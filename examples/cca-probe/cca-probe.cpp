// Two-sequence tensor probe.
//
// Puts the same prompt into sequence 0 and sequence 1 of ONE batch, so both
// sequences share every ubatch. Every tensor in the graph that spans tokens
// (ne == 2n, tokens are sequence-major) or sequences (ne == 2) must then be
// identical across its two halves. The first tensor where it is not is the
// operation through which one sequence contaminates the other. The final
// logits are also compared against a single-sequence decode of the same
// prompt.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

struct probe {
    int n = 0;        // tokens per sequence
    int n_seqs = 2;   // 1 disables the check (reference decode)
    int checked = 0, mismatched = 0, reported = 0;
    std::vector<uint8_t> buf;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    probe * p = (probe *) ud;
    if (ask) return p->n_seqs >= 2 && t->type == GGML_TYPE_F32;
    if (p->n_seqs < 2 || t->type != GGML_TYPE_F32) return true;
    if (strstr(t->name, ".weight") || strstr(t->name, ".bias")) return true;   // a [1, n_head_kv] weight is not a sequence pair
    int d = -1; int64_t half = 0;
    for (int i = 3; i >= 0; --i) {
        if (t->ne[i] == (int64_t) p->n_seqs * p->n) { d = i; half = p->n; break; }
        if (t->ne[i] == (int64_t) p->n_seqs)        { d = i; half = 1;    break; }
    }
    if (d < 0) return true;
    const size_t nbytes = ggml_nbytes(t);
    p->buf.resize(nbytes);
    ggml_backend_tensor_get(t, p->buf.data(), 0, nbytes);
    const char * base = (const char *) p->buf.data();
    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    ne[d] = half;
    double maxd = 0, maxa = 0;
    for (int64_t i3 = 0; i3 < ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
        const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
        const float a = *(const float *) (base + off);
        const float b = *(const float *) (base + off + half*t->nb[d]);
        maxd = std::max(maxd, (double) std::fabs(a - b));
        maxa = std::max(maxa, (double) std::fabs(a));
    }
    p->checked++;
    const bool bad = maxd > 1e-3 * std::max(1.0, maxa);
    if (bad) {
        p->mismatched++;
        if (p->reported < 60) {
            p->reported++;
            LOG("MISMATCH %-30s %-10s ne=[%lld,%lld,%lld,%lld] split_dim=%d max|a-b|=%.4g max|a|=%.4g\n",
                t->name, ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], d, maxd, maxa);
        }
    }
    return true;
}

static void top3(const llama_vocab * vocab, const float * logits, int n_vocab, const char * label) {
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 3, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    LOG("%s top-3:", label);
    for (int k = 0; k < 3; ++k) LOG(" %s(%.2f)", common_token_to_piece(vocab, idx[k]).c_str(), logits[idx[k]]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    llama_numa_init(params.numa);

    probe P;
    params.cb_eval = cb;
    params.cb_eval_user_data = &P;
    params.warmup = false;
    params.n_parallel = 2;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    const int n = (int) tokens.size();
    P.n = n;
    LOG("prompt tokens per sequence: %d\n", n);

    // two sequences, same prompt, one batch
    llama_batch batch = llama_batch_init(2*n, 0, 1);
    for (int s = 0; s < 2; ++s)
        for (int i = 0; i < n; ++i)
            common_batch_add(batch, tokens[i], i, { s }, i == n - 1);
    if (llama_decode(ctx, batch)) { LOG_ERR("decode (2 seqs) failed\n"); return 1; }
    std::vector<float> l0(llama_get_logits_ith(ctx, n - 1),   llama_get_logits_ith(ctx, n - 1)   + n_vocab);
    std::vector<float> l1(llama_get_logits_ith(ctx, 2*n - 1), llama_get_logits_ith(ctx, 2*n - 1) + n_vocab);
    LOG("tensors checked: %d   mismatched: %d\n", P.checked, P.mismatched);

    // reference: seq 0 alone, memory cleared, callback disabled
    P.n_seqs = 1;
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch ref = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) common_batch_add(ref, tokens[i], i, { 0 }, i == n - 1);
    if (llama_decode(ctx, ref)) { LOG_ERR("decode (1 seq) failed\n"); return 1; }
    std::vector<float> lr(llama_get_logits_ith(ctx, n - 1), llama_get_logits_ith(ctx, n - 1) + n_vocab);

    auto maxdiff = [&](const std::vector<float> & a, const std::vector<float> & b) {
        double m = 0; for (int i = 0; i < n_vocab; ++i) m = std::max(m, (double) std::fabs(a[i] - b[i])); return m; };
    LOG("logits max|seq0 - seq1| = %.4g   max|seq0 - alone| = %.4g   max|seq1 - alone| = %.4g\n",
        maxdiff(l0, l1), maxdiff(l0, lr), maxdiff(l1, lr));
    top3(vocab, lr.data(), n_vocab, "alone ");
    top3(vocab, l0.data(), n_vocab, "seq 0 ");
    top3(vocab, l1.data(), n_vocab, "seq 1 ");

    llama_batch_free(batch);
    llama_batch_free(ref);
    llama_backend_free();
    return 0;
}
