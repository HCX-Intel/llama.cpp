set(TARGET llama-cca-probe)
add_executable(${TARGET} cca-probe.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
