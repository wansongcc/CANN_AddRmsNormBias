cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(add_rms_norm_bias_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

add_executable(add_rms_norm_bias_custom main.asc)

target_link_libraries(add_rms_norm_bias_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(add_rms_norm_bias_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(add_rms_norm_bias_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)

add_executable(benchmark_stats_test scripts/test_benchmark_stats.cpp)
target_include_directories(benchmark_stats_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(add_rms_norm_bias_benchmark benchmarks/benchmark_main.asc)

target_link_libraries(add_rms_norm_bias_benchmark PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(add_rms_norm_bias_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(add_rms_norm_bias_benchmark PRIVATE
    BENCHMARK_SOC_ARCH="${SOC_ARCH}"
)

target_compile_options(add_rms_norm_bias_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)
