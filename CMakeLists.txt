cmake_minimum_required(VERSION 3.20)

project(cuda_median_filter LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

include(cmake/CPM.cmake)
CPMAddPackage("gh:quxflux/sorting_network_cpp#98bfe00")

add_library(cuda_median_filter_project_options INTERFACE)
target_compile_features(cuda_median_filter_project_options INTERFACE cxx_std_20)

add_library(cuda_median_filter INTERFACE
    include/cuda_median_filter/cuda_median_filter.h
    include/cuda_median_filter/detail/cuda_median_filter_impl.h
    include/cuda_median_filter/detail/cuda_wrap.h
    include/cuda_median_filter/detail/image_source_target.h
    include/cuda_median_filter/detail/math.h
    include/cuda_median_filter/detail/pitched_array_accessor.h
    include/cuda_median_filter/detail/pointer_arithmetic.h
    include/cuda_median_filter/detail/primitives.h
    include/cuda_median_filter/detail/static_for.h
    )
target_include_directories(cuda_median_filter INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cuda_median_filter INTERFACE sorting_network_cpp)

option(CUDA_MEDIAN_FILTER_BUILD_BENCHMARK_APP "Enable compilation of benchmarking app" OFF)
option(CUDA_MEDIAN_FILTER_BUILD_CLI_APP "Enable compilation of command line app" OFF)
option(CUDA_MEDIAN_FILTER_BUILD_TESTS "Enable Test Builds" OFF)

include(cmake/CompilerWarnings.cmake)
add_library(cuda_median_filter_project_warnings INTERFACE)
set_project_warnings(cuda_median_filter_project_warnings)

include(cmake/CudaSettings.cmake)
add_library(cuda_median_filter_project_cuda_settings INTERFACE)
set_cuda_settings(cuda_median_filter_project_cuda_settings)

if (CUDA_MEDIAN_FILTER_BUILD_TESTS OR CUDA_MEDIAN_FILTER_BUILD_BENCHMARK_APP)
    set(METAL_BUILD_DOC OFF)
    set(METAL_BUILD_EXAMPLES OFF)
    set(METAL_BUILD_TESTS OFF)
    CPMAddPackage("gh:brunocodutra/metal#v2.1.4")
endif()

if (CUDA_MEDIAN_FILTER_BUILD_TESTS)
    CPMAddPackage("gh:google/googletest#release-1.12.0")
endif()

add_subdirectory(src)
