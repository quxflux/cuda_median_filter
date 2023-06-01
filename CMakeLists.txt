cmake_minimum_required(VERSION 3.20)

project(cuda_median_filter_2d LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

include(cmake/DepTargets.cmake)

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

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
target_include_directories(cuda_median_filter INTERFACE include)
target_link_libraries(cuda_median_filter INTERFACE SortingNetworkCpp)

option(BUILD_BENCHMARK_APP "Enable compilation of benchmarking app" ON)
option(BUILD_CLI_APP "Enable compilation of command line app" OFF)
option(BUILD_TESTS "Enable Test Builds" ON)

include(cmake/CompilerWarnings.cmake)
add_library(project_warnings INTERFACE)
set_project_warnings(project_warnings)

include(cmake/CudaSettings.cmake)
add_library(project_cuda_settings INTERFACE)
set_cuda_settings(project_cuda_settings)

include(cmake/CPM.cmake)

if (BUILD_TESTS)
    CPMAddPackage("gh:google/googletest#release-1.12.0")
endif()

add_subdirectory(src)
