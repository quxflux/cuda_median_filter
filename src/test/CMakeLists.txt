project(cuda_median_filter_tests)

add_metal()
add_gtest()

include(GoogleTest)

add_library(cuda_median_filter_tests_base INTERFACE)
target_link_libraries(cuda_median_filter_tests_base INTERFACE
        cuda_median_filter_project_options
        cuda_median_filter_project_warnings
        cuda_median_filter
        cuda_median_filter_shared_utils
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        Metal)

add_executable(cuda_median_filter_tests src/cuda/cuda_median_filter_tests.cu)
target_include_directories(cuda_median_filter_tests PRIVATE src)
target_link_libraries(cuda_median_filter_tests PRIVATE cuda_median_filter_project_cuda_settings cuda_median_filter_tests_base)

if (CUDA_MEDIAN_FILTER_ENABLE_SYCL)
    add_executable(cuda_median_filter_tests_sycl src/sycl/cuda_median_filter_tests.cpp)
    target_include_directories(cuda_median_filter_tests_sycl PRIVATE src)
    target_link_libraries(cuda_median_filter_tests_sycl PRIVATE cuda_median_filter_tests_base sycl_target)
endif ()

gtest_discover_tests(${PROJECT_NAME})
