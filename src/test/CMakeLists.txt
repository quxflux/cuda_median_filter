project(cuda_median_filter_tests)

add_metal()
add_gtest()

include(GoogleTest)

add_executable(${PROJECT_NAME} src/cuda/cuda_median_filter_tests.cu)

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME}
    PRIVATE cuda_median_filter_project_options
        cuda_median_filter_project_warnings
        cuda_median_filter_project_cuda_settings
        cuda_median_filter_shared_utils
        cuda_median_filter
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        Metal)
         
gtest_discover_tests(${PROJECT_NAME})
