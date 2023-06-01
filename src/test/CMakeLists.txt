project(cuda_median_filter_tests)

include(GoogleTest)

add_executable(${PROJECT_NAME} src/cuda_median_filter_tests.cu)

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    ${CMAKE_SOURCE_DIR}/deps/metal/include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE project_options
        project_warnings
        project_cuda_settings
        cuda_median_filter_shared_utils
        cuda_median_filter
        GTest::gtest
        GTest::gmock
        Metal)
         
gtest_discover_tests(${PROJECT_NAME})
