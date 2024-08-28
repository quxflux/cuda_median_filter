project(cuda_median_filter_tests)

include(GoogleTest)

add_executable(${PROJECT_NAME}
    src/cuda_median_filter_tests.cu
    src/image_filter_config.cpp
    src/load_neighbor_pixels.cpp
    src/math.cpp
    src/pointer_arithmetic.cpp
    src/primitives.cpp
    src/static_for.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME}
    PRIVATE cuda_median_filter_project_options
        cuda_median_filter_project_warnings
        cuda_median_filter_project_cuda_settings
        cuda_median_filter_shared_utils
        cuda_median_filter
        GTest::gmock
        GTest::gmock_main
        Metal
)

gtest_discover_tests(${PROJECT_NAME})
