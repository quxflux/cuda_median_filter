project(cuda_median_filter_tests)

include(GoogleTest)

# Function to generate test instantiation files for different data types
# The compilation of all test instantiations in a single translation unit
# causes excessive compilation times, so we split them into separate files.
# Each file will instantiate tests for a specific data type.
function(create_test_instantiation_file DATA_TYPE)
    string(TOLOWER ${DATA_TYPE} type_lower)
    set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/cuda_median_filter_tests_${type_lower}.cu")
    file(WRITE ${OUTPUT_FILE}
"
#include <cuda_median_filter_tests.h>

namespace quxflux
{
  using filter_specs_${type_lower} = rewrap_list<::testing::Types, generate_all_filter_specs<${DATA_TYPE}, metal::numbers<1, 3, 5, 7>>>;
  TYPED_TEST_SUITE(median_filter_test, filter_specs_${type_lower}, test_type_name_generator);
  INSTANTIATE_TYPED_TEST_SUITE_P(scalar_${type_lower}, median_filter_test, filter_specs_${type_lower});
}  // namespace quxflux
")

    message(STATUS "Generated test instantiation file: ${OUTPUT_FILE}")
endfunction()

create_test_instantiation_file("uint8_t")
create_test_instantiation_file("uint16_t")
create_test_instantiation_file("uint32_t")
create_test_instantiation_file("float")

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_median_filter_tests_uint8_t.cu
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_median_filter_tests_uint16_t.cu
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_median_filter_tests_uint32_t.cu
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_median_filter_tests_float.cu
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
