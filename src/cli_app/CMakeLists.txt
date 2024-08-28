project(cuda_median_filter_cli CUDA)

add_executable(${PROJECT_NAME} src/main.cu)

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cuda_median_filter_project_options
        cuda_median_filter_project_warnings
        cuda_median_filter_project_cuda_settings
        cuda_median_filter_shared_utils
        cuda_median_filter)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

option(CUDA_MEDIAN_FILTER_CLI_APP_ENABLE_PROFILING "Pass -lineinfo to nvcc for cli_app compilation to allow profiling with nvidia visual profiler" OFF)

if (CUDA_MEDIAN_FILTER_CLI_APP_ENABLE_PROFILING)
    set_source_files_properties(src/main.cu PROPERTIES COMPILE_OPTIONS "-lineinfo")
endif()
