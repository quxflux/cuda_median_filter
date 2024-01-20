project(cuda_median_filter_cli CUDA)

SET(CLI_SOURCE_FILES src/main.cpp src/run_with_cuda.cu)

if (CUDA_MEDIAN_FILTER_ENABLE_SYCL)
    list(APPEND CLI_SOURCE_FILES src/run_with_sycl.cpp)
endif()

add_executable(${PROJECT_NAME} ${CLI_SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE src)

add_lyra()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cuda_median_filter_project_options
        cuda_median_filter_project_warnings
        cuda_median_filter_project_cuda_settings
        cuda_median_filter_shared_utils
        cuda_median_filter
        lyra)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# uncomment to allow for kernel profiling in nvidia visual profiler
# set_source_files_properties(src/main.cu PROPERTIES COMPILE_OPTIONS "-lineinfo")

if (CUDA_MEDIAN_FILTER_ENABLE_SYCL)
    target_link_libraries(${PROJECT_NAME} PRIVATE sycl_target)
endif()