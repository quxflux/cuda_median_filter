project(cuda_median_filter_benchmark)

list(APPEND source_files
       src/main.cpp
       src/filter_database.cpp
       src/impls/cuda_median_filter/create_cuda_median_filter_impls.cu)

option(BENCHMARK_NPP "Benchmark the NPP median filter implementation" OFF)
option(BENCHMARK_OPENCV "Benchmark the OpenCV median filter implementation" OFF)
option(BENCHMARK_OPENCV_CUDA "Benchmark the OpenCV CUDA median filter implementation" OFF)

if (BENCHMARK_OPENCV)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS opencv_core opencv_imgproc)
    list(APPEND source_files src/impls/opencv/create_opencv_median_filter_impls.cpp)
endif()

if (BENCHMARK_OPENCV_CUDA)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_cudaimgproc opencv_cudaarithm)
    list(APPEND source_files src/impls/opencv/create_opencv_cuda_median_filter_impls.cpp)
endif()

if (BENCHMARK_NPP)
    find_package(CUDAToolkit REQUIRED COMPONENTS nppif)
    list(APPEND source_files src/impls/npp/create_npp_median_filter_impls.cpp)
endif()

add_metal()

add_executable(${PROJECT_NAME} ${source_files})

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cuda_median_filter_project_options
        cuda_median_filter_project_warnings
        cuda_median_filter_project_cuda_settings
        cuda_median_filter_shared_utils
        cuda_median_filter
        Metal)

if (BENCHMARK_NPP)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nppif)
endif()

if (BENCHMARK_OPENCV_CUDA OR BENCHMARK_OPENCV)
    list(APPEND dependencies opencv_imgproc opencv_core)
endif()

if (BENCHMARK_OPENCV_CUDA)
    list(APPEND dependencies opencv_cudafilters opencv_cudaarithm)
endif()

foreach (dependency ${dependencies})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${dependency})

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${dependency}> $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endforeach()

