add_library(sycl_target INTERFACE)

set(CUDA_MEDIAN_FILTER_SYCL_FLAGS  $<$<COMPILE_LANGUAGE:CXX>:-fsycl -fsycl-targets=nvptx64-nvidia-cuda>)
target_compile_options(sycl_target INTERFACE ${CUDA_MEDIAN_FILTER_SYCL_FLAGS})
target_link_options(sycl_target INTERFACE ${CUDA_MEDIAN_FILTER_SYCL_FLAGS})
