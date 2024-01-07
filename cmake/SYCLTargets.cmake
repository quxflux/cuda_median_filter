add_library(sycl_target INTERFACE)
target_compile_options(sycl_target INTERFACE -fsycl)
target_link_options(sycl_target INTERFACE -fsycl)

add_library(sycl_cuda_target INTERFACE)
target_compile_options(sycl_cuda_target INTERFACE -fsycl-targets=nvptx64-nvidia-cuda)
target_link_options(sycl_cuda_target INTERFACE -fsycl-targets=nvptx64-nvidia-cuda)
target_link_libraries(sycl_cuda_target INTERFACE sycl_target)
