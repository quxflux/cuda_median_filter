add_subdirectory(shared_utils)

if(CUDA_MEDIAN_FILTER_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(CUDA_MEDIAN_FILTER_BUILD_BENCHMARK_APP)
  add_subdirectory(benchmark_app)
endif()

if(CUDA_MEDIAN_FILTER_BUILD_CLI_APP)
  add_subdirectory(cli_app)
endif()
