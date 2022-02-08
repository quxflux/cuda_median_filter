add_subdirectory(shared_utils)

if(BUILD_TESTS)
  add_subdirectory(test)
endif()

if(BUILD_BENCHMARK_APP)
  add_subdirectory(benchmark_app)
endif()

if(BUILD_CLI_APP)
  add_subdirectory(cli_app)
endif()
