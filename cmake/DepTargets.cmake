add_library(SortingNetworkCpp INTERFACE)
target_include_directories(SortingNetworkCpp INTERFACE ${CMAKE_SOURCE_DIR}/deps/sorting_network_cpp/include)

add_library(Metal INTERFACE)
target_include_directories(Metal INTERFACE ${CMAKE_SOURCE_DIR}/deps/metal/include)
