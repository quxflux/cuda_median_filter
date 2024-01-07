include(cmake/CPM.cmake)

function(add_sorting_network_cpp)
    CPMAddPackage("gh:quxflux/sorting_network_cpp#98bfe00")
endfunction()

function(add_metal)
    CPMAddPackage(
            NAME metal
            VERSION 2.1.4
            GITHUB_REPOSITORY brunocodutra/metal
            OPTIONS
                "METAL_BUILD_DOC OFF"
                "METAL_BUILD_EXAMPLES OFF"
                "METAL_BUILD_TESTS OFF"
    )
endfunction()

function(add_gtest)
    CPMAddPackage(
            NAME googletest
            VERSION 1.12.0
            GITHUB_REPOSITORY google/googletest
            OPTIONS
                "INSTALL_GTEST OFF"
    )
endfunction()