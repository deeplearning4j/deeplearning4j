include_directories("tests")

# Check for CppUTest and bail out if they don't have it
if(DEFINED ENV{CPPUTEST_HOME})
    message("Using CppUTest found in $ENV{CPPUTEST_HOME}")
else()
    message("CPPUTEST_HOME is not set; You must tell CMake where to find CppUTest")
    return()
endif()
if(LINUX)
    link_directories(/usr/local/lib)
endif()

add_library(CppUTest)


if (CUDA_FOUND)
    message("ADDING CUDA EXECUTABLE")
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-G;-g;-gencode arch=compute_50,code=sm_50")
    CUDA_COMPILE(MAIN_O  gpu/main.cu)
    cuda_add_executable(libnd4jtests ${MAIN_O})

else()
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}" )
    add_executable(libnd4jtests cpu/main.cpp)
endif()
