cmake_minimum_required(VERSION 3.3)
project(nativeoperations)
set( CMAKE_VERBOSE_MAKEFILE on )
# packages
find_package(CUDA)

find_path(STDIO_INCLUDE_PATH stdio.h)

include_directories("src/include")
include_directories("src/tests")
LINK_LIBRARIES(CppUTest)

if (CUDA_FOUND)
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_SEPARABLE_COMPILATION OFF)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-lCppUTest;-G;-g;-gencode arch=compute_50,code=sm_50")
    cuda_add_executable(nativeoperations main.cu)

else()
    LINK_LIBRARIES(CppUTest)
    message("CUDA not found")
    set(SOURCE_FILES
            src/array/shape.cu
            main.cu src/array/templatemath.cu)
    add_executable(nativeoperations ${SOURCE_FILES} src/array/templatemath.cu src/array/templatemath.h)
endif()

