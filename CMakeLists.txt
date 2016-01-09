cmake_minimum_required(VERSION 3.3)
project(nativeoperations)
set( CMAKE_VERBOSE_MAKEFILE on )
# packages
find_package(CUDA)

find_path(STDIO_INCLUDE_PATH stdio.h)

include_directories("src/include")
include_directories("src/tests")
LINK_LIBRARIES(CppUTest)
set(CMAKE_CXX_FLAGS "-lCppUTest -Wall -O3 -fopenmp -std=c++11 -ftree-vectorize -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")

if (CUDA_FOUND)
    message("CUDA found!")
    SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_SEPARABLE_COMPILATION OFF)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-DVERBOSE;-lCppUTest")
    cuda_add_executable(nativeoperations main.cu)

else()
    LINK_LIBRARIES(CppUTest)
    message("CUDA not found")
    set(SOURCE_FILES
            src/array/shape.cu
            main.cu src/array/templatemath.cu)
    add_executable(nativeoperations ${SOURCE_FILES} src/array/templatemath.cu src/array/templatemath.h)
endif()

