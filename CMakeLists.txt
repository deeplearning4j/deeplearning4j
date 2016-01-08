cmake_minimum_required(VERSION 3.3)
project(nativeoperations)
set( CMAKE_VERBOSE_MAKEFILE on )
# packages
find_package(CUDA)

find_path(STDIO_INCLUDE_PATH stdio.h)

include_directories("src/include")
include_directories("src/tests")

if (CUDA_FOUND)
    message("CUDA found!")
   SET(CUDA_VERBOSE_BUILD ON)
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++11;-O3;-DVERBOSE;-lCppUTest")
    set(CUDA_FILES
            src/array/shape.cu
            src/array/templatemath.cu
            main.cu)
    cuda_add_executable(nativeoperations ${CUDA_FILES})

else()
    LINK_LIBRARIES(CppUTest)
    message("CUDA not found")
    set(CMAKE_CXX_FLAGS "-Wall -O0 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")
    set(SOURCE_FILES
            src/array/shape.cu
            main.cu src/array/templatemath.cu)
    add_executable(nativeoperations ${SOURCE_FILES} src/array/templatemath.cu src/array/templatemath.h)
endif()

