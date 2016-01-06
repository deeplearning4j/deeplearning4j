cmake_minimum_required(VERSION 3.3)
project(nativeoperations)
set( CMAKE_VERBOSE_MAKEFILE on )
include_directories("src/include/")
include_directories("src/tests/")
if(UNIX)
    include_directories("/usr/include")
    include_directories("/usr/local/include")
endif()

find_path(STDIO_INCLUDE_PATH stdio.h)
# packages
find_package(CUDA)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include/" "/usr/local/lib" ${CUDA_INCLUDE_DIRS})
SET(CUDA_CC "50" CACHE STRING "CUDA Compute Capability of the target GPU.")
SET(CUDA_NVCC_FLAGS
        "${CUDA_NVCC_FLAGS};-gencode;arch=compute_${CUDA_CC},code=sm_${CUDA_CC};")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC")

# Disable warnings for CUDA
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-w")

SET(CUDA_SEPARABLE_COMPILATION OFF)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-rdc=true")


if (CUDA_FOUND)
    message("CUDA found!")
else()
    message("CUDA not found")
endif()

LINK_DIRECTORIES(/usr/local/lib)
LINK_LIBRARIES(CppUTest)
set(CMAKE_VERBOSE_MAKEFILE 1)
#set(CMAKE_CXX_FLAGS "-Wall -O3 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")
set(CMAKE_CXX_FLAGS "-Wall -O0 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")




SET(PARALLELISM_ENABLED 1)
set(SOURCE_FILES
        src/array/shape.cc
        main.cpp src/tests/reduce3tests.h src/tests/indexreducetests.h)

add_executable(nativeoperations ${SOURCE_FILES})