cmake_minimum_required(VERSION 3.3)
project(nd4j_kernels)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/broadcasting/include/broadcasting.h
    src/common/helper_cuda.h
    src/common/helper_string.h
    src/common/indexing.h
    src/common/sharedmem.h
    src/common/tad.h
    src/indexreduce/include/indexreduce.h
    src/pairwise_transforms/include/pairwise_transform.h
    src/reduce/include/reduce.h
    src/reduce3/include/reduce3.h
    src/scalar/include/scalar.h
    src/transforms/include/transform.h
    LICENSE
    Makefile
    README.md
    vars.mk)

add_executable(nd4j_kernels ${SOURCE_FILES})