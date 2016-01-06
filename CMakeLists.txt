cmake_minimum_required(VERSION 3.3)
project(nativeoperations)

include_directories("src/include/")
include_directories("src/tests/")
include_directories(/usr/java/jdk1.8.0_31/include/)
find_path(STDIO_INCLUDE_PATH stdio.h)
LINK_DIRECTORIES(/usr/local/lib)

LINK_LIBRARIES(CppUTest CppUTestExt)
set(CMAKE_VERBOSE_MAKEFILE 1)
#set(CMAKE_CXX_FLAGS "-Wall -O3 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")
set(CMAKE_CXX_FLAGS "-Wall -O0 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")

SET(PARALLELISM_ENABLED 1)
set(SOURCE_FILES
        src/array/shape.cc
        main.cpp src/tests/reduce3tests.h src/tests/indexreducetests.h)

add_executable(nativeoperations ${SOURCE_FILES})