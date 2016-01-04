cmake_minimum_required(VERSION 3.3)
project(nativeoperations)

include_directories("src/include/")
include_directories("src/tests/")
include_directories(/usr/java/jdk1.8.0_31/include/)
find_path(STDIO_INCLUDE_PATH stdio.h)
LINK_DIRECTORIES(/usr/local/lib)

LINK_LIBRARIES(CppUTest CppUTestExt)

set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++11 -msse2 -ftree-vectorizer-verbose=2 -fassociative-math -funsafe-math-optimizations -march=native")

set(SOURCE_FILES
        src/array/shape.cc
        main.cpp src/templatemath.cpp src/tests/broadcaststests.h src/tests/scalartests.h)

add_executable(nativeoperations ${SOURCE_FILES})