cmake_minimum_required(VERSION 3.3)
project(nativeoperations)

include_directories("src/include/")
include_directories("src/tests/")

find_path(STDIO_INCLUDE_PATH stdio.h)
LINK_DIRECTORIES(/usr/local/lib)
LINK_LIBRARIES(CppUTest CppUTestExt)


if(MSVC) # MSVC compiler (Win32 only)
    # Display more warnings
    set(CMAKE_CXX_FLAGS "/W3")
elseif(UNIX OR CMAKE_COMPILER_IS_GNUCXX) # Clang OR Gcc (Linux, Mac OS or Win32 with MingW)
    # Enable C++11 and displays all warnings
    set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++11 -msse2 -ftree-vectorizer-verbose=2 -fassociative-math -funsafe-math-optimizations -march=native")
    if(APPLE) # Clang / Mac OS only
        # Required on OSX to compile c++11
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -mmacosx-version-min=10.7")
    endif(APPLE)
endif()

set(SOURCE_FILES
        src/array/shape.cc
        main.cpp)

add_executable(nativeoperations ${SOURCE_FILES})