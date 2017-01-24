cmake_minimum_required(VERSION 3.2)
project(libnd4j)
set( CMAKE_VERBOSE_MAKEFILE on)
option(NATIVE "Optimize for build machine (might not work on others)" OFF)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#ensure we create lib files
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)


set(CMAKE_CXX_FLAGS_RELEASE  "-Wall -O3 -std=c++11 -fassociative-math -funsafe-math-optimizations")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0 -std=c++11")


if(NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()


if (NOT DEFINED ENV{CLION_IDE})
    message("NOT CLION")
    include_directories(include/ include/helpers include/loops include/ops include/types)
    add_subdirectory(blas)
endif ()


if ($ENV{CLION_IDE})
    add_subdirectory(tests)
endif ()