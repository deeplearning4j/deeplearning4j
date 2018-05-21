include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../../include ../../include/helpers ../../include/loops ../../include/ops ../../include/types ../../include/cnpy ../../blas)
if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
endif()

if(APPLE)
    message("Using apple")
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()
if(WIN32)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()
endif()


set(CMAKE_CXX_FLAGS  "-Wall -O3 -std=c++11 -fassociative-math -funsafe-math-optimizations")


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O3 -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
endif()


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


message("CPU BLAS")
add_definitions(-D__CPUBLAS__=true)
add_library(nd4jcpu SHARED ../../blas/cpu/NativeOps.cpp   ../../include/nd4jmemset.h ../../include/nd4jmalloc.h)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
    message(FATAL_ERROR "You need at least GCC 4.9")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("OPENMP NOT FOUND")
endif()


if(APPLE)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
elseif(MSYS)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static")
endif()


add_executable(runtests ../../include/cnpy/cnpy.cpp CnpyTests.cpp)
target_link_libraries(runtests nd4jcpu  gtest gtest_main)
