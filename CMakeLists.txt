cmake_minimum_required(VERSION 3.3)
project(nativeoperations)
set( CMAKE_VERBOSE_MAKEFILE on )
include_directories("src/include/")
include_directories("src/tests/")

find_path(STDIO_INCLUDE_PATH stdio.h)
# packages
find_package(CUDA)

if(UNIX)
    message("IN UNIX")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
    include_directories("/usr/include/c++/4.9.2")
    include_directories("/usr/include/c++/4.9.2/backward")
    LINK_DIRECTORIES(/usr/local/lib)

    if(CUDA_FOUND)
        message("RUNNING CUDA")
        set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include/"  ${CUDA_INCLUDE_DIRS})
        SET(CUDA_CC "50" CACHE STRING "CUDA Compute Capability of the target GPU.")
        SET(CUDA_NVCC_FLAGS
                "-x ${CUDA_NVCC_FLAGS};-gencode;arch=compute_${CUDA_CC},code=sm_${CUDA_CC};")
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC")

        # Note: only static libraries are supported by the device linker.
        SET(BUILD_SHARED_LIBS OFF)
        # Disable warnings for CUDA
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -w")

        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -rdc=true")
        include_directories("/usr/include")
        include_directories(/usr/local/cuda-7.5/include/)
        LINK_DIRECTORIES(/usr/local/cuda-7.5/lib64)
        link_libraries(cuda cudart cudadevrt)
    endif()
endif()

LINK_LIBRARIES(CppUTest)
set(CMAKE_VERBOSE_MAKEFILE 1)
#set(CMAKE_CXX_FLAGS "-Wall -O3 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")
set(CMAKE_CXX_FLAGS "-Wall -O0 -fopenmp -std=c++11 -ftree-vectorize -fopt-info-vec-missed  -fopt-info-vec-all -msse2 -ftree-vectorizer-verbose=0 -fassociative-math -funsafe-math-optimizations -march=native")

SET(CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT TRUE)


SET(PARALLELISM_ENABLED 1)




if (CUDA_FOUND)
    message("CUDA found!")
    set(CUDA_FILES
            src/array/shape.cc
            src/array/templatemath.cc
            main.cpp)
    cuda_add_executable(nativeoperations ${CUDA_FILES})


else()
    message("CUDA not found")
    set(SOURCE_FILES
            src/array/shape.cc
            main.cpp src/array/templatemath.cc)
    add_executable(nativeoperations ${SOURCE_FILES} src/array/templatemath.cc src/array/templatemath.h)
endif()

