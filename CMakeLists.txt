cmake_minimum_required(VERSION 3.3)
project(operations)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${root_SOURCE_DIR}/include)
include_directories("/usr/include/")
include_directories("src/include")
find_path(STDIO_INCLUDE_PATH stdio.h)
LINK_DIRECTORIES(/usr/local/lib)
TARGET_LINK_LIBRARIES(CppUTest CppUTestExt)


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
    src/array/array.cu
    src/array/buffer.cu
    src/array/shape.cu
    src/cuda_util/helper_cuda.h
    src/cuda_util/helper_string.h
    src/cuda_util/sharedmem.h
    src/functions/broadcasting.cu
    src/functions/indexreduce.cu
    src/functions/pairwise_transform.cu
    src/functions/postprocess.cu
    src/functions/reduce.cu
    src/functions/reduce3.cu
    src/functions/scalar.cu
    src/functions/transform.cu
     src/tests/unittest/assertions.h
    src/tests/unittest/exceptions.h
    src/tests/unittest/meta.h
    src/tests/unittest/system.h
    src/tests/unittest/testframework.h
    src/tests/unittest/unittest.h
    src/tests/unittest/util.h
    src/tests/buffertests.h
    src/tests/reducetests.h
    src/tests/shapetests.h
    src/tests/testframework.h
    src/tests/testhelpers.h
    src/nd4j-kernels2.cu
    src/testreduce.cu
    main.cpp)

add_executable(operations ${SOURCE_FILES})