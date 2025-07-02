# GCC flags for C++ template duplicate instantiation issues
cmake_minimum_required(VERSION 3.15)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # For C++ template duplicate instantiation errors
    if(NOT SD_CUDA)
        add_compile_options(-fpermissive)
        add_compile_options(-ftemplate-depth=1024)
        add_compile_options(-fno-gnu-unique)
        message(STATUS "Added -fpermissive: Allows duplicate template instantiations")
        message(STATUS "Added template-related flags for C++ duplicate handling")
    else()
        # CUDA-specific flags - try multiple approaches
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fpermissive")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-error")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --disable-warnings")

        # Also set CXX flags for host compilation
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wno-error")

        message(STATUS "Added CUDA-specific compiler flags for duplicate template instantiations")
        message(STATUS "Added --disable-warnings to suppress NVCC errors")
    endif()
endif()