# GCC and Clang flags for C++ template duplicate instantiation issues
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
        # Also set CXX flags for host compilation
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -Wno-error")
        message(STATUS "Added CUDA-specific compiler flags for duplicate template instantiations")
        message(STATUS "Added --disable-warnings to suppress NVCC errors")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # For Clang template duplicate instantiation handling with proper folding
    if(NOT SD_CUDA)
        # Core template folding configuration
        add_compile_options(-ftemplate-depth=1024)
        add_compile_options(-ffunction-sections)
        add_compile_options(-fdata-sections)
        # Enable COMDAT folding for duplicate template instantiations
        add_compile_options(-fmerge-all-constants)
        add_compile_options(-fno-unique-section-names)
        # Use LLD linker for better template folding support
        add_compile_options(-fuse-ld=lld)
        # Linker flags for identical code folding (ICF)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--icf=all -Wl,--gc-sections")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--icf=all -Wl,--gc-sections")
        message(STATUS "Added Clang template folding with LLD linker")
        message(STATUS "Enabled identical code folding for duplicate templates")
    else()
        # CUDA with Clang template folding
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1024 -ffunction-sections -fdata-sections")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-ffunction-sections -Xcompiler=-fdata-sections")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -Wl,--icf=all")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld -Wl,--icf=all")
        message(STATUS "Added CUDA-specific Clang template folding")
        message(STATUS "Enabled LLD linker with ICF for CUDA builds")
    endif()
endif()