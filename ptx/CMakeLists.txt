if (CUDA_FOUND)
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_SEPARABLE_COMPILATION OFF)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-lCppUTest;-G;-g;-gencode arch=compute_50,code=sm_50")
    file (GLOB_RECURSE cufiles *.cu)
    # Tag interop_native_CUDA.cu for OBJ format intead of PTX (which is the default for
    # OPTIX_add_test_executable).  Do this before calling OPTIX_add_test_executable.
    set_source_files_properties(${cufiles}
            PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
            )
    cuda_compile_ptx (ptxfiles ${cufiles})
    cuda_add_executable (cuda_core_test  ${ptxfiles})
endif()

