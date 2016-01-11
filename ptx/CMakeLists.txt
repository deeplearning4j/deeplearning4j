if (CUDA_FOUND AND PTX)
    message("Building ptx")
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_SEPARABLE_COMPILATION OFF)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-G;-g;-gencode arch=compute_50,code=sm_50")
    file (GLOB_RECURSE cufiles *.cu)
    cuda_compile_ptx (ptxfiles ${cufiles})
    add_custom_target (ptx ALL  DEPENDS ${ptxfiles} SOURCES ${ptxfiles})

endif()

