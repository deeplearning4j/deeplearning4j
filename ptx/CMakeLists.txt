if (CUDA_FOUND AND PTX)
    message("Building ptx")
    set( CUDA_ARCHITECTURE_MINIMUM "3.5" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_SEPARABLE_COMPILATION OFF)
    file (GLOB_RECURSE cufiles *.cu)
    cuda_compile_ptx (ptxfiles ${cufiles})
    add_custom_target (ptx ALL  DEPENDS ${ptxfiles} SOURCES ${ptxfiles})

endif()

