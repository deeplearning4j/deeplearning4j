if (CUDA_FOUND AND FATBIN)
    message("Building fatbin")
    set( CUDA_ARCHITECTURE_MINIMUM "3.5" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_SEPARABLE_COMPILATION OFF)
    file (GLOB_RECURSE cufiles *.cu)
    cuda_compile_fatbin (ptxfiles ${cufiles})
    add_custom_target (fatbin ALL  DEPENDS ${ptxfiles} SOURCES ${ptxfiles})

endif()

