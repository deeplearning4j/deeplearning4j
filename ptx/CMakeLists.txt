if (CUDA_FOUND)
    cuda_compile_ptx(
            cuda_ptx_files
            ptx.cu
           ${LIBND4J_SOURCE})
    add_custom_target(ptxoutput  ptx.cu DEPENDS ${cuda_ptx_files} )
endif()

