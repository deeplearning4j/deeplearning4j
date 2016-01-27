include_directories(tests)
if(LINUX)
    link_directories(/usr/local/lib)
endif()



if (CUDA_FOUND)
    message("ADDING CUDA EXECUTABLE")
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-G;-g;-gencode arch=compute_50,code=sm_50")
    CUDA_COMPILE(MAIN_O  gpu/main.cu)
    cuda_add_executable(libnd4jtests ${MAIN_O})
    target_link_libraries(libnd4jtests imp_cpputest)

else()
    message("Adding cpu exec")
    add_executable(libnd4jtests cpu/main.cpp)
    target_link_libraries(libnd4jtests CppUTest)
endif()
