include_directories("tests")
LINK_LIBRARIES(CppUTest)
if (CUDA_FOUND)
    message("ADDING CUDA EXECUTABLE")
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-lCppUTest;-G;-g;-gencode arch=compute_50,code=sm_50")
    CUDA_COMPILE(MAIN_O  main.cu)
    cuda_add_executable(libnd4jtests ${MAIN_O})

else()
   SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -x c++" )
   
    LINK_LIBRARIES(CppUTest)
<<<<<<< HEAD
    message("CUDA not found")
    add_executable(nativeoperations main.cu)
    SET_TARGET_PROPERTIES(nativeoperations PROPERTIES LINKER_LANGUAGE CXX)
    
endif()
=======
    add_executable(libnd4jtests cpu/main.cpp)
endif()
>>>>>>> f1b7c0150c0a40602b56b9b8c079e18f45400805
