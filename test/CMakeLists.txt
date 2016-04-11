include_directories(tests)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
   link_directories(/usr/local/lib)
endif()

set(TEST_FILES
               tests/broadcasttests.h
               tests/pairwiseutiltests.h
               tests/reducetests.h
               tests/summarystatsreducetest.h
               tests/transformtests.h
               tests/indexreducetests.h
               tests/scalartests.h
               tests/testhelpers.h
               tests/pairwise_transform_tests.h
               tests/reduce3tests.h
               tests/shapetests.h
               tests/teststring.h)

if (CUDA_FOUND)
    message("ADDING CUDA EXECUTABLE")
    set( CUDA_ARCHITECTURE_MINIMUM "5.0" CACHE STRING "Minimum required CUDA compute capability" )
    SET(CUDA_VERBOSE_BUILD ON)
    set(CUDA_COMPUTE_CAPABILITY "50")
    list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE;-G;-g;-gencode arch=compute_50,code=sm_50")
    CUDA_COMPILE(MAIN_O  gpu/main.cu)
    cuda_add_executable(libnd4jtests ${TEST_FILES} ${MAIN_O})
    target_link_libraries(libnd4jtests cppunit)

else()
    add_executable(libnd4jtests ${TEST_FILES} cpu/main.cpp)
    target_link_libraries(libnd4jtests CppUTest)
endif()

