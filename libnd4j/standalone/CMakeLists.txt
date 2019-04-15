cmake_minimum_required(VERSION 3.10)

include_directories(../include
../layers
../include/helpers
../include/array
../include/memory
../include/loops
../include/graph
../include/ops
../include/types
../include/cnpy
../blas)




find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DLIBND4J_ALL_OPS=true -pthread")

if(LIBND4J_NAME)

else()
    set(LIBND4J_NAME nd4jcpu)
endif()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../blasbuild/cpu/blas/)

SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -z muldefs")


set(SOURCES GraphExecutor.cpp)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})


add_executable(GraphExecutor ${SOURCES})

target_link_libraries(GraphExecutor 
        ${LIBND4J_NAME}
        ${PNG_LIBRARY}
        ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES})

