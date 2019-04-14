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

SET(PNG_HOME ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libpng-1.6.36 CACHE PATH "")
SET(ZLIB_HOME ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib-1.2.11 CACHE PATH "")
SET(STANDALONE_DEPS_INSTALL_PATH /opt/dl4j/lib CACHE PATH "")
include_directories(${PNG_HOME})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/libpng-1.6.36/)

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

if(${CMAKE_SYSTEM_NAME} MATCHES "Aurora")
    set(LIBND4J_NAME ${LIBND4J_NAME}static)
endif()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../blasbuild/cpu/blas/)

SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -z muldefs")



set(SOURCES GraphExecutor.cpp)

add_subdirectory(${ZLIB_HOME})

SET(ZLIB_FILE libz.so)
SET(ZLIB_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/zlib-1.2.11/)
SET(ZLIB_LIBRARY libz.so)
SET(ZLIB_INCLUDE_DIR 
	${CMAKE_CURRENT_BINARY_DIR}/thirdparty/zlib-1.2.11/ 
	${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/zlib-1.2.11/)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/zlib-1.2.11/)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/thirdparty/libpng-1.6.36/)

if(${CMAKE_SYSTEM_NAME} MATCHES "Aurora")
    SET(ZLIB_STATIC_FILE ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/zlib-1.2.11/libzstatic.a)
else()
    SET(ZLIB_STATIC_FILE ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/zlib-1.2.11/libz.a)
endif()

add_subdirectory(${PNG_HOME})
add_dependencies(png z)
add_dependencies(pngfix z)
add_dependencies(png-fix-itxt z)


add_executable(GraphExecutor ${SOURCES})
add_dependencies(GraphExecutor zstatic)
add_dependencies(GraphExecutor png_static)
add_dependencies(GraphExecutor png)


#target_link_libraries(GraphExecutor ${LIBND4J_NAME}static png_static ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/zlib-1.2.11/libzstatic.a)
target_link_libraries(GraphExecutor 
        ${LIBND4J_NAME}
	${CMAKE_CURRENT_BINARY_DIR}/thirdparty/libpng-1.6.36/libpng16.a
	${ZLIB_STATIC_FILE} ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES})

