include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(../../include ../../layers ../../include/helpers ../../include/memory ../../include/loops ../../include/graph ../../include/ops ../../include/types ../../include/cnpy ../../blas)
if(LINUX)
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()

if(APPLE)
    message("Using apple")
    link_directories(/usr/local/lib)
    link_directories(/usr/lib)
    link_directories(/lib)
endif()
if(WIN32)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()
endif()


set(CMAKE_CXX_FLAGS  " -O3 -std=c++11 -fassociative-math -funsafe-math-optimizations")
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wa,-mbig-obj ")
endif()


if (APPLE)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER 6.0 OR "${CMAKE_C_COMPILER_VERSION}" VERSION_EQUAL 6.0)
            SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wa,-mavx512f -fmax-errors=1")
        endif()
    endif()
endif()


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O3 -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
endif()


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")


message("CPU BLAS")
add_definitions(-D__CPUBLAS__=true)
add_library(nd4jcpu SHARED ../../blas/cpu/NativeOps.cpp ../../blas/cpu/GraphExecutioner.cpp
        ../../blas/cpu/NativeOpExcutioner.cpp ../../blas/cpu/NDArray.cpp ../../blas/cpu/NDArrayFactory.cpp  ../../include/memory/cpu/Workspace.cpp
        ../../include/cnpy/cnpy.cpp  ../../include/nd4jmemset.h ../../include/nd4jmalloc.h ../../include/ops/declarable/generic/parity_ops.cpp
        ../../include/ops/declarable/generic/third_party.cpp ../../include/memory/cpu/Workspace.cpp ../../include/ops/declarable/generic/helpers/impl/convolutions.cpp ../../include/ops/declarable/generic/convo/convo_ops.cpp ../../blas/Environment.cpp ../../include/graph/impl/Stash.cpp ../../include/indexing/impl/NDIndex.cpp ../../include/indexing/impl/IndicesList.cpp ../../include/graph/impl/Variable.cpp ../../include/graph/impl/VariableSpace.cpp ../../include/graph/impl/ShapeList.cpp ../../include/graph/impl/SessionLocalStorage.cpp ../../include/graph/impl/Node.cpp ../../include/graph/impl/Graph.cpp ../../include/graph/impl/Block.cpp ../../include/graph/impl/ArrayList.cpp ../../include/helpers/impl/ArrayUtils.cpp ../../include/helpers/impl/helper_hash.cpp ../../include/helpers/impl/TAD.cpp ../../include/ops/impl/specials_sparse.cpp ../../include/helpers/impl/shape.cpp ../../include/ops/declarable/impl/DeclarableOp.cpp ../../include/ops/declarable/impl/OpRegistrator.cpp ../../include/ops/declarable/impl/DeclarableReductionOp.cpp ../../include/ops/declarable/impl/DeclarableCustomOp.cpp ../../include/ops/impl/specials.cpp ../../include/memory/cpu/MemoryRegistrator.cpp ../../include/ops/impl/gemm.cpp)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
    message(FATAL_ERROR "You need at least GCC 4.9")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("OPENMP NOT FOUND")
endif()


if(APPLE)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    #SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static")
elseif(MSYS)
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static")
endif()


add_executable(runtests StashTests.cpp SessionLocalTests.cpp ConvolutionTests.cpp DeclarableOpsTests.cpp DenseLayerTests.cpp FlatBuffersTests.cpp GraphTests.cpp HashUtilsTests.cpp NDArrayTests.cpp TadTests.cpp VariableSpaceTests.cpp VariableTests.cpp WorkspaceTests.cpp)
target_link_libraries(runtests nd4jcpu gtest gtest_main)
