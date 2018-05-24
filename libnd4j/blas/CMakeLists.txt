set( CMAKE_VERBOSE_MAKEFILE on)

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


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_RELEASE=true -Wl,-rpath,$ORIGIN/")

if ("${LIBND4J_ALL_OPS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIBND4J_ALL_OPS=true")
else()
    message("_OPS: ${LIBND4J_OPS_LIST}")
    foreach(OP "${LIBND4J_OPS_LIST}")
        message(STATUS "${OP}")
    endforeach()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBND4J_OPS_LIST}")
endif()

IF(${ARCH} MATCHES "arm*")
    set(ARCH_TUNE "-march=${ARCH}")
ELSEIF(${ARCH} MATCHES "power*")
    set(ARCH_TUNE "-mcpu=${ARCH} -mtune=${ARCH} -D__POWER")
ELSEIF(${EXTENSION} MATCHES "avx2")
    set(ARCH_TUNE "-march=${ARCH} -mtune=${ARCH} -msse4.1 -msse4.2 -mavx -mavx2 -mfma")
ELSE()
    set(ARCH_TUNE "-march=${ARCH} -mtune=${ARCH}")
ENDIF()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O3 -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++

    set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /EHsc")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    #  -fmax-errors=3

    set(CMAKE_CXX_FLAGS_RELEASE  " -O3 -std=c++11 -fassociative-math -funsafe-math-optimizations")
    set(CMAKE_CXX_FLAGS_DEBUG " -g -O0 -std=c++11")

    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -fPIC")
endif()


IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("/usr/include")
    include_directories("/usr/local/include")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
if(!CUDA_BLAS)
    if(!CPU_BLAS)
        set(CUDA_BLAS FALSE)
        set(CPU_BLAS TRUE)
    endif()
endif()

# TODO: get rid of this once problem confirmed solved
#if (APPLE)
#    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#        if ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER 6.0 OR "${CMAKE_C_COMPILER_VERSION}" VERSION_EQUAL 6.0)
#            SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wa,-mavx512f,-mavx512vl,-mavx512bw,-mavx512dq,-mavx512cd ")
#        endif()
#    endif()
#endif()


if(CUDA_BLAS)
    message("Build cublas")
    find_package(CUDA)
    add_definitions(-D__CUDABLAS__=true)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set (CMAKE_CXX_FLAGS "")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        if ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER 4.9 AND "$ENV{TRICK_NVCC}" STREQUAL "YES" AND CUDA_VERSION VERSION_LESS "8.0")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__GNUC__=4 -D__GNUC_MINOR__=9 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED")
            set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -std=c++98 -Dnullptr=NULL")
            message("TRICKING CUDA INTO SUPPORTING GCC > 4.9 YOU ARE PROCEEDING AT YOUR OWN RISK")
        endif()
    endif()

    # we want OpenMP to be available for hybrid operations, at least for GCC
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()

    if (CUDA_FOUND)
        message("CUDA found!")
        set( CUDA_ARCHITECTURE_MINIMUM "3.0" CACHE STRING "Minimum required CUDA compute capability" )
        SET(CUDA_VERBOSE_BUILD ON)
        SET(CUDA_SEPARABLE_COMPILATION ON)
        #set(CUDA_COMPUTE_CAPABILITY "61")
        set(CUDA_COMPUTE_CAPABILITY "35")
        # make NVCC more verbose to prevent timeouts on CI servers
        #list(APPEND CUDA_NVCC_FLAGS -v)
        if ("${EXPERIMENTAL}" STREQUAL "yes")
            list(APPEND CUDA_NVCC_FLAGS -DEXPERIMENTAL)
        endif()
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            if(CUDA_VERSION VERSION_GREATER "8.0") # cuda 9
                if ("${COMPUTE}" STREQUAL "all")
                    if (APPLE)
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 -w --cudart=static -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62)
                    else()
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 -w --cudart=static -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70)
                    endif()
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 -w --cudart=static -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            elseif (CUDA_VERSION VERSION_GREATER "7.5") # cuda 8.0
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 -w --cudart=static -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 -w --cudart=static -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            else()
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 --cudart=static -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 --cudart=static -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            endif()

        else()
            if(CUDA_VERSION VERSION_GREATER "8.0") # cuda 9
                if ("${COMPUTE}" STREQUAL "all")
                    if (APPLE)
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 -G -g --cudart=static -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62)
                    elseif()
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 -G -g --cudart=static -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70)
                    endif()
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 -G -g --cudart=static -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            elseif (CUDA_VERSION VERSION_GREATER "7.5") # cuda 8
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 -G -g --cudart=static -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 -G -g --cudart=static -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            else()
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 -G -g --cudart=static -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 -G -g --cudart=static -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            endif()
        endif()


        file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
        file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/*.cpp ../include/array/*.h)
        file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/*.cpp ../include/memory/*.h)
        file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.h)
        file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES false ../include/ops/declarable/helpers/cpu/*.cpp)
        file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
        file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
        file(GLOB_RECURSE HELPERS_SOURCES false ../include/helpers/*.cpp ../include/helpers/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/*.cpp ../include/loops/*.h)

            # ../include/loops/cuda/grid_strided.cu ../include/loops/cuda/grid_shaped.cu
            #cuda/NativeOps.cu ../include/loops/cuda/grid_strided.cu ../include/loops/cuda/grid_shaped.cu ../include/loops/cuda/scalar.cu ../include/loops/cuda/reduce.cu ../include/loops/cuda/pairwise_transform.cu
            # ../include/loops/cuda/scalar.cu
        CUDA_ADD_LIBRARY(${LIBND4J_NAME} SHARED cuda/NativeOps.cu ../include/loops/cuda/grid_shaped.cu ../include/loops/cuda/summarystatsreduce.cu ../include/loops/cuda/reduce.cu ../include/loops/cuda/indexreduce.cu ../include/loops/cuda/scalar.cu ../include/loops/cuda/broadcasting.cu ../include/loops/cuda/transform.cu ../include/loops/cuda/grid_strided.cu   ../include/loops/cuda/pairwise_transform.cu ../include/loops/cuda/random.cu ${CUSTOMOPS_HELPERS_SOURCES}
                ../include/cnpy/cnpy.cpp  ../include/nd4jmemset.h ../include/nd4jmalloc.h
                cpu/GraphExecutioner.cpp cpu/NativeOpExcutioner.cpp cpu/NDArray.cpp cpu/NDArrayFactory.cpp
                Environment.cpp Environment.h ${LOOPS_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
                ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES} ${OPS_SOURCES})
        target_link_libraries(${LIBND4J_NAME} ${CUDA_LIBRARIES})

        if(WIN32)
            message("CUDA on Windows: enabling /EHsc")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
            SET_TARGET_PROPERTIES(${LIBND4J_NAME} PROPERTIES COMPILER_FLAGS "/EHsc")
        endif()

        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cuda)

        install(TARGETS ${LIBND4J_NAME} DESTINATION .)
    endif(CUDA_FOUND)
elseif(CPU_BLAS)

    file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/*.cpp ../include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/*.cpp ../include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES false ../include/ops/declarable/helpers/cpu/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES false ../include/helpers/*.cpp ../include/helpers/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/*.cpp ../include/loops/*.h)

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)
if(IOS)
    add_library(${LIBND4J_NAME} STATIC cpu/NativeOps.cpp cpu/GraphExecutioner.cpp
            cpu/NativeOpExcutioner.cpp cpu/NDArray.cpp cpu/NDArrayFactory.cpp
            ../include/cnpy/cnpy.cpp  ../include/nd4jmemset.h ../include/nd4jmalloc.h
            Environment.cpp Environment.h ${LOOPS_SOURCES}  ${ARRAY_SOURCES} ${TYPES_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}  ${CUSTOMOPS_HELPERS_SOURCES}
            ${OPS_SOURCES})
else()

    add_library(${LIBND4J_NAME} SHARED cpu/NativeOps.cpp cpu/GraphExecutioner.cpp
            cpu/NativeOpExcutioner.cpp cpu/NDArray.cpp cpu/NDArrayFactory.cpp
            ../include/cnpy/cnpy.cpp  ../include/nd4jmemset.h ../include/nd4jmalloc.h
            Environment.cpp Environment.h ${LOOPS_SOURCES}  ${ARRAY_SOURCES} ${TYPES_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}  ${CUSTOMOPS_HELPERS_SOURCES}
            ${OPS_SOURCES} )


    if ("${LIBND4J_ALL_OPS}" AND "${LIBND4J_BUILD_MINIFIER}")
        message(STATUS "Building minifier...")
        add_library(nd4jstatic STATIC cpu/NativeOps.cpp cpu/GraphExecutioner.cpp
            cpu/NativeOpExcutioner.cpp cpu/NDArray.cpp cpu/NDArrayFactory.cpp
            ../include/cnpy/cnpy.cpp  ../include/nd4jmemset.h ../include/nd4jmalloc.h
            Environment.cpp Environment.h ${LOOPS_SOURCES}  ${ARRAY_SOURCES} ${TYPES_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}  ${CUSTOMOPS_HELPERS_SOURCES}
            ${OPS_SOURCES} )

        add_executable(minifier ../minifier/minifier.cpp ../minifier/graphopt.cpp)
        target_link_libraries(minifier nd4jstatic)
    endif()
endif()

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 4.9)
      message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    # OpenMP works well pretty much only with GCC
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        find_package(OpenMP)
        if (OPENMP_FOUND)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()

    if(APPLE)
        # We cannot use those to propagate C++ exceptions across shared libraries.
        # SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    elseif(MSYS)
        SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -static")
    endif()

    #install(TARGETS mySharedLib DESTINATION /some/full/path)
    install(TARGETS ${LIBND4J_NAME} DESTINATION  .)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cpu)
endif()
