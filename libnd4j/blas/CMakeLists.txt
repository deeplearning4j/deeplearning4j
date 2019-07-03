################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

set(CMAKE_VERBOSE_MAKEFILE OFF)

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

if (APPLE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPPLE_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPPLE_BUILD=true")
endif()

if (ANDROID_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_BUILD=true")
endif()

if (IOS_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIOS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DIOS_BUILD=true")
endif()

if(WIN32)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
	endif()
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()

    # workaround for long command lines
    SET(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
    SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)

    SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")

    SET(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")
endif()

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
    set(ARCH_TUNE "-march=${ARCH} -mtune=${ARCH} -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -D__F16C__=true")
ELSE()
    if ("${ARCH}" STREQUAL "x86-64")
        set(ARCH_TYPE "generic")
    else()
        set(ARCH_TYPE "${ARCH}")
    endif()

    set(ARCH_TUNE "-march=${ARCH} -mtune=${ARCH_TYPE}")
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
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/")
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
            set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -std=c++11 -Dnullptr=NULL")
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
		message("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
		include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")
        set( CUDA_ARCHITECTURE_MINIMUM "3.0" CACHE STRING "Minimum required CUDA compute capability" )
        SET(CUDA_VERBOSE_BUILD ON)
        SET(CUDA_SEPARABLE_COMPILATION OFF)
        #set(CUDA_COMPUTE_CAPABILITY "61")
        set(CUDA_COMPUTE_CAPABILITY "35")
        # make NVCC more verbose to prevent timeouts on CI servers
        #list(APPEND CUDA_NVCC_FLAGS -v)
        if ("${EXPERIMENTAL}" STREQUAL "yes")
            message("Experimental mode ENABLED")
            list(APPEND CUDA_NVCC_FLAGS -D__ND4J_EXPERIMENTAL__=true)
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
            set (EXPM " -D__ND4J_EXPERIMENTAL__=true")
        endif()
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            if(CUDA_VERSION VERSION_GREATER "9.2") # cuda 10
                if ("${COMPUTE}" STREQUAL "all")
                    if (APPLE)
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_10 ${EXPM} -w --cudart=static -O3 --expt-extended-lambda -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61)
                    else()
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_10 ${EXPM} -w --cudart=static -O3 --expt-extended-lambda -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75)
                    endif()
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_10 ${EXPM} -w --cudart=static --expt-extended-lambda  -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            elseif(CUDA_VERSION VERSION_GREATER "8.0") # cuda 9
                if ("${COMPUTE}" STREQUAL "all")
                    if (APPLE)
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 ${EXPM} -w --cudart=static -O3 --expt-extended-lambda -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61)
                    else()
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 ${EXPM} -w --cudart=static -O3 --expt-extended-lambda -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61)
                    endif()
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 ${EXPM} -w --cudart=static --expt-extended-lambda -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            elseif (CUDA_VERSION VERSION_GREATER "7.5") # cuda 8.0
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 ${EXPM} -w --cudart=static -O3 --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 ${EXPM} -w --cudart=static --expt-extended-lambda -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            else()
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 ${EXPM} --cudart=static --expt-extended-lambda -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 ${EXPM} --cudart=static --expt-extended-lambda -O3 -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            endif()

        else()
            # debug only
            if (LINUX)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--export-dynamic -rdynamic")
                SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} --export-dynamic")
            endif()

			if(CUDA_VERSION VERSION_GREATER "9.2") # cuda 9
				message("CUDA 10 Debug build")
                if ("${COMPUTE}" STREQUAL "all")
                    if (APPLE)
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_10 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62)
                    elseif()
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_10 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70)
                    endif()
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_10 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            elseif(CUDA_VERSION VERSION_GREATER "8.0") # cuda 9
                if ("${COMPUTE}" STREQUAL "all")
                    if (APPLE)
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62)
                    elseif()
                        list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70)
                    endif()
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_9 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            elseif (CUDA_VERSION VERSION_GREATER "7.5") # cuda 8
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61  -gencode arch=compute_62,code=sm_62)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_8 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            else()
                if ("${COMPUTE}" STREQUAL "all")
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53)
                else()
                    list(APPEND CUDA_NVCC_FLAGS -DCUDA_75 ${EXPM} -w -G -g --cudart=static --expt-extended-lambda -arch=compute_${COMPUTE} -code=sm_${COMPUTE})
                endif()
            endif()
        endif()

        file(GLOB_RECURSE EXCEPTIONS_SOURCES false ../include/exceptions/*.cpp ../include/exceptions/*.h)
        file(GLOB_RECURSE EXEC_SOURCES false ../include/execution/impl/*.cpp ../include/execution/*.cu ../include/execution/*.h)
        file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
        file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/impl/*.cpp ../include/array/cuda/*.cu ../include/array/*.h)
        file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/impl/*.cpp ../include/memory/cuda/*.cu ../include/memory/*.h)
        file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.cu ../include/graph/*.h)
        file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES false ../include/ops/declarable/helpers/cuda/*.cu ../include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
        file(GLOB_RECURSE HELPERS_SOURCES false ../include/helpers/impl/*.cpp ../include/helpers/*.cu ../include/helpers/*.cupp ../include/helpers/*.h)
        file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/*.cpp ../include/loops/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA false ../include/loops/*.cu)

        if (NOT BUILD_TESTS)
			CUDA_ADD_LIBRARY(${LIBND4J_NAME} SHARED cuda/NativeOps.cu cuda/NativeOpExecutioner.cu ${LOOPS_SOURCES_CUDA}
                ${CUSTOMOPS_HELPERS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES}
                ../include/cnpy/cnpy.cpp ../include/nd4jmemset.h ../include/nd4jmalloc.h
                cpu/GraphExecutioner.cpp cuda/NDArray.cu cpu/NDArrayFactory.cpp
                Environment.cpp Environment.h ${LOOPS_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
                ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${EXCEPTIONS_SOURCES} ${OPS_SOURCES})
		else()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBUILD_TESTS=true")

			CUDA_ADD_LIBRARY(${LIBND4J_NAME} STATIC cuda/NativeOps.cu cuda/NativeOpExecutioner.cu ${LOOPS_SOURCES_CUDA}
                ${CUSTOMOPS_HELPERS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES}
                ../include/cnpy/cnpy.cpp ../include/nd4jmemset.h ../include/nd4jmalloc.h
                cpu/GraphExecutioner.cpp cuda/NDArray.cu cpu/NDArrayFactory.cpp
                Environment.cpp Environment.h ${LOOPS_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
                ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${EXCEPTIONS_SOURCES} ${OPS_SOURCES})
		endif()


        if(WIN32)
            message("CUDA on Windows: enabling /EHsc")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
            SET_TARGET_PROPERTIES(${LIBND4J_NAME} PROPERTIES COMPILER_FLAGS "/EHsc /bigobj")
        endif()


		target_link_libraries(${LIBND4J_NAME} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
	    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cuda)

		install(TARGETS ${LIBND4J_NAME} DESTINATION .)
    endif(CUDA_FOUND)
elseif(CPU_BLAS)

    if ("${EXPERIMENTAL}" STREQUAL "yes")
        message("Experimental mode ENABLED")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
    endif()

    file(GLOB_RECURSE EXCEPTIONS_SOURCES false ../include/exceptions/*.cpp ../include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES false ../include/execution/*.cpp ../include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/*.cpp ../include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/*.cpp ../include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES false ../include/ops/declarable/helpers/cpu/*.cpp ../include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES false ../include/helpers/*.cpp ../include/helpers/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/*.cpp ../include/loops/*.h)

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)
    add_library(nd4jobj OBJECT cpu/NativeOps.cpp cpu/GraphExecutioner.cpp
            cpu/NativeOpExecutioner.cpp cpu/NDArray.cpp cpu/NDArrayFactory.cpp
            ../include/cnpy/cnpy.cpp  ../include/nd4jmemset.h ../include/nd4jmalloc.h
            Environment.cpp Environment.h ${LOOPS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${EXCEPTIONS_SOURCES} ${INDEXING_SOURCES} ${CUSTOMOPS_HELPERS_SOURCES}
            ${OPS_SOURCES})
    if(IOS)
        add_library(${LIBND4J_NAME}       STATIC $<TARGET_OBJECTS:nd4jobj>)
    else()
        add_library(${LIBND4J_NAME}static STATIC $<TARGET_OBJECTS:nd4jobj>)
        add_library(${LIBND4J_NAME}       SHARED $<TARGET_OBJECTS:nd4jobj>)
    endif()

    target_link_libraries(${LIBND4J_NAME} ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES})

    if ("${LIBND4J_ALL_OPS}" AND "${LIBND4J_BUILD_MINIFIER}")
        message(STATUS "Building minifier...")
        add_executable(minifier ../minifier/minifier.cpp ../minifier/graphopt.cpp)
        target_link_libraries(minifier ${LIBND4J_NAME}static ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES})
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

    if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT(APPLE))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic")
    endif()

    #install(TARGETS mySharedLib DESTINATION /some/full/path)
    install(TARGETS ${LIBND4J_NAME} DESTINATION  .)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cpu)
endif()
