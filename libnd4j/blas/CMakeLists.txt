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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAPPLE_BUILD=true -mmacosx-version-min=10.10")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DAPPLE_BUILD=true -mmacosx-version-min=10.10")
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
    message("Building AVX2 binary...")
    set(ARCH_TUNE "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mprefetchwt1 -DSD_F16C=true -DF_AVX2=true")
ELSE()
    if ("${ARCH}" STREQUAL "x86-64")
        message("Building x86_64 binary...")
        set(ARCH_TYPE "generic")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DF_X64=true")
    else()
        set(ARCH_TYPE "${ARCH}")
    endif()

    IF(${EXTENSION} MATCHES "avx512")
        message("Building AVX512 binary...")
        # we need to set flag here, that we can use hardware f16 conversion + tell that cpu features should be tracked
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq  -mavx512cd -mbmi -mbmi2 -mprefetchwt1 -mclflushopt -mxsavec -mxsaves -DSD_F16C=true -DF_AVX512=true")
    endif()

    if (NOT WIN32)
        # we don't want this definition for msvc
        set(ARCH_TUNE "-march=${ARCH} -mtune=${ARCH_TYPE}")
    endif()
ENDIF()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang" AND X86_BUILD)
    # apple clang but not ios-arm
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O3 -fp-model fast")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
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

#if MKLDNN is enabled - we're building mkldnn-powered helpers
if (HAVE_MKLDNN)
    file(GLOB_RECURSE CUSTOMOPS_MKLDNN_SOURCES false ../include/ops/declarable/platform/mkldnn/*.cpp ../include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
endif()

if(CUDA_BLAS)
    message("Build cublas")
    find_package(CUDA)
    add_definitions(-D__CUDABLAS__=true)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set (CMAKE_CXX_FLAGS "")
    endif()

    if (CUDA_FOUND)
		message("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
		include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")

        if ("${EXPERIMENTAL}" STREQUAL "yes")
            message("Experimental mode ENABLED")
            set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
            set(EXPM " -D__ND4J_EXPERIMENTAL__=true")
        endif()

        # the only difference for debug mode here is host/device debug symbols
        set(CMAKE_CUDA_FLAGS_DEBUG " -G -g")

        # we need -fPIC on Linux/GCC
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            message("Enabling fPIC...")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
        endif()

        if(CUDA_VERSION VERSION_GREATER "9.2") # cuda 10
            if ("${COMPUTE}" STREQUAL "all")
                if (APPLE)
                    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_10 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60")
                else()
                    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_10 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70")
                endif()
            else()
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_10 ${EXPM} -w --cudart=static --expt-extended-lambda -arch=compute_${COMPUTE} -code=sm_${COMPUTE}")
            endif()
        elseif(CUDA_VERSION VERSION_GREATER "8.0") # cuda 9
            if ("${COMPUTE}" STREQUAL "all")
                if (APPLE)
                    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_9 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60")
                else()
                    set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_9 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70")
                endif()
            else()
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_9 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -arch=compute_${COMPUTE} -code=sm_${COMPUTE}")
            endif()
        elseif (CUDA_VERSION VERSION_GREATER "7.5") # cuda 8.0
            if ("${COMPUTE}" STREQUAL "all")
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_8 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60")
            else()
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_8 ${EXPM} -w --cudart=static --expt-extended-lambda --Xfatbin -compress-all -arch=compute_${COMPUTE} -code=sm_${COMPUTE}")
            endif()
        else()
            if ("${COMPUTE}" STREQUAL "all")
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_75 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -gencode arch=compute_30,code=sm_30 -gencode arch=compute_52,code=sm_52")
            else()
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -DCUDA_75 ${EXPM} -w --cudart=static --expt-extended-lambda -Xfatbin -compress-all -arch=compute_${COMPUTE} -code=sm_${COMPUTE}")
            endif()
        endif()

        file(GLOB_RECURSE PERF_SOURCES false ../include/performance/*.cpp ../include/performance/*.h)
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
        file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/impl/*.cpp ../include/loops/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA false ../include/loops/*.cu)

        if (HAVE_CUDNN)
            message("cuDNN included")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES false ../include/ops/declarable/platform/cudnn/*.cu)
        endif()

		add_library(nd4jobj OBJECT cuda/NativeOps.cu cuda/NativeOpExecutioner.cu cuda/BlasVersionHelper.cu Environment.cpp ${LOOPS_SOURCES_CUDA}
                ${CUSTOMOPS_HELPERS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES}
                ../include/cnpy/cnpy.cpp ../include/nd4jmemset.h ../include/nd4jmalloc.h
                cpu/GraphExecutioner.cpp cuda/NDArray.cu cpu/NDArrayFactory.cpp
                Environment.h ${LOOPS_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
                ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${INDEXING_SOURCES} ${EXCEPTIONS_SOURCES} ${OPS_SOURCES} ${PERF_SOURCES} ${CUSTOMOPS_CUDNN_SOURCES} ${CUSTOMOPS_MKLDNN_SOURCES})

        add_library(${LIBND4J_NAME} SHARED $<TARGET_OBJECTS:nd4jobj>)

        if (WIN32)
            message("MSVC runtime for library: ${MSVC_RT_LIB}")
        endif()

        # static library is built only if we're going to build tests, skip otherwise
        if (BUILD_TESTS)
            add_library(${LIBND4J_NAME}static STATIC $<TARGET_OBJECTS:nd4jobj>)
            set_property(TARGET ${LIBND4J_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${LIBND4J_NAME}static DESTINATION .)
        endif()

        # on windows we want to make sure we use MT or MD, but since we use it in one lib, we must use it everywhere to avoid conflicts
        set_property(TARGET nd4jobj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        set_property(TARGET ${LIBND4J_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        if(WIN32)
            message("CUDA on Windows: enabling /EHsc")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /std:c++14")
        endif()

		target_link_libraries(${LIBND4J_NAME} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDNN} ${MKLDNN})
	    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cuda)

		install(TARGETS ${LIBND4J_NAME} DESTINATION .)
    endif(CUDA_FOUND)
elseif(CPU_BLAS)

    if ("${EXPERIMENTAL}" STREQUAL "yes")
        message("Experimental mode ENABLED")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ND4J_EXPERIMENTAL__=true")
    endif()

    file(GLOB_RECURSE PERF_SOURCES false ../include/performance/*.cpp ../include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES false ../include/exceptions/*.cpp ../include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES false ../include/execution/*.cpp ../include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES false ../include/types/*.cpp ../include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES false ../include/array/*.cpp ../include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES false ../include/memory/*.cpp ../include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES false ../include/graph/*.cpp ../include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES false ../include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_GENERIC_SOURCES false ../include/ops/declarable/helpers/cpu/*.cpp ../include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES false ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp  ../include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES false ../include/indexing/*.cpp ../include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES false ../include/helpers/*.cpp ../include/helpers/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES false ../include/loops/*.cpp ../include/loops/*.h)

    if (X86_BUILD)
        # we disable platform optimizations for certains files for linux/macos
        set_source_files_properties(cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        set_source_files_properties(../include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
    endif()

    if(CHECK_VECTORIZATION)
       set(VECT_FILES cpu/NativeOps.cpp ${OPS_SOURCES} ${HELPERS_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES} ${LOOPS_SOURCES})
       if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        
        if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
            set(CHECK_VECT_FLAGS "-ftree-vectorize -fsave-optimization-record")
            #to process fsave-optimization-record we will need our cython version code
            message("Build Auto vectorization helpers")
            execute_process(COMMAND "python3" "${CMAKE_CURRENT_SOURCE_DIR}/../auto_vectorization/cython_setup.py" "build_ext" "--inplace" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../auto_vectorization/" RESULT_VARIABLE ret)
            message("build='${ret}'")

            #remove fail cases that gcc fails produce sometimes
            file(GLOB_RECURSE FAILURE_CASES false ../include/loops/cpu/compilation_units/reduce3*.cpp)
            #message("*****${FAILURE_CASES}")
            foreach(FL_ITEM ${FAILURE_CASES})
                message("Removing failure cases ${FL_ITEM}")
                list(REMOVE_ITEM VECT_FILES ${FL_ITEM})
            endforeach() 
        else()
            set(CHECK_VECT_FLAGS "-ftree-vectorize -fopt-info-vec-optimized-missed")
        endif()
        message("CHECK VECTORIZATION ${CHECK_VECT_FLAGS}")
        set_source_files_properties( ${VECT_FILES}  PROPERTIES COMPILE_FLAGS "${CHECK_VECT_FLAGS}" )
       endif()
    endif()    

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)
    add_library(nd4jobj OBJECT cpu/NativeOps.cpp cpu/GraphExecutioner.cpp
            cpu/NativeOpExecutioner.cpp cpu/NDArray.cpp cpu/NDArrayFactory.cpp
            ../include/cnpy/cnpy.cpp  ../include/nd4jmemset.h ../include/nd4jmalloc.h
            Environment.cpp Environment.h ${LOOPS_SOURCES} ${HELPERS_SOURCES} ${EXEC_SOURCES} ${ARRAY_SOURCES} ${TYPES_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${EXCEPTIONS_SOURCES} ${INDEXING_SOURCES} ${CUSTOMOPS_MKLDNN_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES}
            ${OPS_SOURCES} ${PERF_SOURCES})
    if(IOS)
        add_library(${LIBND4J_NAME}       STATIC $<TARGET_OBJECTS:nd4jobj>)
    else()
        # static library is built only if we're going to build tests, skip otherwise
        if (BUILD_TESTS)
            add_library(${LIBND4J_NAME}static STATIC $<TARGET_OBJECTS:nd4jobj>)
        endif()

        add_library(${LIBND4J_NAME}       SHARED $<TARGET_OBJECTS:nd4jobj>)
    endif()

    # we're including {MKLDNN} here in case of building from sources. in future that'll replace {MKLDNN_LIBRARIES}. same applies to BLAS
    if (NOT BLAS_LIBRARIES)
        set(BLAS_LIBRARIES "")
    endif()
    target_link_libraries(${LIBND4J_NAME} ${MKLDNN} ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES} ${BLAS_LIBRARIES} ${CPU_FEATURES})

    if ("${LIBND4J_ALL_OPS}" AND "${LIBND4J_BUILD_MINIFIER}")
        message(STATUS "Building minifier...")
        add_executable(minifier ../minifier/minifier.cpp ../minifier/graphopt.cpp)
        target_link_libraries(minifier ${LIBND4J_NAME}static ${MKLDNN_LIBRARIES} ${OPENBLAS_LIBRARIES} ${MKLDNN} ${BLAS_LIBRARIES})
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

    if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT(APPLE) AND NOT(WIN32))
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic")
        SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic")
    endif()

    install(TARGETS ${LIBND4J_NAME} DESTINATION  .)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/cpu)
endif()
