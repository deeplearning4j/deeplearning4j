################################################################################
#
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

cmake_minimum_required(VERSION 3.15)
project(libnd4j)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

# Include Directories Based on OS
if(UNIX)
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(APPLE)
    message("Using Apple")
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

# Define Compiler Flags for Specific Builds
if(SD_APPLE_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_APPLE_BUILD=true -mmacosx-version-min=10.10")
endif()

if(SD_ARM_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ARM_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ARM_BUILD=true")
endif()

if(SD_ANDROID_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_ANDROID_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_ANDROID_BUILD=true")
endif()

if(SD_IOS_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_IOS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_IOS_BUILD=true")
endif()

# Windows Specific Configurations
if(WIN32 AND NOT ANDROID)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()

    message(STATUS "Include Directories:")
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()

    # Workaround for Long Command Lines
    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
endif()

# Link Time Optimization
if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Using Link Time Optimization")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# Compile Definitions for Operations
set(DEFINITIONS_CONTENT "")
if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
    message("Adding all ops due to empty op list or SD_ALL_OPS definition: SD_ALL_OPS=${SD_ALL_OPS} SD_OPS_LIST=${SD_OPS_LIST}")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
    message("_OPS: ${SD_OPS_LIST}")
    foreach(OP ${SD_OPS_LIST})
        add_compile_definitions(OP_${OP}=1)
        message(STATUS "OP: ${OP}")
        string(APPEND DEFINITIONS_CONTENT "#define OP_${OP} 1\n")
    endforeach()
endif()

# Compile Definitions for Types
list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
if(SD_TYPES_LIST_COUNT GREATER 0)
    add_compile_definitions(SD_SELECTIVE_TYPES)
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_TYPES\n")
    foreach(SD_TYPE ${SD_TYPES_LIST})
        string(TOUPPER ${SD_TYPE} SD_TYPE_UPPERCASE)
        add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
        message(STATUS "TYPE: ${SD_TYPE_UPPERCASE}")
        string(APPEND DEFINITIONS_CONTENT "#define HAS_${SD_TYPE_UPPERCASE}\n")
    endforeach()
endif()

# Write Definitions to Header File
if(OP_OUTPUT_FILE MATCHES ".h$")
    message("Definitions will be written to \"${OP_OUTPUT_FILE}\"")
    file(WRITE "${OP_OUTPUT_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
endif()

# Architecture Tuning
if(SD_ARCH MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
elseif(CMAKE_SYSTEM_NAME MATCHES "Aurora")
    set_source_files_properties(../include/graph/impl/GraphHolder.cpp PROPERTIES COMPILE_FLAGS -g0)
elseif(SD_ARCH MATCHES "power*")
    set(ARCH_TUNE "-mcpu=${SD_ARCH} -mtune=${SD_ARCH} -D__POWER")
elseif(SD_EXTENSION MATCHES "avx2")
    message("Building AVX2 binary...")
    set(ARCH_TUNE "-mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mprefetchwt1 -DSD_F16C=true -DF_AVX2=true")
    check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" NO_AVX256_SPLIT)
    if(NO_AVX256_SPLIT)
        set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
    endif()
else()
    if("${SD_ARCH}" STREQUAL "x86-64")
        message("Building x86_64 binary...")
        set(ARCH_TYPE "generic")
        add_compile_definitions(F_X64=true)
    else()
        set(ARCH_TYPE "${SD_ARCH}")
    endif()

    if(SD_EXTENSION MATCHES "avx512")
        message("Building AVX512 binary...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse2 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd -mbmi -mbmi2 -mprefetchwt1 -mclflushopt -mxsavec -mxsaves -DSD_F16C=true -DF_AVX512=true")
    endif()

    if(NOT WIN32 AND NOT SD_CUDA)
        set(ARCH_TUNE "-march=${SD_ARCH} -mtune=${ARCH_TYPE}")
    endif()
endif()

# Compiler-Specific Flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND SD_X86_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} -O${SD_OPTIMIZATION_LEVEL} -fp-model fast")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_SYSTEM_NAME MATCHES "Aurora" AND NOT SD_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} ${INFORMATIVE_FLAGS} -std=c++11")
    if(UNIX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,-z,--no-undefined,--verbose")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT APPLE AND NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic,--verbose")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic,--verbose")
    endif()

    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        # See: https://github.com/bombela/backward-cpp

        # Check if compiler is nvcc or nvcc_wrapper
        set(COMPILER_IS_NVCC false)
        get_filename_component(COMPILER_NAME ${CMAKE_CXX_COMPILER} NAME)
        if(COMPILER_NAME MATCHES "^nvcc")
            set(COMPILER_IS_NVCC TRUE)
        endif()

        if(DEFINED ENV{OMPI_CXX} OR DEFINED ENV{MPICH_CXX})
            if("$ENV{OMPI_CXX}" MATCHES "nvcc" OR "$ENV{MPICH_CXX}" MATCHES "nvcc")
                set(COMPILER_IS_NVCC TRUE)
            endif()
        endif()

        # Set C++ standard
        set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
        if(COMPILER_IS_NVCC)
            # GNU C++ extensions are not supported by nvcc
            set(CMAKE_CXX_EXTENSIONS OFF)
        endif()

        # Set C++ compiler and flags
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -lpthread -pthread -MT -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -finstrument-functions -g -O0 -fPIC")
        add_compile_definitions(SD_GCC_FUNCTRACE)
    endif()
endif()

# External Include Directories for Linux
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include")
endif()

# Ensure SD_CPU is TRUE if neither SD_CUDA nor SD_CPU is set
if(NOT SD_CUDA)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

# Set SD_LIBRARY_NAME Based on Build Type
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()

# ONEDNN and ARM Compute Sources
if(HAVE_ONEDNN)
    file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES
            ../include/ops/declarable/platform/mkldnn/*.cpp
            ../include/ops/declarable/platform/mkldnn/mkldnnUtils.h
    )
endif()

if(HAVE_ARMCOMPUTE)
    file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES
            ../include/ops/declarable/platform/armcompute/*.cpp
            ../include/ops/declarable/platform/armcompute/*.h
    )
endif()

# CUDA Configuration
if(SD_CUDA)
    message("Building with CUDA support")

    add_definitions(-D__CUDABLAS__=true)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "")
    endif()

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message("CUDA include directory: ${CUDA_INCLUDE_DIRS} with CXX compiler ${CMAKE_CXX_COMPILER_ID} SD_GCC_FUNCTRACE=${SD_GCC_FUNCTRACE}")
        include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")

        set(CMAKE_CUDA_FLAGS_DEBUG " -g")
        message("CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SD_GCC_FUNCTRACE STREQUAL "ON")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -DSD_GCC_FUNCTRACE=1 -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -finstrument-functions -g -O0")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G")
                add_compile_definitions(SD_GCC_FUNCTRACE)
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
            endif()
        endif()

        if(WIN32)
            message("Configuring CUDA libraries for Windows")
            if(NOT DEFINED CUDA_cublas_LIBRARY)
                set(CUDA_cublas_LIBRARY "${CUDA_HOME}/lib/x64/cublas.lib")
            endif()
            if(NOT DEFINED CUDA_cusolver_LIBRARY)
                set(CUDA_cusolver_LIBRARY "${CUDA_HOME}/lib/x64/cusolver.lib")
            endif()
        endif()

        if("${SD_ARCH}" MATCHES "armv8-a" AND UNIX)
            message("Adding Jetson Nano specific settings")
            # Need to manually specify stubbed links in order
            # for cublas and cusolver to be resolved
            if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CUDA_cublas_LIBRARY}" MATCHES ".*NOTFOUND.*" )
                message("Setting cublas library manually")
                set(CUDA_cublas_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so" CACHE STRING "CUDA CUBLAS LIB" FORCE)
                # Unset NOTFOUND variables
                unset(CUDA_cublas-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            if(NOT DEFINED CUDA_cusolver_LIBRARY OR CUDA_cusolver_LIBRARY MATCHES ".*NOTFOUND.*")
                message("Setting cusolver library manually for Jetson Nano")
                set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so" CACHE STRING "CUDA CUSOLVER LIB" FORCE)
                # Unset NOTFOUND variables
                unset(CUDA_cusolver-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            message("Jetson Nano cublas library: ${CUDA_cublas_LIBRARY}, cusolver library: ${CUDA_cusolver_LIBRARY}")
        endif()

        # Cap the number of registers to prevent resource exhaustion
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=40 ")

        # Define CUDA Architectures
        string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
        if(COMPUTE_CMP STREQUAL "all")
            if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
                set(CUDA_ARCH_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80")
            else()
                set(CUDA_ARCH_FLAGS "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90")
            endif()
            set(CMAKE_CUDA_ARCHITECTURES "${CUDA_DEFAULT_ARCHITECTURES}")
        elseif(COMPUTE_CMP STREQUAL "auto")
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
            set(CMAKE_CUDA_ARCHITECTURES "${CUDA_DEFAULT_ARCHITECTURES}")
        elseif(COMPUTE_CMP MATCHES "^[0-9]+$")
            # Matches user-specified compute capability
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE}")
            set(CMAKE_CUDA_ARCHITECTURES "${COMPUTE}")
        else()
            # Matches specific compute capabilities
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
        endif()

        string(REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(NOT "${CUDA_ARCH_FLAGS}" STREQUAL "")
                set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS} CACHE STRING "CUDA architectures" FORCE)
            else()
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
                    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 CACHE STRING "CUDA architectures for CUDA 11.x compilers" FORCE)
                else()
                    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 CACHE STRING "CUDA architectures for nvcc 12 and up" FORCE)
                endif()
                add_compile_definitions(CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})
            endif()
        endif()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudart -allow-unsupported-compiler -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}")

        # Gather CUDA Source Files
        file(GLOB_RECURSE PERF_SOURCES ../include/performance/*.cpp ../include/performance/*.h)
        file(GLOB_RECURSE EXCEPTIONS_SOURCES ../include/exceptions/*.cpp ../include/exceptions/*.h)
        file(GLOB_RECURSE EXEC_SOURCES ../include/execution/impl/*.cpp ../include/execution/cuda/*.cu ../include/execution/cuda/*.h ../include/execution/*.cu ../include/execution/*.h)
        file(GLOB_RECURSE TYPES_SOURCES ../include/types/*.cpp ../include/types/*.h)
        file(GLOB_RECURSE ARRAY_SOURCES ../include/array/impl/*.cpp ../include/array/cuda/*.cu ../include/array/*.h)
        file(GLOB_RECURSE MEMORY_SOURCES ../include/memory/impl/*.cpp ../include/memory/cuda/*.cu ../include/memory/*.h)
        file(GLOB_RECURSE GRAPH_SOURCES ../include/graph/*.cpp ../include/graph/*.cu ../include/graph/*.h)
        file(GLOB_RECURSE CUSTOMOPS_SOURCES ../include/ops/declarable/generic/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ../include/ops/declarable/helpers/cuda/*.cu ../include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE OPS_SOURCES ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp ../include/ops/*.h)
        file(GLOB_RECURSE HELPERS_SOURCES ../include/build_info.cpp ../include/ConstMessages.cpp ../include/helpers/*.cpp ../include/helpers/*.h)
        file(GLOB_RECURSE INDEXING_SOURCES ../include/indexing/*.cpp ../include/indexing/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES ../include/loops/impl/*.cpp ../include/loops/*.h)
        file(GLOB_RECURSE LEGACY_SOURCES ../include/legacy/impl/*.cpp ../include/legacy/*.cu ../include/legacy/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA ../include/loops/*.cu)
        file(GLOB_RECURSE COMPILATION_UNITS ../include/loops/cuda/compilation_units/*.cu.in ../include/ops/impl/compilation_units/*.cpp.in)

        set(ALL_SOURCES
                ${PERF_SOURCES}
                ${EXCEPTIONS_SOURCES}
                ${EXEC_SOURCES}
                ${TYPES_SOURCES}
                ${ARRAY_SOURCES}
                ${MEMORY_SOURCES}
                ${GRAPH_SOURCES}
                ${CUSTOMOPS_SOURCES}
                ${CUSTOMOPS_ONEDNN_SOURCES}
                ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
                ${CUSTOMOPS_GENERIC_SOURCES}
                ${OPS_SOURCES}
                ${HELPERS_SOURCES}
                ${INDEXING_SOURCES}
                ${LOOPS_SOURCES}
                ${LEGACY_SOURCES}
                ${LOOPS_SOURCES_CUDA}
        )

        # Generate Compilation Units
        foreach(FL_ITEM ${COMPILATION_UNITS})
            genCompilation(${FL_ITEM})
        endforeach()

        # Include cuDNN Sources if Available
        if(HAVE_CUDNN)
            message("cuDNN included")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ../include/ops/declarable/platform/cudnn/*.cu)
        endif()

        # Create Object Library
        if(SD_GCC_FUNCTRACE STREQUAL "ON")
            add_library(samediff_obj OBJECT
                    ${LOOPS_SOURCES_CUDA}
                    ${LEGACY_SOURCES}
                    ${HELPERS_CPP}
                    ${CUSTOMOPS_HELPERS_SOURCES}
                    ${HELPERS_SOURCES}
                    ${EXEC_SOURCES}
                    ${LOOPS_SOURCES}
                    ${ARRAY_SOURCES}
                    ${TYPES_SOURCES}
                    ${MEMORY_SOURCES}
                    ${GRAPH_SOURCES}
                    ${CUSTOMOPS_SOURCES}
                    ${INDEXING_SOURCES}
                    ${EXCEPTIONS_SOURCES}
                    ${OPS_SOURCES}
                    ${PERF_SOURCES}
                    ${CUSTOMOPS_CUDNN_SOURCES}
                    ${CUSTOMOPS_ONEDNN_SOURCES}
                    ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
                    ${CUSTOMOPS_GENERIC_SOURCES}
            )
        else()
            add_library(samediff_obj OBJECT
                    ${LOOPS_SOURCES_CUDA}
                    ${LEGACY_SOURCES}
                    ${HELPERS_CPP}
                    ${CUSTOMOPS_HELPERS_SOURCES}
                    ${HELPERS_SOURCES}
                    ${EXEC_SOURCES}
                    ${LOOPS_SOURCES}
                    ${ARRAY_SOURCES}
                    ${TYPES_SOURCES}
                    ${MEMORY_SOURCES}
                    ${GRAPH_SOURCES}
                    ${CUSTOMOPS_SOURCES}
                    ${INDEXING_SOURCES}
                    ${EXCEPTIONS_SOURCES}
                    ${OPS_SOURCES}
                    ${PERF_SOURCES}
                    ${CUSTOMOPS_CUDNN_SOURCES}
                    ${CUSTOMOPS_ONEDNN_SOURCES}
                    ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
                    ${CUSTOMOPS_GENERIC_SOURCES}
            )
        endif()

        target_include_directories(samediff_obj PUBLIC ${EXTERNAL_INCLUDE_DIRS})

        # Handle MSVC Runtime Library
        set(MSVC_RT_LIB "MultiThreadedDLL")
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        if(WIN32)
            message("Enabling /EHsc for CUDA on Windows")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /std:c++14")
        endif()

        # Create the shared library using SD_LIBRARY_NAME
        if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        endif()

        # Create the static library if needed
        if(SD_STATIC_LIB AND SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}static DESTINATION .)
        elseif(SD_STATIC_LIB)
            add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
        endif()

        # Set MSVC runtime library for the shared library
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        # Link CUDA Libraries
        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                ${CUDA_LIBRARIES}
                ${CUDA_DRIVER_LIBRARY}
                ${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${CUDNN}
                ${MKLDNN}
        )
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cuda")

        # Install CUDA Library
        install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
    endif()

    # CPU Configuration
elseif(SD_CPU)
    message("Building with CPU support")

    # Gather CPU Source Files
    file(GLOB_RECURSE PERF_SOURCES ../include/performance/*.cpp ../include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES ../include/exceptions/*.cpp ../include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES ../include/execution/*.cpp ../include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES ../include/types/*.cpp ../include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES ../include/array/*.cpp ../include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES ../include/memory/*.cpp ../include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES ../include/graph/*.cpp ../include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ../include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_GENERIC_SOURCES ../include/ops/declarable/helpers/cpu/*.cpp ../include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ../include/ops/impl/*.cpp ../include/ops/declarable/impl/*.cpp ../include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES ../include/indexing/*.cpp ../include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES ../include/build_info.cpp ../include/ConstMessages.cpp ../include/helpers/*.cpp ../include/helpers/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES ../include/legacy/impl/*.cpp ../include/legacy/cpu/*.cpp ../include/legacy/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES ../include/loops/*.cpp ../include/loops/*.h)
    file(GLOB_RECURSE COMPILATION_UNITS ../include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in ../include/loops/cpu/compilation_units/*.cpp.in ../include/helpers/cpu/loops/*.cpp.in ../include/ops/impl/compilation_units/*.cpp.in)

    set(ALL_SOURCES
            ${PERF_SOURCES}
            ${EXCEPTIONS_SOURCES}
            ${EXEC_SOURCES}
            ${TYPES_SOURCES}
            ${ARRAY_SOURCES}
            ${MEMORY_SOURCES}
            ${GRAPH_SOURCES}
            ${CUSTOMOPS_SOURCES}
            ${CUSTOMOPS_ONEDNN_SOURCES}
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
            ${CUSTOMOPS_GENERIC_SOURCES}
            ${OPS_SOURCES}
            ${HELPERS_SOURCES}
            ${INDEXING_SOURCES}
            ${LOOPS_SOURCES}
            ${LEGACY_SOURCES}
            ${LOOPS_SOURCES}
    )

    if (SD_X86_BUILD)
        # Disable platform optimizations for certain files on Linux/macOS
        set_source_files_properties(cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        set_source_files_properties(../include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
    endif()

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)

    # Remove Excluded Files if Not All Operations are Included
    if(NOT SD_ALL_OPS)
        message("Not all SD OPS INCLUDED")

        foreach(category IN ITEMS PERF_SOURCES EXCEPTIONS_SOURCES EXEC_SOURCES TYPES_SOURCES ARRAY_SOURCES MEMORY_SOURCES GRAPH_SOURCES CUSTOMOPS_SOURCES OPS_SOURCES HELPERS_SOURCES INDEXING_SOURCES LOOPS_SOURCES LEGACY_SOURCES LOOPS_SOURCES_CUDA HELPERS_CPP)
            list(GET ${category} SOURCES)
            foreach(SRC_FILE ${${category}})
                message("Checking exclusion for: ${SRC_FILE}")
                removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${category}")
            endforeach()
        endforeach()
    endif()

    # Generate Compilation Units
    foreach(FL_ITEM ${COMPILATION_UNITS})
        genCompilation(${FL_ITEM})
    endforeach()

    # Include cuDNN Sources if Available
    if(HAVE_CUDNN)
        message("cuDNN included")
        file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ../include/ops/declarable/platform/cudnn/*.cu)
    endif()

    # Create Object Library
    add_library(samediff_obj OBJECT
            ${LEGACY_SOURCES}
            ${CALLSTACK_SRC}
            ${LOOPS_SOURCES}
            ${HELPERS_SOURCES}
            ${EXEC_SOURCES}
            ${ARRAY_SOURCES}
            ${TYPES_SOURCES}
            ${MEMORY_SOURCES}
            ${GRAPH_SOURCES}
            ${CUSTOMOPS_SOURCES}
            ${EXCEPTIONS_SOURCES}
            ${INDEXING_SOURCES}
            ${CUSTOMOPS_ONEDNN_SOURCES}
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
            ${CUSTOMOPS_GENERIC_SOURCES}
            ${OPS_SOURCES}
            ${PERF_SOURCES}
    )

    target_include_directories(samediff_obj PUBLIC ${EXTERNAL_INCLUDE_DIRS})

    # Add External Dependencies
    foreach(external_dependency IN LISTS EXTERNAL_DEPENDENCY_PROJECTS)
        message("@external ${external_dependency}")
        add_dependencies(samediff_obj ${external_dependency})
    endforeach()

    # Create the shared library using SD_LIBRARY_NAME
    if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        if(ANDROID)
            # Limit the number of compile jobs on systems with few cores
            cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
            if(_logical_cores LESS 4)
                set_target_properties(${SD_LIBRARY_NAME} PROPERTIES JOB_POOL_COMPILE one_jobs)
            endif()
        endif()
    endif()

    # Create the static library if needed
    if (SD_STATIC_LIB AND SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
        set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        install(TARGETS ${SD_LIBRARY_NAME}static DESTINATION .)
    elseif(SD_STATIC_LIB)
        add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
    endif()

    # Handle MSVC Runtime Library Consistently
    set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

    # Link Libraries
    if(NOT "$ENV{CLION_IDE}")
        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                ${EXTERNAL_DEPENDENCY_LIBS}
                ${ONEDNN}
                ${ONEDNN_LIBRARIES}
                ${ARMCOMPUTE_LIBRARIES}
                ${OPENBLAS_LIBRARIES}
                ${BLAS_LIBRARIES}
                ${CPU_FEATURES}
        )
        target_include_directories(${SD_LIBRARY_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})
    endif()

    # Compiler Checks
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    # OpenMP for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OPENMP_FOUND)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()

    # Set Library Output Directories
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cpu")

    # Install the Library
    install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)

endif()

# Preprocessing Configuration
if(SD_PREPROCESS STREQUAL "ON")
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}")
    include_directories(${CMAKE_BINARY_DIR}/../../include)

    # Remove Duplicate Entries
    list(REMOVE_DUPLICATES ALL_SOURCES)

    # Define Output Directory
    set(PREPROCESSED_DIR "${CMAKE_SOURCE_DIR}/preprocessed")
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})

    # Initialize Lists
    set(PREPROCESSED_FILES)
    set(PROCESSED_SOURCES "")

    # Remove Duplicates from ALL_SOURCES
    list(REMOVE_DUPLICATES ALL_SOURCES)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

    # Create Custom Commands for Each Source File
    foreach(src IN LISTS ALL_SOURCES)
        if(NOT src IN_LIST PROCESSED_SOURCES)
            get_filename_component(src_name ${src} NAME_WE)
            get_filename_component(src_dir ${src} DIRECTORY)
            string(REPLACE "/" "_" src_dir_ ${src_dir})

            set(preprocessed_file "${PREPROCESSED_DIR}/${src_dir_}_${src_name}.i")
            message(STATUS "Processing ${src} to ${preprocessed_file}")

            if(NOT EXISTS "${preprocessed_file}")
                if(src MATCHES "\\.c$")
                    set(language "C")
                elseif(src MATCHES "\\.cpp$|\\.cxx$|\\.cc$")
                    set(language "CXX")
                else()
                    set(language "CXX")
                endif()

                get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
                list(APPEND INCLUDE_DIRS
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/array
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/execution
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/ops
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/types
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/memory
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/graph
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/exceptions
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/indexing
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/legacy
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/loops
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/loops/cpu
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/loops/cuda
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/performance
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/helpers
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/system
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/ops/declarable
                        ${CMAKE_CURRENT_SOURCE_DIR}/../include/ops/declarable/helpers
                )

                set(include_dirs "")
                foreach(dir IN LISTS INCLUDE_DIRS)
                    set(include_dirs "${include_dirs} -I${dir}")
                endforeach()

                set(compiler "${CMAKE_CXX_COMPILER}")

                message("COMMAND ${compiler} -E ${include_dirs} ${src} -o ${preprocessed_file}")
                execute_process(
                        COMMAND "${compiler}" -E ${include_dirs} "${src}" -o "${preprocessed_file}"
                        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/../"
                        RESULT_VARIABLE result
                        OUTPUT_VARIABLE stdout_output
                        ERROR_VARIABLE stderr_output
                )

                set(output_log_file "${preprocessed_file}.saved")
                file(WRITE ${output_log_file} "Standard Output:\n${stdout_output}\n")
                file(APPEND ${output_log_file} "COMMAND ${compiler} -E ${include_dirs} ${src} -o ${preprocessed_file}\n")
                file(APPEND ${output_log_file} "Standard Error:\n${stderr_output}\n")

                list(APPEND PREPROCESSED_FILES ${preprocessed_file})
            else()
                message(STATUS "Preprocessed file already exists, skipping: ${preprocessed_file}")
                list(APPEND PREPROCESSED_FILES ${preprocessed_file})
            endif()

            list(APPEND PROCESSED_SOURCES ${src})
        else()
            message(STATUS "Skipping already processed file: ${src}")
        endif()
    endforeach()

    set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

    # Create a Custom Target for All Preprocessed Files
    add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES})
endif()
