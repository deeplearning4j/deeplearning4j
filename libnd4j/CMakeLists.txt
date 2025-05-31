################################################################################
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

# Basic CMake Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Disable debug mode to reduce excessive output
# set(CMAKE_FIND_DEBUG_MODE 1)

# Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
include(CheckCXXCompilerFlag)

# Set Windows specific flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

# Options
option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_BUILD_TESTS "Build tests" OFF)
option(SD_STATIC_LIB "Build static library (ignored, only shared lib is built)" OFF)
option(SD_SHARED_LIB "Build shared library (ignored, this is the default)" ON)
option(SD_SANITIZE "Enable Address Sanitizer" OFF)
option(SD_USE_LTO "Use link time optimization" OFF)
option(PRINT_INDICES "Print indices" OFF)
option(PRINT_MATH "Print math operations" OFF)
option(SD_PREPROCESSOR "Use preprocessor" OFF)
option(SD_GCC_FUNCTRACE "Use call traces" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF)
option(SD_PTXAS "Enable ptxas verbose output" OFF)
option(SD_KEEP_NVCC_OUTPUT "Keep NVCC output files" OFF)
option(SD_PREPROCESS "Enable preprocessing" OFF)

# Handle PRINT_INDICES option
message("PRINT_INDICES: ${PRINT_INDICES}")
if(PRINT_INDICES)
    message("Added print indices compile definition")
    add_compile_definitions(PRINT_INDICES)
endif()

# Handle PRINT_MATH option
message("PRINT_MATH: ${PRINT_MATH}")
if(PRINT_MATH)
    message("Added print indices compile definition")
    add_compile_definitions(SD_PRINT_MATH)
endif()

# Set optimization level based on GCC_FUNCTRACE
if(SD_GCC_FUNCTRACE)
    message("Set optimization for functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL "0")
else()
    message("Set optimization level for no functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL "3")
endif()

message("Set default optimization level ${SD_OPTIMIZATION_LEVEL}")
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)

# Print build type and all variables
message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# Add this near the top with other option definitions
option(SD_PREPROCESSOR "Use preprocessor" OFF)

# Add this section before the flatbuffers download section
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

# Create the directory for generated files if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/generated")

# Write Definitions to include_ops.h file
set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
message("Generating include_ops.h at: ${INCLUDE_OPS_FILE}")
file(WRITE "${INCLUDE_OPS_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")

# Add the generated directory to include paths
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/generated")


print_all_variables()

# Define ARM Compute Library URLs based on architecture
set(ARM_COMPUTE_URL_ARMV7 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-armv7a-cpu-bin.tar.gz")
set(ARM_COMPUTE_URL_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-aarch64-cpu-bin.tar.gz")
set(ARM_COMPUTE_URL_ANDROID_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-android-aarch64-cpu-bin.tar.gz")

# Set the appropriate URL based on architecture
if(${SD_ARCH} MATCHES "armv7")
    set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
    message("Using ARM Compute Library for ARMv7 32-bit")
elseif(${SD_ARCH} MATCHES "armv8-a")
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (matched armv8-a)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched armv8-a)")
    endif()
elseif(${SD_ARCH} MATCHES "arm64")
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (matched arm64)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched arm64)")
    endif()
endif()

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

# Architecture Tuning
if(SD_ARCH MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
elseif(CMAKE_SYSTEM_NAME MATCHES "Aurora")
    set_source_files_properties(./include/graph/impl/GraphHolder.cpp PROPERTIES COMPILE_FLAGS -g0)
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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT SD_CUDA)
     message(STATUS "Adding GCC memory optimization flag: --param ggc-min-expand=10")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=10 ${ARCH_TUNE} ${INFORMATIVE_FLAGS} -std=c++11")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --param ggc-min-expand=10")

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

        set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
        if(COMPILER_IS_NVCC)
            set(CMAKE_CXX_EXTENSIONS OFF)
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fstack-protector -fstack-protector-all  -Wall  -Wextra -Werror -Wno-return-type -Wno-error=int-in-bool-context -Wno-unused-variable -Wno-error=implicit-fallthrough -Wno-return-type -Wno-unused-parameter -Wno-error=unknown-pragmas -ggdb3 -lpthread -pthread -MT -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -finstrument-functions  -O0 -fPIC")
        add_compile_definitions(SD_GCC_FUNCTRACE)
    endif()
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

# Set default engine
if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
endif()

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

# Determine platform type more accurately
set(SD_X86_BUILD false)
set(SD_ARM_BUILD false)

if(SD_ANDROID_BUILD)
    if(ANDROID_ABI MATCHES "x86_64")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86-64")
    elseif(ANDROID_ABI MATCHES "x86")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86")
    elseif(ANDROID_ABI MATCHES "arm64-v8a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "arm64-v8a")
    elseif(ANDROID_ABI MATCHES "armeabi-v7a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "armv7-a")
    endif()
elseif(NOT SD_IOS_BUILD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_X86_BUILD true)
        if(NOT DEFINED SD_ARCH)
            set(SD_ARCH "x86-64")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
        set(SD_ARM_BUILD true)
    endif()
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")

# Platform specific compiler flags
if(SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif(APPLE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "armv8-a")
        set(SD_ARCH armv8-a)
        set(SD_X86_BUILD false)
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
    set(SD_X86_BUILD true)
    if(SD_CUDA)
        set(CMAKE_CXX_FLAGS_RELEASE  "-D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  "  /FS /EHsc")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
        set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
    endif()
else()
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")

    if(SD_SANITIZE)
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIE   -lpthread -ftls-model=local-dynamic  -static-libasan  -fsanitize=${SD_SANITIZERS}  -fno-sanitize-recover=all")
        message("Using sanitizers: ${SD_SANITIZERS} - note you can not use both thread and address sanitizer at the same time. Be careful what sanitizers you specify.
         Note that address and undefined can not be used at the same time or an address overlap error will occur.  See: https://github.com/google/sanitizers/issues/856
         FOR THREADS USE: thread,undefined,float-divide-by-zero,float-cast-overflow
         FOR ADDRESS USE: address,undefined,float-divide-by-zero,float-cast-overflow")
        if(SD_CPU)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${SANITIZE_FLAGS}")
        endif()
        if(SD_CUDA)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${SANITIZE_FLAGS} -lpthread -ftls-model=local-dynamic --relocatable-device-code=true")
        endif()
    endif()
endif()

if(SD_NATIVE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*")
        set(SD_X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# External Include Directories
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include")
endif()

# Initialize job pools for parallel builds
set_property(GLOBAL PROPERTY JOB_POOLS one_jobs=1 two_jobs=2)

# Common functions for both CPU and CUDA builds
function(removeFileIfExcluded)
    cmake_parse_arguments(
            PARSED_ARGS
            ""
            "FILE_ITEM"
            "LIST_ITEM"
            ${ARGN}
    )
    file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
    string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)

    if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
        set(local_list ${${PARSED_ARGS_LIST_ITEM}})
        set(file_removed FALSE)

        foreach(OP ${SD_OPS_LIST})
            string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)

            if(${NOT_EXCLUDED_OP_IDX} LESS 0)
                list(REMOVE_ITEM local_list "${PARSED_ARGS_FILE_ITEM}")
                set(file_removed TRUE)
                break()
            endif()
        endforeach()

        if(file_removed)
            set(${PARSED_ARGS_LIST_ITEM} ${local_list} PARENT_SCOPE)
        endif()
    endif()
endfunction()
function(genCompilation FL_ITEM)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)

    set(EXTENSION "cpp")
    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)

    set(SD_FLOAT_TYPES_GEN 0)
    set(SD_INTEGER_TYPES_GEN 0)
    set(SD_COMMON_TYPES_GEN 0)
    set(SD_PAIRWISE_TYPES_GEN 0)
    set(RANGE_STOP -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    set(SD_INTEGER_TYPES_END 7)
    set(SD_COMMON_TYPES_END 12)
    set(SD_FLOAT_TYPES_END 3)
    set(SD_PAIRWISE_TYPES_END 12)

    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set(SD_INTEGER_TYPES_GEN 1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set(SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set(SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set(SD_PAIRWISE_TYPES_GEN 1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()
    endforeach()

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/compilation_units")

    if(RANGE_STOP GREATER -1)
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set(SD_FLOAT_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set(SD_INTEGER_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set(SD_COMMON_TYPES_GEN 0)
            endif()

            set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            configure_file("${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
            list(APPEND CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE})
        endforeach()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
    list(LENGTH COMB_LIST COMB_COUNT)

    if(NOT (COMBINATION_TYPE EQUAL 3 OR COMBINATION_TYPE EQUAL 2))
        message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 3 or 2.")
    endif()

    if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR
    (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
        message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).")
    endif()

    if(COMBINATION_TYPE EQUAL 3)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        list(GET COMB_LIST 2 COMB3)
        set(PLACEHOLDER1 "@COMB1@")
        set(PLACEHOLDER2 "@COMB2@")
        set(PLACEHOLDER3 "@COMB3@")
    elseif(COMBINATION_TYPE EQUAL 2)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        set(PLACEHOLDER1 "@COMB1@")
        set(PLACEHOLDER2 "@COMB2@")
    endif()

    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    if(COMBINATION_TYPE EQUAL 3)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" TEMP_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" TEMP_CONTENT "${TEMP_CONTENT}")
        string(REPLACE "${PLACEHOLDER3}" "${COMB3}" FINAL_CONTENT "${TEMP_CONTENT}")
    elseif(COMBINATION_TYPE EQUAL 2)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" FINAL_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}")
    endif()

    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
    string(REPLACE "_template_" "_" OUTPUT_BASE_NAME "${TEMPLATE_BASE}")

    if(COMBINATION_TYPE EQUAL 3)
        set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}_${COMB3}.cpp")
    elseif(COMBINATION_TYPE EQUAL 2)
        set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}.cpp")
    endif()

    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")
    file(WRITE "${GENERATED_FILE}" "${FINAL_CONTENT}")

    list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)

    message(STATUS "Generated Instantiation File: ${GENERATED_FILE}")
endfunction()

function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    list(GET COMB_LIST 0 COMB1)
    list(GET COMB_LIST 1 COMB2)
    list(GET COMB_LIST 2 COMB3)

    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)

    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    string(REGEX MATCH "([a-zA-Z0-9_:]+),[ \n\t]*::([a-zA-Z0-9_]+)" FUNCTION_MATCH "${TEMPLATE_CONTENT}")
    set(CLASS_NAME ${CMAKE_MATCH_1})
    set(METHOD_NAME ${CMAKE_MATCH_2})

    string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")

    string(REGEX MATCH "::${METHOD_NAME}\\(([^;]+)\\);" FUNC_ARGS_MATCH "${TEMPLATE_CONTENT}")
    set(FUNCTION_ARGS "${CMAKE_MATCH_1}")

    set(PARAM_COUNT 0)
    set(SIGNATURE_ID "")

    string(REPLACE "," ";" ARGS_LIST "${FUNCTION_ARGS}")
    list(LENGTH ARGS_LIST PARAM_COUNT)

    foreach(ARG ${ARGS_LIST})
        string(REGEX MATCH "^[^*& \t]+" TYPE_NAME "${ARG}")
        if(TYPE_NAME)
            string(APPEND SIGNATURE_ID "_${TYPE_NAME}")
        endif()
    endforeach()

    if(SIGNATURE_ID MATCHES ".{30,}")
        string(MD5 SIGNATURE_HASH "${SIGNATURE_ID}")
        string(SUBSTRING "${SIGNATURE_HASH}" 0 8 SIGNATURE_ID)
        set(SIGNATURE_ID "_h${SIGNATURE_ID}")
    endif()

    set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}${SIGNATURE_ID}_${COMB1}_${COMB2}_${COMB3}.cu")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    if(EXISTS "${GENERATED_FILE}")
        list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
        set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
        return()
    endif()

    set(START_MARKER "ITERATE_COMBINATIONS_3")
    string(FIND "${TEMPLATE_CONTENT}" "${START_MARKER}" START_POS)
    if(START_POS EQUAL -1)
        message(FATAL_ERROR "Could not find ITERATE_COMBINATIONS_3 in template file ${TEMPLATE_FILE}")
    endif()

    string(SUBSTRING "${TEMPLATE_CONTENT}" 0 ${START_POS} HEADER_CONTENT)

    set(NEW_CONTENT "${HEADER_CONTENT}\n\n// Single function instantiation for ${CLASS_NAME}::${METHOD_NAME}\n")
    string(APPEND NEW_CONTENT "template void ${CLASS_NAME}::${METHOD_NAME}<SD_SINGLE_TYPE_${COMB1}, SD_SINGLE_TYPE_${COMB2}, SD_SINGLE_TYPE_${COMB3}>(${FUNCTION_ARGS});\n")

    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    file(WRITE "${GENERATED_FILE}" "${NEW_CONTENT}")

    set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA)
    list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)

    message(STATUS "Generated: ${GENERATED_FILE}")
endfunction()

# Configure BLAS for CPU builds
if(NOT SD_CUDA)
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if("${OPENBLAS_PATH}" STREQUAL "")
        if(NOT APPLE)
            set(BLA_VENDOR "OpenBLAS")
        endif()

        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            message("Found external BLAS implementation: ${BLAS_LIBRARIES} ")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        endif()
    else()
        set(HAVE_OPENBLAS 1)
        message("Setting openblas")
        include_directories(${OPENBLAS_PATH}/include/)
        link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
        set(OPENBLAS_LIBRARIES openblas)
    endif()
endif()

# ARM Compute configuration
set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)
if(HELPERS_armcompute AND NOT HELPERS_armcompute STREQUAL "false" AND NOT HELPERS_armcompute STREQUAL "FALSE" AND NOT HELPERS_armcompute STREQUAL "OFF")
    option(LIBND4J_BUILD_WITH_ARMCOMPUTE "Build with ARM Compute Library support" ON)

    if(LIBND4J_BUILD_WITH_ARMCOMPUTE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64"))
        message(STATUS "Configuring ARM Compute Library download using ExternalProject_Add")
        include(ExternalProject)

        set(ARMCOMPUTE_VERSION "v25.04")
        set(ARMCOMPUTE_ARCH "aarch64")
        set(ARMCOMPUTE_PLATFORM "linux")
        set(ARMCOMPUTE_FLAVOR "cpu")
        set(ARMCOMPUTE_INSTALL_DIR "${CMAKE_BINARY_DIR}/armcompute_install")
        set(ARMCOMPUTE_SOURCE_DIR "${CMAKE_BINARY_DIR}/armcompute_external/src")
        set(ARMCOMPUTE_PKG_NAME "arm_compute-${ARMCOMPUTE_VERSION}-${ARMCOMPUTE_PLATFORM}-${ARMCOMPUTE_ARCH}-${ARMCOMPUTE_FLAVOR}-bin")
        set(ARMCOMPUTE_URL "https://github.com/ARM-software/ComputeLibrary/releases/download/${ARMCOMPUTE_VERSION}/${ARMCOMPUTE_PKG_NAME}.tar.gz")
        set(ARMCOMPUTE_LIB_SUBDIR "lib/armv8a-neon")

        ExternalProject_Add(armcompute_external
                PREFIX            "${CMAKE_BINARY_DIR}/armcompute_external"
                URL               "${ARMCOMPUTE_URL}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                SOURCE_DIR        "${ARMCOMPUTE_SOURCE_DIR}"
                CONFIGURE_COMMAND ""
                BUILD_COMMAND     ""
                INSTALL_COMMAND   ${CMAKE_COMMAND} -E remove_directory "${ARMCOMPUTE_INSTALL_DIR}" &&
                                  ${CMAKE_COMMAND} -E copy_directory "${ARMCOMPUTE_SOURCE_DIR}/${ARMCOMPUTE_PKG_NAME}/include" "${ARMCOMPUTE_INSTALL_DIR}/include" &&
                                  ${CMAKE_COMMAND} -E copy_directory "${ARMCOMPUTE_SOURCE_DIR}/${ARMCOMPUTE_PKG_NAME}/${ARMCOMPUTE_LIB_SUBDIR}" "${ARMCOMPUTE_INSTALL_DIR}/lib"
                BUILD_BYPRODUCTS  "${ARMCOMPUTE_INSTALL_DIR}/include/arm_compute/core/CL/CLKernelLibrary.h"
                                  "${ARMCOMPUTE_INSTALL_DIR}/lib/libarm_compute.so"
                                  "${ARMCOMPUTE_INSTALL_DIR}/lib/libarm_compute_graph.so"
                BUILD_IN_SOURCE   1
        )

        add_library(armcompute_interface INTERFACE)
        target_include_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/include")
        target_link_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/lib")
        target_link_libraries(armcompute_interface INTERFACE arm_compute arm_compute_graph)
        add_dependencies(armcompute_interface armcompute_external)

        set(ARMCOMPUTE_FOUND TRUE)
        set(ARMCOMPUTE_INCLUDE_DIRS "${ARMCOMPUTE_INSTALL_DIR}/include")
        set(ARMCOMPUTE_LIBRARIES armcompute_interface)

    else()
        message(STATUS "Skipping ARM Compute setup via ExternalProject_Add (disabled or not AArch64)")
        set(ARMCOMPUTE_FOUND FALSE)
    endif()
endif()

# Add flatbuffers dependency using ExternalProject_Add directly
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    set(FLATBUFFERS_BUILD_FLATC "ON" CACHE STRING "Enable flatc build" FORCE)
else()
    set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Disable flatc build" FORCE)
endif()

include(ExternalProject)
ExternalProject_Add(flatbuffers_external
    GIT_REPOSITORY    https://github.com/google/flatbuffers/
    GIT_TAG           v25.2.10
    SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src"
    BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build"
    CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=Release
        -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc"
                      "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a"
)

# Make the flatbuffers include path globally available for all targets in this project.
# This ensures that both the main library and external processes like JNI compilation can find the headers.
include_directories("${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")

# Define an INTERFACE library to represent the flatbuffers dependency for other targets.
add_library(flatbuffers_interface INTERFACE)

# Define the full path to the static library that will be built by the external project
set(FLATBUFFERS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/libflatbuffers.a")

# Tell any target linking against our interface to link with the actual library file
target_link_libraries(flatbuffers_interface INTERFACE ${FLATBUFFERS_LIBRARY})

# Crucial: Make our interface target depend on the external project completing its build.
add_dependencies(flatbuffers_interface flatbuffers_external)

# Correct, build-time Flatbuffers code generation
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")

    set(MAIN_GENERATED_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h")

    add_custom_command(
        OUTPUT ${MAIN_GENERATED_HEADER}
        COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}" bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
        DEPENDS flatbuffers_external
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running flatc to generate C++ headers"
        VERBATIM
    )

    add_custom_target(generate_flatbuffers_headers
        DEPENDS ${MAIN_GENERATED_HEADER}
    )

    add_custom_command(
        TARGET generate_flatbuffers_headers POST_BUILD
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Copying generated Java files"
        VERBATIM
    )
endif()

# OneDNN configuration - FIXED: Removed problematic flatbuffers dependency
if(NOT DEFINED ONEDNN_PROCESSING_DONE)
    set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
endif()
if(HELPERS_onednn AND NOT HELPERS_onednn STREQUAL "false" AND NOT HELPERS_onednn STREQUAL "FALSE" AND NOT HELPERS_onednn STREQUAL "OFF")
    include(ExternalProject)
    message(STATUS "Configuring OneDNN dependency using ExternalProject_Add")
    set(HAVE_ONEDNN TRUE)

    set(ONEDNN_INSTALL_DIR "${CMAKE_BINARY_DIR}/onednn_install")
    set(ONEDNN_SOURCE_DIR "${CMAKE_BINARY_DIR}/onednn_external/src")
    set(ONEDNN_BUILD_DIR "${CMAKE_BINARY_DIR}/onednn_external/build")

    ExternalProject_Add(onednn_external
            PREFIX            "${CMAKE_BINARY_DIR}/onednn_external"
            GIT_REPOSITORY    "https://github.com/KonduitAI/oneDNN.git"
            GIT_TAG           "master"
            SOURCE_DIR        "${ONEDNN_SOURCE_DIR}"
            BINARY_DIR        "${ONEDNN_BUILD_DIR}"
            CMAKE_ARGS
                -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DDNNL_LIBRARY_TYPE=STATIC
                -DDNNL_BUILD_TESTS=OFF
                -DDNNL_BUILD_EXAMPLES=OFF
                -DDNNL_VERBOSE=OFF
            BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE}
            INSTALL_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config ${CMAKE_BUILD_TYPE}
            BUILD_BYPRODUCTS  "${ONEDNN_INSTALL_DIR}/include/dnnl.h"
                              "${ONEDNN_INSTALL_DIR}/lib/libdnnl.a"
            LOG_DOWNLOAD      ON
            LOG_CONFIGURE     ON
            LOG_BUILD         ON
            LOG_INSTALL       ON
            # FIXED: Removed "DEPENDS flatbuffers_external" which was causing the error
    )

    add_library(onednn_interface INTERFACE)
    target_include_directories(onednn_interface INTERFACE
            "${ONEDNN_INSTALL_DIR}/include"
    )
    target_link_libraries(onednn_interface INTERFACE
            "${ONEDNN_INSTALL_DIR}/lib/libdnnl.a"
    )
    add_dependencies(onednn_interface onednn_external)
    set(ONEDNN onednn_interface)

    message(STATUS "OneDNN configured via ExternalProject_Add. Build artifacts will be in ${ONEDNN_INSTALL_DIR}")
else()
    message(STATUS "Skipping OneDNN setup via ExternalProject_Add (HELPERS_onednn is OFF)")
    set(HAVE_ONEDNN FALSE)
    set(ONEDNN "")
endif()

# cuDNN configuration
if(HELPERS_cudnn AND NOT HELPERS_cudnn STREQUAL "false" AND NOT HELPERS_cudnn STREQUAL "FALSE" AND NOT HELPERS_cudnn STREQUAL "OFF")
    if(NOT SD_CUDA)
        message(FATAL_ERROR "Can't build cuDNN on non-CUDA platform")
    endif()

    SET(CUDNN_LIBNAME "cudnn")

    if(DEFINED ENV{CUDNN_ROOT_DIR})
        message("Using cudnn root directory from environment")
        set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
    endif()
    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        message("Using cuda root directory from environment")
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
    endif()
    message("Cudnn root dir ${CUDNN_ROOT_DIR} CUDA TOOLKIT ROOT DIR ${CUDA_TOOLKIT_ROOT_DIR}")

    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} /usr/local/cuda
            ENV CUDNN_ROOT_DIR ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES cuda/include include include/cuda)

    find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
            HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} /usr/local/cuda
            ENV CUDNN_ROOT_DIR ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 cuda/lib/x64 lib/x64)

    message("CUDNN_INCLUDE_DIR search result: ${CUDNN_INCLUDE_DIR}")
    message("CUDNN_LIBRARY search result: ${CUDNN_LIBRARY}")

    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message("Found cuDNN: include at ${CUDNN_INCLUDE_DIR}, library at ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true)
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        message(WARNING "cuDNN not found. Continuing without cuDNN support.")
        message("Searched for include in: ${CUDNN_ROOT_DIR}, ${CUDA_TOOLKIT_ROOT_DIR}, and system paths")
        message("Searched for library '${CUDNN_LIBNAME}' in: ${CUDNN_ROOT_DIR}, ${CUDA_TOOLKIT_ROOT_DIR}, and system paths")
        set(HAVE_CUDNN false)
    endif()
endif()

# Configuration file
configure_file(include/config.h.in include/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set type combinations for CUDA generation
set(COMBINATIONS_3
        "0,0,0" "0,0,1" "0,0,2" "0,1,0" "0,1,1" "0,1,2" "0,2,0" "0,2,1" "0,2,2"
        "1,0,0" "1,0,1" "1,0,2" "1,1,0" "1,1,1" "1,1,2" "1,2,0" "1,2,1" "1,2,2"
        "2,0,0" "2,0,1" "2,0,2" "2,1,0" "2,1,1" "2,1,2" "2,2,0" "2,2,1" "2,2,2"
)
set(COMBINATIONS_2
        "0,0" "0,1" "1,0" "1,1" "0,2" "2,0" "1,2" "2,1" "2,2"
)

# Define template locations
set(INSTANTIATION_TEMPLATES_3
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"
)
set(INSTANTIATION_TEMPLATES_2
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_double.cpp.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_single.cpp.in"
)

# CUDA Configuration
if(SD_CUDA)
    message("Building with CUDA support")

    # [FIX] Force the '--allow-unsupported-compiler' flag into the CMake cache
    # before enabling the CUDA language. This is the correct way to ensure the flag
    # is used during the initial compiler identification step, which otherwise fails
    # on systems with MSVC versions newer than what the CUDA Toolkit officially supports.
    if(WIN32)
        message("Forcing Windows-specific CUDA flag: --allow-unsupported-compiler")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler" CACHE STRING "Initial CUDA compiler flags" FORCE)
    endif()

    enable_language(CUDA)
    add_definitions(-D__CUDABLAS__=true)
    find_package(CUDA REQUIRED)

    if(CUDA_FOUND)
        message("CUDA include directory: ${CUDA_INCLUDE_DIRS} with CXX compiler ${CMAKE_CXX_COMPILER_ID} SD_GCC_FUNCTRACE=${SD_GCC_FUNCTRACE}")
        include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")

        set(CMAKE_CUDA_FLAGS_DEBUG " -g")
        message("CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

        if("${SD_PTXAS}" STREQUAL "ON")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
        endif()

        if(SD_KEEP_NVCC_OUTPUT)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --keep")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SD_GCC_FUNCTRACE STREQUAL "ON")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Werror -Wall   -Wno-return-type  -Wno-unknown-pragmas  -Wno-unused-variable -Wno-unused-parameter  -Wreturn-type -W -ggdb3 -fPIC -DSD_GCC_FUNCTRACE=1 -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -finstrument-functions  -O0")
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G")
                add_compile_definitions(SD_GCC_FUNCTRACE)
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
            endif()
        endif()

        if(WIN32)
            # This configuration is now handled by the specific library settings below
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
            if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CUDA_cublas_LIBRARY}" MATCHES ".*NOTFOUND.*" )
                message("Setting cublas library manually")
                set(CUDA_cublas_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so" CACHE STRING "CUDA CUBLAS LIB" FORCE)
                unset(CUDA_cublas-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            if(NOT DEFINED CUDA_cusolver_LIBRARY OR CUDA_cusolver_LIBRARY MATCHES ".*NOTFOUND.*")
                message("Setting cusolver library manually for Jetson Nano")
                set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so" CACHE STRING "CUDA CUSOLVER LIB" FORCE)
                unset(CUDA_cusolver-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            message("Jetson Nano cublas library: ${CUDA_cublas_LIBRARY}, cusolver library: ${CUDA_cusolver_LIBRARY}")
        endif()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=128")

        string(TOLOWER "${COMPUTE}" COMPUTE_CMP)
        if(COMPUTE_CMP STREQUAL "all")
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90")
        elseif(COMPUTE_CMP STREQUAL "auto")
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
        else()
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
            string(REPLACE ";" " " CUDA_ARCH_FLAGS "${TMP_CUDA_ARCH_FLAGS}")
        endif()

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudart -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS}")

        message("Final CUDA flags: ${CMAKE_CUDA_FLAGS}")

        add_compile_definitions(SD_CUDA=true)
        set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")

        file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
        file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
        file(GLOB_RECURSE EXEC_SOURCES ./include/execution/impl/*.cpp ./include/execution/cuda/*.cu ./include/execution/cuda/*.h ./include/execution/*.cu ./include/execution/*.h)
        file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
        file(GLOB_RECURSE ARRAY_SOURCES ./include/array/cuda/*.cu ./include/array/cuda/*.chpp ./include/array/impl/*.cpp ./include/array/cuda/*.cu ./include/array/*.h)
        file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/impl/*.cpp ./include/memory/cuda/*.cu ./include/memory/*.h)
        file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.cu ./include/graph/*.h)
        file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES ./include/ops/declarable/helpers/cuda/*.cu ./include/ops/declarable/helpers/impl/*.cpp)
        file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
        file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/cuda/*.cu ./include/helpers/*.h)
        file(GLOB CPU_HELPERS_TO_EXCLUDE ./include/helpers/cpu/*.cpp)
        list(REMOVE_ITEM HELPERS_SOURCES ${CPU_HELPERS_TO_EXCLUDE})
        file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/impl/*.cpp ./include/loops/*.h)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/*.cu ./include/legacy/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA  ./include/loops/*.cu ./include/loops/cuda/**/*.cu)
        file(GLOB_RECURSE COMPILATION_UNITS ./include/loops/cuda/compilation_units/*.cu.in)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
        foreach(FL_ITEM ${COMPILATION_UNITS})
            genCompilation(${FL_ITEM})
        endforeach()

        set(ALL_SOURCES
                ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES}
                ${ARRAY_SOURCES} ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES}
                ${CUSTOMOPS_HELPERS_SOURCES} ${OPS_SOURCES} ${HELPERS_SOURCES} ${INDEXING_SOURCES}
                ${LOOPS_SOURCES} ${LEGACY_SOURCES} ${LOOPS_SOURCES_CUDA} ${CUSTOMOPS_GENERIC_SOURCES}
        )

        if(HAVE_CUDNN)
            message("cuDNN included")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES ${CUSTOMOPS_CUDNN_SOURCES})
        endif()

        set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
        add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})

        # FIXED: Use flatbuffers_interface instead of flatbuffers_external
        add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)

        if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
            add_dependencies(${OBJECT_LIB_NAME} generate_flatbuffers_headers)
        endif()

        target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})
        set_property(TARGET ${OBJECT_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        if(WIN32)
            message("Enabling /EHsc for CUDA on Windows")
            target_compile_options(${OBJECT_LIB_NAME} INTERFACE "/EHsc" "/bigobj")
            target_compile_options(${OBJECT_LIB_NAME} PRIVATE "/EHsc" "/bigobj")
        endif()

        add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
        set_target_properties(${SD_LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY} ${CUDNN} flatbuffers_interface
        )

        install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)
    endif()
else() # Start of CPU Build Section
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    set(CUSTOMOPS_GENERIC_SOURCES "")

    file(GLOB_RECURSE COMPILATION_UNITS
            ./include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
            ./include/loops/cpu/compilation_units/*.cpp.in
            ./include/helpers/cpu/loops/*.cpp.in)

    foreach(FL_ITEM ${COMPILATION_UNITS})
        genCompilation(${FL_ITEM})
    endforeach()

    file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp ./include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp ./include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp ./include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL_SOURCES ./include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp  ./include/helpers/cpu/*.cpp ./include/helpers/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp ./include/legacy/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp ./include/loops/*.h)

    set(ALL_SOURCES "")

    set(CUSTOMOPS_ONEDNN_SOURCES "")
    if(HAVE_ONEDNN)
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES_TMP ./include/ops/declarable/platform/mkldnn/*.cpp ./include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
        set(CUSTOMOPS_ONEDNN_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES_TMP})
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES})
    endif()

    set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
    if(HAVE_ARMCOMPUTE)
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES_TMP ./include/ops/declarable/platform/armcompute/*.cpp ./include/ops/declarable/platform/armcompute/*.h)
        set(CUSTOMOPS_ARMCOMPUTE_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES_TMP})
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
    endif()

    if (SD_X86_BUILD)
        set_source_files_properties(./include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
    endif()

    set(STATIC_SOURCES_TO_CHECK
            ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
            ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}
            ${LEGACY_SOURCES} ${LOOPS_SOURCES} ${CUSTOMOPS_ONEDNN_SOURCES} ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
    )

    if(NOT SD_ALL_OPS)
        message("Not all SD OPS INCLUDED - Filtering sources")
        set(FILTERED_STATIC_SOURCES "")
        foreach(SRC_FILE ${STATIC_SOURCES_TO_CHECK})
            set(temp_list "${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "temp_list")
            if(temp_list)
                list(APPEND FILTERED_STATIC_SOURCES "${SRC_FILE}")
            else()
                message("Excluding file due to op restrictions: ${SRC_FILE}")
            endif()
        endforeach()
        list(REMOVE_ITEM ALL_SOURCES ${STATIC_SOURCES_TO_CHECK})
        list(APPEND ALL_SOURCES ${FILTERED_STATIC_SOURCES})
    else()
        list(APPEND ALL_SOURCES
                ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES}
                ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
                ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}
                ${LEGACY_SOURCES} ${LOOPS_SOURCES}
        )
        list(REMOVE_DUPLICATES ALL_SOURCES)
    endif()

    list(APPEND ALL_SOURCES ${CUSTOMOPS_GENERIC_SOURCES})

    set(PARTITION_SOURCES "")
    set(CPU_INST_DIR "${CMAKE_BINARY_DIR}/cpu_instantiations")
    file(MAKE_DIRECTORY "${CPU_INST_DIR}")
    set(GENERATED_PARTITION_SOURCES_TEMP "")

    foreach(TEMPLATE_FILE ${INSTANTIATION_TEMPLATES_3})
        foreach(COMBINATION ${COMBINATIONS_3})
            set(CUSTOMOPS_GENERIC_SOURCES "")
            genPartitionCombination(${TEMPLATE_FILE} 3 ${COMBINATION} "${CPU_INST_DIR}")
            list(APPEND GENERATED_PARTITION_SOURCES_TEMP ${CUSTOMOPS_GENERIC_SOURCES})
        endforeach()
    endforeach()

    foreach(TEMPLATE_FILE ${INSTANTIATION_TEMPLATES_2})
        foreach(COMBINATION ${COMBINATIONS_2})
            set(CUSTOMOPS_GENERIC_SOURCES "")
            genPartitionCombination(${TEMPLATE_FILE} 2 ${COMBINATION} "${CPU_INST_DIR}")
            list(APPEND GENERATED_PARTITION_SOURCES_TEMP ${CUSTOMOPS_GENERIC_SOURCES})
        endforeach()
    endforeach()
    set(PARTITION_SOURCES ${GENERATED_PARTITION_SOURCES_TEMP})

    list(APPEND ALL_SOURCES ${PARTITION_SOURCES})
    list(REMOVE_DUPLICATES ALL_SOURCES)

    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})

    # FIXED: Use flatbuffers_interface instead of flatbuffers_external
    add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)

    if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
        add_dependencies(${OBJECT_LIB_NAME} generate_flatbuffers_headers)
    endif()

    target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})
    set_property(TARGET ${OBJECT_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

    add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
    set_target_properties(${SD_LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
    set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

    if(ANDROID)
        cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
        if(_logical_cores LESS 4)
            set_target_properties(${SD_LIBRARY_NAME} PROPERTIES JOB_POOL_COMPILE one_jobs)
        endif()
    endif()

    target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
            ${ONEDNN}
            ${ARMCOMPUTE_LIBRARIES}
            ${OPENBLAS_LIBRARIES}
            ${BLAS_LIBRARIES}
            flatbuffers_interface
    )

    install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    # OpenMP for GCC
    # OpenMP Configuration
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "OpenMP found, linking OpenMP::OpenMP_CXX")
            target_link_libraries(${OBJECT_LIB_NAME} PUBLIC OpenMP::OpenMP_CXX)
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
        else()
            message(WARNING "OpenMP not found, falling back to manual configuration")
            target_compile_options(${OBJECT_LIB_NAME} INTERFACE "-fopenmp")
            target_compile_options(${OBJECT_LIB_NAME} PRIVATE "-fopenmp")
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC "-fopenmp")
        endif()
    endif()
endif() # End of CPU Build Section

# Add tests if enabled
if(SD_BUILD_TESTS)
    include(CTest)
    set(SD_ALL_OPS true)
    enable_testing()
    add_subdirectory(tests_cpu)
endif()

# Preprocessing Configuration
if(SD_PREPROCESS STREQUAL "ON")
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}")
    include_directories(${CMAKE_BINARY_DIR}/.././include)

    get_target_property(FINAL_ALL_SOURCES ${OBJECT_LIB_NAME} SOURCES)
    list(REMOVE_DUPLICATES FINAL_ALL_SOURCES)

    set(PREPROCESSED_DIR "${CMAKE_SOURCE_DIR}/preprocessed")
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})

    set(PREPROCESSED_FILES)
    set(PROCESSED_SOURCES "")

    foreach(src IN LISTS FINAL_ALL_SOURCES)
        if(NOT EXISTS ${src} OR NOT src MATCHES "\\.(c|cpp|cxx|cc|cu)$")
            continue()
        endif()

        if(NOT src IN_LIST PROCESSED_SOURCES)
            get_filename_component(src_name ${src} NAME_WE)
            get_filename_component(src_path ${src} PATH)

            file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${src_path})
            string(REPLACE "/" "_" src_dir_ "${rel_path}")

            if(IS_ABSOLUTE ${src} AND NOT src MATCHES "^${CMAKE_SOURCE_DIR}")
                file(RELATIVE_PATH rel_path_build ${CMAKE_BINARY_DIR} ${src_path})
                string(REPLACE "/" "_" src_dir_build_ "${rel_path_build}")
                set(src_dir_ "build_${src_dir_build_}")
            endif()

            set(preprocessed_file "${PREPROCESSED_DIR}/${src_dir_}_${src_name}.i")
            message(STATUS "Processing ${src} to ${preprocessed_file}")

            if(NOT EXISTS "${preprocessed_file}")
                set(compiler "")
                set(lang_flags "")
                set(includes_list "")
                get_target_property(includes_list ${OBJECT_LIB_NAME} INCLUDE_DIRECTORIES)
                get_target_property(compile_defs ${OBJECT_LIB_NAME} COMPILE_DEFINITIONS)
                get_target_property(compile_opts ${OBJECT_LIB_NAME} COMPILE_OPTIONS)

                set(include_flags "")
                foreach(dir IN LISTS includes_list)
                    string(APPEND include_flags " -I\"${dir}\"")
                endforeach()

                set(defs_flags "")
                foreach(def IN LISTS compile_defs)
                    string(APPEND defs_flags " -D${def}")
                endforeach()

                if(src MATCHES "\\.cu$")
                    set(language "CUDA")
                    set(compiler "${CMAKE_CUDA_COMPILER}")
                elseif(src MATCHES "\\.c$")
                    set(language "C")
                    set(compiler "${CMAKE_C_COMPILER}")
                    set(lang_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
                elseif(src MATCHES "\\.cpp$|\\.cxx$|\\.cc$")
                    set(language "CXX")
                    set(compiler "${CMAKE_CXX_COMPILER}")
                    set(lang_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
                else()
                    message(WARNING "Skipping preprocessing for unknown file type: ${src}")
                    continue()
                endif()

                string(REGEX REPLACE "-O[0-9]" "" lang_flags "${lang_flags}")
                string(REGEX REPLACE "-g" "" lang_flags "${lang_flags}")
                string(REGEX REPLACE "-flto" "" lang_flags "${lang_flags}")
                string(REGEX REPLACE "-fPIC" "" lang_flags "${lang_flags}")

                message("COMMAND ${compiler} -E ${lang_flags} ${defs_flags} ${include_flags} ${src} -o ${preprocessed_file}")
                execute_process(
                        COMMAND ${CMAKE_COMMAND} -E time "${compiler}" -E ${lang_flags} ${defs_flags} ${include_flags} "${src}" -o "${preprocessed_file}"
                        RESULT_VARIABLE result
                        OUTPUT_VARIABLE stdout_output
                        ERROR_VARIABLE stderr_output
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_STRIP_TRAILING_WHITESPACE
                )

                set(output_log_file "${preprocessed_file}.log")
                file(WRITE ${output_log_file} "Command:\n${compiler} -E ${lang_flags} ${defs_flags} ${include_flags} ${src} -o ${preprocessed_file}\n\n")
                file(APPEND ${output_log_file} "Result: ${result}\n\n")
                file(APPEND ${output_log_file} "Standard Output:\n${stdout_output}\n\n")
                file(APPEND ${output_log_file} "Standard Error:\n${stderr_output}\n")

                if(result)
                    message(WARNING "Preprocessing failed for ${src}. See log: ${output_log_file}")
                else()
                    list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                endif()
            else()
                message(STATUS "Preprocessed file already exists, skipping: ${preprocessed_file}")
                list(APPEND PREPROCESSED_FILES ${preprocessed_file})
            endif()

            list(APPEND PROCESSED_SOURCES ${src})
        endif()
    endforeach()

    set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

    if(PREPROCESSED_FILES)
        add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES})
    else()
        add_custom_target(preprocess_sources ALL)
    endif()
endif()