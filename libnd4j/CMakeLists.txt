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

# Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set Windows specific flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

# Options
option(SD_NATIVE "Optimize for build machine (might not work on others)" OFF)
option(SD_CHECK_VECTORIZATION "checks for vectorization" OFF)
option(SD_BUILD_TESTS "Build tests" OFF)
option(SD_STATIC_LIB "Build static library" OFF)
option(SD_SHARED_LIB "Build shared library" ON)
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
    # For Android, trust ANDROID_ABI
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
    # Non-Android, Non-iOS logic
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

# removeFileIfExcluded: removes the file from the given glob list
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

# genCompilation: Generates cpp, cu files from templates
function(genCompilation FL_ITEM)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)

    set(EXTENSION "cpp")

    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)

    # Set all to false - Use local variables to prevent modification of parent scope
    set(LOCAL_SD_FLOAT_TYPES_GEN 0)
    set(LOCAL_SD_INTEGER_TYPES_GEN 0)
    set(LOCAL_SD_COMMON_TYPES_GEN 0)
    set(LOCAL_SD_PAIRWISE_TYPES_GEN 0)
    set(LOCAL_RANGE_STOP -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    # Define end indices for different type lists
    set(SD_INTEGER_TYPES_END 7)
    set(SD_COMMON_TYPES_END 12)
    set(SD_FLOAT_TYPES_END 3)
    set(SD_PAIRWISE_TYPES_END 12)

    # Process type matches and set flags
    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set(LOCAL_SD_INTEGER_TYPES_GEN 1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set(LOCAL_SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set(LOCAL_SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set(LOCAL_SD_PAIRWISE_TYPES_GEN 1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER LOCAL_RANGE_STOP)
            set(LOCAL_RANGE_STOP ${STOP})
        endif()
    endforeach()

    # Generate files based on type flags
    set(LOCAL_CUSTOMOPS_GENERIC_SOURCES "")

    # Create the output directory if it doesn't exist
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/compilation_units")

    if(LOCAL_RANGE_STOP GREATER -1)
        # First pass: collect all filenames to generate to avoid duplicates
        set(FILES_TO_GENERATE "")

        foreach(FL_TYPE_INDEX RANGE 0 ${LOCAL_RANGE_STOP})
            # Copy flags for this iteration
            set(ITER_SD_FLOAT_TYPES_GEN ${LOCAL_SD_FLOAT_TYPES_GEN})
            set(ITER_SD_INTEGER_TYPES_GEN ${LOCAL_SD_INTEGER_TYPES_GEN})
            set(ITER_SD_COMMON_TYPES_GEN ${LOCAL_SD_COMMON_TYPES_GEN})

            # Turn off flags if index exceeds range
            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set(ITER_SD_FLOAT_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set(ITER_SD_INTEGER_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set(ITER_SD_COMMON_TYPES_GEN 0)
            endif()

            # Pass these variables to configure_file via cmake's variable system
            set(SD_FLOAT_TYPES_GEN ${ITER_SD_FLOAT_TYPES_GEN})
            # Pass these variables to configure_file via cmake's variable system
            set(SD_FLOAT_TYPES_GEN ${ITER_SD_FLOAT_TYPES_GEN})
            set(SD_INTEGER_TYPES_GEN ${ITER_SD_INTEGER_TYPES_GEN})
            set(SD_COMMON_TYPES_GEN ${ITER_SD_COMMON_TYPES_GEN})
            set(FL_TYPE_INDEX ${FL_TYPE_INDEX})

            set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            list(APPEND FILES_TO_GENERATE ${GENERATED_SOURCE})
        endforeach()

        # Second pass: actually generate the files
        foreach(GENERATED_SOURCE ${FILES_TO_GENERATE})
            # Extract the type index from the filename
            string(REGEX MATCH "_([0-9]+)\.${EXTENSION}$" _ ${GENERATED_SOURCE})
            set(FL_TYPE_INDEX ${CMAKE_MATCH_1})

            # Configure again with the right flags for this index
            set(ITER_SD_FLOAT_TYPES_GEN ${LOCAL_SD_FLOAT_TYPES_GEN})
            set(ITER_SD_INTEGER_TYPES_GEN ${LOCAL_SD_INTEGER_TYPES_GEN})
            set(ITER_SD_COMMON_TYPES_GEN ${LOCAL_SD_COMMON_TYPES_GEN})

            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set(ITER_SD_FLOAT_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set(ITER_SD_INTEGER_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set(ITER_SD_COMMON_TYPES_GEN 0)
            endif()

            # Set these for the configure_file command
            set(SD_FLOAT_TYPES_GEN ${ITER_SD_FLOAT_TYPES_GEN})
            set(SD_INTEGER_TYPES_GEN ${ITER_SD_INTEGER_TYPES_GEN})
            set(SD_COMMON_TYPES_GEN ${ITER_SD_COMMON_TYPES_GEN})

            # Only generate the file if it doesn't already exist
            if(NOT EXISTS "${GENERATED_SOURCE}")
                configure_file("${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
                message(STATUS "Generated: ${GENERATED_SOURCE}")
            endif()

            list(APPEND LOCAL_CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE})
        endforeach()
    endif()

    # Count how many files were generated
    list(LENGTH LOCAL_CUSTOMOPS_GENERIC_SOURCES NUM_FILES)
    message(STATUS "GenCompilation for ${FL_ITEM} generated ${NUM_FILES} files")

    # Set the parent scope variable without recursive appending
    # This prevents the list from growing infinitely
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} ${LOCAL_CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

# genPartitionCombination: Generates combination files for CPU
function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
    # Split combination string into a list
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    # Determine number of elements
    list(LENGTH COMB_LIST COMB_COUNT)

    # Validate combination
    if(NOT (COMBINATION_TYPE EQUAL 3 OR COMBINATION_TYPE EQUAL 2))
        message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 3 or 2.")
    endif()

    if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR
    (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
        message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).")
    endif()

    # Define placeholders based on combination type
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

    # Read the template content
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Perform placeholder replacements
    if(COMBINATION_TYPE EQUAL 3)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" TEMP_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" TEMP_CONTENT "${TEMP_CONTENT}")
        string(REPLACE "${PLACEHOLDER3}" "${COMB3}" FINAL_CONTENT "${TEMP_CONTENT}")
    elseif(COMBINATION_TYPE EQUAL 2)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" FINAL_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}")
    endif()

    # Define the output file name based on combination
    if(COMBINATION_TYPE EQUAL 3)
        set(OUTPUT_FILE "pairwise_instantiation_${COMB1}_${COMB2}_${COMB3}.cpp")
    elseif(COMBINATION_TYPE EQUAL 2)
        set(OUTPUT_FILE "pairwise_instantiation_${COMB1}_${COMB2}.cpp")
    endif()

    # Define the full path for the generated file
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    # Write the processed content to the output file
    file(WRITE "${GENERATED_FILE}" "${FINAL_CONTENT}")

    # Optionally, collect generated sources for further processing
    list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")

    # Output a message for verification
    message(STATUS "Generated Instantiation File: ${GENERATED_FILE}")
endfunction()

# genSingleFunctionCuda: Generate individual CUDA files for each type combination
function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
    # Split the COMBINATION string into a list
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    # Extract combination values
    list(GET COMB_LIST 0 COMB1)
    list(GET COMB_LIST 1 COMB2)
    list(GET COMB_LIST 2 COMB3)

    # Get the base name from the template file
    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)

    # Read the template content
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Extract class and method names using regex
    string(REGEX MATCH "([a-zA-Z0-9_:]+),[ \n\t]*::([a-zA-Z0-9_]+)" FUNCTION_MATCH "${TEMPLATE_CONTENT}")
    set(CLASS_NAME ${CMAKE_MATCH_1})
    set(METHOD_NAME ${CMAKE_MATCH_2})

    # Clean class name for file naming
    string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")

    # Extract function signature to create a hash
    string(REGEX MATCH "::${METHOD_NAME}\\(([^;]+)\\);" FUNC_ARGS_MATCH "${TEMPLATE_CONTENT}")
    set(FUNCTION_ARGS "${CMAKE_MATCH_1}")

    # Create a signature identifier based on parameter count and types
    set(PARAM_COUNT 0)
    set(SIGNATURE_ID "")

    # Split the function arguments and count them
    string(REPLACE "," ";" ARGS_LIST "${FUNCTION_ARGS}")
    list(LENGTH ARGS_LIST PARAM_COUNT)

    # Create a hash of the signature
    foreach(ARG ${ARGS_LIST})
        # Extract just the type name from the parameter
        string(REGEX MATCH "^[^*& \t]+" TYPE_NAME "${ARG}")
        if(TYPE_NAME)
            # Append to the signature ID
            string(APPEND SIGNATURE_ID "_${TYPE_NAME}")
        endif()
    endforeach()

    # Create a shorter hash if the signature is too long
    if(SIGNATURE_ID MATCHES ".{30,}")
        string(MD5 SIGNATURE_HASH "${SIGNATURE_ID}")
        string(SUBSTRING "${SIGNATURE_HASH}" 0 8 SIGNATURE_ID)
        set(SIGNATURE_ID "_h${SIGNATURE_ID}")
    endif()

    # Output filename with signature identifier
    set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}${SIGNATURE_ID}_${COMB1}_${COMB2}_${COMB3}.cu")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    # Check if this file already exists - if so, no need to regenerate
    if(EXISTS "${GENERATED_FILE}")
        # Add to CUDA_GENERATED_SOURCES without regenerating
        list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
        set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
        return()
    endif()

    # Extract just the necessary header information
    set(START_MARKER "ITERATE_COMBINATIONS_3")
    string(FIND "${TEMPLATE_CONTENT}" "${START_MARKER}" START_POS)
    if(START_POS EQUAL -1)
        message(FATAL_ERROR "Could not find ITERATE_COMBINATIONS_3 in template file ${TEMPLATE_FILE}")
    endif()

    string(SUBSTRING "${TEMPLATE_CONTENT}" 0 ${START_POS} HEADER_CONTENT)

    # Create streamlined content with just the single instantiation needed
    set(NEW_CONTENT "${HEADER_CONTENT}\n\n// Single function instantiation for ${CLASS_NAME}::${METHOD_NAME}\n")
    string(APPEND NEW_CONTENT "template void ${CLASS_NAME}::${METHOD_NAME}<SD_SINGLE_TYPE_${COMB1}, SD_SINGLE_TYPE_${COMB2}, SD_SINGLE_TYPE_${COMB3}>(${FUNCTION_ARGS});\n")

    # Create directory if needed
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # Write the processed content to the output file
    file(WRITE "${GENERATED_FILE}" "${NEW_CONTENT}")

    # Set properties and add to sources
    set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA)
    list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)

    message(STATUS "Generated: ${GENERATED_FILE}")
endfunction()

# Configure BLAS for CPU builds
if(NOT SD_CUDA)
    # We need this definition to avoid global memory use within onednn
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if("${OPENBLAS_PATH}" STREQUAL "")
        # We don't want OpenBLAS on Apple
        if(NOT APPLE)
            # Note: this is not a typo
            set(BLA_VENDOR "OpenBLAS")
        endif()

        # Look around for system blas
        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            message("Found external BLAS implementation: ${BLAS_LIBRARIES} ")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        endif()
    else()
        # Use externally provided OPENBLAS_PATH
        set(HAVE_OPENBLAS 1)
        message("Setting openblas")
        include_directories(${OPENBLAS_PATH}/include/)
        link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
        set(OPENBLAS_LIBRARIES openblas)
    endif()
endif()

# ARM Compute configuration
# Initialize variables to indicate ARM Compute is not yet configured
set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)

# Only attempt ARM Compute configuration if the helper flag is set
if(${HELPERS_armcompute})
    message(STATUS "ARM Compute helper is active, attempting configuration...")

    # First, try to find an existing installation
    find_package(ARMCOMPUTE QUIET)

    if(ARMCOMPUTE_FOUND)
        # Found via find_package
        set(HAVE_ARMCOMPUTE 1)
        include_directories(${ARMCOMPUTE_INCLUDE_DIRS})
        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIBRARIES})
        message(STATUS "Found ARM Compute via find_package. Include: ${ARMCOMPUTE_INCLUDE_DIRS}, Libs: ${ARMCOMPUTE_LIBRARIES}")
    else()
        # Not found via find_package, attempt download if URL is defined and arch matches
        message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
        if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7" OR ${SD_ARCH} MATCHES "armv8-a" OR ${SD_ARCH} MATCHES "arm64"))
            message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")

            # Configure the download project
            configure_file(./CMakeLists.txt.armcompute.in armcompute-download/CMakeLists.txt @ONLY)

            # Execute the CMake configure step for the download
            execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result_config
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
                    OUTPUT_QUIET ERROR_QUIET)

            if(NOT result_config EQUAL 0)
                message(WARNING "CMake configuration step for ARM Compute Library download failed: ${result_config}")
            else()
                # Execute the CMake build step for the download
                execute_process(COMMAND ${CMAKE_COMMAND} --build .
                        RESULT_VARIABLE result_build
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
                        OUTPUT_QUIET ERROR_QUIET)

                if(NOT result_build EQUAL 0)
                    message(WARNING "Build step for ARM Compute Library download failed: ${result_build}")
                else()
                    # Download and build successful, set up paths and variables
                    set(ARMCOMPUTE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/armcompute-src")

                    # Check if the expected library files exist after download/build
                    set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so")
                    set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so")

                    if(EXISTS "${ARMCOMPUTE_LIB_PATH}" AND EXISTS "${ARMCOMPUTE_CORE_LIB_PATH}")
                        add_definitions(-DARMCOMPUTENEON_ENABLED)
                        include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include)
                        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIB_PATH} ${ARMCOMPUTE_CORE_LIB_PATH})
                        set(HAVE_ARMCOMPUTE 1)
                        message(STATUS "ARM Compute Library successfully configured via download at: ${ARMCOMPUTE_ROOT}")
                    else()
                        message(WARNING "ARM Compute download/build step completed, but library files not found at expected location: ${ARMCOMPUTE_ROOT}/lib")
                    endif()
                endif()
            endif()
        else()
            message(STATUS "ARM Compute helper active, but library not found and download conditions not met (URL: ${ARM_COMPUTE_URL}, Arch: ${SD_ARCH}).")
        endif()
    endif()
endif()

if(NOT DEFINED ONEDNN_PROCESSING_DONE)
    set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
endif()

if(${HELPERS_onednn})
    # Check if we've already processed OneDNN to avoid infinite recursion
    if(NOT ONEDNN_PROCESSING_DONE)
        message("Going to pull & build onednn")
        set(HAVE_ONEDNN 1)
        # Set the flag to prevent reprocessing
        set(ONEDNN_PROCESSING_DONE TRUE CACHE INTERNAL "Flag to indicate OneDNN processing is complete")
        set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE)

        # Download and extract only if needed
        if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/onednn-src/CMakeLists.txt")
            message(STATUS "OneDNN source not found, downloading now")
            configure_file(./CMakeLists.txt.onednn.in onednn-download/CMakeLists.txt)

            # Execute separate CMake process to download OneDNN
            execute_process(
                    COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onednn-download"
            )
            if(result)
                set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
                message(FATAL_ERROR "CMake step for onednn failed: ${result}")
            endif()

            execute_process(
                    COMMAND "${CMAKE_COMMAND}" --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onednn-download"
            )
            if(result)
                set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
                message(FATAL_ERROR "Build step for onednn failed: ${result}")
            endif()

            message(STATUS "OneDNN download and extraction completed")
        else()
            message(STATUS "Using existing OneDNN source")
        endif()

        # Create a separate out-of-source build directory
        set(ONEDNN_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn-build-external")
        file(MAKE_DIRECTORY "${ONEDNN_BUILD_DIR}")

        # Only configure and build if the library doesn't exist yet
        if(NOT EXISTS "${ONEDNN_BUILD_DIR}/src/libdnnl.a")
            message(STATUS "Configuring and building OneDNN outside of main build")

            # Configure OneDNN
            execute_process(
                    COMMAND "${CMAKE_COMMAND}"
                    -G "${CMAKE_GENERATOR}"
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DDNNL_LIBRARY_TYPE=STATIC
                    -DDNNL_BUILD_TESTS=OFF
                    -DDNNL_BUILD_EXAMPLES=OFF
                    -DDNNL_VERBOSE=ON
                    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/onednn-install
                    "${CMAKE_CURRENT_SOURCE_DIR}/onednn-src"
                    WORKING_DIRECTORY "${ONEDNN_BUILD_DIR}"
                    RESULT_VARIABLE onednn_config_result
            )

            if(onednn_config_result)
                set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
                message(FATAL_ERROR "OneDNN configuration failed: ${onednn_config_result}")
            endif()

            # Build OneDNN
            execute_process(
                    COMMAND "${CMAKE_COMMAND}" --build .
                    WORKING_DIRECTORY "${ONEDNN_BUILD_DIR}"
                    RESULT_VARIABLE onednn_build_result
            )

            if(onednn_build_result)
                set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
                message(FATAL_ERROR "OneDNN build failed: ${onednn_build_result}")
            endif()

            message(STATUS "OneDNN built successfully")
        else()
            message(STATUS "Using previously built OneDNN")
        endif()

        # Find the OneDNN library
        if(EXISTS "${ONEDNN_BUILD_DIR}/src/libdnnl.a")
            set(ONEDNN_LIBRARY "${ONEDNN_BUILD_DIR}/src/libdnnl.a")
        else()
            # Try to find it using find_library
            find_library(ONEDNN_LIBRARY
                    NAMES dnnl
                    PATHS "${ONEDNN_BUILD_DIR}"
                    "${ONEDNN_BUILD_DIR}/src"
                    NO_DEFAULT_PATH
            )

            if(NOT ONEDNN_LIBRARY)
                set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
                message(FATAL_ERROR "OneDNN library not found")
            endif()
        endif()

        # Set up variables for using OneDNN
        set(onednn_SOURCE_DIR "${ONEDNN_BUILD_DIR}")
        set(onednn_EXT_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn-src")
        set(ONEDNN_PATH "${onednn_SOURCE_DIR}")

        # Add include directories
        include_directories(
                "${onednn_SOURCE_DIR}/include"
                "${onednn_EXT_DIR}/include"
                "${onednn_SOURCE_DIR}"
        )

        # Create an imported target for the library
        add_library(dnnl STATIC IMPORTED GLOBAL)
        set_target_properties(dnnl PROPERTIES
                IMPORTED_LOCATION "${ONEDNN_LIBRARY}"
        )

        set(ONEDNN dnnl)
        message(STATUS "OneDNN library found at: ${ONEDNN_LIBRARY}")
    else()
        message(STATUS "OneDNN processing already completed - using existing configuration")
        # Just make sure the ONEDNN variable is properly set
        set(HAVE_ONEDNN 1)
        set(ONEDNN dnnl)
    endif()
endif()

# cuDNN configuration
if(${HELPERS_cudnn})
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

    # Look for cuDNN in multiple potential locations
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            /usr/local/cuda
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES
            cuda/include
            include
            include/cuda)

    find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            /usr/local/cuda
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES
            lib
            lib64
            cuda/lib
            cuda/lib64
            cuda/lib/x64
            lib/x64)

    # Debug output to help diagnose the issue
    message("CUDNN_INCLUDE_DIR search result: ${CUDNN_INCLUDE_DIR}")
    message("CUDNN_LIBRARY search result: ${CUDNN_LIBRARY}")

    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message("Found cuDNN: include at ${CUDNN_INCLUDE_DIR}, library at ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true)
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        # More detailed error message but continue with build
        message(WARNING "cuDNN not found. Continuing without cuDNN support.")
        message("Searched for include in: ${CUDNN_ROOT_DIR}, ${CUDA_TOOLKIT_ROOT_DIR}, and system paths")
        message("Searched for library '${CUDNN_LIBNAME}' in: ${CUDNN_ROOT_DIR}, ${CUDA_TOOLKIT_ROOT_DIR}, and system paths")
        set(HAVE_CUDNN false)
    endif()
endif()

# Flatbuffers configuration
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    set(FLATBUFFERS_BUILD_FLATC "ON" CACHE STRING "Enable flatc build" FORCE)
else()
    set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Disable flatc build" FORCE)
endif()

# Flatbuffers download
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker")
    configure_file(CMakeLists.txt.in flatbuffers-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download)
    if(result)
        message(FATAL_ERROR "CMake step for flatbuffers failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download)
    if(result)
        message(FATAL_ERROR "Build step for flatbuffers failed: ${result}")
    endif()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker" "Download complete")
else()
    message(STATUS "Flatbuffers already downloaded, skipping download step")
endif()

# Add flatbuffers directly to build
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src
        ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build
        EXCLUDE_FROM_ALL)

# Flatbuffers generation
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    # First, ensure flatc is built
    execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build --target flatc
            RESULT_VARIABLE FLATC_BUILD_RESULT
    )
    if(FLATC_BUILD_RESULT)
        message(FATAL_ERROR "Failed to build flatc: ${FLATC_BUILD_RESULT}")
    endif()

    set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")
    message("Using flatc from: ${FLATC_EXECUTABLE}")

    # Create required directories
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated
            ${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd
    )

    # Run flatc generation
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}"
            bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
            RESULT_VARIABLE FLATC_RESULT
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(FLATC_RESULT)
        message(FATAL_ERROR "Flatbuffer generation failed: ${FLATC_RESULT}")
    endif()

    # Copy Java files
    execute_process(
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
            RESULT_VARIABLE COPY_RESULT
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(COPY_RESULT)
        message(FATAL_ERROR "Java file copying failed: ${COPY_RESULT}")
    endif()
endif()
include_directories(${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include)

# Configuration file
configure_file(include/config.h.in include/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set type combinations for CUDA generation
set(COMBINATIONS_3
        "0,0,0"
        "0,0,2"
        "0,1,2"
        "0,2,0"
        "0,2,1"
        "0,2,2"
        "1,0,0"
        "1,1,1"
        "1,1,2"
        "1,2,0"
        "1,2,1"
        "1,2,2"
        "2,0,0"
        "2,0,1"
        "2,0,2"
        "2,1,0"
        "2,1,1"
        "2,1,2"
        "2,2,0"
        "2,2,1"
        "2,2,2"
)

set(COMBINATIONS_2
        "0,0"
        "0,1"
        "1,0"
        "1,1"
        "0,2"
        "2,0"
        "1,2"
        "2,1"
        "2,2"
)

# Define template locations
set(INSTANTIATION_TEMPLATES_3
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"
)

set(INSTANTIATION_TEMPLATES_2
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in"
)

set(GEN_COMPILATION_TEMPLATES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/compilation_units/pairwise_instantiation_template.cpp.in"
)

# CUDA Configuration
if(SD_CUDA)
    message("Building with CUDA support")

    # Platform-specific settings
    if(WIN32)
        message("Setting up Windows-specific CUDA unsupported compiler flags")
    else()
        message("Setting up Linux-specific CUDA unsupported compiler flags")
    endif()

    # Enable CUDA language
    enable_language(CUDA)

    add_definitions(-D__CUDABLAS__=true)

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        message("CUDA include directory: ${CUDA_INCLUDE_DIRS} with CXX compiler ${CMAKE_CXX_COMPILER_ID} SD_GCC_FUNCTRACE=${SD_GCC_FUNCTRACE}")
        include_directories(${CUDA_INCLUDE_DIRS})
        message("CUDA found!")

        set(CMAKE_CUDA_FLAGS_DEBUG " -g")
        message("CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

        # Add --allow-unsupported-compiler to CUDA flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

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
            # Updated architecture flags for CUDA 12.3
            set(CUDA_ARCH_FLAGS "-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90")
        elseif(COMPUTE_CMP STREQUAL "auto")
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "Auto")
        else()
            CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}")
        endif()


        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudart -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all")

        message("Final CUDA flags: ${CMAKE_CUDA_FLAGS}")
        if(WIN32)
            message("Setting up Windows-specific CUDA unsupported compiler flags")
        else()
            message("Setting up Linux-specific CUDA unsupported compiler flags")
        endif()

        add_compile_definitions(SD_CUDA=true)
        if("$ENV{CLION_IDE}")
            add_compile_definitions(__CUDA_CC__)
        endif()


        set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")

        # Gather CUDA source files
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
        file(GLOB_RECURSE HELPERS_SOURCES
                ./include/build_info.cpp
                ./include/ConstMessages.cpp
                ./include/helpers/*.cpp
                ./include/helpers/cuda/*.cu
                ./include/helpers/*.h)
        file(GLOB CPU_HELPERS_TO_EXCLUDE
                ./include/helpers/cpu/*.cpp)
        list(REMOVE_ITEM HELPERS_SOURCES ${CPU_HELPERS_TO_EXCLUDE})
        file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/impl/*.cpp ./include/loops/*.h)
        file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/*.cu ./include/legacy/*.h)
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA  ./include/loops/*.cu ./include/loops/cuda/**/*.cu)
        file(GLOB_RECURSE COMPILATION_UNITS ./include/loops/cuda/compilation_units/*.cu.in ./include/ops/impl/compilation_units/*.cpp.in)

        foreach(FL_ITEM ${COMPILATION_UNITS})
            genCompilation(${FL_ITEM})
        endforeach()

        if(HAVE_CUDNN)
            message("cuDNN included")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES ./include/ops/declarable/platform/cudnn/*.cu)
            list(APPEND ALL_SOURCES ${CUSTOMOPS_CUDNN_SOURCES})
        endif()

        set(ALL_SOURCES
                ${PERF_SOURCES}
                ${EXCEPTIONS_SOURCES}
                ${EXEC_SOURCES}
                ${TYPES_SOURCES}
                ${ARRAY_SOURCES}
                ${MEMORY_SOURCES}
                ${GRAPH_SOURCES}
                ${CUSTOMOPS_SOURCES}
                ${CUSTOMOPS_HELPERS_SOURCES}
                ${OPS_SOURCES}
                ${HELPERS_SOURCES}
                ${INDEXING_SOURCES}
                ${LOOPS_SOURCES}
                ${LEGACY_SOURCES}
                ${LOOPS_SOURCES_CUDA}
                ${CUSTOMOPS_GENERIC_SOURCES}
                ${CUSTOMOPS_CUDNN_SOURCES}
        )

        add_library(${SD_LIBRARY_NAME} OBJECT
                ${ALL_SOURCES}
        )

        target_include_directories(${SD_LIBRARY_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})

        set(MSVC_RT_LIB "MultiThreadedDLL")
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

        if(WIN32)
            message("Enabling /EHsc for CUDA on Windows")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj")
        endif()

        if(SD_STATIC_LIB AND SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${SD_LIBRARY_NAME}>)
            set_target_properties(${SD_LIBRARY_NAME}_static PROPERTIES OUTPUT_NAME "${SD_LIBRARY_NAME}_static")
            set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}_static DESTINATION .)
        elseif(SD_STATIC_LIB)
            add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${SD_LIBRARY_NAME}>)
            set_target_properties(${SD_LIBRARY_NAME}_static PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
            set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            install(TARGETS ${SD_LIBRARY_NAME}_static DESTINATION .)
        endif()

        if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME}_shared SHARED $<TARGET_OBJECTS:${SD_LIBRARY_NAME}>)
            set_target_properties(${SD_LIBRARY_NAME}_shared PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
            set_property(TARGET ${SD_LIBRARY_NAME}_shared PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

            target_link_libraries(${SD_LIBRARY_NAME}_shared PUBLIC
                    ${CUDA_LIBRARIES}
                    ${CUDA_DRIVER_LIBRARY}
                    ${CUDA_CUBLAS_LIBRARIES}
                    ${CUDA_cusolver_LIBRARY}
                    ${CUDNN}
            )

            install(TARGETS ${SD_LIBRARY_NAME}_shared DESTINATION .)
        endif()

        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cuda")
    endif()
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")

    # Configure BLAS for CPU builds
    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)

    # Gather CPU Source Files
    file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp ./include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp ./include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp ./include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_GENERIC_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp ./include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp ./include/helpers/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp ./include/legacy/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp ./include/loops/*.h)
    file(GLOB_RECURSE COMPILATION_UNITS ./include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in ./include/loops/cpu/compilation_units/*.cpp.in ./include/helpers/cpu/loops/*.cpp.in ./include/ops/impl/compilation_units/*.cpp.in)

    # Include ONEDNN and ARM Compute Sources
    if(HAVE_ONEDNN)
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES
                ./include/ops/declarable/platform/mkldnn/*.cpp
                ./include/ops/declarable/platform/mkldnn/mkldnnUtils.h
        )
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES})
    endif()

    if(HAVE_ARMCOMPUTE)
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES
                ./include/ops/declarable/platform/armcompute/*.cpp
                ./include/ops/declarable/platform/armcompute/*.h
        )
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
    endif()

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
    )

    if (SD_X86_BUILD)
        # Disable platform optimizations for certain files on Linux/macOS
        set_source_files_properties(cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        set_source_files_properties(./include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
    endif()

    # Remove Excluded Files if Not All Operations are Included
    if(NOT SD_ALL_OPS)
        message("Not all SD OPS INCLUDED")

        foreach(category IN ITEMS PERF_SOURCES EXCEPTIONS_SOURCES EXEC_SOURCES TYPES_SOURCES ARRAY_SOURCES MEMORY_SOURCES GRAPH_SOURCES CUSTOMOPS_SOURCES OPS_SOURCES HELPERS_SOURCES INDEXING_SOURCES LOOPS_SOURCES LEGACY_SOURCES LOOPS_SOURCES_CUDA HELPERS_CPP)
            foreach(SRC_FILE ${${category}})
                message("Checking exclusion for: ${SRC_FILE}")
                removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${category}")
            endforeach()
        endforeach()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES "")
    # Generate Compilation Units
    foreach(FL_ITEM ${COMPILATION_UNITS})
        genCompilation(${FL_ITEM})
    endforeach()

    # Process type combinations
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/cpu_instantiations")
    foreach(TEMPLATE_FILE ${INSTANTIATION_TEMPLATES_3})
        foreach(COMBINATION ${COMBINATIONS_3})
            genPartitionCombination(${TEMPLATE_FILE} 3 ${COMBINATION} "${CMAKE_BINARY_DIR}/cpu_instantiations")
        endforeach()
    endforeach()

    foreach(TEMPLATE_FILE ${INSTANTIATION_TEMPLATES_2})
        foreach(COMBINATION ${COMBINATIONS_2})
            genPartitionCombination(${TEMPLATE_FILE} 2 ${COMBINATION} "${CMAKE_BINARY_DIR}/cpu_instantiations")
        endforeach()
    endforeach()

    # Create Object Library
    add_library(${SD_LIBRARY_NAME} OBJECT
            ${LEGACY_SOURCES}
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

    target_include_directories(${SD_LIBRARY_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})

    # Add External Dependencies
    foreach(external_dependency IN LISTS EXTERNAL_DEPENDENCY_PROJECTS)
        message("@external ${external_dependency}")
        add_dependencies(${SD_LIBRARY_NAME} ${external_dependency})
    endforeach()

    # Create the shared library using SD_LIBRARY_NAME
    if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME}_shared SHARED $<TARGET_OBJECTS:${SD_LIBRARY_NAME}>)
        set_target_properties(${SD_LIBRARY_NAME}_shared PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
        if(ANDROID)
            # Limit the number of compile jobs on systems with few cores
            cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
            if(_logical_cores LESS 4)
                set_target_properties(${SD_LIBRARY_NAME}_shared PROPERTIES JOB_POOL_COMPILE one_jobs)
            endif()
        endif()

        # Link required libraries to the shared library
        target_link_libraries(${SD_LIBRARY_NAME}_shared
                ${EXTERNAL_DEPENDENCY_LIBS}
                ${ONEDNN}
                ${ONEDNN_LIBRARIES}
                ${ARMCOMPUTE_LIBRARIES}
                ${OPENBLAS_LIBRARIES}
                ${BLAS_LIBRARIES}
                ${CPU_FEATURES}
        )
        install(TARGETS ${SD_LIBRARY_NAME}_shared DESTINATION .)
    endif()

    # Create the static library if needed
    if (SD_STATIC_LIB AND SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${SD_LIBRARY_NAME}>)
        set_target_properties(${SD_LIBRARY_NAME}_static PROPERTIES OUTPUT_NAME "${SD_LIBRARY_NAME}_static")
        set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        install(TARGETS ${SD_LIBRARY_NAME}_static DESTINATION .)
    elseif(SD_STATIC_LIB)
        add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${SD_LIBRARY_NAME}>)
        set_target_properties(${SD_LIBRARY_NAME}_static PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
        set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        install(TARGETS ${SD_LIBRARY_NAME}_static DESTINATION .)
    endif()

    # Handle MSVC Runtime Library Consistently
    set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

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
endif()

# Add tests if enabled
if(SD_BUILD_TESTS)
    include(CTest)
    # tests are always compiled with all ops included
    set(SD_ALL_OPS true)
    enable_testing()
    add_subdirectory(tests_cpu)
endif()

# Preprocessing Configuration
if(SD_PREPROCESS STREQUAL "ON")
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}")
    include_directories(${CMAKE_BINARY_DIR}/.././include)

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
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/array
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/execution
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/ops
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/types
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/memory
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/graph
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/exceptions
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/indexing
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/legacy
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/loops
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/loops/cpu
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/loops/cuda
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/performance
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/helpers
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/system
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/ops/declarable
                        ${CMAKE_CURRENT_SOURCE_DIR}/./include/ops/declarable/helpers
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

# Set package information
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "agibsonccc <adam@konduit.ai>")
set(CPACK_PACKAGE_VENDOR "Eclipse")
set(CPACK_SETDESTDIR "false")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/lib")
set(CPACK_PACKAGE_NAME "libnd4j")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "8")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "libnd4j")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")



if(UNIX)
    # Determine distribution and release — may require redhat-lsb-core installed on CentOS / RH
    execute_process(COMMAND lsb_release -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND lsb_release -sc OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND uname -i OUTPUT_VARIABLE ARCHITECTURE)

    # Set package name and type (deb vs rpm)
    if(DISTRIBUTION STREQUAL "Ubuntu")
        # Set Ubuntu-specific information
        if(ARCHITECTURE MATCHES ".*x86_64.*")
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
        else()
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
        endif()
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER "agibsonccc")
        set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
        set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "cuda")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
        set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/deeplearning4j/deeplearning4j")
        set(CPACK_GENERATOR "DEB")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${RELEASE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm;" )
    elseif(DISTRIBUTION STREQUAL "CentOS")
        # Set Fedora-specific information
        execute_process(COMMAND lsb_release -sr OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(ARCHITECTURE MATCHES ".*x86_64.*")
            set(CPACK_RPM_PACKAGE_ARCHITECTURE "x86_64")
        else()
            set(CPACK_RPM_PACKAGE_ARCHITECTURE "i686")
        endif()
        set(CPACK_PACKAGE_CONTACT "agibsonccc")
        set(CPACK_RPM_PACKAGE_GROUP "Development/Tools")
        set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
        set(CPACK_RPM_PACKAGE_SUGGESTS "cuda")
        set(CPACK_RPM_PACKAGE_REQUIRES "")
        set(CPACK_RPM_PACKAGE_URL "https://github.com/deeplearning4j/deeplearning4j/libnd4j")
        set(CPACK_GENERATOR "RPM")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}.fc${RELEASE}.${CPACK_RPM_PACKAGE_ARCHITECTURE})
        set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst")
        set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm")
        set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib")
    endif()

endif()


include(CPack)