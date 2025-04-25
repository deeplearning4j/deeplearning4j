cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(libnd4j)

# Include ExternalProject module
include(ExternalProject)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_WINDOWS_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_WINDOWS_BUILD=true")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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
set(CMAKE_VERBOSE_MAKEFILE ON)
option(SD_GCC_FUNCTRACE "Use call traces" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF)

message("PRINT_INDICES: ${PRINT_INDICES}")
if("${PRINT_INDICES}" STREQUAL "ON")
    message("Added print indices compile definition")
    add_compile_definitions(PRINT_INDICES)
endif()

message("PRINT_MATH: ${PRINT_MATH}")
if("${PRINT_MATH}" STREQUAL "ON")
    message("Added print indices compile definition")
    add_compile_definitions(SD_PRINT_MATH)
endif()

if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
    message("Set optimization for functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL   "0")
else()
    message("Set optimization level for no functrace ${SD_GCC_FUNCTRACE}")
    set(SD_OPTIMIZATION_LEVEL  "3")
endif()

message("Set default optimization level ${SD_OPTIMIZATION_LEVEL}")
# FLATBUFFERS_BUILD_FLATC is handled by ExternalProject_Add CMAKE_ARGS below

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

set(ARM_COMPUTE_URL_ARMV7 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-armv7a-cpu-bin.tar.gz")
set(ARM_COMPUTE_URL_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-aarch64-cpu-bin.tar.gz")
set(ARM_COMPUTE_URL_ANDROID_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-android-aarch64-cpu-bin.tar.gz")

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

option(HELPERS_onednn "Build with oneDNN" OFF)
option(HELPERS_armcompute "Build with ARM Compute Library" OFF)
option(HELPERS_cudnn "Build with cuDNN" OFF)

set(MSVC_RT_LIB "MultiThreadedDLL")

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

if (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "armv8-a")
        set(SD_X86_BUILD false)
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
else()
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
    if (SD_SANITIZE)
        if(NOT DEFINED SD_SANITIZERS)
            set(SD_SANITIZERS "address")
        endif()
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIE -lpthread -ftls-model=local-dynamic -static-libasan -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
        message("Using sanitizers: ${SD_SANITIZERS} - note you can not use both thread and address sanitizer at the same time. Be careful what sanitizers you specify.
         Note that address and undefined can not be used at the same time or an address overlap error will occur. See: https://github.com/google/sanitizers/issues/856
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
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*")
        set(SD_X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()

if(NOT SD_CUDA)
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if ("${OPENBLAS_PATH}" STREQUAL "")
        if (NOT APPLE)
            set(BLA_VENDOR "OpenBLAS")
        endif()
        find_package(BLAS REQUIRED)
        if (BLAS_FOUND)
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

set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)

if(${HELPERS_armcompute})
    message(STATUS "ARM Compute helper is active, attempting configuration...")
    find_package(ARMCOMPUTE QUIET)
    if(ARMCOMPUTE_FOUND)
        set(HAVE_ARMCOMPUTE 1)
        include_directories(${ARMCOMPUTE_INCLUDE_DIRS})
        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIBRARIES})
        message(STATUS "Found ARM Compute via find_package. Include: ${ARMCOMPUTE_INCLUDE_DIRS}, Libs: ${ARMCOMPUTE_LIBRARIES}")
        add_definitions(-DHAVE_ARMCOMPUTE=1)
        add_definitions(-DARMCOMPUTENEON_ENABLED)
    else()
        message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
        if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7" OR ${SD_ARCH} MATCHES "armv8-a" OR ${SD_ARCH} MATCHES "arm64"))
            message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.armcompute.in armcompute-download/CMakeLists.txt @ONLY) # Corrected Path
            execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result_config
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
                    OUTPUT_QUIET ERROR_QUIET)
            if(NOT result_config EQUAL 0)
                message(WARNING "CMake configuration step for ARM Compute Library download failed: ${result_config}")
            else()
                execute_process(COMMAND ${CMAKE_COMMAND} --build .
                        RESULT_VARIABLE result_build
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/armcompute-download
                        OUTPUT_QUIET ERROR_QUIET)
                if(NOT result_build EQUAL 0)
                    message(WARNING "Build step for ARM Compute Library download failed: ${result_build}")
                else()
                    set(ARMCOMPUTE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/armcompute-src")
                    set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so")
                    set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so")
                    if(EXISTS "${ARMCOMPUTE_LIB_PATH}" AND EXISTS "${ARMCOMPUTE_CORE_LIB_PATH}")
                        add_definitions(-DARMCOMPUTENEON_ENABLED)
                        include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include)
                        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIB_PATH} ${ARMCOMPUTE_CORE_LIB_PATH})
                        set(HAVE_ARMCOMPUTE 1)
                        add_definitions(-DHAVE_ARMCOMPUTE=1)
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

if (${HELPERS_onednn})
    message("Going to pull & build onednn")
    set(HAVE_ONEDNN 1)
    add_definitions(-DHAVE_ONEDNN=1)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.onednn.in onednn-download/CMakeLists.txt @ONLY) # Corrected Path
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
    if(result)
        message(FATAL_ERROR "CMake step for onednn failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onednn-download )
    if(result)
        message(FATAL_ERROR "Build step for onednn failed: ${result}")
    endif()
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/onednn-src
            ${CMAKE_CURRENT_BINARY_DIR}/onednn-build
            EXCLUDE_FROM_ALL)
    set(onednn_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/onednn-build)
    set(onednn_EXT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onednn-src)
    set(ONEDNN_PATH "${onednn_SOURCE_DIR}")
    include_directories(${onednn_SOURCE_DIR}/include ${onednn_EXT_DIR}/include ${onednn_SOURCE_DIR})
    set(ONEDNN_LIBRARIES dnnl)
endif()

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
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
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
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES
            lib
            lib64
            cuda/lib
            cuda/lib64
            lib/x64)
    message("CUDNN_INCLUDE_DIR search result: ${CUDNN_INCLUDE_DIR}")
    message("CUDNN_LIBRARY search result: ${CUDNN_LIBRARY}")
    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message("Found cuDNN: include at ${CUDNN_INCLUDE_DIR}, library at ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true)
        add_definitions(-DHAVE_CUDNN=1)
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        message(WARNING "cuDNN not found. Build will continue without cuDNN support.")
        set(HAVE_CUDNN false)
    endif()
endif()

# ================== Refactored Flatbuffers Section ==================
set(FLATBUFFERS_TAG "v24.3.25" CACHE STRING "Flatbuffers Git Tag")
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Build Flatbuffers Tests")
set(GENERATE_FLATC OFF)
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC_OPTION)
    set(GENERATE_FLATC ON)
endif()
set(FLATBUFFERS_BUILD_FLATC ${GENERATE_FLATC} CACHE BOOL "Build flatc")

ExternalProject_Add(flatbuffers_external
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG ${FLATBUFFERS_TAG}
        SOURCE_DIR "${CMAKE_BINARY_DIR}/flatbuffers-src"
        BINARY_DIR "${CMAKE_BINARY_DIR}/flatbuffers-build"
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} # Or a specific install dir for deps
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DFLATBUFFERS_BUILD_TESTS=${FLATBUFFERS_BUILD_TESTS}
        -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        # Pass compiler flags if necessary, e.g., for specific architectures or MSVC runtime
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        $<$<BOOL:${WIN32}>:-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}>
        $<$<BOOL:${WIN32}>:-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}>
        $<$<BOOL:${WIN32}>:-DCMAKE_MSVC_RUNTIME_LIBRARY=${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>>
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG>
        INSTALL_COMMAND "" # No install step needed usually
        UPDATE_COMMAND "" # Avoid unnecessary updates unless specified
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_INSTALL TRUE
        LOG_DOWNLOAD TRUE
)

# Define imported targets for the library and potentially flatc
# Note: Check flatbuffers CMakeLists.txt for exact target names if these are wrong
add_library(flatbuffers STATIC IMPORTED GLOBAL)
set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/libflatbuffers.a) # Adjust name/path if needed
add_dependencies(flatbuffers flatbuffers_external)

if(FLATBUFFERS_BUILD_FLATC)
    add_executable(flatc IMPORTED GLOBAL)
    set_property(TARGET flatc PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/flatc) # Adjust name/path if needed
    add_dependencies(flatc flatbuffers_external)
endif()

# Add include directory for flatbuffers headers
include_directories(${CMAKE_BINARY_DIR}/flatbuffers-src/include)
# ================== End Refactored Flatbuffers Section ==================

# Generate Flatbuffers sources if requested
if(GENERATE_FLATC)
    message("Generating flatbuffers sources...")

    # Define generation target depending on the flatc target
    set(FLATC_TARGET flatc) # Use the imported target name
    set(FLATC_INPUT_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/graph.fbs") # Assuming schema location
    set(FLATC_OUTPUT_DIR_CPP "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd")
    set(FLATC_OUTPUT_DIR_JAVA "${CMAKE_CURRENT_SOURCE_DIR}/../../nd4j-serde/nd4j-flatbuffers-java/src/main/java") # Adjust path as needed

    # Ensure output directories exist
    file(MAKE_DIRECTORY ${FLATC_OUTPUT_DIR_CPP})
    file(MAKE_DIRECTORY ${FLATC_OUTPUT_DIR_JAVA}) # Check if this path is correct

    add_custom_command(
            OUTPUT ${FLATC_OUTPUT_DIR_CPP}/graph_generated.h # Primary output file
            COMMAND $<TARGET_FILE:${FLATC_TARGET}> -o ${FLATC_OUTPUT_DIR_CPP} --cpp --gen-object-api ${FLATC_INPUT_SCHEMA}
            COMMAND $<TARGET_FILE:${FLATC_TARGET}> -o ${FLATC_OUTPUT_DIR_JAVA} --java --gen-object-api ${FLATC_INPUT_SCHEMA}
            DEPENDS ${FLATC_TARGET} ${FLATC_INPUT_SCHEMA} # Depends on flatc build and schema
            COMMENT "Generating flatbuffers C++ and Java sources"
            VERBATIM
    )
    add_custom_target(generate_flatbuffers ALL DEPENDS ${FLATC_OUTPUT_DIR_CPP}/graph_generated.h) # Target depends on output

else()
    message("Skipping flatbuffers source generation (FLATBUFFERS_BUILD_FLATC is OFF).")
endif()

configure_file(include/config.h.in include/config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

if(UNIX)
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(APPLE)
    message("Using Apple")
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

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

if(WIN32 AND NOT ANDROID)
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()

    message(STATUS "Include Directories (from blas section):")
    foreach(dir ${dirs})
        message(STATUS "dir='${dir}'")
    endforeach()

    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
endif()

if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Using Link Time Optimization")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

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

if(DEFINED OP_OUTPUT_FILE AND OP_OUTPUT_FILE MATCHES ".h$")
    message("Definitions will be written to \"${OP_OUTPUT_FILE}\"")
    file(WRITE "${OP_OUTPUT_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
endif()

if(SD_ARCH MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
elseif(CMAKE_SYSTEM_NAME MATCHES "Aurora")
    set_source_files_properties(include/graph/impl/GraphHolder.cpp PROPERTIES COMPILE_FLAGS -g0)
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

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include")
endif()

if(NOT SD_CUDA)
    if(NOT SD_CPU)
        set(SD_CUDA FALSE)
        set(SD_CPU TRUE)
    endif()
endif()

if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    else()
        set(SD_LIBRARY_NAME nd4jcpu)
    endif()
endif()

if(HAVE_ONEDNN)
    file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES
            include/ops/declarable/platform/mkldnn/*.cpp
            include/ops/declarable/platform/mkldnn/mkldnnUtils.h
    )
endif()

if(HAVE_ARMCOMPUTE)
    file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES
            include/ops/declarable/platform/armcompute/*.cpp
            include/ops/declarable/platform/armcompute/*.h
    )
endif()

function(removeFileIfExcluded)
    cmake_parse_arguments(
            PARSED_ARGS
            ""
            "FILE_ITEM"
            "LIST_ITEM"
            ${ARGN}
    )
    if(DEFINED ${PARSED_ARGS_LIST_ITEM} AND EXISTS "${PARSED_ARGS_FILE_ITEM}")
        file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
        string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)
        if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
            set(local_list ${${PARSED_ARGS_LIST_ITEM}})
            set(file_removed FALSE)
            set(keep_file FALSE)
            if(SD_ALL_OPS)
                set(keep_file TRUE)
            elseif(DEFINED SD_OPS_LIST AND NOT "${SD_OPS_LIST}" STREQUAL "")
                foreach(OP ${SD_OPS_LIST})
                    string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
                    if(${NOT_EXCLUDED_OP_IDX} GREATER_EQUAL 0)
                        set(keep_file TRUE)
                        break()
                    endif()
                endforeach()
            else()
                set(keep_file TRUE)
            endif()

            if(NOT keep_file)
                list(REMOVE_ITEM local_list "${PARSED_ARGS_FILE_ITEM}")
                set(file_removed TRUE)
                message(STATUS "Excluding file based on OP list: ${PARSED_ARGS_FILE_ITEM}")
            endif()

            if(file_removed)
                set(${PARSED_ARGS_LIST_ITEM} ${local_list} PARENT_SCOPE)
            endif()
        endif()
    else()
    endif()
endfunction()

function(genCompilation FILE_ITEM OUTPUT_LIST_VAR)
    get_filename_component(FILE_ITEM_WE ${FILE_ITEM} NAME_WE)
    set(EXTENSION "cpp")
    if(FILE_ITEM MATCHES "\\.cu\\.in$")
        set(EXTENSION "cu")
    endif()
    file(READ ${FILE_ITEM} CONTENT_FL)
    set (SD_FLOAT_TYPES_GEN     0)
    set (SD_INTEGER_TYPES_GEN       0)
    set (SD_COMMON_TYPES_GEN   0)
    set (SD_PAIRWISE_TYPES_GEN  0)
    set (RANGE_STOP           -1)
    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})
    set(SD_INTEGER_TYPES_END 7)
    set(SD_COMMON_TYPES_END 12)
    set(SD_FLOAT_TYPES_END 3)
    set(SD_PAIRWISE_TYPES_END 12)
    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set (SD_INTEGER_TYPES_GEN  1)
            set(STOP ${SD_INTEGER_TYPES_END})
        elseif(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set (SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        elseif(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set (SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        elseif(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set (SD_PAIRWISE_TYPES_GEN  1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()
    endforeach()
    set(generated_files_this_call "")
    if(RANGE_STOP GREATER -1)
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated_compilation_units")
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            set (CURRENT_SD_FLOAT_TYPES_GEN ${SD_FLOAT_TYPES_GEN})
            set (CURRENT_SD_INTEGER_TYPES_GEN ${SD_INTEGER_TYPES_GEN})
            set (CURRENT_SD_COMMON_TYPES_GEN ${SD_COMMON_TYPES_GEN})
            set (CURRENT_SD_PAIRWISE_TYPES_GEN ${SD_PAIRWISE_TYPES_GEN})

            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set (CURRENT_SD_FLOAT_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set (CURRENT_SD_INTEGER_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set (CURRENT_SD_COMMON_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_PAIRWISE_TYPES_END})
                set (CURRENT_SD_PAIRWISE_TYPES_GEN 0)
            endif()

            if(CURRENT_SD_FLOAT_TYPES_GEN OR CURRENT_SD_INTEGER_TYPES_GEN OR CURRENT_SD_COMMON_TYPES_GEN OR CURRENT_SD_PAIRWISE_TYPES_GEN)
                set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/generated_compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
                configure_file("${FILE_ITEM}" "${GENERATED_SOURCE}" @ONLY DEFINES
                        SD_FLOAT_TYPES_GEN=${CURRENT_SD_FLOAT_TYPES_GEN}
                        SD_INTEGER_TYPES_GEN=${CURRENT_SD_INTEGER_TYPES_GEN}
                        SD_COMMON_TYPES_GEN=${CURRENT_SD_COMMON_TYPES_GEN}
                        SD_PAIRWISE_TYPES_GEN=${CURRENT_SD_PAIRWISE_TYPES_GEN}
                        FL_TYPE_INDEX=${FL_TYPE_INDEX}
                )
                LIST(APPEND generated_files_this_call ${GENERATED_SOURCE} )
            endif()
        endforeach()
    endif()
    set(${OUTPUT_LIST_VAR} ${${OUTPUT_LIST_VAR}} ${generated_files_this_call} PARENT_SCOPE)
endfunction()

function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
    list(GET COMB_LIST 0 COMB1)
    list(GET COMB_LIST 1 COMB2)
    list(GET COMB_LIST 2 COMB3)
    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)
    string(REGEX MATCH "template[ \n\t<].*[ \n\t>]([a-zA-Z0-9_:]+)::([a-zA-Z0-9_]+)<.*>\\(([^)]*)\\);" FUNC_MATCH "${TEMPLATE_CONTENT}")
    if(NOT CMAKE_MATCH_COUNT EQUAL 3)
        string(REGEX MATCH "//[ \t]*Function:[ \t]*([a-zA-Z0-9_:]+)::([a-zA-Z0-9_]+)\\(([^)]*)\\)" FUNC_MATCH "${TEMPLATE_CONTENT}")
    endif()
    if(CMAKE_MATCH_COUNT EQUAL 3)
        set(CLASS_NAME ${CMAKE_MATCH_1})
        set(METHOD_NAME ${CMAKE_MATCH_2})
        set(FUNCTION_ARGS_STRIP ${CMAKE_MATCH_3})
        string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")
        set(SIGNATURE_ID "")
        string(REPLACE "," ";" ARGS_LIST "${FUNCTION_ARGS_STRIP}")
        foreach(ARG ${ARGS_LIST})
            string(STRIP "${ARG}" ARG_STRIPPED)
            string(REGEX MATCH "^([^ ]+[ ]?[\*&]?)" TYPE_NAME "${ARG_STRIPPED}")
            if(CMAKE_MATCH_1)
                string(APPEND SIGNATURE_ID "_${CMAKE_MATCH_1}")
            endif()
        endforeach()
        string(REGEX REPLACE "[*& ]" "" SIGNATURE_ID "${SIGNATURE_ID}")
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
        string(FIND "${TEMPLATE_CONTENT}" "template <" START_POS)
        if(START_POS EQUAL -1)
            string(FIND "${TEMPLATE_CONTENT}" "// Function:" START_POS)
        endif()
        if(START_POS EQUAL -1)
            message(WARNING "Could not find template start or // Function marker in ${TEMPLATE_FILE}. Using full header.")
            set(HEADER_CONTENT ${TEMPLATE_CONTENT})
        else()
            string(SUBSTRING "${TEMPLATE_CONTENT}" 0 ${START_POS} HEADER_CONTENT)
        endif()
        string(REGEX MATCH "(template[ \n\t<][^>]+>[ \n\t]+[a-zA-Z0-9_:]+::[a-zA-Z0-9_]+<[^>]+>\\([^)]*\\));" FULL_SIGNATURE_MATCH "${TEMPLATE_CONTENT}")
        if(NOT CMAKE_MATCH_1)
            message(FATAL_ERROR "Could not extract full function signature from ${TEMPLATE_FILE}")
        endif()
        set(FUNCTION_SIGNATURE_NO_SEMICOLON ${CMAKE_MATCH_1})
        string(REPLACE "T" "SD_SINGLE_TYPE_${COMB1}" TEMP_SIG "${FUNCTION_SIGNATURE_NO_SEMICOLON}")
        string(REPLACE "X" "SD_SINGLE_TYPE_${COMB2}" TEMP_SIG "${TEMP_SIG}")
        string(REPLACE "Y" "SD_SINGLE_TYPE_${COMB3}" INSTANTIATED_SIGNATURE "${TEMP_SIG}")
        set(NEW_CONTENT "${HEADER_CONTENT}\n\n// Single function instantiation for ${CLASS_NAME}::${METHOD_NAME} (${COMB1},${COMB2},${COMB3})\n")
        string(APPEND NEW_CONTENT "template void ${INSTANTIATED_SIGNATURE};\n")
        file(MAKE_DIRECTORY "${OUTPUT_DIR}")
        file(WRITE "${GENERATED_FILE}" "${NEW_CONTENT}")
        set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA)
        list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
        set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Could not extract Class/Method/Args from CUDA template: ${TEMPLATE_FILE}")
    endif()
endfunction()

function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
    list(LENGTH COMB_LIST COMB_COUNT)
    if(NOT (COMBINATION_TYPE EQUAL 3 OR COMBINATION_TYPE EQUAL 2))
        message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 3 or 2.")
    endif()
    if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
        message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).")
    endif()
    get_filename_component(TEMPLATE_NAME ${TEMPLATE_FILE} NAME)
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)
    set(FINAL_CONTENT "${TEMPLATE_CONTENT}")
    set(OUTPUT_FILE_SUFFIX "")
    if(COMBINATION_TYPE EQUAL 3)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        list(GET COMB_LIST 2 COMB3)
        string(REPLACE "@COMB1@" "${COMB1}" FINAL_CONTENT "${FINAL_CONTENT}")
        string(REPLACE "@COMB2@" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}")
        string(REPLACE "@COMB3@" "${COMB3}" FINAL_CONTENT "${FINAL_CONTENT}")
        set(OUTPUT_FILE_SUFFIX "_${COMB1}_${COMB2}_${COMB3}")
    elseif(COMBINATION_TYPE EQUAL 2)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        string(REPLACE "@COMB1@" "${COMB1}" FINAL_CONTENT "${FINAL_CONTENT}")
        string(REPLACE "@COMB2@" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}")
        set(OUTPUT_FILE_SUFFIX "_${COMB1}_${COMB2}")
    endif()
    get_filename_component(TEMPLATE_BASE ${TEMPLATE_FILE} NAME_WE)
    set(OUTPUT_FILE "${TEMPLATE_BASE}${OUTPUT_FILE_SUFFIX}.cpp")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")
    file(WRITE "${GENERATED_FILE}" "${FINAL_CONTENT}")
    list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

if(SD_CUDA)
    message("--- Configuring CUDA Build ---")
    include(TypeMST.cmake)
    set(CUDA_GENERATED_DIR "${CMAKE_BINARY_DIR}/cuda_generated")
    file(MAKE_DIRECTORY "${CUDA_GENERATED_DIR}")
    set(ALL_SOURCES "")
    set(CUDA_GENERATED_SOURCES "")
    set(CUSTOMOPS_ONEDNN_SOURCES "")
    set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
    set(CUSTOMOPS_CUDNN_SOURCES "")
    if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST_COUNT GREATER 0)
        set(NUM_TYPES ${SD_TYPES_LIST_COUNT})
        message(STATUS "Using ${NUM_TYPES} types from SD_TYPES_LIST for CUDA generation")
    else()
        set(NUM_TYPES 10)
        message(STATUS "Using default number of types for CUDA generation: ${NUM_TYPES}")
    endif()
    file(GLOB CUDA_COMPILATION_UNITS_IN include/loops/cuda/compilation_units/*.cu.in)
    file(GLOB CUDA_OPS_COMPILATION_UNITS_IN include/ops/impl/compilation_units/*.cu.in)
    list(APPEND CUDA_COMPILATION_UNITS_IN ${CUDA_OPS_COMPILATION_UNITS_IN})
    foreach(FL_ITEM ${CUDA_COMPILATION_UNITS_IN})
        genCompilation(${FL_ITEM} CUDA_GENERATED_SOURCES)
    endforeach()
    set(SD_USE_MST_TYPES ON)
    if(SD_USE_MST_TYPES)
        generate_mst_combinations()
        message(STATUS "Using optimized MST type combinations for CUDA")
    else()
        message(STATUS "Using all possible type combinations for CUDA (SD_USE_MST_TYPES=OFF)")
    endif()
    file(GLOB CUDA_TEMPLATE_FILES "include/loops/cuda/comb_compilation_units/*.cu.in")
    foreach(TEMPLATE_FILE ${CUDA_TEMPLATE_FILES})
        foreach(COMBINATION ${COMBINATIONS_3})
            genSingleFunctionCuda(${TEMPLATE_FILE} ${COMBINATION} "${CUDA_GENERATED_DIR}")
        endforeach()
    endforeach()

    file(GLOB_RECURSE PERF_SOURCES include/performance/*.cpp include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES include/exceptions/*.cpp include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES include/execution/impl/*.cpp include/execution/*.h)
    file(GLOB_RECURSE EXEC_SOURCES_CUDA include/execution/cuda/*.cu include/execution/cuda/*.h include/execution/*.cu)
    file(GLOB_RECURSE TYPES_SOURCES include/types/*.cpp include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES include/array/impl/*.cpp include/array/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES_CUDA include/array/cuda/*.cu include/array/cuda/*.chpp)
    file(GLOB_RECURSE MEMORY_SOURCES include/memory/impl/*.cpp include/memory/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES_CUDA include/memory/cuda/*.cu)
    file(GLOB_RECURSE GRAPH_SOURCES include/graph/*.cpp include/graph/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES_CUDA include/graph/*.cu)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_CUDA include/ops/declarable/helpers/cuda/*.cu)
    file(GLOB_RECURSE OPS_SOURCES include/ops/impl/*.cpp include/ops/declarable/impl/*.cpp include/ops/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES include/build_info.cpp include/ConstMessages.cpp include/helpers/*.cpp include/helpers/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES_CUDA include/helpers/cuda/*.cu)
    file(GLOB CPU_HELPERS_TO_EXCLUDE include/helpers/cpu/*.cpp)
    list(REMOVE_ITEM HELPERS_SOURCES ${CPU_HELPERS_TO_EXCLUDE})
    file(GLOB_RECURSE INDEXING_SOURCES include/indexing/*.cpp include/indexing/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES include/loops/impl/*.cpp include/loops/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES_CUDA include/loops/*.cu include/loops/cuda/**/*.cu)
    file(GLOB_RECURSE LEGACY_SOURCES include/legacy/impl/*.cpp include/legacy/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES_CUDA include/legacy/*.cu)

    if(HAVE_ONEDNN)
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES include/ops/declarable/platform/mkldnn/*.cpp include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
    endif()
    if(HAVE_ARMCOMPUTE)
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES include/ops/declarable/platform/armcompute/*.cpp include/ops/declarable/platform/armcompute/*.h)
    endif()
    if(HAVE_CUDNN)
        file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES include/ops/declarable/platform/cudnn/*.cu)
    endif()

    set(ALL_SOURCES
            ${CUDA_GENERATED_SOURCES}
            ${PERF_SOURCES}
            ${EXCEPTIONS_SOURCES}
            ${EXEC_SOURCES} ${EXEC_SOURCES_CUDA}
            ${TYPES_SOURCES}
            ${ARRAY_SOURCES} ${ARRAY_SOURCES_CUDA}
            ${MEMORY_SOURCES} ${MEMORY_SOURCES_CUDA}
            ${GRAPH_SOURCES} ${GRAPH_SOURCES_CUDA}
            ${CUSTOMOPS_SOURCES}
            ${CUSTOMOPS_HELPERS_IMPL} ${CUSTOMOPS_HELPERS_CUDA}
            ${OPS_SOURCES}
            ${HELPERS_SOURCES} ${HELPERS_SOURCES_CUDA}
            ${INDEXING_SOURCES}
            ${LOOPS_SOURCES} ${LOOPS_SOURCES_CUDA}
            ${LEGACY_SOURCES} ${LEGACY_SOURCES_CUDA}
            ${CUSTOMOPS_ONEDNN_SOURCES}
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
            ${CUSTOMOPS_CUDNN_SOURCES}
    )

    if(NOT SD_ALL_OPS)
        message("Excluding files based on SD_OPS_LIST for CUDA build...")
        set(FILTERED_SOURCES "")
        foreach(SRC_FILE ${ALL_SOURCES})
            set(keep_file TRUE)
            if(EXISTS "${SRC_FILE}")
                file(READ ${SRC_FILE} FILE_CONTENTS)
                string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)
                if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
                    set(found_matching_op FALSE)
                    if(DEFINED SD_OPS_LIST AND NOT "${SD_OPS_LIST}" STREQUAL "")
                        foreach(OP ${SD_OPS_LIST})
                            string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
                            if(${NOT_EXCLUDED_OP_IDX} GREATER_EQUAL 0)
                                set(found_matching_op TRUE)
                                break()
                            endif()
                        endforeach()
                    endif()
                    if(NOT found_matching_op)
                        set(keep_file FALSE)
                        message(STATUS "Excluding file (no matching NOT_EXCLUDED OP): ${SRC_FILE}")
                    endif()
                endif()
            else()
                set(keep_file FALSE)
            endif()

            if(keep_file)
                list(APPEND FILTERED_SOURCES ${SRC_FILE})
            endif()
        endforeach()
        set(ALL_SOURCES ${FILTERED_SOURCES})
    endif()

    add_library(samediff_obj OBJECT ${ALL_SOURCES})

    target_include_directories(samediff_obj PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CMAKE_BINARY_DIR}/flatbuffers-src/include
            ${CUDA_INCLUDE_DIRS}
            ${EXTERNAL_INCLUDE_DIRS}
            $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_ROOT}/include>
            $<$<BOOL:${HAVE_CUDNN}>:${CUDNN_INCLUDE_DIR}>
    )
    if(WIN32)
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Aurora")
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/impl/GraphHolder.cpp")
            set_source_files_properties(include/graph/impl/GraphHolder.cpp PROPERTIES COMPILE_FLAGS -g0)
        endif()
    endif()

    if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY OUTPUT_NAME ${SD_LIBRARY_NAME})
        if(WIN32)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                ${CUDA_LIBRARIES}
                ${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_cusolver_LIBRARY}
                ${CUDA_DRIVER_LIBRARY}
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN}>
                $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_LIBRARIES}>
                $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_LIBRARIES}>
                Threads::Threads
                dl
                flatbuffers # Link flatbuffers library
        )
        if(OPENMP_FOUND)
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
        endif()
        install(TARGETS ${SD_LIBRARY_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
    endif()

    if(SD_STATIC_LIB)
        set(STATIC_LIB_NAME ${SD_LIBRARY_NAME})
        if(SD_SHARED_LIB)
            set(STATIC_LIB_NAME ${SD_LIBRARY_NAME}static)
        endif()
        add_library(${STATIC_LIB_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
        set_property(TARGET ${STATIC_LIB_NAME} PROPERTY OUTPUT_NAME ${STATIC_LIB_NAME})
        if(WIN32)
            set_property(TARGET ${STATIC_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        # Link flatbuffers to the static library if needed, or ensure users link it
        target_link_libraries(${STATIC_LIB_NAME} PUBLIC flatbuffers)
        install(TARGETS ${STATIC_LIB_NAME} ARCHIVE DESTINATION lib)
    endif()

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

elseif(SD_CPU)
    message("--- Configuring CPU Build ---")
    add_definitions(-D__CPUBLAS__=true)
    set(ALL_SOURCES "")
    set(CPU_GENERATED_SOURCES "")
    set(CUSTOMOPS_GENERIC_SOURCES "")
    set(CUSTOMOPS_ONEDNN_SOURCES "")
    set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")

    file(GLOB CPU_COMPILATION_UNITS_IN
            include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
            include/loops/cpu/compilation_units/*.cpp.in
            include/helpers/cpu/loops/*.cpp.in
            include/ops/impl/compilation_units/*.cpp.in
    )
    foreach(FL_ITEM ${CPU_COMPILATION_UNITS_IN})
        genCompilation(${FL_ITEM} CPU_GENERATED_SOURCES)
    endforeach()

    set(COMBINATIONS_3 "0,0,0;0,0,2;0,1,2;0,2,0;0,2,1;0,2,2;1,0,0;1,1,1;1,1,2;1,2,0;1,2,1;1,2,2;2,0,0;2,0,1;2,0,2;2,1,0;2,1,1;2,1,2;2,2,0;2,2,1;2,2,2")
    set(COMBINATIONS_2 "0,0;0,1;1,0;1,1;0,2;2,0;1,2;2,1;2,2")
    set(INSTANTIATION_TEMPLATES_3 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in")
    set(INSTANTIATION_TEMPLATES_2 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in")
    set(CPU_COMB_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated_combinations")
    file(MAKE_DIRECTORY "${CPU_COMB_GENERATED_DIR}")

    if(EXISTS ${INSTANTIATION_TEMPLATES_3})
        foreach(COMB ${COMBINATIONS_3})
            genPartitionCombination(${INSTANTIATION_TEMPLATES_3} 3 ${COMB} "${CPU_COMB_GENERATED_DIR}")
        endforeach()
    else()
        message(WARNING "CPU Combination Template 3 not found: ${INSTANTIATION_TEMPLATES_3}")
    endif()
    if(EXISTS ${INSTANTIATION_TEMPLATES_2})
        foreach(COMB ${COMBINATIONS_2})
            genPartitionCombination(${INSTANTIATION_TEMPLATES_2} 2 ${COMB} "${CPU_COMB_GENERATED_DIR}")
        endforeach()
    else()
        message(WARNING "CPU Combination Template 2 not found: ${INSTANTIATION_TEMPLATES_2}")
    endif()

    file(GLOB_RECURSE PERF_SOURCES include/performance/*.cpp include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES include/exceptions/*.cpp include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES include/execution/*.cpp include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES include/types/*.cpp include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES include/array/*.cpp include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES include/memory/*.cpp include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES include/graph/*.cpp include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU include/ops/declarable/helpers/cpu/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES include/ops/impl/*.cpp include/ops/declarable/impl/*.cpp include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES include/indexing/*.cpp include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES include/build_info.cpp include/ConstMessages.cpp include/helpers/*.cpp include/helpers/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES include/legacy/impl/*.cpp include/legacy/cpu/*.cpp include/legacy/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES include/loops/*.cpp include/loops/*.h)

    if(HAVE_ONEDNN)
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES include/ops/declarable/platform/mkldnn/*.cpp include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
    endif()
    if(HAVE_ARMCOMPUTE)
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES include/ops/declarable/platform/armcompute/*.cpp include/ops/declarable/platform/armcompute/*.h)
    endif()

    set(ALL_SOURCES
            ${CPU_GENERATED_SOURCES}
            ${CUSTOMOPS_GENERIC_SOURCES}
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
            ${CUSTOMOPS_HELPERS_CPU}
            ${CUSTOMOPS_HELPERS_IMPL}
            ${OPS_SOURCES}
            ${HELPERS_SOURCES}
            ${INDEXING_SOURCES}
            ${LOOPS_SOURCES}
            ${LEGACY_SOURCES}
    )

    if (SD_X86_BUILD AND NOT WIN32)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp")
            set_source_files_properties(include/legacy/cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp")
            set_source_files_properties(include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
    endif()

    message("CPU BLAS (blas section)")
    add_definitions(-D__CPUBLAS__=true)

    if(NOT SD_ALL_OPS)
        message("Not all SD OPS INCLUDED (blas section)")
        set(FILTERED_SOURCES "")
        foreach(SRC_FILE ${ALL_SOURCES})
            set(keep_file TRUE)
            if(EXISTS "${SRC_FILE}")
                file(READ ${SRC_FILE} FILE_CONTENTS)
                string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)
                if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
                    set(found_matching_op FALSE)
                    if(DEFINED SD_OPS_LIST AND NOT "${SD_OPS_LIST}" STREQUAL "")
                        foreach(OP ${SD_OPS_LIST})
                            string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)
                            if(${NOT_EXCLUDED_OP_IDX} GREATER_EQUAL 0)
                                set(found_matching_op TRUE)
                                break()
                            endif()
                        endforeach()
                    endif()
                    if(NOT found_matching_op)
                        set(keep_file FALSE)
                        message(STATUS "Excluding file (no matching NOT_EXCLUDED OP): ${SRC_FILE}")
                    endif()
                endif()
            else()
                set(keep_file FALSE)
            endif()
            if(keep_file)
                list(APPEND FILTERED_SOURCES ${SRC_FILE})
            endif()
        endforeach()
        set(ALL_SOURCES ${FILTERED_SOURCES})
    endif()

    add_library(samediff_obj OBJECT ${ALL_SOURCES})

    target_include_directories(samediff_obj PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}/include
            ${CMAKE_BINARY_DIR}/flatbuffers-src/include
            ${EXTERNAL_INCLUDE_DIRS}
            $<$<BOOL:${HAVE_OPENBLAS}>:${OPENBLAS_PATH}/include>
            $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_ROOT}/include>
    )

    if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        if(ANDROID)
            cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
            if(_logical_cores LESS 4)
                set_target_properties(${SD_LIBRARY_NAME} PROPERTIES JOB_POOL_COMPILE one_jobs)
            endif()
        endif()
    endif()

    if (SD_STATIC_LIB AND SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
        if(WIN32)
            set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        target_link_libraries(${SD_LIBRARY_NAME}static PUBLIC flatbuffers)
        install(TARGETS ${SD_LIBRARY_NAME}static ARCHIVE DESTINATION lib)
    elseif(SD_STATIC_LIB)
        add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
        if(WIN32)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        target_link_libraries(${SD_LIBRARY_NAME} PUBLIC flatbuffers)
        install(TARGETS ${SD_LIBRARY_NAME} ARCHIVE DESTINATION lib)
    endif()

    if(TARGET samediff_obj AND WIN32)
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    endif()
    if(TARGET ${SD_LIBRARY_NAME} AND WIN32)
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    endif()

    if(TARGET ${SD_LIBRARY_NAME})
        if(NOT "$ENV{CLION_IDE}")
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                    $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_LIBRARIES}>
                    $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_LIBRARIES}>
                    ${OPENBLAS_LIBRARIES}
                    ${BLAS_LIBRARIES}
                    Threads::Threads
                    dl
                    flatbuffers
            )
            if(OPENMP_FOUND)
                target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
            endif()
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OPENMP_FOUND)
        endif()
    endif()

    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

    if(TARGET ${SD_LIBRARY_NAME})
        install(TARGETS ${SD_LIBRARY_NAME} LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
    endif()

endif()

option(SD_PREPROCESS "Run C++ Preprocessor" OFF)
if(SD_PREPROCESS STREQUAL "ON")
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}")
    if(NOT ALL_SOURCES)
        message(FATAL_ERROR "SD_PREPROCESS=ON but ALL_SOURCES list is empty. Ensure CUDA or CPU build ran first.")
    endif()
    list(REMOVE_DUPLICATES ALL_SOURCES)
    set(PREPROCESSED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/preprocessed")
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})
    set(PREPROCESSED_FILES "")
    set(PROCESSED_SOURCES "")

    get_target_property(PREPROC_INCLUDES samediff_obj INCLUDE_DIRECTORIES)
    get_target_property(PREPROC_DEFINES samediff_obj COMPILE_DEFINITIONS)
    get_target_property(PREPROC_OPTIONS samediff_obj COMPILE_OPTIONS)
    set(PREPROC_INCLUDE_STR "")
    if(PREPROC_INCLUDES)
        foreach(dir ${PREPROC_INCLUDES})
            if(IS_DIRECTORY "${dir}")
                list(APPEND PREPROC_INCLUDE_STR "-I${dir}")
            endif()
        endforeach()
    endif()
    string(REPLACE ";" " " PREPROC_INCLUDE_STR "${PREPROC_INCLUDE_STR}")
    set(PREPROC_DEFINE_STR "")
    if(PREPROC_DEFINES)
        foreach(d ${PREPROC_DEFINES})
            list(APPEND PREPROC_DEFINE_STR "-D${d}")
        endforeach()
    endif()
    string(REPLACE ";" " " PREPROC_DEFINE_STR "${PREPROC_DEFINE_STR}")
    set(PREPROC_OPTIONS_STR "")
    if(PREPROC_OPTIONS)
        string(REPLACE ";" " " PREPROC_OPTIONS_STR "${PREPROC_OPTIONS}")
    endif()

    if(SD_CUDA AND CMAKE_CUDA_COMPILER)
        set(compiler "${CMAKE_CUDA_COMPILER}")
        set(base_compile_flags "${CMAKE_CUDA_FLAGS}")
    elseif(CMAKE_CXX_COMPILER)
        set(compiler "${CMAKE_CXX_COMPILER}")
        set(base_compile_flags "${CMAKE_CXX_FLAGS}")
    else()
        message(FATAL_ERROR "Cannot determine compiler for preprocessing.")
    endif()

    foreach(src IN LISTS ALL_SOURCES)
        if(EXISTS "${src}" AND NOT src IN_LIST PROCESSED_SOURCES)
            if(src MATCHES "\\.(c|cpp|cxx|cc|cu)$")
                get_filename_component(src_name ${src} NAME_WE)
                get_filename_component(src_dir ${src} DIRECTORY)
                string(REPLACE "${CMAKE_SOURCE_DIR}/" "" rel_path ${src})
                string(REPLACE "${CMAKE_BINARY_DIR}/" "BUILD_DIR/" rel_path ${rel_path})
                string(REPLACE "/" "_" flat_rel_path ${rel_path})
                string(REPLACE "." "_" flat_rel_path ${flat_rel_path})
                set(preprocessed_file "${PREPROCESSED_DIR}/${flat_rel_path}.i")
                message(STATUS "Processing ${src} to ${preprocessed_file}")
                if(NOT EXISTS "${preprocessed_file}")
                    set(pp_command_list ${compiler} -E)
                    list(APPEND pp_command_list ${PREPROC_OPTIONS_STR})
                    list(APPEND pp_command_list ${PREPROC_INCLUDE_STR})
                    list(APPEND pp_command_list ${PREPROC_DEFINE_STR})
                    list(APPEND pp_command_list "${src}")
                    list(APPEND pp_command_list "-o" "${preprocessed_file}")
                    execute_process(
                            COMMAND ${pp_command_list}
                            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                            RESULT_VARIABLE result
                            OUTPUT_VARIABLE stdout_output
                            ERROR_VARIABLE stderr_output
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_STRIP_TRAILING_WHITESPACE
                    )
                    set(output_log_file "${preprocessed_file}.log")
                    file(WRITE ${output_log_file} "COMMAND: ${pp_command_list}\n\nResult: ${result}\n\nStdout:\n${stdout_output}\n\nStderr:\n${stderr_output}\n")
                    if(result EQUAL 0)
                        list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                    else()
                        message(WARNING "Preprocessing failed for ${src}. See ${output_log_file}")
                    endif()
                else()
                    message(STATUS "Preprocessed file already exists, skipping: ${preprocessed_file}")
                    list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                endif()
                list(APPEND PROCESSED_SOURCES ${src})
            endif()
        endif()
    endforeach()
    set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)
    if(PREPROCESSED_FILES)
        add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES})
    endif()
endif()

if(SD_BUILD_TESTS)
    message("--- Configuring Tests ---")
    include(CTest)
    enable_testing()
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests_cpu")
        message(WARNING "SD_BUILD_TESTS enabled, but tests_cpu subdirectory needs its own updated CMakeLists.txt to link correctly.")
    elseif(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        message(WARNING "SD_BUILD_TESTS enabled, but tests subdirectory needs its own updated CMakeLists.txt to link correctly.")
    else()
        message(WARNING "SD_BUILD_TESTS is ON, but a 'tests' or 'tests_cpu' subdirectory was not found.")
    endif()
endif()

message("--- Configuring Installation and Packaging ---")

if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix")
endif()
message("Installation prefix: ${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY include/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl")

if(DEFINED OP_OUTPUT_FILE AND EXISTS "${OP_OUTPUT_FILE}")
    install(FILES ${OP_OUTPUT_FILE} DESTINATION include/${PROJECT_NAME})
endif()
install(FILES ${CMAKE_BINARY_DIR}/include/config.h DESTINATION include/${PROJECT_NAME})

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "Eclipse Deeplearning4j Committers <dl4j-dev@eclipse.org>")
set(CPACK_PACKAGE_VENDOR "Eclipse Foundation")
set(CPACK_PACKAGE_NAME "libnd4j")
if(PROJECT_VERSION)
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
else()
    set(CPACK_PACKAGE_VERSION "1.0.0-SNAPSHOT")
endif()
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+).*" _ ${CPACK_PACKAGE_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")

if(UNIX)
    find_program(LSB_RELEASE_EXEC lsb_release)
    if(LSB_RELEASE_EXEC)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -sc OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        if(EXISTS "/etc/os-release")
            file(STRINGS "/etc/os-release" OS_ID REGEX "^ID=.*")
            if(OS_ID)
                string(REPLACE "ID=" "" DISTRIBUTION "${OS_ID}")
            endif()
        endif()
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(ARCHITECTURE "amd64")
        set(RPM_ARCHITECTURE "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ARCHITECTURE "arm64")
        set(RPM_ARCHITECTURE "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
        set(ARCHITECTURE "armhf")
        set(RPM_ARCHITECTURE "armv7hl")
    else()
        set(ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
        set(RPM_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
    endif()

    if(DISTRIBUTION MATCHES "Ubuntu|Debian")
        set(CPACK_GENERATOR "DEB")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
        set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
        set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/eclipse/deeplearning4j")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
        if(SD_CUDA)
            list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS "cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}")
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS "libcudnn8")
        endif()
        if(CPACK_DEBIAN_PACKAGE_DEPENDS)
            string(REPLACE ";" ", " CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}")
        endif()

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb")
            set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.deb")
        endif()
    elseif(DISTRIBUTION MATCHES "CentOS|Fedora|RHEL")
        set(CPACK_GENERATOR "RPM")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${RPM_ARCHITECTURE})
        set(CPACK_RPM_PACKAGE_ARCHITECTURE ${RPM_ARCHITECTURE})
        set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
        set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries")
        set(CPACK_RPM_PACKAGE_URL "https://github.com/eclipse/deeplearning4j")
        set(CPACK_RPM_PACKAGE_REQUIRES "")
        if(SD_CUDA)
            list(APPEND CPACK_RPM_PACKAGE_REQUIRES "cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}")
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_RPM_PACKAGE_REQUIRES "libcudnn8")
        endif()
        if(CPACK_RPM_PACKAGE_REQUIRES)
            string(REPLACE ";" ", " CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}")
        endif()

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
            set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
            set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
        endif()
    else()
        set(CPACK_GENERATOR "TGZ")
    endif()
elseif(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME}")
    set(CPACK_NSIS_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
    set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-macOS)
endif()

include(CPack)

message("--- CMake Configuration Complete ---")