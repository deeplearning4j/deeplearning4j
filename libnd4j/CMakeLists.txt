# Main CMakeLists.txt (incorporating fixes, comments removed)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(libnd4j)

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
option(SD_PREPROCESS "Use preprocessor" OFF)
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
    message("Added print math compile definition")
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

option(HELPERS_onednn "Build with oneDNN" OFF)
option(HELPERS_armcompute "Build with ARM Compute Library" OFF)
option(HELPERS_cudnn "Build with cuDNN" OFF)

set(MSVC_RT_LIB "MultiThreadedDLL")

set(SD_X86_BUILD false)
set(SD_ARM_BUILD false)

if(DEFINED ENV{SD_ARCH})
    set(SD_ARCH $ENV{SD_ARCH})
elseif(NOT DEFINED SD_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_ARCH "x86-64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(SD_ARCH "arm64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(SD_ARCH "armv7-a")
    else()
        set(SD_ARCH ${CMAKE_SYSTEM_PROCESSOR})
    endif()
    message(STATUS "SD_ARCH not defined via environment or CMake cache, determined as: ${SD_ARCH}")
endif()


if(SD_ANDROID_BUILD)
    if(DEFINED ANDROID_ABI)
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
    else()
        message(WARNING "SD_ANDROID_BUILD is ON, but ANDROID_ABI is not defined by the toolchain file.")
    endif()
elseif(NOT SD_IOS_BUILD)
    if(SD_ARCH MATCHES "x86_64|x86-64|amd64|i[3-6]86")
        set(SD_X86_BUILD true)
    elseif(SD_ARCH MATCHES "arm*|aarch64*")
        set(SD_ARM_BUILD true)
    endif()
elseif(SD_IOS_BUILD)
    if(SD_ARCH MATCHES "arm*|aarch64*")
        set(SD_ARM_BUILD true)
    else()
        set(SD_X86_BUILD true)
    endif()
endif()

if(${SD_ARCH} MATCHES "armv7")
    set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
    message("Using ARM Compute Library for ARMv7 32-bit")
elseif(${SD_ARCH} MATCHES "armv8-a|arm64|aarch64")
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (arch: ${SD_ARCH})")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for Linux ARM64 (arch: ${SD_ARCH})")
    endif()
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_IOS_BUILD=${SD_IOS_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")

if (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE OR SD_IOS_BUILD)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "arm64|aarch64|armv8-a")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
    if(SD_IOS_BUILD)
    endif()
elseif(WIN32)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    endif()
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
        message("Using sanitizers: ${SD_SANITIZERS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
        if(SD_CUDA)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${SANITIZE_FLAGS}\"")
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
        if (NOT APPLE AND NOT SD_IOS_BUILD)
            set(BLA_VENDOR "OpenBLAS" CACHE STRING "Selected BLAS vendor" FORCE)
        endif()

        find_package(BLAS)

        if (BLAS_FOUND)
            message(STATUS "Found external BLAS implementation: ${BLAS_LIBRARIES}")
            set(HAVE_OPENBLAS TRUE)
            set(OPENBLAS_LIBRARIES ${BLAS_LIBRARIES})
        else()
            message(WARNING "No external BLAS implementation found by find_package(BLAS). Performance may be limited.")
        endif()
    else()
        set(HAVE_OPENBLAS TRUE)
        message(STATUS "Using user-provided OpenBLAS path: ${OPENBLAS_PATH}")
        include_directories(${OPENBLAS_PATH}/include)
        link_directories(${OPENBLAS_PATH}/lib)
        find_library(OPENBLAS_LIBRARY NAMES openblas libopenblas HINTS ${OPENBLAS_PATH}/lib REQUIRED)
        if(OPENBLAS_LIBRARY)
            set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
            message(STATUS "Found OpenBLAS library: ${OPENBLAS_LIBRARIES}")
        else()
            message(FATAL_ERROR "OPENBLAS_PATH was provided, but library not found in ${OPENBLAS_PATH}/lib")
        endif()
        add_definitions(-DHAVE_OPENBLAS=1)
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
        add_definitions(-DHAVE_ARMCOMPUTE=1 -DARMCOMPUTENEON_ENABLED)
    else()
        message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
        if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7" OR ${SD_ARCH} MATCHES "armv8-a" OR ${SD_ARCH} MATCHES "arm64"))
            message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")

            include(ExternalProject)

            ExternalProject_Add(armcompute_external
                    URL ${ARM_COMPUTE_URL}
                    SOURCE_DIR "${CMAKE_BINARY_DIR}/armcompute-src"
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    LOG_DOWNLOAD TRUE
            )

            set(ARMCOMPUTE_ROOT "${CMAKE_BINARY_DIR}/armcompute-src")
            set(ARMCOMPUTE_INCLUDE_DIR_LIB "${ARMCOMPUTE_ROOT}/include")
            set(ARMCOMPUTE_INCLUDE_DIR_CORE "${ARMCOMPUTE_ROOT}/include")
            set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so")
            set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so")

            add_library(arm_compute SHARED IMPORTED GLOBAL)
            set_property(TARGET arm_compute PROPERTY IMPORTED_LOCATION ${ARMCOMPUTE_LIB_PATH})
            add_dependencies(arm_compute armcompute_external)

            add_library(arm_compute_core SHARED IMPORTED GLOBAL)
            set_property(TARGET arm_compute_core PROPERTY IMPORTED_LOCATION ${ARMCOMPUTE_CORE_LIB_PATH})
            add_dependencies(arm_compute_core armcompute_external)

            include_directories(${ARMCOMPUTE_INCLUDE_DIR_LIB} ${ARMCOMPUTE_INCLUDE_DIR_CORE})
            set(ARMCOMPUTE_LIBRARIES arm_compute arm_compute_core)
            set(HAVE_ARMCOMPUTE 1)
            add_definitions(-DHAVE_ARMCOMPUTE=1 -DARMCOMPUTENEON_ENABLED)
            message(STATUS "ARM Compute Library configured via download at: ${ARMCOMPUTE_ROOT}")

        else()
            message(STATUS "ARM Compute helper active, but library not found and download conditions not met (URL: ${ARM_COMPUTE_URL}, Arch: ${SD_ARCH}). Build will continue without ARM Compute.")
        endif()
    endif()
endif()


if (${HELPERS_onednn})
    message("Going to pull & build onednn")
    set(HAVE_ONEDNN 1)
    add_definitions(-DHAVE_ONEDNN=1)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Build oneDNN as a static library" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    include(ExternalProject)
    ExternalProject_Add(onednn_external
            GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
            GIT_TAG v2.7
            SOURCE_DIR "${CMAKE_BINARY_DIR}/onednn-src"
            BINARY_DIR "${CMAKE_BINARY_DIR}/onednn-build"
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDNNL_LIBRARY_TYPE=${DNNL_LIBRARY_TYPE}
            -DDNNL_BUILD_EXAMPLES=${DNNL_BUILD_EXAMPLES}
            -DDNNL_BUILD_TESTS=${DNNL_BUILD_TESTS}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
            INSTALL_COMMAND ""
            LOG_CONFIGURE TRUE
            LOG_BUILD TRUE
            LOG_DOWNLOAD TRUE
    )

    add_library(onednn STATIC IMPORTED GLOBAL)
    set_property(TARGET onednn PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/onednn-build/src/libdnnl.a)
    add_dependencies(onednn onednn_external)

    set(ONEDNN_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/onednn-src/include)
    include_directories(${ONEDNN_INCLUDE_DIRS})
    set(ONEDNN_LIBRARIES onednn)

    message(STATUS "oneDNN configured via ExternalProject. Include: ${ONEDNN_INCLUDE_DIRS}")

endif()


if(${HELPERS_cudnn})
    if(NOT SD_CUDA)
        message(FATAL_ERROR "Can't build with cuDNN helper on a non-CUDA platform (SD_CUDA is OFF)")
    endif()

    SET(CUDNN_LIBNAME "cudnn")

    if(DEFINED ENV{CUDNN_ROOT_DIR})
        set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
        message(STATUS "Using CUDNN_ROOT_DIR from environment: ${CUDNN_ROOT_DIR}")
    endif()
    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
        message(STATUS "Using CUDA_TOOLKIT_ROOT_DIR from environment: ${CUDA_TOOLKIT_ROOT_DIR}")
    elseif(DEFINED CUDA_TOOLKIT_ROOT_DIR)
        message(STATUS "Using CUDA_TOOLKIT_ROOT_DIR from CUDA find_package: ${CUDA_TOOLKIT_ROOT_DIR}")
    endif()

    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES include include/cuda cuda/include)

    find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIBNAME} cudnn_static
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES lib lib64 lib/x64 cuda/lib cuda/lib64 targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib)

    message(STATUS "cuDNN search: Include Dir = ${CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN search: Library = ${CUDNN_LIBRARY}")

    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message(STATUS "Found cuDNN: include at ${CUDNN_INCLUDE_DIR}, library at ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true CACHE BOOL "cuDNN Found" FORCE)
        add_definitions(-DHAVE_CUDNN=1)
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        message(WARNING "cuDNN helper was specified, but cuDNN library or include directory not found. Build will continue without cuDNN support.")
        set(HAVE_CUDNN false CACHE BOOL "cuDNN Found" FORCE)
    endif()
endif()

# ================== Refactored Flatbuffers Section ==================
include(ExternalProject)

set(FLATBUFFERS_TAG "v24.3.25" CACHE STRING "Flatbuffers Git Tag")
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Build Flatbuffers Tests")

set(GENERATE_FLATC OFF)
if(DEFINED ENV{GENERATE_FLATC})
    if("$ENV{GENERATE_FLATC}" STREQUAL "ON")
        set(GENERATE_FLATC ON)
        message(STATUS "Building flatc based on ENV{GENERATE_FLATC}=ON")
    endif()
endif()
set(FLATBUFFERS_BUILD_FLATC ${GENERATE_FLATC} CACHE BOOL "Build flatc executable" FORCE)

ExternalProject_Add(flatbuffers_external
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG ${FLATBUFFERS_TAG}
        SOURCE_DIR "${CMAKE_BINARY_DIR}/flatbuffers-src"
        BINARY_DIR "${CMAKE_BINARY_DIR}/flatbuffers-build"
        CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DFLATBUFFERS_BUILD_TESTS=${FLATBUFFERS_BUILD_TESTS}
        -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC}
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        $<$<BOOL:${WIN32}>:-DCMAKE_MSVC_RUNTIME_LIBRARY=${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>>
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG>
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_DOWNLOAD TRUE
)

set(FLATBUFFERS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/flatbuffers-src/include)
include_directories(${FLATBUFFERS_INCLUDE_DIR})
message(STATUS "Flatbuffers include directory set to: ${FLATBUFFERS_INCLUDE_DIR}")

add_library(flatbuffers STATIC IMPORTED GLOBAL)
set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/libflatbuffers.a)
if(WIN32)
    set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/$<IF:$<CONFIG:Debug>,Debug,Release>/flatbuffers.lib)
endif()
add_dependencies(flatbuffers flatbuffers_external)

if(FLATBUFFERS_BUILD_FLATC)
    add_executable(flatc IMPORTED GLOBAL)
    set(FLATC_EXE_PATH "${CMAKE_BINARY_DIR}/flatbuffers-build/flatc")
    if(WIN32)
        set(FLATC_EXE_PATH "${CMAKE_BINARY_DIR}/flatbuffers-build/$<IF:$<CONFIG:Debug>,Debug,Release>/flatc.exe")
    endif()
    set_property(TARGET flatc PROPERTY IMPORTED_LOCATION ${FLATC_EXE_PATH})
    add_dependencies(flatc flatbuffers_external)
    message(STATUS "flatc executable IMPORTED target created. Path will be resolved at build time: ${FLATC_EXE_PATH}")
endif()
# ================== End Refactored Flatbuffers Section ==================


# ================== Generate Flatbuffers Sources Section (FIXED) ==================
if(FLATBUFFERS_BUILD_FLATC)
    message("Generating flatbuffers sources (FLATBUFFERS_BUILD_FLATC is ON)...")

    set(FLATC_TARGET flatc)

    set(FLATC_INPUT_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/scheme/graph.fbs")
    set(FLATC_OUTPUT_DIR_CPP "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd")
    set(FLATC_OUTPUT_DIR_JAVA "${CMAKE_CURRENT_SOURCE_DIR}/../../nd4j-serde/nd4j-flatbuffers-java/src/main/java")

    file(MAKE_DIRECTORY ${FLATC_OUTPUT_DIR_CPP})
    file(MAKE_DIRECTORY ${FLATC_OUTPUT_DIR_JAVA})

    add_custom_command(
            OUTPUT ${FLATC_OUTPUT_DIR_CPP}/graph_generated.h
            COMMAND $<TARGET_FILE:${FLATC_TARGET}>
            -o ${FLATC_OUTPUT_DIR_CPP}
            --cpp
            --gen-object-api
            -o ${FLATC_OUTPUT_DIR_JAVA}
            --java
            --gen-object-api
            ${FLATC_INPUT_SCHEMA}
            DEPENDS ${FLATC_TARGET} ${FLATC_INPUT_SCHEMA}
            COMMENT "Generating flatbuffers C++ and Java sources from ${FLATC_INPUT_SCHEMA}"
            VERBATIM
    )

    add_custom_target(generate_flatbuffers ALL DEPENDS ${FLATC_OUTPUT_DIR_CPP}/graph_generated.h)

else()
    message("Skipping flatbuffers source generation (FLATBUFFERS_BUILD_FLATC is OFF).")
endif()
# ================== End Generate Flatbuffers Sources Section ==================


configure_file(include/config.h.in include/config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ===========================================================================
# Start of Content Pasted Verbatim from blas/CMakeLists.txt
# ===========================================================================

include(CheckCXXCompilerFlag)

if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD)
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(APPLE)
    message(STATUS "Configuring for Apple OS (macOS/iOS)")
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(SD_APPLE_BUILD OR APPLE OR SD_IOS_BUILD)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_APPLE_BUILD=true")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_APPLE_BUILD=true")
    if (NOT SD_IOS_BUILD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.10")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.10")
    endif()
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
    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
        set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
        set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    endif()
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
endif()

if(SD_USE_LTO)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Enabling Link Time Optimization (LTO)")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "Enabling Link Time Optimization (LTCG) for MSVC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /GL")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GL")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /LTCG")
    else()
        message(WARNING "LTO requested but not supported for compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

set(DEFINITIONS_CONTENT "")
if((DEFINED SD_ALL_OPS AND SD_ALL_OPS) OR NOT DEFINED SD_OPS_LIST OR "${SD_OPS_LIST}" STREQUAL "")
    message("Adding all ops due to SD_ALL_OPS=${SD_ALL_OPS} or empty SD_OPS_LIST=${SD_OPS_LIST}")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
    message("Adding selective ops from list: ${SD_OPS_LIST}")
    add_compile_definitions(SD_SELECTIVE_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_OPS 1\n")
    foreach(OP ${SD_OPS_LIST})
        string(TOUPPER ${OP} OP_UPPER)
        add_compile_definitions(OP_${OP_UPPER}=1)
        message(STATUS "Adding op definition: OP_${OP_UPPER}=1")
        string(APPEND DEFINITIONS_CONTENT "#define OP_${OP_UPPER} 1\n")
    endforeach()
endif()

list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST_COUNT GREATER 0)
    message("Adding selective types from list: ${SD_TYPES_LIST}")
    add_compile_definitions(SD_SELECTIVE_TYPES=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_TYPES 1\n")
    foreach(SD_TYPE ${SD_TYPES_LIST})
        string(TOUPPER ${SD_TYPE} SD_TYPE_UPPERCASE)
        add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
        message(STATUS "Adding type definition: HAS_${SD_TYPE_UPPERCASE}")
        string(APPEND DEFINITIONS_CONTENT "#define HAS_${SD_TYPE_UPPERCASE}\n")
    endforeach()
else()
    message("All types enabled (SD_TYPES_LIST not defined or empty)")
endif()

if(DEFINED OP_OUTPUT_FILE AND OP_OUTPUT_FILE)
    if(NOT IS_ABSOLUTE "${OP_OUTPUT_FILE}")
        set(OP_OUTPUT_FILE_ABS "${CMAKE_CURRENT_BINARY_DIR}/${OP_OUTPUT_FILE}")
    else()
        set(OP_OUTPUT_FILE_ABS "${OP_OUTPUT_FILE}")
    endif()
    get_filename_component(OP_OUTPUT_DIR ${OP_OUTPUT_FILE_ABS} DIRECTORY)
    file(MAKE_DIRECTORY ${OP_OUTPUT_DIR})
    message(STATUS "Definitions will be written to \"${OP_OUTPUT_FILE_ABS}\"")
    file(WRITE "${OP_OUTPUT_FILE_ABS}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n\n${DEFINITIONS_CONTENT}\n#endif // SD_DEFINITIONS_GEN_H_\n")
else()
    message(STATUS "OP_OUTPUT_FILE not defined, skipping generation of definitions header.")
endif()

set(ARCH_TUNE "")
if(SD_ARCH)
    if(SD_ARCH MATCHES "armv8|aarch64")
        set(ARCH_TUNE "-march=armv8-a")
    elseif(SD_ARCH MATCHES "armv7")
        set(ARCH_TUNE "-march=armv7-a -mfpu=neon")
    elseif(SD_ARCH MATCHES "power*")
        set(ARCH_TUNE "-mcpu=${SD_ARCH} -mtune=${SD_ARCH} -D__POWER")
    elseif(SD_ARCH MATCHES "x86-64|x86_64|amd64")
        if(DEFINED SD_EXTENSION)
            if(SD_EXTENSION MATCHES "avx512")
                message("Building for x86-64 with AVX512 extension...")
                set(ARCH_TUNE "-mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd")
                set(ARCH_TUNE "-march=skylake-avx512 ${ARCH_TUNE}")
                add_compile_definitions(F_AVX512=true)
            elseif(SD_EXTENSION MATCHES "avx2")
                message("Building for x86-64 with AVX2 extension...")
                set(ARCH_TUNE "-mavx2 -mfma -mf16c")
                set(ARCH_TUNE "-march=haswell ${ARCH_TUNE}")
                add_compile_definitions(F_AVX2=true)
                check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" NO_AVX256_SPLIT)
                if(NO_AVX256_SPLIT)
                    set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
                endif()
            else()
                message("Building for generic x86-64...")
                set(ARCH_TUNE "-march=x86-64")
                add_compile_definitions(F_X64=true)
            endif()
        else()
            message("Building for generic x86-64 (SD_EXTENSION not set)...")
            set(ARCH_TUNE "-march=x86-64")
            add_compile_definitions(F_X64=true)
        endif()
    else()
        message(WARNING "Unhandled SD_ARCH value: ${SD_ARCH}. No specific tuning flags added.")
    endif()

    if(ARCH_TUNE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_TUNE}")
    elseif(ARCH_TUNE AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(SD_EXTENSION MATCHES "avx512")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX512")
        elseif(SD_EXTENSION MATCHES "avx2")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        endif()
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD AND SD_SHARED_LIB)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN'")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT APPLE AND NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
    endif()
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        add_compile_definitions(SD_GCC_FUNCTRACE)
    endif()
    elif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD AND SD_SHARED_LIB)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN'")
    endif()
    elif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast")
    elif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc")
endif()

set(EXTERNAL_INCLUDE_DIRS "")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
endif()

if(NOT DEFINED SD_CUDA AND NOT DEFINED SD_CPU)
    message(WARNING "Neither SD_CUDA nor SD_CPU is defined. Assuming SD_CPU=TRUE.")
    set(SD_CPU TRUE)
    set(SD_CUDA FALSE)
elseif(DEFINED SD_CUDA AND SD_CUDA)
    set(SD_CPU FALSE)
elseif(DEFINED SD_CPU AND SD_CPU)
    set(SD_CUDA FALSE)
endif()
message(STATUS "Build Mode: SD_CPU=${SD_CPU}, SD_CUDA=${SD_CUDA}")

if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    elseif(SD_CPU)
        set(SD_LIBRARY_NAME nd4jcpu)
    else()
        message(FATAL_ERROR "Cannot determine SD_LIBRARY_NAME. Build mode (SD_CPU/SD_CUDA) is unclear.")
    endif()
endif()
message(STATUS "Target library name: ${SD_LIBRARY_NAME}")


set(ALL_SOURCES "")
set(CUSTOMOPS_ONEDNN_SOURCES "")
set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
set(CUSTOMOPS_CUDNN_SOURCES "")
set(CUDA_GENERATED_SOURCES "")
set(CPU_GENERATED_SOURCES "")
set(CUSTOMOPS_GENERIC_SOURCES "")

if(SD_CUDA)
    message(STATUS "--- Configuring CUDA Build ---")

    add_definitions(-D__CUDABLAS__=true)

    if(NOT CUDA_FOUND)
        find_package(CUDA REQUIRED)
    endif()

    if(CUDA_FOUND)
        message(STATUS "CUDA Found: Version=${CUDA_VERSION}, Toolkit Root=${CUDA_TOOLKIT_ROOT_DIR}")
        include_directories(${CUDA_INCLUDE_DIRS})

        set(COMMON_CUDA_FLAGS "-allow-unsupported-compiler --expt-extended-lambda -Xfatbin -compress-all -maxrregcount=128")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS}")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        endif()

        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

        if(NOT DEFINED COMPUTE)
            set(COMPUTE "all")
            message(STATUS "COMPUTE variable not set, defaulting to 'all'")
        endif()
        string(TOLOWER "${COMPUTE}" COMPUTE_LOWER)

        set(CUDA_ARCH_FLAGS_LIST "")
        if(COMPUTE_LOWER STREQUAL "all")
            if(CUDA_VERSION VERSION_LESS 11.0)
                set(CUDA_ARCH_LIST 60 61 70 75)
            elseif(CUDA_VERSION VERSION_LESS 12.0)
                set(CUDA_ARCH_LIST 60 61 70 75 80 86)
            else()
                set(CUDA_ARCH_LIST 75 80 86 90)
            endif()
        elseif(COMPUTE_LOWER STREQUAL "auto")
            set(CUDA_ARCH_LIST "")
        else()
            string(REPLACE " " ";" CUDA_ARCH_LIST_INPUT ${COMPUTE_LOWER})
            set(CUDA_ARCH_LIST "")
            foreach(ARCH_ITEM ${CUDA_ARCH_LIST_INPUT})
                string(REPLACE "." "" ARCH_CLEAN ${ARCH_ITEM})
                list(APPEND CUDA_ARCH_LIST ${ARCH_CLEAN})
            endforeach()
        endif()

        if(CUDA_ARCH_LIST)
            foreach(ARCH_CODE ${CUDA_ARCH_LIST})
                list(APPEND CUDA_ARCH_FLAGS_LIST "-gencode" "arch=compute_${ARCH_CODE},code=compute_${ARCH_CODE}")
                list(APPEND CUDA_ARCH_FLAGS_LIST "-gencode" "arch=compute_${ARCH_CODE},code=sm_${ARCH_CODE}")
            endforeach()
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
            message(STATUS "Targeting CUDA Architectures (sm_): ${CUDA_ARCH_LIST}")
        else()
            message(STATUS "Targeting CUDA Architectures: Auto-detected by CMake/nvcc")
            set(CMAKE_CUDA_ARCHITECTURES "")
        endif()

        string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS_LIST}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

        if(SD_SANITIZE AND SANITIZE_FLAGS)
            message(STATUS "Adding sanitizer flags to CUDA host/device compiler via -Xcompiler")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${SANITIZE_FLAGS}\"")
            if(CUDA_VERSION VERSION_GREATER_EQUAL 10.1)
            endif()
        endif()

        find_package(CUDAToolkit COMPONENTS cublas cusolver)
        if(CUDAToolkit_FOUND)
            message(STATUS "Found CUDA Toolkit components: cuBLAS, cuSolver")
        else()
            message(WARNING "Could not find required CUDA Toolkit components (cuBLAS, cuSolver) via find_package(CUDAToolkit). Linking might fail.")
            find_library(CUDA_cublas_LIBRARY NAMES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
            find_library(CUDA_cusolver_LIBRARY NAMES cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
            if(NOT CUDA_cublas_LIBRARY OR NOT CUDA_cusolver_LIBRARY)
                message(FATAL_ERROR "Manual search for cuBLAS/cuSolver library failed.")
            endif()
            set(CUDAToolkit_LIBRARIES ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
        endif()

        message(STATUS "Configuring CUDA source file generation...")
        set(CUDA_GENERATED_DIR "${CMAKE_BINARY_DIR}/cuda_generated")
        file(MAKE_DIRECTORY ${CUDA_GENERATED_DIR})
        message(STATUS "CUDA source generation configured (Actual generation happens at build time).")

        file(GLOB_RECURSE COMMON_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/generic/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/types/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/types/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/build_info.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ConstMessages.cpp"
                EXCLUDE REGEX ".*/cpu/.*\\.cpp$"
        )

        file(GLOB_RECURSE CUDA_SPECIFIC_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cuda/*.h"
        )

        if(HAVE_CUDNN)
            message(STATUS "Adding cuDNN platform sources")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/cudnn/*.cu")
        endif()

        set(ALL_SOURCES
                ${COMMON_SOURCES}
                ${CUDA_SPECIFIC_SOURCES}
                ${CUDA_GENERATED_SOURCES}
                ${CUSTOMOPS_CUDNN_SOURCES}
        )

        add_library(samediff_obj OBJECT ${ALL_SOURCES})
        target_compile_features(samediff_obj PUBLIC cxx_std_17 cuda_std_17)

        target_include_directories(samediff_obj PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_BINARY_DIR}/include"
                ${FLATBUFFERS_INCLUDE_DIR}
                ${CUDA_INCLUDE_DIRS}
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN_INCLUDE_DIR}>
                $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
        )

        target_compile_definitions(samediff_obj PUBLIC
                $<$<BOOL:${HAVE_CUDNN}>:HAVE_CUDNN=1>
                $<$<BOOL:${HAVE_ONEDNN}>:HAVE_ONEDNN=1>
                __CUDABLAS__=true
        )

        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            target_compile_options(samediff_obj PRIVATE /EHsc /bigobj)
        endif()

        if(SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
            set(FINAL_TARGET ${SD_LIBRARY_NAME})
            message(STATUS "Creating shared library: ${SD_LIBRARY_NAME}")
            if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                set_property(TARGET ${FINAL_TARGET} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            endif()
        endif()

        if(SD_STATIC_LIB)
            add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:samediff_obj>)
            if(NOT SD_SHARED_LIB)
                set(FINAL_TARGET ${SD_LIBRARY_NAME}_static)
            endif()
            message(STATUS "Creating static library: ${SD_LIBRARY_NAME}_static")
            if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            endif()
            target_link_libraries(${SD_LIBRARY_NAME}_static PUBLIC flatbuffers)
            install(TARGETS ${SD_LIBRARY_NAME}_static ARCHIVE DESTINATION lib)
        endif()

        if(NOT DEFINED FINAL_TARGET)
            message(FATAL_ERROR "No library type selected (SD_SHARED_LIB or SD_STATIC_LIB must be ON)")
        endif()

        target_link_libraries(${FINAL_TARGET} PUBLIC
                CUDA::cudart
                ${CUDAToolkit_LIBRARIES}
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN}>
                $<$<BOOL:${HAVE_ONEDNN}>:onednn>
                Threads::Threads
                flatbuffers
                $<$<BOOL:${SD_SANITIZE}>:asan>
        )
        if (UNIX AND NOT APPLE)
            target_link_libraries(${FINAL_TARGET} PUBLIC dl)
        endif()

        if(SD_SHARED_LIB)
            install(TARGETS ${SD_LIBRARY_NAME}
                    LIBRARY DESTINATION lib
                    RUNTIME DESTINATION bin
                    ARCHIVE DESTINATION lib
            )
        endif()

        message(STATUS "CUDA Build Configuration Complete.")

    else()
        message(FATAL_ERROR "CUDA was enabled (SD_CUDA=ON) but not found by find_package(CUDA).")
    endif()

elseif(SD_CPU)
    message(STATUS "--- Configuring CPU Build ---")
    add_definitions(-D__CPUBLAS__=true)

    message(STATUS "Configuring CPU source file generation...")
    set(CPU_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated")
    file(MAKE_DIRECTORY ${CPU_GENERATED_DIR})
    set(CPU_COMB_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated_combinations")
    file(MAKE_DIRECTORY ${CPU_COMB_GENERATED_DIR})

    message(STATUS "CPU source generation configured (Actual generation happens at build time).")

    file(GLOB_RECURSE COMMON_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/generic/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/array/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/array/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/impl/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/build_info.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ConstMessages.cpp"
            EXCLUDE REGEX ".*/cuda/.*\\.cu$"
            EXCLUDE REGEX ".*/cuda/.*\\.h$"
            EXCLUDE REGEX ".*/cudnn/.*\\.cu$"
            EXCLUDE REGEX ".*/armcompute/.*\\.cpp$"
    )

    file(GLOB_RECURSE CPU_SPECIFIC_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/*.h"
    )

    if(HAVE_ONEDNN)
        message(STATUS "Adding oneDNN platform sources")
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/mkldnn/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/mkldnn/*.h")
    endif()
    if(HAVE_ARMCOMPUTE)
        message(STATUS "Adding ARM Compute platform sources")
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/armcompute/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/armcompute/*.h")
    endif()

    set(ALL_SOURCES
            ${COMMON_SOURCES}
            ${CPU_SPECIFIC_SOURCES}
            ${CPU_GENERATED_SOURCES}
            ${CUSTOMOPS_GENERIC_SOURCES}
            ${CUSTOMOPS_ONEDNN_SOURCES}
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
    )

    if (SD_X86_BUILD AND NOT WIN32)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp" PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp" PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
    endif()

    if(NOT SD_ALL_OPS)
        message(STATUS "Filtering sources based on selective ops list...")
    endif()

    add_library(samediff_obj OBJECT ${ALL_SOURCES})
    target_compile_features(samediff_obj PUBLIC cxx_std_17)

    target_include_directories(samediff_obj PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            ${FLATBUFFERS_INCLUDE_DIR}
            $<$<BOOL:${HAVE_OPENBLAS}>:${OPENBLAS_PATH}/include>
            $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_INCLUDE_DIR_LIB}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_INCLUDE_DIR_CORE}>
    )

    target_compile_definitions(samediff_obj PUBLIC
            $<$<BOOL:${HAVE_OPENBLAS}>:HAVE_OPENBLAS=1>
            $<$<BOOL:${HAVE_ONEDNN}>:HAVE_ONEDNN=1>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:HAVE_ARMCOMPUTE=1>
            __CPUBLAS__=true
    )

    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        target_compile_options(samediff_obj PRIVATE /EHsc /bigobj)
    endif()

    if(SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        set(FINAL_TARGET ${SD_LIBRARY_NAME})
        message(STATUS "Creating shared library: ${SD_LIBRARY_NAME}")
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set_property(TARGET ${FINAL_TARGET} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        if(UNIX OR APPLE)
            set_target_properties(${FINAL_TARGET} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)
        endif()
    endif()

    if(SD_STATIC_LIB)
        add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:samediff_obj>)
        if(NOT SD_SHARED_LIB)
            set(FINAL_TARGET ${SD_LIBRARY_NAME}_static)
        endif()
        message(STATUS "Creating static library: ${SD_LIBRARY_NAME}_static")
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        target_link_libraries(${SD_LIBRARY_NAME}_static PUBLIC flatbuffers)
        install(TARGETS ${SD_LIBRARY_NAME}_static ARCHIVE DESTINATION lib)
    endif()

    if(NOT DEFINED FINAL_TARGET)
        message(FATAL_ERROR "No library type selected (SD_SHARED_LIB or SD_STATIC_LIB must be ON)")
    endif()

    find_package(OpenMP)
    if(OPENMP_FOUND)
        message(STATUS "OpenMP found, enabling OpenMP support.")
        set(OpenMP_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    else()
        message(STATUS "OpenMP not found.")
        set(OpenMP_LINK_LIBRARIES "")
    endif()

    target_link_libraries(${FINAL_TARGET} PUBLIC
            ${OPENBLAS_LIBRARIES}
            $<$<BOOL:${HAVE_ONEDNN}>:onednn>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_LIBRARIES}>
            Threads::Threads
            flatbuffers
            ${OpenMP_LINK_LIBRARIES}
            $<$<BOOL:${SD_SANITIZE}>:asan>
    )
    if (UNIX AND NOT APPLE)
        target_link_libraries(${FINAL_TARGET} PUBLIC dl)
    endif()

    if(SD_SHARED_LIB)
        install(TARGETS ${SD_LIBRARY_NAME}
                LIBRARY DESTINATION lib
                RUNTIME DESTINATION bin
                ARCHIVE DESTINATION lib
        )
    endif()

    message(STATUS "CPU Build Configuration Complete.")

else()
    message(FATAL_ERROR "Build mode not specified correctly. Set either -DSD_CPU=ON or -DSD_CUDA=ON.")
endif()

# ===========================================================================
# End of Content Pasted Verbatim from blas/CMakeLists.txt
# ===========================================================================


if(SD_PREPROCESS)
    message(STATUS "--- Configuring Preprocessing Step ---")
    if(NOT ALL_SOURCES)
        message(FATAL_ERROR "SD_PREPROCESS is ON, but the ALL_SOURCES list is empty. Ensure CUDA or CPU configuration ran first.")
    endif()

    list(REMOVE_DUPLICATES ALL_SOURCES)

    set(PREPROCESSED_DIR "${CMAKE_BINARY_DIR}/preprocessed")
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})
    message(STATUS "Preprocessing output directory: ${PREPROCESSED_DIR}")

    set(PREPROCESSED_FILES "")
    set(PROCESSED_SOURCES_SET "")

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

    set(PREPROC_DEFINE_STR "")
    if(PREPROC_DEFINES)
        foreach(d ${PREPROC_DEFINES})
            list(APPEND PREPROC_DEFINE_STR "-D${d}")
        endforeach()
    endif()

    set(PREPROC_OPTIONS_STR "")
    if(PREPROC_OPTIONS)
    endif()

    set(preprocessor "")
    set(base_preproc_flags "")
    if(SD_CUDA AND CMAKE_CUDA_COMPILER)
        set(preprocessor ${CMAKE_CUDA_COMPILER})
        set(base_preproc_flags ${CMAKE_CUDA_FLAGS})
        list(APPEND base_preproc_flags "--preprocess")
    elseif(CMAKE_CXX_COMPILER)
        set(preprocessor ${CMAKE_CXX_COMPILER})
        set(base_preproc_flags ${CMAKE_CXX_FLAGS})
        list(APPEND base_preproc_flags "-E")
    else()
        message(FATAL_ERROR "Cannot determine compiler for preprocessing.")
    endif()

    foreach(src IN LISTS ALL_SOURCES)
        get_filename_component(src_abs ${src} ABSOLUTE)
        if(EXISTS "${src_abs}" AND NOT "${src_abs}" IN_LIST PROCESSED_SOURCES_SET)
            if(src_abs MATCHES "\\.(c|cpp|cxx|cc|cu)$")
                string(REPLACE "${CMAKE_SOURCE_DIR}/" "" rel_path ${src_abs})
                string(REPLACE "${CMAKE_BINARY_DIR}/" "BUILD_DIR/" rel_path ${rel_path})
                string(REPLACE "/" "_" flat_rel_path ${rel_path})
                string(REPLACE "." "_" flat_rel_path ${flat_rel_path})

                set(preprocessed_file "${PREPROCESSED_DIR}/${flat_rel_path}.i")
                message(STATUS "Preprocessing: ${src_abs} -> ${preprocessed_file}")

                set(pp_command ${preprocessor})
                list(APPEND pp_command ${base_preproc_flags})
                list(APPEND pp_command ${PREPROC_OPTIONS_STR})
                list(APPEND pp_command ${PREPROC_INCLUDE_STR})
                list(APPEND pp_command ${PREPROC_DEFINE_STR})
                list(APPEND pp_command "${src_abs}")
                list(APPEND pp_command "-o" "${preprocessed_file}")

                add_custom_command(
                        OUTPUT ${preprocessed_file}
                        COMMAND ${pp_command}
                        DEPENDS ${src_abs}
                        COMMENT "Preprocessing ${src_abs}"
                        VERBATIM
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                )
                list(APPEND PREPROCESSED_FILES ${preprocessed_file})
            endif()
            list(APPEND PROCESSED_SOURCES_SET "${src_abs}")
        endif()
    endforeach()

    if(PREPROCESSED_FILES)
        add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES}
                COMMENT "Running C/C++/CUDA preprocessor on sources..."
        )
        set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)
    endif()
endif()


if(SD_BUILD_TESTS)
    message(STATUS "--- Configuring Tests ---")
    include(CTest)
    enable_testing()

    set(TEST_DIR_NAME "tests")
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR_NAME}")
        message(STATUS "Adding test subdirectory: ${TEST_DIR_NAME}")
        add_subdirectory(${TEST_DIR_NAME})
    else()
        message(WARNING "SD_BUILD_TESTS is ON, but directory '${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR_NAME}' was not found.")
    endif()
endif()


message(STATUS "--- Configuring Installation and Packaging ---")

if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix")
endif()
message(STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}")

set(INSTALL_LIB_DIR "lib" CACHE STRING "Library installation directory")
set(INSTALL_BIN_DIR "bin" CACHE STRING "Runtime executable/DLL installation directory")
set(INSTALL_INCLUDE_DIR "include/${PROJECT_NAME}" CACHE STRING "Header file installation directory")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION ${INSTALL_INCLUDE_DIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl"
        PATTERN "impl" EXCLUDE
        PATTERN "cpu" EXCLUDE
        PATTERN "cuda" EXCLUDE
        PATTERN "generated" EXCLUDE
        PATTERN "platform" EXCLUDE
)

if(DEFINED OP_OUTPUT_FILE_ABS AND EXISTS "${OP_OUTPUT_FILE_ABS}")
    install(FILES ${OP_OUTPUT_FILE_ABS} DESTINATION ${INSTALL_INCLUDE_DIR})
endif()
install(FILES "${CMAKE_BINARY_DIR}/include/config.h" DESTINATION ${INSTALL_INCLUDE_DIR})
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd/graph_generated.h")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd/graph_generated.h"
            DESTINATION ${INSTALL_INCLUDE_DIR}/graph/generated/sd)
endif()

include(CPack)

set(CPACK_PACKAGE_NAME "libnd4j")
set(CPACK_PACKAGE_VENDOR "Eclipse Foundation")
set(CPACK_PACKAGE_CONTACT "Eclipse Deeplearning4j Committers <dl4j-dev@eclipse.org>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations library for Eclipse Deeplearning4j")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/eclipse/deeplearning4j")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

if(PROJECT_VERSION)
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
else()
    set(CPACK_PACKAGE_VERSION "1.0.0-SNAPSHOT")
endif()
if(${CPACK_PACKAGE_VERSION} MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
    set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
else()
    set(CPACK_PACKAGE_VERSION_MAJOR 0)
    set(CPACK_PACKAGE_VERSION_MINOR 0)
    set(CPACK_PACKAGE_VERSION_PATCH 0)
endif()
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")


if(UNIX)
    find_program(LSB_RELEASE_EXEC lsb_release)
    if(LSB_RELEASE_EXEC)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -sr OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -sc OUTPUT_VARIABLE CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)
    elseif(EXISTS "/etc/os-release")
        file(STRINGS "/etc/os-release" OS_INFO REGEX "^(ID|VERSION_ID|VERSION_CODENAME)=")
        foreach(info ${OS_INFO})
            if(info MATCHES "^ID=(.*)")
                set(DISTRIBUTION ${CMAKE_MATCH_1})
                string(STRIP ${DISTRIBUTION} DISTRIBUTION)
            elseif(info MATCHES "^VERSION_ID=(.*)")
                set(RELEASE ${CMAKE_MATCH_1})
                string(STRIP ${RELEASE} RELEASE)
            elseif(info MATCHES "^VERSION_CODENAME=(.*)")
                set(CODENAME ${CMAKE_MATCH_1})
                string(STRIP ${CODENAME} CODENAME)
            endif()
        endforeach()
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

    set(CPACK_GENERATOR "TGZ")

    if(DISTRIBUTION MATCHES "Ubuntu|Debian")
        set(CPACK_GENERATOR "DEB")
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${ARCHITECTURE}")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
        set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
        set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.17)")
        if(SD_CUDA)
            if(CUDA_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
                set(CUDA_MAJOR ${CMAKE_MATCH_1})
                set(CUDA_MINOR ${CMAKE_MATCH_2})
                list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS " cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}")
            endif()
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS " libcudnn8")
        endif()
        string(REPLACE ";" ", " CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}")


        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb")
            set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.deb")
        endif()

    elseif(DISTRIBUTION MATCHES "CentOS|Fedora|RHEL|Rocky|AlmaLinux")
        set(CPACK_GENERATOR "RPM")
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${RPM_ARCHITECTURE}")
        set(CPACK_RPM_PACKAGE_ARCHITECTURE ${RPM_ARCHITECTURE})
        set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
        set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries")
        set(CPACK_RPM_PACKAGE_URL ${CPACK_PACKAGE_HOMEPAGE_URL})
        set(CPACK_RPM_PACKAGE_VENDOR ${CPACK_PACKAGE_VENDOR})
        set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.17")
        if(SD_CUDA)
            if(CUDA_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
                set(CUDA_MAJOR ${CMAKE_MATCH_1})
                set(CUDA_MINOR ${CMAKE_MATCH_2})
                list(APPEND CPACK_RPM_PACKAGE_REQUIRES " cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}")
            endif()
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_RPM_PACKAGE_REQUIRES " libcudnn8")
        endif()
        string(REPLACE ";" ", " CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}")

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
            set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
            set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
        endif()
    endif()

elseif(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME}")
    set(CPACK_NSIS_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
    set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-macOS")
endif()

message(STATUS "CPack Generator: ${CPACK_GENERATOR}")
message(STATUS "CPack Package File Name: ${CPACK_PACKAGE_FILE_NAME}")

message("--- CMake Configuration Complete ---")