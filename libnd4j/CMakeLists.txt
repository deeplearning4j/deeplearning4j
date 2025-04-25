# Main CMakeLists.txt (Uploaded File Content Starts Here)
cmake_minimum_required(VERSION 3.15)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(libnd4j)



set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")

#ensure we create lib files
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
option(SD_PREPROCESSOR "Use preprocessor" OFF) # Note: 'blas' file uses SD_PREPROCESS
set(CMAKE_VERBOSE_MAKEFILE ON)
# GCC specific flag: 	 -finstrument-functions enables call stack logging.
# Useful for debugging segfaults.
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
set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Hack to disable flatc build" FORCE)
# note we may or may not use
# a build type called "none" to prevent default injection
# of flags from cmake.
# We do this when using functrace so we can add symbols
# to a binary but still run from java without freezing.
# # Normally, we would just want to use debug build.
# Running a debug build
# via JNI seems to just freeze though.
# The goal is to just use tools like
# valgrind or compute-sanitizer or even address sanitizer with symbols
# embedded in a binary but still run code from java.
# message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# print_all_variables() # Uncomment to debug if needed

# Define ARM Compute Library URLs based on architecture
# For ARM 32-bit (Raspberry Pi)
set(ARM_COMPUTE_URL_ARMV7 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-armv7a-cpu-bin.tar.gz")
# For ARM 64-bit (Raspberry Pi, Odroid N2)
set(ARM_COMPUTE_URL_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-aarch64-cpu-bin.tar.gz")
# For ARM 64-bit Android
set(ARM_COMPUTE_URL_ANDROID_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-android-aarch64-cpu-bin.tar.gz")

# Set the appropriate URL based on the architecture
# Set the appropriate URL based on the architecture
if(${SD_ARCH} MATCHES "armv7")
    set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
    message("Using ARM Compute Library for ARMv7 32-bit")

    # Corrected block starts here
elseif(${SD_ARCH} MATCHES "armv8-a") # First condition
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (matched armv8-a)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched armv8-a)")
    endif()
elseif(${SD_ARCH} MATCHES "arm64") # Second condition (separate elseif)
    # Same logic as above
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})

        message("Using ARM Compute Library for Android ARM64 (matched arm64)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched arm64)")
    endif()
    # Original endif for the whole if/elseif chain
endif()

# Find the section related to CUDA configuration around line 167
# Note: CUDA configuration logic is now within the pasted block below

# Helper flags for dependencies (can be set via cmake command line)
option(HELPERS_onednn "Build with oneDNN" OFF)
option(HELPERS_armcompute "Build with ARM Compute Library" OFF)
option(HELPERS_cudnn "Build with cuDNN" OFF)

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

set(SD_X86_BUILD false)
# Determine platform type more accurately
set(SD_X86_BUILD false)
set(SD_ARM_BUILD false) # Add similar flag for ARM if needed

if(SD_ANDROID_BUILD)
    # For Android, trust ANDROID_ABI
    if(ANDROID_ABI MATCHES "x86_64")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86-64") # Explicitly set SD_ARCH too
    elseif(ANDROID_ABI MATCHES "x86")
        set(SD_X86_BUILD true)
        set(SD_ARCH "x86") # Explicitly set SD_ARCH too
    elseif(ANDROID_ABI MATCHES "arm64-v8a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "arm64-v8a") # Explicitly set SD_ARCH too
    elseif(ANDROID_ABI MATCHES "armeabi-v7a")
        set(SD_ARM_BUILD true)
        set(SD_ARCH "armv7-a") # Explicitly set SD_ARCH too
    endif()
elseif(NOT SD_IOS_BUILD)
    # Non-Android, Non-iOS logic (can use CMAKE_SYSTEM_PROCESSOR or keep original checks)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_X86_BUILD true)
        # Potentially set SD_ARCH here too if not passed externally
        if(NOT DEFINED SD_ARCH)
            set(SD_ARCH "x86-64")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
        set(SD_ARM_BUILD true)
        # Potentially set SD_ARCH here too if not passed externally
        # Add other non-Android/iOS platforms (PowerPC, etc.) if needed
    endif()
    # Add iOS logic if necessary
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")


# -fsanitize=address
# -fsanitize=leak
if (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "armv8-a")
        #set(SD_ARCH armv8-a) # Let SD_ARCH be determined earlier
        set(SD_X86_BUILD false)
        set(CMAKE_OSX_ARCHITECTURES "arm64")
    endif()


    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
elseif(WIN32)
    # set(SD_X86_BUILD true) # Already handled above
    # CXX Flags for Windows are handled within the pasted `blas` block logic below
    # if (SD_CUDA)
    #     set(CMAKE_CXX_FLAGS_RELEASE  "-D_RELEASE=true")
    #      set(CMAKE_CXX_FLAGS_DEBUG  " /FS /EHsc")
    # else()
    #     set(CMAKE_CXX_FLAGS_RELEASE  "/O2 /D_RELEASE=true /EHsc")
    #     set(CMAKE_CXX_FLAGS_DEBUG  "/Zi /Od /EHsc")
    # endif()
else() # Assuming Linux/other Unix
    # set this as a separate flag.
    # profiler won't work with anything higher than -O0
    # we also have to do this for release because libnd4j has issues when working with java
    # ideally debug builds would work but since it's not consistent it's better to just
    # set this for release flags so we can debug code even from the java level
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        # note we may or may not use a build type called "none" to prevent default injection
        # of flags from cmake. We do this when using functrace so we can add symbols
        # to a binary but still run from java without freezing.
        # # Normally, we would just want to use debug build.
        # Running a debug build
        # via JNI seems to just freeze though.
        # The goal is to just use tools like
        # valgrind or compute-sanitizer or even address sanitizer with symbols
        # embedded in a binary but still run code from java.
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")
    # note on ftls model: https://github.com/microsoft/mimalloc/issues/147 tsanitize sometimes throws errors
    if (SD_SANITIZE)
        # Default sanitizers if not specified via command line
        if(NOT DEFINED SD_SANITIZERS)
            set(SD_SANITIZERS "address") # Default to address sanitizer
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
        # adds stack size to prevent misc errors with address sanitizer
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


if(NOT SD_CUDA) # BLAS/OpenBLAS setup for CPU only

    # we need this definition to avoid global memory use within onednn
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if ("${OPENBLAS_PATH}" STREQUAL "")
        #we don't want OpenBLAS on Apple
        if (NOT APPLE)
            # note: this is not a typo
            set(BLA_VENDOR "OpenBLAS")
        endif()

        # look around for system blas instead, see: https://cmake.org/cmake/help/latest/module/FindBLAS.html
        find_package(BLAS REQUIRED)
        if (BLAS_FOUND)
            message("Found external BLAS implementation: ${BLAS_LIBRARIES} ")
            add_definitions(-D__EXTERNAL_BLAS__=true)
        endif()
    else()
        # if we have externally provided OPENBLAS_PATH - let's use it
        set(HAVE_OPENBLAS 1)
        message("Setting openblas")
        include_directories(${OPENBLAS_PATH}/include/)
        link_directories(${OPENBLAS_PATH} ${OPENBLAS_PATH}/lib/)
        set(OPENBLAS_LIBRARIES openblas)
    endif()

endif()


#arm-compute entry
# Root CMakeLists.txt adjustment

# Initialize variables to indicate ARM Compute is not yet configured
set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)

# Only attempt ARM Compute configuration if the helper flag is set
if(${HELPERS_armcompute})
    message(STATUS "ARM Compute helper is active, attempting configuration...")

    # First, try to find an existing installation
    find_package(ARMCOMPUTE QUIET) # Use QUIET to avoid messages if not found initially

    if(ARMCOMPUTE_FOUND)
        # Found via find_package
        set(HAVE_ARMCOMPUTE 1)
        include_directories(${ARMCOMPUTE_INCLUDE_DIRS}) # Use standard variable from FindARMCOMPUTE
        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIBRARIES}) # Standard variable from FindARMCOMPUTE
        message(STATUS "Found ARM Compute via find_package. Include: ${ARMCOMPUTE_INCLUDE_DIRS}, Libs: ${ARMCOMPUTE_LIBRARIES}")
        add_definitions(-DHAVE_ARMCOMPUTE=1) # Add definition
        add_definitions(-DARMCOMPUTENEON_ENABLED) # Add definition
    else()
        # Not found via find_package, attempt download if URL is defined and arch matches
        message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
        if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7" OR ${SD_ARCH} MATCHES "armv8-a" OR ${SD_ARCH} MATCHES "arm64"))
            message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")

            # Configure the download project
            configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.armcompute.in armcompute-download/CMakeLists.txt @ONLY) # Corrected Path

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
                    set(ARMCOMPUTE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/armcompute-src") # Set the root directory

                    # Check if the expected library files exist after download/build
                    set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so")
                    set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so")

                    if(EXISTS "${ARMCOMPUTE_LIB_PATH}" AND EXISTS "${ARMCOMPUTE_CORE_LIB_PATH}")
                        add_definitions(-DARMCOMPUTENEON_ENABLED) # Add necessary definition
                        include_directories(${ARMCOMPUTE_ROOT} ${ARMCOMPUTE_ROOT}/include) # Include directories
                        set(ARMCOMPUTE_LIBRARIES ${ARMCOMPUTE_LIB_PATH} ${ARMCOMPUTE_CORE_LIB_PATH}) # Set the libraries variable
                        set(HAVE_ARMCOMPUTE 1) # Mark ARM Compute as available
                        add_definitions(-DHAVE_ARMCOMPUTE=1) # Add definition
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

# --- The rest of your root CMakeLists.txt continues below ---
# Make sure the 'else' block that previously defined ARMCOMPUTE_LIBRARIES incorrectly is removed.
if (${HELPERS_onednn})
    message("Going to pull & build onednn")
    set(HAVE_ONEDNN 1)
    add_definitions(-DHAVE_ONEDNN=1) # Add definition
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "Hack to enforce static mode" FORCE)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.onednn.in onednn-download/CMakeLists.txt @ONLY) # Corrected Path
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" . # Corrected Generator argument
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
    set(ONEDNN_LIBRARIES dnnl) # Changed variable name for consistency
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

    # Enhanced search paths for cuDNN
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
        add_definitions(-DHAVE_CUDNN=1) # Add definition
        set(CUDNN ${CUDNN_LIBRARY})
    else()
        # Allow build to continue even without cuDNN
        message(WARNING "cuDNN not found. Build will continue without cuDNN support.")
        set(HAVE_CUDNN false)
    endif()
endif()

# Before the flatbuffers download section
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    set(FLATBUFFERS_BUILD_FLATC "ON" CACHE STRING "Enable flatc build" FORCE)
else()
    set(FLATBUFFERS_BUILD_FLATC "OFF" CACHE STRING "Disable flatc build" FORCE)
endif()

# Existing flatbuffers download section
if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-download-complete.marker")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.in flatbuffers-download/CMakeLists.txt @ONLY) # Corrected Path
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

# After flatbuffers is added, handle generation if enabled
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

configure_file(include/config.h.in include/config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ===========================================================================
# Start of Content Pasted Verbatim from blas/CMakeLists.txt
# Path adjustments made below. Formatting Preserved.
# Redundant cmake_minimum_required, project removed.
# ===========================================================================
# cmake_minimum_required(VERSION 3.15) # Removed Redundant
# project(libnd4j) # Removed Redundant

set(CMAKE_VERBOSE_MAKEFILE OFF) # Note: Main file has ON
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Note: Main file also has ON

include(CheckCXXCompilerFlag)

# Include Directories Based on OS
if(UNIX)
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(APPLE)
    message("Using Apple") # Note: Main file also has this check
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
    get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES) # Already done in Main? Keep for now.
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()

    message(STATUS "Include Directories (from blas section):") # Added note
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
if(SD_USE_LTO) # Note: Option defined in Main
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
if(DEFINED OP_OUTPUT_FILE AND OP_OUTPUT_FILE MATCHES ".h$") # Check if variable is defined
    message("Definitions will be written to \"${OP_OUTPUT_FILE}\"")
    file(WRITE "${OP_OUTPUT_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
endif()

# Architecture Tuning
if(SD_ARCH MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
elseif(CMAKE_SYSTEM_NAME MATCHES "Aurora")
    set_source_files_properties(include/graph/impl/GraphHolder.cpp PROPERTIES COMPILE_FLAGS -g0) # Path Adjusted
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE} ${INFORMATIVE_FLAGS} -std=c++11") # Note: Main file uses C++17
    if(UNIX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,-z,--no-undefined,--verbose")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,$ORIGIN/,--no-undefined,--verbose")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT APPLE AND NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wl,-export-dynamic,--verbose")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -export-dynamic,--verbose")
    endif()

    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON") # Note: Option defined in Main
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

# External Include Directories for Linux
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include") # Note: Main also sets this variable
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
            include/ops/declarable/platform/mkldnn/*.cpp             # Path Adjusted
            include/ops/declarable/platform/mkldnn/mkldnnUtils.h    # Path Adjusted
    )
endif()

if(HAVE_ARMCOMPUTE)
    file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES
            include/ops/declarable/platform/armcompute/*.cpp        # Path Adjusted
            include/ops/declarable/platform/armcompute/*.h          # Path Adjusted
    )
endif()

# CUDA Configuration
if(SD_CUDA)
    message("Building with CUDA support (blas section)") # Added note

    add_definitions(-D__CUDABLAS__=true)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # set(CMAKE_CXX_FLAGS "") # Redundant setting from main CUDA block
    endif()

    # find_package(CUDA REQUIRED) # Already done in main block
    if(CUDA_FOUND)
        message("CUDA include directory: ${CUDA_INCLUDE_DIRS} with CXX compiler ${CMAKE_CXX_COMPILER_ID} SD_GCC_FUNCTRACE=${SD_GCC_FUNCTRACE} (blas section)")
        # include_directories(${CUDA_INCLUDE_DIRS}) # Already done in main block
        message("CUDA found! (blas section)")

        set(CMAKE_CUDA_FLAGS_DEBUG " -g") # Note: Also set in main block
        message("CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID} (blas section)")

        # Common CUDA flags for all platforms - add -allow-unsupported-compiler flag
        set(COMMON_CUDA_FLAGS "-allow-unsupported-compiler") # Note: Also set in main block
        message("Adding -allow-unsupported-compiler flag for CUDA compilation (blas section)")

        # Add platform-specific compiler flags
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(SD_GCC_FUNCTRACE STREQUAL "ON")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Werror -Wall   -Wno-return-type  -Wno-unknown-pragmas  -Wno-unused-variable -Wno-unused-parameter  -Wreturn-type -W -ggdb3 -fPIC -DSD_GCC_FUNCTRACE=1 -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -finstrument-functions  -O0") # Note: Similar flags in main block
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G") # Note: Similar flags in main block
                add_compile_definitions(SD_GCC_FUNCTRACE) # Note: Already done if main block runs
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS} -Xcompiler=-fPIC") # Note: Similar flags in main block
            endif()
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # Add Windows-specific flags, but keep -allow-unsupported-compiler
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS}") # Note: Also set in main block
            message("Using allow-unsupported-compiler flag for Windows CUDA build (blas section)")
        else()
            # For other compilers, ensure we still have the allow-unsupported-compiler flag
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS}") # Note: Also set in main block
        endif()

        if(WIN32)
            message("Configuring CUDA libraries for Windows (blas section)")
            if(NOT DEFINED CUDA_cublas_LIBRARY)
                set(CUDA_cublas_LIBRARY "${CUDA_HOME}/lib/x64/cublas.lib") # Note: Also set in main block
            endif()
            if(NOT DEFINED CUDA_cusolver_LIBRARY)
                set(CUDA_cusolver_LIBRARY "${CUDA_HOME}/lib/x64/cusolver.lib") # Note: Also set in main block
            endif()
        endif()

        # Rest of the CUDA configuration (Jetson Nano, etc.)
        if("${SD_ARCH}" MATCHES "armv8-a" AND UNIX)
            message("Adding Jetson Nano specific settings (blas section)")
            # Need to manually specify stubbed links in order
            # for cublas and cusolver to be resolved
            if(NOT DEFINED CUDA_cublas_LIBRARY OR "${CUDA_cublas_LIBRARY}" MATCHES ".*NOTFOUND.*" )
                message("Setting cublas library manually (blas section)")
                set(CUDA_cublas_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcublas.so" CACHE STRING "CUDA CUBLAS LIB" FORCE) # Note: Also set in main block
                # Unset NOTFOUND variables
                unset(CUDA_cublas-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cublas_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            if(NOT DEFINED CUDA_cusolver_LIBRARY OR CUDA_cusolver_LIBRARY MATCHES ".*NOTFOUND.*")
                message("Setting cusolver library manually for Jetson Nano (blas section)")
                set(CUDA_cusolver_LIBRARY "$ENV{loc_DIR}/cuda/targets/aarch64-linux/lib/stubs/libcusolver.so" CACHE STRING "CUDA CUSOLVER LIB" FORCE) # Note: Also set in main block
                # Unset NOTFOUND variables
                unset(CUDA_cusolver-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND CACHE)
                unset(CUDA_cusolver_LIBRARY-NOTFOUND PARENT_SCOPE)
            endif()

            message("Jetson Nano cublas library: ${CUDA_cublas_LIBRARY}, cusolver library: ${CUDA_cusolver_LIBRARY} (blas section)")
        endif()

        # Cap the number of registers to prevent resource exhaustion
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=128") # Note: Also set in main block

        # Define CUDA Architectures
        string(TOLOWER "${COMPUTE}" COMPUTE_CMP) # Note: COMPUTE var needs definition (e.g. -DCOMPUTE=...)
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
            # Handling for list like "3.7 5.0..." from user command example
            string(REPLACE " " ";" COMPUTE_LIST ${COMPUTE_CMP})
            set(CUDA_ARCH_FLAGS "")
            set(CMAKE_CUDA_ARCHITECTURES "")
            foreach(ARCH ${COMPUTE_LIST})
                string(REPLACE "." "" ARCH_CLEAN ${ARCH}) # 7.5 -> 75
                list(APPEND CUDA_ARCH_FLAGS "-gencode" "arch=compute_${ARCH_CLEAN},code=sm_${ARCH_CLEAN}")
                list(APPEND CMAKE_CUDA_ARCHITECTURES ${ARCH_CLEAN})
            endforeach()
            # CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS "${COMPUTE}") # Original logic
        endif()

        string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}") # Adjusted variable name
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            if(NOT "${CUDA_ARCH_FLAGS_STR}" STREQUAL "") # Use string version
                set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures" FORCE) # Use list version here? Be consistent. Use list: CMAKE_CUDA_ARCHITECTURES
            else()
                if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
                    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 CACHE STRING "CUDA architectures for CUDA 11.x compilers" FORCE)
                else()
                    set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86 CACHE STRING "CUDA architectures for nvcc 12 and up" FORCE)
                endif()
                # add_compile_definitions(CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}) # This seems incorrect
            endif()
        endif()

        # Add all the CUDA flags together, ensuring -allow-unsupported-compiler is included
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcuda -lcudart -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR} -w --cudart=shared --expt-extended-lambda -Xfatbin -compress-all ${CUDA_ARCH_FLAGS_STR}") # Use string version

        # Add a message confirming unsupported compiler flag is enabled
        message("Final CUDA flags: ${CMAKE_CUDA_FLAGS} (blas section)")


        # Helper Functions defined above in Main section

        # Generate CUDA template instantiations
        # Create output directory for CUDA generated files
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/cuda_generated")

        # Initialize the list of CUDA generated sources
        set(CUDA_GENERATED_SOURCES "")


        if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST_COUNT GREATER 0)
            # If selective types are enabled, count them
            set(NUM_TYPES ${SD_TYPES_LIST_COUNT})
            message(STATUS "Using ${NUM_TYPES} types from SD_TYPES_LIST")
        else()
            # Default number of types if not using selective types
            set(NUM_TYPES 10)
            message(STATUS "Using default number of types: ${NUM_TYPES}")
        endif()
        include(TypeMST.cmake) # Corrected Path

        # 2. Process the compilation_units templates
        file(GLOB CUDA_COMPILATION_UNITS include/loops/cuda/compilation_units/*.cu.in) # Path Adjusted
        foreach(FL_ITEM ${CUDA_COMPILATION_UNITS})
            genCompilation(${FL_ITEM} CUDA_GENERATED_SOURCES) # Use helper function
        endforeach()

        # Decide whether to use all combinations or optimized MST combinations
        set(SD_USE_MST_TYPES ON)
        # Decide whether to use all combinations or optimized MST combinations
        if(SD_USE_MST_TYPES)
            # Generate optimized type combinations
            generate_mst_combinations()
            message(STATUS "Using optimized type combinations instead of all possible combinations")
        else()
            # Use all possible combinations (original behavior)
            # set(COMBINATIONS_3 ${ALL_TYPE_COMBINATIONS}) # Needs definition if used
            message(STATUS "Using all possible type combinations")
        endif()


        # Get all CUDA template files from comb_compilation_units directory
        file(GLOB CUDA_TEMPLATE_FILES
                "include/loops/cuda/comb_compilation_units/*.cu.in" # Path Adjusted
        )

        # For each template file, generate individual function files per type combination
        foreach(TEMPLATE_FILE ${CUDA_TEMPLATE_FILES})
            foreach(COMBINATION ${COMBINATIONS_3})
                genSingleFunctionCuda(${TEMPLATE_FILE} ${COMBINATION} "${CMAKE_BINARY_DIR}/cuda_generated") # Use helper function
            endforeach()
        endforeach()


        # Add all generated CUDA files to the sources
        # list(APPEND ALL_SOURCES ${CUDA_GENERATED_SOURCES}) # ALL_SOURCES is gathered below


        # Gather CUDA Source Files
        file(GLOB_RECURSE PERF_SOURCES include/performance/*.cpp include/performance/*.h) # Path Adjusted
        file(GLOB_RECURSE EXCEPTIONS_SOURCES include/exceptions/*.cpp include/exceptions/*.h) # Path Adjusted
        file(GLOB_RECURSE EXEC_SOURCES include/execution/impl/*.cpp include/execution/cuda/*.cu include/execution/cuda/*.h include/execution/*.cu include/execution/*.h) # Path Adjusted
        file(GLOB_RECURSE TYPES_SOURCES include/types/*.cpp include/types/*.h) # Path Adjusted
        file(GLOB_RECURSE ARRAY_SOURCES include/array/cuda/*.cu include/array/cuda/*.chpp include/array/impl/*.cpp include/array/*.h) # Path Adjusted (Combined array sources)
        file(GLOB_RECURSE MEMORY_SOURCES include/memory/impl/*.cpp include/memory/cuda/*.cu include/memory/*.h) # Path Adjusted
        file(GLOB_RECURSE GRAPH_SOURCES include/graph/*.cpp include/graph/*.cu include/graph/*.h) # Path Adjusted
        file(GLOB_RECURSE CUSTOMOPS_SOURCES include/ops/declarable/generic/*.cpp) # Path Adjusted
        file(GLOB_RECURSE CUSTOMOPS_HELPERS_SOURCES include/ops/declarable/helpers/cuda/*.cu include/ops/declarable/helpers/impl/*.cpp) # Path Adjusted
        file(GLOB_RECURSE OPS_SOURCES include/ops/impl/*.cpp include/ops/declarable/impl/*.cpp include/ops/*.h) # Path Adjusted
        file(GLOB_RECURSE HELPERS_SOURCES
                include/build_info.cpp # Path Adjusted
                include/ConstMessages.cpp # Path Adjusted
                include/helpers/*.cpp # Path Adjusted
                include/helpers/cuda/*.cu # Path Adjusted
                include/helpers/*.h) # Path Adjusted
        file(GLOB CPU_HELPERS_TO_EXCLUDE
                include/helpers/cpu/*.cpp) # Path Adjusted
        # remove helpers/cpu
        list(REMOVE_ITEM HELPERS_SOURCES ${CPU_HELPERS_TO_EXCLUDE})
        file(GLOB_RECURSE INDEXING_SOURCES include/indexing/*.cpp include/indexing/*.h) # Path Adjusted
        file(GLOB_RECURSE LOOPS_SOURCES include/loops/impl/*.cpp include/loops/*.h) # Path Adjusted
        file(GLOB_RECURSE LEGACY_SOURCES include/legacy/impl/*.cpp include/legacy/*.cu include/legacy/*.h) # Path Adjusted
        file(GLOB_RECURSE LOOPS_SOURCES_CUDA  include/loops/*.cu include/loops/cuda/**/*.cu) # Path Adjusted
        file(GLOB_RECURSE COMPILATION_UNITS include/loops/cuda/compilation_units/*.cu.in include/ops/impl/compilation_units/*.cpp.in) # Path Adjusted
        file(GLOB_RECURSE COMPILATION_UNITS include/loops/cuda/compilation_units/*.cu.in  include/ops/impl/compilation_units/*.cpp.in) # Path Adjusted (Duplicate?)

        # Generate Compilation Units (This seems redundant with the earlier genCompilation loop)
        # foreach(FL_ITEM ${COMPILATION_UNITS})
        #     genCompilation(${FL_ITEM})
        # endforeach()
        set(ALL_SOURCES
                ${CUDA_GENERATED_SOURCES} # Use list populated by genCompilation/genSingleFunctionCuda
                ${PERF_SOURCES}
                ${EXCEPTIONS_SOURCES}
                ${EXEC_SOURCES}
                ${TYPES_SOURCES}
                ${ARRAY_SOURCES}
                ${MEMORY_SOURCES}
                ${GRAPH_SOURCES}
                ${CUSTOMOPS_SOURCES}
                ${CUSTOMOPS_ONEDNN_SOURCES} # From GLOB above
                ${CUSTOMOPS_ARMCOMPUTE_SOURCES} # From GLOB above
                # ${CUSTOMOPS_GENERIC_SOURCES} # This was for CPU, use CUDA sources
                ${CUSTOMOPS_HELPERS_SOURCES} # Helpers CUDA/Impl
                ${OPS_SOURCES}
                ${HELPERS_SOURCES}
                ${INDEXING_SOURCES}
                ${LOOPS_SOURCES}
                ${LEGACY_SOURCES}
                ${LOOPS_SOURCES_CUDA}
        )


        # Include cuDNN Sources if Available
        if(HAVE_CUDNN)
            message("cuDNN included (blas section)")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES include/ops/declarable/platform/cudnn/*.cu) # Path Adjusted
            list(APPEND ALL_SOURCES ${CUSTOMOPS_CUDNN_SOURCES}) # Append to ALL_SOURCES
        endif()

        # Create Object Library
        # Note: HELPERS_CPP is not defined, removed from list
        add_library(samediff_obj OBJECT ${ALL_SOURCES}) # Simplified object library creation
        # The if/else for SD_GCC_FUNCTRACE seemed redundant as ALL_SOURCES is the same

        target_include_directories(samediff_obj PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include # Source include
                ${CMAKE_CURRENT_BINARY_DIR}/include # Generated headers include
                ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include # Flatbuffers include
                ${CUDA_INCLUDE_DIRS} # CUDA includes
                ${EXTERNAL_INCLUDE_DIRS} # OS Specific includes
                # Conditional includes
                $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
                $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_ROOT}/include> # ARMCompute has multiple include dirs sometimes
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN_INCLUDE_DIR}>
        )

        # Handle MSVC Runtime Library
        set(MSVC_RT_LIB "MultiThreadedDLL") # Note: Also set in Main section
        if(WIN32) # Check OS, not compiler ID here
            set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()

        if(WIN32)
            message("Enabling /EHsc for CUDA on Windows (blas section)")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /bigobj /std:c++14") # Note: Also set in Main section
        endif()

        # Create the shared library using SD_LIBRARY_NAME
        if(NOT SD_STATIC_LIB OR SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        endif()

        # Create the static library if needed
        if(SD_STATIC_LIB AND SD_SHARED_LIB)
            add_library(${SD_LIBRARY_NAME}static STATIC $<TARGET_OBJECTS:samediff_obj>)
            if(WIN32) # Check OS, not compiler ID here
                set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            endif()
            install(TARGETS ${SD_LIBRARY_NAME}static ARCHIVE DESTINATION lib) # Adjusted install
        elseif(SD_STATIC_LIB)
            add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
            if(WIN32) # Check OS, not compiler ID here
                set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            endif()
            install(TARGETS ${SD_LIBRARY_NAME} ARCHIVE DESTINATION lib) # Adjusted install
        endif()

        # Set MSVC runtime library for the shared library if it exists
        if(TARGET ${SD_LIBRARY_NAME} AND WIN32)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()

        # Link CUDA Libraries
        if(TARGET ${SD_LIBRARY_NAME}) # Only link if shared library target exists
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                    ${CUDA_LIBRARIES}
                    ${CUDA_DRIVER_LIBRARY} # From main section find_library
                    ${CUDA_CUBLAS_LIBRARIES} # From find_package or manual set
                    ${CUDA_cusolver_LIBRARY} # From find_package or manual set
                    $<$<BOOL:${HAVE_CUDNN}>:${CUDNN}> # Link CUDNN if found
                    # Link other potential dependencies like oneDNN, ARMCompute if needed by CUDA code
                    $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_LIBRARIES}>
                    $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_LIBRARIES}>
                    Threads::Threads
                    dl
            )
            if(OPENMP_FOUND)
                target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
            endif()
        endif()

        # Install CUDA Library
        if(TARGET ${SD_LIBRARY_NAME})
            install(TARGETS ${SD_LIBRARY_NAME} LIBRARY DESTINATION lib RUNTIME DESTINATION bin) # Adjusted install
        endif()

        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib") # Path Adjusted
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin") # Added for consistency
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib") # Added for consistency

    else()
        # This else corresponds to if(CUDA_FOUND)
        message(WARNING "CUDA was enabled but not found by find_package. Skipping CUDA build steps in blas section.")
    endif()

    # CPU Configuration
elseif(SD_CPU)
    message("Building with CPU support (blas section)") # Added note

    # Initialize source lists specific to this block
    set(ALL_SOURCES "")
    set(CPU_GENERATED_SOURCES "") # List for genCompilation output
    set(CUSTOMOPS_GENERIC_SOURCES "") # List for genPartitionCombination output
    set(CUSTOMOPS_ONEDNN_SOURCES "") # Reset for this scope
    set(CUSTOMOPS_ARMCOMPUTE_SOURCES "") # Reset for this scope

    # Generate CPU specific compilation units (.cpp.in -> .cpp)
    file(GLOB CPU_COMPILATION_UNITS_IN
            include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in  # Path Adjusted
            include/loops/cpu/compilation_units/*.cpp.in                    # Path Adjusted
            include/helpers/cpu/loops/*.cpp.in                              # Path Adjusted
            include/ops/impl/compilation_units/*.cpp.in                     # Path Adjusted
    )
    foreach(FL_ITEM ${CPU_COMPILATION_UNITS_IN})
        genCompilation(${FL_ITEM} CPU_GENERATED_SOURCES) # Populate CPU_GENERATED_SOURCES
    endforeach()

    # Generate CPU combination instantiations (using genPartitionCombination from main file)
    set(COMBINATIONS_3 "0,0,0;0,0,2;0,1,2;0,2,0;0,2,1;0,2,2;1,0,0;1,1,1;1,1,2;1,2,0;1,2,1;1,2,2;2,0,0;2,0,1;2,0,2;2,1,0;2,1,1;2,1,2;2,2,0;2,2,1;2,2,2") # Semicolon separated list
    set(COMBINATIONS_2 "0,0;0,1;1,0;1,1;0,2;2,0;1,2;2,1;2,2") # Semicolon separated list

    # Define template files (adjust paths as needed)
    set(INSTANTIATION_TEMPLATES_3 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in") # Adjusted path
    set(INSTANTIATION_TEMPLATES_2 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in") # Adjusted path

    set(CPU_COMB_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated_combinations")
    file(MAKE_DIRECTORY "${CPU_COMB_GENERATED_DIR}")

    # Generate files for combinations_3
    if(EXISTS ${INSTANTIATION_TEMPLATES_3})
        foreach(COMB ${COMBINATIONS_3})
            genPartitionCombination(${INSTANTIATION_TEMPLATES_3} 3 ${COMB} "${CPU_COMB_GENERATED_DIR}") # Populates CUSTOMOPS_GENERIC_SOURCES
        endforeach()
    else()
        message(WARNING "CPU Combination Template 3 not found: ${INSTANTIATION_TEMPLATES_3}")
    endif()

    # Generate files for combinations_2
    if(EXISTS ${INSTANTIATION_TEMPLATES_2})
        foreach(COMB ${COMBINATIONS_2})
            genPartitionCombination(${INSTANTIATION_TEMPLATES_2} 2 ${COMB} "${CPU_COMB_GENERATED_DIR}") # Populates CUSTOMOPS_GENERIC_SOURCES
        endforeach()
    else()
        message(WARNING "CPU Combination Template 2 not found: ${INSTANTIATION_TEMPLATES_2}")
    endif()

    # Gather CPU Source Files
    file(GLOB_RECURSE PERF_SOURCES include/performance/*.cpp include/performance/*.h) # Path Adjusted
    file(GLOB_RECURSE EXCEPTIONS_SOURCES include/exceptions/*.cpp include/exceptions/*.h) # Path Adjusted
    file(GLOB_RECURSE EXEC_SOURCES include/execution/*.cpp include/execution/*.h) # Path Adjusted
    file(GLOB_RECURSE TYPES_SOURCES include/types/*.cpp include/types/*.h) # Path Adjusted
    file(GLOB_RECURSE ARRAY_SOURCES include/array/*.cpp include/array/*.h) # Path Adjusted
    file(GLOB_RECURSE MEMORY_SOURCES include/memory/*.cpp include/memory/*.h) # Path Adjusted
    file(GLOB_RECURSE GRAPH_SOURCES include/graph/*.cpp include/graph/*.h) # Path Adjusted
    file(GLOB_RECURSE CUSTOMOPS_SOURCES include/ops/declarable/generic/*.cpp) # Path Adjusted
    # Note: CUSTOMOPS_GENERIC_SOURCES is populated by genPartitionCombination calls above
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU include/ops/declarable/helpers/cpu/*.cpp) # Path Adjusted (Used instead of CUSTOMOPS_GENERIC_SOURCES from blas?) - Keep both for now.
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL include/ops/declarable/helpers/impl/*.cpp) # Path Adjusted
    file(GLOB_RECURSE OPS_SOURCES include/ops/impl/*.cpp include/ops/declarable/impl/*.cpp include/ops/*.h) # Path Adjusted
    file(GLOB_RECURSE INDEXING_SOURCES include/indexing/*.cpp include/indexing/*.h) # Path Adjusted
    file(GLOB_RECURSE HELPERS_SOURCES include/build_info.cpp include/ConstMessages.cpp include/helpers/*.cpp include/helpers/*.h) # Path Adjusted
    file(GLOB_RECURSE LEGACY_SOURCES include/legacy/impl/*.cpp include/legacy/cpu/*.cpp include/legacy/*.h) # Path Adjusted
    file(GLOB_RECURSE LOOPS_SOURCES include/loops/*.cpp include/loops/*.h) # Path Adjusted
    # file(GLOB_RECURSE COMPILATION_UNITS ...) # Redundant with CPU_COMPILATION_UNITS_IN above

    # Platform specific sources (ONEDNN, ARMCompute)
    if(HAVE_ONEDNN)
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES include/ops/declarable/platform/mkldnn/*.cpp include/ops/declarable/platform/mkldnn/mkldnnUtils.h) # Path Adjusted
    endif()
    if(HAVE_ARMCOMPUTE)
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES include/ops/declarable/platform/armcompute/*.cpp include/ops/declarable/platform/armcompute/*.h) # Path Adjusted
    endif()


    set(ALL_SOURCES
            ${CPU_GENERATED_SOURCES} # From genCompilation
            ${CUSTOMOPS_GENERIC_SOURCES} # From genPartitionCombination
            ${PERF_SOURCES}
            ${EXCEPTIONS_SOURCES}
            ${EXEC_SOURCES}
            ${TYPES_SOURCES}
            ${ARRAY_SOURCES}
            ${MEMORY_SOURCES}
            ${GRAPH_SOURCES}
            ${CUSTOMOPS_SOURCES}
            ${CUSTOMOPS_ONEDNN_SOURCES} # From GLOB
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES} # From GLOB
            ${CUSTOMOPS_HELPERS_CPU}   # CPU Specific Helpers
            ${CUSTOMOPS_HELPERS_IMPL}  # Impl Helpers
            ${OPS_SOURCES}
            ${HELPERS_SOURCES}
            ${INDEXING_SOURCES}
            ${LOOPS_SOURCES}
            ${LEGACY_SOURCES}
    )

    # Disable platform optimizations for specific files if needed (Path Adjusted)
    if (SD_X86_BUILD AND NOT WIN32)
        # Check if files exist before setting properties
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp") # Adjusted path
            set_source_files_properties(include/legacy/cpu/NativeOps.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp") # Adjusted path
            set_source_files_properties(include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
    endif()

    message("CPU BLAS (blas section)") # Added note
    add_definitions(-D__CPUBLAS__=true) # Note: Already done in main CPU block? Keep for now.

    # Remove Excluded Files if Not All Operations are Included
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
                        set(keep_file FALSE) # Remove if marker exists but no matching OP found
                        message(STATUS "Excluding file (no matching NOT_EXCLUDED OP): ${SRC_FILE}")
                    endif()
                endif()
            else()
                # message(WARNING "Source file listed but not found during exclusion check: ${SRC_FILE}")
                set(keep_file FALSE) # Don't keep non-existent files
            endif()

            if(keep_file)
                list(APPEND FILTERED_SOURCES ${SRC_FILE})
            endif()
        endforeach()
        set(ALL_SOURCES ${FILTERED_SOURCES}) # Replace with filtered list

        # Original blas exclusion logic (seems less safe than above)
        # foreach(category IN ITEMS PERF_SOURCES EXCEPTIONS_SOURCES EXEC_SOURCES TYPES_SOURCES ARRAY_SOURCES MEMORY_SOURCES GRAPH_SOURCES CUSTOMOPS_SOURCES OPS_SOURCES HELPERS_SOURCES INDEXING_SOURCES LOOPS_SOURCES LEGACY_SOURCES LOOPS_SOURCES_CUDA HELPERS_CPP)
        #     foreach(SRC_FILE ${${category}})
        #         message("Checking exclusion for: ${SRC_FILE}")
        #         removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "${category}") # This modifies lists being iterated over potentially
        #     endforeach()
        # endforeach()
    endif()

    # Create Object Library
    # CALLSTACK_SRC is not defined, removed.
    add_library(samediff_obj OBJECT ${ALL_SOURCES}) # Use filtered list if necessary

    target_include_directories(samediff_obj PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include # Source include
            ${CMAKE_CURRENT_BINARY_DIR}/include # Generated headers include
            ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include # Flatbuffers include
            ${EXTERNAL_INCLUDE_DIRS} # OS Specific includes
            # Conditional includes
            $<$<BOOL:${HAVE_OPENBLAS}>:${OPENBLAS_PATH}/include>
            $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_ROOT}/include> # ARMCompute has multiple include dirs sometimes
    )

    # Add External Dependencies (Placeholder from blas - needs definition)
    # foreach(external_dependency IN LISTS EXTERNAL_DEPENDENCY_PROJECTS)
    #     message("@external ${external_dependency}")
    #     add_dependencies(samediff_obj ${external_dependency})
    # endforeach()


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
        if(WIN32)
            set_property(TARGET ${SD_LIBRARY_NAME}static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        install(TARGETS ${SD_LIBRARY_NAME}static ARCHIVE DESTINATION lib) # Adjusted install
    elseif(SD_STATIC_LIB)
        add_library(${SD_LIBRARY_NAME} STATIC $<TARGET_OBJECTS:samediff_obj>)
        if(WIN32)
            set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        install(TARGETS ${SD_LIBRARY_NAME} ARCHIVE DESTINATION lib) # Adjusted install
    endif()

    # Handle MSVC Runtime Library Consistently
    if(TARGET samediff_obj AND WIN32)
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    endif()
    if(TARGET ${SD_LIBRARY_NAME} AND WIN32)
        set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
    endif()


    # Link Libraries
    if(TARGET ${SD_LIBRARY_NAME}) # Only link if shared lib exists
        if(NOT "$ENV{CLION_IDE}")
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
                    # ${EXTERNAL_DEPENDENCY_LIBS} # Needs definition
                    $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_LIBRARIES}>
                    $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_LIBRARIES}>
                    ${OPENBLAS_LIBRARIES} # From main section
                    ${BLAS_LIBRARIES} # From main section
                    # ${CPU_FEATURES} # Needs definition/find_package
                    Threads::Threads
                    dl
            )
            if(OPENMP_FOUND)
                target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
            endif()
            # target_include_directories(${SD_LIBRARY_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS}) # Included via samediff_obj
        endif()
    endif()

    # Compiler Checks
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    # OpenMP for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP) # Already found in main section? Re-finding is ok.
        if(OPENMP_FOUND)
            # Flags already added in main section
            # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()

    # Set Library Output Directories
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib") # Path Adjusted
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin") # Added for consistency
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib") # Added for consistency

    # Install the Library
    if(TARGET ${SD_LIBRARY_NAME})
        install(TARGETS ${SD_LIBRARY_NAME} LIBRARY DESTINATION lib RUNTIME DESTINATION bin) # Adjusted install
    endif()

endif() # End SD_CUDA / SD_CPU block

# Preprocessing Configuration
option(SD_PREPROCESS "Run C++ Preprocessor" OFF) # Use the blas option name SD_PREPROCESS
if(SD_PREPROCESS STREQUAL "ON")
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}") # Output dir is relative to build
    # include_directories(${CMAKE_BINARY_DIR}/../../include) # This seems incorrect, includes handled by target

    # Ensure ALL_SOURCES is populated from the correct build path (CUDA or CPU)
    if(NOT ALL_SOURCES)
        message(FATAL_ERROR "SD_PREPROCESS=ON but ALL_SOURCES list is empty. Ensure CUDA or CPU build ran first.")
    endif()

    # Remove Duplicate Entries
    list(REMOVE_DUPLICATES ALL_SOURCES)

    # Define Output Directory
    set(PREPROCESSED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/preprocessed") # Output relative to source dir
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})

    # Initialize Lists
    set(PREPROCESSED_FILES "")
    set(PROCESSED_SOURCES "")

    # Remove Duplicates from ALL_SOURCES again? Seems redundant.
    # list(REMOVE_DUPLICATES ALL_SOURCES)
    # include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # Path adjusted

    # Get properties from the object library target
    get_target_property(PREPROC_INCLUDES samediff_obj INCLUDE_DIRECTORIES)
    get_target_property(PREPROC_DEFINES samediff_obj COMPILE_DEFINITIONS)
    get_target_property(PREPROC_OPTIONS samediff_obj COMPILE_OPTIONS)

    set(PREPROC_INCLUDE_STR "")
    if(PREPROC_INCLUDES)
        foreach(dir ${PREPROC_INCLUDES})
            # Avoid adding empty or non-existent directories
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

    # Determine compiler
    if(SD_CUDA AND CMAKE_CUDA_COMPILER)
        set(compiler "${CMAKE_CUDA_COMPILER}")
        set(base_compile_flags "${CMAKE_CUDA_FLAGS}") # Base CUDA flags
    elseif(CMAKE_CXX_COMPILER)
        set(compiler "${CMAKE_CXX_COMPILER}")
        set(base_compile_flags "${CMAKE_CXX_FLAGS}") # Base CXX flags
    else()
        message(FATAL_ERROR "Cannot determine compiler for preprocessing.")
    endif()

    # Create Custom Commands for Each Source File
    foreach(src IN LISTS ALL_SOURCES)
        # Check if source file exists and is not already processed
        if(EXISTS "${src}" AND NOT src IN_LIST PROCESSED_SOURCES)
            # Only preprocess source files, not headers
            if(src MATCHES "\\.(c|cpp|cxx|cc|cu)$")
                get_filename_component(src_name ${src} NAME_WE)
                get_filename_component(src_dir ${src} DIRECTORY)

                # Create a unique name based on path relative to source OR build dir
                string(REPLACE "${CMAKE_SOURCE_DIR}/" "" rel_path ${src}) # Relative path from source root
                string(REPLACE "${CMAKE_BINARY_DIR}/" "BUILD_DIR/" rel_path ${rel_path}) # Handle generated files
                string(REPLACE "/" "_" flat_rel_path ${rel_path}) # Flatten path
                string(REPLACE "." "_" flat_rel_path ${flat_rel_path}) # Avoid dots in middle

                set(preprocessed_file "${PREPROCESSED_DIR}/${flat_rel_path}.i")
                message(STATUS "Processing ${src} to ${preprocessed_file}")

                if(NOT EXISTS "${preprocessed_file}")
                    # Construct command list carefully
                    set(pp_command_list ${compiler} -E)
                    # Add base flags (need careful splitting if flags contain spaces)
                    # list(APPEND pp_command_list ${base_compile_flags}) # This might break if flags have spaces
                    # Add target-specific options/includes/defines
                    list(APPEND pp_command_list ${PREPROC_OPTIONS_STR})
                    list(APPEND pp_command_list ${PREPROC_INCLUDE_STR})
                    list(APPEND pp_command_list ${PREPROC_DEFINE_STR})
                    list(APPEND pp_command_list "${src}") # Source file
                    list(APPEND pp_command_list "-o" "${preprocessed_file}") # Output file

                    execute_process(
                            COMMAND ${pp_command_list}
                            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" # Consistent working dir
                            RESULT_VARIABLE result
                            OUTPUT_VARIABLE stdout_output
                            ERROR_VARIABLE stderr_output
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_STRIP_TRAILING_WHITESPACE
                    )

                    # Log output/errors if needed
                    set(output_log_file "${preprocessed_file}.log")
                    file(WRITE ${output_log_file} "COMMAND: ${pp_command_list}\n\nResult: ${result}\n\nStdout:\n${stdout_output}\n\nStderr:\n${stderr_output}\n")

                    if(result EQUAL 0)
                        list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                    else()
                        message(WARNING "Preprocessing failed for ${src}. See ${output_log_file}")
                    endif()
                else()
                    message(STATUS "Preprocessed file already exists, skipping: ${preprocessed_file}")
                    list(APPEND PREPROCESSED_FILES ${preprocessed_file}) # Add existing file to list
                endif()

                list(APPEND PROCESSED_SOURCES ${src})
            endif() # End check if source file
        endif() # End check if exists and not processed
    endforeach()

    set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

    # Create a Custom Target for All Preprocessed Files
    if(PREPROCESSED_FILES)
        add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES})
    endif()
endif() # End SD_PREPROCESS

# ===========================================================================
# End of Content Pasted Verbatim from blas/CMakeLists.txt
# ===========================================================================


# Testing Setup (Optional - from main)
if(SD_BUILD_TESTS)
    message("--- Configuring Tests ---")
    include(CTest)
    enable_testing()

    # Assume tests are in a 'tests_cpu' subdirectory relative to this CMakeLists.txt (original name)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests_cpu")
        # add_subdirectory(tests_cpu) # Requires tests_cpu/CMakeLists.txt to be adapted
        message(WARNING "SD_BUILD_TESTS enabled, but tests_cpu subdirectory needs its own updated CMakeLists.txt to link correctly.")
    elseif(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests") # Check for generic tests dir
        # add_subdirectory(tests) # Requires tests/CMakeLists.txt to be adapted
        message(WARNING "SD_BUILD_TESTS enabled, but tests subdirectory needs its own updated CMakeLists.txt to link correctly.")
    else()
        message(WARNING "SD_BUILD_TESTS is ON, but a 'tests' or 'tests_cpu' subdirectory was not found.")
    endif()
endif()


# Installation and Packaging (from main)
message("--- Configuring Installation and Packaging ---")

# Set install prefix (adjust as needed)
if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix")
endif()
message("Installation prefix: ${CMAKE_INSTALL_PREFIX}")

# Install libraries (already added install commands in CUDA/CPU blocks)
# Install header files (example)
install(DIRECTORY include/
        DESTINATION include/${PROJECT_NAME} # Install into a project-specific subdir
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl")

# Install generated definition header if it exists and was defined
if(DEFINED OP_OUTPUT_FILE AND EXISTS "${OP_OUTPUT_FILE}")
    install(FILES ${OP_OUTPUT_FILE} DESTINATION include/${PROJECT_NAME})
endif()
# Install configured header
install(FILES ${CMAKE_BINARY_DIR}/include/config.h DESTINATION include/${PROJECT_NAME})


# CPack Configuration (from main)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations for nd4j.")
set(CPACK_PACKAGE_RELEASE 1) # Or derive from version string
set(CPACK_PACKAGE_CONTACT "Eclipse Deeplearning4j Committers <dl4j-dev@eclipse.org>") # Updated contact
set(CPACK_PACKAGE_VENDOR "Eclipse Foundation") # Updated vendor
set(CPACK_PACKAGE_NAME "libnd4j")
# Try to get version from project version or define manually
if(PROJECT_VERSION)
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
else()
    # Extract version from project description or set manually if needed
    set(CPACK_PACKAGE_VERSION "1.0.0-SNAPSHOT") # Fallback version
endif()
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+).*" _ ${CPACK_PACKAGE_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE") # Point to license file
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}") # Installation dir within prefix

# Generator specific settings
if(UNIX)
    # Detect Linux distribution
    find_program(LSB_RELEASE_EXEC lsb_release)
    if(LSB_RELEASE_EXEC)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -sc OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        # Fallback detection (e.g., check /etc/os-release)
        if(EXISTS "/etc/os-release")
            file(STRINGS "/etc/os-release" OS_ID REGEX "^ID=.*")
            if(OS_ID) # Check if variable is set
                string(REPLACE "ID=" "" DISTRIBUTION "${OS_ID}")
            endif()
            # Release detection might be harder here
        endif()
    endif()

    # Detect architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(ARCHITECTURE "amd64") # Debian naming
        set(RPM_ARCHITECTURE "x86_64") # RPM naming
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ARCHITECTURE "arm64")
        set(RPM_ARCHITECTURE "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
        set(ARCHITECTURE "armhf") # Example, might depend on distro
        set(RPM_ARCHITECTURE "armv7hl") # Example
    else()
        set(ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
        set(RPM_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
    endif()

    if(DISTRIBUTION MATCHES "Ubuntu|Debian")
        set(CPACK_GENERATOR "DEB")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
        set(CPACK_DEBIAN_PACKAGE_SECTION "libs") # Changed from devel
        set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/eclipse/deeplearning4j") # Updated URL
        # Define dependencies carefully based on actual needs
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "") # Initialize empty list string
        # Add dependencies conditionally
        if(SD_CUDA)
            list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS "cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}") # Example
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS "libcudnn8") # Example, version might vary
        endif()
        # Convert list to comma-separated string if not empty
        if(CPACK_DEBIAN_PACKAGE_DEPENDS)
            string(REPLACE ";" ", " CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}")
        endif()


        # Postinst/Postrm scripts (Ensure these exist and are correct)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb")
            set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.deb")
        endif()
    elseif(DISTRIBUTION MATCHES "CentOS|Fedora|RHEL")
        set(CPACK_GENERATOR "RPM")
        set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${RPM_ARCHITECTURE})
        set(CPACK_RPM_PACKAGE_ARCHITECTURE ${RPM_ARCHITECTURE})
        set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
        set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries") # Changed from Development/Tools
        set(CPACK_RPM_PACKAGE_URL "https://github.com/eclipse/deeplearning4j") # Updated URL
        # Define dependencies carefully
        set(CPACK_RPM_PACKAGE_REQUIRES "") # Initialize empty list string
        # Add dependencies conditionally
        if(SD_CUDA)
            list(APPEND CPACK_RPM_PACKAGE_REQUIRES "cuda-cudart-${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}") # Example
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_RPM_PACKAGE_REQUIRES "libcudnn8") # Example, version might vary
        endif()
        # Convert list to comma-separated string if not empty
        if(CPACK_RPM_PACKAGE_REQUIRES)
            string(REPLACE ";" ", " CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}")
        endif()

        # Postinst/Postrm scripts
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
            set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
            set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
        endif()
        # Exclude auto filelist addition for common paths if needed
        # set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "/usr/local/lib")
    else()
        # Default generator for other Unix-like systems (e.g., TGZ)
        set(CPACK_GENERATOR "TGZ")
    endif()
elseif(WIN32)
    set(CPACK_GENERATOR "NSIS") # Or ZIP
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME}")
    set(CPACK_NSIS_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
    set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop") # Or PKG, TGZ
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-macOS)
endif()

include(CPack)

message("--- CMake Configuration Complete ---")