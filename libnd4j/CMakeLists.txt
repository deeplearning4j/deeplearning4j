# Main CMakeLists.txt (incorporating fixes for flatbuffers dependencies and paths)
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
option(SD_PREPROCESS "Use preprocessor" OFF) # Note: 'blas' file uses SD_PREPROCESS
set(CMAKE_VERBOSE_MAKEFILE ON)
# GCC specific flag:     -finstrument-functions enables call stack logging.
# Useful for debugging segfaults.
option(SD_GCC_FUNCTRACE "Use call traces" OFF)
option(FLATBUFFERS_BUILD_FLATC "Enable the build of the flatbuffers compiler" OFF) # This will be overridden below based on ENV

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

# Define ARM Compute Library URLs based on architecture
# For ARM 32-bit (Raspberry Pi)
set(ARM_COMPUTE_URL_ARMV7 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-armv7a-cpu-bin.tar.gz")
# For ARM 64-bit (Raspberry Pi, Odroid N2)
set(ARM_COMPUTE_URL_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-linux-aarch64-cpu-bin.tar.gz")
# For ARM 64-bit Android
set(ARM_COMPUTE_URL_ANDROID_AARCH64 "https://github.com/ARM-software/ComputeLibrary/releases/download/v25.04/arm_compute-v25.04-android-aarch64-cpu-bin.tar.gz")

# Helper flags for dependencies (can be set via cmake command line)
option(HELPERS_onednn "Build with oneDNN" OFF)
option(HELPERS_armcompute "Build with ARM Compute Library" OFF)
option(HELPERS_cudnn "Build with cuDNN" OFF)

# MSVC runtime lib can be either "MultiThreaded" or "MultiThreadedDLL", /MT and /MD respectively
set(MSVC_RT_LIB "MultiThreadedDLL")

set(SD_X86_BUILD false)
set(SD_ARM_BUILD false) # Add similar flag for ARM if needed

# Determine platform type
if(DEFINED ENV{SD_ARCH})
    set(SD_ARCH $ENV{SD_ARCH})
elseif(NOT DEFINED SD_ARCH) # Correct syntax: elseif
    # Fallback based on system if SD_ARCH env var is not set
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SD_ARCH "x86-64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64") # Correct syntax: elseif
        set(SD_ARCH "arm64") # or armv8-a depending on preference
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm") # Correct syntax: elseif
        set(SD_ARCH "armv7-a") # Example, might need refinement
    else()
        set(SD_ARCH ${CMAKE_SYSTEM_PROCESSOR}) # Default to system processor
    endif()
    message(STATUS "SD_ARCH not defined via environment or CMake cache, determined as: ${SD_ARCH}")
endif()


if(SD_ANDROID_BUILD)
    # For Android, trust ANDROID_ABI if defined by toolchain file
    if(DEFINED ANDROID_ABI)
        if(ANDROID_ABI MATCHES "x86_64")
            set(SD_X86_BUILD true)
            set(SD_ARCH "x86-64")
        elseif(ANDROID_ABI MATCHES "x86") # Correct syntax: elseif
            set(SD_X86_BUILD true)
            set(SD_ARCH "x86")
        elseif(ANDROID_ABI MATCHES "arm64-v8a") # Correct syntax: elseif
            set(SD_ARM_BUILD true)
            set(SD_ARCH "arm64-v8a") # or armv8-a
        elseif(ANDROID_ABI MATCHES "armeabi-v7a") # Correct syntax: elseif
            set(SD_ARM_BUILD true)
            set(SD_ARCH "armv7-a")
        endif()
    else()
        message(WARNING "SD_ANDROID_BUILD is ON, but ANDROID_ABI is not defined by the toolchain file.")
        # Add fallback logic if needed, though typically toolchain defines ABI
    endif()
elseif(NOT SD_IOS_BUILD) # Correct syntax: elseif
    # Non-Android, Non-iOS logic based on determined SD_ARCH
    if(SD_ARCH MATCHES "x86_64|x86-64|amd64|i[3-6]86")
        set(SD_X86_BUILD true)
    elseif(SD_ARCH MATCHES "arm*|aarch64*") # Correct syntax: elseif
        set(SD_ARM_BUILD true)
    endif()
    # Add iOS logic if necessary based on CMAKE_SYSTEM_NAME and SD_ARCH
elseif(SD_IOS_BUILD) # Correct syntax: elseif
    if(SD_ARCH MATCHES "arm*|aarch64*")
        set(SD_ARM_BUILD true)
    else() # Assuming x86 simulator
        set(SD_X86_BUILD true)
    endif()
endif()

# Set ARM Compute URL based on determined architecture
if(${SD_ARCH} MATCHES "armv7")
    set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
    message("Using ARM Compute Library for ARMv7 32-bit")
elseif(${SD_ARCH} MATCHES "armv8-a|arm64|aarch64") # Correct syntax: elseif
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (arch: ${SD_ARCH})")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for Linux ARM64 (arch: ${SD_ARCH})")
    endif()
endif()

message(STATUS "Build flags determined: SD_ANDROID_BUILD=${SD_ANDROID_BUILD}, SD_IOS_BUILD=${SD_IOS_BUILD}, SD_X86_BUILD=${SD_X86_BUILD}, SD_ARM_BUILD=${SD_ARM_BUILD}, SD_ARCH=${SD_ARCH}")


# Platform specific flags
if (SD_ANDROID_BUILD)
    set_property(GLOBAL PROPERTY JOB_POOLS one_job=1 two_jobs=2)
    # Flags typically set by Android toolchain file, but can be augmented here
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-unknown-pragmas -Wno-braced-scalar-init -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else")
elseif (APPLE OR SD_IOS_BUILD) # Correct syntax: elseif - Combined Apple macOS and iOS
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*" OR "${SD_ARCH}" MATCHES "arm64|aarch64|armv8-a")
        set(CMAKE_OSX_ARCHITECTURES "arm64") # Ensure ARM build for Apple Silicon / iOS ARM
    endif()
    # Common flags for Apple platforms
    set(CMAKE_CXX_FLAGS_RELEASE  "-O${SD_OPTIMIZATION_LEVEL} -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true -D_RELEASE=true")
    set(CMAKE_CXX_FLAGS_DEBUG  " -O${SD_OPTIMIZATION_LEVEL} -g -fPIC -Wno-return-type -Wno-braced-scalar-init -Wno-unknown-pragmas -Wno-delete-non-virtual-dtor -Wno-unused-command-line-argument -Wno-dangling-else -D__APPLE_OS__=true")
    # Add iOS specific flags if needed (often handled by toolchain)
    if(SD_IOS_BUILD)
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mios-version-min=...") # Example if needed
    endif()
elseif(WIN32) # Correct syntax: elseif
    # CXX Flags for Windows are handled within the pasted `blas` block logic below (and potentially CUDA section)
    # Ensure /bigobj is added for MSVC if needed
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    endif()
else() # Assuming Linux/other Unix
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -g")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE   "-O${SD_OPTIMIZATION_LEVEL} -fPIC -D_RELEASE=true")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG  " -g -O${SD_OPTIMIZATION_LEVEL} -fPIC")

    if (SD_SANITIZE)
        if(NOT DEFINED SD_SANITIZERS)
            set(SD_SANITIZERS "address") # Default to address sanitizer
        endif()
        set(SANITIZE_FLAGS " -Wall -Wextra -fPIE -lpthread -ftls-model=local-dynamic -static-libasan -fsanitize=${SD_SANITIZERS} -fno-sanitize-recover=all")
        message("Using sanitizers: ${SD_SANITIZERS}")

        # Apply sanitize flags (adjust logic based on whether SD_CUDA affects this)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
        if(SD_CUDA)
            # Ensure CUDA compilation also gets necessary sanitizer flags if possible/needed
            # This might require -Xcompiler flag for nvcc
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${SANITIZE_FLAGS}\"")
            # Linker flags might also be needed for CUDA device linking if using device-side sanitization
        endif()
    endif()
endif()

if(SD_NATIVE)
    IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64*" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm64*")
        set(SD_X86_BUILD false)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
    ELSE() # Assuming x86 or compatible
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    ENDIF()
endif()


# Find BLAS implementation if not using CUDA (CUDA uses cuBLAS)
if(NOT SD_CUDA)
    # we need this definition to avoid global memory use within onednn
    add_definitions(-DDNNL_ENABLE_CONCURRENT_EXEC=true)
    if ("${OPENBLAS_PATH}" STREQUAL "")
        # We don't want OpenBLAS on Apple by default, prefer Accelerate framework
        if (NOT APPLE AND NOT SD_IOS_BUILD)
            set(BLA_VENDOR "OpenBLAS" CACHE STRING "Selected BLAS vendor" FORCE)
        endif()

        # Find any available BLAS (could be OpenBLAS, MKL, Accelerate, generic)
        # Set COMPONENTS if specific BLAS is needed, otherwise finds any
        find_package(BLAS) # Let CMake find the best available BLAS

        if (BLAS_FOUND)
            message(STATUS "Found external BLAS implementation: ${BLAS_LIBRARIES}")
            # This definition might be too generic if BLAS_LIBRARIES is not OpenBLAS
            # Consider more specific checks if needed: if(${BLAS_INFO} MATCHES "openblas")
            # add_definitions(-D__EXTERNAL_BLAS__=true) # Maybe redundant if BLAS_FOUND implies external
            set(HAVE_OPENBLAS TRUE) # Assume BLAS found is usable like OpenBLAS interface
            set(OPENBLAS_LIBRARIES ${BLAS_LIBRARIES}) # Use the found libraries
        else()
            message(WARNING "No external BLAS implementation found by find_package(BLAS). Performance may be limited.")
        endif()
    else()
        # User provided OPENBLAS_PATH, assume it's OpenBLAS
        set(HAVE_OPENBLAS TRUE)
        message(STATUS "Using user-provided OpenBLAS path: ${OPENBLAS_PATH}")
        include_directories(${OPENBLAS_PATH}/include)
        link_directories(${OPENBLAS_PATH}/lib) # Add lib directory if not standard
        # Determine library name (e.g., openblas, libopenblas) - might need adjustment
        find_library(OPENBLAS_LIBRARY NAMES openblas libopenblas HINTS ${OPENBLAS_PATH}/lib REQUIRED)
        if(OPENBLAS_LIBRARY)
            set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
            message(STATUS "Found OpenBLAS library: ${OPENBLAS_LIBRARIES}")
        else()
            message(FATAL_ERROR "OPENBLAS_PATH was provided, but library not found in ${OPENBLAS_PATH}/lib")
        endif()
        # Add definition to indicate OpenBLAS is used
        add_definitions(-DHAVE_OPENBLAS=1)
    endif()
endif()


#arm-compute entry
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
        add_definitions(-DHAVE_ARMCOMPUTE=1 -DARMCOMPUTENEON_ENABLED) # Add necessary definitions
    else()
        message(STATUS "ARM Compute not found via find_package. Checking download conditions.")
        if(DEFINED ARM_COMPUTE_URL AND (${SD_ARCH} MATCHES "armv7" OR ${SD_ARCH} MATCHES "armv8-a" OR ${SD_ARCH} MATCHES "arm64"))
            message(STATUS "Attempting to download ARM Compute Library from: ${ARM_COMPUTE_URL}")

            # Include ExternalProject module if not already done
            include(ExternalProject)

            # Configure the download project using ExternalProject_Add
            ExternalProject_Add(armcompute_external
                    URL ${ARM_COMPUTE_URL}
                    # URL_HASH SHA256=... # Add SHA256 hash for the specific version/file for security
                    SOURCE_DIR "${CMAKE_BINARY_DIR}/armcompute-src"
                    CONFIGURE_COMMAND "" # Pre-built binaries, no configure step needed
                    BUILD_COMMAND ""     # Pre-built binaries, no build step needed
                    INSTALL_COMMAND ""   # We'll use the files directly from SOURCE_DIR
                    LOG_DOWNLOAD TRUE
            )

            # Define the location AFTER the download step (ExternalProject doesn't set these automatically)
            set(ARMCOMPUTE_ROOT "${CMAKE_BINARY_DIR}/armcompute-src") # Root dir after extraction
            set(ARMCOMPUTE_INCLUDE_DIR_LIB "${ARMCOMPUTE_ROOT}/include") # Include for arm_compute library
            set(ARMCOMPUTE_INCLUDE_DIR_CORE "${ARMCOMPUTE_ROOT}/include") # Include for arm_compute_core library
            set(ARMCOMPUTE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute.so") # Path to library
            set(ARMCOMPUTE_CORE_LIB_PATH "${ARMCOMPUTE_ROOT}/lib/libarm_compute_core.so") # Path to core library

            # Create IMPORTED targets for the libraries AFTER ExternalProject_Add
            add_library(arm_compute SHARED IMPORTED GLOBAL)
            set_property(TARGET arm_compute PROPERTY IMPORTED_LOCATION ${ARMCOMPUTE_LIB_PATH})
            add_dependencies(arm_compute armcompute_external) # Depends on download

            add_library(arm_compute_core SHARED IMPORTED GLOBAL)
            set_property(TARGET arm_compute_core PROPERTY IMPORTED_LOCATION ${ARMCOMPUTE_CORE_LIB_PATH})
            add_dependencies(arm_compute_core armcompute_external) # Depends on download

            # Set variables for use in the build
            include_directories(${ARMCOMPUTE_INCLUDE_DIR_LIB} ${ARMCOMPUTE_INCLUDE_DIR_CORE})
            set(ARMCOMPUTE_LIBRARIES arm_compute arm_compute_core) # Use the target names
            set(HAVE_ARMCOMPUTE 1)
            add_definitions(-DHAVE_ARMCOMPUTE=1 -DARMCOMPUTENEON_ENABLED) # Add definitions
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
            GIT_TAG v2.7 # Or specify a desired tag/commit
            SOURCE_DIR "${CMAKE_BINARY_DIR}/onednn-src"
            BINARY_DIR "${CMAKE_BINARY_DIR}/onednn-build"
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} # Example install prefix
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDNNL_LIBRARY_TYPE=${DNNL_LIBRARY_TYPE}
            -DDNNL_BUILD_EXAMPLES=${DNNL_BUILD_EXAMPLES}
            -DDNNL_BUILD_TESTS=${DNNL_BUILD_TESTS}
            # Pass necessary compiler/flags if cross-compiling
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
            INSTALL_COMMAND "" # We will link against the build directory library
            LOG_CONFIGURE TRUE
            LOG_BUILD TRUE
            LOG_DOWNLOAD TRUE
    )

    # Define imported target for oneDNN static library
    add_library(onednn STATIC IMPORTED GLOBAL)
    set_property(TARGET onednn PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/onednn-build/src/libdnnl.a) # Adjust path/name if necessary
    add_dependencies(onednn onednn_external) # Ensure it's built first

    # Set include directories and libraries variable
    set(ONEDNN_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/onednn-src/include) # Include dir from source
    include_directories(${ONEDNN_INCLUDE_DIRS})
    set(ONEDNN_LIBRARIES onednn) # Use the target name

    message(STATUS "oneDNN configured via ExternalProject. Include: ${ONEDNN_INCLUDE_DIRS}")

endif()


if(${HELPERS_cudnn})
    if(NOT SD_CUDA)
        message(FATAL_ERROR "Can't build with cuDNN helper on a non-CUDA platform (SD_CUDA is OFF)")
    endif()

    SET(CUDNN_LIBNAME "cudnn") # Base name

    # Prefer environment variables if set
    if(DEFINED ENV{CUDNN_ROOT_DIR})
        set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
        message(STATUS "Using CUDNN_ROOT_DIR from environment: ${CUDNN_ROOT_DIR}")
    endif()
    # CUDA_TOOLKIT_ROOT_DIR often set by find_package(CUDA) or environment
    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
        message(STATUS "Using CUDA_TOOLKIT_ROOT_DIR from environment: ${CUDA_TOOLKIT_ROOT_DIR}")
    elseif(DEFINED CUDA_TOOLKIT_ROOT_DIR) # Correct syntax: elseif
        message(STATUS "Using CUDA_TOOLKIT_ROOT_DIR from CUDA find_package: ${CUDA_TOOLKIT_ROOT_DIR}")
    endif()

    # Search for include file
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES include include/cuda cuda/include) # Common suffixes

    # Search for library file
    find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIBNAME} cudnn_static # Look for dynamic and static
            HINTS
            ${CUDNN_ROOT_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}
            ENV CUDNN_ROOT_DIR
            ENV CUDA_TOOLKIT_ROOT_DIR
            PATH_SUFFIXES lib lib64 lib/x64 cuda/lib cuda/lib64 targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib) # Common suffixes

    message(STATUS "cuDNN search: Include Dir = ${CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN search: Library = ${CUDNN_LIBRARY}")

    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message(STATUS "Found cuDNN: include at ${CUDNN_INCLUDE_DIR}, library at ${CUDNN_LIBRARY}")
        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true CACHE BOOL "cuDNN Found" FORCE)
        add_definitions(-DHAVE_CUDNN=1) # Add definition
        set(CUDNN ${CUDNN_LIBRARY}) # Set variable for linking
    else()
        message(WARNING "cuDNN helper was specified, but cuDNN library or include directory not found. Build will continue without cuDNN support.")
        set(HAVE_CUDNN false CACHE BOOL "cuDNN Found" FORCE)
    endif()
endif()

# ================== Refactored Flatbuffers Section ==================
# Include ExternalProject module if not already done
include(ExternalProject)

set(FLATBUFFERS_TAG "v25.2.10" CACHE STRING "Flatbuffers Git Tag")
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Build Flatbuffers Tests")

# Determine if flatc needs to be built based on environment variable set by build script
set(GENERATE_FLATC OFF) # Default OFF
if(DEFINED ENV{GENERATE_FLATC})
    if("$ENV{GENERATE_FLATC}" STREQUAL "ON")
        set(GENERATE_FLATC ON)
        message(STATUS "Building flatc based on ENV{GENERATE_FLATC}=ON")
    endif()
endif()
# Cache the decision for the ExternalProject argument
set(FLATBUFFERS_BUILD_FLATC ${GENERATE_FLATC} CACHE BOOL "Build flatc executable" FORCE)

# Define ExternalProject for flatbuffers
ExternalProject_Add(flatbuffers_external
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        GIT_TAG ${FLATBUFFERS_TAG}
        SOURCE_DIR "${CMAKE_BINARY_DIR}/flatbuffers-src"  # Source code location after download
        BINARY_DIR "${CMAKE_BINARY_DIR}/flatbuffers-build" # Where flatbuffers itself will be built
        CMAKE_ARGS # Arguments passed to flatbuffers' CMake configure step
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} # Use main project's install prefix if needed
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DFLATBUFFERS_BUILD_TESTS=${FLATBUFFERS_BUILD_TESTS}
        -DFLATBUFFERS_BUILD_FLATC=${FLATBUFFERS_BUILD_FLATC} # Pass the flag to build flatc
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        # Pass necessary flags, especially for cross-compiling or specific runtimes like MSVC
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
        # Pass MSVC specific flags if on Windows
        $<$<BOOL:${WIN32}>:-DCMAKE_MSVC_RUNTIME_LIBRARY=${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>>
        # Use generator expressions for configuration-specific flags if needed
        # $<$<CONFIG:Debug>:-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}> # Example
        # $<$<CONFIG:Release>:-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}> # Example
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> # Build command
        INSTALL_COMMAND ""             # We don't need to install it system-wide
        UPDATE_COMMAND ""              # Avoid automatic updates unless needed
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_DOWNLOAD TRUE
)

# Get include directory property from the external project AFTER it's configured/built
# This usually requires the path to be set manually based on SOURCE_DIR as ExternalProject doesn't expose include dirs easily before build.
set(FLATBUFFERS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/flatbuffers-src/include)
include_directories(${FLATBUFFERS_INCLUDE_DIR})
message(STATUS "Flatbuffers include directory set to: ${FLATBUFFERS_INCLUDE_DIR}")

# Define imported targets AFTER ExternalProject_Add
# Add STATIC library target for libflatbuffers
add_library(flatbuffers STATIC IMPORTED GLOBAL)
# Adjust path/name based on platform and build type if necessary (e.g., libflatbuffers.a, flatbuffers.lib)
set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/libflatbuffers.a) # Assume Linux/macOS name for now
if(WIN32)
    # Adjust for Windows library name convention if needed (check flatbuffers build output)
    set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/$<IF:$<CONFIG:Debug>,Debug,Release>/flatbuffers.lib) # Example for MSVC Debug/Release dirs
endif()
add_dependencies(flatbuffers flatbuffers_external) # Ensure library depends on the external build

# Add flatc executable target if built (FLATBUFFERS_BUILD_FLATC is ON)
if(FLATBUFFERS_BUILD_FLATC)
    add_executable(flatc IMPORTED GLOBAL)
    # Adjust path based on platform and build type if needed
    set(FLATC_EXE_PATH "${CMAKE_BINARY_DIR}/flatbuffers-build/flatc")
    if(WIN32)
        set(FLATC_EXE_PATH "${CMAKE_BINARY_DIR}/flatbuffers-build/$<IF:$<CONFIG:Debug>,Debug,Release>/flatc.exe") # Example for MSVC Debug/Release dirs
    endif()
    set_property(TARGET flatc PROPERTY IMPORTED_LOCATION ${FLATC_EXE_PATH})
    add_dependencies(flatc flatbuffers_external) # Ensure executable depends on the external build
    message(STATUS "flatc executable IMPORTED target created. Path will be resolved at build time: ${FLATC_EXE_PATH}")
endif()
# ================== End Refactored Flatbuffers Section ==================


# ================== Generate Flatbuffers Sources Section (FIXED) ==================
# Generate Flatbuffers sources only if requested via FLATBUFFERS_BUILD_FLATC
# (which depends on ENV{GENERATE_FLATC} being ON)
if(FLATBUFFERS_BUILD_FLATC)
    message("Generating flatbuffers sources (FLATBUFFERS_BUILD_FLATC is ON)...")

    # --- Configure-time check 'if(TARGET flatc)' REMOVED ---
    # Build-time dependencies handle the availability of the flatc executable.

    set(FLATC_TARGET flatc) # Use the imported target name

    # Define schema and output locations based on project structure
    # Schema file confirmed via user input to be in include/graph/scheme/
    # ${CMAKE_CURRENT_SOURCE_DIR} refers to the libnd4j directory containing this CMakeLists.txt
    set(FLATC_INPUT_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/scheme/graph.fbs")

    # C++ generated sources location
    set(FLATC_OUTPUT_DIR_CPP "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd")

    # Java generated sources location (relative path to the nd4j-serde module)
    set(FLATC_OUTPUT_DIR_JAVA "${CMAKE_CURRENT_SOURCE_DIR}/../../nd4j-serde/nd4j-flatbuffers-java/src/main/java")

    # Ensure output directories exist before generation
    file(MAKE_DIRECTORY ${FLATC_OUTPUT_DIR_CPP})
    file(MAKE_DIRECTORY ${FLATC_OUTPUT_DIR_JAVA})

    # Add command to generate C++ and Java sources from the primary graph schema
    # flatc compiler should handle includes relative to the schema's directory (${CMAKE_CURRENT_SOURCE_DIR}/include/graph/scheme/)
    add_custom_command(
            OUTPUT ${FLATC_OUTPUT_DIR_CPP}/graph_generated.h # Define a primary output file for dependency tracking
            # Use the flatc target (path resolved by CMake at build time via generator expression)
            COMMAND $<TARGET_FILE:${FLATC_TARGET}>
            # Specify C++ output directory and options
            -o ${FLATC_OUTPUT_DIR_CPP}
            --cpp
            --gen-object-api
            # Specify Java output directory and options
            -o ${FLATC_OUTPUT_DIR_JAVA}
            --java
            --gen-object-api
            # Specify the input schema file
            ${FLATC_INPUT_SCHEMA}
            # Command depends on the flatc executable target and the input schema file
            DEPENDS ${FLATC_TARGET} ${FLATC_INPUT_SCHEMA}
            COMMENT "Generating flatbuffers C++ and Java sources from ${FLATC_INPUT_SCHEMA}"
            VERBATIM
    )

    # Add a custom target to ensure the generation command runs
    # This target depends on the primary output file of the custom command
    add_custom_target(generate_flatbuffers ALL DEPENDS ${FLATC_OUTPUT_DIR_CPP}/graph_generated.h)

else()
    # This else now corresponds to if(FLATBUFFERS_BUILD_FLATC)
    message("Skipping flatbuffers source generation (FLATBUFFERS_BUILD_FLATC is OFF).")
endif()
# ================== End Generate Flatbuffers Sources Section ==================


configure_file(include/config.h.in include/config.h @ONLY)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# ===========================================================================
# Start of Content Pasted Verbatim from blas/CMakeLists.txt
# (Ensure paths inside this section are relative to libnd4j/ or use CMAKE_CURRENT_SOURCE_DIR properly)
# ===========================================================================

include(CheckCXXCompilerFlag)

# Include Directories Based on OS
if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD)
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

if(APPLE)
    message(STATUS "Configuring for Apple OS (macOS/iOS)") # Note: Main file also has this check
    link_directories(/usr/local/lib /usr/lib /lib)
endif()

# Define Compiler Flags for Specific Builds (these add to existing flags)
if(SD_APPLE_BUILD OR APPLE OR SD_IOS_BUILD) # Broader check for Apple
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSD_APPLE_BUILD=true") # General Apple flag
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSD_APPLE_BUILD=true")
    if (NOT SD_IOS_BUILD) # macOS specific version min
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

# Windows Specific Configurations
if(WIN32 AND NOT ANDROID)
    # Use response files for long command lines (Ninja usually does this automatically)
    set(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS ON)
    # Check compiler ID for response file flag format
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
        set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Correct syntax: elseif
        set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@") # MSVC uses @ too
        set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")
    endif()
    # Force Ninja response files if using Ninja generator
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set(CMAKE_NINJA_FORCE_RESPONSE_FILE ON CACHE INTERNAL "")
    endif()
    # Big object flag for MinGW
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
endif()

# Link Time Optimization
if(SD_USE_LTO) # Note: Option defined in Main
    # Check compiler support for LTO
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Enabling Link Time Optimization (LTO)")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -flto") # Add for C compiler too
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Correct syntax: elseif
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

# Compile Definitions for Operations
set(DEFINITIONS_CONTENT "")
# Check if SD_ALL_OPS is defined and ON, or if SD_OPS_LIST is empty
if((DEFINED SD_ALL_OPS AND SD_ALL_OPS) OR NOT DEFINED SD_OPS_LIST OR "${SD_OPS_LIST}" STREQUAL "")
    message("Adding all ops due to SD_ALL_OPS=${SD_ALL_OPS} or empty SD_OPS_LIST=${SD_OPS_LIST}")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
    message("Adding selective ops from list: ${SD_OPS_LIST}")
    add_compile_definitions(SD_SELECTIVE_OPS=1) # Add a general flag for selective ops
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_OPS 1\n")
    foreach(OP ${SD_OPS_LIST})
        string(TOUPPER ${OP} OP_UPPER) # Ensure consistency (e.g., use upper case)
        add_compile_definitions(OP_${OP_UPPER}=1)
        message(STATUS "Adding op definition: OP_${OP_UPPER}=1")
        string(APPEND DEFINITIONS_CONTENT "#define OP_${OP_UPPER} 1\n")
    endforeach()
endif()

# Compile Definitions for Types
# Check if SD_TYPES_LIST is defined and not empty
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
    # Optionally add #define SD_ALL_TYPES 1 if needed by code
endif()

# Write Definitions to Header File
# Ensure OP_OUTPUT_FILE is defined (e.g., via build script argument)
if(DEFINED OP_OUTPUT_FILE AND OP_OUTPUT_FILE) # Check if defined and not empty
    # Ensure the path is absolute or relative to the correct directory (e.g., binary dir)
    if(NOT IS_ABSOLUTE "${OP_OUTPUT_FILE}")
        set(OP_OUTPUT_FILE_ABS "${CMAKE_CURRENT_BINARY_DIR}/${OP_OUTPUT_FILE}")
    else()
        set(OP_OUTPUT_FILE_ABS "${OP_OUTPUT_FILE}")
    endif()
    get_filename_component(OP_OUTPUT_DIR ${OP_OUTPUT_FILE_ABS} DIRECTORY)
    file(MAKE_DIRECTORY ${OP_OUTPUT_DIR}) # Ensure directory exists
    message(STATUS "Definitions will be written to \"${OP_OUTPUT_FILE_ABS}\"")
    file(WRITE "${OP_OUTPUT_FILE_ABS}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n\n${DEFINITIONS_CONTENT}\n#endif // SD_DEFINITIONS_GEN_H_\n")
else()
    message(STATUS "OP_OUTPUT_FILE not defined, skipping generation of definitions header.")
endif()

# Architecture Tuning Flags
set(ARCH_TUNE "") # Initialize
if(SD_ARCH) # Check if SD_ARCH is set
    if(SD_ARCH MATCHES "armv8|aarch64") # Covers armv8-a, arm64, etc.
        set(ARCH_TUNE "-march=armv8-a") # Generic ARMv8-A, adjust if specific CPU needed
    elseif(SD_ARCH MATCHES "armv7") # Correct syntax: elseif
        set(ARCH_TUNE "-march=armv7-a -mfpu=neon") # Example for ARMv7 with Neon
    elseif(SD_ARCH MATCHES "power*") # Correct syntax: elseif
        set(ARCH_TUNE "-mcpu=${SD_ARCH} -mtune=${SD_ARCH} -D__POWER")
    elseif(SD_ARCH MATCHES "x86-64|x86_64|amd64") # Correct syntax: elseif
        # x86-64 tuning depends on SD_EXTENSION
        if(DEFINED SD_EXTENSION)
            if(SD_EXTENSION MATCHES "avx512")
                message("Building for x86-64 with AVX512 extension...")
                set(ARCH_TUNE "-mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512cd")
                # Add common prerequisite flags if needed (check compiler defaults)
                set(ARCH_TUNE "-march=skylake-avx512 ${ARCH_TUNE}") # Use a base march if safe
                add_compile_definitions(F_AVX512=true)
            elseif(SD_EXTENSION MATCHES "avx2") # Correct syntax: elseif
                message("Building for x86-64 with AVX2 extension...")
                # Common flags for AVX2 support
                set(ARCH_TUNE "-mavx2 -mfma -mf16c") # -mf16c often useful with AVX2
                # Add common prerequisite flags if needed
                set(ARCH_TUNE "-march=haswell ${ARCH_TUNE}") # Use a base march if safe
                add_compile_definitions(F_AVX2=true)
                # Check for split load/store flags (less common now)
                check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" NO_AVX256_SPLIT)
                if(NO_AVX256_SPLIT)
                    set(ARCH_TUNE "${ARCH_TUNE} -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
                endif()
            else() # Default x86-64 without specific AVX extensions specified
                message("Building for generic x86-64...")
                set(ARCH_TUNE "-march=x86-64") # Generic baseline
                add_compile_definitions(F_X64=true)
            endif()
        else() # Default x86-64 if SD_EXTENSION not defined
            message("Building for generic x86-64 (SD_EXTENSION not set)...")
            set(ARCH_TUNE "-march=x86-64")
            add_compile_definitions(F_X64=true)
        endif()
    else()
        message(WARNING "Unhandled SD_ARCH value: ${SD_ARCH}. No specific tuning flags added.")
    endif()

    # Apply tuning flags based on compiler (avoid applying to MSVC directly)
    if(ARCH_TUNE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_TUNE}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_TUNE}") # Apply to C as well
    elseif(ARCH_TUNE AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Correct syntax: elseif
        # MSVC uses /arch: (e.g., /arch:AVX2, /arch:AVX512)
        if(SD_EXTENSION MATCHES "avx512")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX512")
        elseif(SD_EXTENSION MATCHES "avx2") # Correct syntax: elseif
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
        endif()
        # Note: MSVC doesn't have direct equivalents for all march flags
    endif()
endif()


# Compiler-Specific Flags (Additions)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(NOT SD_CUDA) # Avoid interfering with CUDA's compiler flags if possible
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # Main file sets C++17, ensure consistency
    endif()
    # Rpath settings for shared libraries on Linux/Unix
    if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD AND SD_SHARED_LIB)
        # Use $ORIGIN for libraries to find dependencies in the same directory
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN'")
    endif()
    # Export dynamic symbols for debugging/profiling if needed
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT APPLE AND NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic") # For executables if any
    endif()
    # Functrace flags (already handled by main functrace block, ensure no conflict)
    if("${SD_GCC_FUNCTRACE}" STREQUAL "ON")
        add_compile_definitions(SD_GCC_FUNCTRACE) # Ensure definition exists
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang") # Correct syntax: elseif
    # Clang specific flags if any
    if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD AND SD_SHARED_LIB)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN'")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel") # Correct syntax: elseif
    # Intel specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast") # Example optimization
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Correct syntax: elseif
    # MSVC specific flags are often handled by CMAKE_CXX_FLAGS_RELEASE/DEBUG
    # Ensure /EHsc is set for exception handling (usually default)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc") # Apply to C too if needed
endif()


# External Include Directories (Consider if needed, often handled by find_package)
set(EXTERNAL_INCLUDE_DIRS "")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    # list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include") # Usually not needed
endif()

# Determine primary build mode (CPU or CUDA) - Should be set by build script via -DSD_CUDA=ON or -DSD_CPU=ON
if(NOT DEFINED SD_CUDA AND NOT DEFINED SD_CPU)
    message(WARNING "Neither SD_CUDA nor SD_CPU is defined. Assuming SD_CPU=TRUE.")
    set(SD_CPU TRUE)
    set(SD_CUDA FALSE)
elseif(DEFINED SD_CUDA AND SD_CUDA) # Correct syntax: elseif
    set(SD_CPU FALSE) # Ensure CPU is false if CUDA is true
elseif(DEFINED SD_CPU AND SD_CPU) # Correct syntax: elseif
    set(SD_CUDA FALSE) # Ensure CUDA is false if CPU is true
endif()
message(STATUS "Build Mode: SD_CPU=${SD_CPU}, SD_CUDA=${SD_CUDA}")


# Set SD_LIBRARY_NAME Based on Build Type
if(NOT DEFINED SD_LIBRARY_NAME)
    if(SD_CUDA)
        set(SD_LIBRARY_NAME nd4jcuda)
    elseif(SD_CPU) # Correct syntax: elseif - Use elseif for clarity
        set(SD_LIBRARY_NAME nd4jcpu)
    else()
        message(FATAL_ERROR "Cannot determine SD_LIBRARY_NAME. Build mode (SD_CPU/SD_CUDA) is unclear.")
    endif()
endif()
message(STATUS "Target library name: ${SD_LIBRARY_NAME}")


# Initialize source lists
set(ALL_SOURCES "")
set(CUSTOMOPS_ONEDNN_SOURCES "")
set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
set(CUSTOMOPS_CUDNN_SOURCES "")
set(CUDA_GENERATED_SOURCES "")
set(CPU_GENERATED_SOURCES "")
set(CUSTOMOPS_GENERIC_SOURCES "") # CPU generated combinations

# Helper function to generate CUDA/CPU source files from templates based on types
# (Assumes genCompilation and genPartitionCombination are defined earlier or in included files)
# Example Placeholder - actual implementation needed if used
# function(generate_typed_sources TEMPLATE_LIST TYPE_LIST OUTPUT_VARIABLE)
#    # ... logic to loop through templates and types ...
#    # configure_file(${template} ${output_file} @ONLY)
#    # list(APPEND ${OUTPUT_VARIABLE} ${output_file})
#    set(${OUTPUT_VARIABLE} ${${OUTPUT_VARIABLE}} PARENT_SCOPE)
# endfunction()


# CUDA Configuration Block
if(SD_CUDA)
    message(STATUS "--- Configuring CUDA Build ---")

    add_definitions(-D__CUDABLAS__=true)

    # Find CUDA package (should be found earlier, but check again)
    if(NOT CUDA_FOUND)
        find_package(CUDA REQUIRED)
    endif()

    if(CUDA_FOUND)
        message(STATUS "CUDA Found: Version=${CUDA_VERSION}, Toolkit Root=${CUDA_TOOLKIT_ROOT_DIR}")
        include_directories(${CUDA_INCLUDE_DIRS})

        # Common CUDA flags
        set(COMMON_CUDA_FLAGS "-allow-unsupported-compiler --expt-extended-lambda -Xfatbin -compress-all -maxrregcount=128")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS}")

        # Platform specific flags
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # Pass PIC flag to host compiler via -Xcompiler
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") # Correct syntax: elseif
            # MSVC flags for CUDA (e.g., exception handling, runtime library)
            # These might be passed via CMAKE_CXX_FLAGS automatically by CMake's CUDA support
            # Ensure MSVC runtime library matches main project setting (/MD or /MT)
            # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options /EHsc,/MD") # Example if needed
        endif()

        # Debug/Release specific CUDA flags
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo") # Device debug info
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math") # Example release optimization

        # CUDA Architectures (from COMPUTE variable set by build script)
        if(NOT DEFINED COMPUTE)
            set(COMPUTE "all") # Default if not provided
            message(STATUS "COMPUTE variable not set, defaulting to 'all'")
        endif()
        string(TOLOWER "${COMPUTE}" COMPUTE_LOWER)

        set(CUDA_ARCH_FLAGS_LIST "") # Initialize as list
        if(COMPUTE_LOWER STREQUAL "all")
            # Define default list based on CUDA version
            if(CUDA_VERSION VERSION_LESS 11.0)
                set(CUDA_ARCH_LIST 60 61 70 75) # Example for CUDA 10.x
            elseif(CUDA_VERSION VERSION_LESS 12.0) # Correct syntax: elseif
                set(CUDA_ARCH_LIST 60 61 70 75 80 86) # Example for CUDA 11.x
            else()
                set(CUDA_ARCH_LIST 75 80 86 90) # Example for CUDA 12.x+ (Adjust as needed)
            endif()
        elseif(COMPUTE_LOWER STREQUAL "auto") # Correct syntax: elseif
            # Let CMake detect architectures
            set(CUDA_ARCH_LIST "") # Empty list triggers CMake auto-detection
        else()
            # User provided list (e.g., "7.5 8.6")
            string(REPLACE " " ";" CUDA_ARCH_LIST_INPUT ${COMPUTE_LOWER})
            set(CUDA_ARCH_LIST "")
            foreach(ARCH_ITEM ${CUDA_ARCH_LIST_INPUT})
                string(REPLACE "." "" ARCH_CLEAN ${ARCH_ITEM}) # Convert 7.5 -> 75
                list(APPEND CUDA_ARCH_LIST ${ARCH_CLEAN})
            endforeach()
        endif()

        if(CUDA_ARCH_LIST) # If list is not empty (i.e., not "auto")
            foreach(ARCH_CODE ${CUDA_ARCH_LIST})
                # Add flags for both compute_ and sm_
                list(APPEND CUDA_ARCH_FLAGS_LIST "-gencode" "arch=compute_${ARCH_CODE},code=compute_${ARCH_CODE}")
                list(APPEND CUDA_ARCH_FLAGS_LIST "-gencode" "arch=compute_${ARCH_CODE},code=sm_${ARCH_CODE}")
            endforeach()
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST}) # Set CMake variable for info
            message(STATUS "Targeting CUDA Architectures (sm_): ${CUDA_ARCH_LIST}")
        else()
            message(STATUS "Targeting CUDA Architectures: Auto-detected by CMake/nvcc")
            set(CMAKE_CUDA_ARCHITECTURES "") # Explicitly empty for auto
            # CUDA_ARCH_FLAGS_LIST remains empty, nvcc defaults will be used unless CMAKE_CUDA_ARCHITECTURES is set elsewhere
        endif()

        # Convert list of flags to string
        string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS_LIST}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

        # Sanitize flags for CUDA
        if(SD_SANITIZE AND SANITIZE_FLAGS) # Check if sanitization is enabled
            message(STATUS "Adding sanitizer flags to CUDA host/device compiler via -Xcompiler")
            # Pass sanitize flags to host compiler
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${SANITIZE_FLAGS}\"")
            # Check CUDA version for compute-sanitizer support
            if(CUDA_VERSION VERSION_GREATER_EQUAL 10.1)
                # Add device-side flags if needed (e.g., for compute-sanitizer)
                # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --debug") # Often needed
            endif()
            # Linker flags for sanitizer runtime might be needed (added later in target_link_libraries)
        endif()

        # --- Find CUDA Libraries (cuBLAS, cuSolver) ---
        find_package(CUDAToolkit COMPONENTS cublas cusolver) # Find components
        if(CUDAToolkit_FOUND)
            message(STATUS "Found CUDA Toolkit components: cuBLAS, cuSolver")
            # Libraries are typically in CUDAToolkit_LIBRARIES, check FindCUDAToolkit docs
        else()
            message(WARNING "Could not find required CUDA Toolkit components (cuBLAS, cuSolver) via find_package(CUDAToolkit). Linking might fail.")
            # Fallback: Try finding manually (less robust)
            find_library(CUDA_cublas_LIBRARY NAMES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
            find_library(CUDA_cusolver_LIBRARY NAMES cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
            if(NOT CUDA_cublas_LIBRARY OR NOT CUDA_cusolver_LIBRARY)
                message(FATAL_ERROR "Manual search for cuBLAS/cuSolver library failed.")
            endif()
            # Set CUDAToolkit_LIBRARIES manually if found
            set(CUDAToolkit_LIBRARIES ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
        endif()


        # --- Generate CUDA Source Files (Templates, etc.) ---
        message(STATUS "Configuring CUDA source file generation...")
        set(CUDA_GENERATED_DIR "${CMAKE_BINARY_DIR}/cuda_generated")
        file(MAKE_DIRECTORY ${CUDA_GENERATED_DIR})
        # (Include TypeMST.cmake or define genCompilation/genSingleFunctionCuda here)
        # include(TypeMST.cmake) # Example include

        # Placeholder for actual generation logic using helper functions
        # file(GLOB CUDA_COMPILATION_UNITS include/loops/cuda/compilation_units/*.cu.in)
        # foreach(FL_ITEM ${CUDA_COMPILATION_UNITS})
        #     genCompilation(${FL_ITEM} CUDA_GENERATED_SOURCES) # Appends to CUDA_GENERATED_SOURCES
        # endforeach()
        # ... similar logic for comb_compilation_units using genSingleFunctionCuda ...
        message(STATUS "CUDA source generation configured (Actual generation happens at build time).")


        # --- Gather ALL CUDA Source Files ---
        file(GLOB_RECURSE COMMON_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/exceptions/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/impl/*.cpp" # Common graph parts
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/generic/*.cpp" # Generic ops often needed
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/impl/*.cpp" # Common impl
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/impl/*.cpp" # Common array parts
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/types/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/types/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/impl/*.cpp" # Common execution parts
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/indexing/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/impl/*.cpp" # Common loop parts
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/impl/*.cpp" # Common legacy parts
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/build_info.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ConstMessages.cpp"
                # Exclude CPU specific implementations if present in common globs
                EXCLUDE REGEX ".*/cpu/.*\\.cpp$"
        )

        file(GLOB_RECURSE CUDA_SPECIFIC_SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cuda/*.h" # Include CUDA specific headers too
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
                # Include generated CUDA sources placeholder
                # "${CUDA_GENERATED_DIR}/*.cu" # Add generated sources here if applicable
        )

        # Platform specific sources (cuDNN, potentially oneDNN if used with CUDA ops)
        if(HAVE_CUDNN)
            message(STATUS "Adding cuDNN platform sources")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/cudnn/*.cu")
        endif()
        # if(HAVE_ONEDNN) # If oneDNN has CUDA-specific parts
        #     message(STATUS "Adding oneDNN platform sources (CUDA context)")
        #     file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES ...)
        # endif()

        # Combine all sources
        set(ALL_SOURCES
                ${COMMON_SOURCES}
                ${CUDA_SPECIFIC_SOURCES}
                ${CUDA_GENERATED_SOURCES} # Add generated list
                ${CUSTOMOPS_CUDNN_SOURCES}
                # ${CUSTOMOPS_ONEDNN_SOURCES} # Add if applicable
        )

        # Create Object Library
        add_library(samediff_obj OBJECT ${ALL_SOURCES})
        # --- ADDED DEPENDENCY ---
        # Ensure flatbuffers external project is built before compiling objects
        add_dependencies(samediff_obj flatbuffers_external)
        # --- END ADDED DEPENDENCY ---
        target_compile_features(samediff_obj PUBLIC cxx_std_17 cuda_std_17) # Ensure C++/CUDA standards

        # Set include directories for the object library
        target_include_directories(samediff_obj PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include" # Source include
                "${CMAKE_CURRENT_BINARY_DIR}/include" # Generated config.h include
                ${FLATBUFFERS_INCLUDE_DIR}            # Flatbuffers include
                ${CUDA_INCLUDE_DIRS}                  # CUDA includes
                # Add helper include directories conditionally
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN_INCLUDE_DIR}>
                $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
                # $<$<BOOL:${HAVE_ARMCOMPUTE}>:...> # Unlikely for CUDA build
        )

        # Set compile definitions for the object library
        target_compile_definitions(samediff_obj PUBLIC
                $<$<BOOL:${HAVE_CUDNN}>:HAVE_CUDNN=1>
                $<$<BOOL:${HAVE_ONEDNN}>:HAVE_ONEDNN=1>
                # Add other common definitions
                __CUDABLAS__=true
        )

        # Handle MSVC Runtime Library for object library if needed
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            # Ensure necessary MSVC flags are propagated if not automatic
            target_compile_options(samediff_obj PRIVATE /EHsc /bigobj)
        endif()

        # --- Create Final Library (Shared/Static) ---
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
            if(NOT SD_SHARED_LIB) # If only static is built, use the main name
                set(FINAL_TARGET ${SD_LIBRARY_NAME}_static) # Use static target as primary
            endif()
            message(STATUS "Creating static library: ${SD_LIBRARY_NAME}_static")
            if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
            endif()
            # Link flatbuffers to static lib (needed if users link only static)
            target_link_libraries(${SD_LIBRARY_NAME}_static PUBLIC flatbuffers)
            install(TARGETS ${SD_LIBRARY_NAME}_static ARCHIVE DESTINATION lib)
        endif()

        if(NOT DEFINED FINAL_TARGET)
            message(FATAL_ERROR "No library type selected (SD_SHARED_LIB or SD_STATIC_LIB must be ON)")
        endif()

        # --- Link Libraries to Final Target ---
        target_link_libraries(${FINAL_TARGET} PUBLIC
                # CUDA runtime and core libraries (CMake target ${CUDA_LIBRARIES} usually includes cudart)
                CUDA::cudart # Use CMake imported target if available (CMake >= 3.10)
                # CUDA::cuda_driver # If needed
                # CUDA Toolkit libraries (cuBLAS, cuSolver)
                ${CUDAToolkit_LIBRARIES} # From find_package(CUDAToolkit) or manual set
                # Helper libraries (linked conditionally)
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN}> # Link CUDNN library variable
                $<$<BOOL:${HAVE_ONEDNN}>:onednn> # Link oneDNN imported target
                # Other dependencies
                Threads::Threads # Standard CMake threads target
                flatbuffers      # Flatbuffers imported target
                # Add sanitizer runtime if needed
                $<$<BOOL:${SD_SANITIZE}>:asan> # Example for Address Sanitizer (might need different name)
        )
        if (UNIX AND NOT APPLE)
            target_link_libraries(${FINAL_TARGET} PUBLIC dl) # Link dl library for dynamic loading if needed
        endif()

        # Install shared library and potentially runtime components
        if(SD_SHARED_LIB)
            install(TARGETS ${SD_LIBRARY_NAME}
                    LIBRARY DESTINATION lib
                    RUNTIME DESTINATION bin # Install DLLs/SOs to bin on Windows/Linux if applicable
                    ARCHIVE DESTINATION lib # Install import lib on Windows
            )
        endif()

        message(STATUS "CUDA Build Configuration Complete.")

    else()
        message(FATAL_ERROR "CUDA was enabled (SD_CUDA=ON) but not found by find_package(CUDA).")
    endif()

    # CPU Configuration Block
elseif(SD_CPU) # Correct syntax: elseif
    message(STATUS "--- Configuring CPU Build ---")
    add_definitions(-D__CPUBLAS__=true)

    # --- Generate CPU Source Files (Templates, etc.) ---
    message(STATUS "Configuring CPU source file generation...")
    set(CPU_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated")
    file(MAKE_DIRECTORY ${CPU_GENERATED_DIR})
    set(CPU_COMB_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated_combinations")
    file(MAKE_DIRECTORY ${CPU_COMB_GENERATED_DIR})

    # (Include/Define genCompilation and genPartitionCombination here if needed)
    # Placeholder for actual generation logic
    # file(GLOB CPU_COMPILATION_UNITS_IN include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in ...)
    # foreach(FL_ITEM ${CPU_COMPILATION_UNITS_IN})
    #     genCompilation(${FL_ITEM} CPU_GENERATED_SOURCES) # Appends to CPU_GENERATED_SOURCES
    # endforeach()
    # ... similar logic for combination templates using genPartitionCombination ...
    message(STATUS "CPU source generation configured (Actual generation happens at build time).")


    # --- Gather ALL CPU Source Files ---
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
            # Exclude CUDA specific implementations
            EXCLUDE REGEX ".*/cuda/.*\\.cu$"
            EXCLUDE REGEX ".*/cuda/.*\\.h$"
            EXCLUDE REGEX ".*/cudnn/.*\\.cu$" # Exclude platform specifics not for CPU
            EXCLUDE REGEX ".*/armcompute/.*\\.cpp$"
    )

    file(GLOB_RECURSE CPU_SPECIFIC_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cpu/*.cpp" # If any exist
            "${CMAKE_CURRENT_SOURCE_DIR}/include/memory/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cpu/*.cpp" # If any exist
            "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/*.h"
            # Include generated CPU sources placeholders
            # "${CPU_GENERATED_DIR}/*.cpp"
            # "${CPU_COMB_GENERATED_DIR}/*.cpp"
    )

    # Platform specific sources for CPU helpers (oneDNN, ARMCompute)
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
            ${CPU_GENERATED_SOURCES}     # Add generated list
            ${CUSTOMOPS_GENERIC_SOURCES} # Add generated combinations list
            ${CUSTOMOPS_ONEDNN_SOURCES}  # Add platform specifics
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
    )

    # Source file specific flags (Example: Disable optimizations for specific files)
    if (SD_X86_BUILD AND NOT WIN32)
        # Check if files exist before setting properties
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp" PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp" PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
    endif()

    # Filter sources based on selective ops if needed (example logic)
    if(NOT SD_ALL_OPS)
        message(STATUS "Filtering sources based on selective ops list...")
        # (Add logic here to parse files for markers like NOT_EXCLUDED(OP_XXX) and filter ALL_SOURCES)
        # This logic can be complex and depends on markers in source files.
    endif()

    # Create Object Library
    add_library(samediff_obj OBJECT ${ALL_SOURCES})
    # --- ADDED DEPENDENCY ---
    # Ensure flatbuffers external project is built before compiling objects
    add_dependencies(samediff_obj flatbuffers_external)
    # --- END ADDED DEPENDENCY ---
    target_compile_features(samediff_obj PUBLIC cxx_std_17) # Ensure C++ standard

    # Set include directories for the object library
    target_include_directories(samediff_obj PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include" # Source include
            "${CMAKE_CURRENT_BINARY_DIR}/include" # Generated config.h include
            ${FLATBUFFERS_INCLUDE_DIR}            # Flatbuffers include
            # Add helper include directories conditionally
            $<$<BOOL:${HAVE_OPENBLAS}>:${OPENBLAS_PATH}/include> # If path was used
            $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_INCLUDE_DIR_LIB}> # ARM Compute includes
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_INCLUDE_DIR_CORE}>
    )

    # Set compile definitions for the object library
    target_compile_definitions(samediff_obj PUBLIC
            $<$<BOOL:${HAVE_OPENBLAS}>:HAVE_OPENBLAS=1>
            $<$<BOOL:${HAVE_ONEDNN}>:HAVE_ONEDNN=1>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:HAVE_ARMCOMPUTE=1>
            # Add other common definitions
            __CPUBLAS__=true
    )

    # Handle MSVC Runtime Library for object library
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_property(TARGET samediff_obj PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        target_compile_options(samediff_obj PRIVATE /EHsc /bigobj)
    endif()


    # --- Create Final Library (Shared/Static) ---
    if(SD_SHARED_LIB)
        add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:samediff_obj>)
        set(FINAL_TARGET ${SD_LIBRARY_NAME})
        message(STATUS "Creating shared library: ${SD_LIBRARY_NAME}")
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set_property(TARGET ${FINAL_TARGET} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        # Set SONAME and VERSION for shared libraries on Linux/macOS
        if(UNIX OR APPLE)
            set_target_properties(${FINAL_TARGET} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1) # Adjust SOVERSION as needed
        endif()
    endif()

    if(SD_STATIC_LIB)
        add_library(${SD_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:samediff_obj>)
        if(NOT SD_SHARED_LIB) # If only static is built
            set(FINAL_TARGET ${SD_LIBRARY_NAME}_static)
        endif()
        message(STATUS "Creating static library: ${SD_LIBRARY_NAME}_static")
        if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set_property(TARGET ${SD_LIBRARY_NAME}_static PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")
        endif()
        # Link flatbuffers to static lib
        target_link_libraries(${SD_LIBRARY_NAME}_static PUBLIC flatbuffers)
        install(TARGETS ${SD_LIBRARY_NAME}_static ARCHIVE DESTINATION lib)
    endif()

    if(NOT DEFINED FINAL_TARGET)
        message(FATAL_ERROR "No library type selected (SD_SHARED_LIB or SD_STATIC_LIB must be ON)")
    endif()

    # --- Link Libraries to Final Target ---
    find_package(OpenMP) # Find OpenMP
    if(OPENMP_FOUND)
        message(STATUS "OpenMP found, enabling OpenMP support.")
        # Link OpenMP runtime library (target name varies by CMake version/compiler)
        set(OpenMP_LINK_LIBRARIES OpenMP::OpenMP_CXX) # CMake 3.9+ standard target
    else()
        message(STATUS "OpenMP not found.")
        set(OpenMP_LINK_LIBRARIES "")
    endif()

    target_link_libraries(${FINAL_TARGET} PUBLIC
            # BLAS libraries (found earlier)
            ${OPENBLAS_LIBRARIES} # Contains BLAS_LIBRARIES if found via find_package(BLAS) or specific OpenBLAS lib
            # Helper libraries (linked conditionally)
            $<$<BOOL:${HAVE_ONEDNN}>:onednn> # Link oneDNN imported target
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_LIBRARIES}> # Link ARMCompute targets
            # Other dependencies
            Threads::Threads
            flatbuffers
            ${OpenMP_LINK_LIBRARIES} # Link OpenMP if found
            # Add sanitizer runtime if needed
            $<$<BOOL:${SD_SANITIZE}>:asan> # Example for Address Sanitizer
    )
    if (UNIX AND NOT APPLE)
        target_link_libraries(${FINAL_TARGET} PUBLIC dl)
    endif()
    # MKL linking needs specific handling if find_package(BLAS) found MKL

    # Install shared library
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
endif() # End SD_CUDA / SD_CPU block


# ===========================================================================
# End of Content Pasted Verbatim from blas/CMakeLists.txt
# ===========================================================================


# Preprocessing Step (Optional, controlled by SD_PREPROCESS option)
if(SD_PREPROCESS)
    message(STATUS "--- Configuring Preprocessing Step ---")
    # This section assumes ALL_SOURCES has been populated by either the CUDA or CPU block
    if(NOT ALL_SOURCES)
        message(FATAL_ERROR "SD_PREPROCESS is ON, but the ALL_SOURCES list is empty. Ensure CUDA or CPU configuration ran first.")
    endif()

    list(REMOVE_DUPLICATES ALL_SOURCES)

    set(PREPROCESSED_DIR "${CMAKE_BINARY_DIR}/preprocessed") # Output within build dir
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})
    message(STATUS "Preprocessing output directory: ${PREPROCESSED_DIR}")

    set(PREPROCESSED_FILES "") # List to hold generated .i files
    set(PROCESSED_SOURCES_SET "") # Track processed sources to avoid duplicates

    # Get compile info from the object library target
    get_target_property(PREPROC_INCLUDES samediff_obj INCLUDE_DIRECTORIES)
    get_target_property(PREPROC_DEFINES samediff_obj COMPILE_DEFINITIONS)
    get_target_property(PREPROC_OPTIONS samediff_obj COMPILE_OPTIONS)

    # Convert properties to command line flags
    set(PREPROC_INCLUDE_STR "")
    if(PREPROC_INCLUDES)
        foreach(dir ${PREPROC_INCLUDES})
            if(IS_DIRECTORY "${dir}") # Check if directory exists
                list(APPEND PREPROC_INCLUDE_STR "-I${dir}")
            endif()
        endforeach()
    endif()

    set(PREPROC_DEFINE_STR "")
    if(PREPROC_DEFINES)
        foreach(d ${PREPROC_DEFINES})
            # Handle definitions with values (e.g., FOO=BAR) - needs care
            list(APPEND PREPROC_DEFINE_STR "-D${d}")
        endforeach()
    endif()

    set(PREPROC_OPTIONS_STR "")
    if(PREPROC_OPTIONS)
        # list(APPEND PREPROC_OPTIONS_STR ${PREPROC_OPTIONS}) # Append list directly
    endif()

    # Determine compiler and base flags for preprocessing
    set(preprocessor "")
    set(base_preproc_flags "")
    if(SD_CUDA AND CMAKE_CUDA_COMPILER)
        set(preprocessor ${CMAKE_CUDA_COMPILER})
        set(base_preproc_flags ${CMAKE_CUDA_FLAGS})
        # Add CUDA specific preprocessing flags if any (e.g., --preprocess)
        list(APPEND base_preproc_flags "--preprocess") # Or maybe -E is sufficient? Check nvcc docs.
    elseif(CMAKE_CXX_COMPILER) # Correct syntax: elseif
        set(preprocessor ${CMAKE_CXX_COMPILER})
        set(base_preproc_flags ${CMAKE_CXX_FLAGS})
        list(APPEND base_preproc_flags "-E") # Standard C++ preprocessor flag
    else()
        message(FATAL_ERROR "Cannot determine compiler for preprocessing.")
    endif()

    # Create Custom Commands for Each Source File
    foreach(src IN LISTS ALL_SOURCES)
        get_filename_component(src_abs ${src} ABSOLUTE) # Ensure absolute path
        # Check if source file exists and hasn't been processed yet
        if(EXISTS "${src_abs}" AND NOT "${src_abs}" IN_LIST PROCESSED_SOURCES_SET)
            # Only preprocess actual source files, not headers
            if(src_abs MATCHES "\\.(c|cpp|cxx|cc|cu)$")
                # Create a unique output filename based on relative path
                string(REPLACE "${CMAKE_SOURCE_DIR}/" "" rel_path ${src_abs})
                string(REPLACE "${CMAKE_BINARY_DIR}/" "BUILD_DIR/" rel_path ${rel_path}) # Handle generated files
                string(REPLACE "/" "_" flat_rel_path ${rel_path})
                string(REPLACE "." "_" flat_rel_path ${flat_rel_path}) # Avoid dots within name

                set(preprocessed_file "${PREPROCESSED_DIR}/${flat_rel_path}.i")
                message(STATUS "Preprocessing: ${src_abs} -> ${preprocessed_file}")

                # Construct preprocessing command arguments
                set(pp_command ${preprocessor})
                list(APPEND pp_command ${base_preproc_flags}) # Base flags (includes -E or --preprocess)
                list(APPEND pp_command ${PREPROC_OPTIONS_STR}) # Target specific compile options
                list(APPEND pp_command ${PREPROC_INCLUDE_STR}) # Include paths
                list(APPEND pp_command ${PREPROC_DEFINE_STR})  # Compile definitions
                list(APPEND pp_command "${src_abs}")           # Input source file
                list(APPEND pp_command "-o" "${preprocessed_file}") # Output file

                # Add custom command for this specific file
                add_custom_command(
                        OUTPUT ${preprocessed_file}
                        COMMAND ${pp_command}
                        DEPENDS ${src_abs} # Depends on the source file
                        COMMENT "Preprocessing ${src_abs}"
                        VERBATIM
                        # Working directory might be needed if includes are relative to build dir
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                )
                list(APPEND PREPROCESSED_FILES ${preprocessed_file}) # Add output to list
            endif()
            list(APPEND PROCESSED_SOURCES_SET "${src_abs}") # Mark as processed
        endif()
    endforeach()

    # Create a Custom Target that depends on all generated .i files
    if(PREPROCESSED_FILES)
        add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES}
                COMMENT "Running C/C++/CUDA preprocessor on sources..."
        )
        set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1) # Avoid cleaning generated files on 'clean'
    endif()
endif() # End SD_PREPROCESS


# Testing Setup (Optional - from main)
if(SD_BUILD_TESTS)
    message(STATUS "--- Configuring Tests ---")
    include(CTest)
    enable_testing()

    # Check for test subdirectory (adjust name if needed)
    set(TEST_DIR_NAME "tests") # Or tests_cpu, tests_cuda etc.
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR_NAME}")
        message(STATUS "Adding test subdirectory: ${TEST_DIR_NAME}")
        # Ensure the test subdirectory's CMakeLists links against the main library target
        # (e.g., ${SD_LIBRARY_NAME} or ${SD_LIBRARY_NAME}_static)
        add_subdirectory(${TEST_DIR_NAME})
    else()
        message(WARNING "SD_BUILD_TESTS is ON, but directory '${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIR_NAME}' was not found.")
    endif()
endif()


# Installation and Packaging (from main)
message(STATUS "--- Configuring Installation and Packaging ---")

# Set install prefix (can be overridden by user with -DCMAKE_INSTALL_PREFIX)
if(NOT CMAKE_INSTALL_PREFIX_INITIALIZED)
    set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix")
endif()
message(STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}")

# Define install destination directories relative to prefix
set(INSTALL_LIB_DIR "lib" CACHE STRING "Library installation directory")
set(INSTALL_BIN_DIR "bin" CACHE STRING "Runtime executable/DLL installation directory")
set(INSTALL_INCLUDE_DIR "include/${PROJECT_NAME}" CACHE STRING "Header file installation directory")

# Install header files (public headers)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" # Trailing slash is important
        DESTINATION ${INSTALL_INCLUDE_DIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl"
        # Exclude internal implementation headers if necessary
        PATTERN "impl" EXCLUDE
        PATTERN "cpu" EXCLUDE
        PATTERN "cuda" EXCLUDE
        PATTERN "generated" EXCLUDE # Exclude generated headers from source install
        PATTERN "platform" EXCLUDE
)

# Install generated definition header if created
if(DEFINED OP_OUTPUT_FILE_ABS AND EXISTS "${OP_OUTPUT_FILE_ABS}")
    install(FILES ${OP_OUTPUT_FILE_ABS} DESTINATION ${INSTALL_INCLUDE_DIR})
endif()
# Install configured header (config.h)
install(FILES "${CMAKE_BINARY_DIR}/include/config.h" DESTINATION ${INSTALL_INCLUDE_DIR})
# Install generated flatbuffers header if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd/graph_generated.h")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated/sd/graph_generated.h"
            DESTINATION ${INSTALL_INCLUDE_DIR}/graph/generated/sd) # Keep subdirectory structure
endif()

# Library installation commands are already included within the SD_CUDA/SD_CPU blocks


# CPack Configuration (from main)
include(CPack) # Include CPack module first

# General package information
set(CPACK_PACKAGE_NAME "libnd4j")
set(CPACK_PACKAGE_VENDOR "Eclipse Foundation")
set(CPACK_PACKAGE_CONTACT "Eclipse Deeplearning4j Committers <dl4j-dev@eclipse.org>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native operations library for Eclipse Deeplearning4j")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/eclipse/deeplearning4j")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Version information (use project version if set, otherwise default)
if(PROJECT_VERSION)
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
else()
    set(CPACK_PACKAGE_VERSION "1.0.0-SNAPSHOT") # Fallback version
endif()
# Extract MAJOR.MINOR.PATCH for generators that need it
if(${CPACK_PACKAGE_VERSION} MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
    set(CPACK_PACKAGE_VERSION_PATCH ${CMAKE_MATCH_3})
else()
    # Handle non-standard versions if necessary
    set(CPACK_PACKAGE_VERSION_MAJOR 0)
    set(CPACK_PACKAGE_VERSION_MINOR 0)
    set(CPACK_PACKAGE_VERSION_PATCH 0)
endif()
# Package file name base (Generator specific parts added below)
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")


# Generator specific settings
if(UNIX)
    find_program(LSB_RELEASE_EXEC lsb_release)
    if(LSB_RELEASE_EXEC)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -si OUTPUT_VARIABLE DISTRIBUTION OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -sr OUTPUT_VARIABLE RELEASE OUTPUT_STRIP_TRAILING_WHITESPACE) # Use -sr for release number
        execute_process(COMMAND ${LSB_RELEASE_EXEC} -sc OUTPUT_VARIABLE CODENAME OUTPUT_STRIP_TRAILING_WHITESPACE)
    elseif(EXISTS "/etc/os-release") # Correct syntax: elseif
        # Fallback using /etc/os-release
        file(STRINGS "/etc/os-release" OS_INFO REGEX "^(ID|VERSION_ID|VERSION_CODENAME)=")
        foreach(info ${OS_INFO})
            if(info MATCHES "^ID=(.*)")
                set(DISTRIBUTION ${CMAKE_MATCH_1})
                string(STRIP ${DISTRIBUTION} DISTRIBUTION) # Remove potential quotes
            elseif(info MATCHES "^VERSION_ID=(.*)") # Correct syntax: elseif
                set(RELEASE ${CMAKE_MATCH_1})
                string(STRIP ${RELEASE} RELEASE)
            elseif(info MATCHES "^VERSION_CODENAME=(.*)") # Correct syntax: elseif
                set(CODENAME ${CMAKE_MATCH_1})
                string(STRIP ${CODENAME} CODENAME)
            endif()
        endforeach()
    endif()
    # Normalize distribution name if needed (e.g., "centos" -> "CentOS")

    # Determine architecture naming
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(ARCHITECTURE "amd64")
        set(RPM_ARCHITECTURE "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") # Correct syntax: elseif
        set(ARCHITECTURE "arm64")
        set(RPM_ARCHITECTURE "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7") # Correct syntax: elseif
        set(ARCHITECTURE "armhf")
        set(RPM_ARCHITECTURE "armv7hl")
    else()
        set(ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
        set(RPM_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
    endif()

    # Set default generator
    set(CPACK_GENERATOR "TGZ") # Default fallback

    if(DISTRIBUTION MATCHES "Ubuntu|Debian")
        set(CPACK_GENERATOR "DEB")
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}_${ARCHITECTURE}") # Debian naming convention
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ARCHITECTURE})
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
        set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
        set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
        # Define dependencies (Example - adjust based on actual needs)
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.17)") # Basic C library dependency
        if(SD_CUDA)
            # Example: Depends on CUDA runtime, adjust version match as needed
            # Find CUDA version parts if available (from CUDA_VERSION variable)
            if(CUDA_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
                set(CUDA_MAJOR ${CMAKE_MATCH_1})
                set(CUDA_MINOR ${CMAKE_MATCH_2})
                list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS " cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}")
            endif()
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS " libcudnn8") # Example, check actual package name/version
        endif()
        # Convert list to string
        string(REPLACE ";" ", " CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}")


        # Postinst/Postrm scripts (optional)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb")
            set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.deb;${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.deb")
        endif()

    elseif(DISTRIBUTION MATCHES "CentOS|Fedora|RHEL|Rocky|AlmaLinux") # Correct syntax: elseif
        set(CPACK_GENERATOR "RPM")
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${RPM_ARCHITECTURE}") # RPM naming convention
        set(CPACK_RPM_PACKAGE_ARCHITECTURE ${RPM_ARCHITECTURE})
        set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
        set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries")
        set(CPACK_RPM_PACKAGE_URL ${CPACK_PACKAGE_HOMEPAGE_URL})
        set(CPACK_RPM_PACKAGE_VENDOR ${CPACK_PACKAGE_VENDOR})
        # Define dependencies (Example)
        set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.17")
        if(SD_CUDA)
            if(CUDA_VERSION MATCHES "^([0-9]+)\\.([0-9]+)")
                set(CUDA_MAJOR ${CMAKE_MATCH_1})
                set(CUDA_MINOR ${CMAKE_MATCH_2})
                list(APPEND CPACK_RPM_PACKAGE_REQUIRES " cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}")
            endif()
        endif()
        if(HAVE_CUDNN)
            list(APPEND CPACK_RPM_PACKAGE_REQUIRES " libcudnn8") # Example
        endif()
        # Convert list to string
        string(REPLACE ";" ", " CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}")

        # Postinst/Postrm scripts (optional)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
            set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postinst.rpm")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
            set(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/postrm.rpm")
        endif()
    endif()

elseif(WIN32) # Correct syntax: elseif
    set(CPACK_GENERATOR "NSIS") # Or ZIP
    # Installation directory within prefix
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
    # NSIS specific settings
    set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME}")
    set(CPACK_NSIS_PUBLISHER "${CPACK_PACKAGE_VENDOR}")
    set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
    # Add desktop icons, start menu shortcuts etc. if needed

elseif(APPLE) # Correct syntax: elseif
    # Options: DragNDrop (simple DMG), PackageMaker (requires setup), OSXX11 (X11 apps), Bundle (macOS .app)
    set(CPACK_GENERATOR "DragNDrop") # Simple DMG generator
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-macOS")
    # Add bundle specific settings if using Bundle generator
    # set(CPACK_BUNDLE_NAME ${PROJECT_NAME})
    # set(CPACK_BUNDLE_ICON ...)
    # set(CPACK_BUNDLE_STARTUP_COMMAND ...)
endif()

message(STATUS "CPack Generator: ${CPACK_GENERATOR}")
message(STATUS "CPack Package File Name: ${CPACK_PACKAGE_FILE_NAME}")


message("--- CMake Configuration Complete ---")