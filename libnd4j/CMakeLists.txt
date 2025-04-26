cmake_minimum_required(VERSION 3.15)

# Explicitly enable CXX globally. CUDA is enabled conditionally later if needed.
project(libnd4j LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")

# <<< MERGE START: Add function definitions >>>

# ///////////////////////////////////////////////////////////////////////////////
# genCompilation: Generates cpp, cu files
# INPUT:
# $FILE_ITEM template-configuration that utilizes libnd4j type, macros helpers
# defined inside { include/types/types.h, include/system/type_boilerplate.h}
# OUTPUT:
# $CUSTOMOPS_GENERIC_SOURCES/$CUDA_GENERATED_SOURCES generated files will be added into the appropriate list
# ///////////////////////////////////////////////////////////////////////////////
function(genCompilation FILE_ITEM)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)

    set(EXTENSION "cpp")
    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)
    #check content for types

    #set all to false
    set (SD_FLOAT_TYPES_GEN     0)
    set (SD_INTEGER_TYPES_GEN   0)
    set (SD_COMMON_TYPES_GEN    0)
    set (SD_PAIRWISE_TYPES_GEN  0)
    set (RANGE_STOP         -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    #STOP=COUNT(SD_COMMON_TYPES)-1
    set(SD_INTEGER_TYPES_END 7)  # Indices 0-7
    set(SD_COMMON_TYPES_END 12)  # Indices 0-12
    set(SD_FLOAT_TYPES_END 3)    # Indices 0-3
    set(SD_PAIRWISE_TYPES_END 12) # Indices 0-12

    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set (SD_INTEGER_TYPES_GEN  1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set (SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set (SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set (SD_PAIRWISE_TYPES_GEN  1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()
    endforeach()

    if(RANGE_STOP GREATER -1)
        #message(STATUS "Generating ${FILE_ITEM_WE} for index range 0 to ${RANGE_STOP}") # Reduce verbosity
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            # set OFF if the index is above the specific type group's end index
            set(TEMP_SD_FLOAT_TYPES_GEN ${SD_FLOAT_TYPES_GEN})
            set(TEMP_SD_INTEGER_TYPES_GEN ${SD_INTEGER_TYPES_GEN})
            set(TEMP_SD_COMMON_TYPES_GEN ${SD_COMMON_TYPES_GEN})
            set(TEMP_SD_PAIRWISE_TYPES_GEN ${SD_PAIRWISE_TYPES_GEN})

            if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                set (TEMP_SD_FLOAT_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                set (TEMP_SD_INTEGER_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                set (TEMP_SD_COMMON_TYPES_GEN 0)
            endif()
            if(FL_TYPE_INDEX GREATER ${SD_PAIRWISE_TYPES_END})
                set (TEMP_SD_PAIRWISE_TYPES_GEN 0)
            endif()

            # Use temporary variables for configure_file
            set(SD_FLOAT_TYPES_GEN ${TEMP_SD_FLOAT_TYPES_GEN})
            set(SD_INTEGER_TYPES_GEN ${TEMP_SD_INTEGER_TYPES_GEN})
            set(SD_COMMON_TYPES_GEN ${TEMP_SD_COMMON_TYPES_GEN})
            set(SD_PAIRWISE_TYPES_GEN ${TEMP_SD_PAIRWISE_TYPES_GEN})

            set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            # Ensure the output directory exists before trying to configure the file
            get_filename_component(GENERATED_SOURCE_DIR ${GENERATED_SOURCE} DIRECTORY)
            file(MAKE_DIRECTORY ${GENERATED_SOURCE_DIR})

            configure_file("${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)

            # Append to the correct list based on extension
            if(EXTENSION STREQUAL "cu")
                list(APPEND CUDA_GENERATED_SOURCES ${GENERATED_SOURCE})
            else()
                # This was CUSTOMOPS_GENERIC_SOURCES before, check if CPU_GENERATED_SOURCES is more appropriate
                list(APPEND CPU_GENERATED_SOURCES ${GENERATED_SOURCE})
            endif()

            # Restore original defines for next loop iteration (important!)
            # Use the original values, not the temporary ones
            set(SD_FLOAT_TYPES_GEN ${SD_FLOAT_TYPES_GEN})
            set(SD_INTEGER_TYPES_GEN ${SD_INTEGER_TYPES_GEN})
            set(SD_COMMON_TYPES_GEN ${SD_COMMON_TYPES_GEN})
            set(SD_PAIRWISE_TYPES_GEN ${SD_PAIRWISE_TYPES_GEN})
        endforeach()
    else()
        message(STATUS "Skipping generation for ${FILE_ITEM_WE} as no relevant TYPE_GEN directives found.")
    endif()

    # Propagate lists back to parent scope
    set(CPU_GENERATED_SOURCES ${CPU_GENERATED_SOURCES} PARENT_SCOPE) # Renamed list for clarity
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
endfunction()

# Define a function to generate individual CUDA files for each type combination
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
    # Updated Regex to handle potential namespaces before class name and whitespace
    string(REGEX MATCH "template[ \n\t]+void[ \n\t]+([a-zA-Z0-9_:]+)::([a-zA-Z0-9_]+)<" FUNCTION_MATCH "${TEMPLATE_CONTENT}")
    set(CLASS_NAME ${CMAKE_MATCH_1})
    set(METHOD_NAME ${CMAKE_MATCH_2})

    # Check if regex matched successfully
    if(NOT CLASS_NAME OR NOT METHOD_NAME)
        message(WARNING "Could not extract Class::Method from template ${TEMPLATE_FILE}. Using fallback naming.")
        # Fallback naming if regex fails
        set(CLASS_NAME "UnknownClass")
        set(METHOD_NAME "UnknownMethod")
    endif()

    # Clean class name for file naming
    string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")

    # Extract function signature arguments to create a unique identifier
    # Updated Regex to be more robust against whitespace and pointer/reference symbols
    string(REGEX MATCH "::${METHOD_NAME}<.*>\\(([^)]*)\\);" FUNC_ARGS_MATCH "${TEMPLATE_CONTENT}")
    set(FUNCTION_ARGS_RAW "${CMAKE_MATCH_1}")
    # Clean up args string: remove extra spaces, const, &, * for identifier generation
    string(REGEX REPLACE "[ \t\n]+" " " FUNCTION_ARGS_CLEANED "${FUNCTION_ARGS_RAW}") # Normalize spaces
    string(REGEX REPLACE "const " "" FUNCTION_ARGS_CLEANED "${FUNCTION_ARGS_CLEANED}") # Remove const
    string(REGEX REPLACE "[&*]" "" FUNCTION_ARGS_CLEANED "${FUNCTION_ARGS_CLEANED}") # Remove &, *
    string(REGEX REPLACE "," "_" FUNCTION_ARGS_CLEANED "${FUNCTION_ARGS_CLEANED}") # Replace comma with underscore
    string(REGEX REPLACE "[^a-zA-Z0-9_]" "" FUNCTION_ARGS_CLEANED "${FUNCTION_ARGS_CLEANED}") # Remove remaining non-alphanumeric

    # Create a signature identifier
    set(SIGNATURE_ID "_sig${FUNCTION_ARGS_CLEANED}")

    # Create a shorter hash if the signature identifier is too long (potential filesystem limits)
    string(LENGTH "${SIGNATURE_ID}" SIG_ID_LEN)
    if(SIG_ID_LEN GREATER 50) # Use a reasonable limit
        string(MD5 SIGNATURE_HASH "${SIGNATURE_ID}")
        string(SUBSTRING "${SIGNATURE_HASH}" 0 12 SIGNATURE_ID_SHORT) # Use 12 chars of MD5
        set(SIGNATURE_ID "_h${SIGNATURE_ID_SHORT}")
    endif()


    # Output filename with signature identifier
    set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}${SIGNATURE_ID}_${COMB1}_${COMB2}_${COMB3}.cu")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    # Replace placeholders with actual values
    # Using @ placeholder as assumed by the configure_file command implicitly used by older file's genCompilation
    string(REPLACE "@COMB1@" "${COMB1}" MODIFIED_CONTENT "${TEMPLATE_CONTENT}")
    string(REPLACE "@COMB2@" "${COMB2}" MODIFIED_CONTENT "${MODIFIED_CONTENT}")
    string(REPLACE "@COMB3@" "${COMB3}" MODIFIED_CONTENT "${MODIFIED_CONTENT}")

    # Remove the iteration macro itself from the final file
    # Use regex to remove lines containing ITERATE_COMBINATIONS_3 or similar
    string(REGEX REPLACE "[^\n]*ITERATE_COMBINATIONS_3[^\n]*\n?" "" MODIFIED_CONTENT "${MODIFIED_CONTENT}")
    # Add removal for other potential iteration macros if needed

    # Create directory if needed
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # Write the content
    file(WRITE "${GENERATED_FILE}" "${MODIFIED_CONTENT}")

    # Set properties and add to sources
    set_source_files_properties("${GENERATED_FILE}" PROPERTIES LANGUAGE CUDA)
    list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)

    #message(STATUS "Generated CUDA single function: ${GENERATED_FILE}") # Keep this less verbose unless debugging
endfunction()

# <<< MERGE END: Add function definitions >>>


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

# <<< FIX: Enable CUDA Language if SD_CUDA is TRUE >>>
if(SD_CUDA)
    message(STATUS "Enabling CUDA language for the project.")
    # This ensures CMake knows how to handle .cu files later
    enable_language(CUDA)
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


# Set DEFAULT_ENGINE based on build type
if(SD_CUDA)
    set(DEFAULT_ENGINE "samediff::ENGINE_CUDA")
else()
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")
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
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> # Build command
        INSTALL_COMMAND ""             # We don't need to install it system-wide
        UPDATE_COMMAND ""              # Avoid automatic updates unless needed
        LOG_CONFIGURE TRUE
        LOG_BUILD TRUE
        LOG_DOWNLOAD TRUE
)

# Get include directory property from the external project AFTER it's configured/built
set(FLATBUFFERS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/flatbuffers-src/include)
include_directories(${FLATBUFFERS_INCLUDE_DIR})
message(STATUS "Flatbuffers include directory set to: ${FLATBUFFERS_INCLUDE_DIR}")

# Define imported targets AFTER ExternalProject_Add
add_library(flatbuffers STATIC IMPORTED GLOBAL)
set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/libflatbuffers.a) # Assume Linux/macOS name for now
if(WIN32)
    set_property(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/flatbuffers-build/$<IF:$<CONFIG:Debug>,Debug,Release>/flatbuffers.lib) # Example for MSVC Debug/Release dirs
endif()
add_dependencies(flatbuffers flatbuffers_external) # Ensure library depends on the external build

# Add flatc executable target if built
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


# ================== Generate Flatbuffers Sources Section ==================
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
            -o ${FLATC_OUTPUT_DIR_CPP} --cpp --gen-object-api
            -o ${FLATC_OUTPUT_DIR_JAVA} --java --gen-object-api
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
add_custom_target(generate_config DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# ===========================================================================
# Start of Build Logic Section (Modified for Merged Functionality)
# ===========================================================================

# Architecture Tuning Flags
set(ARCH_TUNE "") # Initialize
if(SD_ARCH) # Check if SD_ARCH is set
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


# Compiler-Specific Flags (Additions)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(NOT SD_CUDA)
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # Use C++17 from top
    endif()
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(UNIX AND NOT APPLE AND NOT SD_IOS_BUILD AND SD_SHARED_LIB)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'$ORIGIN'")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc")
endif()


# External Include Directories
set(EXTERNAL_INCLUDE_DIRS "")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    # list(APPEND EXTERNAL_INCLUDE_DIRS "/usr/include" "/usr/local/include") # Usually not needed
endif()


# Set SD_LIBRARY_NAME Based on Build Type
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


# Initialize source lists
set(ALL_SOURCES "")
set(CUSTOMOPS_ONEDNN_SOURCES "")
set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
set(CUSTOMOPS_CUDNN_SOURCES "")
set(CUDA_GENERATED_SOURCES "")
set(CPU_GENERATED_SOURCES "")
set(CUSTOMOPS_GENERIC_SOURCES "") # CPU generated combinations from templates


# CUDA Configuration Block
if(SD_CUDA)
    message(STATUS "--- Configuring CUDA Build ---")
    add_definitions(-D__CUDABLAS__=true SD_CUDA=true)

    # Find CUDA package (already enabled, but find components)
    if(NOT CUDA_FOUND)
        find_package(CUDA REQUIRED) # Should already be found
    endif()

    if(CUDA_FOUND)
        message(STATUS "CUDA Found: Version=${CUDA_VERSION}, Toolkit Root=${CUDA_TOOLKIT_ROOT_DIR}")
        include_directories(${CUDA_INCLUDE_DIRS})

        # Common CUDA flags
        set(COMMON_CUDA_FLAGS "-allow-unsupported-compiler --expt-extended-lambda -Xfatbin -compress-all -maxrregcount=128")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${COMMON_CUDA_FLAGS}")

        # Platform specific flags
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
        endif()
        # Debug/Release specific CUDA flags
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")
        # CUDA Architectures
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

        # Sanitize flags for CUDA
        if(SD_SANITIZE AND SANITIZE_FLAGS)
            message(STATUS "Adding sanitizer flags to CUDA host/device compiler via -Xcompiler")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"${SANITIZE_FLAGS}\"")
        endif()

        # Find CUDA Libraries (cuBLAS, cuSolver)
        find_package(CUDAToolkit COMPONENTS cublas cusolver)
        if(NOT CUDAToolkit_FOUND)
            message(WARNING "Could not find required CUDA Toolkit components (cuBLAS, cuSolver) via find_package(CUDAToolkit). Linking might fail.")
            find_library(CUDA_cublas_LIBRARY NAMES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
            find_library(CUDA_cusolver_LIBRARY NAMES cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
            if(NOT CUDA_cublas_LIBRARY OR NOT CUDA_cusolver_LIBRARY)
                message(FATAL_ERROR "Manual search for cuBLAS/cuSolver library failed.")
            endif()
            set(CUDAToolkit_LIBRARIES ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})
        endif()


        # --- Generate CUDA Source Files (Templates, etc.) ---
        message(STATUS "Configuring CUDA source file generation...")
        set(CUDA_GENERATED_DIR "${CMAKE_BINARY_DIR}/cuda_generated")
        file(MAKE_DIRECTORY ${CUDA_GENERATED_DIR})
        set(CUDA_GENERATED_SOURCES "") # Ensure list is initialized

        # Determine NUM_TYPES based on SD_TYPES_LIST
        if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST_COUNT GREATER 0)
            set(NUM_TYPES ${SD_TYPES_LIST_COUNT})
            message(STATUS "Using ${NUM_TYPES} types from SD_TYPES_LIST for CUDA generation")
        else()
            set(NUM_TYPES 10)
            message(STATUS "Using default number of types for CUDA generation: ${NUM_TYPES}")
        endif()

        # Process compilation_units templates
        file(GLOB CUDA_COMPILATION_UNITS "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/compilation_units/*.cu.in")
        foreach(FL_ITEM ${CUDA_COMPILATION_UNITS})
            genCompilation(${FL_ITEM}) # Uses function defined above
        endforeach()

        # Define explicit combinations (copied from previous sample file)
        set(COMBINATIONS_3
                "0,0,0" "0,0,2" "0,1,2" "0,2,0" "0,2,1" "0,2,2"
                "1,0,0" "1,1,1" "1,1,2" "1,2,0" "1,2,1" "1,2,2"
                "2,0,0" "2,0,1" "2,0,2" "2,1,0" "2,1,1" "2,1,2"
                "2,2,0" "2,2,1" "2,2,2"
        )
        message(STATUS "Using explicit list for CUDA type combinations")

        # Process comb_compilation_units templates
        file(GLOB CUDA_TEMPLATE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/comb_compilation_units/*.cu.in")
        foreach(TEMPLATE_FILE ${CUDA_TEMPLATE_FILES})
            foreach(COMBINATION ${COMBINATIONS_3})
                genSingleFunctionCuda(${TEMPLATE_FILE} ${COMBINATION} "${CUDA_GENERATED_DIR}") # Uses function defined above
            endforeach()
        endforeach()
        message(STATUS "Finished CUDA source generation.")
        message(STATUS "DEBUG: Generated CUDA Sources List: ${CUDA_GENERATED_SOURCES}")


        # --- Gather ALL CUDA Source Files ---
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
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/array/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/execution/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cuda/*.cu"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cuda/*.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/*.cu"
        )
        message(STATUS "DEBUG: CUDA Specific Sources (Combined GLOB): ${CUDA_SPECIFIC_SOURCES}")

        # Platform specific sources (cuDNN)
        if(HAVE_CUDNN)
            message(STATUS "Adding cuDNN platform sources")
            file(GLOB_RECURSE CUSTOMOPS_CUDNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/platform/cudnn/*.cu")
        endif()

        # Combine all sources
        set(ALL_SOURCES
                ${COMMON_SOURCES}
                ${CUDA_SPECIFIC_SOURCES}    # Combined manual + specific CUDA sources
                ${CUDA_GENERATED_SOURCES}   # Generated CUDA sources
                ${CUSTOMOPS_CUDNN_SOURCES}  # Conditional cuDNN sources
        )
        list(REMOVE_DUPLICATES ALL_SOURCES)
        message(STATUS "DEBUG: Final ALL_SOURCES for CUDA build: ${ALL_SOURCES}")

        # Create Object Library
        add_library(samediff_obj OBJECT ${ALL_SOURCES})
        add_dependencies(samediff_obj generate_config)
        add_dependencies(samediff_obj flatbuffers_external)
        target_compile_features(samediff_obj PUBLIC cxx_std_17 cuda_std_17)

        # Set include directories for the object library
        target_include_directories(samediff_obj PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_BINARY_DIR}/include"
                ${FLATBUFFERS_INCLUDE_DIR}
                ${CUDA_INCLUDE_DIRS}
                $<$<BOOL:${HAVE_CUDNN}>:${CUDNN_INCLUDE_DIR}>
                $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
        )

        # Set compile definitions for the object library
        target_compile_definitions(samediff_obj PUBLIC
                $<$<BOOL:${HAVE_CUDNN}>:HAVE_CUDNN=1>
                $<$<BOOL:${HAVE_ONEDNN}>:HAVE_ONEDNN=1>
                __CUDABLAS__=true
                SD_CUDA=true
        )

        # Handle MSVC Runtime Library for object library if needed
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
            # Add PROJECT_VERSION variable definition if not already present
            if(NOT DEFINED PROJECT_VERSION)
                set(PROJECT_VERSION "1.0.0-SNAPSHOT") # Or get from elsewhere
            endif()
            if(UNIX OR APPLE)
                set_target_properties(${FINAL_TARGET} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1) # Adjust SOVERSION as needed
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

        # --- Link Libraries to Final Target ---
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

        # Install shared library
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

    # CPU Configuration Block
elseif(SD_CPU)
    message(STATUS "--- Configuring CPU Build ---")
    add_definitions(-D__CPUBLAS__=true SD_CPU=true)

    # --- Generate CPU Source Files (Templates, etc.) ---
    message(STATUS "Configuring CPU source file generation...")
    # Assume CPU might also use genCompilation for .cpp.in files
    # Initialize list for generated CPU sources
    set(CPU_GENERATED_SOURCES "")
    set(CUSTOMOPS_GENERIC_SOURCES "") # Also used by CPU gen? Check template content

    set(CPU_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated") # Different dir? or use compilation_units?
    file(MAKE_DIRECTORY ${CPU_GENERATED_DIR})
    set(CPU_COMB_GENERATED_DIR "${CMAKE_BINARY_DIR}/cpu_generated_combinations")
    file(MAKE_DIRECTORY ${CPU_COMB_GENERATED_DIR})

    # Process compilation_units templates for CPU
    file(GLOB CPU_COMPILATION_UNITS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/compilation_units/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/loops/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/*.cpp.in"
    )
    foreach(FL_ITEM ${CPU_COMPILATION_UNITS})
        genCompilation(${FL_ITEM}) # Uses function defined above, appends to CPU_GENERATED_SOURCES
    endforeach()

    # Process combination templates for CPU (if any - e.g., pairwise_instantiation)
    # The previous file had templates like pairwise_instantiation_template_2/3.cpp.in
    # Assume these might use genCompilation or a similar mechanism.
    # For now, we only include the genCompilation calls above.
    # Add calls to genPartitionCombination if needed, definition needs to be added.

    message(STATUS "CPU source generation configured (Actual generation happens at build time).")
    message(STATUS "DEBUG: Generated CPU Sources List: ${CPU_GENERATED_SOURCES}")


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
            EXCLUDE REGEX ".*/armcompute/.*\\.cpp$" # Excluded if helper not enabled below
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
    else()
        # Exclude armcompute sources if helper not enabled
        list(FILTER COMMON_SOURCES EXCLUDE REGEX ".*/armcompute/.*\\.cpp$")
    endif()

    set(ALL_SOURCES
            ${COMMON_SOURCES}
            ${CPU_SPECIFIC_SOURCES}
            ${CPU_GENERATED_SOURCES}     # Add generated CPU list
            ${CUSTOMOPS_GENERIC_SOURCES} # Contains generated CPU pairwise if used
            ${CUSTOMOPS_ONEDNN_SOURCES}  # Add platform specifics
            ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
    )
    list(REMOVE_DUPLICATES ALL_SOURCES) # Clean up overlaps
    message(STATUS "DEBUG: Final ALL_SOURCES for CPU build: ${ALL_SOURCES}")

    # Source file specific flags
    if (SD_X86_BUILD AND NOT WIN32)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/include/legacy/cpu/NativeOps.cpp" PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/impl/OpTracker.cpp" PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
        endif()
    endif()

    # Filter sources based on selective ops if needed
    # Function removeFileIfExcluded needs to be defined if this is used
    # if(NOT SD_ALL_OPS)
    #    message(STATUS "Filtering sources based on selective ops list...")
    #    # foreach(category IN ITEMS ALL_SOURCES) # Simplified loop
    #    #    removeFileIfExcluded(FILE_ITEM "${category}" LIST_ITEM ALL_SOURCES)
    #    # endforeach()
    # endif()

    # Create Object Library
    add_library(samediff_obj OBJECT ${ALL_SOURCES})
    add_dependencies(samediff_obj generate_config)
    add_dependencies(samediff_obj flatbuffers_external)
    # Add dependency on generated CPU sources if applicable
    # add_custom_target(generate_cpu_sources DEPENDS ${CPU_GENERATED_SOURCES} ${CUSTOMOPS_GENERIC_SOURCES}) # Example
    # add_dependencies(samediff_obj generate_cpu_sources) # Example

    target_compile_features(samediff_obj PUBLIC cxx_std_17)

    # Set include directories for the object library
    target_include_directories(samediff_obj PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            ${FLATBUFFERS_INCLUDE_DIR}
            $<$<BOOL:${HAVE_OPENBLAS}>:${OPENBLAS_PATH}/include>
            $<$<BOOL:${HAVE_ONEDNN}>:${ONEDNN_INCLUDE_DIRS}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_INCLUDE_DIR_LIB}>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:${ARMCOMPUTE_INCLUDE_DIR_CORE}>
    )

    # Set compile definitions for the object library
    target_compile_definitions(samediff_obj PUBLIC
            $<$<BOOL:${HAVE_OPENBLAS}>:HAVE_OPENBLAS=1>
            $<$<BOOL:${HAVE_ONEDNN}>:HAVE_ONEDNN=1>
            $<$<BOOL:${HAVE_ARMCOMPUTE}>:HAVE_ARMCOMPUTE=1>
            __CPUBLAS__=true SD_CPU=true
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
        if(UNIX OR APPLE)
            # Add PROJECT_VERSION variable definition if not already present
            if(NOT DEFINED PROJECT_VERSION)
                set(PROJECT_VERSION "1.0.0-SNAPSHOT") # Or get from elsewhere
            endif()
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

    # --- Link Libraries to Final Target ---
    find_package(OpenMP)
    if(OPENMP_FOUND)
        message(STATUS "OpenMP found, enabling OpenMP support.")
        set(OpenMP_LINK_LIBRARIES OpenMP::OpenMP_CXX)
    else()
        message(STATUS "OpenMP not found.")
        set(OpenMP_LINK_LIBRARIES "")
    endif()

    target_link_libraries(${FINAL_TARGET} PUBLIC
            ${OPENBLAS_LIBRARIES} # Contains BLAS_LIBRARIES if found via find_package(BLAS) or specific OpenBLAS lib
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
# End of Build Logic Section
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
                #message(STATUS "Preprocessing: ${src_abs} -> ${preprocessed_file}") # Reduce verbosity

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
if(NOT DEFINED PROJECT_VERSION)
    set(PROJECT_VERSION "1.0.0-SNAPSHOT") # Or get from elsewhere if needed by CPack/install
endif()
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

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
        # Ensure RELEASE variable exists, provide default if not found
        if(NOT DEFINED RELEASE)
            set(RELEASE "1") # Default release number
        endif()
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}-${RELEASE}_${ARCHITECTURE}") # Debian naming convention
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
        # Ensure RELEASE variable exists, provide default if not found
        if(NOT DEFINED RELEASE)
            set(RELEASE "1") # Default release number
        endif()
        set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${RELEASE}.${RPM_ARCHITECTURE}") # RPM naming convention
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