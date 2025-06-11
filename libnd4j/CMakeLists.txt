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

# --- FIX FOR UNSUPPORTED HOST COMPILERS (ALL PLATFORMS) ---
# This must be set BEFORE the project() command to ensure the flags are used
# during CMake's initial CUDA compiler identification step.
#   --allow-unsupported-compiler: The flag for nvcc itself.
#   -D__NVCC_ALLOW_UNSUPPORTED_COMPILER__: The macro that the CUDA headers check for.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -D__NVCC_ALLOW_UNSUPPORTED_COMPILER__")
# Also add it as a compile definition to ensure it gets to the preprocessor
add_compile_definitions(__NVCC_ALLOW_UNSUPPORTED_COMPILER__)
project(libnd4j)

# Basic CMake Configuration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message("CMAKE MODULE PATH ${CMAKE_MODULE_PATH}")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Include the enhanced type validation functions
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TypeValidation.cmake)

# Add new CMake options for type validation
option(SD_STRICT_TYPE_VALIDATION "Enable strict type validation" OFF)
option(SD_DEBUG_AUTO_REDUCE "Auto-reduce types for debug builds" ON)
option(SD_SHOW_TYPE_IMPACT "Show estimated build impact" ON)

# Add debug type profile option
set(SD_DEBUG_TYPE_PROFILE "" CACHE STRING "Debug type profile (MINIMAL_INDEXING, ESSENTIAL, FLOATS_ONLY, etc.)")
set(SD_DEBUG_CUSTOM_TYPES "" CACHE STRING "Custom types for debug profile")

# Set default validation mode based on build configuration
if(NOT DEFINED SD_TYPES_VALIDATION_MODE)
    if(SD_GCC_FUNCTRACE STREQUAL "ON")
        set(SD_TYPES_VALIDATION_MODE "DEBUG")
    elseif(SD_STRICT_TYPE_VALIDATION)
        set(SD_TYPES_VALIDATION_MODE "STRICT")
    else()
        set(SD_TYPES_VALIDATION_MODE "NORMAL")
    endif()
endif()

# Enable debug auto-reduction by default for debug builds
if(NOT DEFINED SD_DEBUG_AUTO_REDUCE AND SD_GCC_FUNCTRACE STREQUAL "ON")
    set(SD_DEBUG_AUTO_REDUCE TRUE)
endif()

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

# Helper options - these control whether helper libraries are enabled
option(HELPERS_armcompute "Enable ARM Compute Library helper" OFF)
option(HELPERS_onednn "Enable OneDNN helper" OFF)
option(HELPERS_cudnn "Enable cuDNN helper" OFF)

# Handle PRINT_INDICES option
message("PRINT_INDICES: ${PRINT_INDICES}")

function(print_status_colored type message)
    if(type STREQUAL "ERROR")
        message(FATAL_ERROR "❌ ${message}")
    elseif(type STREQUAL "WARNING")
        message(WARNING "⚠️  ${message}")
    elseif(type STREQUAL "SUCCESS")
        message(STATUS "✅ ${message}")
    elseif(type STREQUAL "INFO")
        message(STATUS "ℹ️  ${message}")
    elseif(type STREQUAL "DEBUG")
        message(STATUS "🔍 ${message}")
    elseif(type STREQUAL "NOTICE")
        message(NOTICE "📢 ${message}")
    else()
        message(STATUS "${message}")
    endif()
endfunction()

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

# Enhanced build messages with type information
message(STATUS "")
print_status_colored("INFO" "=== LIBND4J BUILD CONFIGURATION ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Chip: ${SD_CHIP}")
message(STATUS "Platform: ${SD_PLATFORM}")

if(SD_TYPES_LIST_COUNT GREATER 0)
    message(STATUS "Type Selection: SELECTIVE (${SD_TYPES_LIST_COUNT} types)")
else()
    message(STATUS "Type Selection: ALL")
endif()

if(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "")
    message(STATUS "Debug Type Profile: ${SD_DEBUG_TYPE_PROFILE}")
endif()

if(SD_GCC_FUNCTRACE STREQUAL "ON")
    message(STATUS "Function Tracing: ENABLED")
    if(SD_DEBUG_AUTO_REDUCE)
        message(STATUS "Debug Type Reduction: ENABLED")
    endif()
endif()

message(STATUS "Type Validation: ${SD_TYPES_VALIDATION_MODE}")
message(STATUS "")

macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# Type aliases for normalization
set(TYPE_ALIAS_float "float32")
set(TYPE_ALIAS_half "float16")
set(TYPE_ALIAS_long "int64")
set(TYPE_ALIAS_unsignedlong "uint64")
set(TYPE_ALIAS_int "int32")
set(TYPE_ALIAS_bfloat "bfloat16")
set(TYPE_ALIAS_float64 "double")

# All supported types list
set(ALL_SUPPORTED_TYPES
        "bool" "int8" "uint8" "int16" "uint16" "int32" "uint32"
        "int64" "uint64" "float16" "bfloat16" "float32" "double"
        "float" "half" "long" "unsignedlong" "int" "bfloat" "float64"
        "utf8" "utf16" "utf32"
)

# Minimum required types for basic functionality
set(MINIMUM_REQUIRED_TYPES "int32" "int64" "float32")

# Debug type profiles
set(DEBUG_PROFILE_MINIMAL_INDEXING "float32;double;int32;int64")
set(DEBUG_PROFILE_ESSENTIAL "float32;double;int32;int64;int8;int16")
set(DEBUG_PROFILE_FLOATS_ONLY "float32;double;float16")
set(DEBUG_PROFILE_INTEGERS_ONLY "int8;int16;int32;int64;uint8;uint16;uint32;uint64")
set(DEBUG_PROFILE_SINGLE_PRECISION "float32;int32;int64")
set(DEBUG_PROFILE_DOUBLE_PRECISION "double;int32;int64")

# Function to normalize a type name (handle aliases)
function(normalize_type input_type output_var)
    set(normalized_type "${input_type}")

    # Check if it's an alias
    if(DEFINED TYPE_ALIAS_${input_type})
        set(normalized_type "${TYPE_ALIAS_${input_type}}")
    endif()

    set(${output_var} "${normalized_type}" PARENT_SCOPE)
endfunction()

# Function to check if a type is supported
function(is_type_supported type result_var)
    normalize_type("${type}" normalized_type)

    if("${normalized_type}" IN_LIST ALL_SUPPORTED_TYPES)
        set(${result_var} TRUE PARENT_SCOPE)
    else()
        set(${result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Function to show available types
function(show_available_types)
    message(STATUS "")
    print_status_colored("INFO" "=== AVAILABLE DATA TYPES ===")
    message(STATUS "")
    message(STATUS "Core Types:")
    message(STATUS "  bool     - Boolean type")
    message(STATUS "  int8     - 8-bit signed integer")
    message(STATUS "  uint8    - 8-bit unsigned integer")
    message(STATUS "  int16    - 16-bit signed integer")
    message(STATUS "  uint16   - 16-bit unsigned integer")
    message(STATUS "  int32    - 32-bit signed integer")
    message(STATUS "  uint32   - 32-bit unsigned integer")
    message(STATUS "  int64    - 64-bit signed integer")
    message(STATUS "  uint64   - 64-bit unsigned integer")
    message(STATUS "")
    message(STATUS "Floating Point Types:")
    message(STATUS "  float16  - 16-bit floating point (half precision)")
    message(STATUS "  bfloat16 - 16-bit brain floating point")
    message(STATUS "  float32  - 32-bit floating point (single precision)")
    message(STATUS "  double   - 64-bit floating point (double precision)")
    message(STATUS "")
    message(STATUS "Type Aliases:")
    message(STATUS "  float    -> float32")
    message(STATUS "  half     -> float16")
    message(STATUS "  long     -> int64")
    message(STATUS "  int      -> int32")
    message(STATUS "  bfloat   -> bfloat16")
    message(STATUS "  float64  -> double")
    message(STATUS "")
endfunction()

# Function to resolve debug type profile
function(resolve_debug_profile profile custom_types result_var)
    if(profile STREQUAL "CUSTOM")
        if(custom_types AND NOT custom_types STREQUAL "")
            # Ensure minimum indexing types are included
            set(minimum_types "int32;int64;float32")
            set(combined_types "${minimum_types}")

            # Add custom types, avoiding duplicates
            string(REPLACE ";" ";" CUSTOM_LIST "${custom_types}")
            foreach(type IN LISTS CUSTOM_LIST)
                if(NOT type IN_LIST combined_types)
                    set(combined_types "${combined_types};${type}")
                endif()
            endforeach()
            set(${result_var} "${combined_types}" PARENT_SCOPE)
        else()
            message(FATAL_ERROR "CUSTOM profile specified but no custom types provided!")
        endif()
    elseif(DEFINED DEBUG_PROFILE_${profile})
        set(${result_var} "${DEBUG_PROFILE_${profile}}" PARENT_SCOPE)
    else()
        print_status_colored("WARNING" "Unknown debug profile '${profile}', using MINIMAL_INDEXING")
        set(${result_var} "${DEBUG_PROFILE_MINIMAL_INDEXING}" PARENT_SCOPE)
    endif()
endfunction()

# Function to estimate build impact
function(estimate_build_impact types_string build_type)
    if(NOT types_string OR types_string STREQUAL "" OR types_string STREQUAL "all" OR types_string STREQUAL "ALL")
        print_status_colored("INFO" "=== BUILD IMPACT ESTIMATION ===")
        message(STATUS "Using ALL types - expect full compilation with all template instantiations")
        return()
    endif()

    string(REPLACE ";" ";" TYPES_LIST "${types_string}")
    list(LENGTH TYPES_LIST type_count)

    if(type_count GREATER 0)
        math(EXPR est_2_combinations "${type_count} * ${type_count}")
        math(EXPR est_3_combinations "${type_count} * ${type_count} * ${type_count}")
        math(EXPR est_binary_size_mb "${est_3_combinations} * 10 / 27")  # Rough estimate

        print_status_colored("INFO" "=== BUILD IMPACT ESTIMATION ===")
        message(STATUS "Type count: ${type_count}")
        message(STATUS "Estimated 2-type combinations: ${est_2_combinations}")
        message(STATUS "Estimated 3-type combinations: ${est_3_combinations}")
        message(STATUS "Estimated binary size: ~${est_binary_size_mb}MB")

        if(build_type STREQUAL "Debug" AND est_3_combinations GREATER 125)
            print_status_colored("WARNING" "HIGH COMBINATION COUNT DETECTED!")
            print_status_colored("WARNING" "${est_3_combinations} 3-type combinations may cause:")
            print_status_colored("WARNING" "- Binary size >2GB (x86-64 limit exceeded)")
            print_status_colored("WARNING" "- Compilation failure due to PLT overflow")
            print_status_colored("WARNING" "- Very long build times")
            message(STATUS "")
            print_status_colored("WARNING" "Consider using fewer types for debug builds:")
            print_status_colored("WARNING" "-DSD_DEBUG_TYPE_PROFILE=MINIMAL_INDEXING")
            print_status_colored("WARNING" "-DSD_TYPES_LIST=\"float32;double;int32;int64\"")
        elseif(est_binary_size_mb GREATER 1000)
            print_status_colored("WARNING" "Large binary size warning: ~${est_binary_size_mb}MB")
        endif()
    endif()
endfunction()

# Function to validate a list of types
function(validate_type_list types_string validation_mode)
    print_status_colored("INFO" "=== CMAKE TYPE VALIDATION ===")

    # Handle empty or special cases
    if(NOT types_string OR types_string STREQUAL "" OR types_string STREQUAL "all" OR types_string STREQUAL "ALL")
        if(validation_mode STREQUAL "STRICT")
            print_status_colored("ERROR" "No data types specified and strict mode enabled!")
        else()
            print_status_colored("WARNING" "No data types specified, using ALL types")
            return()
        endif()
    endif()

    # Parse semicolon-separated types
    string(REPLACE ";" ";" TYPES_LIST "${types_string}")
    set(invalid_types "")
    set(valid_types "")
    set(normalized_types "")

    foreach(type IN LISTS TYPES_LIST)
        # Trim whitespace
        string(STRIP "${type}" type)

        if(NOT type STREQUAL "")
            is_type_supported("${type}" is_valid)

            if(is_valid)
                normalize_type("${type}" normalized_type)
                list(APPEND valid_types "${type}")
                list(APPEND normalized_types "${normalized_type}")

                if(NOT type STREQUAL normalized_type)
                    message(STATUS "  ✅ ${type} (normalized to: ${normalized_type})")
                else()
                    message(STATUS "  ✅ ${type}")
                endif()
            else()
                list(APPEND invalid_types "${type}")
                message(STATUS "  ❌ ${type} (INVALID)")
            endif()
        endif()
    endforeach()

    # Check for invalid types
    list(LENGTH invalid_types invalid_count)
    if(invalid_count GREATER 0)
        string(REPLACE ";" ", " invalid_types_str "${invalid_types}")
        print_status_colored("ERROR" "Found ${invalid_count} invalid type(s): ${invalid_types_str}")
        show_available_types()
        message(FATAL_ERROR "Type validation failed!")
    endif()

    # Check for no valid types
    list(LENGTH valid_types valid_count)
    if(valid_count EQUAL 0)
        print_status_colored("ERROR" "No valid types found!")
        show_available_types()
        message(FATAL_ERROR "Type validation failed!")
    endif()

    # Check for minimum required types
    set(missing_essential "")
    foreach(req_type IN LISTS MINIMUM_REQUIRED_TYPES)
        if(NOT req_type IN_LIST normalized_types)
            list(APPEND missing_essential "${req_type}")
        endif()
    endforeach()

    list(LENGTH missing_essential missing_count)
    if(missing_count GREATER 0)
        string(REPLACE ";" ", " missing_essential_str "${missing_essential}")
        print_status_colored("WARNING" "Missing recommended essential types: ${missing_essential_str}")
        print_status_colored("WARNING" "Array indexing and basic operations may fail at runtime!")

        if(validation_mode STREQUAL "STRICT")
            string(REPLACE ";" ", " required_types_str "${MINIMUM_REQUIRED_TYPES}")
            print_status_colored("ERROR" "Strict mode requires essential types: ${required_types_str}")
            message(FATAL_ERROR "Essential types missing in strict mode!")
        endif()
    endif()

    # Check for excessive type combinations in debug builds
    if(validation_mode STREQUAL "DEBUG" AND valid_count GREATER 6)
        math(EXPR estimated_combinations "${valid_count} * ${valid_count} * ${valid_count}")
        print_status_colored("WARNING" "Debug build with ${valid_count} types may generate ~${estimated_combinations} combinations")
        print_status_colored("WARNING" "This could result in very large binaries and long compile times!")
        print_status_colored("WARNING" "Consider using a debug type profile: -DSD_DEBUG_TYPE_PROFILE=MINIMAL_INDEXING")
    endif()

    string(REPLACE ";" ", " normalized_types_str "${normalized_types}")
    print_status_colored("SUCCESS" "Type validation passed: ${valid_count} valid types")
    message(STATUS "Selected types: ${normalized_types_str}")
endfunction()

# FAIL-FAST TYPE VALIDATION
function(validate_generated_defines_failfast)
    if(SD_TYPES_LIST_COUNT GREATER 0)
        print_status_colored("INFO" "=== FAIL-FAST VALIDATION: Checking generated defines ===")

        # Build list of expected defines
        set(EXPECTED_DEFINES "")
        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            string(TOUPPER ${normalized_type} SD_TYPE_UPPERCASE)
            list(APPEND EXPECTED_DEFINES "HAS_${SD_TYPE_UPPERCASE}")
        endforeach()

        # Check the generated file
        set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
        if(NOT EXISTS "${INCLUDE_OPS_FILE}")
            print_status_colored("WARNING" "Generated defines file will be created: ${INCLUDE_OPS_FILE}")
            return()
        endif()

        file(READ "${INCLUDE_OPS_FILE}" GENERATED_CONTENT)

        # Check each expected define exists
        set(MISSING_DEFINES "")
        foreach(EXPECTED_DEFINE ${EXPECTED_DEFINES})
            string(FIND "${GENERATED_CONTENT}" "#define ${EXPECTED_DEFINE}" DEFINE_FOUND)
            if(DEFINE_FOUND EQUAL -1)
                list(APPEND MISSING_DEFINES "${EXPECTED_DEFINE}")
            endif()
        endforeach()

        # FAIL FAST if any defines are missing
        list(LENGTH MISSING_DEFINES missing_count)
        if(missing_count GREATER 0)
            string(REPLACE ";" ", " missing_str "${MISSING_DEFINES}")
            string(REPLACE ";" ", " expected_str "${EXPECTED_DEFINES}")

            message(STATUS "")
            print_status_colored("ERROR" "❌ VALIDATION FAILURE: Type processing failed")
            print_status_colored("ERROR" "")
            print_status_colored("ERROR" "Requested types: ${SD_TYPES_LIST}")
            print_status_colored("ERROR" "Expected defines: ${expected_str}")
            print_status_colored("ERROR" "Missing defines: ${missing_str}")
            print_status_colored("ERROR" "")
            print_status_colored("ERROR" "Generated file content:")
            message(STATUS "${GENERATED_CONTENT}")
            print_status_colored("ERROR" "")
            message(FATAL_ERROR "❌ BUILD TERMINATED: ${missing_count} type(s) failed to process correctly")
        endif()

        print_status_colored("SUCCESS" "✅ All ${SD_TYPES_LIST_COUNT} types validated successfully")
    endif()
endfunction()

function(validate_and_process_types_failfast)
    # Do the original processing
    validate_and_process_types()

    # Immediately validate the results
    validate_generated_defines_failfast()
endfunction()

# Main validation function to be called from CMakeLists.txt
function(validate_and_process_types)
    # Determine validation mode
    set(validation_mode "NORMAL")
    if(SD_GCC_FUNCTRACE STREQUAL "ON")
        set(validation_mode "DEBUG")
    endif()
    if(SD_STRICT_TYPE_VALIDATION)
        set(validation_mode "STRICT")
    endif()

    # Handle debug builds with auto-reduction
    if(SD_GCC_FUNCTRACE STREQUAL "ON" AND SD_DEBUG_AUTO_REDUCE)
        print_status_colored("INFO" "=== DEBUG BUILD TYPE REDUCTION ACTIVE ===")

        if(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "")
            resolve_debug_profile("${SD_DEBUG_TYPE_PROFILE}" "${SD_DEBUG_CUSTOM_TYPES}" resolved_types)
            set(SD_TYPES_LIST "${resolved_types}" PARENT_SCOPE)
            message(STATUS "Debug Profile: ${SD_DEBUG_TYPE_PROFILE}")
            message(STATUS "Resolved Types: ${resolved_types}")
        elseif(NOT SD_TYPES_LIST OR SD_TYPES_LIST STREQUAL "")
            # No types specified and no profile - use minimal safe default
            resolve_debug_profile("MINIMAL_INDEXING" "" resolved_types)
            set(SD_TYPES_LIST "${resolved_types}" PARENT_SCOPE)
            print_status_colored("WARNING" "Auto-selected MINIMAL_INDEXING profile for debug build")
            message(STATUS "Types: ${resolved_types}")
        endif()

        message(STATUS "=============================================")
    endif()

    # Validate the final datatypes
    if(SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        validate_type_list("${SD_TYPES_LIST}" "${validation_mode}")
        estimate_build_impact("${SD_TYPES_LIST}" "${CMAKE_BUILD_TYPE}")

        # Show configuration summary
        print_status_colored("INFO" "=== TYPE CONFIGURATION SUMMARY ===")

        if(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "")
            message(STATUS "Debug Type Profile: ${SD_DEBUG_TYPE_PROFILE}")
        endif()

        message(STATUS "Type Selection: SELECTIVE")
        message(STATUS "Building with types: ${SD_TYPES_LIST}")
        message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
        message(STATUS "")
    else()
        print_status_colored("INFO" "=== TYPE CONFIGURATION SUMMARY ===")
        message(STATUS "Type Selection: ALL (default)")
        message(STATUS "Building with all supported data types")
        message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
        message(STATUS "")
    endif()
endfunction()

macro(SETUP_LIBND4J_TYPE_VALIDATION)
    # Set default validation mode
    if(NOT DEFINED SD_TYPES_VALIDATION_MODE)
        if(SD_GCC_FUNCTRACE STREQUAL "ON")
            set(SD_TYPES_VALIDATION_MODE "DEBUG")
        elseif(SD_STRICT_TYPE_VALIDATION)
            set(SD_TYPES_VALIDATION_MODE "STRICT")
        else()
            set(SD_TYPES_VALIDATION_MODE "NORMAL")
        endif()
    endif()

    # Enable debug auto-reduction by default for debug builds
    if(NOT DEFINED SD_DEBUG_AUTO_REDUCE AND SD_GCC_FUNCTRACE STREQUAL "ON")
        set(SD_DEBUG_AUTO_REDUCE TRUE)
    endif()

    # Call the main validation function
    validate_and_process_types()

    # Update the count after validation
    if(SD_TYPES_LIST)
        list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
    else()
        set(SD_TYPES_LIST_COUNT 0)
    endif()
endmacro()

# Add this section before the flatbuffers download section
# Compile Definitions for Operations
# Call the main validation and processing function FIRST
validate_and_process_types_failfast()

# Compile Definitions for Operations (unchanged)
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

# Enhanced Compile Definitions for Types
list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
if(SD_TYPES_LIST_COUNT GREATER 0)
    add_compile_definitions(SD_SELECTIVE_TYPES)
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_TYPES\n")

    # Display the types being built with
    print_status_colored("SUCCESS" "=== BUILDING WITH SELECTED TYPES ===")
    set(TYPE_DISPLAY_LIST "")
    foreach(SD_TYPE ${SD_TYPES_LIST})
        normalize_type("${SD_TYPE}" normalized_type)
        string(TOUPPER ${normalized_type} SD_TYPE_UPPERCASE)
        add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
        message(STATUS "✅ TYPE: ${normalized_type}")
        string(APPEND DEFINITIONS_CONTENT "#define HAS_${SD_TYPE_UPPERCASE}\n")
        list(APPEND TYPE_DISPLAY_LIST "${normalized_type}")
    endforeach()

    # Create a summary of types being built
    list(LENGTH TYPE_DISPLAY_LIST TYPE_COUNT)
    string(REPLACE ";" ", " TYPE_DISPLAY_STRING "${TYPE_DISPLAY_LIST}")
    print_status_colored("SUCCESS" "Building with ${TYPE_COUNT} data types: ${TYPE_DISPLAY_STRING}")

else()
    # Building with all types
    print_status_colored("INFO" "=== BUILDING WITH ALL TYPES ===")
    message(STATUS "No selective types specified - building with all supported data types")
    message(STATUS "This will include all type instantiations and may result in larger binaries")
endif()

# Create the directory for generated files if it doesn't exist
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/generated")

# Write Definitions to include_ops.h file
set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
message("Generating include_ops.h at: ${INCLUDE_OPS_FILE}")
file(WRITE "${INCLUDE_OPS_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")

# Add the generated directory to include paths
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/generated")

# Enhanced type validation and processing
if(SD_TYPES_LIST_COUNT GREATER 0)
    add_compile_definitions(SD_SELECTIVE_TYPES)
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_TYPES\n")

    # Display the types being built with
    print_status_colored("SUCCESS" "=== BUILDING WITH SELECTED TYPES ===")
    set(TYPE_DISPLAY_LIST "")
    foreach(SD_TYPE ${SD_TYPES_LIST})
        normalize_type("${SD_TYPE}" normalized_type)
        string(TOUPPER ${normalized_type} SD_TYPE_UPPERCASE)
        add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
        message(STATUS "✅ TYPE: ${normalized_type}")
        string(APPEND DEFINITIONS_CONTENT "#define HAS_${SD_TYPE_UPPERCASE}\n")
        list(APPEND TYPE_DISPLAY_LIST "${normalized_type}")
    endforeach()

    # Create a summary of types being built
    list(LENGTH TYPE_DISPLAY_LIST TYPE_COUNT)
    string(REPLACE ";" ", " TYPE_DISPLAY_STRING "${TYPE_DISPLAY_LIST}")
    print_status_colored("SUCCESS" "Building with ${TYPE_COUNT} data types: ${TYPE_DISPLAY_STRING}")

    # Estimate and warn about potential compilation issues
    estimate_build_impact("${TYPE_DISPLAY_STRING}" "${CMAKE_BUILD_TYPE}")

else()
    # Building with all types
    print_status_colored("INFO" "=== BUILDING WITH ALL TYPES ===")
    message(STATUS "No selective types specified - building with all supported data types")
    message(STATUS "This will include all type instantiations and may result in larger binaries")
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
if(HELPERS_armcompute AND "${SD_ARCH}" MATCHES "armv7")
    set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ARMV7})
    message("Using ARM Compute Library for ARMv7 32-bit")
elseif(HELPERS_armcompute AND "${SD_ARCH}" MATCHES "armv8-a")
    if(SD_ANDROID_BUILD)
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_ANDROID_AARCH64})
        message("Using ARM Compute Library for Android ARM64 (matched armv8-a)")
    else()
        set(ARM_COMPUTE_URL ${ARM_COMPUTE_URL_AARCH64})
        message("Using ARM Compute Library for ARM64 (matched armv8-a)")
    endif()
elseif(HELPERS_armcompute AND "${SD_ARCH}" MATCHES "arm64")
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

if(SD_ARM_BUILD)
    if(NOT SD_ARCH) # This is TRUE for undefined OR empty strings
        message(STATUS "Warning: SD_ARCH was not set for this ARM build. Defaulting to 'aarch64'.")
        set(SD_ARCH "aarch64")
    endif()
endif()


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
        if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
            set(SD_ARCH "x86-64")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm*|aarch64")
        set(SD_ARM_BUILD true)
        if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
                set(SD_ARCH "armv8-a")
            else()
                set(SD_ARCH "armv7-a")
            endif()
        endif()
    endif()
endif()
if(SD_ARM_BUILD)
    if(NOT DEFINED SD_ARCH OR SD_ARCH STREQUAL "")
        message(STATUS "Warning: SD_ARCH was not set for this ARM build. Defaulting to 'armv8-a'.")
        set(SD_ARCH "armv8-a")
    endif()

    if(SD_ANDROID)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    endif()
endif()



# Apply type-based combination reduction for debug builds
if(SD_GCC_FUNCTRACE STREQUAL "ON" AND ESTIMATED_3_COMBINATIONS GREATER 64)
    message(STATUS "Applying debug build combination reduction...")

    # Generate combinations based on actual selected types
    set(ORIGINAL_COMBINATIONS_3 ${COMBINATIONS_3})
    set(ORIGINAL_COMBINATIONS_2 ${COMBINATIONS_2})

    # Map selected types to indices
    set(TYPE_INDEX_MAP "")
    set(TYPE_INDEX 0)
    foreach(TYPE_NAME ${SD_TYPES_LIST})
        list(APPEND TYPE_INDEX_MAP ${TYPE_INDEX})
        math(EXPR TYPE_INDEX "${TYPE_INDEX} + 1")
    endforeach()

    # Generate new combinations based on selected types only
    set(NEW_COMBINATIONS_3 "")
    set(NEW_COMBINATIONS_2 "")

    foreach(i ${TYPE_INDEX_MAP})
        foreach(j ${TYPE_INDEX_MAP})
            list(APPEND NEW_COMBINATIONS_2 "${i},${j}")
            foreach(k ${TYPE_INDEX_MAP})
                list(APPEND NEW_COMBINATIONS_3 "${i},${j},${k}")
            endforeach()
        endforeach()
    endforeach()

    set(COMBINATIONS_3 ${NEW_COMBINATIONS_3})
    set(COMBINATIONS_2 ${NEW_COMBINATIONS_2})

    list(LENGTH COMBINATIONS_3 FINAL_3_COUNT)
    list(LENGTH COMBINATIONS_2 FINAL_2_COUNT)
    message(STATUS "Reduced combinations: ${FINAL_3_COUNT} 3-type, ${FINAL_2_COUNT} 2-type")
endif()

# Enhanced type validation
# Enhanced type validation with CMake functions
if(SD_TYPES_LIST_COUNT GREATER 0)
    # Validate that all specified types are supported
    set(INVALID_TYPES "")
    set(VALID_TYPES "")
    set(NORMALIZED_TYPES "")

    foreach(SD_TYPE ${SD_TYPES_LIST})
        is_type_supported("${SD_TYPE}" is_valid)

        if(is_valid)
            normalize_type("${SD_TYPE}" normalized_type)
            list(APPEND VALID_TYPES "${SD_TYPE}")
            list(APPEND NORMALIZED_TYPES "${normalized_type}")
        else()
            list(APPEND INVALID_TYPES "${SD_TYPE}")
        endif()
    endforeach()

    # Report invalid types
    list(LENGTH INVALID_TYPES invalid_count)
    if(invalid_count GREATER 0)
        string(REPLACE ";" ", " invalid_types_str "${INVALID_TYPES}")
        message(FATAL_ERROR "❌ Invalid data types specified: ${invalid_types_str}")
    endif()

    # Check for missing critical types
    set(MISSING_INDEXING_TYPES "")

    if(NOT "int32" IN_LIST NORMALIZED_TYPES)
        list(APPEND MISSING_INDEXING_TYPES "int32")
    endif()
    if(NOT "int64" IN_LIST NORMALIZED_TYPES)
        list(APPEND MISSING_INDEXING_TYPES "int64")
    endif()
    if(NOT "float32" IN_LIST NORMALIZED_TYPES)
        list(APPEND MISSING_INDEXING_TYPES "float32")
    endif()

    list(LENGTH MISSING_INDEXING_TYPES missing_count)
    if(missing_count GREATER 0)
        string(REPLACE ";" ", " missing_types_str "${MISSING_INDEXING_TYPES}")
        print_status_colored("WARNING" "Missing critical indexing types: ${missing_types_str}")
        print_status_colored("WARNING" "Array indexing operations may fail at runtime!")
        print_status_colored("WARNING" "Consider adding these to your type list.")
    endif()

    # Show final type selection
    string(REPLACE ";" ", " normalized_types_str "${NORMALIZED_TYPES}")
    print_status_colored("SUCCESS" "✅ Selected data types validated: ${normalized_types_str}")
endif()


# Architecture Tuning
if(SD_ARCH MATCHES "armv8")
    set(ARCH_TUNE "-march=${SD_ARCH}")
elseif(SD_ARCH MATCHES "armv7")
    set(ARCH_TUNE "-march=${SD_ARCH} -mfpu=neon")
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

    # FIXED: Only set architecture flags if we have valid values
    if(NOT WIN32 AND NOT SD_CUDA)
        if(DEFINED SD_ARCH AND NOT SD_ARCH STREQUAL "" AND DEFINED ARCH_TYPE AND NOT ARCH_TYPE STREQUAL "")
            set(ARCH_TUNE "-march=${SD_ARCH} -mtune=${ARCH_TYPE}")
        elseif(DEFINED SD_ARCH AND NOT SD_ARCH STREQUAL "")
            # Fallback if ARCH_TYPE is not set
            set(ARCH_TUNE "-march=${SD_ARCH}")
        endif()
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=10 ${ARCH_TUNE} ${INFORMATIVE_FLAGS} -std=c++17 -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --param ggc-min-expand=10 -fPIC")

    # Add flags to handle large relocations for x86_64
    if(SD_X86_BUILD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-relax")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--no-relax")
    endif()

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

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ftemplate-backtrace-limit=0 -gno-record-gcc-switches -ftrack-macro-expansion=0 -fstack-protector -fstack-protector-all  -Wall  -Wextra -Werror -Wno-return-type -Wno-error=int-in-bool-context -Wno-unused-variable -Wno-error=implicit-fallthrough -Wno-return-type -Wno-unused-parameter -Wno-error=unknown-pragmas -ggdb3 -lpthread -pthread -MT -Bsymbolic -lbfd -rdynamic -lunwind -ldw -ldl -fno-omit-frame-pointer -fno-optimize-sibling-calls -rdynamic -finstrument-functions  -O0 -fPIC")
        add_compile_definitions(SD_GCC_FUNCTRACE)
    endif()
endif()


# Comprehensive linker fix for PLT overflow
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SD_X86_BUILD)
    message(STATUS "Configuring linker for large template library with PLT overflow prevention")

    # Clear any existing conflicting linker flags
    string(REGEX REPLACE "-fuse-ld=[a-zA-Z]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    string(REGEX REPLACE "-fuse-ld=[a-zA-Z]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

    # Try mold first (best for large C++ template libraries)
    execute_process(COMMAND mold --version OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE MOLD_AVAILABLE)
    if(MOLD_AVAILABLE EQUAL 0)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=mold")
        message(STATUS "✓ Using mold linker (handles large PLT tables)")
    else()
        # Test if linker supports --plt-align before using it
        execute_process(
                COMMAND ${CMAKE_LINKER} --help
                OUTPUT_VARIABLE LD_HELP_OUTPUT
                ERROR_QUIET
        )

        string(FIND "${LD_HELP_OUTPUT}" "--plt-align" PLT_ALIGN_SUPPORTED)

        if(PLT_ALIGN_SUPPORTED GREATER -1)
            # Use GNU LD with PLT-specific optimizations if supported
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--plt-align=32 -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--plt-align=32 -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
            message(STATUS "✓ Using GNU LD with PLT overflow prevention")
        else()
            # Fall back to basic optimizations for older linkers
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--hash-style=both -Wl,-z,max-page-size=0x200000")
            message(STATUS "✓ Using GNU LD with basic optimizations (--plt-align not supported)")
        endif()
    endif()

    # Use large memory model (required for your template scale)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=large")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcmodel=large")

    # Memory optimization during compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param ggc-min-expand=100 --param ggc-min-heapsize=131072")

    # Section splitting for better linker handling
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")

    message(STATUS "Applied PLT overflow prevention for large template library")
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

# ARM Compute configuration - FIXED: Only configure when explicitly enabled
# ARM Compute configuration - FIXED: Better error handling and download robustness
set(ARMCOMPUTE_LIBRARIES "")
set(HAVE_ARMCOMPUTE 0)
if(HELPERS_armcompute)
    message(STATUS "ARM Compute Library helper is enabled")
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

        # Create directories upfront to avoid permission issues
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/armcompute_external/src")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/downloads")
        file(MAKE_DIRECTORY "${ARMCOMPUTE_INSTALL_DIR}")

        ExternalProject_Add(armcompute_external
                PREFIX            "${CMAKE_BINARY_DIR}/armcompute_external"
                URL               "${ARMCOMPUTE_URL}"
                DOWNLOAD_DIR      "${CMAKE_BINARY_DIR}/downloads"
                SOURCE_DIR        "${ARMCOMPUTE_SOURCE_DIR}"
                CONFIGURE_COMMAND ""
                BUILD_COMMAND     ""
                # FIXED: More robust install command with better error handling
                INSTALL_COMMAND
                ${CMAKE_COMMAND} -E echo "Starting ARM Compute installation..." &&
                ${CMAKE_COMMAND} -E remove_directory "${ARMCOMPUTE_INSTALL_DIR}" &&
                ${CMAKE_COMMAND} -E make_directory "${ARMCOMPUTE_INSTALL_DIR}" &&
                ${CMAKE_COMMAND} -E make_directory "${ARMCOMPUTE_INSTALL_DIR}/include" &&
                ${CMAKE_COMMAND} -E make_directory "${ARMCOMPUTE_INSTALL_DIR}/lib" &&
                ${CMAKE_COMMAND} -E copy_directory "${ARMCOMPUTE_SOURCE_DIR}/${ARMCOMPUTE_PKG_NAME}/include" "${ARMCOMPUTE_INSTALL_DIR}/include" &&
                ${CMAKE_COMMAND} -E copy_directory "${ARMCOMPUTE_SOURCE_DIR}/${ARMCOMPUTE_PKG_NAME}/${ARMCOMPUTE_LIB_SUBDIR}" "${ARMCOMPUTE_INSTALL_DIR}/lib" &&
                ${CMAKE_COMMAND} -E echo "ARM Compute installation completed successfully"
                BUILD_BYPRODUCTS
                "${ARMCOMPUTE_INSTALL_DIR}/include/arm_compute/core/CL/CLKernelLibrary.h"
                "${ARMCOMPUTE_INSTALL_DIR}/lib/libarm_compute.so"
                "${ARMCOMPUTE_INSTALL_DIR}/lib/libarm_compute_graph.so"
                BUILD_IN_SOURCE   1
                # FIXED: Add timeout and retry options
                TIMEOUT           300
                DOWNLOAD_NO_PROGRESS 1
                # FIXED: Use UPDATE_DISCONNECTED to avoid timestamp issues
                UPDATE_DISCONNECTED 1
                # FIXED: Add LOG options for better debugging
                LOG_DOWNLOAD      ON
                LOG_INSTALL       ON
        )

        add_library(armcompute_interface INTERFACE)
        target_include_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/include")
        target_link_directories(armcompute_interface INTERFACE "${ARMCOMPUTE_INSTALL_DIR}/lib")
        target_link_libraries(armcompute_interface INTERFACE arm_compute arm_compute_graph)
        add_dependencies(armcompute_interface armcompute_external)

        set(ARMCOMPUTE_FOUND TRUE)
        set(ARMCOMPUTE_INCLUDE_DIRS "${ARMCOMPUTE_INSTALL_DIR}/include")
        set(ARMCOMPUTE_LIBRARIES armcompute_interface)
        set(HAVE_ARMCOMPUTE 1)

        # FIXED: Add verification step
        add_custom_command(TARGET armcompute_external POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "Verifying ARM Compute installation..."
                COMMAND test -f "${ARMCOMPUTE_INSTALL_DIR}/include/arm_compute/core/CL/CLKernelLibrary.h" ||
        (echo "ERROR: ARM Compute headers not found!" && exit 1)
                COMMAND test -f "${ARMCOMPUTE_INSTALL_DIR}/lib/libarm_compute.so" ||
        (echo "ERROR: ARM Compute library not found!" && exit 1)
                COMMAND ${CMAKE_COMMAND} -E echo "ARM Compute verification completed successfully"
                COMMENT "Verifying ARM Compute Library installation"
        )

    else()
        message(STATUS "Skipping ARM Compute setup via ExternalProject_Add (disabled or not AArch64)")
        set(ARMCOMPUTE_FOUND FALSE)
    endif()
else()
    message(STATUS "ARM Compute Library helper is disabled (HELPERS_armcompute=OFF)")
    set(ARMCOMPUTE_FOUND FALSE)
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

# OneDNN configuration - FIXED: Only configure when explicitly enabled
if(NOT DEFINED ONEDNN_PROCESSING_DONE)
    set(ONEDNN_PROCESSING_DONE FALSE CACHE INTERNAL "Flag to indicate OneDNN processing status")
endif()
if(HELPERS_onednn)
    message(STATUS "OneDNN helper is enabled")
    include(ExternalProject)
    message(STATUS "Configuring OneDNN dependency using ExternalProject_Add")
    set(HAVE_ONEDNN TRUE)

    set(ONEDNN_INSTALL_DIR "${CMAKE_BINARY_DIR}/onednn_install")
    set(ONEDNN_SOURCE_DIR "${CMAKE_BINARY_DIR}/onednn_external/src")
    set(ONEDNN_BUILD_DIR "${CMAKE_BINARY_DIR}/onednn_external/build")

    ExternalProject_Add(onednn_external
            PREFIX            "${CMAKE_BINARY_DIR}/onednn_external"
            GIT_REPOSITORY    "https://github.com/uxlfoundation/oneDNN.git"
            GIT_TAG           "main"
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
    message(STATUS "OneDNN helper is disabled (HELPERS_onednn=OFF)")
    set(HAVE_ONEDNN FALSE)
    set(ONEDNN "")
endif()

# cuDNN configuration - FIXED: Better Windows support and debugging
if(HELPERS_cudnn)
    message(STATUS "cuDNN helper is enabled")
    if(NOT SD_CUDA)
        message(FATAL_ERROR "Can't build cuDNN on non-CUDA platform")
    endif()

    # Add debug output to verify the option is being triggered
    message(STATUS "HELPERS_cudnn is ON - proceeding with cuDNN detection")
    message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
    message(STATUS "CUDA found: ${CUDA_FOUND}")

    SET(CUDNN_LIBNAME "cudnn")

    # Windows-specific cuDNN library names and paths
    if(WIN32)
        # On Windows, cuDNN might be named differently
        SET(CUDNN_LIBNAME_WIN "cudnn64_8")  # Common naming for cuDNN 8.x
        SET(CUDNN_LIBNAME_WIN_ALT "cudnn")  # Fallback naming

        # Try to find CUDA installation directory if not set
        if(NOT DEFINED ENV{CUDA_PATH} AND NOT DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
            # Common CUDA installation paths on Windows
            set(CUDA_POSSIBLE_PATHS
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7"
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
                    "C:/cuda"
            )
            foreach(CUDA_PATH_CANDIDATE ${CUDA_POSSIBLE_PATHS})
                if(EXISTS "${CUDA_PATH_CANDIDATE}")
                    set(ENV{CUDA_TOOLKIT_ROOT_DIR} "${CUDA_PATH_CANDIDATE}")
                    message(STATUS "Auto-detected CUDA path: ${CUDA_PATH_CANDIDATE}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    # Set cuDNN root directory from environment variables
    if(DEFINED ENV{CUDNN_ROOT_DIR})
        message(STATUS "Using cuDNN root directory from environment: $ENV{CUDNN_ROOT_DIR}")
        set(CUDNN_ROOT_DIR $ENV{CUDNN_ROOT_DIR})
    endif()

    if(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
        message(STATUS "Using CUDA root directory from environment: $ENV{CUDA_TOOLKIT_ROOT_DIR}")
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_TOOLKIT_ROOT_DIR})
    endif()

    if(DEFINED ENV{CUDA_PATH})
        message(STATUS "Using CUDA path from environment: $ENV{CUDA_PATH}")
        set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
    endif()

    message(STATUS "cuDNN root dir: ${CUDNN_ROOT_DIR}")
    message(STATUS "CUDA toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")

    # Enhanced search paths for Windows
    set(CUDNN_SEARCH_PATHS "")
    if(WIN32)
        list(APPEND CUDNN_SEARCH_PATHS
                ${CUDNN_ROOT_DIR}
                ${CUDA_TOOLKIT_ROOT_DIR}
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
                "C:/cuda"
                "C:/cudnn"
                "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA"
        )
    else()
        list(APPEND CUDNN_SEARCH_PATHS
                ${CUDNN_ROOT_DIR}
                ${CUDA_TOOLKIT_ROOT_DIR}
                /usr/local/cuda
                /opt/cuda
        )
    endif()

    # Find cuDNN include directory
    find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDNN_SEARCH_PATHS}
            PATH_SUFFIXES
            include
            cuda/include
            include/cuda
            CUDNN/include  # Windows cuDNN standalone installation
            DOC "cuDNN include directory"
    )

    # Find cuDNN library
    if(WIN32)
        # Try multiple library names on Windows
        find_library(CUDNN_LIBRARY
                NAMES ${CUDNN_LIBNAME_WIN} ${CUDNN_LIBNAME_WIN_ALT} ${CUDNN_LIBNAME}
                HINTS ${CUDNN_SEARCH_PATHS}
                PATH_SUFFIXES
                lib
                lib64
                lib/x64
                cuda/lib
                cuda/lib64
                cuda/lib/x64
                CUDNN/lib/x64  # Windows cuDNN standalone installation
                DOC "cuDNN library"
        )
    else()
        find_library(CUDNN_LIBRARY ${CUDNN_LIBNAME}
                HINTS ${CUDNN_SEARCH_PATHS}
                PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64
                DOC "cuDNN library"
        )
    endif()

    # Debug output
    message(STATUS "cuDNN include search result: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN library search result: ${CUDNN_LIBRARY}")

    # Additional debugging - list what was actually searched
    if(NOT CUDNN_INCLUDE_DIR)
        message(STATUS "cuDNN include search paths tried:")
        foreach(search_path ${CUDNN_SEARCH_PATHS})
            message(STATUS "  ${search_path}/include")
            message(STATUS "  ${search_path}/cuda/include")
            message(STATUS "  ${search_path}/CUDNN/include")
        endforeach()
    endif()

    if(NOT CUDNN_LIBRARY)
        message(STATUS "cuDNN library search paths tried:")
        foreach(search_path ${CUDNN_SEARCH_PATHS})
            if(WIN32)
                message(STATUS "  ${search_path}/lib/x64")
                message(STATUS "  ${search_path}/CUDNN/lib/x64")
            endif()
            message(STATUS "  ${search_path}/lib")
            message(STATUS "  ${search_path}/lib64")
        endforeach()
        if(WIN32)
            message(STATUS "cuDNN library names tried: ${CUDNN_LIBNAME_WIN}, ${CUDNN_LIBNAME_WIN_ALT}, ${CUDNN_LIBNAME}")
        else()
            message(STATUS "cuDNN library name tried: ${CUDNN_LIBNAME}")
        endif()
    endif()

    # Verify cuDNN was found
    if(CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message(STATUS "✓ Found cuDNN:")
        message(STATUS "  Include: ${CUDNN_INCLUDE_DIR}")
        message(STATUS "  Library: ${CUDNN_LIBRARY}")

        # Verify the files actually exist
        if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
            message(STATUS "  ✓ cudnn.h found")
        else()
            message(WARNING "  ✗ cudnn.h NOT found at ${CUDNN_INCLUDE_DIR}/cudnn.h")
        endif()

        if(EXISTS "${CUDNN_LIBRARY}")
            message(STATUS "  ✓ cuDNN library file exists")
        else()
            message(WARNING "  ✗ cuDNN library file NOT found at ${CUDNN_LIBRARY}")
        endif()

        include_directories(${CUDNN_INCLUDE_DIR})
        set(HAVE_CUDNN true)
        set(CUDNN ${CUDNN_LIBRARY})

        # Add cuDNN preprocessor definition
        add_definitions(-DHAVE_CUDNN=1)

    else()
        message(WARNING "✗ cuDNN not found. Continuing without cuDNN support.")
        message(STATUS "To enable cuDNN, ensure:")
        message(STATUS "  1. cuDNN is installed")
        message(STATUS "  2. Set CUDNN_ROOT_DIR environment variable, or")
        message(STATUS "  3. Set CUDA_TOOLKIT_ROOT_DIR environment variable, or")
        if(WIN32)
            message(STATUS "  4. Install cuDNN to a standard location like C:/cudnn")
            message(STATUS "  5. Make sure cuDNN DLLs are in your PATH")
        else()
            message(STATUS "  4. Install cuDNN to /usr/local/cuda or /opt/cuda")
        endif()
        set(HAVE_CUDNN false)
        set(CUDNN "")
    endif()
else()
    message(STATUS "cuDNN helper is disabled (HELPERS_cudnn=OFF)")
    set(HAVE_CUDNN false)
    set(CUDNN "")
endif()

# Add a summary message about cuDNN status
if(SD_CUDA)
    if(HAVE_CUDNN)
        message(STATUS "CUDA BUILD: cuDNN integration ENABLED")
    else()
        message(STATUS "CUDA BUILD: cuDNN integration DISABLED")
        message(STATUS "  To enable cuDNN, set -DHELPERS_cudnn=ON and ensure cuDNN is installed")
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
    set(CMAKE_CUDA_COMPILER_LAUNCHER "" CACHE STRING "CUDA compiler launcher")
    # THE LINE BELOW IS MOVED TO THE TOP OF THE FILE AND IS NOW INEFFECTIVE HERE
    # set(CMAKE_CUDA_FLAGS "--allow-unsupported-compiler" CACHE STRING "CUDA compile flags" FORCE)

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
                set(CMAKE_CUDA_FLAGS " ${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC --device-debug -lineinfo -G")
                add_compile_definitions(SD_GCC_FUNCTRACE)
            else()
                set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
            endif()
        endif()

        if(WIN32)
            if(MSVC)
                # Enable conforming preprocessor to fix __VA_ARGS__ expansion issues
                add_compile_options(/Zc:preprocessor)
            endif()
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

    # OneDNN sources - FIXED: Only include when helper is enabled
    set(CUSTOMOPS_ONEDNN_SOURCES "")
    if(HAVE_ONEDNN)
        message(STATUS "Including OneDNN platform sources")
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES_TMP ./include/ops/declarable/platform/mkldnn/*.cpp ./include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
        set(CUSTOMOPS_ONEDNN_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES_TMP})
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES})
    else()
        message(STATUS "Skipping OneDNN platform sources (OneDNN helper disabled)")
    endif()

    # ARM Compute sources - FIXED: Only include when helper is enabled
    set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
    if(HAVE_ARMCOMPUTE)
        message(STATUS "Including ARM Compute platform sources")
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES_TMP ./include/ops/declarable/platform/armcompute/*.cpp ./include/ops/declarable/platform/armcompute/*.h)
        set(CUSTOMOPS_ARMCOMPUTE_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES_TMP})
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
    else()
        message(STATUS "Skipping ARM Compute platform sources (ARM Compute helper disabled)")
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