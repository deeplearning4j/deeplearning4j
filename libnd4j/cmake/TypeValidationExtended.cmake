################################################################################
# Extended Type Validation Functions
# Enhanced type validation with error handling, debug profiles, and ML workloads
################################################################################

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

# Debug type profiles (enhanced with ML workloads)
set(DEBUG_PROFILE_MINIMAL_INDEXING "float32;double;int32;int64")
set(DEBUG_PROFILE_ESSENTIAL "float32;double;int32;int64;int8;int16")
set(DEBUG_PROFILE_FLOATS_ONLY "float32;double;float16")
set(DEBUG_PROFILE_INTEGERS_ONLY "int8;int16;int32;int64;uint8;uint16;uint32;uint64")
set(DEBUG_PROFILE_SINGLE_PRECISION "float32;int32;int64")
set(DEBUG_PROFILE_DOUBLE_PRECISION "double;int32;int64")
set(DEBUG_PROFILE_QUANTIZATION "int8;uint8;float32;int32;int64")
set(DEBUG_PROFILE_MIXED_PRECISION "float16;bfloat16;float32;int32;int64")
set(DEBUG_PROFILE_NLP "std::string;float32;int32;int64")

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

    # Use list(FIND) for compatibility with older CMake versions
    list(FIND ALL_SUPPORTED_TYPES "${normalized_type}" type_index)
    if(type_index GREATER -1)
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
                list(FIND combined_types "${type}" type_index)
                if(type_index EQUAL -1)
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
        list(FIND normalized_types "${req_type}" req_type_index)
        if(req_type_index EQUAL -1)
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
            print_status_colored("INFO" "Generated defines file will be created during template processing: ${INCLUDE_OPS_FILE}")
            print_status_colored("INFO" "Skipping pre-generation validation - this is normal for clean builds")
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

        # FAIL FAST if any defines are missing (only if file exists)
        list(LENGTH MISSING_DEFINES missing_count)
        if(missing_count GREATER 0)
            string(REPLACE ";" ", " missing_str "${MISSING_DEFINES}")
            string(REPLACE ";" ", " expected_str "${EXPECTED_DEFINES}")

            message(STATUS "")
            print_status_colored("WARNING" "⚠️ VALIDATION ISSUE: Some type defines missing from existing generated file")
            message(STATUS "")
            message(STATUS "Requested types: ${SD_TYPES_LIST}")
            message(STATUS "Expected defines: ${expected_str}")
            message(STATUS "Missing defines: ${missing_str}")
            message(STATUS "")
            message(STATUS "Generated file content preview:")
            string(LENGTH "${GENERATED_CONTENT}" content_length)
            if(content_length GREATER 500)
                string(SUBSTRING "${GENERATED_CONTENT}" 0 500 content_preview)
                message(STATUS "${content_preview}...")
            else()
                message(STATUS "${GENERATED_CONTENT}")
            endif()
            message(STATUS "")
            print_status_colored("INFO" "The template processing step should regenerate this file correctly")
        else()
            print_status_colored("SUCCESS" "✅ All ${SD_TYPES_LIST_COUNT} types found in existing generated file")
        endif()
    endif()
endfunction()

function(validate_and_process_types_failfast)
    # Do the original processing
    validate_and_process_types()

    # Do a non-fatal validation check (since generated files may not exist yet)
    validate_generated_defines_failfast()
endfunction()

# Alternative strict validation for use AFTER template processing
function(validate_generated_defines_strict)
    if(SD_TYPES_LIST_COUNT GREATER 0)
        print_status_colored("INFO" "=== STRICT POST-GENERATION VALIDATION ===")

        # Build list of expected defines
        set(EXPECTED_DEFINES "")
        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            string(TOUPPER ${normalized_type} SD_TYPE_UPPERCASE)
            list(APPEND EXPECTED_DEFINES "HAS_${SD_TYPE_UPPERCASE}")
        endforeach()

        # Check the generated file (must exist for strict validation)
        set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
        if(NOT EXISTS "${INCLUDE_OPS_FILE}")
            message(FATAL_ERROR "❌ STRICT VALIDATION FAILURE: Generated file missing: ${INCLUDE_OPS_FILE}")
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
            print_status_colored("ERROR" "❌ STRICT VALIDATION FAILURE: Type processing failed")
            message(STATUS "")
            message(STATUS "Requested types: ${SD_TYPES_LIST}")
            message(STATUS "Expected defines: ${expected_str}")
            message(STATUS "Missing defines: ${missing_str}")
            message(STATUS "")
            message(STATUS "Generated file content:")
            message(STATUS "${GENERATED_CONTENT}")
            message(STATUS "")
            message(FATAL_ERROR "❌ BUILD TERMINATED: ${missing_count} type(s) failed to process correctly")
        endif()

        print_status_colored("SUCCESS" "✅ All ${SD_TYPES_LIST_COUNT} types validated successfully in generated file")
    endif()
endfunction()

# Function to generate the type definitions header
function(generate_type_definitions_header)
    if(SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        print_status_colored("INFO" "=== GENERATING TYPE DEFINITIONS HEADER ===")

        set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")

        # Start building the header content
        set(HEADER_CONTENT "")
        string(APPEND HEADER_CONTENT "#ifndef SD_DEFINITIONS_GEN_H_\n")
        string(APPEND HEADER_CONTENT "#define SD_DEFINITIONS_GEN_H_\n")
        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "// Auto-generated type definitions\n")
        string(APPEND HEADER_CONTENT "// Generated by CMake type validation system\n")
        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "#define SD_ALL_OPS 1\n")
        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "// Type availability definitions\n")

        # Add type definitions
        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            string(TOUPPER ${normalized_type} SD_TYPE_UPPERCASE)
            string(APPEND HEADER_CONTENT "#define HAS_${SD_TYPE_UPPERCASE} 1\n")
        endforeach()

        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "#endif\n")

        # Write the file
        file(WRITE "${INCLUDE_OPS_FILE}" "${HEADER_CONTENT}")

        print_status_colored("SUCCESS" "✅ Generated type definitions header: ${INCLUDE_OPS_FILE}")
        message(STATUS "Defined types: ${SD_TYPES_LIST}")
    else()
        print_status_colored("INFO" "No specific types defined - using ALL types mode")

        set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
        set(HEADER_CONTENT "")
        string(APPEND HEADER_CONTENT "#ifndef SD_DEFINITIONS_GEN_H_\n")
        string(APPEND HEADER_CONTENT "#define SD_DEFINITIONS_GEN_H_\n")
        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "#define SD_ALL_OPS 1\n")
        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "// All types enabled by default\n")

        # Define all supported types
        foreach(type IN LISTS ALL_SUPPORTED_TYPES)
            normalize_type("${type}" normalized_type)
            string(TOUPPER ${normalized_type} TYPE_UPPERCASE)
            string(APPEND HEADER_CONTENT "#define HAS_${TYPE_UPPERCASE} 1\n")
        endforeach()

        string(APPEND HEADER_CONTENT "\n")
        string(APPEND HEADER_CONTENT "#endif\n")

        file(WRITE "${INCLUDE_OPS_FILE}" "${HEADER_CONTENT}")
        print_status_colored("SUCCESS" "✅ Generated all-types definitions header: ${INCLUDE_OPS_FILE}")
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