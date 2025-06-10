# =============================================================================
# TypeValidation.cmake - Comprehensive type validation for libnd4j
# =============================================================================


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

# Function to create a type configuration summary file
function(generate_type_config_summary)
    set(CONFIG_FILE "${CMAKE_BINARY_DIR}/type_configuration_summary.txt")

    file(WRITE "${CONFIG_FILE}" "LibND4J Type Configuration Summary\n")
    file(APPEND "${CONFIG_FILE}" "=====================================\n\n")
    file(APPEND "${CONFIG_FILE}" "Generated: ${CMAKE_CURRENT_LIST_DIR}\n")
    file(APPEND "${CONFIG_FILE}" "Build Type: ${CMAKE_BUILD_TYPE}\n")
    file(APPEND "${CONFIG_FILE}" "Platform: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}\n")
    file(APPEND "${CONFIG_FILE}" "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n\n")

    if(SD_TYPES_LIST_COUNT GREATER 0)
        file(APPEND "${CONFIG_FILE}" "Type Selection: SELECTIVE\n")
        file(APPEND "${CONFIG_FILE}" "Selected Types (${SD_TYPES_LIST_COUNT}):\n")
        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            file(APPEND "${CONFIG_FILE}" "  - ${normalized_type}")
            if(NOT SD_TYPE STREQUAL normalized_type)
                file(APPEND "${CONFIG_FILE}" " (from ${SD_TYPE})")
            endif()
            file(APPEND "${CONFIG_FILE}" "\n")
        endforeach()
    else()
        file(APPEND "${CONFIG_FILE}" "Type Selection: ALL\n")
        file(APPEND "${CONFIG_FILE}" "Building with all supported data types\n")
    endif()

    if(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "")
        file(APPEND "${CONFIG_FILE}" "\nDebug Type Profile: ${SD_DEBUG_TYPE_PROFILE}\n")
    endif()

    if(SD_GCC_FUNCTRACE STREQUAL "ON")
        file(APPEND "${CONFIG_FILE}" "Function Tracing: ENABLED\n")
        if(SD_DEBUG_AUTO_REDUCE)
            file(APPEND "${CONFIG_FILE}" "Debug Auto-Reduction: ENABLED\n")
        endif()
    endif()

    file(APPEND "${CONFIG_FILE}" "\nValidation Mode: ${SD_TYPES_VALIDATION_MODE}\n")

    message(STATUS "Type configuration summary written to: ${CONFIG_FILE}")
endfunction()

# Function to validate CMake variables and provide helpful messages
function(validate_cmake_type_variables)
    # Check for common variable naming mistakes
    if(DEFINED SD_TYPE_LIST AND NOT DEFINED SD_TYPES_LIST)
        message(WARNING "Found SD_TYPE_LIST but expected SD_TYPES_LIST. Did you mean SD_TYPES_LIST?")
    endif()

    if(DEFINED LIBND4J_DATATYPES AND NOT DEFINED SD_TYPES_LIST)
        message(STATUS "Converting LIBND4J_DATATYPES to SD_TYPES_LIST")
        set(SD_TYPES_LIST "${LIBND4J_DATATYPES}" PARENT_SCOPE)
    endif()

    # Validate that list variables are properly formatted
    if(SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        # Check for common formatting issues
        if(SD_TYPES_LIST MATCHES ".*,.*")
            message(WARNING "SD_TYPES_LIST contains commas. Expected semicolon-separated list.")
            string(REPLACE "," ";" SD_TYPES_LIST "${SD_TYPES_LIST}")
            set(SD_TYPES_LIST "${SD_TYPES_LIST}" PARENT_SCOPE)
            message(STATUS "Converted comma-separated to semicolon-separated list")
        endif()

        # Check for whitespace issues
        set(CLEANED_TYPES_LIST "")
        foreach(TYPE_ITEM ${SD_TYPES_LIST})
            string(STRIP "${TYPE_ITEM}" CLEANED_TYPE)
            if(NOT CLEANED_TYPE STREQUAL "")
                list(APPEND CLEANED_TYPES_LIST "${CLEANED_TYPE}")
            endif()
        endforeach()

        list(LENGTH CLEANED_TYPES_LIST CLEANED_COUNT)
        list(LENGTH SD_TYPES_LIST ORIGINAL_COUNT)

        if(NOT CLEANED_COUNT EQUAL ORIGINAL_COUNT)
            message(STATUS "Cleaned up whitespace in type list")
            set(SD_TYPES_LIST "${CLEANED_TYPES_LIST}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Function to set up type-based compile definitions
function(setup_type_definitions)
    if(SD_TYPES_LIST_COUNT GREATER 0)
        # Create normalized list for definitions
        set(NORMALIZED_TYPES "")
        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            list(APPEND NORMALIZED_TYPES "${normalized_type}")
        endforeach()

        # Remove duplicates
        list(REMOVE_DUPLICATES NORMALIZED_TYPES)

        # Set up compile definitions
        foreach(NORM_TYPE ${NORMALIZED_TYPES})
            string(TOUPPER "${NORM_TYPE}" TYPE_UPPER)
            add_compile_definitions(HAS_${TYPE_UPPER})

            # Handle special cases
            if(NORM_TYPE STREQUAL "float32")
                add_compile_definitions(HAS_FLOAT)
            elseif(NORM_TYPE STREQUAL "float16")
                add_compile_definitions(HAS_HALF)
            elseif(NORM_TYPE STREQUAL "int64")
                add_compile_definitions(HAS_LONG)
            elseif(NORM_TYPE STREQUAL "uint64")
                add_compile_definitions(HAS_UNSIGNEDLONG)
            elseif(NORM_TYPE STREQUAL "int32")
                add_compile_definitions(HAS_INT)
            elseif(NORM_TYPE STREQUAL "bfloat16")
                add_compile_definitions(HAS_BFLOAT)
            elseif(NORM_TYPE STREQUAL "double")
                add_compile_definitions(HAS_FLOAT64)
            endif()
        endforeach()

        # Set selective types flag
        add_compile_definitions(SD_SELECTIVE_TYPES)

        # Store normalized list for later use
        set(SD_NORMALIZED_TYPES_LIST "${NORMALIZED_TYPES}" PARENT_SCOPE)
    endif()
endfunction()

# Function to generate type mapping header
function(generate_type_mapping_header)
    set(HEADER_FILE "${CMAKE_BINARY_DIR}/include/type_mapping_generated.h")

    file(WRITE "${HEADER_FILE}" "/* Generated type mapping header */\n")
    file(APPEND "${HEADER_FILE}" "#ifndef LIBND4J_TYPE_MAPPING_GENERATED_H\n")
    file(APPEND "${HEADER_FILE}" "#define LIBND4J_TYPE_MAPPING_GENERATED_H\n\n")

    file(APPEND "${HEADER_FILE}" "/* Build-time type configuration */\n")
    if(SD_TYPES_LIST_COUNT GREATER 0)
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_SELECTIVE_TYPES 1\n")
        file(APPEND "${HEADER_FILE}" "#define SD_SELECTED_TYPE_COUNT ${SD_TYPES_LIST_COUNT}\n\n")

        file(APPEND "${HEADER_FILE}" "/* Selected types */\n")
        set(TYPE_INDEX 0)
        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            string(TOUPPER "${normalized_type}" TYPE_UPPER)
            file(APPEND "${HEADER_FILE}" "#define SD_SELECTED_TYPE_${TYPE_INDEX} ${TYPE_UPPER}\n")
            math(EXPR TYPE_INDEX "${TYPE_INDEX} + 1")
        endforeach()
    else()
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_SELECTIVE_TYPES 0\n")
        file(APPEND "${HEADER_FILE}" "#define SD_SELECTED_TYPE_COUNT 0\n")
    endif()

    file(APPEND "${HEADER_FILE}" "\n#endif /* LIBND4J_TYPE_MAPPING_GENERATED_H */\n")

    message(STATUS "Generated type mapping header: ${HEADER_FILE}")
endfunction()

# Function to validate type consistency across build
function(validate_type_consistency)
    # Check that essential combinations are possible
    if(SD_TYPES_LIST_COUNT GREATER 0)
        set(HAS_INTEGER_TYPE FALSE)
        set(HAS_FLOAT_TYPE FALSE)

        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)

            # Check for integer types
            if(normalized_type MATCHES "^(int|uint)[0-9]+$" OR normalized_type STREQUAL "bool")
                set(HAS_INTEGER_TYPE TRUE)
            endif()

            # Check for floating point types
            if(normalized_type MATCHES "^(float|double|bfloat)[0-9]*$" OR normalized_type STREQUAL "double")
                set(HAS_FLOAT_TYPE TRUE)
            endif()
        endforeach()

        if(NOT HAS_INTEGER_TYPE)
            print_status_colored("WARNING" "No integer types selected - this may limit functionality")
        endif()

        if(NOT HAS_FLOAT_TYPE)
            print_status_colored("WARNING" "No floating point types selected - this may limit ML/AI operations")
        endif()
    endif()
endfunction()

# Main function that orchestrates all type validation
function(libnd4j_validate_and_setup_types)
    print_status_colored("INFO" "=== LIBND4J TYPE VALIDATION SYSTEM ===")

    # Step 1: Validate and clean up CMake variables
    validate_cmake_type_variables()

    # Step 2: Update SD_TYPES_LIST_COUNT after cleanup
    if(SD_TYPES_LIST)
        list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
        set(SD_TYPES_LIST_COUNT "${SD_TYPES_LIST_COUNT}" PARENT_SCOPE)
    else()
        set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)
    endif()

    # Step 3: Run main validation
    validate_and_process_types()

    # Step 4: Set up compile definitions
    setup_type_definitions()

    # Step 5: Validate consistency
    validate_type_consistency()

    # Step 6: Generate additional files
    generate_type_mapping_header()
    generate_type_config_summary()

    print_status_colored("SUCCESS" "Type validation and setup completed successfully")
endfunction()

# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

# Macro to easily add type validation to existing CMakeLists.txt
macro(LIBND4J_SETUP_TYPE_VALIDATION)
    # Set up paths
    if(NOT DEFINED CMAKE_VALIDATION_DIR)
        set(CMAKE_VALIDATION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    endif()

    # Include validation if not already included
    if(NOT COMMAND validate_and_process_types)
        if(EXISTS "${CMAKE_VALIDATION_DIR}/TypeValidation.cmake")
            include("${CMAKE_VALIDATION_DIR}/TypeValidation.cmake")
        else()
            message(WARNING "TypeValidation.cmake not found at ${CMAKE_VALIDATION_DIR}")
        endif()
    endif()

    # Run validation
    libnd4j_validate_and_setup_types()
endmacro()

# Function to print usage help
function(print_type_validation_help)
    message(STATUS "")
    message(STATUS "LibND4J Type Validation System Help")
    message(STATUS "===================================")
    message(STATUS "")
    message(STATUS "CMake Variables:")
    message(STATUS "  SD_TYPES_LIST              - Semicolon-separated list of types")
    message(STATUS "  SD_STRICT_TYPE_VALIDATION  - Enable strict validation (ON/OFF)")
    message(STATUS "  SD_DEBUG_TYPE_PROFILE      - Debug type profile name")
    message(STATUS "  SD_DEBUG_CUSTOM_TYPES      - Custom types for debug profile")
    message(STATUS "  SD_DEBUG_AUTO_REDUCE       - Auto-reduce types for debug (ON/OFF)")
    message(STATUS "")
    message(STATUS "Example Usage:")
    message(STATUS "  cmake -DSD_TYPES_LIST=\"float32;double;int32;int64\" ..")
    message(STATUS "  cmake -DSD_DEBUG_TYPE_PROFILE=MINIMAL_INDEXING ..")
    message(STATUS "  cmake -DSD_STRICT_TYPE_VALIDATION=ON ..")
    message(STATUS "")
    message(STATUS "For more information, see the documentation.")
    message(STATUS "")
endfunction()

# Check if help was requested
if(LIBND4J_TYPE_VALIDATION_HELP)
    print_type_validation_help()
endif()