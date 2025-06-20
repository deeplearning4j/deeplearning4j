# =============================================================================
# TypeValidation.cmake - Enhanced with ML-aware pattern recognition
# =============================================================================

# Include required modules (with proper error handling)
# Remove this include - TypeCombinationEngine will be included from CMakeLists.txt
# TypeCombinationEngine functions will be available when needed

# Global variables for type combinations
set(GENERATED_TYPE_COMBINATIONS "" CACHE INTERNAL "Generated type combinations")
set(PROCESSED_TEMPLATE_FILES "" CACHE INTERNAL "Processed template files")

# Enhanced function to validate ML-specific type combinations
function(validate_ml_type_combinations types_list)
    set(valid_combinations "")
    set(invalid_combinations "")

    foreach(type1 ${types_list})
        foreach(type2 ${types_list})
            foreach(type3 ${types_list})
                if(COMMAND is_semantically_valid_combination)
                    is_semantically_valid_combination(${type1} ${type2} ${type3} "ml_validation" is_valid)
                    if(is_valid)
                        list(APPEND valid_combinations "${type1},${type2},${type3}")
                    else()
                        list(APPEND invalid_combinations "${type1},${type2},${type3}")
                    endif()
                endif()
            endforeach()
        endforeach()
    endforeach()

    list(LENGTH valid_combinations valid_count)
    list(LENGTH invalid_combinations invalid_count)

    message(STATUS "ML Type Combination Validation:")
    message(STATUS "  Valid combinations: ${valid_count}")
    message(STATUS "  Invalid combinations: ${invalid_count}")

    if(invalid_count GREATER 0 AND valid_count GREATER 0)
        math(EXPR reduction_percent "100 * ${invalid_count} / (${valid_count} + ${invalid_count})")
        message(STATUS "  Semantic filtering reduced combinations by ${reduction_percent}%")
    endif()
endfunction()

# Enhanced function to estimate combination impact with ML considerations
function(estimate_combination_impact types_list operation_types result_var)
    list(LENGTH types_list type_count)

    set(total_combinations 0)
    set(filtered_combinations 0)

    foreach(op_type ${operation_types})
        if(op_type STREQUAL "pairwise" OR op_type STREQUAL "transform")
            # 3-type combinations: n^3
            math(EXPR op_combinations "${type_count} * ${type_count} * ${type_count}")

            # Apply semantic filtering estimation
            if(op_type STREQUAL "pairwise")
                # Pairwise operations typically have 70% valid combinations
                math(EXPR filtered_op_combinations "${op_combinations} * 70 / 100")
            else()
                # Transform operations typically have 60% valid combinations
                math(EXPR filtered_op_combinations "${op_combinations} * 60 / 100")
            endif()
        else()
            # 2-type combinations: n^2
            math(EXPR op_combinations "${type_count} * ${type_count}")
            # 2-type operations typically have 80% valid combinations
            math(EXPR filtered_op_combinations "${op_combinations} * 80 / 100")
        endif()

        math(EXPR total_combinations "${total_combinations} + ${op_combinations}")
        math(EXPR filtered_combinations "${filtered_combinations} + ${filtered_op_combinations}")
    endforeach()

    set(impact_info "")
    string(APPEND impact_info "Total combinations: ${total_combinations}\n")
    string(APPEND impact_info "Filtered combinations: ${filtered_combinations}\n")

    if(total_combinations GREATER 0)
        math(EXPR reduction_percent "100 * (${total_combinations} - ${filtered_combinations}) / ${total_combinations}")
        string(APPEND impact_info "Reduction: ${reduction_percent}%\n")
    endif()

    set(${result_var} "${impact_info}" PARENT_SCOPE)
endfunction()

# Function to suggest type optimizations based on workload
function(suggest_type_optimizations current_types target_workload result_var)
    set(suggestions "")

    if(target_workload STREQUAL "quantization")
        # Check for quantization-friendly types
        set(has_int8 FALSE)
        set(has_uint8 FALSE)
        set(has_float32 FALSE)

        foreach(type ${current_types})
            if(type STREQUAL "int8_t")
                set(has_int8 TRUE)
            elseif(type STREQUAL "uint8_t")
                set(has_uint8 TRUE)
            elseif(type STREQUAL "float")
                set(has_float32 TRUE)
            endif()
        endforeach()

        if(NOT has_int8)
            list(APPEND suggestions "Add int8_t for quantization operations")
        endif()
        if(NOT has_uint8)
            list(APPEND suggestions "Add uint8_t for unsigned quantization")
        endif()
        if(NOT has_float32)
            list(APPEND suggestions "Add float for quantization scale/zero-point operations")
        endif()

        list(APPEND suggestions "Focus on INT8/UINT8 + FP32 combinations for quantization")
        list(APPEND suggestions "Preserve quantization patterns: (float, float, int8), (int8, float, float)")

    elseif(target_workload STREQUAL "training")
        # Check for mixed precision training types
        set(has_float16 FALSE)
        set(has_bfloat16 FALSE)
        set(has_float32 FALSE)

        foreach(type ${current_types})
            if(type STREQUAL "float16")
                set(has_float16 TRUE)
            elseif(type STREQUAL "bfloat16")
                set(has_bfloat16 TRUE)
            elseif(type STREQUAL "float")
                set(has_float32 TRUE)
            endif()
        endforeach()

        if(NOT has_float16 AND NOT has_bfloat16)
            list(APPEND suggestions "Add float16 or bfloat16 for mixed precision training")
        endif()
        if(NOT has_float32)
            list(APPEND suggestions "Add float32 for gradient accumulation")
        endif()

        list(APPEND suggestions "Optimize for mixed precision (FP16/FP32) combinations")
        list(APPEND suggestions "Include gradient accumulation types")
        list(APPEND suggestions "Preserve high precision combinations for training stability")

    elseif(target_workload STREQUAL "inference")
        list(APPEND suggestions "Focus on deployment-optimized types")
        list(APPEND suggestions "Optimize for quantized inference patterns")
        list(APPEND suggestions "Remove training-specific combinations")
        list(APPEND suggestions "Prioritize int8_t, uint8_t, float16, float combinations")

    elseif(target_workload STREQUAL "nlp")
        # Check for string types
        set(has_string FALSE)
        foreach(type ${current_types})
            if(type MATCHES "std::.*string")
                set(has_string TRUE)
                break()
            endif()
        endforeach()

        if(NOT has_string)
            list(APPEND suggestions "Add std::string for tokenization operations")
            list(APPEND suggestions "Consider std::u16string for Unicode support")
        endif()

        list(APPEND suggestions "Optimize string + numeric combinations for tokenization")
        list(APPEND suggestions "Include int32_t/int64_t for token indices")
        list(APPEND suggestions "Preserve text encoding type combinations")

    elseif(target_workload STREQUAL "cv")
        # Check for image processing types
        set(has_uint8 FALSE)
        set(has_float16 FALSE)

        foreach(type ${current_types})
            if(type STREQUAL "uint8_t")
                set(has_uint8 TRUE)
            elseif(type STREQUAL "float16")
                set(has_float16 TRUE)
            endif()
        endforeach()

        if(NOT has_uint8)
            list(APPEND suggestions "Add uint8_t for image pixel data")
        endif()
        if(NOT has_float16)
            list(APPEND suggestions "Add float16 for memory-efficient convolutions")
        endif()

        list(APPEND suggestions "Optimize for image processing patterns")
        list(APPEND suggestions "Focus on uint8_t, float16, float combinations")
        list(APPEND suggestions "Optimize convolution operations")
    endif()

    set(${result_var} "${suggestions}" PARENT_SCOPE)
endfunction()

# Enhanced normalization with quantization support
function(normalize_type_enhanced input_type output_var)
    set(normalized_type "${input_type}")

    # Standard aliases
    if(normalized_type STREQUAL "float32")
        set(normalized_type "float")
    elseif(normalized_type STREQUAL "float64")
        set(normalized_type "double")
    elseif(normalized_type STREQUAL "half")
        set(normalized_type "float16")
    elseif(normalized_type STREQUAL "long")
        set(normalized_type "int64_t")
    elseif(normalized_type STREQUAL "int")
        set(normalized_type "int32_t")
    elseif(normalized_type STREQUAL "bfloat")
        set(normalized_type "bfloat16")
        # Quantization-specific aliases
    elseif(normalized_type STREQUAL "qint8")
        set(normalized_type "int8_t")
    elseif(normalized_type STREQUAL "quint8")
        set(normalized_type "uint8_t")
    elseif(normalized_type STREQUAL "qint16")
        set(normalized_type "int16_t")
    elseif(normalized_type STREQUAL "quint16")
        set(normalized_type "uint16_t")
        # String type aliases
    elseif(normalized_type STREQUAL "utf8")
        set(normalized_type "std::string")
    elseif(normalized_type STREQUAL "utf16")
        set(normalized_type "std::u16string")
    elseif(normalized_type STREQUAL "utf32")
        set(normalized_type "std::u32string")
    endif()

    set(${output_var} "${normalized_type}" PARENT_SCOPE)
endfunction()

# Function to add semantic validation beyond syntax checking
function(validate_semantic_patterns types_list result_var)
    set(warnings "")
    set(errors "")

    # Check for problematic patterns
    set(has_string_types FALSE)
    set(string_count 0)
    set(has_quantization_types FALSE)
    set(has_float_types FALSE)

    foreach(type ${types_list})
        normalize_type_enhanced("${type}" normalized_type)

        # Count string types
        if(normalized_type MATCHES "std::.*string")
            set(has_string_types TRUE)
            math(EXPR string_count "${string_count} + 1")
        endif()

        # Check for quantization types
        if(normalized_type MATCHES "int8_t|uint8_t")
            set(has_quantization_types TRUE)
        endif()

        # Check for floating point types
        if(normalized_type MATCHES "float|double|bfloat16|float16")
            set(has_float_types TRUE)
        endif()
    endforeach()

    # Semantic validation rules
    if(has_string_types AND string_count GREATER 2)
        list(APPEND warnings "Multiple string types detected - may lead to excessive string-to-string combinations")
    endif()

    if(has_quantization_types AND NOT has_float_types)
        list(APPEND errors "Quantization types without floating point types - quantization patterns impossible")
    endif()

    if(has_string_types AND NOT has_quantization_types AND NOT has_float_types)
        list(APPEND warnings "String types without numeric types - limited ML functionality")
    endif()

    # Check for balance
    list(LENGTH types_list total_types)
    if(total_types GREATER 10)
        list(APPEND warnings "Large number of types (${total_types}) may result in excessive combinations")
    endif()

    set(validation_result "")
    if(warnings)
        string(APPEND validation_result "WARNINGS:\n")
        foreach(warning ${warnings})
            string(APPEND validation_result "  - ${warning}\n")
        endforeach()
    endif()

    if(errors)
        string(APPEND validation_result "ERRORS:\n")
        foreach(error ${errors})
            string(APPEND validation_result "  - ${error}\n")
        endforeach()
    endif()

    if(NOT warnings AND NOT errors)
        string(APPEND validation_result "All semantic patterns are valid\n")
    endif()

    set(${result_var} "${validation_result}" PARENT_SCOPE)
endfunction()

# Enhanced get_all_types function with ML awareness
function(get_all_types result_var)
    set(all_types
            "bool" "int8_t" "uint8_t" "int16_t" "uint16_t"
            "int32_t" "uint32_t" "int64_t" "uint64_t"
            "float16" "bfloat16" "float" "double"
    )

    # Add string types if enabled
    if(SD_ENABLE_STRING_OPERATIONS)
        list(APPEND all_types "std::string" "std::u16string" "std::u32string")
    endif()

    set(${result_var} "${all_types}" PARENT_SCOPE)
endfunction()

# ML pattern recognition for type priority
function(get_ml_type_priority type_name result_var)
    normalize_type_enhanced("${type_name}" normalized_type)

    # Priority ranking for ML workloads (higher = more important)
    if(normalized_type STREQUAL "float")
        set(priority 10)  # Essential for most ML operations
    elseif(normalized_type STREQUAL "int32_t")
        set(priority 9)   # Critical for indexing
    elseif(normalized_type STREQUAL "int64_t")
        set(priority 9)   # Critical for indexing
    elseif(normalized_type STREQUAL "double")
        set(priority 8)   # High precision operations
    elseif(normalized_type STREQUAL "int8_t")
        set(priority 7)   # Quantization
    elseif(normalized_type STREQUAL "uint8_t")
        set(priority 7)   # Quantization, image data
    elseif(normalized_type STREQUAL "float16")
        set(priority 6)   # Mixed precision
    elseif(normalized_type STREQUAL "bfloat16")
        set(priority 6)   # ML training
    elseif(normalized_type STREQUAL "bool")
        set(priority 5)   # Masking operations
    elseif(normalized_type MATCHES "std::.*string")
        set(priority 4)   # NLP operations
    else()
        set(priority 3)   # Other types
    endif()

    set(${result_var} ${priority} PARENT_SCOPE)
endfunction()

# Function to sort types by ML importance
function(sort_types_by_ml_priority types_list result_var)
    set(type_priority_pairs "")

    foreach(type ${types_list})
        get_ml_type_priority("${type}" priority)
        list(APPEND type_priority_pairs "${priority}:${type}")
    endforeach()

    # Sort by priority (descending)
    list(SORT type_priority_pairs COMPARE NATURAL ORDER DESCENDING)

    set(sorted_types "")
    foreach(pair ${type_priority_pairs})
        string(REGEX REPLACE "^[0-9]+:" "" type "${pair}")
        list(APPEND sorted_types "${type}")
    endforeach()

    set(${result_var} "${sorted_types}" PARENT_SCOPE)
endfunction()

# =============================================================================
# ORIGINAL FUNCTIONS (Enhanced)
# =============================================================================

# Type aliases for normalization
set(TYPE_ALIAS_float "float32")
set(TYPE_ALIAS_half "float16")
set(TYPE_ALIAS_long "int64")
set(TYPE_ALIAS_unsignedlong "uint64")
set(TYPE_ALIAS_int "int32")
set(TYPE_ALIAS_bfloat "bfloat16")
set(TYPE_ALIAS_float64 "double")

# All supported types list (enhanced)
set(ALL_SUPPORTED_TYPES
        "bool" "int8" "uint8" "int16" "uint16" "int32" "uint32"
        "int64" "uint64" "float16" "bfloat16" "float32" "double"
        "float" "half" "long" "unsignedlong" "int" "bfloat" "float64"
        "utf8" "utf16" "utf32"
        # Quantization aliases
        "qint8" "quint8" "qint16" "quint16"
)

# Minimum required types for basic functionality (enhanced)
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

# Function to normalize a type name (enhanced)
function(normalize_type input_type output_var)
    normalize_type_enhanced("${input_type}" result)
    set(${output_var} "${result}" PARENT_SCOPE)
endfunction()

# Enhanced function to check if a type is supported
function(is_type_supported type result_var)
    normalize_type("${type}" normalized_type)

    if("${normalized_type}" IN_LIST ALL_SUPPORTED_TYPES)
        set(${result_var} TRUE PARENT_SCOPE)
    else()
        set(${result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Enhanced show_available_types function
function(show_available_types)
    message(STATUS "")
    if(COMMAND print_status_colored)
        print_status_colored("INFO" "=== AVAILABLE DATA TYPES ===")
    else()
        message(STATUS "=== AVAILABLE DATA TYPES ===")
    endif()
    message(STATUS "")
    message(STATUS "Core Types:")
    message(STATUS "  bool     - Boolean type")
    message(STATUS "  int8     - 8-bit signed integer (quantization)")
    message(STATUS "  uint8    - 8-bit unsigned integer (quantization, images)")
    message(STATUS "  int16    - 16-bit signed integer")
    message(STATUS "  uint16   - 16-bit unsigned integer")
    message(STATUS "  int32    - 32-bit signed integer (indexing)")
    message(STATUS "  uint32   - 32-bit unsigned integer")
    message(STATUS "  int64    - 64-bit signed integer (indexing)")
    message(STATUS "  uint64   - 64-bit unsigned integer")
    message(STATUS "")
    message(STATUS "Floating Point Types:")
    message(STATUS "  float16  - 16-bit floating point (mixed precision)")
    message(STATUS "  bfloat16 - 16-bit brain floating point (ML training)")
    message(STATUS "  float32  - 32-bit floating point (primary ML type)")
    message(STATUS "  double   - 64-bit floating point (high precision)")
    message(STATUS "")
    message(STATUS "String Types (NLP):")
    message(STATUS "  utf8     - UTF-8 strings")
    message(STATUS "  utf16    - UTF-16 strings")
    message(STATUS "  utf32    - UTF-32 strings")
    message(STATUS "")
    message(STATUS "Type Aliases:")
    message(STATUS "  float    -> float32")
    message(STATUS "  half     -> float16")
    message(STATUS "  long     -> int64")
    message(STATUS "  int      -> int32")
    message(STATUS "  bfloat   -> bfloat16")
    message(STATUS "  float64  -> double")
    message(STATUS "")
    message(STATUS "Quantization Aliases:")
    message(STATUS "  qint8    -> int8_t")
    message(STATUS "  quint8   -> uint8_t")
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
        if(COMMAND print_status_colored)
            print_status_colored("WARNING" "Unknown debug profile '${profile}', using MINIMAL_INDEXING")
        else()
            message(WARNING "Unknown debug profile '${profile}', using MINIMAL_INDEXING")
        endif()
        set(${result_var} "${DEBUG_PROFILE_MINIMAL_INDEXING}" PARENT_SCOPE)
    endif()
endfunction()

# Enhanced estimate_build_impact with ML considerations
function(estimate_build_impact types_string build_type)
    if(NOT types_string OR types_string STREQUAL "" OR types_string STREQUAL "all" OR types_string STREQUAL "ALL")
        if(COMMAND print_status_colored)
            print_status_colored("INFO" "=== BUILD IMPACT ESTIMATION ===")
        else()
            message(STATUS "=== BUILD IMPACT ESTIMATION ===")
        endif()
        message(STATUS "Using ALL types - expect full compilation with all template instantiations")
        return()
    endif()

    string(REPLACE ";" ";" TYPES_LIST "${types_string}")
    list(LENGTH TYPES_LIST type_count)

    if(type_count GREATER 0)
        # Get impact estimation using ML-aware filtering
        set(operation_types "pairwise" "reduction" "indexreduction" "transform" "generic")
        estimate_combination_impact("${TYPES_LIST}" "${operation_types}" impact_info)

        # Parse the impact info
        string(REGEX MATCH "Total combinations: ([0-9]+)" _ "${impact_info}")
        if(CMAKE_MATCH_1)
            set(est_total_combinations ${CMAKE_MATCH_1})
        else()
            set(est_total_combinations 0)
        endif()

        string(REGEX MATCH "Filtered combinations: ([0-9]+)" _ "${impact_info}")
        if(CMAKE_MATCH_1)
            set(est_filtered_combinations ${CMAKE_MATCH_1})
        else()
            set(est_filtered_combinations 0)
        endif()

        math(EXPR est_binary_size_mb "${est_filtered_combinations} * 8 / 1000")  # Adjusted estimate

        if(COMMAND print_status_colored)
            print_status_colored("INFO" "=== BUILD IMPACT ESTIMATION (ML-Aware) ===")
        else()
            message(STATUS "=== BUILD IMPACT ESTIMATION (ML-Aware) ===")
        endif()
        message(STATUS "Type count: ${type_count}")
        message(STATUS "Total combinations (unfiltered): ${est_total_combinations}")
        message(STATUS "Filtered combinations (semantic): ${est_filtered_combinations}")
        message(STATUS "Estimated binary size: ~${est_binary_size_mb}MB")

        # Get semantic validation
        validate_semantic_patterns("${TYPES_LIST}" semantic_result)
        if(NOT semantic_result STREQUAL "All semantic patterns are valid\n")
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "Semantic Pattern Issues Detected:")
            else()
                message(WARNING "Semantic Pattern Issues Detected:")
            endif()
            message(STATUS "${semantic_result}")
        endif()

        if(build_type STREQUAL "Debug" AND est_filtered_combinations GREATER 125)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "HIGH COMBINATION COUNT DETECTED!")
                print_status_colored("WARNING" "${est_filtered_combinations} filtered combinations may cause:")
                print_status_colored("WARNING" "- Long build times")
                print_status_colored("WARNING" "- Large memory usage during compilation")
                print_status_colored("WARNING" "Consider using fewer types for debug builds:")
                print_status_colored("WARNING" "-DSD_DEBUG_TYPE_PROFILE=QUANTIZATION (4 types)")
                print_status_colored("WARNING" "-DSD_DEBUG_TYPE_PROFILE=MIXED_PRECISION (5 types)")
            else()
                message(WARNING "HIGH COMBINATION COUNT DETECTED!")
                message(WARNING "${est_filtered_combinations} filtered combinations may cause:")
                message(WARNING "- Long build times")
                message(WARNING "- Large memory usage during compilation")
                message(WARNING "Consider using fewer types for debug builds:")
                message(WARNING "-DSD_DEBUG_TYPE_PROFILE=QUANTIZATION (4 types)")
                message(WARNING "-DSD_DEBUG_TYPE_PROFILE=MIXED_PRECISION (5 types)")
            endif()
        elseif(est_binary_size_mb GREATER 500)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "Large binary size warning: ~${est_binary_size_mb}MB")
            else()
                message(WARNING "Large binary size warning: ~${est_binary_size_mb}MB")
            endif()
        endif()
    endif()
endfunction()

# Enhanced validate_type_list with ML awareness
function(validate_type_list types_string validation_mode)
    if(COMMAND print_status_colored)
        print_status_colored("INFO" "=== ENHANCED CMAKE TYPE VALIDATION ===")
    else()
        message(STATUS "=== ENHANCED CMAKE TYPE VALIDATION ===")
    endif()

    # Handle empty or special cases
    if(NOT types_string OR types_string STREQUAL "" OR types_string STREQUAL "all" OR types_string STREQUAL "ALL")
        if(validation_mode STREQUAL "STRICT")
            if(COMMAND print_status_colored)
                print_status_colored("ERROR" "No data types specified and strict mode enabled!")
            else()
                message(FATAL_ERROR "No data types specified and strict mode enabled!")
            endif()
        else()
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "No data types specified, using ALL types")
            else()
                message(WARNING "No data types specified, using ALL types")
            endif()
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
        if(COMMAND print_status_colored)
            print_status_colored("ERROR" "Found ${invalid_count} invalid type(s): ${invalid_types_str}")
        else()
            message(FATAL_ERROR "Found ${invalid_count} invalid type(s): ${invalid_types_str}")
        endif()
        show_available_types()
        message(FATAL_ERROR "Type validation failed!")
    endif()

    # Check for no valid types
    list(LENGTH valid_types valid_count)
    if(valid_count EQUAL 0)
        if(COMMAND print_status_colored)
            print_status_colored("ERROR" "No valid types found!")
        else()
            message(FATAL_ERROR "No valid types found!")
        endif()
        show_available_types()
        message(FATAL_ERROR "Type validation failed!")
    endif()

    # Enhanced validation with ML awareness
    validate_semantic_patterns("${normalized_types}" semantic_result)
    if(COMMAND print_status_colored)
        print_status_colored("INFO" "ML Semantic Validation:")
    else()
        message(STATUS "ML Semantic Validation:")
    endif()
    message(STATUS "${semantic_result}")

    # Sort types by ML priority
    sort_types_by_ml_priority("${normalized_types}" sorted_types)
    string(REPLACE ";" ", " sorted_types_str "${sorted_types}")
    message(STATUS "Types sorted by ML priority: ${sorted_types_str}")

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
        if(COMMAND print_status_colored)
            print_status_colored("WARNING" "Missing recommended essential types: ${missing_essential_str}")
            print_status_colored("WARNING" "Array indexing and basic operations may fail at runtime!")
        else()
            message(WARNING "Missing recommended essential types: ${missing_essential_str}")
            message(WARNING "Array indexing and basic operations may fail at runtime!")
        endif()

        if(validation_mode STREQUAL "STRICT")
            string(REPLACE ";" ", " required_types_str "${MINIMUM_REQUIRED_TYPES}")
            if(COMMAND print_status_colored)
                print_status_colored("ERROR" "Strict mode requires essential types: ${required_types_str}")
            else()
                message(FATAL_ERROR "Strict mode requires essential types: ${required_types_str}")
            endif()
            message(FATAL_ERROR "Essential types missing in strict mode!")
        endif()
    endif()

    # ML-specific validation
    if(COMMAND validate_ml_type_combinations)
        validate_ml_type_combinations("${normalized_types}")
    endif()

    # Check for excessive type combinations in debug builds
    if(validation_mode STREQUAL "DEBUG" AND valid_count GREATER 6)
        set(operation_types "pairwise" "reduction" "transform")
        estimate_combination_impact("${normalized_types}" "${operation_types}" impact_info)
        if(COMMAND print_status_colored)
            print_status_colored("WARNING" "Debug build impact analysis:")
        else()
            message(WARNING "Debug build impact analysis:")
        endif()
        message(STATUS "${impact_info}")
        if(COMMAND print_status_colored)
            print_status_colored("WARNING" "Consider using a debug type profile: -DSD_DEBUG_TYPE_PROFILE=QUANTIZATION")
        else()
            message(WARNING "Consider using a debug type profile: -DSD_DEBUG_TYPE_PROFILE=QUANTIZATION")
        endif()
    endif()

    string(REPLACE ";" ", " normalized_types_str "${normalized_types}")
    if(COMMAND print_status_colored)
        print_status_colored("SUCCESS" "Enhanced type validation passed: ${valid_count} valid types")
    else()
        message(STATUS "Enhanced type validation passed: ${valid_count} valid types")
    endif()
    message(STATUS "Selected types: ${normalized_types_str}")
endfunction()

# FAIL-FAST TYPE VALIDATION
function(validate_generated_defines_failfast)
    if(SD_TYPES_LIST_COUNT GREATER 0)
        if(COMMAND print_status_colored)
            print_status_colored("INFO" "=== FAIL-FAST VALIDATION: Checking generated defines ===")
        else()
            message(STATUS "=== FAIL-FAST VALIDATION: Checking generated defines ===")
        endif()

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
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "Generated defines file will be created: ${INCLUDE_OPS_FILE}")
            else()
                message(WARNING "Generated defines file will be created: ${INCLUDE_OPS_FILE}")
            endif()
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
            if(COMMAND print_status_colored)
                print_status_colored("ERROR" "❌ VALIDATION FAILURE: Type processing failed")
                print_status_colored("ERROR" "")
                print_status_colored("ERROR" "Requested types: ${SD_TYPES_LIST}")
                print_status_colored("ERROR" "Expected defines: ${expected_str}")
                print_status_colored("ERROR" "Missing defines: ${missing_str}")
                print_status_colored("ERROR" "")
                print_status_colored("ERROR" "Generated file content:")
            else()
                message(FATAL_ERROR "❌ VALIDATION FAILURE: Type processing failed")
                message(FATAL_ERROR "")
                message(FATAL_ERROR "Requested types: ${SD_TYPES_LIST}")
                message(FATAL_ERROR "Expected defines: ${expected_str}")
                message(FATAL_ERROR "Missing defines: ${missing_str}")
                message(FATAL_ERROR "")
                message(FATAL_ERROR "Generated file content:")
            endif()
            message(STATUS "${GENERATED_CONTENT}")
            if(COMMAND print_status_colored)
                print_status_colored("ERROR" "")
            else()
                message(FATAL_ERROR "")
            endif()
            message(FATAL_ERROR "❌ BUILD TERMINATED: ${missing_count} type(s) failed to process correctly")
        endif()

        if(COMMAND print_status_colored)
            print_status_colored("SUCCESS" "✅ All ${SD_TYPES_LIST_COUNT} types validated successfully")
        else()
            message(STATUS "✅ All ${SD_TYPES_LIST_COUNT} types validated successfully")
        endif()
    endif()
endfunction()

function(validate_and_process_types_failfast)
    # Do the original processing
    validate_and_process_types()

    # Immediately validate the results
    validate_generated_defines_failfast()
endfunction()

# Main validation function to be called from CMakeLists.txt (enhanced)
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
        if(COMMAND print_status_colored)
            print_status_colored("INFO" "=== DEBUG BUILD TYPE REDUCTION ACTIVE ===")
        else()
            message(STATUS "=== DEBUG BUILD TYPE REDUCTION ACTIVE ===")
        endif()

        if(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "")
            resolve_debug_profile("${SD_DEBUG_TYPE_PROFILE}" "${SD_DEBUG_CUSTOM_TYPES}" resolved_types)
            set(SD_TYPES_LIST "${resolved_types}" PARENT_SCOPE)
            message(STATUS "Debug Profile: ${SD_DEBUG_TYPE_PROFILE}")
            message(STATUS "Resolved Types: ${resolved_types}")
        elseif(NOT SD_TYPES_LIST OR SD_TYPES_LIST STREQUAL "")
            # No types specified and no profile - use minimal safe default
            resolve_debug_profile("MINIMAL_INDEXING" "" resolved_types)
            set(SD_TYPES_LIST "${resolved_types}" PARENT_SCOPE)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "Auto-selected MINIMAL_INDEXING profile for debug build")
            else()
                message(WARNING "Auto-selected MINIMAL_INDEXING profile for debug build")
            endif()
            message(STATUS "Types: ${resolved_types}")
        endif()

        message(STATUS "=============================================")
    endif()

    # Validate the final datatypes
    if(SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        validate_type_list("${SD_TYPES_LIST}" "${validation_mode}")
        estimate_build_impact("${SD_TYPES_LIST}" "${CMAKE_BUILD_TYPE}")

        # Show configuration summary
        if(COMMAND print_status_colored)
            print_status_colored("INFO" "=== TYPE CONFIGURATION SUMMARY ===")
        else()
            message(STATUS "=== TYPE CONFIGURATION SUMMARY ===")
        endif()

        if(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "")
            message(STATUS "Debug Type Profile: ${SD_DEBUG_TYPE_PROFILE}")
        endif()

        message(STATUS "Type Selection: SELECTIVE")
        message(STATUS "Building with types: ${SD_TYPES_LIST}")
        message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
        message(STATUS "")
    else()
        if(COMMAND print_status_colored)
            print_status_colored("INFO" "=== TYPE CONFIGURATION SUMMARY ===")
        else()
            message(STATUS "=== TYPE CONFIGURATION SUMMARY ===")
        endif()
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

# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS (Enhanced)
# =============================================================================

# Function to create a type configuration summary file
function(generate_type_config_summary)
    set(CONFIG_FILE "${CMAKE_BINARY_DIR}/type_configuration_summary.txt")

    file(WRITE "${CONFIG_FILE}" "LibND4J Enhanced Type Configuration Summary\n")
    file(APPEND "${CONFIG_FILE}" "===============================================\n\n")
    file(APPEND "${CONFIG_FILE}" "Generated: ${CMAKE_CURRENT_LIST_DIR}\n")
    file(APPEND "${CONFIG_FILE}" "Build Type: ${CMAKE_BUILD_TYPE}\n")
    file(APPEND "${CONFIG_FILE}" "Platform: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}\n")
    file(APPEND "${CONFIG_FILE}" "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\n\n")

    if(SD_TYPES_LIST_COUNT GREATER 0)
        file(APPEND "${CONFIG_FILE}" "Type Selection: SELECTIVE (ML-Aware)\n")
        file(APPEND "${CONFIG_FILE}" "Selected Types (${SD_TYPES_LIST_COUNT}):\n")

        # Sort types by ML priority for the summary
        if(COMMAND sort_types_by_ml_priority)
            sort_types_by_ml_priority("${SD_TYPES_LIST}" sorted_types)
        else()
            set(sorted_types "${SD_TYPES_LIST}")
        endif()

        foreach(SD_TYPE ${sorted_types})
            normalize_type("${SD_TYPE}" normalized_type)
            if(COMMAND get_ml_type_priority)
                get_ml_type_priority("${normalized_type}" priority)
                file(APPEND "${CONFIG_FILE}" "  - ${normalized_type} (ML priority: ${priority})")
            else()
                file(APPEND "${CONFIG_FILE}" "  - ${normalized_type}")
            endif()
            if(NOT SD_TYPE STREQUAL normalized_type)
                file(APPEND "${CONFIG_FILE}" " [from ${SD_TYPE}]")
            endif()
            file(APPEND "${CONFIG_FILE}" "\n")
        endforeach()

        # Add semantic analysis
        if(COMMAND validate_semantic_patterns)
            validate_semantic_patterns("${SD_TYPES_LIST}" semantic_result)
            file(APPEND "${CONFIG_FILE}" "\nSemantic Analysis:\n${semantic_result}")
        endif()

        # Add combination impact
        if(COMMAND estimate_combination_impact)
            set(operation_types "pairwise" "reduction" "transform")
            estimate_combination_impact("${SD_TYPES_LIST}" "${operation_types}" impact_info)
            file(APPEND "${CONFIG_FILE}" "\nCombination Impact:\n${impact_info}")
        endif()

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

    # Add optimization recommendations
    if(SD_TYPES_LIST_COUNT GREATER 0 AND COMMAND suggest_type_optimizations)
        # Try to detect workload from types
        set(detected_workload "generic")
        if("int8_t" IN_LIST SD_TYPES_LIST AND "uint8_t" IN_LIST SD_TYPES_LIST)
            set(detected_workload "quantization")
        elseif("float16" IN_LIST SD_TYPES_LIST OR "bfloat16" IN_LIST SD_TYPES_LIST)
            set(detected_workload "training")
        endif()

        suggest_type_optimizations("${SD_TYPES_LIST}" "${detected_workload}" suggestions)
        if(suggestions)
            file(APPEND "${CONFIG_FILE}" "\nOptimization Recommendations:\n")
            foreach(suggestion ${suggestions})
                file(APPEND "${CONFIG_FILE}" "  - ${suggestion}\n")
            endforeach()
        endif()
    endif()

    message(STATUS "Enhanced type configuration summary written to: ${CONFIG_FILE}")
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

# Enhanced function to set up type-based compile definitions
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
            # Handle special characters in type names
            string(REPLACE "::" "_" TYPE_UPPER "${TYPE_UPPER}")
            string(REPLACE "T" "" TYPE_UPPER "${TYPE_UPPER}")
            add_compile_definitions(HAS_${TYPE_UPPER})

            # Handle special cases
            if(NORM_TYPE STREQUAL "float")
                add_compile_definitions(HAS_FLOAT32)
            elseif(NORM_TYPE STREQUAL "float16")
                add_compile_definitions(HAS_HALF)
            elseif(NORM_TYPE STREQUAL "int64_t")
                add_compile_definitions(HAS_LONG)
            elseif(NORM_TYPE STREQUAL "uint64_t")
                add_compile_definitions(HAS_UNSIGNEDLONG)
            elseif(NORM_TYPE STREQUAL "int32_t")
                add_compile_definitions(HAS_INT)
            elseif(NORM_TYPE STREQUAL "bfloat16")
                add_compile_definitions(HAS_BFLOAT)
            elseif(NORM_TYPE STREQUAL "double")
                add_compile_definitions(HAS_FLOAT64)
            elseif(NORM_TYPE MATCHES "std::.*string")
                add_compile_definitions(SD_ENABLE_STRING_OPERATIONS)
            endif()
        endforeach()

        # Set selective types flag
        add_compile_definitions(SD_SELECTIVE_TYPES)

        # Store normalized list for later use
        set(SD_NORMALIZED_TYPES_LIST "${NORMALIZED_TYPES}" PARENT_SCOPE)

        list(LENGTH NORMALIZED_TYPES list_length)
        message(STATUS "Set up ${list_length} type-specific compile definitions")
    endif()
endfunction()

# Enhanced function to generate type mapping header
function(generate_type_mapping_header)
    set(HEADER_FILE "${CMAKE_BINARY_DIR}/include/type_mapping_generated.h")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")

    file(WRITE "${HEADER_FILE}" "/* Generated type mapping header with ML awareness */\n")
    file(APPEND "${HEADER_FILE}" "#ifndef LIBND4J_TYPE_MAPPING_GENERATED_H\n")
    file(APPEND "${HEADER_FILE}" "#define LIBND4J_TYPE_MAPPING_GENERATED_H\n\n")

    file(APPEND "${HEADER_FILE}" "/* Build-time type configuration */\n")
    if(SD_TYPES_LIST_COUNT GREATER 0)
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_SELECTIVE_TYPES 1\n")
        file(APPEND "${HEADER_FILE}" "#define SD_SELECTED_TYPE_COUNT ${SD_TYPES_LIST_COUNT}\n\n")

        file(APPEND "${HEADER_FILE}" "/* Selected types (sorted by ML priority) */\n")
        if(COMMAND sort_types_by_ml_priority)
            sort_types_by_ml_priority("${SD_TYPES_LIST}" sorted_types)
        else()
            set(sorted_types "${SD_TYPES_LIST}")
        endif()

        set(TYPE_INDEX 0)
        foreach(SD_TYPE ${sorted_types})
            normalize_type("${SD_TYPE}" normalized_type)
            string(TOUPPER "${normalized_type}" TYPE_UPPER)
            string(REPLACE "::" "_" TYPE_UPPER "${TYPE_UPPER}")
            string(REPLACE "T" "" TYPE_UPPER "${TYPE_UPPER}")

            file(APPEND "${HEADER_FILE}" "#define SD_SELECTED_TYPE_${TYPE_INDEX} ${TYPE_UPPER}\n")
            file(APPEND "${HEADER_FILE}" "#define SD_TYPE_${TYPE_INDEX}_NAME \"${normalized_type}\"\n")

            if(COMMAND get_ml_type_priority)
                get_ml_type_priority("${normalized_type}" priority)
                file(APPEND "${HEADER_FILE}" "#define SD_TYPE_${TYPE_INDEX}_ML_PRIORITY ${priority}\n")
            endif()

            math(EXPR TYPE_INDEX "${TYPE_INDEX} + 1")
        endforeach()

        # Generate ML pattern detection macros
        file(APPEND "${HEADER_FILE}" "\n/* ML Pattern Detection Macros */\n")

        # Check for quantization types
        set(has_quantization FALSE)
        foreach(type ${SD_TYPES_LIST})
            if(type MATCHES "int8|uint8")
                set(has_quantization TRUE)
                break()
            endif()
        endforeach()

        if(has_quantization)
            file(APPEND "${HEADER_FILE}" "#define SD_HAS_QUANTIZATION_TYPES 1\n")
        else()
            file(APPEND "${HEADER_FILE}" "#define SD_HAS_QUANTIZATION_TYPES 0\n")
        endif()

        # Check for mixed precision types
        set(has_mixed_precision FALSE)
        foreach(type ${SD_TYPES_LIST})
            if(type MATCHES "float16|bfloat16")
                set(has_mixed_precision TRUE)
                break()
            endif()
        endforeach()

        if(has_mixed_precision)
            file(APPEND "${HEADER_FILE}" "#define SD_HAS_MIXED_PRECISION_TYPES 1\n")
        else()
            file(APPEND "${HEADER_FILE}" "#define SD_HAS_MIXED_PRECISION_TYPES 0\n")
        endif()

        # Check for string types
        set(has_string_types FALSE)
        foreach(type ${SD_TYPES_LIST})
            if(type MATCHES "std::.*string")
                set(has_string_types TRUE)
                break()
            endif()
        endforeach()

        if(has_string_types)
            file(APPEND "${HEADER_FILE}" "#define SD_HAS_STRING_TYPES 1\n")
        else()
            file(APPEND "${HEADER_FILE}" "#define SD_HAS_STRING_TYPES 0\n")
        endif()

    else()
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_SELECTIVE_TYPES 0\n")
        file(APPEND "${HEADER_FILE}" "#define SD_SELECTED_TYPE_COUNT 0\n")
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_QUANTIZATION_TYPES 1\n")
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_MIXED_PRECISION_TYPES 1\n")
        file(APPEND "${HEADER_FILE}" "#define SD_HAS_STRING_TYPES 1\n")
    endif()

    file(APPEND "${HEADER_FILE}" "\n#endif /* LIBND4J_TYPE_MAPPING_GENERATED_H */\n")

    message(STATUS "Generated enhanced type mapping header: ${HEADER_FILE}")
endfunction()

# Enhanced function to validate type consistency across build
function(validate_type_consistency)
    # Check that essential combinations are possible
    if(SD_TYPES_LIST_COUNT GREATER 0)
        set(HAS_INTEGER_TYPE FALSE)
        set(HAS_FLOAT_TYPE FALSE)
        set(HAS_INDEXING_TYPE FALSE)
        set(HAS_QUANTIZATION_TYPE FALSE)

        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)

            # Check for integer types
            if(normalized_type MATCHES "^(int|uint)[0-9]+_t$" OR normalized_type STREQUAL "bool")
                set(HAS_INTEGER_TYPE TRUE)
                if(normalized_type MATCHES "int32_t|int64_t")
                    set(HAS_INDEXING_TYPE TRUE)
                endif()
                if(normalized_type MATCHES "int8_t|uint8_t")
                    set(HAS_QUANTIZATION_TYPE TRUE)
                endif()
            endif()

            # Check for floating point types
            if(normalized_type MATCHES "^(float|double|bfloat)[0-9]*$" OR normalized_type STREQUAL "double")
                set(HAS_FLOAT_TYPE TRUE)
            endif()
        endforeach()

        if(NOT HAS_INTEGER_TYPE)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "No integer types selected - this may limit functionality")
            else()
                message(WARNING "No integer types selected - this may limit functionality")
            endif()
        endif()

        if(NOT HAS_FLOAT_TYPE)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "No floating point types selected - this may limit ML/AI operations")
            else()
                message(WARNING "No floating point types selected - this may limit ML/AI operations")
            endif()
        endif()

        if(NOT HAS_INDEXING_TYPE)
            if(COMMAND print_status_colored)
                print_status_colored("ERROR" "No indexing types (int32_t, int64_t) selected - array operations will fail!")
            else()
                message(FATAL_ERROR "No indexing types (int32_t, int64_t) selected - array operations will fail!")
            endif()
        endif()

        if(HAS_QUANTIZATION_TYPE AND NOT HAS_FLOAT_TYPE)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "Quantization types without float types - quantization operations may be limited")
            else()
                message(WARNING "Quantization types without float types - quantization operations may be limited")
            endif()
        endif()
    endif()
endfunction()

# Main function that orchestrates all enhanced type validation
function(libnd4j_validate_and_setup_types)
    if(COMMAND print_status_colored)
        print_status_colored("INFO" "=== LIBND4J ENHANCED TYPE VALIDATION SYSTEM ===")
    else()
        message(STATUS "=== LIBND4J ENHANCED TYPE VALIDATION SYSTEM ===")
    endif()

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

    if(COMMAND print_status_colored)
        print_status_colored("SUCCESS" "Enhanced type validation and setup completed successfully")
    else()
        message(STATUS "Enhanced type validation and setup completed successfully")
    endif()
endfunction()

# =============================================================================
# INTEGRATION HELPERS (Enhanced)
# =============================================================================

# Enhanced macro to easily add type validation to existing CMakeLists.txt
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

    # Include type combination engine
    if(EXISTS "${CMAKE_VALIDATION_DIR}/TypeCombinationEngine.cmake")
        include("${CMAKE_VALIDATION_DIR}/TypeCombinationEngine.cmake")
    endif()

    # Include type profiles
    if(EXISTS "${CMAKE_VALIDATION_DIR}/TypeProfiles.cmake")
        include("${CMAKE_VALIDATION_DIR}/TypeProfiles.cmake")
    endif()

    # Run enhanced validation
    libnd4j_validate_and_setup_types()
endmacro()

# Enhanced function to print usage help
function(print_type_validation_help)
    message(STATUS "")
    message(STATUS "LibND4J Enhanced Type Validation System Help")
    message(STATUS "===========================================")
    message(STATUS "")
    message(STATUS "CMake Variables:")
    message(STATUS "  SD_TYPES_LIST              - Semicolon-separated list of types")
    message(STATUS "  SD_STRICT_TYPE_VALIDATION  - Enable strict validation (ON/OFF)")
    message(STATUS "  SD_DEBUG_TYPE_PROFILE      - Debug type profile name")
    message(STATUS "  SD_DEBUG_CUSTOM_TYPES      - Custom types for debug profile")
    message(STATUS "  SD_DEBUG_AUTO_REDUCE       - Auto-reduce types for debug (ON/OFF)")
    message(STATUS "  SD_TYPE_PROFILE            - ML workload profile (quantization/training/inference/nlp/cv)")
    message(STATUS "")
    message(STATUS "Enhanced Debug Profiles:")
    message(STATUS "  QUANTIZATION               - int8;uint8;float32;int32;int64")
    message(STATUS "  MIXED_PRECISION            - float16;bfloat16;float32;int32;int64")
    message(STATUS "  NLP                        - std::string;float32;int32;int64")
    message(STATUS "  MINIMAL_INDEXING           - float32;double;int32;int64")
    message(STATUS "")
    message(STATUS "Example Usage:")
    message(STATUS "  cmake -DSD_TYPES_LIST=\"float32;double;int32;int64\" ..")
    message(STATUS "  cmake -DSD_DEBUG_TYPE_PROFILE=QUANTIZATION ..")
    message(STATUS "  cmake -DSD_TYPE_PROFILE=quantization ..")
    message(STATUS "  cmake -DSD_STRICT_TYPE_VALIDATION=ON ..")
    message(STATUS "")
    message(STATUS "ML Workload Profiles:")
    message(STATUS "  quantization - INT8/UINT8 inference optimization")
    message(STATUS "  training     - Mixed precision training")
    message(STATUS "  inference    - Deployment optimization")
    message(STATUS "  nlp          - String processing + embeddings")
    message(STATUS "  cv           - Computer vision quantization")
    message(STATUS "")
    message(STATUS "For more information, see the documentation.")
    message(STATUS "")
endfunction()

# Check if help was requested
if(LIBND4J_TYPE_VALIDATION_HELP)
    print_type_validation_help()
endif()