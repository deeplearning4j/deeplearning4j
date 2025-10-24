# =============================================================================
# TypeValidation.cmake - Enhanced with ML-aware pattern recognition
# MINIMAL FIXES ONLY - PRESERVING ORIGINAL STRUCTURE
# =============================================================================

# Global variables for type combinations
set(GENERATED_TYPE_COMBINATIONS "" CACHE INTERNAL "Generated type combinations")
set(PROCESSED_TEMPLATE_FILES "" CACHE INTERNAL "Processed template files")


# Define the missing srcore_normalize_type function that was causing the build error
function(srcore_normalize_type input_type output_var)
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
    elseif(normalized_type STREQUAL "qint8")
        set(normalized_type "int8_t")
    elseif(normalized_type STREQUAL "quint8")
        set(normalized_type "uint8_t")
    elseif(normalized_type STREQUAL "qint16")
        set(normalized_type "int16_t")
    elseif(normalized_type STREQUAL "quint16")
        set(normalized_type "uint16_t")
    elseif(normalized_type STREQUAL "utf8")
        set(normalized_type "std::string")
    elseif(normalized_type STREQUAL "utf16")
        set(normalized_type "std::u16string")
    elseif(normalized_type STREQUAL "utf32")
        set(normalized_type "std::u32string")
    endif()

    set(${output_var} "${normalized_type}" PARENT_SCOPE)
endfunction()

# Define any other missing functions that might be called
# In TypeValidation.cmake, replace the stub function with real validation
function(is_semantically_valid_combination type1 type2 type3 mode result_var)
    # Normalize types first
    srcore_normalize_type("${type1}" norm_t1)
    srcore_normalize_type("${type2}" norm_t2)
    srcore_normalize_type("${type3}" norm_t3)
    
    # Rule 1: Same type combinations are ALWAYS valid
    if(norm_t1 STREQUAL norm_t2 AND norm_t2 STREQUAL norm_t3)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 2: Same input types with different output - check if sensible
    if(norm_t1 STREQUAL norm_t2)
        # float,float -> double (precision upgrade) - VALID
        if(norm_t1 MATCHES "float" AND norm_t3 STREQUAL "double")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
        
        # int,int -> float (division result) - VALID
        if(norm_t1 MATCHES "int" AND norm_t3 MATCHES "float|double")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
        
        # any,any -> bool (comparison) - VALID
        if(norm_t3 STREQUAL "bool")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
        
        # float16,float16 -> float (mixed precision) - VALID
        if(norm_t1 MATCHES "float16|bfloat16" AND norm_t3 STREQUAL "float")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
        
        # int8,int8 -> int32 (quantization accumulation) - VALID
        if(norm_t1 MATCHES "int8_t|uint8_t" AND norm_t3 MATCHES "int32_t")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Rule 3: REJECT nonsensical mixed type combinations
    # Examples that should be REJECTED:
    # - bool,float16,double (what operation is this?)
    # - float16,int8_t,int32_t (mixing float and int without clear pattern)
    # - bfloat16,sd::LongType,float (random mixing)
    
    # If we get here, it's an invalid combination
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# =============================================================================
# ORIGINAL FUNCTIONS - PRESERVED EXACTLY AS THEY WERE
# =============================================================================

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
        list(APPEND suggestions "Prioritize int8_t, UnsignedChar, float16, float combinations")

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
# ORIGINAL FUNCTIONS (Enhanced) - ALL PRESERVED
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

# Function to normalize a type name (enhanced) - FIXED TO USE srcore_normalize_type
function(normalize_type input_type output_var)
    srcore_normalize_type("${input_type}" result)
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
        message(FATAL_ERROR "Type validation failed!")
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


function(validate_and_process_types)
    # Determine validation mode
    set(validation_mode "NORMAL")

    message(STATUS "🎯 =================================================================")
    message(STATUS "🎯 TYPE VALIDATION: Determining type selection mode...")
    message(STATUS "🎯 =================================================================")

    # STEP 1: Check for EXPLICIT user override for ALL types (highest priority)
    set(EXPLICIT_ALL_TYPES_REQUEST FALSE)
    if(DEFINED SD_FORCE_ALL_TYPES AND SD_FORCE_ALL_TYPES)
        set(EXPLICIT_ALL_TYPES_REQUEST TRUE)
        message(STATUS "🎯 SD_FORCE_ALL_TYPES=ON detected - USER EXPLICITLY REQUESTED ALL TYPES")
    endif()

    # STEP 2: Check if the current SD_TYPES_LIST matches any debug profile (CRITICAL FIX)
    set(TYPES_ARE_DEBUG_PROFILE FALSE)
    if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        string(STRIP "${SD_TYPES_LIST}" stripped_types)

        # Check if this matches any debug profile exactly
        if(stripped_types STREQUAL "float32;double;int32;int64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_MINIMAL_INDEXING")
        elseif(stripped_types STREQUAL "float32;double;int32;int64;int8;int16")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_ESSENTIAL")
        elseif(stripped_types STREQUAL "float32;double;float16")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_FLOATS_ONLY")
        elseif(stripped_types STREQUAL "int8;int16;int32;int64;uint8;uint16;uint32;uint64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_INTEGERS_ONLY")
        elseif(stripped_types STREQUAL "float32;int32;int64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_SINGLE_PRECISION")
        elseif(stripped_types STREQUAL "double;int32;int64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_DOUBLE_PRECISION")
        elseif(stripped_types STREQUAL "int8;uint8;float32;int32;int64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_QUANTIZATION")
        elseif(stripped_types STREQUAL "float16;bfloat16;float32;int32;int64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_MIXED_PRECISION")
        elseif(stripped_types STREQUAL "std::string;float32;int32;int64")
            set(TYPES_ARE_DEBUG_PROFILE TRUE)
            set(matched_profile "DEBUG_PROFILE_NLP")
        endif()

        if(TYPES_ARE_DEBUG_PROFILE)
            message(STATUS "🎯 DETECTED: Current SD_TYPES_LIST matches debug profile ${matched_profile}")
            message(STATUS "🎯 Types: ${stripped_types}")
            message(STATUS "🎯 This is NOT user-provided - this is an auto-applied debug profile!")
        endif()
    endif()

    # STEP 3: Check if user EXPLICITLY provided specific types via command line
    set(USER_EXPLICITLY_PROVIDED_TYPES FALSE)
    if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        string(STRIP "${SD_TYPES_LIST}" stripped_types)
        if(NOT stripped_types STREQUAL "")
            if(DEFINED SD_FORCE_SELECTIVE_TYPES AND SD_FORCE_SELECTIVE_TYPES)
                set(USER_EXPLICITLY_PROVIDED_TYPES TRUE)
                message(STATUS "🎯 SD_FORCE_SELECTIVE_TYPES=ON - User explicitly wants selective types: ${SD_TYPES_LIST}")
            elseif(TYPES_ARE_DEBUG_PROFILE)
                # CRITICAL FIX: If types match a debug profile, they are NOT user-provided
                set(USER_EXPLICITLY_PROVIDED_TYPES FALSE)
                message(STATUS "🎯 Types match debug profile ${matched_profile} - NOT treating as user-provided")
            elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" AND SD_GCC_FUNCTRACE)
                message(STATUS "🎯 Debug mode detected with types: ${SD_TYPES_LIST}")
                message(STATUS "🎯 These appear to be auto-generated debug types, not user-provided")
                set(USER_EXPLICITLY_PROVIDED_TYPES FALSE)
            else()
                # Only treat as user-provided if it doesn't match any debug profile
                set(USER_EXPLICITLY_PROVIDED_TYPES TRUE)
                message(STATUS "🎯 Non-debug build with types: ${SD_TYPES_LIST} - treating as user-provided")
            endif()
        endif()
    endif()

    # STEP 4: Check if debug auto-reduction should apply
    set(DEBUG_AUTO_REDUCTION_APPLIES FALSE)

    # STEP 5: DECIDE THE FINAL MODE BASED ON CORRECTED PRIORITY
    message(STATUS "🎯 -----------------------------------------------------------------")
    message(STATUS "🎯 DECISION LOGIC:")
    message(STATUS "🎯   EXPLICIT_ALL_TYPES_REQUEST: ${EXPLICIT_ALL_TYPES_REQUEST}")
    message(STATUS "🎯   USER_EXPLICITLY_PROVIDED_TYPES: ${USER_EXPLICITLY_PROVIDED_TYPES}")
    message(STATUS "🎯   TYPES_ARE_DEBUG_PROFILE: ${TYPES_ARE_DEBUG_PROFILE}")
    message(STATUS "🎯   DEBUG_AUTO_REDUCTION_APPLIES: ${DEBUG_AUTO_REDUCTION_APPLIES}")
    message(STATUS "🎯   CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    message(STATUS "🎯   SD_GCC_FUNCTRACE: ${SD_GCC_FUNCTRACE}")
    message(STATUS "🎯 -----------------------------------------------------------------")

    if(EXPLICIT_ALL_TYPES_REQUEST)
        # PRIORITY 1: User explicitly requested ALL types via SD_FORCE_ALL_TYPES=ON
        message(STATUS "🎯 ✅ DECISION: ALL TYPES MODE (explicit user request)")
        message(STATUS "🎯 Reason: SD_FORCE_ALL_TYPES=ON overrides everything")
        set(USE_ALL_TYPES TRUE)
        set(FINAL_TYPES_LIST "")

        # Clear any auto-generated types to ensure ALL mode
        set(SD_TYPES_LIST "" PARENT_SCOPE)
        set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)

        # CRITICAL: Also clear all cached variables to prevent them from overriding
        unset(SD_TYPES_LIST CACHE)
        unset(SD_TYPES_LIST_COUNT CACHE)
        unset(SRCORE_VALIDATED_TYPES CACHE)
        unset(SRCORE_USE_SELECTIVE_TYPES CACHE)

    elseif(USER_EXPLICITLY_PROVIDED_TYPES AND NOT TYPES_ARE_DEBUG_PROFILE)
        # PRIORITY 2: User explicitly provided specific types (and they're not from a debug profile)
        message(STATUS "🎯 ✅ DECISION: SELECTIVE TYPES MODE (user-provided types)")
        message(STATUS "🎯 Reason: User explicitly specified types: ${SD_TYPES_LIST}")
        set(USE_ALL_TYPES FALSE)
        set(FINAL_TYPES_LIST "${SD_TYPES_LIST}")

        # Validate the user-provided types
        validate_type_list("${SD_TYPES_LIST}" "${validation_mode}")

    elseif(TYPES_ARE_DEBUG_PROFILE)
        # CRITICAL FIX: If types are from a debug profile, ignore them and use ALL types
        message(STATUS "🎯 ✅ DECISION: ALL TYPES MODE (debug profile detected and ignored)")
        message(STATUS "🎯 Reason: SD_TYPES_LIST matches debug profile ${matched_profile} - ignoring and using ALL types")
        set(USE_ALL_TYPES TRUE)
        set(FINAL_TYPES_LIST "")

        # Clear the debug profile types to ensure ALL mode
        set(SD_TYPES_LIST "" PARENT_SCOPE)
        set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)

        # CRITICAL: Also clear all cached variables to prevent them from overriding
        unset(SD_TYPES_LIST CACHE)
        unset(SD_TYPES_LIST_COUNT CACHE)
        unset(SRCORE_VALIDATED_TYPES CACHE)
        unset(SRCORE_USE_SELECTIVE_TYPES CACHE)

    else()
        # PRIORITY 4: DEFAULT = ALL TYPES
        message(STATUS "🎯 ✅ DECISION: ALL TYPES MODE (default behavior)")
        message(STATUS "🎯 Reason: No explicit user requests detected - using default ALL types")
        set(USE_ALL_TYPES TRUE)
        set(FINAL_TYPES_LIST "")

        # Clear any existing types to ensure ALL mode
        set(SD_TYPES_LIST "" PARENT_SCOPE)
        set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)

        # CRITICAL: Also clear all cached variables to prevent them from overriding
        unset(SD_TYPES_LIST CACHE)
        unset(SD_TYPES_LIST_COUNT CACHE)
        unset(SRCORE_VALIDATED_TYPES CACHE)
        unset(SRCORE_USE_SELECTIVE_TYPES CACHE)
    endif()

    # STEP 6: SET UP THE FINAL CONFIGURATION AND EXPORT IMMEDIATELY
    message(STATUS "🎯 =================================================================")
    if(USE_ALL_TYPES)
        message(STATUS "🎯 FINAL CONFIGURATION: ALL TYPES MODE")
        message(STATUS "🎯 All available data types will be included")
        message(STATUS "🎯 SD_SELECTIVE_TYPES will NOT be defined")

        set(SRCORE_USE_SELECTIVE_TYPES FALSE CACHE INTERNAL "Use selective type discovery")
        set(SRCORE_VALIDATED_TYPES "" CACHE INTERNAL "Validated types for selective rendering")
        message(STATUS "🎯 Exported ALL_TYPES mode for SelectiveRenderingCore")

    else()
        message(STATUS "🎯 FINAL CONFIGURATION: SELECTIVE TYPES MODE")
        message(STATUS "🎯 Building with specific types: ${FINAL_TYPES_LIST}")
        message(STATUS "🎯 SD_SELECTIVE_TYPES will be defined")

        # Update count using the local variable
        if(FINAL_TYPES_LIST)
            list(LENGTH FINAL_TYPES_LIST final_count)
            set(SD_TYPES_LIST_COUNT ${final_count} PARENT_SCOPE)
        else()
            set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)
        endif()

        set(SRCORE_USE_SELECTIVE_TYPES TRUE CACHE INTERNAL "Use selective type discovery")
        set(SRCORE_VALIDATED_TYPES "${FINAL_TYPES_LIST}" CACHE INTERNAL "Validated types for selective rendering")
        message(STATUS "🎯 Exported SELECTIVE types for SelectiveRenderingCore: ${FINAL_TYPES_LIST}")
    endif()
    message(STATUS "🎯 =================================================================")
endfunction()

macro(SETUP_LIBND4J_TYPE_VALIDATION)
    set(SD_TYPES_VALIDATION_MODE "NORMAL")




    # Call the main validation function
    validate_and_process_types()

    # Update the count after validation
    if(SD_TYPES_LIST)
        list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
    else()
        set(SD_TYPES_LIST_COUNT 0)
    endif()
endmacro()


# Export type validation results for use by SelectiveRenderingCore
function(export_validated_types_for_selective_rendering)
    # Export the validated type list and mode for SelectiveRenderingCore to use
    if(SD_TYPES_LIST_COUNT GREATER 0)
        # SELECTIVE mode - export the specific types that were validated
        set(SRCORE_USE_SELECTIVE_TYPES TRUE PARENT_SCOPE)
        set(SRCORE_VALIDATED_TYPES "${SD_TYPES_LIST}" PARENT_SCOPE)
        message(STATUS "📤 Exporting SELECTIVE types for SelectiveRenderingCore: ${SD_TYPES_LIST}")
    else()
        # ALL types mode - let SelectiveRenderingCore discover all types
        set(SRCORE_USE_SELECTIVE_TYPES FALSE PARENT_SCOPE)
        set(SRCORE_VALIDATED_TYPES "" PARENT_SCOPE)
        message(STATUS "📤 Exporting ALL_TYPES mode for SelectiveRenderingCore")
    endif()
endfunction()



function(setup_type_definitions)
    # Try to find the target from various sources, including actual target names
    set(target_name "")

    # Check if CURRENT_BUILD_TARGET is set
    if(DEFINED CURRENT_BUILD_TARGET AND NOT CURRENT_BUILD_TARGET STREQUAL "")
        set(target_name ${CURRENT_BUILD_TARGET})
        message(STATUS "Using CURRENT_BUILD_TARGET: ${target_name}")
        # Check SD_LIBRARY_NAME first (this is set in CMakeLists.txt)
    elseif(DEFINED SD_LIBRARY_NAME AND TARGET ${SD_LIBRARY_NAME})
        set(target_name ${SD_LIBRARY_NAME})
        message(STATUS "Found SD_LIBRARY_NAME target: ${target_name}")
        # Check for actual target names based on build type
    elseif(SD_CUDA AND TARGET nd4jcuda)
        set(target_name "nd4jcuda")
        message(STATUS "Found CUDA target: nd4jcuda")
    elseif(SD_CPU AND TARGET nd4jcpu)
        set(target_name "nd4jcpu")
        message(STATUS "Found CPU target: nd4jcpu")
        # Fallback to common target names
    elseif(TARGET nd4jcpu)
        set(target_name "nd4jcpu")
        message(STATUS "Found nd4jcpu target")
    elseif(TARGET nd4jcuda)
        set(target_name "nd4jcuda")
        message(STATUS "Found nd4jcuda target")
    elseif(TARGET nd4j)
        set(target_name "nd4j")
        message(STATUS "Found nd4j target")
    elseif(TARGET libnd4j)
        set(target_name "libnd4j")
        message(STATUS "Found libnd4j target")
    else()
        # Get all targets and try to find one that looks like our library
        get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
        foreach(tgt ${all_targets})
            if(tgt MATCHES ".*(nd4j|libnd4j).*")
                set(target_name ${tgt})
                message(STATUS "Auto-detected target: ${target_name}")
                break()
            endif()
        endforeach()
    endif()

    if(target_name STREQUAL "")
        # No targets found yet - defer the setup
        message(STATUS "No targets found yet, deferring type definitions setup")
        set_property(GLOBAL PROPERTY DEFERRED_TYPE_SETUP_NEEDED TRUE)

        # Store the type configuration for later use
        if(DEFINED SD_TYPES_LIST)
            set_property(GLOBAL PROPERTY DEFERRED_SD_TYPES_LIST "${SD_TYPES_LIST}")
        endif()
        if(DEFINED SD_TYPES_LIST_COUNT)
            set_property(GLOBAL PROPERTY DEFERRED_SD_TYPES_LIST_COUNT "${SD_TYPES_LIST_COUNT}")
        endif()

        return()
    endif()

    setup_type_definitions_for_target(${target_name})
endfunction()


# Remove the complex deferred logic since MainBuildFlow handles it
function(LIBND4J_SETUP_TYPE_VALIDATION)
    message(STATUS "🎯 LIBND4J Type validation setup - will be applied when targets are created")

    # Just validate the type configuration, don't apply to targets yet
    if(NOT DEFINED SD_TYPES_LIST_COUNT)
        set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)
    endif()



    message(STATUS "✅ Type validation configuration ready")
endfunction()

# Automatic application when any nd4j-related target is created
function(auto_apply_deferred_type_definitions target_name)
    if(target_name MATCHES ".*nd4j.*" OR target_name MATCHES ".*libnd4j.*")
        apply_deferred_type_definitions_to_target(${target_name})
    endif()
endfunction()


function(validate_type_consistency)
    # Check that essential combinations are possible
    if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        set(HAS_INTEGER_TYPE FALSE)
        set(HAS_FLOAT_TYPE FALSE)
        set(HAS_INDEXING_TYPE FALSE)
        set(HAS_QUANTIZATION_TYPE FALSE)

        message(STATUS "🔍 Validating type consistency for: ${SD_TYPES_LIST}")

        foreach(SD_TYPE ${SD_TYPES_LIST})
            normalize_type("${SD_TYPE}" normalized_type)
            message(STATUS "   Checking: ${SD_TYPE} -> ${normalized_type}")

            # Check for indexing types FIRST - this is the most critical check
            if(SD_TYPE STREQUAL "int32" OR SD_TYPE STREQUAL "int64" OR
                    normalized_type STREQUAL "int32_t" OR normalized_type STREQUAL "int64_t")
                set(HAS_INDEXING_TYPE TRUE)
                set(HAS_INTEGER_TYPE TRUE)
                message(STATUS "     ✅ INDEXING type detected: ${SD_TYPE} -> ${normalized_type}")
            endif()

            # Check for any integer types
            if(SD_TYPE MATCHES "^(int|uint)[0-9]+$" OR SD_TYPE STREQUAL "bool" OR
                    normalized_type MATCHES "^(int|uint)[0-9]+_t$" OR normalized_type STREQUAL "bool")
                set(HAS_INTEGER_TYPE TRUE)
                message(STATUS "     ✅ INTEGER type detected: ${SD_TYPE} -> ${normalized_type}")
            endif()

            # Check for quantization types (int8/uint8)
            if(SD_TYPE MATCHES "^(int8|uint8)$" OR normalized_type MATCHES "^(int8_t|uint8_t)$")
                set(HAS_QUANTIZATION_TYPE TRUE)
                message(STATUS "     ✅ QUANTIZATION type detected: ${SD_TYPE} -> ${normalized_type}")
            endif()

            # Check for floating point types
            if(SD_TYPE MATCHES "^(float|double|bfloat|half)" OR
                    normalized_type MATCHES "^(float|double|bfloat|half)")
                set(HAS_FLOAT_TYPE TRUE)
                message(STATUS "     ✅ FLOAT type detected: ${SD_TYPE} -> ${normalized_type}")
            endif()
        endforeach()

        message(STATUS "")
        message(STATUS "🔍 Final type consistency results:")
        message(STATUS "   HAS_INTEGER_TYPE: ${HAS_INTEGER_TYPE}")
        message(STATUS "   HAS_FLOAT_TYPE: ${HAS_FLOAT_TYPE}")
        message(STATUS "   HAS_INDEXING_TYPE: ${HAS_INDEXING_TYPE}")
        message(STATUS "   HAS_QUANTIZATION_TYPE: ${HAS_QUANTIZATION_TYPE}")
        message(STATUS "")

        # Only show warnings/errors if the checks actually fail
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
                print_status_colored("ERROR" "No indexing types (int32, int64) selected - array operations will fail!")
                print_status_colored("ERROR" "Available types were: ${SD_TYPES_LIST}")
                print_status_colored("ERROR" "Normalized types were checked for: int32_t, int64_t")
            else()
                message(FATAL_ERROR "No indexing types (int32, int64) selected - array operations will fail!")
            endif()
        endif()

        if(HAS_QUANTIZATION_TYPE AND NOT HAS_FLOAT_TYPE)
            if(COMMAND print_status_colored)
                print_status_colored("WARNING" "Quantization types without float types - quantization operations may be limited")
            else()
                message(WARNING "Quantization types without float types - quantization operations may be limited")
            endif()
        endif()
    else()
        # ALL TYPES mode - no consistency check needed
        message(STATUS "🔍 ALL TYPES MODE: Type consistency check skipped")
        message(STATUS "    All available types are included, ensuring full functionality")
    endif()
endfunction()
# Main function that orchestrates all enhanced type validation
function(libnd4j_validate_and_setup_types)
    if(COMMAND print_status_colored)
        print_status_colored("INFO" "=== LIBND4J ENHANCED TYPE VALIDATION SYSTEM ===")
    else()
        message(STATUS "=== LIBND4J ENHANCED TYPE VALIDATION SYSTEM ===")
    endif()

    # Step 1: Update SD_TYPES_LIST_COUNT after cleanup
    if(SD_TYPES_LIST)
        list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
        set(SD_TYPES_LIST_COUNT "${SD_TYPES_LIST_COUNT}" PARENT_SCOPE)
    else()
        set(SD_TYPES_LIST_COUNT 0 PARENT_SCOPE)
    endif()

    # Step 2: Run main validation
    validate_and_process_types()

    # Step 3: Set up compile definitions - THIS IS THE CRITICAL STEP
    setup_type_definitions()

    # Step 4: Validate consistency - FIXED FUNCTION
    validate_type_consistency()

    if(COMMAND print_status_colored)
        print_status_colored("SUCCESS" "Enhanced type validation and setup completed successfully")
    else()
        message(STATUS "Enhanced type validation and setup completed successfully")
    endif()
endfunction()
# =============================================================================
# MAIN SETUP MACRO - THIS IS WHAT GETS CALLED
# =============================================================================

macro(LIBND4J_SETUP_TYPE_VALIDATION)
    # Set default validation mode
    if(NOT DEFINED SD_TYPES_VALIDATION_MODE)
        set(SD_TYPES_VALIDATION_MODE "NORMAL")

    endif()



    # Call the main validation function
    libnd4j_validate_and_setup_types()

    # Update the count after validation
    if(SD_TYPES_LIST)
        list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
    else()
        set(SD_TYPES_LIST_COUNT 0)
    endif()

    if(SD_TYPES_LIST_COUNT GREATER 0)
        set(SRCORE_USE_SELECTIVE_TYPES TRUE CACHE INTERNAL "Use selective type discovery")
        set(SRCORE_VALIDATED_TYPES "${SD_TYPES_LIST}" CACHE INTERNAL "Validated types for selective rendering")
        message(STATUS "📤 CACHE: Exported SELECTIVE types for SelectiveRenderingCore: ${SD_TYPES_LIST}")
    else()
        set(SRCORE_USE_SELECTIVE_TYPES FALSE CACHE INTERNAL "Use selective type discovery")
        set(SRCORE_VALIDATED_TYPES "" CACHE INTERNAL "Validated types for selective rendering")
        message(STATUS "📤 CACHE: Exported ALL_TYPES mode for SelectiveRenderingCore")
    endif()
endmacro()


function(apply_type_definitions_to_target target_name)
    message(STATUS "🔧 APPLYING type definitions to target: ${target_name}")

    if(DEFINED SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        # Selective types mode - add definitions for specific types
        message(STATUS "🎯 Adding SELECTIVE type definitions for: ${SD_TYPES_LIST}")

        foreach(type_name ${SD_TYPES_LIST})
            string(TOLOWER "${type_name}" lower_type)

            # Floating point types
            if(lower_type MATCHES "^(float|float32)$")
                target_compile_definitions(${target_name} PRIVATE HAS_FLOAT=1 HAS_FLOAT32=1)
                message(STATUS "   ✅ Added: HAS_FLOAT, HAS_FLOAT32")
            elseif(lower_type MATCHES "^(double|float64)$")
                target_compile_definitions(${target_name} PRIVATE HAS_DOUBLE=1 HAS_FLOAT64=1)
                message(STATUS "   ✅ Added: HAS_DOUBLE, HAS_FLOAT64")
            elseif(lower_type MATCHES "^(float16|half)$")
                target_compile_definitions(${target_name} PRIVATE HAS_FLOAT16=1 HAS_HALF=1)
                message(STATUS "   ✅ Added: HAS_FLOAT16, HAS_HALF")
            elseif(lower_type MATCHES "^(bfloat16|bfloat)$")
                target_compile_definitions(${target_name} PRIVATE HAS_BFLOAT16=1 HAS_BFLOAT=1)
                message(STATUS "   ✅ Added: HAS_BFLOAT16, HAS_BFLOAT")
            elseif(lower_type MATCHES "^float8$")
                target_compile_definitions(${target_name} PRIVATE HAS_FLOAT8=1)
                message(STATUS "   ✅ Added: HAS_FLOAT8")
            elseif(lower_type MATCHES "^half2$")
                target_compile_definitions(${target_name} PRIVATE HAS_HALF2=1)
                message(STATUS "   ✅ Added: HAS_HALF2")

                # Integer types
            elseif(lower_type MATCHES "^(int8|int8_t)$")
                target_compile_definitions(${target_name} PRIVATE HAS_INT8=1 HAS_INT8_T=1)
                message(STATUS "   ✅ Added: HAS_INT8, HAS_INT8_T")
            elseif(lower_type MATCHES "^(int16|int16_t)$")
                target_compile_definitions(${target_name} PRIVATE HAS_INT16=1 HAS_INT16_T=1)
                message(STATUS "   ✅ Added: HAS_INT16, HAS_INT16_T")
            elseif(lower_type MATCHES "^(int32|int32_t|int)$")
                target_compile_definitions(${target_name} PRIVATE HAS_INT32=1 HAS_INT32_T=1 HAS_INT=1)
                message(STATUS "   ✅ Added: HAS_INT32, HAS_INT32_T, HAS_INT")
            elseif(lower_type MATCHES "^(int64|int64_t|long)$")
                target_compile_definitions(${target_name} PRIVATE HAS_INT64=1 HAS_INT64_T=1 HAS_LONG=1)
                message(STATUS "   ✅ Added: HAS_INT64, HAS_INT64_T, HAS_LONG")

                # Unsigned integer types
            elseif(lower_type MATCHES "^(uint8|uint8_t)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UINT8=1 HAS_UINT8_T=1)
                message(STATUS "   ✅ Added: HAS_UINT8, HAS_UINT8_T")
            elseif(lower_type MATCHES "^(uint16|uint16_t)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UINT16=1 HAS_UINT16_T=1)
                message(STATUS "   ✅ Added: HAS_UINT16, HAS_UINT16_T")
            elseif(lower_type MATCHES "^(uint32|uint32_t)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UINT32=1 HAS_UINT32_T=1)
                message(STATUS "   ✅ Added: HAS_UINT32, HAS_UINT32_T")
            elseif(lower_type MATCHES "^(uint64|uint64_t|unsignedlong)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UINT64=1 HAS_UINT64_T=1 HAS_UNSIGNEDLONG=1)
                message(STATUS "   ✅ Added: HAS_UINT64, HAS_UINT64_T, HAS_UNSIGNEDLONG")

                # Quantized types
            elseif(lower_type MATCHES "^qint8$")
                target_compile_definitions(${target_name} PRIVATE HAS_QINT8=1)
                message(STATUS "   ✅ Added: HAS_QINT8")
            elseif(lower_type MATCHES "^qint16$")
                target_compile_definitions(${target_name} PRIVATE HAS_QINT16=1)
                message(STATUS "   ✅ Added: HAS_QINT16")

                # Boolean type
            elseif(lower_type MATCHES "^bool$")
                target_compile_definitions(${target_name} PRIVATE HAS_BOOL=1)
                message(STATUS "   ✅ Added: HAS_BOOL")

                # String types
            elseif(lower_type MATCHES "^(utf8|string|std::string)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UTF8=1 HAS_STD_STRING=1)
                message(STATUS "   ✅ Added: HAS_UTF8, HAS_STD_STRING")
            elseif(lower_type MATCHES "^(utf16|u16string|std::u16string)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UTF16=1 HAS_STD_U16STRING=1)
                message(STATUS "   ✅ Added: HAS_UTF16, HAS_STD_U16STRING")
            elseif(lower_type MATCHES "^(utf32|u32string|std::u32string)$")
                target_compile_definitions(${target_name} PRIVATE HAS_UTF32=1 HAS_STD_U32STRING=1)
                message(STATUS "   ✅ Added: HAS_UTF32, HAS_STD_U32STRING")
            else()
                message(WARNING "   ⚠️ Unknown type: ${type_name}")
            endif()
        endforeach()

        # Always add the selective types flag
        target_compile_definitions(${target_name} PRIVATE SD_SELECTIVE_TYPES=1)
        message(STATUS "   ✅ Added: SD_SELECTIVE_TYPES")
    else()
        # All types mode - add ALL type definitions from the header
        message(STATUS "🎯 Adding ALL type definitions")
        target_compile_definitions(${target_name} PRIVATE
                # Boolean
                HAS_BOOL=1

                # Floating point types
                HAS_FLOAT=1 HAS_FLOAT32=1
                HAS_DOUBLE=1 HAS_FLOAT64=1
                HAS_FLOAT16=1 HAS_HALF=1
                HAS_BFLOAT16=1 HAS_BFLOAT=1
                HAS_FLOAT8=1
                HAS_HALF2=1

                # Signed integer types
                HAS_INT8=1 HAS_INT8_T=1
                HAS_INT16=1 HAS_INT16_T=1
                HAS_INT32=1 HAS_INT32_T=1 HAS_INT=1
                HAS_INT64=1 HAS_INT64_T=1 HAS_LONG=1

                # Unsigned integer types
                HAS_UINT8=1 HAS_UINT8_T=1
                HAS_UINT16=1 HAS_UINT16_T=1
                HAS_UINT32=1 HAS_UINT32_T=1
                HAS_UINT64=1 HAS_UINT64_T=1 HAS_UNSIGNEDLONG=1

                # Quantized types
                HAS_QINT8=1
                HAS_QINT16=1

                # String types (only if string operations enabled)
                HAS_UTF8=1 HAS_STD_STRING=1
                HAS_UTF16=1 HAS_STD_U16STRING=1
                HAS_UTF32=1 HAS_STD_U32STRING=1
        )

        # Enable string operations for all types mode
        target_compile_definitions(${target_name} PRIVATE SD_ENABLE_STRING_OPERATIONS=1)
        message(STATUS "   ✅ Added: All data types and string operations")
    endif()
endfunction()


function(setup_type_definitions_for_target target_name)
    message(STATUS "🎯 Setting up type definitions for target: ${target_name}")

    # Validate inputs and run type validation
    if(NOT DEFINED SD_TYPES_LIST_COUNT)
        set(SD_TYPES_LIST_COUNT 0)
    endif()

    # Set up debug auto-reduction based on build type and tracing
    set(DEBUG_AUTO_REDUCTION_APPLIES FALSE)


    # Apply the type definitions to the target
    if(TARGET ${target_name})
        apply_type_definitions_to_target(${target_name})
        message(STATUS "✅ Type definitions applied to ${target_name}")
    else()
        message(WARNING "Target ${target_name} not found, cannot apply type definitions")
    endif()
endfunction()

# =============================================================================
# UTILITY FUNCTIONS - ALL PRESERVED
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


    file(APPEND "${CONFIG_FILE}" "\nValidation Mode: ${SD_TYPES_VALIDATION_MODE}\n")

    message(STATUS "Enhanced type configuration summary written to: ${CONFIG_FILE}")
endfunction()

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
    message(STATUS "  cmake -DSD_STRICT_TYPE_VALIDATION=ON ..")
    message(STATUS "")
endfunction()

# Check if help was requested
if(LIBND4J_TYPE_VALIDATION_HELP)
    print_type_validation_help()
endif()