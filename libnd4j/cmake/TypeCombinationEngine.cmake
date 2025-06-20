# TypeCombinationEngine.cmake
# Complete template processing and semantic filtering system for type combinations

# Include required modules
include(TypeValidation)

# Global variables for type combinations
set(GENERATED_TYPE_COMBINATIONS "" CACHE INTERNAL "Generated type combinations")
set(PROCESSED_TEMPLATE_FILES "" CACHE INTERNAL "Processed template files")

# Core type ranking mapping (from type_promote.h)
set(TYPE_RANKS
    "bool:1"
    "int8_t:2"
    "uint8_t:2"
    "int16_t:3"
    "uint16_t:3"
    "int32_t:4"
    "uint32_t:4"
    "int64_t:5"
    "uint64_t:5"
    "float16:6"
    "bfloat16:6"
    "float:7"
    "double:8"
    "std::string:-10"
    "std::u16string:-10"
    "std::u32string:-10"
)


function(initialize_dynamic_combinations)
    if(NOT DEFINED SD_COMMON_TYPES_COUNT OR SD_COMMON_TYPES_COUNT EQUAL 0)
        message(FATAL_ERROR "❌ Cannot initialize combinations: SD_COMMON_TYPES_COUNT is ${SD_COMMON_TYPES_COUNT}")
    endif()

    message(STATUS "=== INITIALIZING TYPE COMBINATIONS ===")
    message(STATUS "Available types: ${SD_COMMON_TYPES_COUNT}")

    set(combinations3 "")
    set(combinations2 "")

    # Check if semantic filtering is enabled
    if(SD_ENABLE_SEMANTIC_FILTERING)
        message(STATUS "🔧 Semantic filtering ENABLED - using selective combinations")

        # Apply type profile filtering first
        set(active_type_indices "")
        if(SD_TYPE_PROFILE STREQUAL "MINIMAL")
            # Only essential types: int32, int64, float32, double
            set(essential_types "int32;int64;float32;double")
        elseif(SD_TYPE_PROFILE STREQUAL "ESSENTIAL")
            # Core types for most operations
            set(essential_types "bool;int8;int16;int32;int64;uint8;uint16;uint32;uint64;float16;float32;double")
        elseif(SD_TYPE_PROFILE STREQUAL "QUANTIZATION")
            # Focus on quantization-friendly types
            set(essential_types "int8;uint8;int16;uint16;int32;int64;float16;float32;double")
        else()
            # Default to essential if unknown profile
            set(essential_types "bool;int8;int16;int32;int64;uint8;uint16;uint32;uint64;float16;float32;double")
        endif()

        # Find indices for essential types
        foreach(target_type ${essential_types})
            math(EXPR max_index "${SD_COMMON_TYPES_COUNT} - 1")
            foreach(i RANGE 0 ${max_index})
                get_type_name_from_index(${i} type_name)
                if(type_name STREQUAL target_type)
                    list(APPEND active_type_indices ${i})
                    break()
                endif()
            endforeach()
        endforeach()

        list(LENGTH active_type_indices active_count)
        message(STATUS "🎯 Using ${active_count} filtered types: ${essential_types}")

        # Generate combinations only for filtered types
        foreach(i ${active_type_indices})
            foreach(j ${active_type_indices})
                # Apply semantic validation for 2-type combinations
                get_type_name_from_index(${i} type1)
                get_type_name_from_index(${j} type2)

                # Simple semantic check: avoid obviously invalid combinations
                set(is_valid TRUE)
                if(type1 MATCHES "utf" OR type2 MATCHES "utf")
                    if(NOT (type1 MATCHES "int" OR type2 MATCHES "int"))
                        set(is_valid FALSE)  # String ops need integer indices
                    endif()
                endif()

                if(is_valid)
                    list(APPEND combinations2 "${i},${j}")
                endif()

                foreach(k ${active_type_indices})
                    # Apply semantic validation for 3-type combinations
                    get_type_name_from_index(${k} type3)

                    set(is_valid_3 TRUE)
                    # Avoid triple string combinations
                    if(type1 MATCHES "utf" AND type2 MATCHES "utf" AND type3 MATCHES "utf")
                        set(is_valid_3 FALSE)
                    endif()

                    # Apply quantization-aware filtering
                    if(SD_TYPE_PROFILE STREQUAL "QUANTIZATION")
                        # Only allow combinations that make sense for quantization
                        if(NOT ((type1 MATCHES "int8|uint8" AND type2 MATCHES "float" AND type3 MATCHES "float") OR
                                (type1 MATCHES "float" AND type2 MATCHES "float" AND type3 MATCHES "int8|uint8") OR
                                (type1 STREQUAL type2 AND type2 STREQUAL type3)))
                            set(is_valid_3 FALSE)
                        endif()
                    endif()

                    if(is_valid_3)
                        list(APPEND combinations3 "${i},${j},${k}")
                    endif()
                endforeach()
            endforeach()
        endforeach()

    else()
        message(STATUS "⚠️ Semantic filtering DISABLED - using all combinations")
        # Original brute force approach
        math(EXPR max_index "${SD_COMMON_TYPES_COUNT} - 1")
        foreach(i RANGE 0 ${max_index})
            foreach(j RANGE 0 ${max_index})
                list(APPEND combinations2 "${i},${j}")
                foreach(k RANGE 0 ${max_index})
                    list(APPEND combinations3 "${i},${j},${k}")
                endforeach()
            endforeach()
        endforeach()
    endif()

    list(LENGTH combinations2 count_2)
    list(LENGTH combinations3 count_3)

    # Apply safety limits
    if(DEFINED SD_MAX_TEMPLATE_COMBINATIONS)
        if(count_3 GREATER ${SD_MAX_TEMPLATE_COMBINATIONS})
            message(WARNING "⚠️ 3-type combinations (${count_3}) exceed limit (${SD_MAX_TEMPLATE_COMBINATIONS})")
            message(WARNING "   Consider using a more restrictive type profile")
        endif()
    endif()

    message(STATUS "Generated ${count_2} 2-type and ${count_3} 3-type combinations")

    if(SD_ENABLE_SEMANTIC_FILTERING)
        math(EXPR total_possible_2 "${SD_COMMON_TYPES_COUNT} * ${SD_COMMON_TYPES_COUNT}")
        math(EXPR total_possible_3 "${SD_COMMON_TYPES_COUNT} * ${SD_COMMON_TYPES_COUNT} * ${SD_COMMON_TYPES_COUNT}")
        math(EXPR reduction_2 "100 - (100 * ${count_2} / ${total_possible_2})")
        math(EXPR reduction_3 "100 - (100 * ${count_3} / ${total_possible_3})")
        message(STATUS "🎯 Filtering reduced combinations by: 2-type=${reduction_2}%, 3-type=${reduction_3}%")
    endif()

    set(COMBINATIONS_3 "${combinations3}" PARENT_SCOPE)
    set(COMBINATIONS_2 "${combinations2}" PARENT_SCOPE)

    # Also cache globally
    set(COMBINATIONS_3 "${combinations3}" CACHE INTERNAL "3-type combinations" FORCE)
    set(COMBINATIONS_2 "${combinations2}" CACHE INTERNAL "2-type combinations" FORCE)
endfunction()


function(process_template_unified TEMPLATE_FILE COMBINATION_TYPE COMBINATIONS OUTPUT_DIR)
    set(GENERATED_SOURCES_LOCAL "")
    foreach(COMBINATION ${COMBINATIONS})
        string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        if(COMBINATION_TYPE EQUAL 3)
            list(GET COMB_LIST 2 COMB3)
        else()
            set(COMB3 ${COMB1})
        endif()
        get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
        string(REPLACE "_template" "" OUTPUT_BASE_NAME "${TEMPLATE_BASE}")
        if(COMBINATION_TYPE EQUAL 3)
            set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}_${COMB3}.cpp")
        else()
            set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}.cpp")
        endif()
        set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")
        configure_file("${TEMPLATE_FILE}" "${GENERATED_FILE}" @ONLY)
        list(APPEND GENERATED_SOURCES_LOCAL "${GENERATED_FILE}")
    endforeach()
    set(GENERATED_SOURCES ${GENERATED_SOURCES_LOCAL} PARENT_SCOPE)
endfunction()

function(detect_template_requirements TEMPLATE_FILE NEEDS_2_TYPE_VAR NEEDS_3_TYPE_VAR)
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)
    set(NEEDS_2_TYPE FALSE)
    set(NEEDS_3_TYPE FALSE)
    if(TEMPLATE_CONTENT MATCHES "TYPE_INDEX_3|COMB3")
        set(NEEDS_3_TYPE TRUE)
    endif()
    if(TEMPLATE_CONTENT MATCHES "TYPE_INDEX_2|COMB2")
        set(NEEDS_2_TYPE TRUE)
    endif()
    if(NOT NEEDS_2_TYPE AND NOT NEEDS_3_TYPE)
        set(NEEDS_2_TYPE TRUE)
        set(NEEDS_3_TYPE TRUE)
    endif()
    set(${NEEDS_2_TYPE_VAR} ${NEEDS_2_TYPE} PARENT_SCOPE)
    set(${NEEDS_3_TYPE_VAR} ${NEEDS_3_TYPE} PARENT_SCOPE)
endfunction()

# Get type promotion rank for semantic filtering
function(get_type_promotion_rank type_name result_var)
    foreach(rank_pair ${TYPE_RANKS})
        string(REPLACE ":" ";" rank_parts ${rank_pair})
        list(GET rank_parts 0 type)
        list(GET rank_parts 1 rank)
        if(type STREQUAL type_name)
            set(${result_var} ${rank} PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${result_var} -1 PARENT_SCOPE)  # Unknown type
endfunction()

# Check if type combination represents a quantization pattern
function(is_quantization_pattern type1 type2 type3 result_var)
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)
    
    # Forward quantization: (float32, float32, int8)
    if(rank1 GREATER_EQUAL 5 AND rank2 GREATER_EQUAL 5 AND rank3 LESS_EQUAL 3)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Dequantization: (int8, float32, float32)
    if(rank1 LESS_EQUAL 3 AND rank2 GREATER_EQUAL 5 AND rank3 GREATER_EQUAL 5)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Scale/Zero-point operations
    if((rank1 LESS_EQUAL 3 AND rank2 GREATER_EQUAL 5) OR (rank1 GREATER_EQUAL 5 AND rank2 LESS_EQUAL 3))
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Check if type combination represents a dequantization pattern
function(is_dequantization_pattern type1 type2 type3 result_var)
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)
    
    # Dequantization patterns: quantized -> float
    if(rank1 LESS_EQUAL 3 AND rank2 GREATER_EQUAL 5 AND rank3 GREATER_EQUAL 5)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Mixed quantization operations
    if(rank1 LESS_EQUAL 3 AND rank2 GREATER_EQUAL 5 AND rank3 LESS_EQUAL 3)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Semantic validation for type combinations
function(is_semantically_valid_combination type1 type2 type3 operation_type result_var)
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)
    
    # Rule 1: Type availability check
    if(rank1 EQUAL -1 OR rank2 EQUAL -1 OR rank3 EQUAL -1)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 2: String operation semantics
    if(rank1 EQUAL -10 AND rank2 EQUAL -10 AND rank3 EQUAL -10)
        set(${result_var} FALSE PARENT_SCOPE)  # Triple string operations invalid
        return()
    endif()
    
    # Rule 3: Quantization pattern recognition
    is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
    if(is_quant)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    is_dequantization_pattern(${type1} ${type2} ${type3} is_dequant)
    if(is_dequant)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 4: Precision waste prevention (only reject if not quantization)
    math(EXPR rank_diff "${rank1} - ${rank3}")
    if(rank_diff GREATER 5)
        set(${result_var} FALSE PARENT_SCOPE)  # Extreme precision loss
        return()
    endif()
    
    # Rule 5: Boolean/masking operations (allow all - valid in ML)
    if(type1 STREQUAL "bool" OR type2 STREQUAL "bool" OR type3 STREQUAL "bool")
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 6: String + numeric (tokenization, encoding, feature extraction)
    if((rank1 EQUAL -10 AND rank2 GREATER 0) OR (rank1 GREATER 0 AND rank2 EQUAL -10))
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Default: allow combination if no specific rules reject it
    set(${result_var} TRUE PARENT_SCOPE)
endfunction()

# Apply quantization-specific pattern validation
function(apply_quantization_patterns type_list result_var)
    set(filtered_combinations "")
    
    foreach(combination ${type_list})
        string(REPLACE "," ";" combo_parts ${combination})
        list(LENGTH combo_parts combo_len)
        
        if(combo_len EQUAL 3)
            list(GET combo_parts 0 type1)
            list(GET combo_parts 1 type2)
            list(GET combo_parts 2 type3)
            
            # Check quantization patterns
            is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
            is_dequantization_pattern(${type1} ${type2} ${type3} is_dequant)
            
            if(is_quant OR is_dequant)
                list(APPEND filtered_combinations ${combination})
            endif()
        else()
            # For 2-type combinations, apply simpler filtering
            list(GET combo_parts 0 type1)
            list(GET combo_parts 1 type2)
            
            get_type_promotion_rank(${type1} rank1)
            get_type_promotion_rank(${type2} rank2)
            
            # Allow quantization-related pairs
            if((rank1 LESS_EQUAL 3 AND rank2 GREATER_EQUAL 5) OR (rank1 GREATER_EQUAL 5 AND rank2 LESS_EQUAL 3))
                list(APPEND filtered_combinations ${combination})
            endif()
        endif()
    endforeach()
    
    set(${result_var} "${filtered_combinations}" PARENT_SCOPE)
endfunction()

# Get operation type from template file name
function(get_operation_type_from_template template_file result_var)
    get_filename_component(template_name ${template_file} NAME_WE)
    
    if(template_name MATCHES "pairwise")
        set(${result_var} "pairwise" PARENT_SCOPE)
    elseif(template_name MATCHES "reduction")
        set(${result_var} "reduction" PARENT_SCOPE)
    elseif(template_name MATCHES "indexreduction")
        set(${result_var} "indexreduction" PARENT_SCOPE)
    elseif(template_name MATCHES "transform")
        set(${result_var} "transform" PARENT_SCOPE)
    else()
        set(${result_var} "generic" PARENT_SCOPE)
    endif()
endfunction()

# Generate type combinations for specific operation
function(generate_combinations_for_operation operation_type types_list result_var)
    set(valid_combinations "")
    
    # Get available types (from TypeValidation.cmake)
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()
    
    # Generate 2-type combinations
    foreach(type1 ${SD_ALL_TYPES_LIST})
        foreach(type2 ${SD_ALL_TYPES_LIST})
            is_semantically_valid_combination(${type1} ${type2} ${type1} ${operation_type} is_valid)
            if(is_valid)
                list(APPEND valid_combinations "${type1},${type2}")
            endif()
        endforeach()
    endforeach()
    
    # Generate 3-type combinations if needed
    if(operation_type STREQUAL "pairwise" OR operation_type STREQUAL "transform")
        foreach(type1 ${SD_ALL_TYPES_LIST})
            foreach(type2 ${SD_ALL_TYPES_LIST})
                foreach(type3 ${SD_ALL_TYPES_LIST})
                    is_semantically_valid_combination(${type1} ${type2} ${type3} ${operation_type} is_valid)
                    if(is_valid)
                        list(APPEND valid_combinations "${type1},${type2},${type3}")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    endif()
    
    set(${result_var} "${valid_combinations}" PARENT_SCOPE)
endfunction()

# Process template files with generated combinations
function(process_template_files)
    # Find all template files
    file(GLOB_RECURSE TEMPLATE_FILES 
         "${CMAKE_SOURCE_DIR}/include/loops/**/*.cpp.in"
         "${CMAKE_SOURCE_DIR}/include/loops/**/*.cu.in")
    
    foreach(template_file ${TEMPLATE_FILES})
        get_operation_type_from_template(${template_file} operation_type)
        generate_combinations_for_operation(${operation_type} "" combinations)
        
        # Generate macro definitions for combinations
        generate_combination_macros(${combinations} macro_definitions)
        
        # Set template variables
        set(COMB1 "")
        set(COMB2 "")
        set(COMB3 "")
        
        # Process combinations into template variables
        foreach(combination ${combinations})
            string(REPLACE "," ";" combo_parts ${combination})
            list(LENGTH combo_parts combo_len)
            
            if(combo_len EQUAL 2)
                list(GET combo_parts 0 type1)
                list(GET combo_parts 1 type2)
                string(APPEND COMB1 "INSTANTIATE_TEMPLATE(${type1}, ${type2});\n")
            elseif(combo_len EQUAL 3)
                list(GET combo_parts 0 type1)
                list(GET combo_parts 1 type2)
                list(GET combo_parts 2 type3)
                string(APPEND COMB3 "INSTANTIATE_TEMPLATE_3(${type1}, ${type2}, ${type3});\n")
            endif()
        endforeach()
        
        # Configure output file
        get_filename_component(template_dir ${template_file} DIRECTORY)
        get_filename_component(template_name ${template_file} NAME_WE)
        set(output_file "${CMAKE_BINARY_DIR}/generated/${template_name}.cpp")
        
        configure_file(${template_file} ${output_file} @ONLY)
        
        # Add to processed files list
        list(APPEND PROCESSED_TEMPLATE_FILES ${output_file})
    endforeach()
    
    set(PROCESSED_TEMPLATE_FILES ${PROCESSED_TEMPLATE_FILES} CACHE INTERNAL "Processed template files")
endfunction()

# Generate combination macros
function(generate_combination_macros combinations result_var)
    set(macro_content "")
    
    # Generate SD_COMMON_TYPES_PART_X macros
    set(part_counter 1)
    set(current_part "")
    set(combinations_per_part 50)  # Limit combinations per macro part
    
    set(combo_count 0)
    foreach(combination ${combinations})
        string(REPLACE "," ";" combo_parts ${combination})
        list(LENGTH combo_parts combo_len)
        
        if(combo_len EQUAL 2)
            list(GET combo_parts 0 type1)
            list(GET combo_parts 1 type2)
            string(APPEND current_part "(${type1}, ${type2}) \\\n")
        elseif(combo_len EQUAL 3)
            list(GET combo_parts 0 type1)
            list(GET combo_parts 1 type2)
            list(GET combo_parts 2 type3)
            string(APPEND current_part "(${type1}, ${type2}, ${type3}) \\\n")
        endif()
        
        math(EXPR combo_count "${combo_count} + 1")
        
        if(combo_count GREATER_EQUAL combinations_per_part)
            string(APPEND macro_content "#define SD_COMMON_TYPES_PART_${part_counter} \\\n${current_part}\n\n")
            math(EXPR part_counter "${part_counter} + 1")
            set(current_part "")
            set(combo_count 0)
        endif()
    endforeach()
    
    # Add remaining combinations
    if(NOT current_part STREQUAL "")
        string(APPEND macro_content "#define SD_COMMON_TYPES_PART_${part_counter} \\\n${current_part}\n\n")
    endif()
    
    set(${result_var} "${macro_content}" PARENT_SCOPE)
endfunction()

# Main function to generate all type combinations
function(generate_type_combinations)
    message(STATUS "Generating semantic type combinations...")
    
    # Get current type profile
    if(DEFINED SD_TYPE_PROFILE)
        message(STATUS "Using type profile: ${SD_TYPE_PROFILE}")
        apply_type_profile("${SD_TYPE_PROFILE}")
    endif()
    
    # Generate combinations for each operation type
    set(operation_types "pairwise" "reduction" "indexreduction" "transform" "generic")
    
    foreach(op_type ${operation_types})
        generate_combinations_for_operation(${op_type} "" combinations)
        message(STATUS "Generated ${op_type} combinations: ${CMAKE_MATCH_COUNT}")
        
        # Store combinations for later use
        set(COMBINATIONS_${op_type} ${combinations} CACHE INTERNAL "Combinations for ${op_type}")
    endforeach()
    
    # Process all template files
    process_template_files()
    
    message(STATUS "Type combination generation completed")
endfunction()

# Estimate combination impact
function(estimate_combination_impact types_list operation_types result_var)
    list(LENGTH types_list type_count)
    
    set(total_combinations 0)
    foreach(op_type ${operation_types})
        if(op_type STREQUAL "pairwise" OR op_type STREQUAL "transform")
            # 3-type combinations: n^3
            math(EXPR op_combinations "${type_count} * ${type_count} * ${type_count}")
        else()
            # 2-type combinations: n^2
            math(EXPR op_combinations "${type_count} * ${type_count}")
        endif()
        
        math(EXPR total_combinations "${total_combinations} + ${op_combinations}")
    endforeach()
    
    set(${result_var} ${total_combinations} PARENT_SCOPE)
endfunction()

# Suggest type optimizations
function(suggest_type_optimizations current_types target_workload result_var)
    set(suggestions "")
    
    if(target_workload STREQUAL "quantization")
        list(APPEND suggestions "Focus on INT8/UINT8 + FP32 combinations")
        list(APPEND suggestions "Preserve quantization patterns")
        list(APPEND suggestions "Remove precision waste combinations")
    elseif(target_workload STREQUAL "training")
        list(APPEND suggestions "Optimize for mixed precision (FP16/FP32)")
        list(APPEND suggestions "Include gradient accumulation types")
        list(APPEND suggestions "Preserve high precision combinations")
    elseif(target_workload STREQUAL "inference")
        list(APPEND suggestions "Focus on deployment types")
        list(APPEND suggestions "Optimize for quantized inference")
        list(APPEND suggestions "Remove training-specific combinations")
    endif()
    
    set(${result_var} "${suggestions}" PARENT_SCOPE)
endfunction()

# Function to categorize types for semantic filtering
function(categorize_type type_name result_var)
    normalize_type_enhanced("${type_name}" normalized_type)

    if(normalized_type MATCHES "^(float|double|bfloat16|float16)$")
        set(${result_var} "FLOATING_POINT" PARENT_SCOPE)
    elseif(normalized_type MATCHES "^(int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)$")
        set(${result_var} "INTEGER" PARENT_SCOPE)
    elseif(normalized_type STREQUAL "bool")
        set(${result_var} "BOOLEAN" PARENT_SCOPE)
    elseif(normalized_type MATCHES "std::.*string")
        set(${result_var} "STRING" PARENT_SCOPE)
    else()
        set(${result_var} "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Enhanced semantic validation with proper category-based logic
# Enhanced semantic validation with proper category-based logic
function(is_semantically_valid_pairwise_combination type1 type2 type3 result_var)
    # Get type categories
    categorize_type(${type1} cat1)
    categorize_type(${type2} cat2)
    categorize_type(${type3} cat3)

    # Get promotion ranks for fallback logic
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)

    # Rule 1: Reject unknown types immediately
    if(rank1 EQUAL -1 OR rank2 EQUAL -1 OR rank3 EQUAL -1)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Same-category homogeneous operations (most common and valid)
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # All same category - apply category-specific rules
        if(cat1 STREQUAL "FLOATING_POINT")
            # Float -> Float: must follow precision hierarchy
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow precision reduction only if quantization-like
            is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
            set(${result_var} ${is_quant} PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "INTEGER")
            # Integer -> Integer: follow bit-width hierarchy with exceptions
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow downcast for quantization (large int -> int8/uint8)
            if(rank3 LESS_EQUAL 2 AND max_input_rank GREATER_EQUAL 4)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "BOOLEAN")
            # Bool -> Bool: always valid for masking/logical operations
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "STRING")
            # String -> String: generally invalid (concatenation handled differently)
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 3: Mixed category operations - ML-specific patterns
    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER")
        # (Float, Float) -> Integer: Quantization pattern
        if(rank3 LESS_EQUAL 2)  # Targeting int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "FLOATING_POINT")
        # (Integer, Float) -> Float: Dequantization pattern
        if(rank1 LESS_EQUAL 2)  # From int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "INTEGER" AND cat3 STREQUAL "FLOATING_POINT")
        # (Float, Integer) -> Float: Scale/bias operations
        if(rank2 LESS_EQUAL 2)  # Using int8/uint8 as parameter
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # NEW Rule 4: Computer Vision Patterns
    if(cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER")
        # Pattern: (Integer, Float, Integer) - Common in CV operations
        # Examples: crop_and_resize (uint16_t, float, uint32_t)
        # Allow if input/output are reasonable integer types for CV
        if((rank1 GREATER_EQUAL 2 AND rank1 LESS_EQUAL 5) AND  # uint8_t to uint64_t range
        (rank3 GREATER_EQUAL 2 AND rank3 LESS_EQUAL 5) AND  # uint8_t to uint64_t range
        (rank2 GREATER_EQUAL 6))  # float16 or higher for interpolation
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 5: Boolean broadcasting - valid with any numeric type
    if(cat1 STREQUAL "BOOLEAN" OR cat2 STREQUAL "BOOLEAN")
        # Boolean can broadcast with any numeric type
        if((cat1 STREQUAL "BOOLEAN" AND (cat2 STREQUAL "FLOATING_POINT" OR cat2 STREQUAL "INTEGER")) OR
        (cat2 STREQUAL "BOOLEAN" AND (cat1 STREQUAL "FLOATING_POINT" OR cat1 STREQUAL "INTEGER")))
            # Output should follow non-boolean input type category
            if(cat1 STREQUAL "BOOLEAN")
                categorize_type(${type2} expected_cat)
            else()
                categorize_type(${type1} expected_cat)
            endif()

            if(cat3 STREQUAL expected_cat)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # Rule 6: String + Numeric operations (NLP tokenization, encoding)
    if((cat1 STREQUAL "STRING" AND (cat2 STREQUAL "INTEGER" OR cat2 STREQUAL "FLOATING_POINT")) OR
    (cat2 STREQUAL "STRING" AND (cat1 STREQUAL "INTEGER" OR cat1 STREQUAL "FLOATING_POINT")))
        # String operations typically output numeric results
        if(cat3 STREQUAL "INTEGER" OR cat3 STREQUAL "FLOATING_POINT")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 7: Type promotion within same category with different precision
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # Within same category, check for valid promotion
        if(rank1 GREATER rank2)
            set(max_input_rank ${rank1})
        else()
            set(max_input_rank ${rank2})
        endif()
        if(rank3 EQUAL max_input_rank)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 8: Integer indexing operations (critical for array operations)
    if((cat1 STREQUAL "INTEGER" OR cat2 STREQUAL "INTEGER") AND cat3 STREQUAL "INTEGER")
        # Allow if at least one input and output are integers
        if((rank1 GREATER_EQUAL 4 OR rank2 GREATER_EQUAL 4) AND rank3 GREATER_EQUAL 4)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: reject unlikely combinations to reduce template explosion
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Function to normalize a type name (enhanced)
function(normalize_type input_type output_var)
    normalize_type_enhanced("${input_type}" result)
    set(${output_var} "${result}" PARENT_SCOPE)
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

# Function to analyze and report type combination patterns
function(analyze_combination_patterns combinations workload_profile)
    set(pattern_counts "")
    set(quantization_patterns 0)
    set(dequantization_patterns 0)
    set(same_type_patterns 0)
    set(mixed_precision_patterns 0)
    set(string_numeric_patterns 0)
    set(boolean_patterns 0)

    foreach(combination ${combinations})
        string(REPLACE "," ";" combo_parts ${combination})
        list(LENGTH combo_parts combo_len)
        if(combo_len GREATER_EQUAL 3)
            list(GET combo_parts 0 type1)
            list(GET combo_parts 1 type2)
            list(GET combo_parts 2 type3)

            # Analyze pattern types
            is_quantization_pattern("${type1}" "${type2}" "${type3}" is_quant)
            if(is_quant)
                math(EXPR quantization_patterns "${quantization_patterns} + 1")
            endif()

            is_dequantization_pattern("${type1}" "${type2}" "${type3}" is_dequant)
            if(is_dequant)
                math(EXPR dequantization_patterns "${dequantization_patterns} + 1")
            endif()

            if(type1 STREQUAL type2 AND type2 STREQUAL type3)
                math(EXPR same_type_patterns "${same_type_patterns} + 1")
            endif()

            if((type1 MATCHES "float16|bfloat16" AND type2 MATCHES "float16|bfloat16|float") OR
            (type1 MATCHES "float" AND type2 MATCHES "float16|bfloat16"))
                math(EXPR mixed_precision_patterns "${mixed_precision_patterns} + 1")
            endif()

            categorize_type("${type1}" cat1)
            categorize_type("${type2}" cat2)
            if((cat1 STREQUAL "STRING" AND cat2 STREQUAL "INTEGER") OR
            (cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "STRING"))
                math(EXPR string_numeric_patterns "${string_numeric_patterns} + 1")
            endif()

            if(type1 STREQUAL "bool" OR type2 STREQUAL "bool" OR type3 STREQUAL "bool")
                math(EXPR boolean_patterns "${boolean_patterns} + 1")
            endif()
        endif()
    endforeach()

    list(LENGTH combinations total_combinations)

    message(STATUS "")
    message(STATUS "Pattern Analysis for workload '${workload_profile}':")
    message(STATUS "  Total combinations: ${total_combinations}")
    message(STATUS "  Same-type patterns: ${same_type_patterns}")
    message(STATUS "  Quantization patterns: ${quantization_patterns}")
    message(STATUS "  Dequantization patterns: ${dequantization_patterns}")
    message(STATUS "  Mixed precision patterns: ${mixed_precision_patterns}")
    message(STATUS "  String+Numeric patterns: ${string_numeric_patterns}")
    message(STATUS "  Boolean patterns: ${boolean_patterns}")
    message(STATUS "")
endfunction()

# Main orchestration function for enhanced semantic validation system
function(setup_enhanced_semantic_validation)
    message(STATUS "=== SETTING UP ENHANCED SEMANTIC VALIDATION SYSTEM ===")

    # Determine workload profile
    set(workload_profile "generic")
    if(DEFINED SD_WORKLOAD_PROFILE)
        set(workload_profile "${SD_WORKLOAD_PROFILE}")
    elseif(DEFINED SD_TYPE_PROFILE)
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    message(STATUS "Active workload profile: ${workload_profile}")

    # Generate optimized combinations
    generate_pairwise_combinations_enhanced("${workload_profile}" optimized_combinations)

    # Store results
    set(SD_OPTIMIZED_COMBINATIONS "${optimized_combinations}" PARENT_SCOPE)

    message(STATUS "Enhanced semantic validation setup completed")
    list(LENGTH optimized_combinations combo_count)
    message(STATUS "Generated ${combo_count} optimized type combinations")
endfunction()

function(apply_workload_filter_pairwise type1 type2 type3 workload_profile result_var)
    if(NOT workload_profile OR workload_profile STREQUAL "")
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Get type categories and ranks
    categorize_type(${type1} cat1)
    categorize_type(${type2} cat2)
    categorize_type(${type3} cat3)
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)

    if(workload_profile STREQUAL "quantization")
        # Focus on quantization-related patterns
        # High priority: quantization patterns
        if((cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER" AND rank3 LESS_EQUAL 2) OR
        (cat1 STREQUAL "INTEGER" AND rank1 LESS_EQUAL 2 AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "FLOATING_POINT") OR
        (cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "INTEGER" AND rank2 LESS_EQUAL 2 AND cat3 STREQUAL "FLOATING_POINT"))
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()

        # Medium priority: essential numeric operations on quantization-friendly types
        if((type1 MATCHES "int8_t|uint8_t|float" AND type2 MATCHES "int8_t|uint8_t|float" AND type3 MATCHES "int8_t|uint8_t|float") OR
        (type1 MATCHES "int32_t|int64_t" AND type2 MATCHES "int32_t|int64_t" AND type3 MATCHES "int32_t|int64_t"))
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()

        set(${result_var} FALSE PARENT_SCOPE)

    elseif(workload_profile STREQUAL "training")
        # High priority: mixed precision patterns (fp16/bf16 + fp32)
        if((type1 MATCHES "float16|bfloat16" AND type2 MATCHES "float16|bfloat16|float" AND type3 MATCHES "float16|bfloat16|float") OR
        (type1 MATCHES "float" AND type2 MATCHES "float16|bfloat16" AND type3 MATCHES "float16|bfloat16|float"))
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()

        # Essential: indexing operations
        if(type1 MATCHES "int32_t|int64_t" AND type2 MATCHES "int32_t|int64_t" AND type3 MATCHES "int32_t|int64_t")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()

        set(${result_var} FALSE PARENT_SCOPE)

    else()
        # Unknown workload profile - allow all valid combinations
        set(${result_var} TRUE PARENT_SCOPE)
    endif()
endfunction()


# Generate semantically filtered pairwise combinations with enhanced validation
function(generate_pairwise_combinations_enhanced workload_profile result_var)
    set(valid_combinations "")
    set(total_checked 0)
    set(semantic_passed 0)
    set(workload_passed 0)

    # Get available types
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()

    message(STATUS "Generating pairwise combinations for workload: ${workload_profile}")

    foreach(type1 ${SD_ALL_TYPES_LIST})
        foreach(type2 ${SD_ALL_TYPES_LIST})
            foreach(type3 ${SD_ALL_TYPES_LIST})
                math(EXPR total_checked "${total_checked} + 1")

                # First pass: semantic validation
                is_semantically_valid_pairwise_combination("${type1}" "${type2}" "${type3}" is_semantic_valid)
                if(is_semantic_valid)
                    math(EXPR semantic_passed "${semantic_passed} + 1")
                    list(APPEND valid_combinations "${type1},${type2},${type3}")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # Report filtering statistics
    message(STATUS "Combination filtering results:")
    message(STATUS "  Total checked: ${total_checked}")
    message(STATUS "  Semantic valid: ${semantic_passed}")

    if(total_checked GREATER 0)
        math(EXPR semantic_reduction "100 * (${total_checked} - ${semantic_passed}) / ${total_checked}")
        message(STATUS "  Semantic filtering reduced by: ${semantic_reduction}%")
    endif()

    set(${result_var} "${valid_combinations}" PARENT_SCOPE)
endfunction()

# Main orchestration function for enhanced semantic validation system
function(setup_enhanced_semantic_validation)
    message(STATUS "=== SETTING UP ENHANCED SEMANTIC VALIDATION SYSTEM ===")

    # Determine workload profile
    set(workload_profile "generic")
    if(DEFINED SD_WORKLOAD_PROFILE)
        set(workload_profile "${SD_WORKLOAD_PROFILE}")
    elseif(DEFINED SD_TYPE_PROFILE)
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    message(STATUS "Active workload profile: ${workload_profile}")

    # Generate optimized combinations
    generate_pairwise_combinations_enhanced("${workload_profile}" optimized_combinations)

    # Store results
    set(SD_OPTIMIZED_COMBINATIONS "${optimized_combinations}" PARENT_SCOPE)

    message(STATUS "Enhanced semantic validation setup completed")
    list(LENGTH optimized_combinations combo_count)
    message(STATUS "Generated ${combo_count} optimized type combinations")
endfunction()

# Function to categorize types for semantic filtering
function(categorize_type type_name result_var)
    normalize_type_enhanced("${type_name}" normalized_type)

    if(normalized_type MATCHES "^(float|double|bfloat16|float16)$")
        set(${result_var} "FLOATING_POINT" PARENT_SCOPE)
    elseif(normalized_type MATCHES "^(int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)$")
        set(${result_var} "INTEGER" PARENT_SCOPE)
    elseif(normalized_type STREQUAL "bool")
        set(${result_var} "BOOLEAN" PARENT_SCOPE)
    elseif(normalized_type MATCHES "std::.*string")
        set(${result_var} "STRING" PARENT_SCOPE)
    else()
        set(${result_var} "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Enhanced semantic validation with proper category-based logic
function(is_semantically_valid_pairwise_combination type1 type2 type3 result_var)
    # Get type categories
    categorize_type(${type1} cat1)
    categorize_type(${type2} cat2)
    categorize_type(${type3} cat3)

    # Get promotion ranks for fallback logic
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)

    # Rule 1: Reject unknown types immediately
    if(rank1 EQUAL -1 OR rank2 EQUAL -1 OR rank3 EQUAL -1)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Same-category homogeneous operations (most common and valid)
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # All same category - apply category-specific rules
        if(cat1 STREQUAL "FLOATING_POINT")
            # Float -> Float: must follow precision hierarchy
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow precision reduction only if quantization-like
            is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
            set(${result_var} ${is_quant} PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "INTEGER")
            # Integer -> Integer: follow bit-width hierarchy with exceptions
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow downcast for quantization (large int -> int8/uint8)
            if(rank3 LESS_EQUAL 2 AND max_input_rank GREATER_EQUAL 4)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "BOOLEAN")
            # Bool -> Bool: always valid for masking/logical operations
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "STRING")
            # String -> String: generally invalid (concatenation handled differently)
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 3: Mixed category operations - ML-specific patterns
    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER")
        # (Float, Float) -> Integer: Quantization pattern
        if(rank3 LESS_EQUAL 2)  # Targeting int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "FLOATING_POINT")
        # (Integer, Float) -> Float: Dequantization pattern
        if(rank1 LESS_EQUAL 2)  # From int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "INTEGER" AND cat3 STREQUAL "FLOATING_POINT")
        # (Float, Integer) -> Float: Scale/bias operations
        if(rank2 LESS_EQUAL 2)  # Using int8/uint8 as parameter
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 4: Boolean broadcasting - valid with any numeric type
    if(cat1 STREQUAL "BOOLEAN" OR cat2 STREQUAL "BOOLEAN")
        # Boolean can broadcast with any numeric type
        if((cat1 STREQUAL "BOOLEAN" AND (cat2 STREQUAL "FLOATING_POINT" OR cat2 STREQUAL "INTEGER")) OR
           (cat2 STREQUAL "BOOLEAN" AND (cat1 STREQUAL "FLOATING_POINT" OR cat1 STREQUAL "INTEGER")))
            # Output should follow non-boolean input type category
            if(cat1 STREQUAL "BOOLEAN")
                categorize_type(${type2} expected_cat)
            else()
                categorize_type(${type1} expected_cat)
            endif()

            if(cat3 STREQUAL expected_cat)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # Rule 5: String + Numeric operations (NLP tokenization, encoding)
    if((cat1 STREQUAL "STRING" AND (cat2 STREQUAL "INTEGER" OR cat2 STREQUAL "FLOATING_POINT")) OR
       (cat2 STREQUAL "STRING" AND (cat1 STREQUAL "INTEGER" OR cat1 STREQUAL "FLOATING_POINT")))
        # String operations typically output numeric results
        if(cat3 STREQUAL "INTEGER" OR cat3 STREQUAL "FLOATING_POINT")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 6: Type promotion within same category with different precision
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # Within same category, check for valid promotion
        if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
        if(rank3 EQUAL max_input_rank)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 7: Integer indexing operations (critical for array operations)
    if((cat1 STREQUAL "INTEGER" OR cat2 STREQUAL "INTEGER") AND cat3 STREQUAL "INTEGER")
        # Allow if at least one input and output are integers
        if((rank1 GREATER_EQUAL 4 OR rank2 GREATER_EQUAL 4) AND rank3 GREATER_EQUAL 4)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: reject unlikely combinations to reduce template explosion
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Generate semantically filtered pairwise combinations with enhanced validation
function(generate_pairwise_combinations_enhanced workload_profile result_var)
    set(valid_combinations "")
    set(total_checked 0)
    set(semantic_passed 0)
    set(workload_passed 0)

    # Get available types
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()

    message(STATUS "Generating pairwise combinations for workload: ${workload_profile}")

    foreach(type1 ${SD_ALL_TYPES_LIST})
        foreach(type2 ${SD_ALL_TYPES_LIST})
            foreach(type3 ${SD_ALL_TYPES_LIST})
                math(EXPR total_checked "${total_checked} + 1")

                # First pass: semantic validation
                is_semantically_valid_pairwise_combination("${type1}" "${type2}" "${type3}" is_semantic_valid)
                if(is_semantic_valid)
                    math(EXPR semantic_passed "${semantic_passed} + 1")
                    list(APPEND valid_combinations "${type1},${type2},${type3}")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # Report filtering statistics
    message(STATUS "Combination filtering results:")
    message(STATUS "  Total checked: ${total_checked}")
    message(STATUS "  Semantic valid: ${semantic_passed}")

    if(total_checked GREATER 0)
        math(EXPR semantic_reduction "100 * (${total_checked} - ${semantic_passed}) / ${total_checked}")
        message(STATUS "  Semantic filtering reduced by: ${semantic_reduction}%")
    endif()

    set(${result_var} "${valid_combinations}" PARENT_SCOPE)
endfunction()

# Main orchestration function for enhanced semantic validation system
function(setup_enhanced_semantic_validation)
    message(STATUS "=== SETTING UP ENHANCED SEMANTIC VALIDATION SYSTEM ===")

    # Determine workload profile
    set(workload_profile "generic")
    if(DEFINED SD_WORKLOAD_PROFILE)
        set(workload_profile "${SD_WORKLOAD_PROFILE}")
    elseif(DEFINED SD_TYPE_PROFILE)
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    message(STATUS "Active workload profile: ${workload_profile}")

    # Generate optimized combinations
    generate_pairwise_combinations_enhanced("${workload_profile}" optimized_combinations)

    # Store results
    set(SD_OPTIMIZED_COMBINATIONS "${optimized_combinations}" PARENT_SCOPE)

    message(STATUS "Enhanced semantic validation setup completed")
    list(LENGTH optimized_combinations combo_count)
    message(STATUS "Generated ${combo_count} optimized type combinations")
endfunction()

# Function to categorize types for semantic filtering
function(categorize_type type_name result_var)
    normalize_type_enhanced("${type_name}" normalized_type)

    if(normalized_type MATCHES "^(float|double|bfloat16|float16)$")
        set(${result_var} "FLOATING_POINT" PARENT_SCOPE)
    elseif(normalized_type MATCHES "^(int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)$")
        set(${result_var} "INTEGER" PARENT_SCOPE)
    elseif(normalized_type STREQUAL "bool")
        set(${result_var} "BOOLEAN" PARENT_SCOPE)
    elseif(normalized_type MATCHES "std::.*string")
        set(${result_var} "STRING" PARENT_SCOPE)
    else()
        set(${result_var} "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Enhanced semantic validation with proper category-based logic
function(is_semantically_valid_pairwise_combination type1 type2 type3 result_var)
    # Get type categories
    categorize_type(${type1} cat1)
    categorize_type(${type2} cat2)
    categorize_type(${type3} cat3)

    # Get promotion ranks for fallback logic
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)

    # Rule 1: Reject unknown types immediately
    if(rank1 EQUAL -1 OR rank2 EQUAL -1 OR rank3 EQUAL -1)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Same-category homogeneous operations (most common and valid)
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # All same category - apply category-specific rules
        if(cat1 STREQUAL "FLOATING_POINT")
            # Float -> Float: must follow precision hierarchy
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow precision reduction only if quantization-like
            is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
            set(${result_var} ${is_quant} PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "INTEGER")
            # Integer -> Integer: follow bit-width hierarchy with exceptions
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow downcast for quantization (large int -> int8/uint8)
            if(rank3 LESS_EQUAL 2 AND max_input_rank GREATER_EQUAL 4)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "BOOLEAN")
            # Bool -> Bool: always valid for masking/logical operations
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "STRING")
            # String -> String: generally invalid (concatenation handled differently)
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 3: Mixed category operations - ML-specific patterns
    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER")
        # (Float, Float) -> Integer: Quantization pattern
        if(rank3 LESS_EQUAL 2)  # Targeting int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "FLOATING_POINT")
        # (Integer, Float) -> Float: Dequantization pattern
        if(rank1 LESS_EQUAL 2)  # From int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "INTEGER" AND cat3 STREQUAL "FLOATING_POINT")
        # (Float, Integer) -> Float: Scale/bias operations
        if(rank2 LESS_EQUAL 2)  # Using int8/uint8 as parameter
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 4: Boolean broadcasting - valid with any numeric type
    if(cat1 STREQUAL "BOOLEAN" OR cat2 STREQUAL "BOOLEAN")
        # Boolean can broadcast with any numeric type
        if((cat1 STREQUAL "BOOLEAN" AND (cat2 STREQUAL "FLOATING_POINT" OR cat2 STREQUAL "INTEGER")) OR
           (cat2 STREQUAL "BOOLEAN" AND (cat1 STREQUAL "FLOATING_POINT" OR cat1 STREQUAL "INTEGER")))
            # Output should follow non-boolean input type category
            if(cat1 STREQUAL "BOOLEAN")
                categorize_type(${type2} expected_cat)
            else()
                categorize_type(${type1} expected_cat)
            endif()

            if(cat3 STREQUAL expected_cat)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # Rule 5: String + Numeric operations (NLP tokenization, encoding)
    if((cat1 STREQUAL "STRING" AND (cat2 STREQUAL "INTEGER" OR cat2 STREQUAL "FLOATING_POINT")) OR
       (cat2 STREQUAL "STRING" AND (cat1 STREQUAL "INTEGER" OR cat1 STREQUAL "FLOATING_POINT")))
        # String operations typically output numeric results
        if(cat3 STREQUAL "INTEGER" OR cat3 STREQUAL "FLOATING_POINT")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 6: Type promotion within same category with different precision
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # Within same category, check for valid promotion
        if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
        if(rank3 EQUAL max_input_rank)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 7: Integer indexing operations (critical for array operations)
    if((cat1 STREQUAL "INTEGER" OR cat2 STREQUAL "INTEGER") AND cat3 STREQUAL "INTEGER")
        # Allow if at least one input and output are integers
        if((rank1 GREATER_EQUAL 4 OR rank2 GREATER_EQUAL 4) AND rank3 GREATER_EQUAL 4)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: reject unlikely combinations to reduce template explosion
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Generate semantically filtered pairwise combinations with enhanced validation
function(generate_pairwise_combinations_enhanced workload_profile result_var)
    set(valid_combinations "")
    set(total_checked 0)
    set(semantic_passed 0)
    set(workload_passed 0)

    # Get available types
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()

    message(STATUS "Generating pairwise combinations for workload: ${workload_profile}")

    foreach(type1 ${SD_ALL_TYPES_LIST})
        foreach(type2 ${SD_ALL_TYPES_LIST})
            foreach(type3 ${SD_ALL_TYPES_LIST})
                math(EXPR total_checked "${total_checked} + 1")

                # First pass: semantic validation
                is_semantically_valid_pairwise_combination("${type1}" "${type2}" "${type3}" is_semantic_valid)
                if(is_semantic_valid)
                    math(EXPR semantic_passed "${semantic_passed} + 1")
                    list(APPEND valid_combinations "${type1},${type2},${type3}")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # Report filtering statistics
    message(STATUS "Combination filtering results:")
    message(STATUS "  Total checked: ${total_checked}")
    message(STATUS "  Semantic valid: ${semantic_passed}")

    if(total_checked GREATER 0)
        math(EXPR semantic_reduction "100 * (${total_checked} - ${semantic_passed}) / ${total_checked}")
        message(STATUS "  Semantic filtering reduced by: ${semantic_reduction}%")
    endif()

    set(${result_var} "${valid_combinations}" PARENT_SCOPE)
endfunction()

# Main orchestration function for enhanced semantic validation system
function(setup_enhanced_semantic_validation)
    message(STATUS "=== SETTING UP ENHANCED SEMANTIC VALIDATION SYSTEM ===")

    # Determine workload profile
    set(workload_profile "generic")
    if(DEFINED SD_WORKLOAD_PROFILE)
        set(workload_profile "${SD_WORKLOAD_PROFILE}")
    elseif(DEFINED SD_TYPE_PROFILE)
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    message(STATUS "Active workload profile: ${workload_profile}")

    # Generate optimized combinations
    generate_pairwise_combinations_enhanced("${workload_profile}" optimized_combinations)

    # Store results
    set(SD_OPTIMIZED_COMBINATIONS "${optimized_combinations}" PARENT_SCOPE)

    message(STATUS "Enhanced semantic validation setup completed")
    list(LENGTH optimized_combinations combo_count)
    message(STATUS "Generated ${combo_count} optimized type combinations")
endfunction()

# Function to categorize types for semantic filtering
function(categorize_type type_name result_var)
    normalize_type_enhanced("${type_name}" normalized_type)

    if(normalized_type MATCHES "^(float|double|bfloat16|float16)$")
        set(${result_var} "FLOATING_POINT" PARENT_SCOPE)
    elseif(normalized_type MATCHES "^(int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)$")
        set(${result_var} "INTEGER" PARENT_SCOPE)
    elseif(normalized_type STREQUAL "bool")
        set(${result_var} "BOOLEAN" PARENT_SCOPE)
    elseif(normalized_type MATCHES "std::.*string")
        set(${result_var} "STRING" PARENT_SCOPE)
    else()
        set(${result_var} "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Enhanced semantic validation with proper category-based logic
function(is_semantically_valid_pairwise_combination type1 type2 type3 result_var)
    # Get type categories
    categorize_type(${type1} cat1)
    categorize_type(${type2} cat2)
    categorize_type(${type3} cat3)

    # Get promotion ranks for fallback logic
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)

    # Rule 1: Reject unknown types immediately
    if(rank1 EQUAL -1 OR rank2 EQUAL -1 OR rank3 EQUAL -1)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Same-category homogeneous operations (most common and valid)
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # All same category - apply category-specific rules
        if(cat1 STREQUAL "FLOATING_POINT")
            # Float -> Float: must follow precision hierarchy
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow precision reduction only if quantization-like
            is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
            set(${result_var} ${is_quant} PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "INTEGER")
            # Integer -> Integer: follow bit-width hierarchy with exceptions
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow downcast for quantization (large int -> int8/uint8)
            if(rank3 LESS_EQUAL 2 AND max_input_rank GREATER_EQUAL 4)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "BOOLEAN")
            # Bool -> Bool: always valid for masking/logical operations
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "STRING")
            # String -> String: generally invalid (concatenation handled differently)
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 3: Mixed category operations - ML-specific patterns
    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER")
        # (Float, Float) -> Integer: Quantization pattern
        if(rank3 LESS_EQUAL 2)  # Targeting int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "FLOATING_POINT")
        # (Integer, Float) -> Float: Dequantization pattern
        if(rank1 LESS_EQUAL 2)  # From int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "INTEGER" AND cat3 STREQUAL "FLOATING_POINT")
        # (Float, Integer) -> Float: Scale/bias operations
        if(rank2 LESS_EQUAL 2)  # Using int8/uint8 as parameter
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 4: Boolean broadcasting - valid with any numeric type
    if(cat1 STREQUAL "BOOLEAN" OR cat2 STREQUAL "BOOLEAN")
        # Boolean can broadcast with any numeric type
        if((cat1 STREQUAL "BOOLEAN" AND (cat2 STREQUAL "FLOATING_POINT" OR cat2 STREQUAL "INTEGER")) OR
           (cat2 STREQUAL "BOOLEAN" AND (cat1 STREQUAL "FLOATING_POINT" OR cat1 STREQUAL "INTEGER")))
            # Output should follow non-boolean input type category
            if(cat1 STREQUAL "BOOLEAN")
                categorize_type(${type2} expected_cat)
            else()
                categorize_type(${type1} expected_cat)
            endif()

            if(cat3 STREQUAL expected_cat)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # Rule 5: String + Numeric operations (NLP tokenization, encoding)
    if((cat1 STREQUAL "STRING" AND (cat2 STREQUAL "INTEGER" OR cat2 STREQUAL "FLOATING_POINT")) OR
       (cat2 STREQUAL "STRING" AND (cat1 STREQUAL "INTEGER" OR cat1 STREQUAL "FLOATING_POINT")))
        # String operations typically output numeric results
        if(cat3 STREQUAL "INTEGER" OR cat3 STREQUAL "FLOATING_POINT")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 6: Type promotion within same category with different precision
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # Within same category, check for valid promotion
        if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
        if(rank3 EQUAL max_input_rank)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 7: Integer indexing operations (critical for array operations)
    if((cat1 STREQUAL "INTEGER" OR cat2 STREQUAL "INTEGER") AND cat3 STREQUAL "INTEGER")
        # Allow if at least one input and output are integers
        if((rank1 GREATER_EQUAL 4 OR rank2 GREATER_EQUAL 4) AND rank3 GREATER_EQUAL 4)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: reject unlikely combinations to reduce template explosion
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Generate semantically filtered pairwise combinations with enhanced validation
function(generate_pairwise_combinations_enhanced workload_profile result_var)
    set(valid_combinations "")
    set(total_checked 0)
    set(semantic_passed 0)
    set(workload_passed 0)

    # Get available types
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()

    message(STATUS "Generating pairwise combinations for workload: ${workload_profile}")

    foreach(type1 ${SD_ALL_TYPES_LIST})
        foreach(type2 ${SD_ALL_TYPES_LIST})
            foreach(type3 ${SD_ALL_TYPES_LIST})
                math(EXPR total_checked "${total_checked} + 1")

                # First pass: semantic validation
                is_semantically_valid_pairwise_combination("${type1}" "${type2}" "${type3}" is_semantic_valid)
                if(is_semantic_valid)
                    math(EXPR semantic_passed "${semantic_passed} + 1")
                    list(APPEND valid_combinations "${type1},${type2},${type3}")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # Report filtering statistics
    message(STATUS "Combination filtering results:")
    message(STATUS "  Total checked: ${total_checked}")
    message(STATUS "  Semantic valid: ${semantic_passed}")

    if(total_checked GREATER 0)
        math(EXPR semantic_reduction "100 * (${total_checked} - ${semantic_passed}) / ${total_checked}")
        message(STATUS "  Semantic filtering reduced by: ${semantic_reduction}%")
    endif()

    set(${result_var} "${valid_combinations}" PARENT_SCOPE)
endfunction()

# Main orchestration function for enhanced semantic validation system
function(setup_enhanced_semantic_validation)
    message(STATUS "=== SETTING UP ENHANCED SEMANTIC VALIDATION SYSTEM ===")

    # Determine workload profile
    set(workload_profile "generic")
    if(DEFINED SD_WORKLOAD_PROFILE)
        set(workload_profile "${SD_WORKLOAD_PROFILE}")
    elseif(DEFINED SD_TYPE_PROFILE)
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    message(STATUS "Active workload profile: ${workload_profile}")

    # Generate optimized combinations
    generate_pairwise_combinations_enhanced("${workload_profile}" optimized_combinations)

    # Store results
    set(SD_OPTIMIZED_COMBINATIONS "${optimized_combinations}" PARENT_SCOPE)

    message(STATUS "Enhanced semantic validation setup completed")
    list(LENGTH optimized_combinations combo_count)
    message(STATUS "Generated ${combo_count} optimized type combinations")
endfunction()

# Function to categorize types for semantic filtering
function(categorize_type type_name result_var)
    normalize_type_enhanced("${type_name}" normalized_type)

    if(normalized_type MATCHES "^(float|double|bfloat16|float16)$")
        set(${result_var} "FLOATING_POINT" PARENT_SCOPE)
    elseif(normalized_type MATCHES "^(int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t)$")
        set(${result_var} "INTEGER" PARENT_SCOPE)
    elseif(normalized_type STREQUAL "bool")
        set(${result_var} "BOOLEAN" PARENT_SCOPE)
    elseif(normalized_type MATCHES "std::.*string")
        set(${result_var} "STRING" PARENT_SCOPE)
    else()
        set(${result_var} "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Enhanced semantic validation with proper category-based logic
function(is_semantically_valid_pairwise_combination type1 type2 type3 result_var)
    # Get type categories
    categorize_type(${type1} cat1)
    categorize_type(${type2} cat2)
    categorize_type(${type3} cat3)

    # Get promotion ranks for fallback logic
    get_type_promotion_rank(${type1} rank1)
    get_type_promotion_rank(${type2} rank2)
    get_type_promotion_rank(${type3} rank3)

    # Rule 1: Reject unknown types immediately
    if(rank1 EQUAL -1 OR rank2 EQUAL -1 OR rank3 EQUAL -1)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Same-category homogeneous operations (most common and valid)
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # All same category - apply category-specific rules
        if(cat1 STREQUAL "FLOATING_POINT")
            # Float -> Float: must follow precision hierarchy
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow precision reduction only if quantization-like
            is_quantization_pattern(${type1} ${type2} ${type3} is_quant)
            set(${result_var} ${is_quant} PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "INTEGER")
            # Integer -> Integer: follow bit-width hierarchy with exceptions
            if(rank1 GREATER rank2)
                set(max_input_rank ${rank1})
            else()
                set(max_input_rank ${rank2})
            endif()
            if(rank3 GREATER_EQUAL max_input_rank)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            # Allow downcast for quantization (large int -> int8/uint8)
            if(rank3 LESS_EQUAL 2 AND max_input_rank GREATER_EQUAL 4)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "BOOLEAN")
            # Bool -> Bool: always valid for masking/logical operations
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        elseif(cat1 STREQUAL "STRING")
            # String -> String: generally invalid (concatenation handled differently)
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 3: Mixed category operations - ML-specific patterns
    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "INTEGER")
        # (Float, Float) -> Integer: Quantization pattern
        if(rank3 LESS_EQUAL 2)  # Targeting int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "INTEGER" AND cat2 STREQUAL "FLOATING_POINT" AND cat3 STREQUAL "FLOATING_POINT")
        # (Integer, Float) -> Float: Dequantization pattern
        if(rank1 LESS_EQUAL 2)  # From int8/uint8
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(cat1 STREQUAL "FLOATING_POINT" AND cat2 STREQUAL "INTEGER" AND cat3 STREQUAL "FLOATING_POINT")
        # (Float, Integer) -> Float: Scale/bias operations
        if(rank2 LESS_EQUAL 2)  # Using int8/uint8 as parameter
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 4: Boolean broadcasting - valid with any numeric type
    if(cat1 STREQUAL "BOOLEAN" OR cat2 STREQUAL "BOOLEAN")
        # Boolean can broadcast with any numeric type
        if((cat1 STREQUAL "BOOLEAN" AND (cat2 STREQUAL "FLOATING_POINT" OR cat2 STREQUAL "INTEGER")) OR
           (cat2 STREQUAL "BOOLEAN" AND (cat1 STREQUAL "FLOATING_POINT" OR cat1 STREQUAL "INTEGER")))
            # Output should follow non-boolean input type category
            if(cat1 STREQUAL "BOOLEAN")
                categorize_type(${type2} expected_cat)
            else()
                categorize_type(${type1} expected_cat)
            endif()

            if(cat3 STREQUAL expected_cat)
                set(${result_var} TRUE PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()

    # Rule 5: String + Numeric operations (NLP tokenization, encoding)
    if((cat1 STREQUAL "STRING" AND (cat2 STREQUAL "INTEGER" OR cat2 STREQUAL "FLOATING_POINT")) OR
       (cat2 STREQUAL "STRING" AND (cat1 STREQUAL "INTEGER" OR cat1 STREQUAL "FLOATING_POINT")))
        # String operations typically output numeric results
        if(cat3 STREQUAL "INTEGER" OR cat3 STREQUAL "FLOATING_POINT")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 6: Type promotion within same category with different precision
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # Within same category, check for valid promotion
        if(rank1 GREATER rank2)
            set(max_input_rank ${rank1})
        else()
            set(max_input_rank ${rank2})
        endif()
        if(rank3 EQUAL max_input_rank)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 7: Integer indexing operations (critical for array operations)
    if((cat1 STREQUAL "INTEGER" OR cat2 STREQUAL "INTEGER") AND cat3 STREQUAL "INTEGER")
        # Allow if at least one input and output are integers
        if((rank1 GREATER_EQUAL 4 OR rank2 GREATER_EQUAL 4) AND rank3 GREATER_EQUAL 4)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: reject unlikely combinations to reduce template explosion
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Generate semantically filtered pairwise combinations with enhanced validation
function(generate_pairwise_combinations_enhanced workload_profile result_var)
    set(valid_combinations "")
    set(total_checked 0)
    set(semantic_passed 0)
    set(workload_passed 0)

    # Get available types
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()

    message(STATUS "Generating pairwise combinations for workload: ${workload_profile}")

    foreach(type1 ${SD_ALL_TYPES_LIST})
        foreach(type2 ${SD_ALL_TYPES_LIST})
            foreach(type3 ${SD_ALL_TYPES_LIST})
                math(EXPR total_checked "${total_checked} + 1")

                # First pass: semantic validation
                is_semantically_valid_pairwise_combination("${type1}" "${type2}" "${type3}" is_semantic_valid)
                if(is_semantic_valid)
                    math(EXPR semantic_passed "${semantic_passed} + 1")
                    list(APPEND valid_combinations "${type1},${type2},${type3}")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # Report filtering statistics
    message(STATUS "Combination filtering results:")
    message(STATUS "  Total checked: ${total_checked}")
    message(STATUS "  Semantic valid: ${semantic_passed}")

    if(total_checked GREATER 0)
        math(EXPR semantic_reduction "100 * (${total_checked} - ${semantic_passed}) / ${total_checked}")
        message(STATUS "  Semantic filtering reduced by: ${semantic_reduction}%")
    endif()

    set(${result_var} "${valid_combinations}" PARENT_SCOPE)
endfunction()

# Main orchestration function for enhanced semantic validation system
function(setup_enhanced_semantic_validation)
    message(STATUS "=== SETTING UP ENHANCED SEMANTIC VALIDATION SYSTEM ===")

    # Determine workload profile
    set(workload_profile "generic")
    if(DEFINED SD_WORKLOAD_PROFILE)
        set(workload_profile "${SD_WORKLOAD_PROFILE}")
    elseif(DEFINED SD_TYPE_PROFILE)
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    message(STATUS "Active workload profile: ${workload_profile}")

    # Generate optimized combinations
    generate_pairwise_combinations_enhanced("${workload_profile}" optimized_combinations)

    # Store results
    set(SD_OPTIMIZED_COMBINATIONS "${optimized_combinations}" PARENT_SCOPE)

    message(STATUS "Enhanced semantic validation setup completed")
    list(LENGTH optimized_combinations combo_count)
    message(STATUS "Generated ${combo_count} optimized type combinations")
endfunction()
