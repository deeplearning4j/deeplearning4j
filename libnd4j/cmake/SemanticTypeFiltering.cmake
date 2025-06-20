################################################################################
# Semantic Type Filtering Functions
# Enhanced semantic filtering for ML workload-specific type combinations
################################################################################

# Function to categorize a type for semantic validation
function(categorize_type type result_var)
    string(TOLOWER "${type}" type_lower)
    
    if(type_lower MATCHES "bool")
        set(${result_var} "BOOLEAN" PARENT_SCOPE)
    elseif(type_lower MATCHES "int8|uint8|int16|uint16|int32|uint32|int64|uint64|long")
        set(${result_var} "INTEGER" PARENT_SCOPE)
    elseif(type_lower MATCHES "float16|bfloat16|float32|float|double|float64|half|bfloat")
        set(${result_var} "FLOATING_POINT" PARENT_SCOPE)
    elseif(type_lower MATCHES "string|utf8|utf16|utf32")
        set(${result_var} "STRING" PARENT_SCOPE)
    else()
        set(${result_var} "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Function to get type promotion rank for semantic validation
function(get_type_promotion_rank type result_var)
    string(TOLOWER "${type}" type_lower)
    
    # Boolean types
    if(type_lower MATCHES "bool")
        set(${result_var} 0 PARENT_SCOPE)
    # 8-bit integers
    elseif(type_lower MATCHES "int8|uint8")
        set(${result_var} 1 PARENT_SCOPE)
    # 16-bit integers
    elseif(type_lower MATCHES "int16|uint16")
        set(${result_var} 2 PARENT_SCOPE)
    # 32-bit integers
    elseif(type_lower MATCHES "int32|int")
        set(${result_var} 3 PARENT_SCOPE)
    # 64-bit integers
    elseif(type_lower MATCHES "int64|long|uint64|unsignedlong")
        set(${result_var} 4 PARENT_SCOPE)
    # 16-bit floats
    elseif(type_lower MATCHES "float16|half|bfloat16|bfloat")
        set(${result_var} 5 PARENT_SCOPE)
    # 32-bit floats
    elseif(type_lower MATCHES "float32|float")
        set(${result_var} 6 PARENT_SCOPE)
    # 64-bit floats
    elseif(type_lower MATCHES "double|float64")
        set(${result_var} 7 PARENT_SCOPE)
    # String types
    elseif(type_lower MATCHES "string|utf8|utf16|utf32")
        set(${result_var} 8 PARENT_SCOPE)
    else()
        set(${result_var} -1 PARENT_SCOPE)  # Unknown type
    endif()
endfunction()

# Enhanced semantic validation for pairwise operations
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

    # Rule 2: Pairwise operations typically work with compatible numeric types
    if(cat1 STREQUAL cat2 AND cat2 STREQUAL cat3)
        # Same category operations are generally valid for pairwise
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Mixed category operations - ML-specific patterns for pairwise
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

    # Rule 4: Boolean operations with numeric types (masking operations)
    if(cat1 STREQUAL "BOOLEAN" OR cat2 STREQUAL "BOOLEAN")
        if((cat1 STREQUAL "BOOLEAN" AND (cat2 STREQUAL "FLOATING_POINT" OR cat2 STREQUAL "INTEGER")) OR
           (cat2 STREQUAL "BOOLEAN" AND (cat1 STREQUAL "FLOATING_POINT" OR cat1 STREQUAL "INTEGER")))
            # Output should follow non-boolean input type category for pairwise ops
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

    # Rule 5: Type promotion within same category (essential for pairwise)
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

    # Default: reject unlikely combinations for pairwise operations
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()

# Function to apply workload-specific filtering for pairwise operations
function(apply_workload_filter_pairwise type1 type2 type3 workload_profile result_var)
    if(workload_profile STREQUAL "quantization")
        # Prioritize quantization-friendly type combinations
        string(TOLOWER "${type1};${type2};${type3}" types_combined)
        if(types_combined MATCHES "int8|uint8" AND types_combined MATCHES "float32")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    elseif(workload_profile STREQUAL "training")
        # Prioritize mixed precision training combinations
        string(TOLOWER "${type1};${type2};${type3}" types_combined)
        if(types_combined MATCHES "float16|bfloat16" OR types_combined MATCHES "float32")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    elseif(workload_profile STREQUAL "inference")
        # Prioritize inference-optimized combinations
        string(TOLOWER "${type1};${type2};${type3}" types_combined)
        if(types_combined MATCHES "int8|uint8|float16|float32")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    elseif(workload_profile STREQUAL "nlp")
        # Prioritize NLP-friendly combinations
        string(TOLOWER "${type1};${type2};${type3}" types_combined)
        if(types_combined MATCHES "float32|double|int32|int64")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    elseif(workload_profile STREQUAL "cv")
        # Prioritize computer vision combinations
        string(TOLOWER "${type1};${type2};${type3}" types_combined)
        if(types_combined MATCHES "uint8|int8|float16|float32")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: accept all combinations for unknown/generic workloads
    set(${result_var} TRUE PARENT_SCOPE)
endfunction()

# Function to generate semantically filtered pairwise combinations
function(generate_pairwise_semantic_combinations workload_profile result_var)
    # Get all available types
    if(NOT DEFINED SD_ALL_TYPES_LIST)
        get_all_types(SD_ALL_TYPES_LIST)
    endif()

    set(valid_pairwise_combinations "")

    # Apply workload-specific filtering for pairwise operations
    if(workload_profile STREQUAL "quantization")
        # Focus on quantization-friendly types for pairwise ops
        set(pairwise_priority_types "int8_t;uint8_t;float;int32_t;int64_t")
        set(pairwise_secondary_types "double;int16_t;uint16_t")
    elseif(workload_profile STREQUAL "training")
        # Focus on mixed precision training types
        set(pairwise_priority_types "float16;bfloat16;float;double;int32_t;int64_t")
        set(pairwise_secondary_types "int8_t;uint8_t")
    elseif(workload_profile STREQUAL "inference")
        # Focus on inference-optimized types
        set(pairwise_priority_types "int8_t;uint8_t;float16;float;int32_t")
        set(pairwise_secondary_types "double;int64_t")
    elseif(workload_profile STREQUAL "nlp")
        # Focus on NLP-friendly types
        set(pairwise_priority_types "float;double;int32_t;int64_t")
        set(pairwise_secondary_types "float16;int8_t;uint8_t")
    elseif(workload_profile STREQUAL "cv")
        # Focus on computer vision types
        set(pairwise_priority_types "uint8_t;int8_t;float16;float;int32_t")
        set(pairwise_secondary_types "double;int64_t")
    else()
        # Default/Unknown profile: use essential types for pairwise operations
        message(STATUS "Using default pairwise types for profile: '${workload_profile}'")
        set(pairwise_priority_types "float;double;int32_t;int64_t;bool")
        set(pairwise_secondary_types "int8_t;uint8_t;float16")
    endif()

    message(STATUS "Pairwise priority types: ${pairwise_priority_types}")
    message(STATUS "Pairwise secondary types: ${pairwise_secondary_types}")

    set(${result_var} "${valid_pairwise_combinations}" PARENT_SCOPE)
endfunction()

# Function to partition pairwise combinations into chunks
function(partition_pairwise_combinations combinations chunk_size result_var)
    list(LENGTH combinations total_combinations)

    if(total_combinations EQUAL 0)
        set(${result_var} "" PARENT_SCOPE)
        return()
    endif()

    math(EXPR num_chunks "(${total_combinations} + ${chunk_size} - 1) / ${chunk_size}")
    set(partitioned_chunks "")

    set(chunk_index 0)
    while(chunk_index LESS num_chunks)
        math(EXPR start_index "${chunk_index} * ${chunk_size}")
        math(EXPR end_index_temp "${start_index} + ${chunk_size} - 1")

        if(end_index_temp GREATER_EQUAL total_combinations)
            math(EXPR end_index "${total_combinations} - 1")
        else()
            set(end_index ${end_index_temp})
        endif()

        set(chunk_combinations "")
        set(current_index ${start_index})
        while(current_index LESS_EQUAL end_index)
            list(GET combinations ${current_index} combination)
            list(APPEND chunk_combinations ${combination})
            math(EXPR current_index "${current_index} + 1")
        endwhile()

        list(APPEND partitioned_chunks "${chunk_combinations}")
        math(EXPR chunk_index "${chunk_index} + 1")
    endwhile()

    set(${result_var} "${partitioned_chunks}" PARENT_SCOPE)
endfunction()

# Function to validate ML type combinations
function(validate_ml_type_combinations types_list)
    message(STATUS "")
    print_status_colored("INFO" "=== ML TYPE COMBINATION VALIDATION ===")
    
    set(quantization_types 0)
    set(training_types 0)
    set(inference_types 0)
    
    foreach(type ${types_list})
        string(TOLOWER "${type}" type_lower)
        if(type_lower MATCHES "int8|uint8")
            math(EXPR quantization_types "${quantization_types} + 1")
        endif()
        if(type_lower MATCHES "float16|bfloat16")
            math(EXPR training_types "${training_types} + 1")
        endif()
        if(type_lower MATCHES "float32|int32")
            math(EXPR inference_types "${inference_types} + 1")
        endif()
    endforeach()
    
    message(STATUS "Quantization-friendly types: ${quantization_types}")
    message(STATUS "Training-friendly types: ${training_types}")
    message(STATUS "Inference-friendly types: ${inference_types}")
    
    if(quantization_types GREATER 0 AND training_types GREATER 0)
        message(STATUS "✅ Mixed workload support detected")
    elseif(quantization_types GREATER 0)
        message(STATUS "🔢 Quantization workload optimization")
    elseif(training_types GREATER 0)
        message(STATUS "🎯 Training workload optimization")
    else()
        message(STATUS "📊 General purpose configuration")
    endif()
endfunction()

# Function to analyze combination patterns
function(analyze_combination_patterns combinations workload_profile)
    if(NOT combinations OR combinations STREQUAL "")
        return()
    endif()
    
    list(LENGTH combinations total_combinations)
    message(STATUS "")
    print_status_colored("INFO" "=== COMBINATION PATTERN ANALYSIS ===")
    message(STATUS "Workload Profile: ${workload_profile}")
    message(STATUS "Total Combinations: ${total_combinations}")
    
    # Analyze type distribution in combinations
    set(int_combinations 0)
    set(float_combinations 0)
    set(mixed_combinations 0)
    
    foreach(combination ${combinations})
        string(REPLACE "," ";" combo_parts "${combination}")
        set(has_int FALSE)
        set(has_float FALSE)
        
        foreach(part ${combo_parts})
            string(TOLOWER "${part}" part_lower)
            if(part_lower MATCHES "int|bool")
                set(has_int TRUE)
            elseif(part_lower MATCHES "float|double|half|bfloat")
                set(has_float TRUE)
            endif()
        endforeach()
        
        if(has_int AND has_float)
            math(EXPR mixed_combinations "${mixed_combinations} + 1")
        elseif(has_int)
            math(EXPR int_combinations "${int_combinations} + 1")
        elseif(has_float)
            math(EXPR float_combinations "${float_combinations} + 1")
        endif()
    endforeach()
    
    message(STATUS "Integer-only combinations: ${int_combinations}")
    message(STATUS "Float-only combinations: ${float_combinations}")
    message(STATUS "Mixed-type combinations: ${mixed_combinations}")
    
    if(mixed_combinations GREATER 0)
        math(EXPR mixed_percent "100 * ${mixed_combinations} / ${total_combinations}")
        message(STATUS "Mixed-type percentage: ${mixed_percent}%")
    endif()
endfunction()

# Function to get operation type from template file
function(get_operation_type_from_template template_file result_var)
    get_filename_component(template_name "${template_file}" NAME)
    
    if(template_name MATCHES "reduce")
        set(${result_var} "REDUCE" PARENT_SCOPE)
    elseif(template_name MATCHES "pairwise")
        set(${result_var} "PAIRWISE" PARENT_SCOPE)
    elseif(template_name MATCHES "transform")
        set(${result_var} "TRANSFORM" PARENT_SCOPE)
    elseif(template_name MATCHES "broadcast")
        set(${result_var} "BROADCAST" PARENT_SCOPE)
    elseif(template_name MATCHES "indexreduce")
        set(${result_var} "INDEX_REDUCE" PARENT_SCOPE)
    else()
        set(${result_var} "GENERIC" PARENT_SCOPE)
    endif()
endfunction()

# Function to generate enhanced pairwise combinations
function(generate_pairwise_combinations_enhanced workload_profile result_var)
    # Define core types for different workloads
    if(workload_profile STREQUAL "quantization")
        set(priority_types "int8;uint8;float32;int32")
        set(secondary_types "int16;uint16;double")
    elseif(workload_profile STREQUAL "training")
        set(priority_types "float16;bfloat16;float32;double")
        set(secondary_types "int32;int64")
    elseif(workload_profile STREQUAL "inference")
        set(priority_types "int8;uint8;float16;float32")
        set(secondary_types "int32;double")
    elseif(workload_profile STREQUAL "nlp")
        set(priority_types "float32;double;int32;int64")
        set(secondary_types "float16;int8")
    elseif(workload_profile STREQUAL "cv")
        set(priority_types "uint8;int8;float16;float32")
        set(secondary_types "int32;double")
    else()
        # Generic profile
        set(priority_types "float32;double;int32;int64")
        set(secondary_types "int8;uint8;float16")
    endif()
    
    # Generate 3-type combinations prioritizing workload-specific types
    set(enhanced_combinations "")
    string(REPLACE ";" "," priority_list "${priority_types}")
    string(REPLACE ";" "," secondary_list "${secondary_types}")
    
    # Add priority combinations first
    foreach(type1 ${priority_types})
        foreach(type2 ${priority_types})
            foreach(type3 ${priority_types})
                list(APPEND enhanced_combinations "${type1},${type2},${type3}")
            endforeach()
        endforeach()
    endforeach()
    
    # Add selective secondary combinations
    list(LENGTH priority_types priority_count)
    if(priority_count LESS 6)  # Only add secondary if we have room
        foreach(type1 ${priority_types})
            foreach(type2 ${secondary_types})
                list(GET priority_types 0 primary_output)
                list(APPEND enhanced_combinations "${type1},${type2},${primary_output}")
            endforeach()
        endforeach()
    endif()
    
    # Remove duplicates and return
    list(REMOVE_DUPLICATES enhanced_combinations)
    set(${result_var} "${enhanced_combinations}" PARENT_SCOPE)
endfunction()

# Function to setup enhanced semantic validation
function(setup_enhanced_semantic_validation)
    message(STATUS "Setting up enhanced semantic validation system...")
    
    # Initialize semantic filtering flags
    set(SD_SEMANTIC_VALIDATION_ACTIVE TRUE CACHE BOOL "Semantic validation is active")
    
    # Set up workload-specific type priorities
    set(SD_WORKLOAD_TYPE_PRIORITIES_quantization "int8;uint8;float32;int32" CACHE STRING "Quantization workload type priorities")
    set(SD_WORKLOAD_TYPE_PRIORITIES_training "float16;bfloat16;float32;double" CACHE STRING "Training workload type priorities")
    set(SD_WORKLOAD_TYPE_PRIORITIES_inference "int8;uint8;float16;float32" CACHE STRING "Inference workload type priorities")
    set(SD_WORKLOAD_TYPE_PRIORITIES_nlp "float32;double;int32;int64" CACHE STRING "NLP workload type priorities")
    set(SD_WORKLOAD_TYPE_PRIORITIES_cv "uint8;int8;float16;float32" CACHE STRING "CV workload type priorities")
    
    message(STATUS "Enhanced semantic validation system initialized")
endfunction()
