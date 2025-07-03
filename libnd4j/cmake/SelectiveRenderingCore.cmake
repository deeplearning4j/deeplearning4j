# ============================================================================
# SelectiveRenderingCore.cmake (v19 - With Semantic Filtering)
#
# Enhanced version that implements proper semantic filtering to reduce
# template instantiation combinations from 125 to ~30-40 meaningful ones.
# This file contains ALL logic for the selective rendering system.
# ============================================================================


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

# ============================================================================
# SECTION 1: SEMANTIC FILTERING LOGIC
# ============================================================================
function(_internal_srcore_is_type_numeric type_name output_var)
    set(numeric_types "DOUBLE;FLOAT32;INT32;INT64;FLOAT16;BFLOAT16")
    list(FIND numeric_types "${type_name}" found_index)
    if(found_index GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_is_type_floating type_name output_var)
    set(floating_types "DOUBLE;FLOAT32;FLOAT16;BFLOAT16")
    list(FIND floating_types "${type_name}" found_index)
    if(found_index GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_is_type_integer type_name output_var)
    set(integer_types "INT32;INT64;BOOL")
    list(FIND integer_types "${type_name}" found_index)
    if(found_index GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_get_type_priority type_name output_var)
    # Higher numbers = higher precision/priority
    if(type_name STREQUAL "DOUBLE")
        set(${output_var} 8 PARENT_SCOPE)
    elseif(type_name STREQUAL "FLOAT32")
        set(${output_var} 6 PARENT_SCOPE)
    elseif(type_name STREQUAL "FLOAT16")
        set(${output_var} 4 PARENT_SCOPE)
    elseif(type_name STREQUAL "BFLOAT16")
        set(${output_var} 4 PARENT_SCOPE)
    elseif(type_name STREQUAL "INT64")
        set(${output_var} 7 PARENT_SCOPE)
    elseif(type_name STREQUAL "INT32")
        set(${output_var} 5 PARENT_SCOPE)
    elseif(type_name STREQUAL "BOOL")
        set(${output_var} 1 PARENT_SCOPE)
    else()
        set(${output_var} 0 PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_is_valid_pair type1 type2 output_var)
    # Get type properties
    _internal_srcore_is_type_numeric("${type1}" is_numeric_1)
    _internal_srcore_is_type_numeric("${type2}" is_numeric_2)
    _internal_srcore_is_type_floating("${type1}" is_float_1)
    _internal_srcore_is_type_floating("${type2}" is_float_2)
    _internal_srcore_is_type_integer("${type1}" is_int_1)
    _internal_srcore_is_type_integer("${type2}" is_int_2)

    # Rule 1: Identical types are always valid
    if(type1 STREQUAL type2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Both must be numeric for mixed pairs
    if(NOT is_numeric_1 OR NOT is_numeric_2)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Bool can pair with any numeric type (for broadcasting)
    if(type1 STREQUAL "BOOL" OR type2 STREQUAL "BOOL")
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4: Floating types can pair with each other
    if(is_float_1 AND is_float_2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 5: Integer types can pair with each other
    if(is_int_1 AND is_int_2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 6: Integer-to-Float promotion patterns
    if((is_int_1 AND is_float_2) OR (is_float_1 AND is_int_2))
        # Allow common promotion patterns
        if((type1 STREQUAL "INT32" AND type2 STREQUAL "FLOAT32") OR
        (type1 STREQUAL "FLOAT32" AND type2 STREQUAL "INT32") OR
        (type1 STREQUAL "INT64" AND type2 STREQUAL "DOUBLE") OR
        (type1 STREQUAL "DOUBLE" AND type2 STREQUAL "INT64"))
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Default: invalid combination
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

function(_internal_srcore_is_valid_triple type1 type2 type3 output_var)
    # Get type properties
    _internal_srcore_is_type_numeric("${type1}" is_numeric_1)
    _internal_srcore_is_type_numeric("${type2}" is_numeric_2)
    _internal_srcore_is_type_numeric("${type3}" is_numeric_3)
    _internal_srcore_is_type_floating("${type1}" is_float_1)
    _internal_srcore_is_type_floating("${type2}" is_float_2)
    _internal_srcore_is_type_floating("${type3}" is_float_3)
    _internal_srcore_get_type_priority("${type1}" priority_1)
    _internal_srcore_get_type_priority("${type2}" priority_2)
    _internal_srcore_get_type_priority("${type3}" priority_3)

    # Rule 1: All three types must be numeric
    if(NOT is_numeric_1 OR NOT is_numeric_2 OR NOT is_numeric_3)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Identical types are always valid (T, T, T)
    if(type1 STREQUAL type2 AND type2 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Output type should be highest precision among inputs
    # For (A, B, C), C should be >= max(A, B) in precision
    # Rule 3: Output type should be highest precision among inputs
    # For (A, B, C), C should be >= max(A, B) in precision
    if(priority_1 GREATER priority_2)
        set(max_input_priority ${priority_1})
    else()
        set(max_input_priority ${priority_2})
    endif()

    if(priority_3 LESS max_input_priority)
        # Exception: Allow bool output for comparison operations
        if(NOT type3 STREQUAL "BOOL")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 4: Common operation patterns
    # Pattern A: (T, T, U) - Same input types, different output
    if(type1 STREQUAL type2)
        # Allow same-type inputs with promoted output
        if(priority_3 GREATER_EQUAL priority_1 OR type3 STREQUAL "BOOL")
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Pattern B: (T, U, T) - Mixed inputs, output matches first input
    if(type1 STREQUAL type3)
        _internal_srcore_is_valid_pair("${type1}" "${type2}" pair_valid)
        if(pair_valid)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Pattern C: (T, U, U) - Mixed inputs, output matches second input
    if(type2 STREQUAL type3)
        _internal_srcore_is_valid_pair("${type1}" "${type2}" pair_valid)
        if(pair_valid)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Pattern D: Broadcasting with bool
    if(type1 STREQUAL "BOOL" OR type2 STREQUAL "BOOL")
        # Bool can broadcast with any numeric type
        _internal_srcore_is_valid_pair("${type2}" "${type3}" pair_valid_23)
        _internal_srcore_is_valid_pair("${type1}" "${type3}" pair_valid_13)
        if(pair_valid_23 OR pair_valid_13)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Pattern E: Type promotion chains (int -> float -> double)
    if(is_float_3)
        # Allow promotion to floating point output
        if((type1 STREQUAL "INT32" OR type1 STREQUAL "INT64") AND
        (type2 STREQUAL "INT32" OR type2 STREQUAL "INT64" OR is_float_2))
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 5: Specific beneficial combinations for ML/scientific computing
    # Common ML patterns: mixed precision training
    if((type1 STREQUAL "FLOAT16" AND type2 STREQUAL "FLOAT32" AND type3 STREQUAL "FLOAT32") OR
    (type1 STREQUAL "BFLOAT16" AND type2 STREQUAL "FLOAT32" AND type3 STREQUAL "FLOAT32") OR
    (type1 STREQUAL "FLOAT32" AND type2 STREQUAL "FLOAT16" AND type3 STREQUAL "FLOAT32") OR
    (type1 STREQUAL "FLOAT32" AND type2 STREQUAL "BFLOAT16" AND type3 STREQUAL "FLOAT32"))
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Comparison operations: (T, T, bool)
    if(type3 STREQUAL "BOOL" AND type1 STREQUAL type2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Index operations: (T, INT32/INT64, T)
    if((type2 STREQUAL "INT32" OR type2 STREQUAL "INT64") AND type1 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Default: invalid combination
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# ============================================================================
# SECTION 2: ENHANCED COMBINATION GENERATION WITH FILTERING
# ============================================================================
function(_internal_srcore_generate_combinations active_indices type_names profile result_2_var result_3_var)
    list(LENGTH active_indices type_count)
    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types provided for combination generation")
    endif()
    set(combinations_2 "")
    set(combinations_3 "")
    set(filtered_count_2 0)
    set(filtered_count_3 0)
    set(total_possible_2 0)
    set(total_possible_3 0)

    math(EXPR max_index "${type_count} - 1")
    message(STATUS "🔍 Generating filtered combinations for ${type_count} types...")
    # Generate 2-type combinations with filtering
    foreach(i RANGE ${max_index})
        list(GET type_names ${i} type_name_i)
        foreach(j RANGE ${max_index})
            list(GET type_names ${j} type_name_j)
            math(EXPR total_possible_2 "${total_possible_2} + 1")

            _internal_srcore_is_valid_pair("${type_name_i}" "${type_name_j}" is_valid)
            if(is_valid)
                list(APPEND combinations_2 "${i},${j}")
                math(EXPR filtered_count_2 "${filtered_count_2} + 1")
            endif()
        endforeach()
    endforeach()
    # Generate 3-type combinations with filtering
    foreach(i RANGE ${max_index})
        list(GET type_names ${i} type_name_i)
        foreach(j RANGE ${max_index})
            list(GET type_names ${j} type_name_j)
            foreach(k RANGE ${max_index})
                list(GET type_names ${k} type_name_k)
                math(EXPR total_possible_3 "${total_possible_3} + 1")

                _internal_srcore_is_valid_triple("${type_name_i}" "${type_name_j}" "${type_name_k}" is_valid)
                if(is_valid)
                    list(APPEND combinations_3 "${i},${j},${k}")
                    math(EXPR filtered_count_3 "${filtered_count_3} + 1")
                endif()
            endforeach()
        endforeach()
    endforeach()
    # Calculate savings
    math(EXPR savings_2 "${total_possible_2} - ${filtered_count_2}")
    math(EXPR savings_3 "${total_possible_3} - ${filtered_count_3}")
    if(total_possible_2 GREATER 0)
        math(EXPR percent_saved_2 "100 * ${savings_2} / ${total_possible_2}")
    else()
        set(percent_saved_2 0)
    endif()
    if(total_possible_3 GREATER 0)
        math(EXPR percent_saved_3 "100 * ${savings_3} / ${total_possible_3}")
    else()
        set(percent_saved_3 0)
    endif()
    message(STATUS "✅ Semantic Filtering Results:")
    message(STATUS "   2-type: ${filtered_count_2}/${total_possible_2} combinations (${percent_saved_2}% filtered)")
    message(STATUS "   3-type: ${filtered_count_3}/${total_possible_3} combinations (${percent_saved_3}% filtered)")
    # Optional: Show sample filtered combinations for verification
    if(SRCORE_ENABLE_DIAGNOSTICS)
        message(STATUS "🔍 Sample valid 3-type combinations:")
        set(sample_count 0)
        foreach(combo ${combinations_3})
            if(sample_count GREATER_EQUAL 5)
                break()
            endif()
            string(REPLACE "," ";" combo_parts "${combo}")
            list(GET combo_parts 0 i)
            list(GET combo_parts 1 j)
            list(GET combo_parts 2 k)
            list(GET type_names ${i} name_i)
            list(GET type_names ${j} name_j)
            list(GET type_names ${k} name_k)
            message(STATUS "     (${name_i}, ${name_j}, ${name_k})")
            math(EXPR sample_count "${sample_count} + 1")
        endforeach()
    endif()
    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()

# ============================================================================
# SECTION 3: ORIGINAL HELPER FUNCTIONS (Updated)
# ============================================================================
function(_internal_srcore_debug_message message)
    if(DEFINED SRCORE_ENABLE_DIAGNOSTICS AND SRCORE_ENABLE_DIAGNOSTICS)
        message(STATUS "🔧 SelectiveRenderingCore: ${message}")
    endif()
endfunction()

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
function(is_semantically_valid_combination type1 type2 type3 mode result_var)
    # Simple validation - just return true for now
    set(${result_var} TRUE PARENT_SCOPE)
endfunction()

function(get_all_types result_var)
    # Based on DataType.h enum, get all available types
    set(all_types
            "bool"      # BOOL = 1
            "float8"    # FLOAT8 = 2
            "float16"   # HALF = 3
            "half2"     # HALF2 = 4
            "float32"   # FLOAT32 = 5
            "double"    # DOUBLE = 6
            "int8"      # INT8 = 7
            "int16"     # INT16 = 8
            "int32"     # INT32 = 9
            "int64"     # INT64 = 10
            "uint8"     # UINT8 = 11
            "uint16"    # UINT16 = 12
            "uint32"    # UINT32 = 13
            "uint64"    # UINT64 = 14
            "qint8"     # QINT8 = 15
            "qint16"    # QINT16 = 16
            "bfloat16"  # BFLOAT16 = 17
            "utf8"      # UTF8 = 50
            "utf16"     # UTF16 = 51
            "utf32"     # UTF32 = 52
    )

    set(${result_var} "${all_types}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_discover_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
    message(STATUS "🔍 SelectiveRenderingCore: Starting type discovery...")

    # Debug: Show what variables we received from TypeValidation
    message(STATUS "🔍 DEBUG: SRCORE_USE_SELECTIVE_TYPES = ${SRCORE_USE_SELECTIVE_TYPES}")
    message(STATUS "🔍 DEBUG: SRCORE_VALIDATED_TYPES = ${SRCORE_VALIDATED_TYPES}")

    # Check cache variables first (these are set by TypeValidation.cmake)
    if(NOT DEFINED SRCORE_USE_SELECTIVE_TYPES)
        # Fallback: try to read from cache
        get_property(cache_selective CACHE SRCORE_USE_SELECTIVE_TYPES PROPERTY VALUE)
        get_property(cache_types CACHE SRCORE_VALIDATED_TYPES PROPERTY VALUE)

        if(DEFINED cache_selective)
            set(SRCORE_USE_SELECTIVE_TYPES "${cache_selective}")
            set(SRCORE_VALIDATED_TYPES "${cache_types}")
            message(STATUS "🔍 Read from CACHE: USE_SELECTIVE=${SRCORE_USE_SELECTIVE_TYPES}, TYPES=${SRCORE_VALIDATED_TYPES}")
        else()
            message(STATUS "🔍 No exported variables found - defaulting to ALL types discovery")
            set(SRCORE_USE_SELECTIVE_TYPES FALSE)
            set(SRCORE_VALIDATED_TYPES "")
        endif()
    endif()

    # DECISION: Use the exported information to determine discovery mode
    if(SRCORE_USE_SELECTIVE_TYPES AND DEFINED SRCORE_VALIDATED_TYPES AND NOT SRCORE_VALIDATED_TYPES STREQUAL "")
        message(STATUS "🎯 Using SELECTIVE type discovery for types: ${SRCORE_VALIDATED_TYPES}")
        _internal_srcore_discover_selective_types("${SRCORE_VALIDATED_TYPES}" discovered_indices discovered_names discovered_enums discovered_cpp_types)
    else()
        message(STATUS "🎯 Using ALL types discovery (no selective types specified)")
        _internal_srcore_discover_all_types(discovered_indices discovered_names discovered_enums discovered_cpp_types)
    endif()

    # Set return variables
    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_names}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()


function(_internal_srcore_discover_selective_types validated_types_list result_indices_var result_names_var result_enums_var result_cpp_types_var)
    message(STATUS "🔍 SELECTIVE: Discovering validated types: ${validated_types_list}")

    # Find types.h
    set(possible_headers
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/system/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/types/types.h"
            "${CMAKE_SOURCE_DIR}/libnd4j/include/types/types.h"
            "${CMAKE_SOURCE_DIR}/include/types/types.h")

    set(types_header "")
    foreach(header_path ${possible_headers})
        if(EXISTS "${header_path}")
            set(types_header "${header_path}")
            message(STATUS "Found types.h at: ${header_path}")
            break()
        endif()
    endforeach()

    if(NOT types_header)
        message(FATAL_ERROR "❌ SelectiveRenderingCore: Could not find types.h in any expected location")
    endif()

    file(READ "${types_header}" types_content)

    # Enhanced mapping from user input types to types.h type keys
    set(type_mapping_float32 "FLOAT32")
    set(type_mapping_float "FLOAT32")
    set(type_mapping_double "DOUBLE")
    set(type_mapping_int32 "INT32")
    set(type_mapping_int64 "INT64")
    set(type_mapping_bool "BOOL")
    set(type_mapping_float16 "HALF")
    set(type_mapping_half "HALF")
    set(type_mapping_bfloat16 "BFLOAT16")
    set(type_mapping_bfloat "BFLOAT16")
    set(type_mapping_int8 "INT8")
    set(type_mapping_uint8 "UINT8")
    set(type_mapping_int16 "INT16")
    set(type_mapping_uint16 "UINT16")
    set(type_mapping_uint32 "UINT32")
    set(type_mapping_uint64 "UINT64")
    set(type_mapping_qint8 "QINT8")
    set(type_mapping_qint16 "QINT16")
    set(type_mapping_utf8 "UTF8")
    set(type_mapping_utf16 "UTF16")
    set(type_mapping_utf32 "UTF32")
    set(type_mapping_float8 "FLOAT8")
    set(type_mapping_half2 "HALF2")

    set(discovered_types "")
    set(discovered_indices "")
    set(discovered_enums "")
    set(discovered_cpp_types "")
    set(type_index 0)

    # Convert validated types list to type keys and discover them
    string(REPLACE ";" ";" validated_list "${validated_types_list}")
    foreach(user_type ${validated_list})
        # Normalize the user type
        string(STRIP "${user_type}" user_type)

        # Map to types.h key
        set(type_key "")
        if(DEFINED type_mapping_${user_type})
            set(type_key "${type_mapping_${user_type}}")
        else()
            # Try direct match with uppercase
            string(TOUPPER "${user_type}" upper_type)
            set(type_key "${upper_type}")
        endif()

        if(NOT type_key)
            message(WARNING "⚠️ Could not map user type '${user_type}' to types.h type key")
            continue()
        endif()

        # Search for this type in types.h
        string(REGEX MATCH "#define[ \t]+TTYPE_${type_key}[ \t]*,[ \t]*\\(([^)]+)\\)" type_match "${types_content}")
        if(type_match)
            list(APPEND discovered_types "${type_key}")
            list(APPEND discovered_indices ${type_index})

            string(REGEX MATCH "\\(([^)]+)\\)" tuple_match "${type_match}")
            string(SUBSTRING "${tuple_match}" 1 -1 type_tuple)
            string(REGEX REPLACE "^([^,]+),[ \t]*(.+)$" "\\1;\\2" tuple_parts "${type_tuple}")
            list(GET tuple_parts 0 enum_part)
            list(GET tuple_parts 1 cpp_part)
            string(STRIP "${enum_part}" enum_part)
            string(STRIP "${cpp_part}" cpp_part)
            string(REGEX REPLACE "\\)$" "" cpp_part "${cpp_part}")

            list(APPEND discovered_enums "${enum_part}")
            list(APPEND discovered_cpp_types "${cpp_part}")

            message(STATUS "✅ SELECTIVE Type ${type_index}: ${type_key} (from ${user_type}) -> enum: ${enum_part}, cpp: ${cpp_part}")
            math(EXPR type_index "${type_index} + 1")
        else()
            message(WARNING "⚠️ Could not find type definition for ${type_key} (from ${user_type}) in types.h")
        endif()
    endforeach()

    if(type_index EQUAL 0)
        message(FATAL_ERROR "❌ SELECTIVE: No valid types discovered from validated list: ${validated_types_list}")
    endif()

    message(STATUS "✅ SELECTIVE: Successfully discovered ${type_index} types")

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()


function(_internal_srcore_discover_all_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
    message(STATUS "🔍 ALL: Discovering all available types from types.h")

    # Find types.h in the expected locations
    set(possible_headers
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/system/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../include/types/types.h"
            "${CMAKE_SOURCE_DIR}/libnd4j/include/types/types.h"
            "${CMAKE_SOURCE_DIR}/include/types/types.h")

    set(types_header "")
    foreach(header_path ${possible_headers})
        if(EXISTS "${header_path}")
            set(types_header "${header_path}")
            message(STATUS "Found types.h at: ${header_path}")
            break()
        endif()
    endforeach()

    if(NOT types_header)
        message(FATAL_ERROR "❌ SelectiveRenderingCore: Could not find types.h in any expected location")
    endif()

    file(READ "${types_header}" types_content)

    # Complete list of all types from DataType.h enum (enhanced with all 20+ types)
    set(all_types
            "BOOL"      # DataType::BOOL = 1
            "FLOAT8"    # DataType::FLOAT8 = 2
            "HALF"      # DataType::HALF = 3 (float16)
            "HALF2"     # DataType::HALF2 = 4
            "FLOAT32"   # DataType::FLOAT32 = 5
            "DOUBLE"    # DataType::DOUBLE = 6
            "INT8"      # DataType::INT8 = 7
            "INT16"     # DataType::INT16 = 8
            "INT32"     # DataType::INT32 = 9
            "INT64"     # DataType::INT64 = 10
            "UINT8"     # DataType::UINT8 = 11
            "UINT16"    # DataType::UINT16 = 12
            "UINT32"    # DataType::UINT32 = 13
            "UINT64"    # DataType::UINT64 = 14
            "QINT8"     # DataType::QINT8 = 15
            "QINT16"    # DataType::QINT16 = 16
            "BFLOAT16"  # DataType::BFLOAT16 = 17
            "UTF8"      # DataType::UTF8 = 50
            "UTF16"     # DataType::UTF16 = 51
            "UTF32"     # DataType::UTF32 = 52
    )

    set(discovered_types "")
    set(discovered_indices "")
    set(discovered_enums "")
    set(discovered_cpp_types "")
    set(type_index 0)

    foreach(type_key ${all_types})
        # Look for the type definition in types.h
        string(REGEX MATCH "#define[ \t]+TTYPE_${type_key}[ \t]*,[ \t]*\\(([^)]+)\\)" type_match "${types_content}")
        if(type_match)
            list(APPEND discovered_types "${type_key}")
            list(APPEND discovered_indices ${type_index})

            # Parse the type tuple (DataType::ENUM, cpp_type)
            string(REGEX MATCH "\\(([^)]+)\\)" tuple_match "${type_match}")
            string(SUBSTRING "${tuple_match}" 1 -1 type_tuple)
            string(REGEX REPLACE "^([^,]+),[ \t]*(.+)$" "\\1;\\2" tuple_parts "${type_tuple}")
            list(GET tuple_parts 0 enum_part)
            list(GET tuple_parts 1 cpp_part)
            string(STRIP "${enum_part}" enum_part)
            string(STRIP "${cpp_part}" cpp_part)
            string(REGEX REPLACE "\\)$" "" cpp_part "${cpp_part}")

            list(APPEND discovered_enums "${enum_part}")
            list(APPEND discovered_cpp_types "${cpp_part}")

            message(STATUS "✅ ALL Type ${type_index}: ${type_key} -> enum: ${enum_part}, cpp: ${cpp_part}")
            math(EXPR type_index "${type_index} + 1")
        else()
            message(STATUS "⚠️ Type ${type_key} not found in types.h (may not be implemented)")
        endif()
    endforeach()

    if(type_index EQUAL 0)
        message(FATAL_ERROR "❌ No types discovered from types.h")
    endif()

    message(STATUS "✅ ALL: Successfully discovered ${type_index} types from types.h")

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()
# ============================================================================
# SECTION 4: ORIGINAL PUBLIC FUNCTIONS
# ============================================================================
function(srcore_discover_active_types result_var result_enums_var result_cpp_types_var)
    _internal_srcore_discover_types(active_indices active_names discovered_enums discovered_cpp_types)
    set(SRCORE_ACTIVE_TYPES "${active_names}" PARENT_SCOPE)
    list(LENGTH active_indices type_count)
    set(SRCORE_ACTIVE_TYPE_COUNT ${type_count} PARENT_SCOPE)

    set(${result_var} "${active_indices}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()

function(srcore_generate_combinations active_indices profile result_2_var result_3_var)
    _internal_srcore_generate_combinations("${active_indices}" "${SRCORE_ACTIVE_TYPES}" "${profile}" combinations_2 combinations_3)
    set(SRCORE_COMBINATIONS_2 "${combinations_2}" PARENT_SCOPE)
    set(SRCORE_COMBINATIONS_3 "${combinations_3}" PARENT_SCOPE)
    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()

# ============================================================================
# SECTION 5: REMAINING ORIGINAL FUNCTIONS
# ============================================================================

function(srcore_generate_headers active_indices combinations_2 combinations_3 output_dir type_enums type_cpp_types)
    _internal_srcore_generate_validity_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")
    _internal_generate_override_content("${active_indices}" "${combinations_2}" "${combinations_3}" FINAL_HEADER_CONTENT)
endfunction()

function(srcore_validate_output active_indices combinations_2 combinations_3)
    # Basic validation
    list(LENGTH active_indices type_count)
    list(LENGTH combinations_2 combo_2_count)
    list(LENGTH combinations_3 combo_3_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types found")
    endif()
    if(combo_2_count EQUAL 0)
        message(FATAL_ERROR "No 2-type combinations generated")
    endif()
    if(combo_3_count EQUAL 0)
        message(FATAL_ERROR "No 3-type combinations generated")
    endif()

    message(STATUS "Validation passed: ${type_count} types, ${combo_2_count} pairs, ${combo_3_count} triples")
endfunction()

function(srcore_emergency_fallback)
    # Fallback with minimal type set
    set(UNIFIED_ACTIVE_TYPES "float;double;int32_t;bool" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_2 "0,0;0,1;1,0;1,1;2,2;3,3" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_3 "0,0,0;1,1,1;2,2,2;3,3,3" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT 4 PARENT_SCOPE)
    message(WARNING "Using emergency fallback type configuration")
endfunction()

function(srcore_auto_setup)
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        message(STATUS "Auto-setup: Running selective rendering setup")
        setup_selective_rendering_unified_safe()
    endif()
endfunction()

function(_internal_srcore_generate_validity_header active_indices type_enums type_cpp_types combinations_2 combinations_3 output_dir)
    file(MAKE_DIRECTORY "${output_dir}/system")
    set(header_file "${output_dir}/system/selective_rendering.h")

    # Function to convert C++ types to simple macro names
    function(cpp_type_to_macro_name cpp_type output_var)
        set(macro_name "${cpp_type}")

        # Handle basic types
        if(macro_name STREQUAL "bool")
            set(macro_name "BOOL")
        elseif(macro_name STREQUAL "float")
            set(macro_name "FLOAT32")
        elseif(macro_name STREQUAL "double")
            set(macro_name "DOUBLE")
        elseif(macro_name STREQUAL "int32_t")
            set(macro_name "INT32")
        elseif(macro_name STREQUAL "sd::LongType")
            set(macro_name "INT64")
        elseif(macro_name STREQUAL "uint64_t")
            set(macro_name "UINT64")
        elseif(macro_name STREQUAL "int8_t")
            set(macro_name "int8_t")
        elseif(macro_name STREQUAL "int16_t")
            set(macro_name "int16_t")
        elseif(macro_name STREQUAL "uint8_t")
            set(macro_name "uint8_t")
        elseif(macro_name STREQUAL "uint16_t")
            set(macro_name "uint16_t")
        elseif(macro_name STREQUAL "uint32_t")
            set(macro_name "uint32_t")
        elseif(macro_name STREQUAL "float16")
            set(macro_name "HALF")
        elseif(macro_name STREQUAL "bfloat16")
            set(macro_name "BFLOAT16")

            # CRITICAL FIX: Handle std:: namespace types properly
        elseif(macro_name STREQUAL "std::string")
            set(macro_name "std_string")
        elseif(macro_name STREQUAL "std::u16string")
            set(macro_name "std_u16string")
        elseif(macro_name STREQUAL "std::u32string")
            set(macro_name "std_u32string")
        elseif(macro_name STREQUAL "std::wstring")
            set(macro_name "std_wstring")

            # Fallback: Handle any other std:: types that might appear
        elseif(macro_name MATCHES "^std::")
            # Replace :: with _ to make it preprocessor-safe
            string(REPLACE "::" "_" macro_name "${macro_name}")

            # Handle any other namespace types
        elseif(macro_name MATCHES "::")
            # Replace all :: with _ to make any namespace safe for preprocessor
            string(REPLACE "::" "_" macro_name "${macro_name}")
        endif()

        set(${output_var} "${macro_name}" PARENT_SCOPE)
    endfunction()

    # Function to convert enum values to integer constants
    function(enum_to_int_value enum_value output_var)
        string(REGEX REPLACE ".*::" "" datatype_name "${enum_value}")
        if(datatype_name STREQUAL "BOOL")
            set(int_value "1")
        elseif(datatype_name STREQUAL "FLOAT32")
            set(int_value "5")
        elseif(datatype_name STREQUAL "DOUBLE")
            set(int_value "6")
        elseif(datatype_name STREQUAL "INT32")
            set(int_value "9")
        elseif(datatype_name STREQUAL "INT64")
            set(int_value "10")
        elseif(datatype_name STREQUAL "UINT64")
            set(int_value "14")
        elseif(datatype_name STREQUAL "HALF")
            set(int_value "3")
        elseif(datatype_name STREQUAL "BFLOAT16")
            set(int_value "17")
        else()
            set(int_value "255")  # UNKNOWN
        endif()
        set(${output_var} "${int_value}" PARENT_SCOPE)
    endfunction()

    set(header_content "/* AUTOMATICALLY GENERATED by SelectiveRenderingCore.cmake */\n")
    string(APPEND header_content "#ifndef SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_TYPE_VALID_CHECK_AVAILABLE 1\n\n")

    # Define the COLON macro for namespace resolution
    string(APPEND header_content "// Macro to handle namespace resolution in preprocessor\n")
    string(APPEND header_content "#define COLON ::\n\n")

    # Generate individual type combination checks for triples
    string(APPEND header_content "// Define individual type combination checks that work with #if\n")
    foreach(combo IN LISTS combinations_3)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        list(GET parts 2 k)

        list(LENGTH type_cpp_types num_types)
        if(i LESS ${num_types} AND j LESS ${num_types} AND k LESS ${num_types})
            list(GET type_cpp_types ${i} cpp_i)
            list(GET type_cpp_types ${j} cpp_j)
            list(GET type_cpp_types ${k} cpp_k)
            cpp_type_to_macro_name("${cpp_i}" macro_i)
            cpp_type_to_macro_name("${cpp_j}" macro_j)
            cpp_type_to_macro_name("${cpp_k}" macro_k)
            string(APPEND header_content "#define SD_TRIPLE_${macro_i}_${macro_j}_${macro_k} 1\n")
        endif()
    endforeach()

    # Generate individual pair combination checks
    string(APPEND header_content "\n// Define individual pair combination checks that work with #if\n")
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)

        list(LENGTH type_cpp_types num_types)
        if(i LESS ${num_types} AND j LESS ${num_types})
            list(GET type_cpp_types ${i} cpp_i)
            list(GET type_cpp_types ${j} cpp_j)
            cpp_type_to_macro_name("${cpp_i}" macro_i)
            cpp_type_to_macro_name("${cpp_j}" macro_j)
            string(APPEND header_content "#define SD_PAIR_${macro_i}_${macro_j} 1\n")
        endif()
    endforeach()

    string(APPEND header_content "// UTF8 combinations are NOT defined, so they default to 0\n\n")

    # Add the compile-time checking helper macros
    string(APPEND header_content "// Helper macro to check if a specific triple is compiled\n")
    string(APPEND header_content "#define SD_IS_TRIPLE_COMPILED(T1, T2, T3) SD_TRIPLE_ ## T1 ## _ ## T2 ## _ ## T3\n\n")

    string(APPEND header_content "// Helper macro to check if a specific pair is compiled\n")
    string(APPEND header_content "#define SD_IS_PAIR_COMPILED(T1, T2) SD_PAIR_ ## T1 ## _ ## T2\n\n")

    # Generate the runtime checking macro for triples using integer comparisons
    string(APPEND header_content "// The main runtime checking macro for triples (for actual use)\n")
    string(APPEND header_content "#define SD_IS_TRIPLE_TYPE_COMPILED(TYPE1_ENUM, TYPE2_ENUM, TYPE3_ENUM) \\\n")
    string(APPEND header_content "    (")

    set(first_triple TRUE)
    list(LENGTH type_cpp_types num_types)
    foreach(combo IN LISTS combinations_3)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        list(GET parts 2 k)

        if(i LESS ${num_types} AND j LESS ${num_types} AND k LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            list(GET type_enums ${j} enum_j)
            list(GET type_enums ${k} enum_k)
            enum_to_int_value("${enum_i}" int_i)
            enum_to_int_value("${enum_j}" int_j)
            enum_to_int_value("${enum_k}" int_k)

            if(NOT first_triple)
                string(APPEND header_content " || \\\n     ")
            else()
                set(first_triple FALSE)
            endif()
            string(APPEND header_content "((TYPE1_ENUM) == ${int_i} && (TYPE2_ENUM) == ${int_j} && (TYPE3_ENUM) == ${int_k})")
        endif()
    endforeach()
    string(APPEND header_content ")\n\n")

    # Generate single type validity flags
    string(APPEND header_content "// Single Type Validity\n")
    foreach(i RANGE 0 ${num_types})
        if(i LESS ${num_types})
            list(GET type_cpp_types ${i} cpp_type)
            cpp_type_to_macro_name("${cpp_type}" macro_name)
            string(APPEND header_content "#define SD_TYPE_${macro_name}_VALID 1\n")
        endif()
    endforeach()

    # Generate pairwise type validity flags
    string(APPEND header_content "\n// Pairwise Type Validity\n")
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)

        if(i LESS ${num_types} AND j LESS ${num_types})
            list(GET type_cpp_types ${i} cpp_i)
            list(GET type_cpp_types ${j} cpp_j)
            cpp_type_to_macro_name("${cpp_i}" macro_i)
            cpp_type_to_macro_name("${cpp_j}" macro_j)
            string(APPEND header_content "#define SD_TYPE_PAIR_${macro_i}_${macro_j}_VALID 1\n")
        endif()
    endforeach()

    # Generate helper macros using integer values
    string(APPEND header_content "\n// Helper Macros for Dynamic Type Checking\n")

    # Single type macro
    string(APPEND header_content "#define SD_IS_SINGLE_TYPE_COMPILED(TYPE_ENUM) \\\n")
    string(APPEND header_content "    (")

    set(first_single TRUE)
    foreach(i RANGE 0 ${num_types})
        if(i LESS ${num_types})
            list(GET type_enums ${i} enum_value)
            enum_to_int_value("${enum_value}" int_value)

            if(NOT first_single)
                string(APPEND header_content " || \\\n     ")
            else()
                set(first_single FALSE)
            endif()
            string(APPEND header_content "((TYPE_ENUM) == ${int_value})")
        endif()
    endforeach()
    string(APPEND header_content ")\n\n")

    # Pair type macro
    string(APPEND header_content "#define SD_IS_PAIR_TYPE_COMPILED(TYPE1_ENUM, TYPE2_ENUM) \\\n")
    string(APPEND header_content "    (")

    set(first_pair TRUE)
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)

        if(i LESS ${num_types} AND j LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            list(GET type_enums ${j} enum_j)
            enum_to_int_value("${enum_i}" int_i)
            enum_to_int_value("${enum_j}" int_j)

            if(NOT first_pair)
                string(APPEND header_content " || \\\n     ")
            else()
                set(first_pair FALSE)
            endif()
            string(APPEND header_content "((TYPE1_ENUM) == ${int_i} && (TYPE2_ENUM) == ${int_j})")
        endif()
    endforeach()
    string(APPEND header_content ")\n\n")

    string(APPEND header_content "#endif // SD_SELECTIVE_RENDERING_H\n")
    file(WRITE "${header_file}" "${header_content}")

    list(LENGTH combinations_3 total_triple_combinations)
    list(LENGTH combinations_2 total_pair_combinations)
    message(STATUS "✅ Generated selective_rendering.h with ${num_types} types, ${total_pair_combinations} pair combinations, ${total_triple_combinations} triple combinations")
endfunction()

function(_internal_prepare_double_map active_indices combinations_2)
    foreach(index_x IN LISTS active_indices)
        set(SD_TEMP_COMBO_MAP_${index_x} "")
    endforeach()
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" combo_parts "${combo}")
        list(GET combo_parts 0 index_x)
        list(GET combo_parts 1 index_y)
        set(SD_TEMP_COMBO_MAP_${index_x} "${SD_TEMP_COMBO_MAP_${index_x}};${index_y}")
    endforeach()
endfunction()

function(_internal_generate_override_content active_indices combinations_2 combinations_3 output_var)
    _internal_srcore_debug_message("Generating override content with ${CMAKE_CURRENT_LIST_LENGTH} active indices")

    # Start building the header content
    set(header_content "")

    # Add necessary includes and setup
    string(APPEND header_content "// Core infrastructure includes\n")
    string(APPEND header_content "#include <system/op_boilerplate.h>\n")
    string(APPEND header_content "#include <array/DataTypeValidation.h>\n")
    string(APPEND header_content "#include <stdexcept>\n\n")

    # Ensure THROW_EXCEPTION is defined
    string(APPEND header_content "// Ensure THROW_EXCEPTION is available\n")
    string(APPEND header_content "#ifndef THROW_EXCEPTION\n")
    string(APPEND header_content "#define THROW_EXCEPTION(msg) throw std::runtime_error(msg)\n")
    string(APPEND header_content "#endif\n\n")

    # Add simple validity check macros - FIXED VERSION
    string(APPEND header_content "// Simple validity check macros\n")
    string(APPEND header_content "#ifndef SD_IF_VALID\n")
    string(APPEND header_content "#define SD_IF_VALID(condition, code) code\n")
    string(APPEND header_content "#endif\n\n")

    string(APPEND header_content "#ifndef SD_IS_SINGLE_TYPE_VALID\n")
    string(APPEND header_content "#define SD_IS_SINGLE_TYPE_VALID(type) 1\n")
    string(APPEND header_content "#endif\n\n")

    string(APPEND header_content "#ifndef SD_IS_TYPE_PAIR_VALID\n")
    string(APPEND header_content "#define SD_IS_TYPE_PAIR_VALID(type1, type2) 1\n")
    string(APPEND header_content "#endif\n\n")

    string(APPEND header_content "#ifndef SD_IS_TYPE_TRIPLE_VALID\n")
    string(APPEND header_content "#define SD_IS_TYPE_TRIPLE_VALID(type1, type2, type3) 1\n")
    string(APPEND header_content "#endif\n\n")

    # Generate BUILD_SINGLE_SELECTOR - FIXED VERSION
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// BUILD_SINGLE_SELECTOR - Fixed Version\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef BUILD_SINGLE_SELECTOR\n")
    string(APPEND header_content "#define BUILD_SINGLE_SELECTOR(XTYPE, NAME, SIGNATURE, ...) \\\\\n")
    string(APPEND header_content "    switch (XTYPE) { \\\\\n")

    list(LENGTH active_indices num_types)
    foreach(index IN LISTS active_indices)
        if(DEFINED SRCORE_TYPE_CPP_${index} AND DEFINED SRCORE_TYPE_ENUM_${index})
            string(APPEND header_content "        case ${SRCORE_TYPE_ENUM_${index}}: { NAME<${SRCORE_TYPE_CPP_${index}}> SIGNATURE; break; } \\\\\n")
        endif()
    endforeach()

    string(APPEND header_content "        default: { \\\\\n")
    string(APPEND header_content "            auto e = sd::DataTypeValidation::getDataTypeErrorMessage(XTYPE, #NAME \"_SINGLE\"); \\\\\n")
    string(APPEND header_content "            THROW_EXCEPTION(e.c_str()); \\\\\n")
    string(APPEND header_content "        } \\\\\n")
    string(APPEND header_content "    }\n\n")

    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// BUILD_DOUBLE_SELECTOR - Fixed Version\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef BUILD_DOUBLE_SELECTOR\n")
    string(APPEND header_content "#define BUILD_DOUBLE_SELECTOR(XTYPE, YTYPE, NAME, SIGNATURE, ...) \\\\\n")
    string(APPEND header_content "    switch (XTYPE) { \\\\\n")

    foreach(index_x IN LISTS active_indices)
        if(DEFINED SRCORE_TYPE_ENUM_${index_x})
            string(APPEND header_content "        case ${SRCORE_TYPE_ENUM_${index_x}}: { \\\\\n")
            string(APPEND header_content "            switch (YTYPE) { \\\\\n")

            # Generate valid Y-type cases for this X-type
            foreach(combo IN LISTS combinations_2)
                string(REPLACE "," ";" combo_parts "${combo}")
                list(GET combo_parts 0 combo_x)
                list(GET combo_parts 1 combo_y)

                # Only generate cases where X matches current index_x
                if(combo_x EQUAL index_x)
                    if(DEFINED SRCORE_TYPE_CPP_${combo_x} AND DEFINED SRCORE_TYPE_CPP_${combo_y} AND DEFINED SRCORE_TYPE_ENUM_${combo_y})
                        string(APPEND header_content "                case ${SRCORE_TYPE_ENUM_${combo_y}}: { NAME<${SRCORE_TYPE_CPP_${combo_x}}, ${SRCORE_TYPE_CPP_${combo_y}}> SIGNATURE; break; } \\\\\n")
                    endif()
                endif()
            endforeach()

            string(APPEND header_content "                default: { \\\\\n")
            string(APPEND header_content "                    auto e = sd::DataTypeValidation::getDataTypeErrorMessage(YTYPE, #NAME \"_DOUBLE_Y\"); \\\\\n")
            string(APPEND header_content "                    THROW_EXCEPTION(e.c_str()); \\\\\n")
            string(APPEND header_content "                } \\\\\n")
            string(APPEND header_content "            } \\\\\n")
            string(APPEND header_content "            break; \\\\\n")
            string(APPEND header_content "        } \\\\\n")
        endif()
    endforeach()

    string(APPEND header_content "        default: { \\\\\n")
    string(APPEND header_content "            auto e = sd::DataTypeValidation::getDataTypeErrorMessage(XTYPE, #NAME \"_DOUBLE_X\"); \\\\\n")
    string(APPEND header_content "            THROW_EXCEPTION(e.c_str()); \\\\\n")
    string(APPEND header_content "        } \\\\\n")
    string(APPEND header_content "    }\n\n")

    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// BUILD_TRIPLE_SELECTOR - Fixed Version\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef BUILD_TRIPLE_SELECTOR\n")
    string(APPEND header_content "#define BUILD_TRIPLE_SELECTOR(XTYPE, YTYPE, ZTYPE, NAME, SIGNATURE, ...) \\\\\n")
    string(APPEND header_content "    switch (XTYPE) { \\\\\n")

    foreach(index_x IN LISTS active_indices)
        if(DEFINED SRCORE_TYPE_ENUM_${index_x})
            string(APPEND header_content "        case ${SRCORE_TYPE_ENUM_${index_x}}: { \\\\\n")
            string(APPEND header_content "            switch (YTYPE) { \\\\\n")

            # Group 3-type combinations by X,Y pairs for this X
            set(handled_xy_pairs "")
            foreach(combo IN LISTS combinations_3)
                string(REPLACE "," ";" combo_parts "${combo}")
                list(GET combo_parts 0 combo_x)
                list(GET combo_parts 1 combo_y)
                list(GET combo_parts 2 combo_z)

                if(combo_x EQUAL index_x)
                    # Check if we've already handled this X,Y pair
                    set(xy_pair "${combo_x},${combo_y}")
                    list(FIND handled_xy_pairs "${xy_pair}" found_idx)
                    if(found_idx EQUAL -1)
                        list(APPEND handled_xy_pairs "${xy_pair}")

                        if(DEFINED SRCORE_TYPE_ENUM_${combo_y})
                            string(APPEND header_content "                case ${SRCORE_TYPE_ENUM_${combo_y}}: { \\\\\n")
                            string(APPEND header_content "                    switch (ZTYPE) { \\\\\n")

                            # Generate all Z cases for this X,Y pair
                            foreach(z_combo IN LISTS combinations_3)
                                string(REPLACE "," ";" z_parts "${z_combo}")
                                list(GET z_parts 0 z_x)
                                list(GET z_parts 1 z_y)
                                list(GET z_parts 2 z_z)

                                if(z_x EQUAL combo_x AND z_y EQUAL combo_y)
                                    if(DEFINED SRCORE_TYPE_CPP_${z_x} AND DEFINED SRCORE_TYPE_CPP_${z_y} AND DEFINED SRCORE_TYPE_CPP_${z_z} AND DEFINED SRCORE_TYPE_ENUM_${z_z})
                                        string(APPEND header_content "                        case ${SRCORE_TYPE_ENUM_${z_z}}: { NAME<${SRCORE_TYPE_CPP_${z_x}}, ${SRCORE_TYPE_CPP_${z_y}}, ${SRCORE_TYPE_CPP_${z_z}}> SIGNATURE; break; } \\\\\n")
                                    endif()
                                endif()
                            endforeach()

                            string(APPEND header_content "                        default: { \\\\\n")
                            string(APPEND header_content "                            auto e = sd::DataTypeValidation::getDataTypeErrorMessage(ZTYPE, #NAME \"_TRIPLE_Z\"); \\\\\n")
                            string(APPEND header_content "                            THROW_EXCEPTION(e.c_str()); \\\\\n")
                            string(APPEND header_content "                        } \\\\\n")
                            string(APPEND header_content "                    } \\\\\n")
                            string(APPEND header_content "                    break; \\\\\n")
                            string(APPEND header_content "                } \\\\\n")
                        endif()
                    endif()
                endif()
            endforeach()

            string(APPEND header_content "                default: { \\\\\n")
            string(APPEND header_content "                    auto e = sd::DataTypeValidation::getDataTypeErrorMessage(YTYPE, #NAME \"_TRIPLE_Y\"); \\\\\n")
            string(APPEND header_content "                    THROW_EXCEPTION(e.c_str()); \\\\\n")
            string(APPEND header_content "                } \\\\\n")
            string(APPEND header_content "            } \\\\\n")
            string(APPEND header_content "            break; \\\\\n")
            string(APPEND header_content "        } \\\\\n")
        endif()
    endforeach()

    string(APPEND header_content "        default: { \\\\\n")
    string(APPEND header_content "            auto e = sd::DataTypeValidation::getDataTypeErrorMessage(XTYPE, #NAME \"_TRIPLE_X\"); \\\\\n")
    string(APPEND header_content "            THROW_EXCEPTION(e.c_str()); \\\\\n")
    string(APPEND header_content "        } \\\\\n")
    string(APPEND header_content "    }\n\n")

    # Add selector macro overrides - FIXED VERSION
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// Selector Macro Overrides\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef _SELECTOR_SINGLE\n")
    string(APPEND header_content "#define _SELECTOR_SINGLE(A, B, C, D) \\\\\n")
    string(APPEND header_content "    case C: { A<D> B; break; }\n\n")

    string(APPEND header_content "#undef _SELECTOR_DOUBLE_2\n")
    string(APPEND header_content "#define _SELECTOR_DOUBLE_2(NAME, SIGNATURE, TYPE_A, ENUM, TYPE_B) \\\\\n")
    string(APPEND header_content "    case ENUM: { NAME<TYPE_A, TYPE_B> SIGNATURE; break; }\n\n")

    string(APPEND header_content "#undef _SELECTOR_TRIPLE_3\n")
    string(APPEND header_content "#define _SELECTOR_TRIPLE_3(NAME, SIGNATURE, TYPE_X, TYPE_Y, ENUM_Z, TYPE_Z) \\\\\n")
    string(APPEND header_content "    case ENUM_Z: { NAME<TYPE_X, TYPE_Y, TYPE_Z> SIGNATURE; break; }\n\n")

    set(${output_var} "${header_content}" PARENT_SCOPE)
endfunction()
# ==========================================================================================
# SECTION 6: THE SINGLE PUBLIC ORCHESTRATOR FUNCTION
# ==========================================================================================
function(SETUP_AND_GENERATE_ALL_RENDERING_FILES)
    message(STATUS "Executing unified selective rendering setup and generation...")
    cmake_parse_arguments(SRCORE "" "TYPE_PROFILE;OUTPUT_DIR" "" ${ARGN})

    # --- PART A: RUN CORE ANALYSIS ---
    _internal_srcore_discover_types(active_indices active_names)
    _internal_srcore_generate_combinations("${active_indices}" "${active_names}" "${SRCORE_TYPE_PROFILE}" combinations_2 combinations_3)

    list(LENGTH active_indices type_count)
    if(type_count EQUAL 0)
        message(FATAL_ERROR "Analysis failed: No active types were discovered.")
    endif()
    message(STATUS "Analysis complete. Active types found: ${type_count}")

    # --- PART B: GENERATE HEADERS ---
    set(output_dir "${SRCORE_OUTPUT_DIR}")
    if(NOT output_dir)
        set(output_dir "${CMAKE_BINARY_DIR}/include")
    endif()

    # Generate selective_rendering.h with the validity flags
    _internal_srcore_generate_validity_header("${active_indices}" "${combinations_2}" "${combinations_3}" "${output_dir}")

    # Generate type_boilerplate_overrides.h with the new BUILD_* macro definitions
    _internal_generate_override_content("${active_indices}" "${combinations_2}" "${combinations_3}" FINAL_HEADER_CONTENT)

    message(STATUS "✅ Successfully generated all selective rendering headers to ${GENERATED_DIR}")
endfunction()
# Add this entire function to your SelectiveRenderingCore.cmake file.

function(setup_selective_rendering_unified)
    set(options "")
    set(one_value_args TYPE_PROFILE OUTPUT_DIR)
    set(multi_value_args "")
    cmake_parse_arguments(SRCORE "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT SRCORE_TYPE_PROFILE)
        set(SRCORE_TYPE_PROFILE "${SD_TYPE_PROFILE}")
    endif()
    if(NOT SRCORE_OUTPUT_DIR)
        set(SRCORE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")
    endif()

    # Phase 1: Discover active types - NOW WITH ALL DATA
    srcore_discover_active_types(active_types_indices discovered_enums discovered_cpp_types)
    list(LENGTH active_types_indices type_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "❌ SelectiveRenderingCore: No active types discovered!")
    endif()

    # Phase 2: Generate combinations
    srcore_generate_combinations("${active_types_indices}" "${SRCORE_TYPE_PROFILE}" combinations_2 combinations_3)

    # Phase 3: Generate headers - NOW WITH TYPE DATA
    srcore_generate_headers("${active_types_indices}" "${combinations_2}" "${combinations_3}" "${SRCORE_OUTPUT_DIR}" "${discovered_enums}" "${discovered_cpp_types}")

    # Phase 4: Set output variables
    set(UNIFIED_COMBINATIONS_2 "${combinations_2}" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_3 "${combinations_3}" PARENT_SCOPE)
    set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT ${type_count} PARENT_SCOPE)

    set(UNIFIED_COMBINATIONS_2 "${combinations_2}" CACHE INTERNAL "Unified 2-type combinations")
    set(UNIFIED_COMBINATIONS_3 "${combinations_3}" CACHE INTERNAL "Unified 3-type combinations")
    set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" CACHE INTERNAL "Active types for build")
    set(UNIFIED_TYPE_COUNT ${type_count} CACHE INTERNAL "Unified active type count")
endfunction()

function(setup_selective_rendering_unified_safe)
    # Try normal setup first
    if(NOT CMAKE_CROSSCOMPILING AND NOT ANDROID)
        # Full setup for normal builds
        setup_selective_rendering_unified(${ARGN})

        # CRITICAL FIX: Propagate variables from nested function to parent scope
        if(DEFINED UNIFIED_COMBINATIONS_3)
            set(UNIFIED_COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_COMBINATIONS_2)
            set(UNIFIED_COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_ACTIVE_TYPES)
            set(UNIFIED_ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_TYPE_COUNT)
            set(UNIFIED_TYPE_COUNT "${UNIFIED_TYPE_COUNT}" PARENT_SCOPE)
        endif()

        srcore_map_to_legacy_variables()

        # Generate diagnostic report
        srcore_generate_diagnostic_report()
    else()
        # Simplified setup for cross-compilation/Android
        srcore_debug_message("Cross-compilation detected, using simplified setup")
        setup_selective_rendering_unified(${ARGN})

        # CRITICAL FIX: Propagate variables here too
        if(DEFINED UNIFIED_COMBINATIONS_3)
            set(UNIFIED_COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_COMBINATIONS_2)
            set(UNIFIED_COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_ACTIVE_TYPES)
            set(UNIFIED_ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_TYPE_COUNT)
            set(UNIFIED_TYPE_COUNT "${UNIFIED_TYPE_COUNT}" PARENT_SCOPE)
        endif()

        srcore_map_to_legacy_variables()
    endif()

    # Final verification that we have usable results
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        message(WARNING "⚠️ Unified setup failed, falling back to emergency mode")
        srcore_emergency_fallback()

        # CRITICAL FIX: Propagate emergency fallback variables to parent scope
        if(DEFINED UNIFIED_COMBINATIONS_3)
            set(UNIFIED_COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_COMBINATIONS_2)
            set(UNIFIED_COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_ACTIVE_TYPES)
            set(UNIFIED_ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_TYPE_COUNT)
            set(UNIFIED_TYPE_COUNT "${UNIFIED_TYPE_COUNT}" PARENT_SCOPE)
        endif()

        srcore_map_to_legacy_variables()
    endif()

    # FINAL VERIFICATION: Ensure variables are properly set before returning
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        message(FATAL_ERROR "❌ CRITICAL: Unable to establish UNIFIED_COMBINATIONS_3 even with emergency fallback!")
    endif()

    # Debug output for verification
    list(LENGTH UNIFIED_COMBINATIONS_3 final_combo_count)
    srcore_debug_message("✅ Final verification passed: ${final_combo_count} combinations ready")
endfunction()

function(srcore_debug_message message)
    if(SRCORE_ENABLE_DIAGNOSTICS)
        message(STATUS "🔧 SelectiveRenderingCore: ${message}")
    endif()
endfunction()

function(srcore_map_to_legacy_variables)
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" CACHE INTERNAL "Legacy 2-type combinations")
    endif()

    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" CACHE INTERNAL "Legacy 3-type combinations")
    endif()

    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" CACHE INTERNAL "Legacy active types")

        # Also set SD_COMMON_TYPES_COUNT for legacy compatibility
        list(LENGTH UNIFIED_ACTIVE_TYPES legacy_count)
        set(SD_COMMON_TYPES_COUNT ${legacy_count} PARENT_SCOPE)
        set(SD_COMMON_TYPES_COUNT ${legacy_count} CACHE INTERNAL "Legacy type count")

        # Set TYPE_NAME_X variables for legacy lookup
        set(type_index 0)
        foreach(type_name ${UNIFIED_ACTIVE_TYPES})
            set(TYPE_NAME_${type_index} "${type_name}" PARENT_SCOPE)
            set(TYPE_NAME_${type_index} "${type_name}" CACHE INTERNAL "Legacy reverse type lookup")
            math(EXPR type_index "${type_index} + 1")
        endforeach()
    endif()
endfunction()

function(srcore_generate_diagnostic_report)
    if(NOT SRCORE_ENABLE_DIAGNOSTICS)
        return()
    endif()

    set(report_file "${CMAKE_BINARY_DIR}/selective_rendering_diagnostic_report.txt")
    set(report_content "")

    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND report_content "SelectiveRenderingCore Diagnostic Report\n")
    string(APPEND report_content "Generated: ${current_time}\n")
    string(APPEND report_content "========================================\n\n")

    # System configuration
    string(APPEND report_content "Configuration:\n")
    string(APPEND report_content "- SD_ENABLE_SEMANTIC_FILTERING: ${SD_ENABLE_SEMANTIC_FILTERING}\n")
    string(APPEND report_content "- SD_TYPE_PROFILE: ${SD_TYPE_PROFILE}\n")
    string(APPEND report_content "- SD_SELECTIVE_TYPES: ${SD_SELECTIVE_TYPES}\n")
    string(APPEND report_content "- SRCORE_ENABLE_CACHING: ${SRCORE_ENABLE_CACHING}\n")
    string(APPEND report_content "\n")

    # Active types
    if(DEFINED SRCORE_ACTIVE_TYPES)
        list(LENGTH SRCORE_ACTIVE_TYPES type_count)
        string(APPEND report_content "Active Types (${type_count}):\n")
        set(index 0)
        foreach(type_name ${SRCORE_ACTIVE_TYPES})
            string(APPEND report_content "  [${index}] ${type_name}\n")
            math(EXPR index "${index} + 1")
        endforeach()
        string(APPEND report_content "\n")
    endif()

    # Combination statistics
    if(DEFINED SRCORE_COMBINATIONS_2 AND DEFINED SRCORE_COMBINATIONS_3)
        list(LENGTH SRCORE_COMBINATIONS_2 count_2)
        list(LENGTH SRCORE_COMBINATIONS_3 count_3)
        string(APPEND report_content "Combination Statistics:\n")
        string(APPEND report_content "- 2-type combinations: ${count_2}\n")
        string(APPEND report_content "- 3-type combinations: ${count_3}\n")

        if(DEFINED SRCORE_ACTIVE_TYPE_COUNT)
            math(EXPR total_possible "${SRCORE_ACTIVE_TYPE_COUNT} * ${SRCORE_ACTIVE_TYPE_COUNT} * ${SRCORE_ACTIVE_TYPE_COUNT}")
            if(total_possible GREATER 0)
                math(EXPR usage_percent "100 * ${count_3} / ${total_possible}")
                math(EXPR savings_percent "100 - ${usage_percent}")
                string(APPEND report_content "- Template usage: ${usage_percent}% (${savings_percent}% saved)\n")
            endif()
        endif()
        string(APPEND report_content "\n")
    endif()

    # Sample combinations
    if(DEFINED SRCORE_COMBINATIONS_3)
        string(APPEND report_content "Sample 3-type combinations (first 10):\n")
        set(sample_count 0)
        foreach(combo ${SRCORE_COMBINATIONS_3})
            if(sample_count GREATER_EQUAL 10)
                break()
            endif()

            string(REPLACE "," ";" combo_parts "${combo}")
            list(GET combo_parts 0 i)
            list(GET combo_parts 1 j)
            list(GET combo_parts 2 k)

            if(DEFINED SRCORE_TYPE_NAME_${i} AND DEFINED SRCORE_TYPE_NAME_${j} AND DEFINED SRCORE_TYPE_NAME_${k})
                string(APPEND report_content "  (${SRCORE_TYPE_NAME_${i}}, ${SRCORE_TYPE_NAME_${j}}, ${SRCORE_TYPE_NAME_${k}}) -> (${i},${j},${k})\n")
            else()
                string(APPEND report_content "  (${i},${j},${k}) -> [type names not available]\n")
            endif()

            math(EXPR sample_count "${sample_count} + 1")
        endforeach()
        string(APPEND report_content "\n")
    endif()

    # Write report
    file(WRITE "${report_file}" "${report_content}")
    srcore_debug_message("Diagnostic report written to: ${report_file}")
endfunction()