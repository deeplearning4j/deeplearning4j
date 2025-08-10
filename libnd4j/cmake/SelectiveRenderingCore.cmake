# ============================================================================
# SelectiveRenderingCore.cmake (v20 - Production Optimized) - FIXED
#
# Optimized version with debug profiles removed for production builds.
# Conditional diagnostics only when explicitly enabled via SD_ENABLE_DIAGNOSTICS.
# ============================================================================

# Export type validation results for use by SelectiveRenderingCore
function(export_validated_types_for_selective_rendering)
    if(SD_TYPES_LIST_COUNT GREATER 0)
        set(SRCORE_USE_SELECTIVE_TYPES TRUE PARENT_SCOPE)
        set(SRCORE_VALIDATED_TYPES "${SD_TYPES_LIST}" PARENT_SCOPE)
        if(SD_ENABLE_DIAGNOSTICS)
            message(STATUS "Exporting SELECTIVE types for SelectiveRenderingCore: ${SD_TYPES_LIST}")
        endif()
    else()
        set(SRCORE_USE_SELECTIVE_TYPES FALSE PARENT_SCOPE)
        set(SRCORE_VALIDATED_TYPES "" PARENT_SCOPE)
        if(SD_ENABLE_DIAGNOSTICS)
            message(STATUS "Exporting ALL_TYPES mode for SelectiveRenderingCore")
        endif()
    endif()
endfunction()

# ============================================================================
# SECTION 1: SEMANTIC FILTERING LOGIC (Optimized)
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
    _internal_srcore_is_type_numeric("${type1}" is_numeric_1)
    _internal_srcore_is_type_numeric("${type2}" is_numeric_2)
    _internal_srcore_is_type_floating("${type1}" is_float_1)
    _internal_srcore_is_type_floating("${type2}" is_float_2)
    _internal_srcore_is_type_integer("${type1}" is_int_1)
    _internal_srcore_is_type_integer("${type2}" is_int_2)

    # Identical types are always valid
    if(type1 STREQUAL type2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Both must be numeric for mixed pairs
    if(NOT is_numeric_1 OR NOT is_numeric_2)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Bool can pair with any numeric type
    if(type1 STREQUAL "BOOL" OR type2 STREQUAL "BOOL")
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Floating types can pair with each other
    if(is_float_1 AND is_float_2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Integer types can pair with each other
    if(is_int_1 AND is_int_2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Integer-to-Float promotion patterns
    if((is_int_1 AND is_float_2) OR (is_float_1 AND is_int_2))
        if((type1 STREQUAL "INT32" AND type2 STREQUAL "FLOAT32") OR
        (type1 STREQUAL "FLOAT32" AND type2 STREQUAL "INT32") OR
        (type1 STREQUAL "INT64" AND type2 STREQUAL "DOUBLE") OR
        (type1 STREQUAL "DOUBLE" AND type2 STREQUAL "INT64"))
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

function(_internal_srcore_is_valid_triple type1 type2 type3 output_var)
    _internal_srcore_is_type_numeric("${type1}" is_numeric_1)
    _internal_srcore_is_type_numeric("${type2}" is_numeric_2)
    _internal_srcore_is_type_numeric("${type3}" is_numeric_3)
    _internal_srcore_is_type_floating("${type1}" is_float_1)
    _internal_srcore_is_type_floating("${type2}" is_float_2)
    _internal_srcore_is_type_floating("${type3}" is_float_3)
    _internal_srcore_get_type_priority("${type1}" priority_1)
    _internal_srcore_get_type_priority("${type2}" priority_2)
    _internal_srcore_get_type_priority("${type3}" priority_3)

    # All three types must be numeric
    if(NOT is_numeric_1 OR NOT is_numeric_2 OR NOT is_numeric_3)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Identical types are always valid
    if(type1 STREQUAL type2 AND type2 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Output type should be highest precision among inputs
    if(priority_1 GREATER priority_2)
        set(max_input_priority ${priority_1})
    else()
        set(max_input_priority ${priority_2})
    endif()

    if(priority_3 LESS max_input_priority)
        if(NOT type3 STREQUAL "BOOL")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Common operation patterns
    if(type1 STREQUAL type2)
        if(priority_3 GREATER_EQUAL priority_1 OR type3 STREQUAL "BOOL")
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(type1 STREQUAL type3)
        _internal_srcore_is_valid_pair("${type1}" "${type2}" pair_valid)
        if(pair_valid)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(type2 STREQUAL type3)
        _internal_srcore_is_valid_pair("${type1}" "${type2}" pair_valid)
        if(pair_valid)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Broadcasting with bool
    if(type1 STREQUAL "BOOL" OR type2 STREQUAL "BOOL")
        _internal_srcore_is_valid_pair("${type2}" "${type3}" pair_valid_23)
        _internal_srcore_is_valid_pair("${type1}" "${type3}" pair_valid_13)
        if(pair_valid_23 OR pair_valid_13)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Type promotion chains
    if(is_float_3)
        if((type1 STREQUAL "INT32" OR type1 STREQUAL "INT64") AND
        (type2 STREQUAL "INT32" OR type2 STREQUAL "INT64" OR is_float_2))
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    # ML-specific patterns
    if((type1 STREQUAL "FLOAT16" AND type2 STREQUAL "FLOAT32" AND type3 STREQUAL "FLOAT32") OR
    (type1 STREQUAL "BFLOAT16" AND type2 STREQUAL "FLOAT32" AND type3 STREQUAL "FLOAT32") OR
    (type1 STREQUAL "FLOAT32" AND type2 STREQUAL "FLOAT16" AND type3 STREQUAL "FLOAT32") OR
    (type1 STREQUAL "FLOAT32" AND type2 STREQUAL "BFLOAT16" AND type3 STREQUAL "FLOAT32"))
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Comparison operations
    if(type3 STREQUAL "BOOL" AND type1 STREQUAL type2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Index operations
    if((type2 STREQUAL "INT32" OR type2 STREQUAL "INT64") AND type1 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# ============================================================================
# SECTION 2: OPTIMIZED COMBINATION GENERATION
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

    # Statistics only if diagnostics enabled
    if(SD_ENABLE_DIAGNOSTICS)
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
        message(STATUS "Semantic Filtering Results:")
        message(STATUS "   2-type: ${filtered_count_2}/${total_possible_2} combinations (${percent_saved_2}% filtered)")
        message(STATUS "   3-type: ${filtered_count_3}/${total_possible_3} combinations (${percent_saved_3}% filtered)")
    endif()

    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()

# ============================================================================
# SECTION 3: CORE HELPER FUNCTIONS (Optimized)
# ============================================================================
function(srcore_normalize_type input_type output_var)
    set(normalized_type "${input_type}")

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

function(is_semantically_valid_combination type1 type2 type3 mode result_var)
    set(${result_var} TRUE PARENT_SCOPE)
endfunction()

function(get_all_types result_var)
    set(all_types
            "bool" "float8" "float16" "half2" "float32" "double"
            "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64"
            "qint8" "qint16" "bfloat16" "utf8" "utf16" "utf32"
    )
    set(${result_var} "${all_types}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_discover_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
    if(NOT DEFINED SRCORE_USE_SELECTIVE_TYPES)
        get_property(cache_selective CACHE SRCORE_USE_SELECTIVE_TYPES PROPERTY VALUE)
        get_property(cache_types CACHE SRCORE_VALIDATED_TYPES PROPERTY VALUE)

        if(DEFINED cache_selective)
            set(SRCORE_USE_SELECTIVE_TYPES "${cache_selective}")
            set(SRCORE_VALIDATED_TYPES "${cache_types}")
        else()
            set(SRCORE_USE_SELECTIVE_TYPES FALSE)
            set(SRCORE_VALIDATED_TYPES "")
        endif()
    endif()

    if(SRCORE_USE_SELECTIVE_TYPES AND DEFINED SRCORE_VALIDATED_TYPES AND NOT SRCORE_VALIDATED_TYPES STREQUAL "")
        _internal_srcore_discover_selective_types("${SRCORE_VALIDATED_TYPES}" discovered_indices discovered_names discovered_enums discovered_cpp_types)
    else()
        _internal_srcore_discover_all_types(discovered_indices discovered_names discovered_enums discovered_cpp_types)
    endif()

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_names}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_discover_selective_types validated_types_list result_indices_var result_names_var result_enums_var result_cpp_types_var)
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
            break()
        endif()
    endforeach()

    if(NOT types_header)
        message(FATAL_ERROR "Could not find types.h in any expected location")
    endif()

    file(READ "${types_header}" types_content)

    # Type mapping
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

    string(REPLACE ";" ";" validated_list "${validated_types_list}")
    foreach(user_type ${validated_list})
        string(STRIP "${user_type}" user_type)

        set(type_key "")
        if(DEFINED type_mapping_${user_type})
            set(type_key "${type_mapping_${user_type}}")
        else()
            string(TOUPPER "${user_type}" upper_type)
            set(type_key "${upper_type}")
        endif()

        if(NOT type_key)
            continue()
        endif()

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

            math(EXPR type_index "${type_index} + 1")
        endif()
    endforeach()

    if(type_index EQUAL 0)
        message(FATAL_ERROR "No valid types discovered from validated list: ${validated_types_list}")
    endif()

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_discover_all_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
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
            break()
        endif()
    endforeach()

    if(NOT types_header)
        message(FATAL_ERROR "Could not find types.h in any expected location")
    endif()

    file(READ "${types_header}" types_content)

    set(all_types
            "BOOL" "FLOAT8" "HALF" "HALF2" "FLOAT32" "DOUBLE"
            "INT8" "INT16" "INT32" "INT64" "UINT8" "UINT16" "UINT32" "UINT64"
            "QINT8" "QINT16" "BFLOAT16" "UTF8" "UTF16" "UTF32"
    )

    set(discovered_types "")
    set(discovered_indices "")
    set(discovered_enums "")
    set(discovered_cpp_types "")
    set(type_index 0)

    foreach(type_key ${all_types})
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

            math(EXPR type_index "${type_index} + 1")
        endif()
    endforeach()

    if(type_index EQUAL 0)
        message(FATAL_ERROR "No types discovered from types.h")
    endif()

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()

# ============================================================================
# SECTION 4: PUBLIC API FUNCTIONS
# ============================================================================
function(srcore_discover_active_types result_var result_enums_var result_cpp_types_var)
    _internal_srcore_discover_types(active_indices active_names discovered_enums discovered_cpp_types)
    set(SRCORE_ACTIVE_TYPES "${active_names}" PARENT_SCOPE)
    list(LENGTH active_indices type_count)
    set(SRCORE_ACTIVE_TYPE_COUNT ${type_count} PARENT_SCOPE)

    # Store type mappings for later use
    set(type_index 0)
    foreach(type_enum IN LISTS discovered_enums)
        set(SRCORE_TYPE_ENUM_${type_index} "${type_enum}" PARENT_SCOPE)
        math(EXPR type_index "${type_index} + 1")
    endforeach()

    set(type_index 0)
    foreach(type_cpp IN LISTS discovered_cpp_types)
        set(SRCORE_TYPE_CPP_${type_index} "${type_cpp}" PARENT_SCOPE)
        math(EXPR type_index "${type_index} + 1")
    endforeach()

    set(type_index 0)
    foreach(type_name IN LISTS active_names)
        set(SRCORE_TYPE_NAME_${type_index} "${type_name}" PARENT_SCOPE)
        math(EXPR type_index "${type_index} + 1")
    endforeach()

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

function(srcore_generate_headers active_indices combinations_2 combinations_3 output_dir type_enums type_cpp_types)
    # Generate the base validity header
    _internal_srcore_generate_validity_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")



    if(SD_ENABLE_DIAGNOSTICS)
        message(STATUS "Generated BUILD_ macro overrides: ${override_header_file}")
    endif()

    # Also enhance the main selective_rendering.h with runtime dispatch
    srcore_generate_enhanced_header("${active_indices}" "${combinations_2}" "${combinations_3}" "${output_dir}" "${type_enums}" "${type_cpp_types}")
endfunction()

function(srcore_validate_output active_indices combinations_2 combinations_3)
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
endfunction()

function(srcore_emergency_fallback)
    set(UNIFIED_ACTIVE_TYPES "float;double;int32_t;bool" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_2 "0,0;0,1;1,0;1,1;2,2;3,3" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_3 "0,0,0;1,1,1;2,2,2;3,3,3" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT 4 PARENT_SCOPE)
    if(SD_ENABLE_DIAGNOSTICS)
        message(WARNING "Using emergency fallback type configuration")
    endif()
endfunction()

function(srcore_auto_setup)
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        setup_selective_rendering_unified_safe()
    endif()
endfunction()

function(_internal_srcore_generate_validity_header active_indices type_enums type_cpp_types combinations_2 combinations_3 output_dir)
    file(MAKE_DIRECTORY "${output_dir}/system")
    set(header_file "${output_dir}/system/selective_rendering.h")

    # Helper function to convert enum value to integer
    function(enum_to_int_value enum_value output_var)
        string(REGEX REPLACE ".*::" "" datatype_name "${enum_value}")
        if(datatype_name STREQUAL "BOOL")
            set(int_value "1")
        elseif(datatype_name STREQUAL "FLOAT8")
            set(int_value "2")
        elseif(datatype_name STREQUAL "HALF")
            set(int_value "3")
        elseif(datatype_name STREQUAL "HALF2")
            set(int_value "4")
        elseif(datatype_name STREQUAL "FLOAT32")
            set(int_value "5")
        elseif(datatype_name STREQUAL "DOUBLE")
            set(int_value "6")
        elseif(datatype_name STREQUAL "INT8")
            set(int_value "7")
        elseif(datatype_name STREQUAL "INT16")
            set(int_value "8")
        elseif(datatype_name STREQUAL "INT32")
            set(int_value "9")
        elseif(datatype_name STREQUAL "INT64")
            set(int_value "10")
        elseif(datatype_name STREQUAL "UINT8")
            set(int_value "11")
        elseif(datatype_name STREQUAL "UINT16")
            set(int_value "12")
        elseif(datatype_name STREQUAL "UINT32")
            set(int_value "13")
        elseif(datatype_name STREQUAL "UINT64")
            set(int_value "14")
        elseif(datatype_name STREQUAL "QINT8")
            set(int_value "15")
        elseif(datatype_name STREQUAL "QINT16")
            set(int_value "16")
        elseif(datatype_name STREQUAL "BFLOAT16")
            set(int_value "17")
        elseif(datatype_name STREQUAL "UTF8")
            set(int_value "50")
        elseif(datatype_name STREQUAL "UTF16")
            set(int_value "51")
        elseif(datatype_name STREQUAL "UTF32")
            set(int_value "52")
        else()
            set(int_value "255")  # UNKNOWN
        endif()
        set(${output_var} "${int_value}" PARENT_SCOPE)
    endfunction()

    # Start building the header content
    set(header_content "/* AUTOMATICALLY GENERATED - Selective Rendering Header */\n")
    string(APPEND header_content "/* Generated by SelectiveRenderingCore.cmake */\n")
    string(APPEND header_content "#ifndef SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_SELECTIVE_RENDERING_H\n\n")

    # ============================================================================
    # SECTION 1: RAW COMPILATION FLAGS
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 1: RAW COMPILATION FLAGS\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Collect all compiled type numbers
    list(LENGTH type_enums num_types)
    set(compiled_type_numbers "")

    foreach(i RANGE 0 ${num_types})
        if(i LESS ${num_types})
            list(GET type_enums ${i} enum_value)
            enum_to_int_value("${enum_value}" int_value)
            list(FIND compiled_type_numbers "${int_value}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_value}")
            endif()
        endif()
    endforeach()

    # Generate single type compilation flags for ALL possible types
    string(APPEND header_content "// Single type compilation flags\n")
    set(all_possible_types "1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;50;51;52")
    foreach(type_num IN LISTS all_possible_types)
        list(FIND compiled_type_numbers "${type_num}" found_idx)
        if(found_idx GREATER_EQUAL 0)
            string(APPEND header_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED 1\n")
        else()
            string(APPEND header_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED 0\n")
        endif()
    endforeach()
    string(APPEND header_content "\n")

    # Generate pair type compilation flags
    string(APPEND header_content "// Pair type compilation flags\n")
    set(all_pair_keys "")

    # Collect all valid pairs from combinations_2
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        if(i LESS ${num_types} AND j LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            list(GET type_enums ${j} enum_j)
            enum_to_int_value("${enum_i}" int_i)
            enum_to_int_value("${enum_j}" int_j)
            set(pair_key "${int_i}_${int_j}")
            list(FIND all_pair_keys "${pair_key}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND all_pair_keys "${pair_key}")
            endif()
        endif()
    endforeach()

    # Generate all pair combinations
    foreach(type1 IN LISTS all_possible_types)
        foreach(type2 IN LISTS all_possible_types)
            set(pair_key "${type1}_${type2}")
            list(FIND all_pair_keys "${pair_key}" found_idx)
            if(found_idx GREATER_EQUAL 0)
                string(APPEND header_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED 1\n")
            else()
                string(APPEND header_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED 0\n")
            endif()
        endforeach()
    endforeach()
    string(APPEND header_content "\n")

    # Generate triple type compilation flags
    string(APPEND header_content "// Triple type compilation flags\n")
    set(all_triple_keys "")

    # Collect all valid triples from combinations_3
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
            set(triple_key "${int_i}_${int_j}_${int_k}")
            list(FIND all_triple_keys "${triple_key}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND all_triple_keys "${triple_key}")
            endif()
        endif()
    endforeach()

    # Generate all triple combinations
    foreach(type1 IN LISTS all_possible_types)
        foreach(type2 IN LISTS all_possible_types)
            foreach(type3 IN LISTS all_possible_types)
                set(triple_key "${type1}_${type2}_${type3}")
                list(FIND all_triple_keys "${triple_key}" found_idx)
                if(found_idx GREATER_EQUAL 0)
                    string(APPEND header_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED 1\n")
                else()
                    string(APPEND header_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED 0\n")
                endif()
            endforeach()
        endforeach()
    endforeach()
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 2: MAPPING TABLES
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 2: MAPPING TABLES\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Generate enum to number mappings (handling namespace prefixes)
    string(APPEND header_content "// DataType enum to number mappings (with namespace handling)\n")

    # Define ALL mappings regardless of what's in type_enums
    string(APPEND header_content "#define SD_ENUM_TO_NUM_BOOL 1\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_FLOAT8 2\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_HALF 3\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_HALF2 4\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_FLOAT32 5\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_DOUBLE 6\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT8 7\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT16 8\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT32 9\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT64 10\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT8 11\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT16 12\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT32 13\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT64 14\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_QINT8 15\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_QINT16 16\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_BFLOAT16 17\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UTF8 50\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UTF16 51\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UTF32 52\n")
    string(APPEND header_content "\n")

    # Also add the alias mappings
    string(APPEND header_content "// Constexpr alias to number mappings\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_BOOL 1\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_FLOAT8 2\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_HALF 3\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_HALF2 4\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_FLOAT32 5\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_DOUBLE 6\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT8 7\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT16 8\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT32 9\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT64 10\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT8 11\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT16 12\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT32 13\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT64 14\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_QINT8 15\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_QINT16 16\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_BFLOAT16 17\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UTF8 50\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UTF16 51\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UTF32 52\n")
    string(APPEND header_content "\n")

    # Generate C++ type to number mappings
    string(APPEND header_content "// C++ type name to number mappings\n")
    foreach(i RANGE 0 ${num_types})
        if(i LESS ${num_types})
            list(GET type_cpp_types ${i} cpp_type)
            list(GET type_enums ${i} enum_value)
            enum_to_int_value("${enum_value}" int_value)

            # Clean up the C++ type name for macro usage
            string(REPLACE "::" "__" safe_cpp_type "${cpp_type}")
            string(REPLACE " " "_" safe_cpp_type "${safe_cpp_type}")
            string(APPEND header_content "#define SD_TYPE_TO_NUM_${safe_cpp_type} ${int_value}\n")

            # Add common variations
            if(cpp_type STREQUAL "float")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_float ${int_value}\n")
            elseif(cpp_type STREQUAL "double")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_double ${int_value}\n")
            elseif(cpp_type STREQUAL "bool")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_bool ${int_value}\n")
            elseif(cpp_type STREQUAL "int8_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_int8_t ${int_value}\n")
            elseif(cpp_type STREQUAL "int16_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_int16_t ${int_value}\n")
            elseif(cpp_type STREQUAL "int32_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_int32_t ${int_value}\n")
            elseif(cpp_type STREQUAL "sd::LongType")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_sd__LongType ${int_value}\n")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_int64_t ${int_value}\n")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_long_long ${int_value}\n")
            elseif(cpp_type STREQUAL "uint8_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_uint8_t ${int_value}\n")
            elseif(cpp_type STREQUAL "uint16_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_uint16_t ${int_value}\n")
            elseif(cpp_type STREQUAL "uint32_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_uint32_t ${int_value}\n")
            elseif(cpp_type STREQUAL "uint64_t")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_uint64_t ${int_value}\n")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_sd__UnsignedLong ${int_value}\n")
            elseif(cpp_type STREQUAL "float16")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_float16 ${int_value}\n")
            elseif(cpp_type STREQUAL "bfloat16")
                string(APPEND header_content "#define SD_TYPE_TO_NUM_bfloat16 ${int_value}\n")
            endif()
        endif()
    endforeach()
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 3: CONDITIONAL EXPANSION PRIMITIVES (FIXED FOR WHITESPACE)
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 3: CONDITIONAL EXPANSION PRIMITIVES (VARIADIC)\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// CRITICAL: Using variadic macros to preserve whitespace\n")
    string(APPEND header_content "// Statement context expansions (for use in function bodies)\n")
    string(APPEND header_content "#define SD_IF_1(...) __VA_ARGS__\n")
    string(APPEND header_content "#define SD_IF_0(...) do {} while(0);\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Declaration context expansions (for use at file/namespace scope)\n")
    string(APPEND header_content "#define SD_IF_DECL_1(...) __VA_ARGS__\n")
    string(APPEND header_content "#define SD_IF_DECL_0(...) /* filtered out */\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Expression context expansions (for use in expressions)\n")
    string(APPEND header_content "#define SD_IF_EXPR_1(...) __VA_ARGS__\n")
    string(APPEND header_content "#define SD_IF_EXPR_0(...) ((void)0)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Special whitespace-preserving helpers\n")
    string(APPEND header_content "#define SD_UNPAREN(...) SD_UNPAREN_IMPL __VA_ARGS__\n")
    string(APPEND header_content "#define SD_UNPAREN_IMPL(...) __VA_ARGS__\n")
    string(APPEND header_content "#define SD_IDENTITY(...) __VA_ARGS__\n")
    string(APPEND header_content "#define SD_EXPAND(...) __VA_ARGS__\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 4: TOKEN MANIPULATION HELPERS
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 4: TOKEN MANIPULATION HELPERS\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// Basic concatenation macros\n")
    string(APPEND header_content "#define SD_CAT(a, b) SD_CAT_I(a, b)\n")
    string(APPEND header_content "#define SD_CAT_I(a, b) a ## b\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Three-token concatenation\n")
    string(APPEND header_content "#define SD_CAT3(a, b, c) SD_CAT3_I(a, b, c)\n")
    string(APPEND header_content "#define SD_CAT3_I(a, b, c) a ## b ## c\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Five-token concatenation\n")
    string(APPEND header_content "#define SD_CAT5(a, b, c, d, e) SD_CAT5_I(a, b, c, d, e)\n")
    string(APPEND header_content "#define SD_CAT5_I(a, b, c, d, e) a ## b ## c ## d ## e\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Seven-token concatenation\n")
    string(APPEND header_content "#define SD_CAT7(a, b, c, d, e, f, g) SD_CAT7_I(a, b, c, d, e, f, g)\n")
    string(APPEND header_content "#define SD_CAT7_I(a, b, c, d, e, f, g) a ## b ## c ## d ## e ## f ## g\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 5: COMPILATION CHECK MACROS
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 5: COMPILATION CHECK MACROS\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// Check if type combinations are compiled (returns 0 or 1)\n")
    string(APPEND header_content "#define SD_IS_SINGLE_COMPILED(NUM) \\\n")
    string(APPEND header_content "    SD_CAT3(SD_SINGLE_TYPE_, NUM, _COMPILED)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IS_PAIR_COMPILED(NUM1, NUM2) \\\n")
    string(APPEND header_content "    SD_CAT5(SD_PAIR_TYPE_, NUM1, _, NUM2, _COMPILED)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IS_TRIPLE_COMPILED(NUM1, NUM2, NUM3) \\\n")
    string(APPEND header_content "    SD_CAT7(SD_TRIPLE_TYPE_, NUM1, _, NUM2, _, NUM3, _COMPILED)\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 6: UNIFIED INTERFACE - NUMERIC (FIXED)
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 6: UNIFIED INTERFACE - NUMERIC (VARIADIC)\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// Direct numeric type ID interfaces (statement context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_COMPILED(TYPE_NUM, ...) \\\n")
    string(APPEND header_content "    SD_CAT(SD_IF_, SD_IS_SINGLE_COMPILED(TYPE_NUM))(__VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_COMPILED(TYPE1_NUM, TYPE2_NUM, ...) \\\n")
    string(APPEND header_content "    SD_CAT(SD_IF_, SD_IS_PAIR_COMPILED(TYPE1_NUM, TYPE2_NUM))(__VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_COMPILED(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM, ...) \\\n")
    string(APPEND header_content "    SD_CAT(SD_IF_, SD_IS_TRIPLE_COMPILED(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM))(__VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Direct numeric type ID interfaces (declaration context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_COMPILED_DECL(TYPE_NUM, ...) \\\n")
    string(APPEND header_content "    SD_CAT(SD_IF_DECL_, SD_IS_SINGLE_COMPILED(TYPE_NUM))(__VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_COMPILED_DECL(TYPE1_NUM, TYPE2_NUM, ...) \\\n")
    string(APPEND header_content "    SD_CAT(SD_IF_DECL_, SD_IS_PAIR_COMPILED(TYPE1_NUM, TYPE2_NUM))(__VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_COMPILED_DECL(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM, ...) \\\n")
    string(APPEND header_content "    SD_CAT(SD_IF_DECL_, SD_IS_TRIPLE_COMPILED(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM))(__VA_ARGS__)\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 7: UNIFIED INTERFACE - DATATYPE ENUM (FIXED)
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 7: UNIFIED INTERFACE - DATATYPE ENUM (VARIADIC)\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Add helper macros for proper evaluation with indirection
    string(APPEND header_content "// Helper macros for forcing evaluation through indirection\n")
    string(APPEND header_content "#define SD_EVAL_ENUM_TO_NUM_I(x) x\n")
    string(APPEND header_content "#define SD_EVAL_ENUM_TO_NUM(DTYPE) SD_EVAL_ENUM_TO_NUM_I(SD_CAT(SD_ENUM_TO_NUM_, DTYPE))\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// DataType enum interfaces (statement context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_DATATYPE_COMPILED(DTYPE, ...) \\\n")
    string(APPEND header_content "    SD_IF_SINGLE_COMPILED(SD_EVAL_ENUM_TO_NUM(DTYPE), __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_DATATYPE_COMPILED(DTYPE1, DTYPE2, ...) \\\n")
    string(APPEND header_content "    SD_IF_PAIR_COMPILED( \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE1), \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE2), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_DATATYPE_COMPILED(DTYPE1, DTYPE2, DTYPE3, ...) \\\n")
    string(APPEND header_content "    SD_IF_TRIPLE_COMPILED( \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE1), \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE2), \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE3), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// DataType enum interfaces (declaration context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_DATATYPE_COMPILED_DECL(DTYPE, ...) \\\n")
    string(APPEND header_content "    SD_IF_SINGLE_COMPILED_DECL(SD_EVAL_ENUM_TO_NUM(DTYPE), __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_DATATYPE_COMPILED_DECL(DTYPE1, DTYPE2, ...) \\\n")
    string(APPEND header_content "    SD_IF_PAIR_COMPILED_DECL( \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE1), \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE2), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_DATATYPE_COMPILED_DECL(DTYPE1, DTYPE2, DTYPE3, ...) \\\n")
    string(APPEND header_content "    SD_IF_TRIPLE_COMPILED_DECL( \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE1), \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE2), \\\n")
    string(APPEND header_content "        SD_EVAL_ENUM_TO_NUM(DTYPE3), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 8: UNIFIED INTERFACE - CONSTEXPR ALIASES (FIXED)
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 8: UNIFIED INTERFACE - CONSTEXPR ALIASES (VARIADIC)\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// Constexpr alias interfaces (statement context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_ALIAS_COMPILED(ALIAS, ...) \\\n")
    string(APPEND header_content "    SD_IF_SINGLE_COMPILED(SD_CAT(SD_ALIAS_TO_NUM_, ALIAS), __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_ALIAS_COMPILED(ALIAS1, ALIAS2, ...) \\\n")
    string(APPEND header_content "    SD_IF_PAIR_COMPILED( \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_ALIAS_COMPILED(ALIAS1, ALIAS2, ALIAS3, ...) \\\n")
    string(APPEND header_content "    SD_IF_TRIPLE_COMPILED( \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS3), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Constexpr alias interfaces (declaration context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_ALIAS_COMPILED_DECL(ALIAS, ...) \\\n")
    string(APPEND header_content "    SD_IF_SINGLE_COMPILED_DECL(SD_CAT(SD_ALIAS_TO_NUM_, ALIAS), __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_ALIAS_COMPILED_DECL(ALIAS1, ALIAS2, ...) \\\n")
    string(APPEND header_content "    SD_IF_PAIR_COMPILED_DECL( \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_ALIAS_COMPILED_DECL(ALIAS1, ALIAS2, ALIAS3, ...) \\\n")
    string(APPEND header_content "    SD_IF_TRIPLE_COMPILED_DECL( \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS3), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 9: UNIFIED INTERFACE - C++ TYPE NAMES (FIXED)
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 9: UNIFIED INTERFACE - C++ TYPE NAMES (VARIADIC)\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// C++ type name interfaces (statement context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_TYPE_COMPILED(TYPE, ...) \\\n")
    string(APPEND header_content "    SD_IF_SINGLE_COMPILED(SD_CAT(SD_TYPE_TO_NUM_, TYPE), __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_TYPE_COMPILED(TYPE1, TYPE2, ...) \\\n")
    string(APPEND header_content "    SD_IF_PAIR_COMPILED( \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_TYPE_COMPILED(TYPE1, TYPE2, TYPE3, ...) \\\n")
    string(APPEND header_content "    SD_IF_TRIPLE_COMPILED( \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE3), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// C++ type name interfaces (declaration context)\n")
    string(APPEND header_content "#define SD_IF_SINGLE_TYPE_COMPILED_DECL(TYPE, ...) \\\n")
    string(APPEND header_content "    SD_IF_SINGLE_COMPILED_DECL(SD_CAT(SD_TYPE_TO_NUM_, TYPE), __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_PAIR_TYPE_COMPILED_DECL(TYPE1, TYPE2, ...) \\\n")
    string(APPEND header_content "    SD_IF_PAIR_COMPILED_DECL( \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IF_TRIPLE_TYPE_COMPILED_DECL(TYPE1, TYPE2, TYPE3, ...) \\\n")
    string(APPEND header_content "    SD_IF_TRIPLE_COMPILED_DECL( \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \\\n")
    string(APPEND header_content "        SD_CAT(SD_TYPE_TO_NUM_, TYPE3), \\\n")
    string(APPEND header_content "        __VA_ARGS__)\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 10: BACKWARD COMPATIBILITY
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 10: BACKWARD COMPATIBILITY\n")
    string(APPEND header_content "// ============================================================================\n\n")

    string(APPEND header_content "// Legacy macros for existing code\n")
    string(APPEND header_content "#define SD_IS_SINGLE_DATATYPE_COMPILED(DTYPE) \\\n")
    string(APPEND header_content "    SD_IS_SINGLE_COMPILED(SD_CAT(SD_ENUM_TO_NUM_, DTYPE))\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IS_PAIR_DATATYPE_COMPILED(DTYPE1, DTYPE2) \\\n")
    string(APPEND header_content "    SD_IS_PAIR_COMPILED( \\\n")
    string(APPEND header_content "        SD_CAT(SD_ENUM_TO_NUM_, DTYPE1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ENUM_TO_NUM_, DTYPE2))\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "#define SD_IS_TRIPLE_DATATYPE_COMPILED(DTYPE1, DTYPE2, DTYPE3) \\\n")
    string(APPEND header_content "    SD_IS_TRIPLE_COMPILED( \\\n")
    string(APPEND header_content "        SD_CAT(SD_ENUM_TO_NUM_, DTYPE1), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ENUM_TO_NUM_, DTYPE2), \\\n")
    string(APPEND header_content "        SD_CAT(SD_ENUM_TO_NUM_, DTYPE3))\n")
    string(APPEND header_content "\n")

    string(APPEND header_content "// Whitespace-preserving wrappers for BUILD_* macros\n")
    string(APPEND header_content "// These help preserve whitespace when selective rendering is used\n")
    string(APPEND header_content "#define SD_PRESERVE_WS(...) __VA_ARGS__\n")
    string(APPEND header_content "#define SD_PASS_THROUGH(...) __VA_ARGS__\n")
    string(APPEND header_content "\n")
    
    string(APPEND header_content "// Helper for fixing parenthesized arguments\n")
    string(APPEND header_content "#define SD_FIX_PAREN(x) SD_FIX_PAREN_IMPL x\n")
    string(APPEND header_content "#define SD_FIX_PAREN_IMPL(...) __VA_ARGS__\n")
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 11: DIAGNOSTIC INFORMATION
    # ============================================================================

    if(SD_ENABLE_DIAGNOSTICS)
        string(APPEND header_content "// ============================================================================\n")
        string(APPEND header_content "// SECTION 11: DIAGNOSTIC INFORMATION\n")
        string(APPEND header_content "// ============================================================================\n\n")

        string(APPEND header_content "// Generated with the following active types:\n")
        foreach(i RANGE 0 ${num_types})
            if(i LESS ${num_types})
                list(GET type_enums ${i} enum_value)
                list(GET type_cpp_types ${i} cpp_type)
                enum_to_int_value("${enum_value}" int_value)
                string(APPEND header_content "// [${i}] ${enum_value} -> ${cpp_type} (ID: ${int_value})\n")
            endif()
        endforeach()
        string(APPEND header_content "\n")

        list(LENGTH all_pair_keys num_pairs)
        list(LENGTH all_triple_keys num_triples)
        string(APPEND header_content "// Statistics:\n")
        string(APPEND header_content "// - Single types compiled: ${num_types}\n")
        string(APPEND header_content "// - Pair combinations compiled: ${num_pairs}\n")
        string(APPEND header_content "// - Triple combinations compiled: ${num_triples}\n")
        string(APPEND header_content "// - VARIADIC MACROS ENABLED FOR WHITESPACE PRESERVATION\n")
        string(APPEND header_content "\n")
    endif()

    # Close the header guard
    string(APPEND header_content "#endif // SD_SELECTIVE_RENDERING_H\n")

    # Write the header file
    file(WRITE "${header_file}" "${header_content}")

    # Report generation results
    if(SD_ENABLE_DIAGNOSTICS)
        list(LENGTH all_triple_keys total_triple_combinations)
        list(LENGTH all_pair_keys total_pair_combinations)
        list(LENGTH compiled_type_numbers total_single_types)
        message(STATUS "Generated selective_rendering.h (with variadic fixes):")
        message(STATUS "  - Location: ${header_file}")
        message(STATUS "  - Single types: ${total_single_types}")
        message(STATUS "  - Pair combinations: ${total_pair_combinations}")
        message(STATUS "  - Triple combinations: ${total_triple_combinations}")
        message(STATUS "  - VARIADIC MACROS ENABLED")
    endif()
endfunction()
# ============================================================================
# SECTION 5: MAIN ORCHESTRATOR FUNCTIONS
# ============================================================================
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

    srcore_discover_active_types(active_types_indices discovered_enums discovered_cpp_types)
    list(LENGTH active_types_indices type_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types discovered!")
    endif()

    srcore_generate_combinations("${active_types_indices}" "${SRCORE_TYPE_PROFILE}" combinations_2 combinations_3)
    srcore_generate_headers("${active_types_indices}" "${combinations_2}" "${combinations_3}" "${SRCORE_OUTPUT_DIR}" "${discovered_enums}" "${discovered_cpp_types}")


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
    if(NOT CMAKE_CROSSCOMPILING AND NOT ANDROID)
        setup_selective_rendering_unified(${ARGN})
        srcore_map_to_legacy_variables()
        if(SD_ENABLE_DIAGNOSTICS)
            srcore_generate_diagnostic_report()
        endif()
    else()
        setup_selective_rendering_unified(${ARGN})
        srcore_map_to_legacy_variables()
    endif()

    # Propagate variables to parent scope
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

    # Final verification
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        srcore_emergency_fallback()
        srcore_map_to_legacy_variables()

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
    endif()

    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        message(FATAL_ERROR "Unable to establish UNIFIED_COMBINATIONS_3 even with emergency fallback!")
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

        list(LENGTH UNIFIED_ACTIVE_TYPES legacy_count)
        set(SD_COMMON_TYPES_COUNT ${legacy_count} PARENT_SCOPE)
        set(SD_COMMON_TYPES_COUNT ${legacy_count} CACHE INTERNAL "Legacy type count")

        set(type_index 0)
        foreach(type_name ${UNIFIED_ACTIVE_TYPES})
            set(TYPE_NAME_${type_index} "${type_name}" PARENT_SCOPE)
            set(TYPE_NAME_${type_index} "${type_name}" CACHE INTERNAL "Legacy reverse type lookup")
            math(EXPR type_index "${type_index} + 1")
        endforeach()
    endif()
endfunction()

function(srcore_generate_diagnostic_report)
    if(NOT SD_ENABLE_DIAGNOSTICS)
        return()
    endif()

    set(report_file "${CMAKE_BINARY_DIR}/selective_rendering_diagnostic_report.txt")
    set(report_content "")

    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND report_content "SelectiveRenderingCore Diagnostic Report\n")
    string(APPEND report_content "Generated: ${current_time}\n")
    string(APPEND report_content "========================================\n\n")

    string(APPEND report_content "Configuration:\n")
    string(APPEND report_content "- SD_ENABLE_SEMANTIC_FILTERING: ${SD_ENABLE_SEMANTIC_FILTERING}\n")
    string(APPEND report_content "- SD_TYPE_PROFILE: ${SD_TYPE_PROFILE}\n")
    string(APPEND report_content "- SD_SELECTIVE_TYPES: ${SD_SELECTIVE_TYPES}\n")
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

    # Sample combinations (only if diagnostics enabled)
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
                string(APPEND report_content "  (${i},${j},${k})\n")
            endif()

            math(EXPR sample_count "${sample_count} + 1")
        endforeach()
        string(APPEND report_content "\n")
    endif()

    file(WRITE "${report_file}" "${report_content}")
endfunction()

# ============================================================================
# OPTIMIZED WRAPPER FUNCTIONS (Production Ready)
# ============================================================================

# Main wrapper for existing code
function(setup_selective_rendering)
    setup_selective_rendering_unified_safe()

    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()
endfunction()

# Legacy wrapper functions (now no-ops for performance)
function(track_combination_states active_types combinations_3)
    # Handled internally - no action needed
endfunction()

function(generate_selective_rendering_header)
    # Handled internally - no action needed
endfunction()

function(generate_selective_wrapper_header)
    # Handled internally - no action needed
endfunction()

function(setup_definitive_semantic_filtering_with_selective_rendering)
    set(SD_ENABLE_SEMANTIC_FILTERING TRUE PARENT_SCOPE)
    set(SD_ENABLE_SELECTIVE_RENDERING TRUE PARENT_SCOPE)

    setup_selective_rendering_unified_safe()

    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
endfunction()

function(enhanced_semantic_filtering_setup)
    setup_definitive_semantic_filtering_with_selective_rendering()
endfunction()

function(setup_definitive_semantic_filtering)
    set(SD_ENABLE_SEMANTIC_FILTERING TRUE PARENT_SCOPE)
    setup_selective_rendering_unified_safe(TYPE_PROFILE "${SD_TYPE_PROFILE}")

    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" CACHE INTERNAL "2-type combinations" FORCE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" CACHE INTERNAL "3-type combinations" FORCE)
    endif()
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" CACHE INTERNAL "Active type list" FORCE)
    endif()
endfunction()

function(initialize_definitive_combinations)
    setup_definitive_semantic_filtering()
endfunction()

function(extract_definitive_types result_var)
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    else()
        srcore_auto_setup()
        set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()
endfunction()

function(generate_definitive_combinations active_types result_2_var result_3_var)
    if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
        set(${result_2_var} "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        set(${result_3_var} "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    else()
        srcore_auto_setup()
        set(${result_2_var} "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        set(${result_3_var} "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
endfunction()

function(validate_critical_types_coverage active_types combinations_3)
    # Handled internally - no action needed
endfunction()

# ============================================================================
# PRODUCTION-OPTIMIZED SEMANTIC ENGINE INTEGRATION
# ============================================================================

# Simplified version without debug overhead
function(setup_enhanced_semantic_validation)
    # Core validation logic without debug output
    if(SD_ENABLE_SEMANTIC_FILTERING)
        if(NOT SD_TYPE_PROFILE OR SD_TYPE_PROFILE STREQUAL "")
            if(SD_TYPES_LIST_COUNT GREATER 0)
                set(detected_profile "")
                if("int8_t" IN_LIST SD_TYPES_LIST AND "uint8_t" IN_LIST SD_TYPES_LIST)
                    set(detected_profile "quantization")
                elseif("float16" IN_LIST SD_TYPES_LIST OR "bfloat16" IN_LIST SD_TYPES_LIST)
                    set(detected_profile "training")
                elseif(SD_TYPES_LIST MATCHES ".*string.*")
                    set(detected_profile "nlp")
                endif()

                if(NOT detected_profile STREQUAL "")
                    set(SD_TYPE_PROFILE "${detected_profile}" PARENT_SCOPE)
                else()
                    set(SD_TYPE_PROFILE "inference" PARENT_SCOPE)
                endif()
            else()
                set(SD_TYPE_PROFILE "inference" PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

# Remove debug function calls to avoid GCC function tracing overhead
macro(print_status_colored level message)
    # Only output if diagnostics are explicitly enabled
    if(SD_ENABLE_DIAGNOSTICS)
        message(STATUS "${message}")
    endif()
endmacro()



function(_internal_srcore_generate_helper_macros output_var)
    set(helper_content "")

    string(APPEND helper_content "#define SD_BUILD_TRIPLE_IF_VALID(t1, t2, t3, build_macro) \\\n")
    string(APPEND helper_content "    do { \\\n")
    string(APPEND helper_content "        if (SD_IS_TRIPLE_TYPE_COMPILED(t1, t2, t3)) { \\\n")
    string(APPEND helper_content "            SD_DISPATCH_TRIPLE_RUNTIME(t1, t2, t3, build_macro); \\\n")
    string(APPEND helper_content "        } \\\n")
    string(APPEND helper_content "    } while(0)\n\n")

    string(APPEND helper_content "#define SD_BUILD_PAIR_IF_VALID(t1, t2, build_macro) \\\n")
    string(APPEND helper_content "    do { \\\n")
    string(APPEND helper_content "        if (SD_IS_PAIR_TYPE_COMPILED(t1, t2)) { \\\n")
    string(APPEND helper_content "            SD_DISPATCH_PAIR_RUNTIME(t1, t2, build_macro); \\\n")
    string(APPEND helper_content "        } \\\n")
    string(APPEND helper_content "    } while(0)\n\n")

    string(APPEND helper_content "#define SD_BUILD_SINGLE_IF_VALID(t1, build_macro) \\\n")
    string(APPEND helper_content "    do { \\\n")
    string(APPEND helper_content "        if (SD_IS_SINGLE_TYPE_COMPILED(t1)) { \\\n")
    string(APPEND helper_content "            SD_DISPATCH_SINGLE_RUNTIME(t1, build_macro); \\\n")
    string(APPEND helper_content "        } \\\n")
    string(APPEND helper_content "    } while(0)\n\n")

    set(${output_var} "${helper_content}" PARENT_SCOPE)
endfunction()

function(srcore_generate_enhanced_header active_indices combinations_2 combinations_3 output_dir type_enums type_cpp_types)
    _internal_srcore_generate_validity_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")
    _internal_srcore_append_runtime_dispatch_to_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")
endfunction()

function(_internal_srcore_append_runtime_dispatch_to_header active_indices type_enums type_cpp_types combinations_2 combinations_3 output_dir)
    set(header_file "${output_dir}/system/selective_rendering.h")

    _internal_srcore_generate_helper_macros(helper_macros)

    if(EXISTS "${header_file}")
        file(READ "${header_file}" existing_content)

        string(REGEX REPLACE "\n#endif // SD_SELECTIVE_RENDERING_H\n?$" "" content_without_endif "${existing_content}")

        set(new_content "${content_without_endif}")
        string(APPEND new_content "\n${dispatch_macros}")
        string(APPEND new_content "${helper_macros}")
        string(APPEND new_content "#endif // SD_SELECTIVE_RENDERING_H\n")

        file(WRITE "${header_file}" "${new_content}")

        if(SD_ENABLE_DIAGNOSTICS)
            list(LENGTH combinations_3 total_triple_combinations)
            list(LENGTH combinations_2 total_pair_combinations)
            message(STATUS "Enhanced selective_rendering.h with runtime dispatch - ${total_pair_combinations} pair dispatches, ${total_triple_combinations} triple dispatches")
        endif()
    else()
        message(FATAL_ERROR "Cannot append runtime dispatch - header file does not exist: ${header_file}")
    endif()
endfunction()

function(srcore_enable_runtime_dispatch)
    set(SD_ENABLE_RUNTIME_DISPATCH TRUE PARENT_SCOPE)
    set(SD_ENABLE_RUNTIME_DISPATCH TRUE CACHE BOOL "Enable runtime dispatch macro generation")

    if(SD_ENABLE_DIAGNOSTICS)
        message(STATUS "Runtime dispatch enabled - will generate SD_DISPATCH_*_RUNTIME macros")
    endif()
endfunction()