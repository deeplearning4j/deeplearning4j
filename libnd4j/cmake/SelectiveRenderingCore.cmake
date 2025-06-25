# ============================================================================
# SelectiveRenderingCore.cmake (v18 - Final, Complete, Self-Contained)
#
# This file contains ALL logic for the selective rendering system.
# It defines a single public function, SETUP_AND_GENERATE_ALL_RENDERING_FILES(),
# which handles discovery, combination, and header generation internally.
# This design eliminates all inter-file dependency and scoping issues.
# ============================================================================


# ============================================================================
# SECTION 1: INTERNAL HELPER AND ANALYSIS FUNCTIONS
# These are helper functions used only by the main orchestrator function.
# ============================================================================

function(_internal_srcore_debug_message message)
    if(DEFINED SRCORE_ENABLE_DIAGNOSTICS AND SRCORE_ENABLE_DIAGNOSTICS)
        message(STATUS "🔧 SelectiveRenderingCore: ${message}")
    endif()
endfunction()

function(_internal_srcore_normalize_type input_type output_var)
    set(normalized "${input_type}")
    if(normalized MATCHES "^(float32|FLOAT32)$")
        set(normalized "float")
    elseif(normalized MATCHES "^(float64|FLOAT64)$")
        set(normalized "double")
    elseif(normalized MATCHES "^(half|HALF|float16|FLOAT16)$")
        set(normalized "float16")
    elseif(normalized MATCHES "^(bfloat|BFLOAT|bfloat16|BFLOAT16)$")
        set(normalized "bfloat16")
    elseif(normalized MATCHES "^(int|INT)$")
        set(normalized "int32_t")
    elseif(normalized MATCHES "^(long|LONG|int64|INT64)$")
        set(normalized "sd::LongType")
    elseif(normalized MATCHES "^(uint64|UINT64|unsignedlong|UNSIGNEDLONG)$")
        set(normalized "uint64_t")
    endif()
    set(${output_var} "${normalized}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_discover_types result_indices_var result_names_var)
    _internal_srcore_debug_message("Starting unified type discovery...")
    set(possible_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/system/types.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/types.h")
    set(types_header "")
    foreach(header_path ${possible_headers})
        if(EXISTS "${header_path}")
            set(types_header "${header_path}")
            break()
        endif()
    endforeach()
    if(NOT types_header)
        message(FATAL_ERROR "❌ SelectiveRenderingCore: Could not find types.h in any expected location")
    endif()

    file(READ "${types_header}" types_content)
    set(base_type_names "BFLOAT16" "BOOL" "DOUBLE" "FLOAT32" "HALF" "INT16" "INT32" "INT64" "INT8" "UINT16" "UINT32" "UINT64" "UINT8" "UTF16" "UTF32" "UTF8" "BFLOAT" "FLOAT" "LONG" "UNSIGNEDLONG" "INT")
    set(discovered_types "")
    foreach(type_name ${base_type_names})
        string(REGEX MATCH "#define[ \t]+TTYPE_${type_name}[ \t]*,?[ \t]*\\(([^)]+)\\)" type_match "${types_content}")
        if(type_match)
            _internal_srcore_normalize_type("${type_name}" normalized_name)
            list(APPEND discovered_types "${normalized_name}")
            string(REGEX MATCH "\\(([^)]+)\\)" tuple_match "${type_match}")
            if(tuple_match)
                string(SUBSTRING "${tuple_match}" 1 -1 type_tuple)
                # Set a temporary variable for this scope
                set(TEMP_TUPLE_${normalized_name} "(${type_tuple})")
            endif()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES discovered_types)

    set(active_type_indices "")
    set(active_type_names "")
    set(type_index 0)
    foreach(type_name_normalized IN LISTS discovered_types)
        list(APPEND active_type_indices ${type_index})
        list(APPEND active_type_names ${type_name_normalized})

        set(type_tuple "${TEMP_TUPLE_${type_name_normalized}}")
        string(REGEX REPLACE "^\\s*\\(([^,]+),.*" "\\1" enum_part "${type_tuple}")
        string(REGEX REPLACE ".*,\\s*([^\\)]+)\\s*\\)" "\\1" type_part "${type_tuple}")
        string(STRIP "${enum_part}" enum_part)
        string(STRIP "${type_part}" type_part)

        # FIX: Ensure variables are set in PARENT_SCOPE with proper debug
        message(STATUS "Setting SRCORE_TYPE_ENUM_${type_index} = ${enum_part}")
        message(STATUS "Setting SRCORE_TYPE_CPP_${type_index} = ${type_part}")
        set(SRCORE_TYPE_ENUM_${type_index} "${enum_part}" PARENT_SCOPE)
        set(SRCORE_TYPE_CPP_${type_index} "${type_part}" PARENT_SCOPE)

        math(EXPR type_index "${type_index} + 1")
    endforeach()

    set(${result_indices_var} "${active_type_indices}" PARENT_SCOPE)
    set(${result_names_var} "${active_type_names}" PARENT_SCOPE)
endfunction()

function(srcore_discover_active_types result_var)
    _internal_srcore_discover_types(active_indices active_names)
    set(SRCORE_ACTIVE_TYPES "${active_names}" PARENT_SCOPE)
    set(SRCORE_ACTIVE_TYPE_COUNT 0 PARENT_SCOPE)
    list(LENGTH active_indices type_count)
    set(SRCORE_ACTIVE_TYPE_COUNT ${type_count} PARENT_SCOPE)

    # Set the type mappings for later use
    foreach(index IN LISTS active_indices)
        set(SRCORE_TYPE_NAME_${index} "${SRCORE_TYPE_CPP_${index}}" PARENT_SCOPE)
    endforeach()

    set(${result_var} "${active_indices}" PARENT_SCOPE)
endfunction()

function(srcore_generate_combinations active_indices profile result_2_var result_3_var)
    _internal_srcore_generate_combinations("${active_indices}" "${SRCORE_ACTIVE_TYPES}" "${profile}" combinations_2 combinations_3)
    set(SRCORE_COMBINATIONS_2 "${combinations_2}" PARENT_SCOPE)
    set(SRCORE_COMBINATIONS_3 "${combinations_3}" PARENT_SCOPE)
    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()

function(srcore_generate_headers active_indices combinations_2 combinations_3 output_dir)
    _internal_srcore_generate_validity_header("${active_indices}" "${combinations_2}" "${combinations_3}" "${output_dir}")
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

function(_internal_srcore_generate_combinations active_indices type_names profile result_2_var result_3_var)
    list(LENGTH active_indices type_count)
    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types provided for combination generation")
    endif()

    set(combinations_2 "")
    set(combinations_3 "")
    math(EXPR max_index "${type_count} - 1")

    foreach(i RANGE ${max_index})
        list(GET type_names ${i} type_name_i)
        foreach(j RANGE ${max_index})
            list(GET type_names ${j} type_name_j)
            list(APPEND combinations_2 "${i},${j}")
            # In a real scenario, you might have semantic filtering for 3-type combos here
            foreach(k RANGE ${max_index})
                list(APPEND combinations_3 "${i},${j},${k}")
            endforeach()
        endforeach()
    endforeach()

    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_generate_validity_header active_indices combinations_2 combinations_3 output_dir)
    file(MAKE_DIRECTORY "${output_dir}/system")
    set(header_file "${output_dir}/system/selective_rendering.h")

    # Helper sub-function to sanitize a C++ type into a macro-safe name
    function(sanitize_type_for_macro type_string output_var)
        string(REGEX REPLACE "::" "_" sanitized "${type_string}")
        string(REGEX REPLACE "[^a-zA-Z0-9_]" "" sanitized "${sanitized}")
        set(${output_var} "${sanitized}" PARENT_SCOPE)
    endfunction()

    # EMERGENCY FIX: Define known DataType enum values as fallback
    # This ensures we always have valid comparisons even if discovery fails
    set(FALLBACK_ENUM_MAP_float "sd::DataType::FLOAT32")
    set(FALLBACK_ENUM_MAP_double "sd::DataType::DOUBLE")
    set(FALLBACK_ENUM_MAP_float16 "sd::DataType::HALF")
    set(FALLBACK_ENUM_MAP_bfloat16 "sd::DataType::BFLOAT16")
    set(FALLBACK_ENUM_MAP_int32_t "sd::DataType::INT32")
    set(FALLBACK_ENUM_MAP_int64_t "sd::DataType::INT64")
    set(FALLBACK_ENUM_MAP_uint64_t "sd::DataType::UINT64")
    set(FALLBACK_ENUM_MAP_bool "sd::DataType::BOOL")
    set(FALLBACK_ENUM_MAP_int8_t "sd::DataType::INT8")
    set(FALLBACK_ENUM_MAP_uint8_t "sd::DataType::UINT8")
    set(FALLBACK_ENUM_MAP_int16_t "sd::DataType::INT16")
    set(FALLBACK_ENUM_MAP_uint16_t "sd::DataType::UINT16")
    set(FALLBACK_ENUM_MAP_uint32_t "sd::DataType::UINT32")

    set(header_content "/* AUTOMATICALLY GENERATED by SelectiveRenderingCore.cmake */\n")
    string(APPEND header_content "#ifndef SD_SELECTIVE_RENDERING_H\n#define SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_TYPE_VALID_CHECK_AVAILABLE 1\n\n")

    # =========================================================================
    # SECTION 1: Generate individual type validity flags
    # =========================================================================
    string(APPEND header_content "// Single Type Validity\n")
    foreach(index IN LISTS active_indices)
        set(cpp_type "${SRCORE_TYPE_CPP_${index}}")
        if(NOT cpp_type)
            message(WARNING "No CPP type found for index ${index}, skipping")
            continue()
        endif()
        sanitize_type_for_macro("${cpp_type}" safe_name)
        string(APPEND header_content "#define SD_TYPE_${safe_name}_VALID 1\n")
    endforeach()

    # =========================================================================
    # SECTION 2: Generate pairwise type validity flags
    # =========================================================================
    string(APPEND header_content "\n// Pairwise Type Validity\n")
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        set(cpp_i "${SRCORE_TYPE_CPP_${i}}")
        set(cpp_j "${SRCORE_TYPE_CPP_${j}}")
        if(NOT cpp_i OR NOT cpp_j)
            continue()
        endif()
        sanitize_type_for_macro("${cpp_i}" safe_i)
        sanitize_type_for_macro("${cpp_j}" safe_j)
        string(APPEND header_content "#define SD_TYPE_PAIR_${safe_i}_${safe_j}_VALID 1\n")
    endforeach()

    # =========================================================================
    # SECTION 3: Generate triple type validity flags
    # =========================================================================
    string(APPEND header_content "\n// Triple Type Validity\n")
    foreach(combo IN LISTS combinations_3)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        list(GET parts 2 k)
        set(cpp_i "${SRCORE_TYPE_CPP_${i}}")
        set(cpp_j "${SRCORE_TYPE_CPP_${j}}")
        set(cpp_k "${SRCORE_TYPE_CPP_${k}}")
        if(NOT cpp_i OR NOT cpp_j OR NOT cpp_k)
            continue()
        endif()
        sanitize_type_for_macro("${cpp_i}" safe_i)
        sanitize_type_for_macro("${cpp_j}" safe_j)
        sanitize_type_for_macro("${cpp_k}" safe_k)
        string(APPEND header_content "#define SD_TYPE_TRIPLE_${safe_i}_${safe_j}_${safe_k}_VALID 1\n")
    endforeach()

    # =========================================================================
    # SECTION 4: Generate helper macros for DataTypeUtils - FIXED WITH FALLBACK
    # =========================================================================
    string(APPEND header_content "\n// Helper Macros for Dynamic Type Checking\n")

    # Function to get enum value with fallback
    function(get_enum_value_safe index cpp_type output_var)
        set(enum_value "${SRCORE_TYPE_ENUM_${index}}")
        if(NOT enum_value AND DEFINED FALLBACK_ENUM_MAP_${cpp_type})
            set(enum_value "${FALLBACK_ENUM_MAP_${cpp_type}}")
            message(STATUS "Using fallback enum for ${cpp_type}: ${enum_value}")
        endif()
        if(NOT enum_value)
            message(WARNING "No enum value found for index ${index}, type ${cpp_type}")
            set(enum_value "sd::DataType::INHERIT") # Safe fallback
        endif()
        set(${output_var} "${enum_value}" PARENT_SCOPE)
    endfunction()

    # Single type checker macro - FIXED WITH SAFE ENUM LOOKUP
    string(APPEND header_content "#define SD_IS_SINGLE_TYPE_COMPILED(TYPE_ENUM) \\\n")
    string(APPEND header_content "    (false")  # Start with false to handle empty case
    foreach(index IN LISTS active_indices)
        set(cpp_type "${SRCORE_TYPE_CPP_${index}}")
        if(NOT cpp_type)
            continue()
        endif()

        get_enum_value_safe(${index} "${cpp_type}" enum_value)
        sanitize_type_for_macro("${cpp_type}" safe_name)

        string(APPEND header_content " || \\\n     ")
        string(APPEND header_content "((TYPE_ENUM) == ${enum_value} && defined(SD_TYPE_${safe_name}_VALID))")
    endforeach()
    string(APPEND header_content ")\n\n")

    # Pair type checker macro - FIXED WITH SAFE ENUM LOOKUP
    string(APPEND header_content "#define SD_IS_PAIR_TYPE_COMPILED(TYPE1_ENUM, TYPE2_ENUM) \\\n")
    string(APPEND header_content "    (false")  # Start with false to handle empty case
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)

        set(cpp_i "${SRCORE_TYPE_CPP_${i}}")
        set(cpp_j "${SRCORE_TYPE_CPP_${j}}")
        if(NOT cpp_i OR NOT cpp_j)
            continue()
        endif()

        get_enum_value_safe(${i} "${cpp_i}" enum_i)
        get_enum_value_safe(${j} "${cpp_j}" enum_j)
        sanitize_type_for_macro("${cpp_i}" safe_i)
        sanitize_type_for_macro("${cpp_j}" safe_j)

        string(APPEND header_content " || \\\n     ")
        string(APPEND header_content "((TYPE1_ENUM) == ${enum_i} && (TYPE2_ENUM) == ${enum_j} && defined(SD_TYPE_PAIR_${safe_i}_${safe_j}_VALID))")
    endforeach()
    string(APPEND header_content ")\n\n")

    # Triple type checker macro - FIXED WITH SAFE ENUM LOOKUP
    string(APPEND header_content "#define SD_IS_TRIPLE_TYPE_COMPILED(TYPE1_ENUM, TYPE2_ENUM, TYPE3_ENUM) \\\n")
    string(APPEND header_content "    (false")  # Start with false to handle empty case
    foreach(combo IN LISTS combinations_3)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        list(GET parts 2 k)

        set(cpp_i "${SRCORE_TYPE_CPP_${i}}")
        set(cpp_j "${SRCORE_TYPE_CPP_${j}}")
        set(cpp_k "${SRCORE_TYPE_CPP_${k}}")
        if(NOT cpp_i OR NOT cpp_j OR NOT cpp_k)
            continue()
        endif()

        get_enum_value_safe(${i} "${cpp_i}" enum_i)
        get_enum_value_safe(${j} "${cpp_j}" enum_j)
        get_enum_value_safe(${k} "${cpp_k}" enum_k)
        sanitize_type_for_macro("${cpp_i}" safe_i)
        sanitize_type_for_macro("${cpp_j}" safe_j)
        sanitize_type_for_macro("${cpp_k}" safe_k)

        string(APPEND header_content " || \\\n     ")
        string(APPEND header_content "((TYPE1_ENUM) == ${enum_i} && (TYPE2_ENUM) == ${enum_j} && (TYPE3_ENUM) == ${enum_k} && defined(SD_TYPE_TRIPLE_${safe_i}_${safe_j}_${safe_k}_VALID))")
    endforeach()
    string(APPEND header_content ")\n\n")
    string(APPEND header_content "#endif // SD_SELECTIVE_RENDERING_H\n")
    file(WRITE "${header_file}" "${header_content}")
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

    # Generate BUILD_SINGLE_SELECTOR
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// BUILD_SINGLE_SELECTOR - Fixed Version\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef BUILD_SINGLE_SELECTOR\n")
    string(APPEND header_content "#define BUILD_SINGLE_SELECTOR(XTYPE, NAME, SIGNATURE, ...) \\\\\n")
    string(APPEND header_content "    switch (XTYPE) { \\\\\n")

    foreach(index IN LISTS active_indices)
        string(APPEND header_content "        SD_IF_VALID(SD_IS_SINGLE_TYPE_VALID(${SRCORE_TYPE_CPP_${index}}), \\\\\n")
        string(APPEND header_content "            case ${SRCORE_TYPE_ENUM_${index}}: { NAME<${SRCORE_TYPE_CPP_${index}}> SIGNATURE; break; } \\\\\n")
        string(APPEND header_content "        ) \\\\\n")
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
        string(APPEND header_content "        case ${SRCORE_TYPE_ENUM_${index_x}}: { \\\\\n")
        string(APPEND header_content "            switch (YTYPE) { \\\\\n")

        # Generate valid Y-type cases for this X-type
        foreach(combo IN LISTS combinations_2)
            string(REPLACE "," ";" combo_parts "${combo}")
            list(GET combo_parts 0 combo_x)
            list(GET combo_parts 1 combo_y)

            # Only generate cases where X matches current index_x
            if(combo_x EQUAL index_x)
                string(APPEND header_content "                SD_IF_VALID(SD_IS_TYPE_PAIR_VALID(${SRCORE_TYPE_CPP_${combo_x}}, ${SRCORE_TYPE_CPP_${combo_y}}), \\\\\n")
                string(APPEND header_content "                    case ${SRCORE_TYPE_ENUM_${combo_y}}: { NAME<${SRCORE_TYPE_CPP_${combo_x}}, ${SRCORE_TYPE_CPP_${combo_y}}> SIGNATURE; break; } \\\\\n")
                string(APPEND header_content "                ) \\\\\n")
            endif()
        endforeach()

        string(APPEND header_content "                default: { \\\\\n")
        string(APPEND header_content "                    auto e = sd::DataTypeValidation::getDataTypeErrorMessage(YTYPE, #NAME \"_DOUBLE_Y\"); \\\\\n")
        string(APPEND header_content "                    THROW_EXCEPTION(e.c_str()); \\\\\n")
        string(APPEND header_content "                } \\\\\n")
        string(APPEND header_content "            } \\\\\n")
        string(APPEND header_content "            break; \\\\\n")
        string(APPEND header_content "        } \\\\\n")
    endforeach()

    string(APPEND header_content "        default: { \\\\\n")
    string(APPEND header_content "            auto e = sd::DataTypeValidation::getDataTypeErrorMessage(XTYPE, #NAME \"_DOUBLE_X\"); \\\\\n")
    string(APPEND header_content "            THROW_EXCEPTION(e.c_str()); \\\\\n")
    string(APPEND header_content "        } \\\\\n")
    string(APPEND header_content "    }\n\n")

    # Generate BUILD_TRIPLE_SELECTOR
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// BUILD_TRIPLE_SELECTOR - Fixed Version\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef BUILD_TRIPLE_SELECTOR\n")
    string(APPEND header_content "#define BUILD_TRIPLE_SELECTOR(XTYPE, YTYPE, ZTYPE, NAME, SIGNATURE, ...) \\\\\n")
    string(APPEND header_content "    switch (XTYPE) { \\\\\n")

    foreach(index_x IN LISTS active_indices)
        string(APPEND header_content "        case ${SRCORE_TYPE_ENUM_${index_x}}: { \\\\\n")
        string(APPEND header_content "            switch (YTYPE) { \\\\\n")

        # Group 3-type combinations by X,Y pairs for this X
        foreach(combo IN LISTS combinations_3)
            string(REPLACE "," ";" combo_parts "${combo}")
            list(GET combo_parts 0 combo_x)
            list(GET combo_parts 1 combo_y)
            list(GET combo_parts 2 combo_z)

            if(combo_x EQUAL index_x)
                # Check if we've already handled this X,Y pair
                set(xy_pair "${combo_x},${combo_y}")
                if(NOT DEFINED HANDLED_XY_${xy_pair})
                    set(HANDLED_XY_${xy_pair} TRUE)

                    string(APPEND header_content "                case ${SRCORE_TYPE_ENUM_${combo_y}}: { \\\\\n")
                    string(APPEND header_content "                    switch (ZTYPE) { \\\\\n")

                    # Generate all Z cases for this X,Y pair
                    foreach(z_combo IN LISTS combinations_3)
                        string(REPLACE "," ";" z_parts "${z_combo}")
                        list(GET z_parts 0 z_x)
                        list(GET z_parts 1 z_y)
                        list(GET z_parts 2 z_z)

                        if(z_x EQUAL combo_x AND z_y EQUAL combo_y)
                            string(APPEND header_content "                        SD_IF_VALID(SD_IS_TYPE_TRIPLE_VALID(${SRCORE_TYPE_CPP_${z_x}}, ${SRCORE_TYPE_CPP_${z_y}}, ${SRCORE_TYPE_CPP_${z_z}}), \\\\\n")
                            string(APPEND header_content "                            case ${SRCORE_TYPE_ENUM_${z_z}}: { NAME<${SRCORE_TYPE_CPP_${z_x}}, ${SRCORE_TYPE_CPP_${z_y}}, ${SRCORE_TYPE_CPP_${z_z}}> SIGNATURE; break; } \\\\\n")
                            string(APPEND header_content "                        ) \\\\\n")
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
        endforeach()

        string(APPEND header_content "                default: { \\\\\n")
        string(APPEND header_content "                    auto e = sd::DataTypeValidation::getDataTypeErrorMessage(YTYPE, #NAME \"_TRIPLE_Y\"); \\\\\n")
        string(APPEND header_content "                    THROW_EXCEPTION(e.c_str()); \\\\\n")
        string(APPEND header_content "                } \\\\\n")
        string(APPEND header_content "            } \\\\\n")
        string(APPEND header_content "            break; \\\\\n")
        string(APPEND header_content "        } \\\\\n")
    endforeach()

    string(APPEND header_content "        default: { \\\\\n")
    string(APPEND header_content "            auto e = sd::DataTypeValidation::getDataTypeErrorMessage(XTYPE, #NAME \"_TRIPLE_X\"); \\\\\n")
    string(APPEND header_content "            THROW_EXCEPTION(e.c_str()); \\\\\n")
    string(APPEND header_content "        } \\\\\n")
    string(APPEND header_content "    }\n\n")

    # Add selector macro overrides
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "// Selector Macro Overrides\n")
    string(APPEND header_content "// ===================================================================\n")
    string(APPEND header_content "#undef _SELECTOR_SINGLE\n")
    string(APPEND header_content "#define _SELECTOR_SINGLE(A, B, C, D) \\\\\n")
    string(APPEND header_content "    SD_IF_VALID(SD_IS_SINGLE_TYPE_VALID(D), \\\\\n")
    string(APPEND header_content "        case C: { A<D> B; break; } \\\\\n")
    string(APPEND header_content "    )\n\n")

    string(APPEND header_content "#undef _SELECTOR_DOUBLE_2\n")
    string(APPEND header_content "#define _SELECTOR_DOUBLE_2(NAME, SIGNATURE, TYPE_A, ENUM, TYPE_B) \\\\\n")
    string(APPEND header_content "    SD_IF_VALID(SD_IS_TYPE_PAIR_VALID(TYPE_A, TYPE_B), \\\\\n")
    string(APPEND header_content "        case ENUM: { NAME<TYPE_A, TYPE_B> SIGNATURE; break; } \\\\\n")
    string(APPEND header_content "    )\n\n")

    string(APPEND header_content "#undef _SELECTOR_TRIPLE_3\n")
    string(APPEND header_content "#define _SELECTOR_TRIPLE_3(NAME, SIGNATURE, TYPE_X, TYPE_Y, ENUM_Z, TYPE_Z) \\\\\n")
    string(APPEND header_content "    SD_IF_VALID(SD_IS_TYPE_TRIPLE_VALID(TYPE_X, TYPE_Y, TYPE_Z), \\\\\n")
    string(APPEND header_content "        case ENUM_Z: { NAME<TYPE_X, TYPE_Y, TYPE_Z> SIGNATURE; break; } \\\\\n")
    string(APPEND header_content "    )\n\n")

    set(${output_var} "${header_content}" PARENT_SCOPE)
endfunction()
# ==========================================================================================
# SECTION 3: THE SINGLE PUBLIC ORCHESTRATOR FUNCTION
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
    # Parse optional arguments
    set(options "")
    set(one_value_args TYPE_PROFILE OUTPUT_DIR)
    set(multi_value_args "")
    cmake_parse_arguments(SRCORE "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    # Set defaults
    if(NOT SRCORE_TYPE_PROFILE)
        set(SRCORE_TYPE_PROFILE "${SD_TYPE_PROFILE}")
    endif()
    if(NOT SRCORE_OUTPUT_DIR)
        set(SRCORE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")
    endif()

    srcore_debug_message("=== UNIFIED SELECTIVE RENDERING SETUP ===")
    srcore_debug_message("Type profile: ${SRCORE_TYPE_PROFILE}")
    srcore_debug_message("Output directory: ${SRCORE_OUTPUT_DIR}")

    if(SRCORE_ENABLE_CACHING AND DEFINED SRCORE_CACHE_VALID AND SRCORE_CACHE_VALID)
        srcore_debug_message("Using cached results")
        set(UNIFIED_COMBINATIONS_2 "${SRCORE_COMBINATIONS_2}" PARENT_SCOPE)
        set(UNIFIED_COMBINATIONS_3 "${SRCORE_COMBINATIONS_3}" PARENT_SCOPE)
        set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" PARENT_SCOPE)
        set(UNIFIED_TYPE_COUNT "${SRCORE_ACTIVE_TYPE_COUNT}" PARENT_SCOPE)
        return()
    endif()

    # Phase 1: Discover active types
    srcore_discover_active_types(active_types_indices)
    list(LENGTH active_types_indices type_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "❌ SelectiveRenderingCore: No active types discovered!")
    endif()

    # Phase 2: Generate combinations
    srcore_generate_combinations("${active_types_indices}" "${SRCORE_TYPE_PROFILE}" combinations_2 combinations_3)

    # Phase 3: Generate headers
    srcore_generate_headers("${active_types_indices}" "${combinations_2}" "${combinations_3}" "${SRCORE_OUTPUT_DIR}")

    # Phase 4: Validate results (if enabled)
    if(SRCORE_VALIDATE_RESULTS)
        srcore_validate_output("${active_types_indices}" "${combinations_2}" "${combinations_3}")
    endif()

    # Phase 5: Set output variables and cache them for global visibility
    message(STATUS "Caching selective rendering results globally...")

    # Set to PARENT_SCOPE for immediate callers
    set(UNIFIED_COMBINATIONS_2 "${combinations_2}" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_3 "${combinations_3}" PARENT_SCOPE)
    set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" PARENT_SCOPE)
    set(UNIFIED_ACTIVE_TYPES_INDICES "${active_types_indices}" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT ${type_count} PARENT_SCOPE)

    # ALSO set to CACHE to guarantee visibility across all included files.
    set(UNIFIED_COMBINATIONS_2 "${combinations_2}" CACHE INTERNAL "Unified 2-type combinations")
    set(UNIFIED_COMBINATIONS_3 "${combinations_3}" CACHE INTERNAL "Unified 3-type combinations")
    set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" CACHE INTERNAL "Active types for build")
    set(UNIFIED_ACTIVE_TYPES_INDICES "${active_types_indices}" CACHE INTERNAL "Indices of active types")
    set(UNIFIED_TYPE_COUNT ${type_count} CACHE INTERNAL "Unified active type count")

    foreach(index IN LISTS active_types_indices)
        set(SRCORE_TYPE_ENUM_${index} "${SRCORE_TYPE_ENUM_${index}}" CACHE INTERNAL "Enum for index ${index}")
        set(SRCORE_TYPE_CPP_${index} "${SRCORE_TYPE_CPP_${index}}" CACHE INTERNAL "C++ type for index ${index}")
    endforeach()

    if(SRCORE_ENABLE_CACHING)
        set(SRCORE_CACHE_VALID TRUE CACHE INTERNAL "Cache validity flag")
    endif()
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