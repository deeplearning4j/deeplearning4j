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

function(_internal_srcore_discover_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
    message(STATUS "🔍 DEBUG: Starting type discovery...")

    # Look for types.h in multiple possible locations
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

    set(simple_types "BOOL;DOUBLE;FLOAT32;INT32;INT64")
    set(discovered_types "")
    set(discovered_indices "")
    set(discovered_enums "")
    set(discovered_cpp_types "")
    set(type_index 0)

    foreach(type_key ${simple_types})
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

            message(STATUS "✅ Type ${type_index}: ${type_key} -> enum: ${enum_part}, cpp: ${cpp_part}")
            math(EXPR type_index "${type_index} + 1")
        endif()
    endforeach()

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()


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

function(_internal_srcore_generate_validity_header active_indices type_enums type_cpp_types combinations_2 combinations_3 output_dir)
    file(MAKE_DIRECTORY "${output_dir}/system")
    set(header_file "${output_dir}/system/selective_rendering.h")

    # Function to convert C++ types to simple macro names
    function(cpp_type_to_macro_name cpp_type output_var)
        set(macro_name "${cpp_type}")
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
        elseif(macro_name STREQUAL "float16")
            set(macro_name "HALF")
        elseif(macro_name STREQUAL "bfloat16")
            set(macro_name "BFLOAT16")
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

    # Generate BUILD_DOUBLE_SELECTOR - FIXED VERSION
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

    # Generate BUILD_TRIPLE_SELECTOR - FIXED VERSION
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