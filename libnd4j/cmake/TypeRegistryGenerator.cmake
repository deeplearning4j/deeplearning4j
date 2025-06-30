function(dump_type_macros_to_disk)
    set(output_file "${CMAKE_BINARY_DIR}/resolved_type_macros.txt")
    message(STATUS "🔍 Extracting macro values: ${output_file}")

    # Get include directories from existing target
    set(object_lib_name "${SD_LIBRARY_NAME}_object")
    get_target_property(target_includes "${object_lib_name}" INCLUDE_DIRECTORIES)

    # Build include flags
    set(include_flags "")
    if(target_includes AND NOT target_includes STREQUAL "target_includes-NOTFOUND")
        foreach(include_dir ${target_includes})
            list(APPEND include_flags "-I${include_dir}")
        endforeach()
    endif()

    # Create simple extraction file - just put macros between unique delimiters
    set(extraction_file "${CMAKE_BINARY_DIR}/simple_macro_extraction.cpp")

    file(WRITE "${extraction_file}" "
#include <types/types.h>
#include <loops/pairwise_instantiations.h>

===BEGIN_SD_COMMON_TYPES===
SD_COMMON_TYPES
===END_SD_COMMON_TYPES===

===BEGIN_SD_FLOAT_TYPES===
SD_FLOAT_TYPES
===END_SD_FLOAT_TYPES===

===BEGIN_SD_INTEGER_TYPES===
SD_INTEGER_TYPES
===END_SD_INTEGER_TYPES===

===BEGIN_SD_NUMERIC_TYPES===
SD_NUMERIC_TYPES
===END_SD_NUMERIC_TYPES===

===BEGIN_SD_INDEXING_TYPES===
SD_INDEXING_TYPES
===END_SD_INDEXING_TYPES===

===BEGIN_SD_LONG_TYPES===
SD_LONG_TYPES
===END_SD_LONG_TYPES===

===BEGIN_SD_BOOL_TYPES===
SD_BOOL_TYPES
===END_SD_BOOL_TYPES===

===BEGIN_SD_STRING_TYPES===
SD_STRING_TYPES
===END_SD_STRING_TYPES===

===BEGIN_SD_GENERIC_NUMERIC_TYPES===
SD_GENERIC_NUMERIC_TYPES
===END_SD_GENERIC_NUMERIC_TYPES===

===BEGIN_SD_COMMON_TYPES_EXTENDED===
SD_COMMON_TYPES_EXTENDED
===END_SD_COMMON_TYPES_EXTENDED===

===BEGIN_SD_NATIVE_FLOAT_TYPES===
SD_NATIVE_FLOAT_TYPES
===END_SD_NATIVE_FLOAT_TYPES===

===BEGIN_SD_COMMON_TYPES_ALL===
SD_COMMON_TYPES_ALL
===END_SD_COMMON_TYPES_ALL===

===BEGIN_SD_COMMON_TYPES_PART_0===
SD_COMMON_TYPES_PART_0
===END_SD_COMMON_TYPES_PART_0===

===BEGIN_SD_COMMON_TYPES_PART_1===
SD_COMMON_TYPES_PART_1
===END_SD_COMMON_TYPES_PART_1===

===BEGIN_SD_COMMON_TYPES_PART_2===
SD_COMMON_TYPES_PART_2
===END_SD_COMMON_TYPES_PART_2===

===BEGIN_SD_NUMERIC_TYPES_PART_0===
SD_NUMERIC_TYPES_PART_0
===END_SD_NUMERIC_TYPES_PART_0===

===BEGIN_SD_NUMERIC_TYPES_PART_1===
SD_NUMERIC_TYPES_PART_1
===END_SD_NUMERIC_TYPES_PART_1===

===BEGIN_SD_NUMERIC_TYPES_PART_2===
SD_NUMERIC_TYPES_PART_2
===END_SD_NUMERIC_TYPES_PART_2===

===BEGIN_TTYPE_BOOL===
TTYPE_BOOL
===END_TTYPE_BOOL===

===BEGIN_TTYPE_FLOAT32===
TTYPE_FLOAT32
===END_TTYPE_FLOAT32===

===BEGIN_TTYPE_DOUBLE===
TTYPE_DOUBLE
===END_TTYPE_DOUBLE===

===BEGIN_TTYPE_HALF===
TTYPE_HALF
===END_TTYPE_HALF===

===BEGIN_TTYPE_INT32===
TTYPE_INT32
===END_TTYPE_INT32===

===BEGIN_TTYPE_INT64===
TTYPE_INT64
===END_TTYPE_INT64===
")

    # Run preprocessor
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -E
            ${include_flags}
            -std=c++11
            "${extraction_file}"
            OUTPUT_VARIABLE preprocessed_output
            ERROR_VARIABLE compiler_errors
            RESULT_VARIABLE compiler_result
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Clean up temp file
    file(REMOVE "${extraction_file}")

    # Initialize output
    set(output_content "")
    string(APPEND output_content "RESOLVED MACRO VALUES\n")
    string(APPEND output_content "=====================\n\n")
    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND output_content "Generated: ${current_time}\n")
    string(APPEND output_content "Compiler: ${CMAKE_CXX_COMPILER}\n")
    if(SD_CUDA)
        string(APPEND output_content "Platform: CUDA\n")
    else()
        string(APPEND output_content "Platform: CPU\n")
    endif()
    string(APPEND output_content "\n")

    if(compiler_result EQUAL 0 AND preprocessed_output)
        message(STATUS "   ✅ Preprocessor succeeded")

        # Extract each macro using simple delimiters
        set(macro_names
                "SD_COMMON_TYPES"
                "SD_FLOAT_TYPES"
                "SD_INTEGER_TYPES"
                "SD_NUMERIC_TYPES"
                "SD_INDEXING_TYPES"
                "SD_LONG_TYPES"
                "SD_BOOL_TYPES"
                "SD_STRING_TYPES"
                "SD_GENERIC_NUMERIC_TYPES"
                "SD_COMMON_TYPES_EXTENDED"
                "SD_NATIVE_FLOAT_TYPES"
                "SD_COMMON_TYPES_ALL"
                "SD_COMMON_TYPES_PART_0"
                "SD_COMMON_TYPES_PART_1"
                "SD_COMMON_TYPES_PART_2"
                "SD_NUMERIC_TYPES_PART_0"
                "SD_NUMERIC_TYPES_PART_1"
                "SD_NUMERIC_TYPES_PART_2"
                "TTYPE_BOOL"
                "TTYPE_FLOAT32"
                "TTYPE_DOUBLE"
                "TTYPE_HALF"
                "TTYPE_INT32"
                "TTYPE_INT64"
        )

        set(found_macros 0)
        foreach(macro_name ${macro_names})
            # Look for content between delimiters
            set(begin_marker "===BEGIN_${macro_name}===")
            set(end_marker "===END_${macro_name}===")

            string(FIND "${preprocessed_output}" "${begin_marker}" begin_pos)
            string(FIND "${preprocessed_output}" "${end_marker}" end_pos)

            if(begin_pos GREATER -1 AND end_pos GREATER begin_pos)
                string(LENGTH "${begin_marker}" begin_len)
                math(EXPR content_start "${begin_pos} + ${begin_len}")
                math(EXPR content_length "${end_pos} - ${content_start}")

                if(content_length GREATER 0)
                    string(SUBSTRING "${preprocessed_output}" ${content_start} ${content_length} macro_value)

                    # Clean up the value
                    string(STRIP "${macro_value}" macro_value)
                    string(REGEX REPLACE "[\r\n]+" " " macro_value "${macro_value}")
                    string(REGEX REPLACE "  +" " " macro_value "${macro_value}")
                    string(REGEX REPLACE "^, " "" macro_value "${macro_value}")

                    # Check if macro was actually expanded (not just echoed back)
                    if(NOT macro_value STREQUAL "" AND NOT macro_value STREQUAL macro_name)
                        string(APPEND output_content "${macro_name}:\n")
                        string(APPEND output_content "  ${macro_value}\n\n")
                        math(EXPR found_macros "${found_macros} + 1")
                    else()
                        string(APPEND output_content "${macro_name}: (undefined or not expanded)\n\n")
                    endif()
                else()
                    string(APPEND output_content "${macro_name}: (empty content)\n\n")
                endif()
            else()
                string(APPEND output_content "${macro_name}: (delimiters not found)\n\n")
            endif()
        endforeach()

        message(STATUS "   📊 Found ${found_macros} expanded macros")

        # Add build statistics
        if(DEFINED UNIFIED_ACTIVE_TYPES)
            string(APPEND output_content "BUILD STATISTICS:\n")
            string(APPEND output_content "=================\n")
            list(LENGTH UNIFIED_ACTIVE_TYPES active_count)
            string(APPEND output_content "Active types: ${active_count}\n")
            string(APPEND output_content "Type list: ${UNIFIED_ACTIVE_TYPES}\n")

            if(DEFINED UNIFIED_COMBINATIONS_3)
                list(LENGTH UNIFIED_COMBINATIONS_3 combo_count)
                string(APPEND output_content "3-type combinations: ${combo_count}\n")
            endif()
            string(APPEND output_content "\n")
        endif()

    else()
        string(APPEND output_content "ERROR: Preprocessor failed\n")
        string(APPEND output_content "Result: ${compiler_result}\n")
        if(compiler_errors)
            string(APPEND output_content "Errors:\n${compiler_errors}\n")
        endif()
    endif()

    # Write results
    file(WRITE "${output_file}" "${output_content}")
    message(STATUS "✅ Extraction complete: ${output_file}")

    # Quick summary to console
    if(compiler_result EQUAL 0)
        if("${output_content}" MATCHES "SD_COMMON_TYPES_PART_0:[\r\n]+  ([^\r\n]+)")
            message(STATUS "   Found SD_COMMON_TYPES_PART_0: ${CMAKE_MATCH_1}")
        else()
            message(STATUS "   ❌ SD_COMMON_TYPES_PART_0 not expanded")
        endif()
    endif()

endfunction()