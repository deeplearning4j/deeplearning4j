function(dump_type_macros_to_disk)
    set(output_file "${CMAKE_BINARY_DIR}/resolved_type_macros.txt")
    message(STATUS "🔍 Extracting macro values: ${output_file}")

    # Create a stub header that blocks CUDA includes
    set(stub_header "${CMAKE_BINARY_DIR}/cuda_stub.h")
    file(WRITE "${stub_header}" "#ifndef CUDA_STUB_H
#define CUDA_STUB_H
#define __CUDA_RUNTIME_H__
#define __CUDA_H__
#define __DRIVER_TYPES_H__
struct cudaStream_t {};
struct cublasHandle_t {};
#endif")

    # Create minimal extraction file
    set(extraction_file "${CMAKE_BINARY_DIR}/extract_types.cpp")
    file(WRITE "${extraction_file}" "#include \"${stub_header}\"
#define SD_CUDA 0
#define HAVE_ONEDNN 0
#define HAVE_ARMCOMPUTE 0
#define HAVE_CUDNN 0
#define COUNT_NARG(...) 1

#include <types/types.h>

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

===BEGIN_SD_LONG_TYPES===
SD_LONG_TYPES
===END_SD_LONG_TYPES===

===BEGIN_SD_COMMON_TYPES_PART_0===
SD_COMMON_TYPES_PART_0
===END_SD_COMMON_TYPES_PART_0===

===BEGIN_SD_COMMON_TYPES_PART_1===
SD_COMMON_TYPES_PART_1
===END_SD_COMMON_TYPES_PART_1===

===BEGIN_SD_COMMON_TYPES_PART_2===
SD_COMMON_TYPES_PART_2
===END_SD_COMMON_TYPES_PART_2===

===BEGIN_SD_STRING_TYPES===
SD_STRING_TYPES
===END_SD_STRING_TYPES===")

    # Get basic include paths
    set(include_flags
            "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-I${CMAKE_CURRENT_BINARY_DIR}/include"
            "-I${CMAKE_CURRENT_BINARY_DIR}"
    )

    # Add FlatBuffers if available
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
        list(APPEND include_flags "-I${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-src/include")
    endif()

    # Run preprocessor
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -E -P -w -std=c++11
            ${include_flags}
            -DSD_CUDA=0 -DHAVE_ONEDNN=0 -DHAVE_ARMCOMPUTE=0
            "${extraction_file}"
            OUTPUT_VARIABLE preprocessed_output
            ERROR_VARIABLE compiler_errors
            RESULT_VARIABLE compiler_result
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Initialize output
    set(output_content "RESOLVED MACRO VALUES\n=====================\n\n")
    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND output_content "Generated: ${current_time}\n")
    string(APPEND output_content "Compiler: ${CMAKE_CXX_COMPILER}\n\n")

    if(compiler_result EQUAL 0 AND preprocessed_output)
        message(STATUS "✅ Macro extraction succeeded")

        # Extract macros
        set(macro_names "SD_COMMON_TYPES" "SD_FLOAT_TYPES" "SD_INTEGER_TYPES" "SD_NUMERIC_TYPES"
                "SD_LONG_TYPES" "SD_COMMON_TYPES_PART_0" "SD_COMMON_TYPES_PART_1" "SD_COMMON_TYPES_PART_2" "SD_STRING_TYPES")

        set(found_macros 0)
        foreach(macro_name ${macro_names})
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
                    string(STRIP "${macro_value}" macro_value)
                    string(REGEX REPLACE "[\r\n]+" " " macro_value "${macro_value}")
                    string(REGEX REPLACE "  +" " " macro_value "${macro_value}")

                    if(NOT macro_value STREQUAL "" AND NOT macro_value STREQUAL macro_name)
                        string(APPEND output_content "${macro_name}:\n  ${macro_value}\n\n")
                        math(EXPR found_macros "${found_macros} + 1")
                    else()
                        string(APPEND output_content "${macro_name}: (undefined)\n\n")
                    endif()
                endif()
            else()
                string(APPEND output_content "${macro_name}: (not found)\n\n")
            endif()
        endforeach()

        message(STATUS "📊 Found ${found_macros} expanded macros")
    else()
        string(APPEND output_content "ERROR: Preprocessor failed\nResult: ${compiler_result}\n")
        if(compiler_errors)
            string(APPEND output_content "Errors:\n${compiler_errors}\n")
        endif()
        message(STATUS "❌ Macro extraction failed")
    endif()

    # Write results
    file(WRITE "${output_file}" "${output_content}")
    message(STATUS "✅ Results written to: ${output_file}")

    # Cleanup
    file(REMOVE "${extraction_file}" "${stub_header}")

endfunction()