function(dump_type_macros_to_disk)
    set(output_file "${CMAKE_BINARY_DIR}/resolved_type_macros.txt")
    message(STATUS "🔍 Extracting macro values: ${output_file}")

    # ========================================================================
    # Generate config.h BEFORE extraction (required by op_boilerplate.h)
    # This is the same logic as PostBuild.cmake but runs earlier
    # ========================================================================
    set(CONFIG_H_IN "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in")
    set(CONFIG_H_OUT "${CMAKE_CURRENT_BINARY_DIR}/include/config.h")

    # Ensure include directory exists
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

    # Read the template
    file(READ "${CONFIG_H_IN}" CONFIG_H_CONTENT)

    # Build TYPE_DEFINES for config.h
    set(TYPE_DEFINES "")
    if(DEFINED SD_TYPES_LIST AND NOT SD_TYPES_LIST STREQUAL "")
        # Selective types mode - export only enabled types
        get_directory_property(COMPILE_DEFS COMPILE_DEFINITIONS)
        foreach(def IN LISTS COMPILE_DEFS)
            if(def MATCHES "^HAS_")
                string(APPEND TYPE_DEFINES "#define ${def}\n")
            endif()
        endforeach()
    else()
        # All types mode - define all HAS_* macros
        string(APPEND TYPE_DEFINES "#define HAS_BOOL 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_FLOAT 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_FLOAT32 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_DOUBLE 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_FLOAT64 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_FLOAT16 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_HALF 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_BFLOAT16 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_BFLOAT 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_FLOAT8 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_HALF2 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT8 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT8_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT16 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT16_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT32 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT32_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT64 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_INT64_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_LONG 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT8 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT8_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT16 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT16_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT32 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT32_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT64 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UINT64_T 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UNSIGNEDLONG 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UTF8 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_STD_STRING 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UTF16 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_STD_U16STRING 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_UTF32 1\n")
        string(APPEND TYPE_DEFINES "#define HAS_STD_U32STRING 1\n")
    endif()

    # Perform variable substitutions
    if(HAVE_ONEDNN)
        string(REPLACE "#cmakedefine01 HAVE_ONEDNN" "#define HAVE_ONEDNN 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_ONEDNN" "#define HAVE_ONEDNN 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_ARMCOMPUTE)
        string(REPLACE "#cmakedefine01 HAVE_ARMCOMPUTE" "#define HAVE_ARMCOMPUTE 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_ARMCOMPUTE" "#define HAVE_ARMCOMPUTE 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_CUDNN)
        string(REPLACE "#cmakedefine01 HAVE_CUDNN" "#define HAVE_CUDNN 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_CUDNN" "#define HAVE_CUDNN 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_OPENBLAS)
        string(REPLACE "#cmakedefine01 HAVE_OPENBLAS" "#define HAVE_OPENBLAS 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_OPENBLAS" "#define HAVE_OPENBLAS 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_FLATBUFFERS)
        string(REPLACE "#cmakedefine01 HAVE_FLATBUFFERS" "#define HAVE_FLATBUFFERS 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_FLATBUFFERS" "#define HAVE_FLATBUFFERS 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    # HAVE_TRITON_CPU must be replaced BEFORE HAVE_TRITON to avoid substring match
    if(HAVE_TRITON_CPU)
        string(REPLACE "#cmakedefine01 HAVE_TRITON_CPU" "#define HAVE_TRITON_CPU 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_TRITON_CPU" "#define HAVE_TRITON_CPU 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_TRITON)
        string(REPLACE "#cmakedefine01 HAVE_TRITON" "#define HAVE_TRITON 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_TRITON" "#define HAVE_TRITON 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_MLIR)
        string(REPLACE "#cmakedefine01 HAVE_MLIR" "#define HAVE_MLIR 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_MLIR" "#define HAVE_MLIR 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_MLX)
        string(REPLACE "#cmakedefine01 HAVE_MLX" "#define HAVE_MLX 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_MLX" "#define HAVE_MLX 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_NNAPI)
        string(REPLACE "#cmakedefine01 HAVE_NNAPI" "#define HAVE_NNAPI 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_NNAPI" "#define HAVE_NNAPI 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_CUTLASS)
        string(REPLACE "#cmakedefine01 HAVE_CUTLASS" "#define HAVE_CUTLASS 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_CUTLASS" "#define HAVE_CUTLASS 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(HAVE_OPENVINO)
        string(REPLACE "#cmakedefine01 HAVE_OPENVINO" "#define HAVE_OPENVINO 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 HAVE_OPENVINO" "#define HAVE_OPENVINO 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(SD_SELECTIVE_TYPES)
        string(REPLACE "#cmakedefine01 SD_SELECTIVE_TYPES" "#define SD_SELECTIVE_TYPES 1" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REPLACE "#cmakedefine01 SD_SELECTIVE_TYPES" "#define SD_SELECTIVE_TYPES 0" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    # Handle @VAR@ substitutions
    if(DEFINED SD_LIBRARY_NAME)
        string(REPLACE "@SD_LIBRARY_NAME@" "${SD_LIBRARY_NAME}" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REGEX REPLACE "#cmakedefine SD_LIBRARY_NAME[^\n]*\n" "" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(DEFINED ONEDNN_PATH)
        string(REPLACE "@ONEDNN_PATH@" "${ONEDNN_PATH}" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REGEX REPLACE "#cmakedefine ONEDNN_PATH[^\n]*\n" "" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(DEFINED OPENBLAS_PATH)
        string(REPLACE "@OPENBLAS_PATH@" "${OPENBLAS_PATH}" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REGEX REPLACE "#cmakedefine OPENBLAS_PATH[^\n]*\n" "" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(DEFINED FLATBUFFERS_PATH)
        string(REPLACE "@FLATBUFFERS_PATH@" "${FLATBUFFERS_PATH}" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REGEX REPLACE "#cmakedefine FLATBUFFERS_PATH[^\n]*\n" "" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    if(DEFINED DEFAULT_ENGINE)
        string(REPLACE "@DEFAULT_ENGINE@" "${DEFAULT_ENGINE}" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    else()
        string(REGEX REPLACE "#cmakedefine DEFAULT_ENGINE[^\n]*\n" "" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")
    endif()

    # Substitute TYPE_DEFINES
    string(REPLACE "@TYPE_DEFINES@" "${TYPE_DEFINES}" CONFIG_H_CONTENT "${CONFIG_H_CONTENT}")

    # Write config.h
    file(WRITE "${CONFIG_H_OUT}" "${CONFIG_H_CONTENT}")
    message(STATUS "✅ Generated config.h: ${CONFIG_H_OUT}")

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
    
    # Determine SD_CUDA value for extraction
    # IMPORTANT: We must UNDEFINE SD_CUDA (not set to 0) for CPU builds because
    # op_boilerplate.h uses #if defined(SD_CUDA) which is true even when SD_CUDA=0
    if(SD_CUDA)
        set(SD_CUDA_DEFINE "#define SD_CUDA 1")
    else()
        set(SD_CUDA_DEFINE "// SD_CUDA not defined (CPU build)")
    endif()
    
    file(WRITE "${extraction_file}" "#include \"${stub_header}\"
${SD_CUDA_DEFINE}
#define HAVE_ONEDNN 0
#define HAVE_ARMCOMPUTE 0
#define HAVE_CUDNN 0
#define COUNT_NARG(...) 1

#include <types/types.h>

===BEGIN_SD_COMMON_TYPES===
SD_COMMON_TYPES
===END_SD_COMMON_TYPES===

===BEGIN_SD_COMMON_TYPES_ALL===
SD_COMMON_TYPES_ALL
===END_SD_COMMON_TYPES_ALL===

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
    # IMPORTANT: Do NOT pass -DSD_CUDA=0 here because #if defined(SD_CUDA) checks
    # if the macro exists, not its value. For CPU builds, SD_CUDA should not be defined at all.
    if(SD_CUDA)
        set(extractor_cuda_defines "-DSD_CUDA=1")
    else()
        set(extractor_cuda_defines "")
    endif()
    
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -E -P -w -std=c++11
            ${include_flags}
            ${extractor_cuda_defines} -DHAVE_ONEDNN=0 -DHAVE_ARMCOMPUTE=0
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
        set(macro_names "SD_COMMON_TYPES" "SD_COMMON_TYPES_ALL" "SD_FLOAT_TYPES" "SD_INTEGER_TYPES" "SD_NUMERIC_TYPES"
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