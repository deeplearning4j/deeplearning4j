# Stage one build artifact without duplicating blocks when source and destination
# share a filesystem. This keeps multi-gigabyte native SDK staging bounded while
# retaining a normal copy fallback for cross-filesystem publishers.
if(NOT DEFINED INPUT_FILE OR INPUT_FILE STREQUAL "")
    message(FATAL_ERROR "INPUT_FILE is required")
endif()
if(NOT DEFINED OUTPUT_FILE OR OUTPUT_FILE STREQUAL "")
    message(FATAL_ERROR "OUTPUT_FILE is required")
endif()
if(NOT EXISTS "${INPUT_FILE}" OR IS_DIRECTORY "${INPUT_FILE}")
    message(FATAL_ERROR "Staging input is not a regular file: ${INPUT_FILE}")
endif()

get_filename_component(_input_real "${INPUT_FILE}" REALPATH)
get_filename_component(_output_parent "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${_output_parent}")

get_filename_component(_output_real "${OUTPUT_FILE}" REALPATH)
if(_input_real STREQUAL _output_real)
    return()
endif()

string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _stage_nonce)
set(_stage_tmp "${OUTPUT_FILE}.stage-${_stage_nonce}")
file(REMOVE "${_stage_tmp}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_hardlink
        "${_input_real}" "${_stage_tmp}"
    RESULT_VARIABLE _hardlink_result
    ERROR_QUIET)

if(NOT _hardlink_result EQUAL 0)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_input_real}" "${_stage_tmp}"
        RESULT_VARIABLE _copy_result)
    if(NOT _copy_result EQUAL 0)
        file(REMOVE "${_stage_tmp}")
        message(FATAL_ERROR
            "Could not stage '${_input_real}' at '${OUTPUT_FILE}': "
            "hard-link result=${_hardlink_result}, copy result=${_copy_result}")
    endif()
endif()

if(NOT EXISTS "${_stage_tmp}")
    message(FATAL_ERROR "Staging did not materialize temporary output: ${_stage_tmp}")
endif()
file(REMOVE "${OUTPUT_FILE}")
file(RENAME "${_stage_tmp}" "${OUTPUT_FILE}")
if(NOT EXISTS "${OUTPUT_FILE}")
    message(FATAL_ERROR "Staging output is missing after publication: ${OUTPUT_FILE}")
endif()
