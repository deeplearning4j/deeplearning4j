if(WIN32)
    message(STATUS "Shared-runtime symlink alias contract is Unix-only")
    return()
endif()

foreach(_required_variable IN ITEMS
        LIBND4J_SOURCE_DIR TEST_BINARY_DIR TEST_CXX_COMPILER TEST_READELF)
    if(NOT DEFINED ${_required_variable} OR
       "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} is required")
    endif()
endforeach()

set(_test_root "${TEST_BINARY_DIR}/shared-runtime-alias-contract")
set(_runtime_root "${_test_root}/runtime")
set(_output_root "${_test_root}/output")
set(_source_file "${_test_root}/nvcuda.cpp")
set(_real_runtime "${_runtime_root}/libnvcuda.so")
set(_cuda_alias "${_runtime_root}/libcuda.so")

file(REMOVE_RECURSE "${_test_root}")
file(MAKE_DIRECTORY "${_runtime_root}" "${_output_root}")
file(WRITE "${_source_file}" "extern \"C\" int dl4j_zluda_alias_contract() { return 0; }\n")

execute_process(
    COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
        -Wl,-soname,libnvcuda.so
        -o "${_real_runtime}" "${_source_file}"
    RESULT_VARIABLE _compile_result
    ERROR_VARIABLE _compile_error)
if(NOT _compile_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to compile shared-runtime alias fixture: ${_compile_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "libnvcuda.so" "${_cuda_alias}"
    RESULT_VARIABLE _symlink_result
    ERROR_VARIABLE _symlink_error)
if(NOT _symlink_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to create shared-runtime alias fixture: ${_symlink_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DRUNTIME_LIBRARIES_PIPE=${_cuda_alias}"
        "-DRUNTIME_SEARCH_ROOTS_PIPE=${_runtime_root}"
        "-DOUTPUT_DIR=${_output_root}"
        "-DCXX_COMPILER=${TEST_CXX_COMPILER}"
        "-DREADELF=${TEST_READELF}"
        -P "${LIBND4J_SOURCE_DIR}/cmake/StageSharedRuntime.cmake"
    RESULT_VARIABLE _stage_result
    OUTPUT_VARIABLE _stage_output
    ERROR_VARIABLE _stage_error)
if(NOT _stage_result EQUAL 0)
    message(FATAL_ERROR
        "Shared-runtime alias staging failed:\n${_stage_output}\n${_stage_error}")
endif()

set(_manifest "${_output_root}/shared-runtime-manifest.txt")
if(NOT EXISTS "${_manifest}")
    message(FATAL_ERROR "Shared-runtime alias staging did not write a manifest")
endif()
file(STRINGS "${_manifest}" _manifest_entries)
foreach(_required_runtime IN ITEMS libnvcuda.so libcuda.so)
    list(FIND _manifest_entries "${_required_runtime}" _runtime_index)
    if(_runtime_index EQUAL -1)
        message(FATAL_ERROR
            "Shared-runtime manifest omitted '${_required_runtime}': ${_manifest_entries}")
    endif()
    if(NOT EXISTS "${_output_root}/${_required_runtime}")
        message(FATAL_ERROR
            "Shared-runtime staging omitted '${_required_runtime}'")
    endif()
endforeach()

message(STATUS "Shared-runtime symlink aliases are preserved before canonical deduplication")
