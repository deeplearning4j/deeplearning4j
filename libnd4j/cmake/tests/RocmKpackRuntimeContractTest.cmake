if(NOT UNIX OR APPLE)
    message(STATUS "ROCm kpack runtime contract is Linux-only")
    return()
endif()

foreach(_required_variable IN ITEMS
        LIBND4J_SOURCE_DIR TEST_BINARY_DIR TEST_CXX_COMPILER TEST_READELF)
    if(NOT DEFINED ${_required_variable} OR
       "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} is required")
    endif()
endforeach()

set(_test_root "${TEST_BINARY_DIR}/rocm-kpack-runtime-contract")
set(_runtime_root "${_test_root}/runtime")
set(_output_root "${_test_root}/output")
set(_package_root "${_output_root}/classifier-runtime")
set(_rocblas_source "${_test_root}/rocblas.cpp")
set(_rocsparse_source "${_test_root}/rocsparse.cpp")
set(_rocblas_runtime "${_runtime_root}/librocblas.so")
set(_rocsparse_runtime "${_runtime_root}/librocsparse.so")
set(_blas_kpack "${_runtime_root}/.kpack/blas_lib_gfx1103.kpack")

file(REMOVE_RECURSE "${_test_root}")
file(MAKE_DIRECTORY "${_runtime_root}/.kpack" "${_output_root}")
file(WRITE "${_rocblas_source}"
    "extern \"C\" int dl4j_rocm_blas_contract() { return 10; }\n")
file(WRITE "${_rocsparse_source}"
    "extern \"C\" int dl4j_rocm_sparse_contract() { return 10; }\n")

foreach(_fixture IN ITEMS rocblas rocsparse)
    execute_process(
        COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
            "-Wl,-soname,lib${_fixture}.so"
            -o "${_runtime_root}/lib${_fixture}.so"
            "${_test_root}/${_fixture}.cpp"
        RESULT_VARIABLE _compile_result
        ERROR_VARIABLE _compile_error)
    if(NOT _compile_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to compile ${_fixture} kpack fixture: ${_compile_error}")
    endif()
endforeach()

file(WRITE "${_blas_kpack}" "gfx1103 optimized BLAS kernels")

set(_stage_command
    "${CMAKE_COMMAND}"
    "-DRUNTIME_LIBRARIES_PIPE=${_rocblas_runtime}|${_rocsparse_runtime}"
    "-DRUNTIME_SEARCH_ROOTS_PIPE=${_runtime_root}"
    "-DOUTPUT_DIR=${_output_root}"
    "-DPACKAGE_DIR=${_package_root}"
    "-DCXX_COMPILER=${TEST_CXX_COMPILER}"
    "-DREADELF=${TEST_READELF}"
    "-DROCM_RESOURCE_FORMAT=kpack"
    "-DROCM_RUNTIME_ARCH=gfx1103"
    "-DROCM_KERNEL_PACK_GROUPS_CSV=blas"
    -P "${LIBND4J_SOURCE_DIR}/cmake/StageSharedRuntime.cmake")

execute_process(
    COMMAND ${_stage_command}
    RESULT_VARIABLE _stage_result
    OUTPUT_VARIABLE _stage_output
    ERROR_VARIABLE _stage_error)
if(NOT _stage_result EQUAL 0)
    message(FATAL_ERROR
        "ROCm kpack staging failed:\n${_stage_output}\n${_stage_error}")
endif()

foreach(_required_pack IN ITEMS ".kpack/blas_lib_gfx1103.kpack")
    if(NOT EXISTS "${_package_root}/${_required_pack}")
        message(FATAL_ERROR
            "Classifier runtime package omitted '${_required_pack}'")
    endif()
endforeach()
foreach(_required_runtime IN ITEMS librocblas.so librocsparse.so)
    if(NOT EXISTS "${_package_root}/${_required_runtime}")
        message(FATAL_ERROR
            "Classifier runtime package omitted '${_required_runtime}'")
    endif()
endforeach()

set(_manifest "${_package_root}/shared-runtime-manifest.txt")
if(NOT EXISTS "${_manifest}")
    message(FATAL_ERROR "ROCm kpack staging did not write a package manifest")
endif()
file(STRINGS "${_manifest}" _manifest_entries)
foreach(_required_resource IN ITEMS
        "# resource=.kpack/blas_lib_gfx1103.kpack")
    list(FIND _manifest_entries "${_required_resource}" _resource_index)
    if(_resource_index EQUAL -1)
        message(FATAL_ERROR
            "ROCm kpack manifest omitted '${_required_resource}': ${_manifest_entries}")
    endif()
endforeach()

# A declared Core SDK kpack contract must not fall back to the legacy Tensile
# path or accept a missing, nested, or renamed signed-package resource.
file(REMOVE "${_blas_kpack}")
execute_process(
    COMMAND ${_stage_command}
    RESULT_VARIABLE _missing_result
    OUTPUT_VARIABLE _missing_output
    ERROR_VARIABLE _missing_error)
if(_missing_result EQUAL 0)
    message(FATAL_ERROR
        "ROCm kpack staging accepted a missing gfx1103 BLAS kernel pack")
endif()
string(CONCAT _missing_diagnostics "${_missing_output}" "${_missing_error}")
string(FIND "${_missing_diagnostics}" "blas kernel pack" _missing_message)
if(_missing_message EQUAL -1)
    message(FATAL_ERROR
        "Missing BLAS kpack failed for the wrong reason:\n${_missing_diagnostics}")
endif()

file(MAKE_DIRECTORY "${_runtime_root}/.kpack/nested")
file(WRITE
    "${_runtime_root}/.kpack/nested/blas_lib_gfx1103.kpack"
    "nested BLAS pack must be rejected")
execute_process(
    COMMAND ${_stage_command}
    RESULT_VARIABLE _nested_result
    OUTPUT_VARIABLE _nested_output
    ERROR_VARIABLE _nested_error)
if(_nested_result EQUAL 0)
    message(FATAL_ERROR
        "ROCm kpack staging accepted a nested gfx1103 BLAS kernel pack")
endif()

file(REMOVE_RECURSE "${_runtime_root}/.kpack/nested")
file(WRITE
    "${_runtime_root}/.kpack/blas_renamed_gfx1103.kpack"
    "renamed BLAS pack must be rejected")
execute_process(
    COMMAND ${_stage_command}
    RESULT_VARIABLE _renamed_result
    OUTPUT_VARIABLE _renamed_output
    ERROR_VARIABLE _renamed_error)
if(_renamed_result EQUAL 0)
    message(FATAL_ERROR
        "ROCm kpack staging accepted a renamed gfx1103 BLAS kernel pack")
endif()

file(REMOVE_RECURSE "${_test_root}")
