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
set(_package_root "${_output_root}/classifier-runtime")
set(_source_file "${_test_root}/nvcuda.cpp")
set(_real_runtime "${_runtime_root}/libnvcuda.so")
set(_cuda_alias "${_runtime_root}/libcuda.so")
set(_no_soname_source "${_test_root}/cublas.cpp")
set(_no_soname_runtime "${_runtime_root}/libcublas.so")

file(REMOVE_RECURSE "${_test_root}")
file(MAKE_DIRECTORY "${_runtime_root}" "${_output_root}")
file(WRITE "${_source_file}" "extern \"C\" int dl4j_zluda_alias_contract() { return 0; }\n")
file(WRITE "${_no_soname_source}"
    "extern \"C\" int dl4j_zluda_no_soname_contract() { return 0; }\n")

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
    COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
        -o "${_no_soname_runtime}" "${_no_soname_source}"
    RESULT_VARIABLE _no_soname_compile_result
    ERROR_VARIABLE _no_soname_compile_error)
if(NOT _no_soname_compile_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to compile no-SONAME runtime fixture: ${_no_soname_compile_error}")
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
        "-DRUNTIME_LIBRARIES_PIPE=${_cuda_alias}|${_no_soname_runtime}"
        "-DRUNTIME_SEARCH_ROOTS_PIPE=${_runtime_root}"
        "-DOUTPUT_DIR=${_output_root}"
        "-DPACKAGE_DIR=${_package_root}"
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
foreach(_required_runtime IN ITEMS libnvcuda.so libcublas.so)
    list(FIND _manifest_entries "${_required_runtime}" _runtime_index)
    if(_runtime_index EQUAL -1)
        message(FATAL_ERROR
            "Shared-runtime manifest omitted canonical runtime "
            "'${_required_runtime}': ${_manifest_entries}")
    endif()
endforeach()
list(FIND _manifest_entries "libcuda.so" _alias_manifest_index)
if(NOT _alias_manifest_index EQUAL -1)
    message(FATAL_ERROR
        "Shared-runtime preload manifest contains package-only alias "
        "'libcuda.so': ${_manifest_entries}")
endif()
list(FIND _manifest_entries
    "# runtime-alias=libcuda.so->libnvcuda.so"
    _alias_mapping_index)
list(FIND _manifest_entries "# runtime-alias-count=1"
    _alias_count_index)
if(_alias_mapping_index EQUAL -1 OR _alias_count_index EQUAL -1)
    message(FATAL_ERROR
        "Shared-runtime manifest omitted the non-preloaded libcuda alias "
        "mapping/count: ${_manifest_entries}")
endif()
foreach(_required_runtime IN ITEMS libnvcuda.so libcuda.so libcublas.so)
    if(NOT EXISTS "${_output_root}/${_required_runtime}")
        message(FATAL_ERROR
            "Shared-runtime staging omitted '${_required_runtime}'")
    endif()
    if(NOT EXISTS "${_package_root}/${_required_runtime}")
        message(FATAL_ERROR
            "Classifier runtime package omitted '${_required_runtime}'")
    endif()
endforeach()
if(NOT EXISTS "${_package_root}/shared-runtime-manifest.txt")
    message(FATAL_ERROR
        "Classifier runtime package omitted shared-runtime-manifest.txt")
endif()

# The linked backend, rather than the caller's seed ordering, must be the root of
# the managed dependency walk. The mixed-case MIOpen-style SONAME also verifies
# that the POSIX closure audit preserves case while classifying runtime families
# case-insensitively.
set(_managed_source "${_test_root}/MIOpenContract.cpp")
set(_managed_runtime "${_runtime_root}/libMIOpenContract-concrete.so.7")
set(_managed_link_alias "${_runtime_root}/libMIOpenContract.so")
set(_selected_hsakmt_source "${_test_root}/hsakmt.cpp")
set(_selected_hsakmt_runtime "${_runtime_root}/libhsakmt.so.1.99")
set(_selected_hsakmt_link_alias "${_runtime_root}/libhsakmt.so")
set(_selected_hsa_source "${_test_root}/hsa-runtime.cpp")
set(_selected_hsa_runtime "${_runtime_root}/libhsa-runtime64.so.1.99")
set(_selected_hsa_link_alias "${_runtime_root}/libhsa-runtime64.so")
set(_primary_source "${_test_root}/primary.cpp")
set(_primary_runtime "${_test_root}/libprimary.so")
set(_primary_output "${_test_root}/primary-output")
set(_primary_package "${_primary_output}/classifier-runtime")
set(_primary_runtime_policy "default")
set(_primary_patchelf "")
if(DEFINED TEST_PATCHELF AND
   NOT TEST_PATCHELF STREQUAL "" AND
   EXISTS "${TEST_PATCHELF}")
    set(_primary_runtime_policy "zluda-amd")
    set(_primary_patchelf "${TEST_PATCHELF}")
endif()
file(WRITE "${_managed_source}"
    "extern \"C\" int dl4j_managed_runtime_contract() { return 7; }\n")
file(WRITE "${_selected_hsakmt_source}"
    "extern \"C\" int dl4j_selected_hsakmt_contract() { return 5; }\n")
file(WRITE "${_selected_hsa_source}"
    "extern \"C\" int dl4j_selected_hsakmt_contract();\n"
    "extern \"C\" int dl4j_selected_hsa_contract() { return dl4j_selected_hsakmt_contract() + 11; }\n")
file(WRITE "${_primary_source}"
    "extern \"C\" int dl4j_managed_runtime_contract();\n"
    "extern \"C\" int dl4j_selected_hsa_contract();\n"
    "extern \"C\" int dl4j_primary_contract() { return dl4j_managed_runtime_contract() + dl4j_selected_hsa_contract(); }\n")
execute_process(
    COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
        -Wl,-soname,libMIOpenContract.so.1
        -o "${_managed_runtime}" "${_managed_source}"
    RESULT_VARIABLE _managed_compile_result
    ERROR_VARIABLE _managed_compile_error)
if(NOT _managed_compile_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to compile managed-runtime fixture: ${_managed_compile_error}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "libMIOpenContract-concrete.so.7" "${_managed_link_alias}"
    RESULT_VARIABLE _managed_symlink_result
    ERROR_VARIABLE _managed_symlink_error)
if(NOT _managed_symlink_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to create managed-runtime link alias: ${_managed_symlink_error}")
endif()
execute_process(
    COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
        -Wl,-soname,libhsakmt.so.1
        -o "${_selected_hsakmt_runtime}" "${_selected_hsakmt_source}"
    RESULT_VARIABLE _selected_hsakmt_compile_result
    ERROR_VARIABLE _selected_hsakmt_compile_error)
if(NOT _selected_hsakmt_compile_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to compile selected-ROCm HSAKMT fixture: ${_selected_hsakmt_compile_error}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "libhsakmt.so.1.99" "${_selected_hsakmt_link_alias}"
    RESULT_VARIABLE _selected_hsakmt_symlink_result
    ERROR_VARIABLE _selected_hsakmt_symlink_error)
if(NOT _selected_hsakmt_symlink_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to create selected-ROCm HSAKMT link alias: ${_selected_hsakmt_symlink_error}")
endif()
execute_process(
    COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
        -Wl,-soname,libhsa-runtime64.so.1
        "-L${_runtime_root}" -lhsakmt
        -o "${_selected_hsa_runtime}" "${_selected_hsa_source}"
    RESULT_VARIABLE _selected_hsa_compile_result
    ERROR_VARIABLE _selected_hsa_compile_error)
if(NOT _selected_hsa_compile_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to compile selected-ROCm HSA fixture: ${_selected_hsa_compile_error}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "libhsa-runtime64.so.1.99" "${_selected_hsa_link_alias}"
    RESULT_VARIABLE _selected_hsa_symlink_result
    ERROR_VARIABLE _selected_hsa_symlink_error)
if(NOT _selected_hsa_symlink_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to create selected-ROCm HSA link alias: ${_selected_hsa_symlink_error}")
endif()
execute_process(
    COMMAND "${TEST_CXX_COMPILER}" -shared -fPIC
        -Wl,-soname,libprimary.so
        -Wl,--no-as-needed
        "-L${_runtime_root}" "-Wl,-rpath-link,${_runtime_root}"
        -lMIOpenContract -lhsa-runtime64
        -o "${_primary_runtime}" "${_primary_source}"
    RESULT_VARIABLE _primary_compile_result
    ERROR_VARIABLE _primary_compile_error)
if(NOT _primary_compile_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to compile primary-runtime fixture: ${_primary_compile_error}")
endif()
if(_primary_runtime_policy STREQUAL "zluda-amd")
    # Model ZLUDA's compatibility-patched binaries: the DSO requests a
    # development alias even though the selected dependency has a canonical
    # SONAME. Staging must rewrite this back to the canonical identity.
    execute_process(
        COMMAND "${TEST_PATCHELF}" --replace-needed
            libMIOpenContract.so.1 libMIOpenContract.so
            "${_primary_runtime}"
        RESULT_VARIABLE _fixture_alias_result
        ERROR_VARIABLE _fixture_alias_error)
    if(NOT _fixture_alias_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to create aliased DT_NEEDED fixture: ${_fixture_alias_error}")
    endif()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DRUNTIME_LIBRARIES_PIPE="
        "-DRUNTIME_SEARCH_ROOTS_PIPE=${_runtime_root}"
        "-DPRIMARY_RUNTIME=${_primary_runtime}"
        "-DRUNTIME_POLICY=${_primary_runtime_policy}"
        "-DOUTPUT_DIR=${_primary_output}"
        "-DPACKAGE_DIR=${_primary_package}"
        "-DPATCHELF_EXECUTABLE=${_primary_patchelf}"
        "-DCXX_COMPILER=${TEST_CXX_COMPILER}"
        "-DREADELF=${TEST_READELF}"
        -P "${LIBND4J_SOURCE_DIR}/cmake/StageSharedRuntime.cmake"
    RESULT_VARIABLE _primary_stage_result
    OUTPUT_VARIABLE _primary_stage_output
    ERROR_VARIABLE _primary_stage_error)
if(NOT _primary_stage_result EQUAL 0)
    message(FATAL_ERROR
        "Primary-root closure staging failed:\n"
        "${_primary_stage_output}\n${_primary_stage_error}")
endif()
file(STRINGS "${_primary_output}/shared-runtime-manifest.txt"
    _primary_manifest_entries)
list(FIND _primary_manifest_entries "libMIOpenContract.so.1"
    _managed_runtime_index)
list(FIND _primary_manifest_entries "libMIOpenContract.so"
    _managed_alias_index)
if(_managed_runtime_index EQUAL -1 OR
   NOT _managed_alias_index EQUAL -1 OR
   NOT EXISTS "${_primary_output}/libMIOpenContract.so.1")
    message(FATAL_ERROR
        "Primary-root preload closure did not isolate the canonical "
        "libMIOpenContract.so.1 identity: ${_primary_manifest_entries}")
endif()
list(FIND _primary_manifest_entries "libhsa-runtime64.so.1"
    _selected_hsa_manifest_index)
list(FIND _primary_manifest_entries "libhsa-runtime64.so"
    _selected_hsa_alias_index)
if(_selected_hsa_manifest_index EQUAL -1 OR
   NOT _selected_hsa_alias_index EQUAL -1 OR
   NOT EXISTS "${_primary_output}/libhsa-runtime64.so.1" OR
   NOT EXISTS "${_primary_package}/libhsa-runtime64.so.1")
    message(FATAL_ERROR
        "Runtime closure did not preserve exactly one canonical HSA identity "
        "from the selected ROCm SDK: ${_primary_manifest_entries}")
endif()
list(FIND _primary_manifest_entries "libhsakmt.so.1"
    _selected_hsakmt_manifest_index)
list(FIND _primary_manifest_entries "libhsakmt.so"
    _selected_hsakmt_alias_index)
if(_selected_hsakmt_manifest_index EQUAL -1 OR
   NOT _selected_hsakmt_alias_index EQUAL -1 OR
   NOT EXISTS "${_primary_output}/libhsakmt.so.1" OR
   NOT EXISTS "${_primary_package}/libhsakmt.so.1")
    message(FATAL_ERROR
        "Runtime closure did not preserve the canonical ROCt thunk required by "
        "the selected HSA runtime: ${_primary_manifest_entries}")
endif()

set(_required_packaged_runtimes
    libprimary.so libMIOpenContract.so.1 libhsa-runtime64.so.1
    libhsakmt.so.1 shared-runtime-manifest.txt)
if(_primary_runtime_policy STREQUAL "zluda-amd")
    list(APPEND _required_packaged_runtimes libMIOpenContract.so)
endif()
foreach(_packaged_runtime IN LISTS _required_packaged_runtimes)
    if(NOT EXISTS "${_primary_package}/${_packaged_runtime}")
        message(FATAL_ERROR
            "Classifier runtime package omitted '${_packaged_runtime}'")
    endif()
endforeach()
if(_primary_runtime_policy STREQUAL "zluda-amd")
    foreach(_packaged_runtime IN ITEMS
            libprimary.so libMIOpenContract.so.1 libhsa-runtime64.so.1
            libhsakmt.so.1)
        execute_process(
            COMMAND "${TEST_PATCHELF}" --print-rpath
                "${_primary_package}/${_packaged_runtime}"
            RESULT_VARIABLE _runpath_result
            OUTPUT_VARIABLE _runpath_value
            ERROR_VARIABLE _runpath_error)
        string(STRIP "${_runpath_value}" _runpath_value)
        if(NOT _runpath_result EQUAL 0 OR
           NOT _runpath_value STREQUAL "$ORIGIN")
            message(FATAL_ERROR
                "Classifier runtime '${_packaged_runtime}' does not use "
                "the required $ORIGIN RUNPATH: '${_runpath_value}' "
                "(${_runpath_error})")
        endif()
        execute_process(
            COMMAND "${TEST_READELF}" -d
                "${_primary_package}/${_packaged_runtime}"
            RESULT_VARIABLE _dynamic_section_result
            OUTPUT_VARIABLE _dynamic_section
            ERROR_VARIABLE _dynamic_section_error)
        if(NOT _dynamic_section_result EQUAL 0 OR
           NOT _dynamic_section MATCHES
               "[(]RUNPATH[)][^\n]*[$]ORIGIN")
            message(FATAL_ERROR
                "Classifier runtime '${_packaged_runtime}' does not contain "
                "a DT_RUNPATH entry for $ORIGIN: ${_dynamic_section_error}\n"
                "${_dynamic_section}")
        endif()
    endforeach()
    execute_process(
        COMMAND "${TEST_READELF}" -d
            "${_primary_package}/libprimary.so"
        RESULT_VARIABLE _primary_needed_result
        OUTPUT_VARIABLE _primary_needed_section
        ERROR_VARIABLE _primary_needed_error)
    if(NOT _primary_needed_result EQUAL 0 OR
       NOT _primary_needed_section MATCHES
           "\\[libMIOpenContract[.]so[.]1\\]" OR
       NOT _primary_needed_section MATCHES
           "\\[libhsa-runtime64[.]so[.]1\\]" OR
       _primary_needed_section MATCHES
           "\\[libMIOpenContract[.]so\\]")
        message(FATAL_ERROR
            "Classifier primary runtime retained an alias DT_NEEDED instead "
            "of the canonical SONAME: ${_primary_needed_error}\n"
            "${_primary_needed_section}")
    endif()
else()
    message(STATUS
        "patchelf is unavailable; skipping the Linux ZLUDA identity and RUNPATH sub-contracts")
endif()
if(EXISTS "${_primary_package}/javacpp-build-toolchain.properties")
    message(FATAL_ERROR
        "Classifier runtime package leaked build-only toolchain metadata")
endif()

message(STATUS
    "Shared-runtime aliases, no-SONAME DSOs, primary-root managed closure, and classifier package are preserved")
