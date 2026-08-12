# Stage concrete non-system shared runtimes beside a backend library.
#
# A target path may be an unversioned regular file even though its embedded
# loader name is versioned (for example libLLVM.so with SONAME
# libLLVM.so.22.0git). The classifier must carry the embedded loader name:
# that is the exact name recorded in the backend's dynamic dependencies.
#
# RUNTIME_LIBRARIES_PIPE carries every project-managed runtime path selected by
# the backend, separated by '|', which is not valid in Windows file names. An
# explicitly empty set is valid and still produces a manifest, ensuring a reused
# backend output directory cannot retain runtimes from an earlier configuration.
if(POLICY CMP0009)
    cmake_policy(SET CMP0009 NEW)
endif()

if(NOT DEFINED RUNTIME_LIBRARIES_PIPE)
    message(FATAL_ERROR
        "RUNTIME_LIBRARIES_PIPE must be defined when staging shared runtimes")
endif()
string(REPLACE "|" ";" _runtime_libraries "${RUNTIME_LIBRARIES_PIPE}")
list(REMOVE_DUPLICATES _runtime_libraries)

foreach(_runtime_library IN LISTS _runtime_libraries)
    if(NOT EXISTS "${_runtime_library}")
        message(FATAL_ERROR
            "Cannot stage shared runtime: '${_runtime_library}' does not exist")
    endif()
endforeach()

if(NOT DEFINED CXX_COMPILER OR CXX_COMPILER STREQUAL "")
    message(FATAL_ERROR
        "CXX_COMPILER is required so JavaCPP uses the same compiler as CMake")
endif()
get_filename_component(_cxx_compiler_real "${CXX_COMPILER}" REALPATH)
if(NOT EXISTS "${_cxx_compiler_real}" OR IS_DIRECTORY "${_cxx_compiler_real}")
    message(FATAL_ERROR
        "CMake C++ compiler '${CXX_COMPILER}' does not resolve to a compiler executable")
endif()

function(_shared_runtime_loader_name _out_var _library_path)
    if(WIN32)
        get_filename_component(_loader_name "${_library_path}" NAME)
    elseif(APPLE)
        if(NOT DEFINED OTOOL OR OTOOL STREQUAL "" OR NOT EXISTS "${OTOOL}")
            message(FATAL_ERROR
                "OTOOL from the active CMake toolchain is required to stage '${_library_path}'")
        endif()
        execute_process(
            COMMAND "${OTOOL}" -D "${_library_path}"
            RESULT_VARIABLE _metadata_result
            OUTPUT_VARIABLE _metadata_output
            ERROR_VARIABLE _metadata_error)
        if(NOT _metadata_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to read install name from '${_library_path}': ${_metadata_error}")
        endif()
        string(REGEX MATCH "\n[ \t]*([^ \t\r\n]+)" _install_name_match "${_metadata_output}")
        if(CMAKE_MATCH_1 STREQUAL "")
            message(FATAL_ERROR
                "Shared runtime '${_library_path}' has no Mach-O install name")
        endif()
        get_filename_component(_loader_name "${CMAKE_MATCH_1}" NAME)
    else()
        if(NOT DEFINED READELF OR READELF STREQUAL "" OR NOT EXISTS "${READELF}")
            message(FATAL_ERROR
                "READELF from the active CMake toolchain is required to stage '${_library_path}'")
        endif()
        execute_process(
            COMMAND "${READELF}" -d "${_library_path}"
            RESULT_VARIABLE _metadata_result
            OUTPUT_VARIABLE _metadata_output
            ERROR_VARIABLE _metadata_error)
        if(NOT _metadata_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to read SONAME from '${_library_path}': ${_metadata_error}")
        endif()
        string(REGEX MATCH "\\(SONAME\\)[^\n]*\\[([^]]+)\\]" _soname_match "${_metadata_output}")
        if(CMAKE_MATCH_1 STREQUAL "")
            # An ELF DSO is not required to declare DT_SONAME. In that case the
            # filename used by the link/load contract is its only loader name.
            # Several upstream ZLUDA CUDA-ABI replacement DSOs intentionally use
            # this form, so preserve the managed filename instead of rejecting it.
            get_filename_component(_loader_name "${_library_path}" NAME)
        else()
            set(_loader_name "${CMAKE_MATCH_1}")
        endif()
    endif()

    set(${_out_var} "${_loader_name}" PARENT_SCOPE)
endfunction()

# Best-effort loader-name inspection for files discovered under a declared
# runtime root. Development packages may expose only an unversioned linker alias
# and a concrete file whose basename differs from its SONAME. Invalid linker
# scripts and non-library files are skipped here; explicitly selected runtimes
# still use the strict helper above.
function(_try_shared_runtime_loader_name _out_var _success_var _library_path)
    set(_loader_name "")
    set(_loader_success FALSE)
    if(WIN32)
        get_filename_component(_loader_name "${_library_path}" NAME)
        set(_loader_success TRUE)
    elseif(APPLE)
        if(DEFINED OTOOL AND EXISTS "${OTOOL}")
            execute_process(
                COMMAND "${OTOOL}" -D "${_library_path}"
                RESULT_VARIABLE _metadata_result
                OUTPUT_VARIABLE _metadata_output
                ERROR_QUIET)
            if(_metadata_result EQUAL 0)
                string(REGEX MATCH "\n[ \t]*([^ \t\r\n]+)"
                    _install_name_match "${_metadata_output}")
                if(NOT CMAKE_MATCH_1 STREQUAL "")
                    get_filename_component(_loader_name
                        "${CMAKE_MATCH_1}" NAME)
                    set(_loader_success TRUE)
                endif()
            endif()
        endif()
    else()
        if(DEFINED READELF AND EXISTS "${READELF}")
            execute_process(
                COMMAND "${READELF}" -d "${_library_path}"
                RESULT_VARIABLE _metadata_result
                OUTPUT_VARIABLE _metadata_output
                ERROR_QUIET)
            if(_metadata_result EQUAL 0)
                string(REGEX MATCH
                    "\\(SONAME\\)[^\n]*\\[([^]]+)\\]"
                    _soname_match "${_metadata_output}")
                if(NOT CMAKE_MATCH_1 STREQUAL "")
                    set(_loader_name "${CMAKE_MATCH_1}")
                    set(_loader_success TRUE)
                endif()
            endif()
        endif()
    endif()
    set(${_out_var} "${_loader_name}" PARENT_SCOPE)
    set(${_success_var} "${_loader_success}" PARENT_SCOPE)
endfunction()

function(_shared_runtime_needed_names _out_var _library_path)
    if(WIN32)
        if(NOT DEFINED OBJDUMP OR OBJDUMP STREQUAL "" OR NOT EXISTS "${OBJDUMP}")
            message(FATAL_ERROR
                "OBJDUMP from the active CMake toolchain is required to inspect '${_library_path}'")
        endif()
        execute_process(
            COMMAND "${OBJDUMP}" -p "${_library_path}"
            RESULT_VARIABLE _needed_result
            OUTPUT_VARIABLE _needed_output
            ERROR_VARIABLE _needed_error)
        set(_needed_pattern "DLL Name:[ \t]*[^\r\n]+")
    elseif(APPLE)
        if(NOT DEFINED OTOOL OR OTOOL STREQUAL "" OR NOT EXISTS "${OTOOL}")
            message(FATAL_ERROR
                "OTOOL from the active CMake toolchain is required to inspect '${_library_path}'")
        endif()
        execute_process(
            COMMAND "${OTOOL}" -L "${_library_path}"
            RESULT_VARIABLE _needed_result
            OUTPUT_VARIABLE _needed_output
            ERROR_VARIABLE _needed_error)
        set(_needed_pattern "\n[ \t]*[^ \t\r\n]+")
    else()
        if(NOT DEFINED READELF OR READELF STREQUAL "" OR NOT EXISTS "${READELF}")
            message(FATAL_ERROR
                "READELF from the active CMake toolchain is required to inspect '${_library_path}'")
        endif()
        execute_process(
            COMMAND "${READELF}" -d "${_library_path}"
            RESULT_VARIABLE _needed_result
            OUTPUT_VARIABLE _needed_output
            ERROR_VARIABLE _needed_error)
        set(_needed_pattern "\\(NEEDED\\)[^\n]*\\[[^]]+\\]")
    endif()
    if(NOT _needed_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to inspect runtime dependencies of '${_library_path}': ${_needed_error}")
    endif()

    string(REGEX MATCHALL "${_needed_pattern}" _needed_entries "${_needed_output}")
    set(_needed_names "")
    foreach(_needed_entry IN LISTS _needed_entries)
        if(WIN32)
            string(REGEX REPLACE ".*DLL Name:[ \t]*" "" _needed_name "${_needed_entry}")
        elseif(APPLE)
            string(STRIP "${_needed_entry}" _needed_name)
            string(REGEX REPLACE "[ \t].*" "" _needed_name "${_needed_name}")
            get_filename_component(_needed_name "${_needed_name}" NAME)
        else()
            string(REGEX REPLACE ".*\\[([^]]+)\\].*" "\\1"
                _needed_name "${_needed_entry}")
        endif()
        string(STRIP "${_needed_name}" _needed_name)
        if(NOT _needed_name STREQUAL "")
            list(APPEND _needed_names "${_needed_name}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _needed_names)
    set(${_out_var} "${_needed_names}" PARENT_SCOPE)
endfunction()

function(_set_zluda_origin_runpath _library_path)
    if(NOT DEFINED PATCHELF_EXECUTABLE OR
       PATCHELF_EXECUTABLE STREQUAL "" OR
       NOT EXISTS "${PATCHELF_EXECUTABLE}")
        message(FATAL_ERROR
            "Linux ZLUDA runtime packaging requires the patchelf executable")
    endif()
    execute_process(
        COMMAND "${PATCHELF_EXECUTABLE}" --set-rpath "$ORIGIN" "${_library_path}"
        RESULT_VARIABLE _runpath_result
        ERROR_VARIABLE _runpath_error)
    if(NOT _runpath_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to set the self-contained ZLUDA RUNPATH on "
            "'${_library_path}': ${_runpath_error}")
    endif()
    execute_process(
        COMMAND "${PATCHELF_EXECUTABLE}" --print-rpath "${_library_path}"
        RESULT_VARIABLE _runpath_verify_result
        OUTPUT_VARIABLE _runpath_value
        ERROR_VARIABLE _runpath_verify_error)
    string(STRIP "${_runpath_value}" _runpath_value)
    if(NOT _runpath_verify_result EQUAL 0 OR
       NOT _runpath_value STREQUAL "$ORIGIN")
        message(FATAL_ERROR
            "Self-contained ZLUDA RUNPATH verification failed for "
            "'${_library_path}': expected '$ORIGIN', found "
            "'${_runpath_value}' (${_runpath_verify_error})")
    endif()
endfunction()

function(_is_managed_gpu_runtime_name _out_var _runtime_name)
    string(TOLOWER "${_runtime_name}" _runtime_name_lower)
    if(_runtime_name_lower MATCHES
            "^(lib)?(amd|hsa-runtime|miopen|roc|hip).*")
        set(${_out_var} TRUE PARENT_SCOPE)
    else()
        set(${_out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Some backends ship a project-managed user-space driver stack rather than a
# single DSO. RUNTIME_SEARCH_ROOTS_PIPE makes that dependency closure explicit:
# only dependencies resolved below one of these roots are bundled, so ordinary
# operating-system libraries remain host-provided. This is used by ZLUDA to
# carry its ROCm/HIP/MIOpen closure without copying glibc or the kernel driver.
set(_runtime_search_roots "")
if(DEFINED RUNTIME_SEARCH_ROOTS_PIPE AND
   NOT RUNTIME_SEARCH_ROOTS_PIPE STREQUAL "")
    string(REPLACE "|" ";" _runtime_search_roots
        "${RUNTIME_SEARCH_ROOTS_PIPE}")
    set(_normalized_runtime_search_roots "")
    foreach(_runtime_search_root IN LISTS _runtime_search_roots)
        if(NOT IS_DIRECTORY "${_runtime_search_root}")
            message(FATAL_ERROR
                "Shared-runtime search root does not exist: '${_runtime_search_root}'")
        endif()
        get_filename_component(_runtime_search_root_real
            "${_runtime_search_root}" REALPATH)
        list(APPEND _normalized_runtime_search_roots
            "${_runtime_search_root_real}")
    endforeach()
    list(REMOVE_DUPLICATES _normalized_runtime_search_roots)
    set(_runtime_search_roots "${_normalized_runtime_search_roots}")
endif()

set(_runtime_alias_entries "")
if(_runtime_search_roots)
    # Build a deterministic filename index for the permitted roots once. The
    # dependency walk below may inspect many ROCm libraries, so rescanning the
    # complete runtime tree for every DT_NEEDED entry would be unnecessarily slow.
    set(_runtime_root_files "")
    foreach(_runtime_search_root IN LISTS _runtime_search_roots)
        # Keep direct entries as spelled so ABI symlink names remain searchable.
        # The recursive index supplies transitive libraries in nested ROCm roots.
        file(GLOB _runtime_root_direct_candidates LIST_DIRECTORIES FALSE
            "${_runtime_search_root}/*")
        file(GLOB_RECURSE _runtime_root_recursive_candidates
            LIST_DIRECTORIES FALSE "${_runtime_search_root}/*")
        list(APPEND _runtime_root_files
            ${_runtime_root_direct_candidates}
            ${_runtime_root_recursive_candidates})
    endforeach()
    list(REMOVE_DUPLICATES _runtime_root_files)
    list(SORT _runtime_root_files)

    set(_runtime_queue ${_runtime_libraries})
    set(_primary_runtime_real "")
    if(DEFINED PRIMARY_RUNTIME AND NOT PRIMARY_RUNTIME STREQUAL "" AND
       EXISTS "${PRIMARY_RUNTIME}")
        get_filename_component(_primary_runtime_real
            "${PRIMARY_RUNTIME}" REALPATH)
        list(PREPEND _runtime_queue "${PRIMARY_RUNTIME}")
    endif()
    set(_runtime_inspected "")
    set(_runtime_libraries "")
    while(_runtime_queue)
        list(POP_FRONT _runtime_queue _runtime_candidate)
        get_filename_component(_runtime_candidate_real
            "${_runtime_candidate}" REALPATH)
        set(_runtime_candidate_is_primary FALSE)
        if(_primary_runtime_real AND
           _runtime_candidate_real STREQUAL _primary_runtime_real)
            set(_runtime_candidate_is_primary TRUE)
        endif()

        # Preserve the filename of every explicitly selected symlink before
        # canonical-path deduplication. ZLUDA exposes multiple CUDA ABI names
        # (for example libcuda.so and libcuda.so.1) through one libnvcuda DSO;
        # resolving those seeds first would otherwise discard the ABI aliases.
        if(NOT _runtime_candidate_is_primary)
            get_filename_component(_runtime_candidate_name
                "${_runtime_candidate}" NAME)
            _shared_runtime_loader_name(_runtime_candidate_loader_name
                "${_runtime_candidate_real}")
            if(NOT _runtime_candidate_name STREQUAL
               _runtime_candidate_loader_name)
                list(APPEND _runtime_alias_entries
                    "${_runtime_candidate_real}|${_runtime_candidate_name}")
            endif()
        endif()

        list(FIND _runtime_inspected "${_runtime_candidate_real}"
            _runtime_candidate_index)
        if(NOT _runtime_candidate_index EQUAL -1)
            continue()
        endif()
        list(APPEND _runtime_inspected "${_runtime_candidate_real}")
        if(NOT _runtime_candidate_is_primary)
            list(APPEND _runtime_libraries "${_runtime_candidate_real}")
        endif()

        _shared_runtime_needed_names(_needed_names "${_runtime_candidate_real}")
        foreach(_needed_name IN LISTS _needed_names)
            set(_needed_path "")
            if(WIN32)
                string(TOLOWER "${_needed_name}" _needed_name_compare)
            else()
                set(_needed_name_compare "${_needed_name}")
            endif()

            # Prefer an explicitly seeded runtime with the requested loader name.
            foreach(_seed_runtime IN LISTS _runtime_libraries _runtime_queue)
                _shared_runtime_loader_name(_seed_loader_name "${_seed_runtime}")
                if(WIN32)
                    string(TOLOWER "${_seed_loader_name}" _seed_loader_compare)
                    string(TOLOWER "${_needed_name}" _needed_name_compare)
                else()
                    set(_seed_loader_compare "${_seed_loader_name}")
                    set(_needed_name_compare "${_needed_name}")
                endif()
                if(_seed_loader_compare STREQUAL _needed_name_compare)
                    set(_needed_path "${_seed_runtime}")
                    break()
                endif()
            endforeach()

            _is_managed_gpu_runtime_name(_managed_gpu_runtime "${_needed_name}")
            if(NOT _needed_path AND _managed_gpu_runtime)
                foreach(_root_runtime_file IN LISTS _runtime_root_files)
                    get_filename_component(_root_runtime_name
                        "${_root_runtime_file}" NAME)
                    if(WIN32)
                        string(TOLOWER "${_root_runtime_name}" _root_runtime_lower)
                    else()
                        set(_root_runtime_lower "${_root_runtime_name}")
                    endif()
                    if(_root_runtime_lower STREQUAL _needed_name_compare)
                        set(_needed_path "${_root_runtime_file}")
                        break()
                    endif()
                endforeach()
            endif()

            # If no file is spelled with the requested loader name, resolve by
            # embedded SONAME. ROCm packages such as MIOpen can install a
            # concrete versioned DSO whose filename and SONAME differ.
            if(NOT _needed_path AND _managed_gpu_runtime)
                foreach(_root_runtime_file IN LISTS _runtime_root_files)
                    get_filename_component(_root_runtime_name
                        "${_root_runtime_file}" NAME)
                    if(WIN32)
                        if(NOT _root_runtime_name MATCHES "\\.[Dd][Ll][Ll]$")
                            continue()
                        endif()
                    elseif(APPLE)
                        if(NOT _root_runtime_name MATCHES "\\.dylib($|\\.)")
                            continue()
                        endif()
                    else()
                        if(NOT _root_runtime_name MATCHES "\\.so($|\\.)")
                            continue()
                        endif()
                    endif()
                    _try_shared_runtime_loader_name(
                        _root_loader_name _root_loader_success
                        "${_root_runtime_file}")
                    if(NOT _root_loader_success)
                        continue()
                    endif()
                    if(WIN32)
                        string(TOLOWER "${_root_loader_name}"
                            _root_loader_compare)
                    else()
                        set(_root_loader_compare "${_root_loader_name}")
                    endif()
                    if(_root_loader_compare STREQUAL _needed_name_compare)
                        set(_needed_path "${_root_runtime_file}")
                        break()
                    endif()
                endforeach()
            endif()
            if(_needed_path)
                list(APPEND _runtime_queue "${_needed_path}")
                _shared_runtime_loader_name(_needed_loader_name "${_needed_path}")
                if(NOT _needed_loader_name STREQUAL _needed_name)
                    list(APPEND _runtime_alias_entries
                        "${_needed_path}|${_needed_name}")
                endif()
            elseif(_managed_gpu_runtime)
                message(FATAL_ERROR
                    "Project-managed GPU runtime '${_runtime_candidate_real}' requires "
                    "'${_needed_name}', but it was not found below the declared runtime roots: "
                    "${_runtime_search_roots}")
            endif()
        endforeach()
    endwhile()
    list(REMOVE_DUPLICATES _runtime_alias_entries)
endif()

# ZLUDA classifiers may use CUDA headers and static implementation archives at
# build time, but the final backend must have no dynamic NVIDIA runtime
# dependency. CUDA ABI names implemented by ZLUDA and AMD/ROCm names are allowed
# only when their exact loader names are present in the packaged closure.
if(DEFINED RUNTIME_POLICY AND RUNTIME_POLICY STREQUAL "zluda-amd")
    if(NOT DEFINED PRIMARY_RUNTIME OR PRIMARY_RUNTIME STREQUAL "" OR
       NOT EXISTS "${PRIMARY_RUNTIME}")
        message(FATAL_ERROR
            "zluda-amd runtime policy requires the linked PRIMARY_RUNTIME")
    endif()

    set(_available_runtime_names "")
    foreach(_available_runtime IN LISTS _runtime_libraries)
        _shared_runtime_loader_name(_available_loader_name "${_available_runtime}")
        list(APPEND _available_runtime_names "${_available_loader_name}")
    endforeach()
    foreach(_runtime_alias_entry IN LISTS _runtime_alias_entries)
        string(REPLACE "|" ";" _runtime_alias_parts "${_runtime_alias_entry}")
        list(GET _runtime_alias_parts 1 _runtime_alias_name)
        list(APPEND _available_runtime_names "${_runtime_alias_name}")
    endforeach()
    list(REMOVE_DUPLICATES _available_runtime_names)

    _shared_runtime_needed_names(_primary_needed_names "${PRIMARY_RUNTIME}")
    foreach(_primary_needed_name IN LISTS _primary_needed_names)
        string(TOLOWER "${_primary_needed_name}" _primary_needed_lower)
        if(WIN32)
            set(_primary_needed_compare "${_primary_needed_lower}")
        else()
            # ELF SONAMEs and Mach-O install names are case-sensitive. Keep the
            # original spelling for closure membership while using the lowercase
            # copy only for case-insensitive policy-family classification below.
            set(_primary_needed_compare "${_primary_needed_name}")
        endif()
        if(_primary_needed_lower MATCHES
                "^(lib)?(cudart|cusolver|nvrtc|nvjitlink|curand).*")
            message(FATAL_ERROR
                "ZLUDA backend retained forbidden dynamic NVIDIA dependency '${_primary_needed_name}'; link its build-time implementation statically")
        endif()

        set(_packaged_gpu_runtime FALSE)
        if(_primary_needed_lower MATCHES
                "^(lib)?(cuda|nvcuda|cublas|cublaslt|cusparse|cufft|cudnn|amd|hsa-runtime|miopen|roc|hip).*")
            set(_packaged_gpu_runtime TRUE)
        endif()
        if(_packaged_gpu_runtime)
            set(_primary_runtime_found FALSE)
            foreach(_available_runtime_name IN LISTS _available_runtime_names)
                if(WIN32)
                    string(TOLOWER "${_available_runtime_name}" _available_runtime_lower)
                else()
                    set(_available_runtime_lower "${_available_runtime_name}")
                endif()
                if(_available_runtime_lower STREQUAL _primary_needed_compare)
                    set(_primary_runtime_found TRUE)
                    break()
                endif()
            endforeach()
            if(NOT _primary_runtime_found)
                message(FATAL_ERROR
                    "ZLUDA backend requires '${_primary_needed_name}', but the classifier runtime closure does not provide it")
            endif()
        endif()
    endforeach()
endif()

if(NOT DEFINED OUTPUT_DIR OR OUTPUT_DIR STREQUAL "")
    message(FATAL_ERROR "OUTPUT_DIR is required when staging shared runtimes")
endif()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# nd4jcpu and the standalone SDX target can finish linking concurrently and
# stage the same LLVM/MLIR closure into one classifier directory. Serialize the
# complete cleanup/copy/manifest transaction: copy_if_different alone is not
# sufficient because another stager may remove the destination between its copy
# and verification. GUARD PROCESS releases the lock even on a fatal error.
set(_runtime_stage_lock "${OUTPUT_DIR}/.shared-runtime-stage.lock")
file(LOCK "${_runtime_stage_lock}" GUARD PROCESS TIMEOUT 300
    RESULT_VARIABLE _runtime_stage_lock_result)
if(NOT _runtime_stage_lock_result STREQUAL "0")
    message(FATAL_ERROR
        "Could not acquire shared-runtime staging lock '${_runtime_stage_lock}': "
        "${_runtime_stage_lock_result}")
endif()

# Backend build directories are reusable. Remove only files recorded by the
# previous staging pass; broad LLVM/MLIR globs would erase a second legitimate
# runtime version before the complete current dependency set is staged.
set(_runtime_manifest_path "${OUTPUT_DIR}/shared-runtime-manifest.txt")
if(EXISTS "${_runtime_manifest_path}")
    file(STRINGS "${_runtime_manifest_path}" _previous_runtime_names)
    foreach(_previous_runtime_name IN LISTS _previous_runtime_names)
        if(NOT _previous_runtime_name STREQUAL "" AND
           NOT _previous_runtime_name MATCHES "^#" AND
           NOT _previous_runtime_name MATCHES "[/\\\\]")
            file(REMOVE "${OUTPUT_DIR}/${_previous_runtime_name}")
        endif()
    endforeach()
endif()

set(_staged_runtime_names "")
foreach(_runtime_library IN LISTS _runtime_libraries)
    get_filename_component(_runtime_real_path "${_runtime_library}" REALPATH)
    if(NOT EXISTS "${_runtime_real_path}" OR IS_DIRECTORY "${_runtime_real_path}")
        message(FATAL_ERROR
            "Cannot stage shared runtime: resolved path '${_runtime_real_path}' is not a file")
    endif()
    _shared_runtime_loader_name(_runtime_name "${_runtime_real_path}")
    set(_runtime_output "${OUTPUT_DIR}/${_runtime_name}")

    list(FIND _staged_runtime_names "${_runtime_name}" _runtime_name_index)
    if(NOT _runtime_name_index EQUAL -1)
        file(SHA256 "${_runtime_real_path}" _runtime_source_hash)
        file(SHA256 "${_runtime_output}" _runtime_staged_hash)
        if(NOT _runtime_source_hash STREQUAL _runtime_staged_hash)
            message(FATAL_ERROR
                "Two shared runtime targets resolve to different libraries with the "
                "same loader name '${_runtime_name}'. A classifier cannot represent "
                "that ambiguous dynamic-link closure.")
        endif()
        continue()
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_runtime_real_path}" "${_runtime_output}"
        RESULT_VARIABLE _copy_result)
    if(NOT _copy_result EQUAL 0 OR NOT EXISTS "${_runtime_output}")
        message(FATAL_ERROR
            "Failed to stage shared runtime '${_runtime_real_path}' at '${_runtime_output}'")
    endif()
    list(APPEND _staged_runtime_names "${_runtime_name}")
    message(STATUS "Staged shared runtime: ${_runtime_output}")
endforeach()

# Preserve dependency filenames that intentionally differ from a library's
# SONAME. ZLUDA v6 patches ROCm DT_NEEDED entries to unversioned names for ROCm
# 6/7 compatibility, while the selected build library retains a versioned
# SONAME; both names must exist in JavaCPP's extraction directory.
foreach(_runtime_alias_entry IN LISTS _runtime_alias_entries)
    string(REPLACE "|" ";" _runtime_alias_parts "${_runtime_alias_entry}")
    list(GET _runtime_alias_parts 0 _runtime_alias_source)
    list(GET _runtime_alias_parts 1 _runtime_alias_name)
    if(_runtime_alias_name MATCHES "[/\\\\]")
        message(FATAL_ERROR
            "Unsafe runtime dependency alias '${_runtime_alias_name}'")
    endif()
    get_filename_component(_runtime_alias_real "${_runtime_alias_source}" REALPATH)
    set(_runtime_alias_output "${OUTPUT_DIR}/${_runtime_alias_name}")
    list(FIND _staged_runtime_names "${_runtime_alias_name}"
        _runtime_alias_index)
    if(NOT _runtime_alias_index EQUAL -1)
        file(SHA256 "${_runtime_alias_real}" _runtime_alias_source_hash)
        file(SHA256 "${_runtime_alias_output}" _runtime_alias_output_hash)
        if(NOT _runtime_alias_source_hash STREQUAL _runtime_alias_output_hash)
            message(FATAL_ERROR
                "Runtime alias '${_runtime_alias_name}' resolves to conflicting libraries")
        endif()
        continue()
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_runtime_alias_real}" "${_runtime_alias_output}"
        RESULT_VARIABLE _runtime_alias_copy_result)
    if(NOT _runtime_alias_copy_result EQUAL 0 OR
       NOT EXISTS "${_runtime_alias_output}")
        message(FATAL_ERROR
            "Failed to stage runtime alias '${_runtime_alias_name}'")
    endif()
    list(APPEND _staged_runtime_names "${_runtime_alias_name}")
    message(STATUS "Staged shared runtime alias: ${_runtime_alias_output}")
endforeach()

# JavaCPP extracts every classifier member into one directory, but the dynamic
# loader does not search a DSO's sibling directory unless that DSO says so.
# Normalize every Linux ZLUDA/ROCm library, including the linked backend, to a
# relocatable RUNPATH before materializing the classifier package. This keeps
# build-host CUDA/ROCm paths out of the consumer contract.
if(DEFINED RUNTIME_POLICY AND RUNTIME_POLICY STREQUAL "zluda-amd" AND
   UNIX AND NOT APPLE)
    set(_zluda_runpath_files "")
    foreach(_staged_runtime_name IN LISTS _staged_runtime_names)
        list(APPEND _zluda_runpath_files
            "${OUTPUT_DIR}/${_staged_runtime_name}")
    endforeach()
    if(DEFINED PRIMARY_RUNTIME AND
       NOT PRIMARY_RUNTIME STREQUAL "" AND
       EXISTS "${PRIMARY_RUNTIME}" AND
       NOT IS_DIRECTORY "${PRIMARY_RUNTIME}")
        list(APPEND _zluda_runpath_files "${PRIMARY_RUNTIME}")
    endif()
    list(REMOVE_DUPLICATES _zluda_runpath_files)
    foreach(_zluda_runpath_file IN LISTS _zluda_runpath_files)
        _set_zluda_origin_runpath("${_zluda_runpath_file}")
        message(STATUS
            "Set self-contained ZLUDA RUNPATH: ${_zluda_runpath_file}")
    endforeach()
endif()

list(SORT _staged_runtime_names)
list(LENGTH _staged_runtime_names _runtime_count)
set(_runtime_manifest
    "# nd4j-shared-runtime-manifest-v1\n# runtime-count=${_runtime_count}\n")
if(_staged_runtime_names)
    string(REPLACE ";" "\n" _runtime_entries "${_staged_runtime_names}")
    string(APPEND _runtime_manifest "${_runtime_entries}\n")
endif()
file(WRITE "${_runtime_manifest_path}" "${_runtime_manifest}")

# A binding module must not rediscover native dependencies or copy broad build
# directory globs.  When PACKAGE_DIR is supplied, materialize the exact package
# payload selected above: the linked backend, every manifest-owned runtime and
# the manifest itself.  Keeping this operation in CMake makes native dependency
# resolution and classifier contents one fail-closed contract.
if(DEFINED PACKAGE_DIR AND NOT PACKAGE_DIR STREQUAL "")
    get_filename_component(_runtime_output_dir_absolute "${OUTPUT_DIR}" ABSOLUTE)
    get_filename_component(_runtime_package_dir_absolute "${PACKAGE_DIR}" ABSOLUTE)
    file(RELATIVE_PATH _runtime_package_relative
        "${_runtime_output_dir_absolute}" "${_runtime_package_dir_absolute}")
    if(_runtime_package_relative STREQUAL "" OR
       _runtime_package_relative STREQUAL "." OR
       _runtime_package_relative MATCHES "^\\.\\.[/\\\\]" OR
       IS_ABSOLUTE "${_runtime_package_relative}")
        message(FATAL_ERROR
            "PACKAGE_DIR must be a dedicated child of OUTPUT_DIR: "
            "'${PACKAGE_DIR}' is not safely contained by '${OUTPUT_DIR}'")
    endif()

    file(REMOVE_RECURSE "${_runtime_package_dir_absolute}")
    file(MAKE_DIRECTORY "${_runtime_package_dir_absolute}")

    set(_runtime_package_sources "")
    if(DEFINED PRIMARY_RUNTIME AND
       NOT PRIMARY_RUNTIME STREQUAL "" AND
       EXISTS "${PRIMARY_RUNTIME}" AND
       NOT IS_DIRECTORY "${PRIMARY_RUNTIME}")
        list(APPEND _runtime_package_sources "${PRIMARY_RUNTIME}")
    endif()
    foreach(_staged_runtime_name IN LISTS _staged_runtime_names)
        list(APPEND _runtime_package_sources
            "${OUTPUT_DIR}/${_staged_runtime_name}")
    endforeach()

    foreach(_runtime_package_source IN LISTS _runtime_package_sources)
        get_filename_component(_runtime_package_name
            "${_runtime_package_source}" NAME)
        set(_runtime_package_output
            "${_runtime_package_dir_absolute}/${_runtime_package_name}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_runtime_package_source}" "${_runtime_package_output}"
            RESULT_VARIABLE _runtime_package_copy_result)
        if(NOT _runtime_package_copy_result EQUAL 0 OR
           NOT EXISTS "${_runtime_package_output}")
            message(FATAL_ERROR
                "Failed to copy classifier runtime '${_runtime_package_source}' "
                "to '${_runtime_package_output}'")
        endif()
    endforeach()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_runtime_manifest_path}"
            "${_runtime_package_dir_absolute}/shared-runtime-manifest.txt"
        RESULT_VARIABLE _runtime_package_manifest_copy_result)
    if(NOT _runtime_package_manifest_copy_result EQUAL 0 OR
       NOT EXISTS "${_runtime_package_dir_absolute}/shared-runtime-manifest.txt")
        message(FATAL_ERROR
            "Failed to copy the classifier shared-runtime manifest")
    endif()
    list(LENGTH _runtime_package_sources _runtime_package_file_count)
    message(STATUS
        "Materialized classifier runtime package at "
        "${_runtime_package_dir_absolute} (${_runtime_package_file_count} libraries)")
endif()

# Build-only metadata: JavaCPP consumes this from platform.linkpath to compile its
# JNI wrapper with the exact C++ driver selected by CMake. The binding POM copies
# only shared-runtime-manifest.txt, so no build-host compiler path enters the jar.
file(WRITE "${OUTPUT_DIR}/javacpp-build-toolchain.properties"
    "platform.compiler=${_cxx_compiler_real}\n")
