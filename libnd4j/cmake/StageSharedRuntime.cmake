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

if(DEFINED RUNTIME_LIBRARIES_FILE AND
   NOT RUNTIME_LIBRARIES_FILE STREQUAL "")
    if(NOT EXISTS "${RUNTIME_LIBRARIES_FILE}")
        message(FATAL_ERROR
            "Runtime library list file does not exist: '${RUNTIME_LIBRARIES_FILE}'")
    endif()
    file(READ "${RUNTIME_LIBRARIES_FILE}" _runtime_libraries_contents)
    string(REPLACE "\r\n" "\n" _runtime_libraries_contents
        "${_runtime_libraries_contents}")
    string(REPLACE "\n" ";" _runtime_libraries
        "${_runtime_libraries_contents}")
    set(_runtime_libraries_without_empty "")
    foreach(_runtime_library IN LISTS _runtime_libraries)
        string(STRIP "${_runtime_library}" _runtime_library)
        if(NOT _runtime_library STREQUAL "")
            list(APPEND _runtime_libraries_without_empty "${_runtime_library}")
        endif()
    endforeach()
    set(_runtime_libraries "${_runtime_libraries_without_empty}")
elseif(DEFINED RUNTIME_LIBRARIES_PIPE)
    string(REPLACE "|" ";" _runtime_libraries "${RUNTIME_LIBRARIES_PIPE}")
else()
    message(FATAL_ERROR
        "RUNTIME_LIBRARIES_PIPE or RUNTIME_LIBRARIES_FILE must be defined "
        "when staging shared runtimes")
endif()
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
        if(_soname_match STREQUAL "")
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
        # MSVC generators do not define CMAKE_OBJDUMP. Prefer it when a
        # MinGW/LLVM toolchain supplied one, but use the matching Visual Studio
        # dumpbin executable for the normal MSVC path. dumpbin lives beside
        # cl.exe, so this remains tied to the active CMake compiler rather than
        # depending on a machine-specific PATH entry.
        set(_windows_dependency_inspector "")
        set(_windows_dependency_mode "")
        if(DEFINED OBJDUMP AND NOT OBJDUMP STREQUAL "" AND EXISTS "${OBJDUMP}")
            set(_windows_dependency_inspector "${OBJDUMP}")
            set(_windows_dependency_mode "objdump")
        else()
            get_filename_component(_cxx_compiler_dir "${CXX_COMPILER}" DIRECTORY)
            find_program(_dumpbin_executable
                NAMES dumpbin.exe dumpbin
                HINTS "${_cxx_compiler_dir}")
            if(_dumpbin_executable)
                set(_windows_dependency_inspector "${_dumpbin_executable}")
                set(_windows_dependency_mode "dumpbin")
            endif()
        endif()
        if(_windows_dependency_inspector STREQUAL "")
            message(FATAL_ERROR
                "A Windows dependency inspector is required to inspect '${_library_path}': "
                "CMAKE_OBJDUMP/OBJDUMP was not configured and dumpbin.exe was not found beside CXX_COMPILER")
        endif()
        if(_windows_dependency_mode STREQUAL "objdump")
            execute_process(
                COMMAND "${_windows_dependency_inspector}" -p "${_library_path}"
                RESULT_VARIABLE _needed_result
                OUTPUT_VARIABLE _needed_output
                ERROR_VARIABLE _needed_error)
            set(_needed_pattern "DLL Name:[ \t]*[^\r\n]+")
        else()
            execute_process(
                COMMAND "${_windows_dependency_inspector}" /DEPENDENTS "${_library_path}"
                RESULT_VARIABLE _needed_result
                OUTPUT_VARIABLE _needed_output
                ERROR_VARIABLE _needed_error)
            # dumpbin prints one DLL basename per indented line after
            # "Image has the following dependencies:". Parse only those lines
            # so the header and summary text cannot become runtime names.
            string(REPLACE "\r\n" "\n" _needed_output "${_needed_output}")
            string(REPLACE "\n" ";" _needed_lines "${_needed_output}")
            set(_needed_entries "")
            foreach(_needed_line IN LISTS _needed_lines)
                string(STRIP "${_needed_line}" _needed_line)
                if(_needed_line MATCHES "^[^ \t/\\\\]+\\\\.(dll|DLL)$")
                    list(APPEND _needed_entries "${_needed_line}")
                endif()
            endforeach()
        endif()
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

    if(NOT (WIN32 AND _windows_dependency_mode STREQUAL "dumpbin"))
        string(REGEX MATCHALL "${_needed_pattern}" _needed_entries "${_needed_output}")
    endif()
    set(_needed_names "")
    foreach(_needed_entry IN LISTS _needed_entries)
        if(WIN32 AND _windows_dependency_mode STREQUAL "dumpbin")
            set(_needed_name "${_needed_entry}")
        elseif(WIN32)
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

# A classifier archive cannot preserve the symlink topology used by ELF
# development packages. When two alias filenames are extracted as regular
# files, dlopen treats them as independent DSOs even though their bytes and
# SONAME match. Normalize every managed dependency to the concrete library's
# loader name before packaging so the process has one runtime identity.
function(_normalize_zluda_needed_aliases _library_path)
    _shared_runtime_needed_names(_needed_names "${_library_path}")
    set(_replaced_aliases "")
    foreach(_runtime_alias_entry IN LISTS _runtime_alias_entries)
        string(REPLACE "|" ";" _runtime_alias_parts "${_runtime_alias_entry}")
        list(GET _runtime_alias_parts 0 _runtime_alias_source)
        list(GET _runtime_alias_parts 1 _runtime_alias_name)
        _shared_runtime_loader_name(_runtime_canonical_name
            "${_runtime_alias_source}")
        if(_runtime_alias_name STREQUAL _runtime_canonical_name)
            continue()
        endif()

        list(FIND _needed_names "${_runtime_alias_name}" _needed_alias_index)
        if(_needed_alias_index EQUAL -1)
            continue()
        endif()
        execute_process(
            COMMAND "${PATCHELF_EXECUTABLE}" --replace-needed
                "${_runtime_alias_name}" "${_runtime_canonical_name}"
                "${_library_path}"
            RESULT_VARIABLE _replace_needed_result
            ERROR_VARIABLE _replace_needed_error)
        if(NOT _replace_needed_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to normalize ZLUDA runtime dependency "
                "'${_runtime_alias_name}' to '${_runtime_canonical_name}' in "
                "'${_library_path}': ${_replace_needed_error}")
        endif()
        list(APPEND _replaced_aliases
            "${_runtime_alias_name}|${_runtime_canonical_name}")
        list(REMOVE_ITEM _needed_names "${_runtime_alias_name}")
        list(APPEND _needed_names "${_runtime_canonical_name}")
        message(STATUS
            "Normalized ZLUDA runtime dependency: ${_runtime_alias_name} -> "
            "${_runtime_canonical_name} in ${_library_path}")
    endforeach()

    if(_replaced_aliases)
        _shared_runtime_needed_names(_normalized_needed_names "${_library_path}")
        foreach(_replaced_alias IN LISTS _replaced_aliases)
            string(REPLACE "|" ";" _replaced_alias_parts "${_replaced_alias}")
            list(GET _replaced_alias_parts 0 _replaced_alias_name)
            list(GET _replaced_alias_parts 1 _replaced_canonical_name)
            list(FIND _normalized_needed_names "${_replaced_alias_name}"
                _stale_alias_index)
            list(FIND _normalized_needed_names "${_replaced_canonical_name}"
                _canonical_needed_index)
            if(NOT _stale_alias_index EQUAL -1 OR
               _canonical_needed_index EQUAL -1)
                message(FATAL_ERROR
                    "ZLUDA runtime dependency normalization did not produce "
                    "'${_replaced_canonical_name}' without stale alias "
                    "'${_replaced_alias_name}' in '${_library_path}': "
                    "${_normalized_needed_names}")
            endif()
        endforeach()
    endif()
endfunction()

function(_is_managed_gpu_runtime_name _out_var _runtime_name)
    string(TOLOWER "${_runtime_name}" _runtime_name_lower)
    if(_runtime_name_lower MATCHES
            "^(lib)?(amd|hsa-runtime|hsakmt|miopen|roc|hip|cuda|nvcuda|cublas|cublaslt|cusparse|cufft|cudnn|nvrtc|nvjitlink|curand|nvptxcompiler).*")
        set(${_out_var} TRUE PARENT_SCOPE)
    else()
        set(${_out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Some backends ship a project-managed user-space driver stack rather than a
# single DSO. RUNTIME_SEARCH_ROOTS_PIPE makes that dependency closure explicit:
# only dependencies resolved below one of these roots are bundled, so ordinary
# operating-system libraries remain host-provided. This is used by ZLUDA to
# carry its version-matched ROCm/HIP/HSA/ROCt/MIOpen user-space closure
# without copying glibc or the kernel driver. The amdgpu/KFD kernel driver
# remains host-owned; HIP, HSA, and HSAKMT come from one selected ROCM_PATH.
set(_runtime_search_roots "")
if(DEFINED RUNTIME_SEARCH_ROOTS_FILE AND
   NOT RUNTIME_SEARCH_ROOTS_FILE STREQUAL "")
    if(NOT EXISTS "${RUNTIME_SEARCH_ROOTS_FILE}")
        message(FATAL_ERROR
            "Runtime search-root list file does not exist: '${RUNTIME_SEARCH_ROOTS_FILE}'")
    endif()
    file(READ "${RUNTIME_SEARCH_ROOTS_FILE}" _runtime_search_roots_contents)
    string(REPLACE "\r\n" "\n" _runtime_search_roots_contents
        "${_runtime_search_roots_contents}")
    string(REPLACE "\n" ";" _runtime_search_roots
        "${_runtime_search_roots_contents}")
    set(_runtime_search_roots_without_empty "")
    foreach(_runtime_search_root IN LISTS _runtime_search_roots)
        string(STRIP "${_runtime_search_root}" _runtime_search_root)
        if(NOT _runtime_search_root STREQUAL "")
            list(APPEND _runtime_search_roots_without_empty "${_runtime_search_root}")
        endif()
    endforeach()
    set(_runtime_search_roots "${_runtime_search_roots_without_empty}")
elseif(DEFINED RUNTIME_SEARCH_ROOTS_PIPE AND
       NOT RUNTIME_SEARCH_ROOTS_PIPE STREQUAL "")
    string(REPLACE "|" ";" _runtime_search_roots
        "${RUNTIME_SEARCH_ROOTS_PIPE}")
endif()
if(_runtime_search_roots)
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

# JavaCPP preloads the manifest entries in order. The closure walk above is
# intentionally breadth-first so it can discover every transitive dependency,
# but breadth-first order is unsafe for ELF constructors: a consumer such as
# MIOpen may be loaded before libamd_comgr or librocprofiler-register. Build the
# dependency graph from each staged library's DT_NEEDED names and emit a stable
# dependency-first topological order. The graph is derived from the binaries and
# aliases; no ROCm library names are hardcoded here.
if(_runtime_libraries)
    set(_runtime_dependency_edges "")
    foreach(_runtime_consumer IN LISTS _runtime_libraries)
        get_filename_component(_runtime_consumer_real
            "${_runtime_consumer}" REALPATH)
        _shared_runtime_needed_names(_runtime_consumer_needed
            "${_runtime_consumer_real}")
        foreach(_runtime_needed_name IN LISTS _runtime_consumer_needed)
            set(_runtime_dependency_real "")
            foreach(_runtime_dependency IN LISTS _runtime_libraries)
                _shared_runtime_loader_name(_runtime_dependency_loader
                    "${_runtime_dependency}")
                if(WIN32)
                    string(TOLOWER "${_runtime_dependency_loader}"
                        _runtime_dependency_loader_compare)
                    string(TOLOWER "${_runtime_needed_name}"
                        _runtime_needed_compare)
                else()
                    set(_runtime_dependency_loader_compare
                        "${_runtime_dependency_loader}")
                    set(_runtime_needed_compare "${_runtime_needed_name}")
                endif()
                if(_runtime_dependency_loader_compare STREQUAL
                   _runtime_needed_compare)
                    get_filename_component(_runtime_dependency_real
                        "${_runtime_dependency}" REALPATH)
                    break()
                endif()
            endforeach()

            # A DT_NEEDED entry may use a compatibility filename whose target has
            # a different SONAME. Resolve that alias to its canonical staged path.
            if(NOT _runtime_dependency_real)
                foreach(_runtime_alias_entry IN LISTS _runtime_alias_entries)
                    string(REPLACE "|" ";" _runtime_alias_parts
                        "${_runtime_alias_entry}")
                    list(GET _runtime_alias_parts 0 _runtime_alias_source)
                    list(GET _runtime_alias_parts 1 _runtime_alias_name)
                    if(WIN32)
                        string(TOLOWER "${_runtime_alias_name}"
                            _runtime_alias_compare)
                        string(TOLOWER "${_runtime_needed_name}"
                            _runtime_needed_compare)
                    else()
                        set(_runtime_alias_compare "${_runtime_alias_name}")
                        set(_runtime_needed_compare "${_runtime_needed_name}")
                    endif()
                    if(_runtime_alias_compare STREQUAL _runtime_needed_compare)
                        get_filename_component(_runtime_dependency_real
                            "${_runtime_alias_source}" REALPATH)
                        break()
                    endif()
                endforeach()
            endif()

            if(_runtime_dependency_real AND
               NOT _runtime_dependency_real STREQUAL _runtime_consumer_real)
                list(APPEND _runtime_dependency_edges
                    "${_runtime_consumer_real}|${_runtime_dependency_real}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _runtime_dependency_edges)

    set(_runtime_ordered_libraries "")
    set(_runtime_remaining_libraries "${_runtime_libraries}")
    while(_runtime_remaining_libraries)
        set(_runtime_order_progress FALSE)
        foreach(_runtime_candidate IN LISTS _runtime_remaining_libraries)
            get_filename_component(_runtime_candidate_real
                "${_runtime_candidate}" REALPATH)
            set(_runtime_candidate_blocked FALSE)
            foreach(_runtime_edge IN LISTS _runtime_dependency_edges)
                string(REPLACE "|" ";" _runtime_edge_parts "${_runtime_edge}")
                list(GET _runtime_edge_parts 0 _runtime_edge_consumer)
                list(GET _runtime_edge_parts 1 _runtime_edge_dependency)
                if(NOT _runtime_edge_consumer STREQUAL
                   _runtime_candidate_real)
                    continue()
                endif()
                list(FIND _runtime_remaining_libraries
                    "${_runtime_edge_dependency}" _runtime_dependency_index)
                if(NOT _runtime_dependency_index EQUAL -1)
                    set(_runtime_candidate_blocked TRUE)
                    break()
                endif()
            endforeach()
            if(NOT _runtime_candidate_blocked)
                list(APPEND _runtime_ordered_libraries
                    "${_runtime_candidate_real}")
                list(REMOVE_ITEM _runtime_remaining_libraries
                    "${_runtime_candidate}")
                set(_runtime_order_progress TRUE)
            endif()
        endforeach()
        if(NOT _runtime_order_progress)
            message(FATAL_ERROR
                "Shared-runtime dependency graph contains a cycle; cannot produce "
                "a dependency-first preload manifest")
        endif()
    endwhile()
    set(_runtime_libraries "${_runtime_ordered_libraries}")
endif()

# ZLUDA classifiers may use CUDA headers and implementation libraries at build
# time. Every CUDA/ROCm runtime needed by the final backend must be explicitly
# staged into the classifier closure; an AMD consumer must not need a host CUDA
# or ROCm installation.
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
        set(_packaged_gpu_runtime FALSE)
        if(_primary_needed_lower MATCHES
                "^(lib)?(cuda|nvcuda|cublas|cublaslt|cusparse|cufft|cudnn|cudart|cusolver|nvrtc|nvjitlink|curand|nvptxcompiler|amd|hsa-runtime|hsakmt|miopen|roc|hip).*")
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
set(_staged_resource_names "")
set(_staged_package_names "")
set(_staged_runtime_alias_names "")
set(_staged_runtime_alias_targets "")
set(_staged_runtime_alias_mappings "")
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
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_FILE=${_runtime_real_path}"
            "-DOUTPUT_FILE=${_runtime_output}"
            -P "${CMAKE_CURRENT_LIST_DIR}/StageArtifact.cmake"
        RESULT_VARIABLE _copy_result)
    if(NOT _copy_result EQUAL 0 OR NOT EXISTS "${_runtime_output}")
        message(FATAL_ERROR
            "Failed to stage shared runtime '${_runtime_real_path}' at '${_runtime_output}'")
    endif()
    list(APPEND _staged_runtime_names "${_runtime_name}")
    list(APPEND _staged_package_names "${_runtime_name}")
    message(STATUS "Staged shared runtime: ${_runtime_output}")
endforeach()

# Preserve compatibility filenames that intentionally differ from a library's
# SONAME in the classifier payload. They are package-only aliases: the preload
# manifest contains only canonical loader identities, and the ZLUDA policy below
# rewrites managed DT_NEEDED aliases to those canonical identities. This avoids
# loading byte-identical alias copies as independent HIP/ZLUDA runtimes.
foreach(_runtime_alias_entry IN LISTS _runtime_alias_entries)
    string(REPLACE "|" ";" _runtime_alias_parts "${_runtime_alias_entry}")
    list(GET _runtime_alias_parts 0 _runtime_alias_source)
    list(GET _runtime_alias_parts 1 _runtime_alias_name)
    if(_runtime_alias_name MATCHES "[/\\\\]")
        message(FATAL_ERROR
            "Unsafe runtime dependency alias '${_runtime_alias_name}'")
    endif()
    get_filename_component(_runtime_alias_real "${_runtime_alias_source}" REALPATH)
    _shared_runtime_loader_name(_runtime_alias_target
        "${_runtime_alias_real}")
    list(FIND _staged_runtime_names "${_runtime_alias_target}"
        _runtime_alias_target_index)
    if(_runtime_alias_target_index EQUAL -1)
        message(FATAL_ERROR
            "Runtime alias '${_runtime_alias_name}' targets canonical runtime "
            "'${_runtime_alias_target}', which was not staged")
    endif()
    list(FIND _staged_runtime_alias_names "${_runtime_alias_name}"
        _runtime_alias_mapping_index)
    if(_runtime_alias_mapping_index EQUAL -1)
        list(APPEND _staged_runtime_alias_names "${_runtime_alias_name}")
        list(APPEND _staged_runtime_alias_targets "${_runtime_alias_target}")
        list(APPEND _staged_runtime_alias_mappings
            "${_runtime_alias_name}->${_runtime_alias_target}")
    else()
        list(GET _staged_runtime_alias_targets
            ${_runtime_alias_mapping_index} _previous_runtime_alias_target)
        if(NOT _previous_runtime_alias_target STREQUAL _runtime_alias_target)
            message(FATAL_ERROR
                "Runtime alias '${_runtime_alias_name}' maps to both "
                "'${_previous_runtime_alias_target}' and "
                "'${_runtime_alias_target}'")
        endif()
    endif()
    set(_runtime_alias_output "${OUTPUT_DIR}/${_runtime_alias_name}")
    list(FIND _staged_package_names "${_runtime_alias_name}"
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
        COMMAND "${CMAKE_COMMAND}"
            "-DINPUT_FILE=${_runtime_alias_real}"
            "-DOUTPUT_FILE=${_runtime_alias_output}"
            -P "${CMAKE_CURRENT_LIST_DIR}/StageArtifact.cmake"
        RESULT_VARIABLE _runtime_alias_copy_result)
    if(NOT _runtime_alias_copy_result EQUAL 0 OR
       NOT EXISTS "${_runtime_alias_output}")
        message(FATAL_ERROR
            "Failed to stage runtime alias '${_runtime_alias_name}'")
    endif()
    list(APPEND _staged_package_names "${_runtime_alias_name}")
    message(STATUS "Staged package-only runtime alias: ${_runtime_alias_output}")
endforeach()

# JavaCPP extracts every classifier member into one directory, but the dynamic
# loader does not search a DSO's sibling directory unless that DSO says so.
# Normalize every Linux ZLUDA/ROCm library, including the linked backend, to
# canonical managed DT_NEEDED names and a relocatable RUNPATH before
# materializing the classifier package. This keeps build-host CUDA/ROCm paths
# out of the consumer contract and guarantees a single runtime identity.
if(DEFINED RUNTIME_POLICY AND RUNTIME_POLICY STREQUAL "zluda-amd" AND
   UNIX AND NOT APPLE)
    set(_zluda_runpath_files "")
    foreach(_staged_runtime_name IN LISTS _staged_package_names)
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
        _normalize_zluda_needed_aliases("${_zluda_runpath_file}")
        _set_zluda_origin_runpath("${_zluda_runpath_file}")
        message(STATUS
            "Set self-contained ZLUDA RUNPATH: ${_zluda_runpath_file}")
    endforeach()
endif()

# A binding module must not rediscover native dependencies or copy broad build
# directory globs.  When PACKAGE_DIR is supplied, materialize the exact package
# payload selected above: the linked backend, every canonical manifest-owned
# runtime, compatibility aliases, and the manifest itself. Keeping this
# operation in CMake makes native dependency
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
    foreach(_staged_runtime_name IN LISTS _staged_package_names)
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

    # rocBLAS loads its Tensile dispatch data relative to a bundled
    # rocblas/library directory. Shared-library dependency closure alone cannot
    # discover these non-ELF resources, so copy the version-matched ROCm data
    # tree into the classifier at the path expected by librocblas.so.
    set(_rocblas_resource_dirs "")
    foreach(_runtime_search_root IN LISTS _runtime_search_roots)
        foreach(_rocblas_candidate
                "${_runtime_search_root}/rocblas/library"
                "${_runtime_search_root}/lib/rocblas/library"
                "${_runtime_search_root}/lib64/rocblas/library"
                "${_runtime_search_root}/lib/x86_64-linux-gnu/rocblas/library")
            if(IS_DIRECTORY "${_rocblas_candidate}")
                list(APPEND _rocblas_resource_dirs "${_rocblas_candidate}")
            endif()
        endforeach()
    endforeach()

    # ROCm package revisions differ on whether the install step preserves the
    # intermediate `library` directory. Locate the sentinel recursively so both
    # lib/rocblas/TensileLibrary.dat and lib/rocblas/library/TensileLibrary.dat
    # layouts are accepted while the classifier path remains canonical.
    foreach(_runtime_search_root IN LISTS _runtime_search_roots)
        file(GLOB_RECURSE _rocblas_tensile_candidates LIST_DIRECTORIES FALSE
            "${_runtime_search_root}/TensileLibrary.dat"
            "${_runtime_search_root}/*/TensileLibrary.dat")
        foreach(_rocblas_tensile_candidate IN LISTS _rocblas_tensile_candidates)
            get_filename_component(_rocblas_tensile_dir
                "${_rocblas_tensile_candidate}" DIRECTORY)
            list(APPEND _rocblas_resource_dirs "${_rocblas_tensile_dir}")
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _rocblas_resource_dirs)

    # TheRock-based ROCm Core SDKs replace the legacy Tensile directory with
    # architecture-specific kernel packs. Preserve the SDK-relative .kpack
    # layout in the classifier and copy exactly the groups attested by the
    # release SDK contract. The declared resource format and group list come
    # from release attestation; never infer a fallback from partial contents.
    set(_rocm_kpack_dirs "")
    set(_rocm_kpack_groups "")
    if(DEFINED ROCM_RESOURCE_FORMAT AND ROCM_RESOURCE_FORMAT STREQUAL "kpack")
        if(NOT DEFINED ROCM_RUNTIME_ARCH OR ROCM_RUNTIME_ARCH STREQUAL "")
            message(FATAL_ERROR
                "ROCm kpack runtime staging requires an attested ROCM_RUNTIME_ARCH")
        endif()
        if(NOT DEFINED ROCM_KERNEL_PACK_GROUPS_CSV OR
           ROCM_KERNEL_PACK_GROUPS_CSV STREQUAL "")
            message(FATAL_ERROR
                "ROCm kpack runtime staging requires attested kernel-pack groups")
        endif()
        string(REPLACE "," ";" _rocm_kpack_groups
            "${ROCM_KERNEL_PACK_GROUPS_CSV}")
        list(REMOVE_DUPLICATES _rocm_kpack_groups)
        foreach(_rocm_kpack_group IN LISTS _rocm_kpack_groups)
            if(NOT _rocm_kpack_group MATCHES "^[A-Za-z0-9_-]+$")
                message(FATAL_ERROR
                    "Invalid ROCm kernel-pack group '${_rocm_kpack_group}'")
            endif()
        endforeach()
        foreach(_runtime_search_root IN LISTS _runtime_search_roots)
            if(IS_DIRECTORY "${_runtime_search_root}/.kpack")
                list(APPEND _rocm_kpack_dirs "${_runtime_search_root}/.kpack")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _rocm_kpack_dirs)
        list(LENGTH _rocm_kpack_dirs _rocm_kpack_dir_count)
        if(NOT _rocm_kpack_dir_count EQUAL 1)
            message(FATAL_ERROR
                "ROCm ${ROCM_RUNTIME_ARCH} requires exactly one canonical .kpack root; found ${_rocm_kpack_dir_count}")
        endif()
    endif()

    set(_rocblas_runtime_present FALSE)
    foreach(_staged_runtime_name IN LISTS _staged_runtime_names)
        if(_staged_runtime_name MATCHES "^librocblas[.]so($|[.])")
            set(_rocblas_runtime_present TRUE)
            break()
        endif()
    endforeach()
    set(_rocblas_tensile_found FALSE)
    set(_rocblas_resource_count 0)
    foreach(_rocblas_resource_dir IN LISTS _rocblas_resource_dirs)
        file(GLOB_RECURSE _rocblas_resource_files LIST_DIRECTORIES FALSE
            "${_rocblas_resource_dir}/*")
        foreach(_rocblas_resource_file IN LISTS _rocblas_resource_files)
            file(RELATIVE_PATH _rocblas_resource_relative
                "${_rocblas_resource_dir}" "${_rocblas_resource_file}")
            set(_rocblas_resource_output
                "${_runtime_package_dir_absolute}/rocblas/library/${_rocblas_resource_relative}")
            get_filename_component(_rocblas_resource_output_dir
                "${_rocblas_resource_output}" DIRECTORY)
            file(MAKE_DIRECTORY "${_rocblas_resource_output_dir}")
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${_rocblas_resource_file}" "${_rocblas_resource_output}"
                RESULT_VARIABLE _rocblas_resource_copy_result)
            if(NOT _rocblas_resource_copy_result EQUAL 0 OR
               NOT EXISTS "${_rocblas_resource_output}")
                message(FATAL_ERROR
                    "Failed to stage rocBLAS resource '${_rocblas_resource_file}'")
            endif()
            get_filename_component(_rocblas_resource_name
                "${_rocblas_resource_file}" NAME)
            if(_rocblas_resource_name STREQUAL "TensileLibrary.dat")
                set(_rocblas_tensile_found TRUE)
            endif()
            list(APPEND _staged_resource_names
                "rocblas/library/${_rocblas_resource_relative}")
            math(EXPR _rocblas_resource_count "${_rocblas_resource_count} + 1")
        endforeach()
    endforeach()

    set(_rocm_kpack_resource_count 0)
    if(_rocm_kpack_dirs)
        list(GET _rocm_kpack_dirs 0 _rocm_kpack_dir)
        foreach(_rocm_kpack_group IN LISTS _rocm_kpack_groups)
            set(_rocm_kpack_name
                "${_rocm_kpack_group}_lib_${ROCM_RUNTIME_ARCH}.kpack")
            set(_rocm_kpack_file
                "${_rocm_kpack_dir}/${_rocm_kpack_name}")
            if(NOT EXISTS "${_rocm_kpack_file}" OR
               IS_DIRECTORY "${_rocm_kpack_file}")
                message(FATAL_ERROR
                    "ROCm ${ROCM_RUNTIME_ARCH} canonical ${_rocm_kpack_group} kernel pack is missing: ${_rocm_kpack_file}")
            endif()
            file(SIZE "${_rocm_kpack_file}" _rocm_kpack_size)
            if(_rocm_kpack_size EQUAL 0)
                message(FATAL_ERROR
                    "ROCm kernel pack is empty: ${_rocm_kpack_file}")
            endif()
            set(_rocm_kpack_output
                "${_runtime_package_dir_absolute}/.kpack/${_rocm_kpack_name}")
            get_filename_component(_rocm_kpack_output_dir
                "${_rocm_kpack_output}" DIRECTORY)
            file(MAKE_DIRECTORY "${_rocm_kpack_output_dir}")
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${_rocm_kpack_file}" "${_rocm_kpack_output}"
                RESULT_VARIABLE _rocm_kpack_copy_result)
            if(NOT _rocm_kpack_copy_result EQUAL 0 OR
               NOT EXISTS "${_rocm_kpack_output}")
                message(FATAL_ERROR
                    "Failed to stage ROCm kernel pack '${_rocm_kpack_file}'")
            endif()
            list(APPEND _staged_resource_names
                ".kpack/${_rocm_kpack_name}")
            math(EXPR _rocm_kpack_resource_count
                "${_rocm_kpack_resource_count} + 1")
        endforeach()
    endif()
    list(REMOVE_DUPLICATES _staged_resource_names)
    if(NOT (DEFINED ROCM_RESOURCE_FORMAT AND
            ROCM_RESOURCE_FORMAT STREQUAL "kpack") AND
       _rocblas_runtime_present AND NOT _rocblas_tensile_found)
        message(FATAL_ERROR
            "librocblas.so is staged but its rocblas/library/TensileLibrary.dat "
            "resource tree was not found in the declared ROCm runtime roots")
    endif()
    if(_rocblas_resource_count GREATER 0)
        message(STATUS
            "Staged rocBLAS dispatch resources: ${_rocblas_resource_count} files")
    endif()
    if(_rocm_kpack_resource_count GREATER 0)
        message(STATUS
            "Staged ROCm ${ROCM_RUNTIME_ARCH} kernel packs: ${_rocm_kpack_resource_count} files")
    endif()

    list(LENGTH _runtime_package_sources _runtime_package_file_count)
    message(STATUS
        "Materialized classifier runtime package at "
        "${_runtime_package_dir_absolute} (${_runtime_package_file_count} libraries)")
endif()

# Preserve the dependency-first order produced above. Resource entries are
# extraction metadata, not preload entries; SharedCompilerRuntime materializes
# them beside the classifier libraries before rocBLAS is loaded.
list(LENGTH _staged_runtime_names _runtime_count)
set(_runtime_manifest
    "# nd4j-shared-runtime-manifest-v1\n# runtime-count=${_runtime_count}\n")
if(DEFINED PACKAGE_DIR AND NOT PACKAGE_DIR STREQUAL "")
    list(SORT _staged_runtime_alias_mappings)
    list(LENGTH _staged_runtime_alias_mappings _runtime_alias_count)
    string(APPEND _runtime_manifest
        "# runtime-alias-count=${_runtime_alias_count}\n")
    foreach(_runtime_alias_mapping IN LISTS _staged_runtime_alias_mappings)
        string(APPEND _runtime_manifest
            "# runtime-alias=${_runtime_alias_mapping}\n")
    endforeach()
endif()
list(REMOVE_DUPLICATES _staged_resource_names)
list(SORT _staged_resource_names)
list(LENGTH _staged_resource_names _resource_count)
string(APPEND _runtime_manifest "# resource-count=${_resource_count}\n")
foreach(_staged_resource_name IN LISTS _staged_resource_names)
    string(APPEND _runtime_manifest
        "# resource=${_staged_resource_name}\n")
endforeach()
if(_staged_runtime_names)
    string(REPLACE ";" "\n" _runtime_entries "${_staged_runtime_names}")
    string(APPEND _runtime_manifest "${_runtime_entries}\n")
endif()
file(WRITE "${_runtime_manifest_path}" "${_runtime_manifest}")
if(DEFINED PACKAGE_DIR AND NOT PACKAGE_DIR STREQUAL "")
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
endif()

# Build-only metadata: JavaCPP consumes this from platform.linkpath to compile its
# JNI wrapper with the exact C++ driver selected by CMake. The binding POM copies
# only shared-runtime-manifest.txt, so no build-host compiler path enters the jar.
file(WRITE "${OUTPUT_DIR}/javacpp-build-toolchain.properties"
    "platform.compiler=${_cxx_compiler_real}\n")
