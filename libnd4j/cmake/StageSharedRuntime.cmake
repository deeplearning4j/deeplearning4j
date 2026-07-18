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
            message(FATAL_ERROR
                "Shared runtime '${_library_path}' has no ELF SONAME")
        endif()
        set(_loader_name "${CMAKE_MATCH_1}")
    endif()

    set(${_out_var} "${_loader_name}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED OUTPUT_DIR OR OUTPUT_DIR STREQUAL "")
    message(FATAL_ERROR "OUTPUT_DIR is required when staging shared runtimes")
endif()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

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

list(SORT _staged_runtime_names)
list(LENGTH _staged_runtime_names _runtime_count)
set(_runtime_manifest
    "# nd4j-shared-runtime-manifest-v1\n# runtime-count=${_runtime_count}\n")
if(_staged_runtime_names)
    string(REPLACE ";" "\n" _runtime_entries "${_staged_runtime_names}")
    string(APPEND _runtime_manifest "${_runtime_entries}\n")
endif()
file(WRITE "${_runtime_manifest_path}" "${_runtime_manifest}")

# Build-only metadata: JavaCPP consumes this from platform.linkpath to compile its
# JNI wrapper with the exact C++ driver selected by CMake. The binding POM copies
# only shared-runtime-manifest.txt, so no build-host compiler path enters the jar.
file(WRITE "${OUTPUT_DIR}/javacpp-build-toolchain.properties"
    "platform.compiler=${_cxx_compiler_real}\n")
