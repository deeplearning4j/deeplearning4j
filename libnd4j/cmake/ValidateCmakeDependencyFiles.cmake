# ValidateCmakeDependencyFiles.cmake
#
# CMake's Makefile dependency scanner assumes compiler-generated .d files are
# textual make dependencies. An interrupted compiler-cache write can leave a
# valid object beside a zero-filled or otherwise malformed depfile; consuming
# that file can abort CMake before the compiler launcher has a chance to repair
# the cache entry. Remove only malformed generated depfiles so the normal build
# regenerates them through the compiler launcher.
#
# Required:
#   BUILD_DIR  configured CMake binary directory to validate

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED BUILD_DIR OR NOT IS_DIRECTORY "${BUILD_DIR}")
    message(FATAL_ERROR
        "ValidateCmakeDependencyFiles: BUILD_DIR is missing or invalid: '${BUILD_DIR}'")
endif()

file(GLOB_RECURSE _sd_dependency_files LIST_DIRECTORIES FALSE
    "${BUILD_DIR}/*.o.d"
    "${BUILD_DIR}/*.obj.d")

set(_sd_removed_dependency_files 0)
foreach(_sd_dependency_file IN LISTS _sd_dependency_files)
    file(SIZE "${_sd_dependency_file}" _sd_dependency_file_size)
    set(_sd_dependency_file_valid TRUE)

    if(_sd_dependency_file_size EQUAL 0)
        set(_sd_dependency_file_valid FALSE)
    else()
        # A GCC-style depfile starts with a make target followed by ':'. Reading
        # strings also rejects NUL-filled/binary cache artifacts without making
        # assumptions about target paths or dependency line wrapping.
        file(STRINGS "${_sd_dependency_file}" _sd_dependency_header
            LIMIT_COUNT 1
            LIMIT_INPUT 65536
            REGEX "^[^:]+:")
        if(NOT _sd_dependency_header)
            set(_sd_dependency_file_valid FALSE)
        endif()
    endif()

    if(NOT _sd_dependency_file_valid)
        file(REMOVE "${_sd_dependency_file}")
        if(EXISTS "${_sd_dependency_file}")
            message(FATAL_ERROR
                "ValidateCmakeDependencyFiles: failed to remove malformed generated depfile: ${_sd_dependency_file}")
        endif()
        math(EXPR _sd_removed_dependency_files
            "${_sd_removed_dependency_files} + 1")
    endif()
endforeach()

message(STATUS
    "Validated CMake compiler dependency files in ${BUILD_DIR}; removed ${_sd_removed_dependency_files} malformed file(s)")
