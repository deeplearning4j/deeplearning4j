# =============================================================================
# CUDA TEMPORARY FILE CLEANUP HOOKS
#
# This module provides functions and hooks for cleaning up CUDA temporary files
# that accumulate during builds. NVCC creates tmpxft_* files in /tmp that can
# consume significant disk space and are not always cleaned up properly.
#
# Usage:
#   include(CudaCleanup)
#   cuda_cleanup_configure_time()  # Clean at cmake configure time
#   cuda_create_cleanup_target()   # Create 'cuda_cleanup' target
#
# The cleanup target can be used as a dependency:
#   add_dependencies(my_cuda_target cuda_cleanup_pre_build)
# =============================================================================

# Track if cleanup has already been performed this configure
set(_CUDA_CLEANUP_PERFORMED FALSE CACHE INTERNAL "Track if cleanup was done")

# =============================================================================
# FUNCTION: cuda_cleanup_tmp_files
# Cleans NVCC temporary files from /tmp directory
# =============================================================================
function(cuda_cleanup_tmp_files)
    set(options QUIET VERBOSE DRY_RUN)
    set(oneValueArgs MAX_AGE_MINUTES)
    set(multiValueArgs "")
    cmake_parse_arguments(CLEANUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default to cleaning files older than 0 minutes (all files) if not specified
    if(NOT DEFINED CLEANUP_MAX_AGE_MINUTES)
        set(CLEANUP_MAX_AGE_MINUTES 0)
    endif()

    if(NOT CLEANUP_QUIET)
        message(STATUS "🧹 Cleaning CUDA temporary files from /tmp...")
    endif()

    # Check both /tmp and build directory for temp files
    # NOTE: Do NOT clean compiler_tmp — GCC uses it for active .s assembler temp files.
    # Cleaning it during builds causes "can't open .s for reading" errors.
    set(CLEANUP_PATTERNS
        "/tmp/tmpxft_*"
        "/tmp/cuda-dbg"
        "/tmp/fatbin*"
        "/tmp/nvcc*"
        "${CMAKE_BINARY_DIR}/nvcc_tmp/*"
    )

    set(TOTAL_CLEANED 0)
    set(TOTAL_SIZE 0)

    foreach(pattern ${CLEANUP_PATTERNS})
        file(GLOB files_to_clean "${pattern}")
        list(LENGTH files_to_clean file_count)

        if(file_count GREATER 0)
            foreach(file_path ${files_to_clean})
                # Check if file/directory exists
                if(EXISTS "${file_path}")
                    # Get file info for reporting
                    if(IS_DIRECTORY "${file_path}")
                        set(file_type "directory")
                    else()
                        set(file_type "file")
                        # Try to get file size
                        file(SIZE "${file_path}" file_size)
                        math(EXPR TOTAL_SIZE "${TOTAL_SIZE} + ${file_size}")
                    endif()

                    if(CLEANUP_VERBOSE)
                        message(STATUS "   Removing ${file_type}: ${file_path}")
                    endif()

                    if(NOT CLEANUP_DRY_RUN)
                        file(REMOVE_RECURSE "${file_path}")
                        math(EXPR TOTAL_CLEANED "${TOTAL_CLEANED} + 1")
                    else()
                        if(NOT CLEANUP_QUIET)
                            message(STATUS "   [DRY-RUN] Would remove: ${file_path}")
                        endif()
                    endif()
                endif()
            endforeach()
        endif()
    endforeach()

    if(NOT CLEANUP_QUIET)
        if(TOTAL_CLEANED GREATER 0)
            # Convert bytes to MB for display
            math(EXPR size_mb "${TOTAL_SIZE} / 1048576")
            message(STATUS "✅ Cleaned ${TOTAL_CLEANED} CUDA temp files (~${size_mb} MB)")
        else()
            message(STATUS "ℹ️  No CUDA temp files found to clean")
        endif()
    endif()

    # Return count to parent scope
    set(CUDA_CLEANUP_COUNT ${TOTAL_CLEANED} PARENT_SCOPE)
endfunction()

# =============================================================================
# FUNCTION: cuda_cleanup_build_intermediates
# Cleans intermediate CUDA files from the build directory
# =============================================================================
function(cuda_cleanup_build_intermediates)
    set(options QUIET VERBOSE DRY_RUN)
    set(oneValueArgs BUILD_DIR)
    set(multiValueArgs "")
    cmake_parse_arguments(CLEANUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT CLEANUP_BUILD_DIR)
        set(CLEANUP_BUILD_DIR "${CMAKE_BINARY_DIR}")
    endif()

    if(NOT CLEANUP_QUIET)
        message(STATUS "🧹 Cleaning CUDA intermediate files from build directory...")
    endif()

    # Patterns for intermediate files that can be safely removed
    # Note: We're careful not to remove .ptx files that might be needed
    set(CLEANUP_PATTERNS
        "*.ii"           # Preprocessed files
        "*.cudafe1.cpp"  # CUDA frontend output
        "*.cudafe1.c"
        "*.cudafe1.stub.c"
        "*.cudafe2.c"
        "*.cudafe2.cpp"
        "*.module_id"
        "*.hash"
        "*.cpp1.ii"
        "*.cpp4.ii"
    )

    set(TOTAL_CLEANED 0)

    foreach(pattern ${CLEANUP_PATTERNS})
        file(GLOB_RECURSE files_to_clean "${CLEANUP_BUILD_DIR}/${pattern}")
        list(LENGTH files_to_clean file_count)

        if(file_count GREATER 0)
            foreach(file_path ${files_to_clean})
                if(EXISTS "${file_path}")
                    if(CLEANUP_VERBOSE)
                        message(STATUS "   Removing: ${file_path}")
                    endif()

                    if(NOT CLEANUP_DRY_RUN)
                        file(REMOVE "${file_path}")
                        math(EXPR TOTAL_CLEANED "${TOTAL_CLEANED} + 1")
                    endif()
                endif()
            endforeach()
        endif()
    endforeach()

    if(NOT CLEANUP_QUIET)
        if(TOTAL_CLEANED GREATER 0)
            message(STATUS "✅ Cleaned ${TOTAL_CLEANED} intermediate CUDA files")
        else()
            message(STATUS "ℹ️  No intermediate files found to clean")
        endif()
    endif()
endfunction()

# =============================================================================
# FUNCTION: cuda_cleanup_configure_time
# Performs cleanup at CMake configure time (when cmake is run)
# =============================================================================
function(cuda_cleanup_configure_time)
    set(options FORCE QUIET)
    set(oneValueArgs "")
    set(multiValueArgs "")
    cmake_parse_arguments(CLEANUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Only clean once per configure unless FORCE is specified
    if(_CUDA_CLEANUP_PERFORMED AND NOT CLEANUP_FORCE)
        if(NOT CLEANUP_QUIET)
            message(STATUS "ℹ️  CUDA cleanup already performed this configure")
        endif()
        return()
    endif()

    if(NOT CLEANUP_QUIET)
        message(STATUS "")
        message(STATUS "╔════════════════════════════════════════════════════════════╗")
        message(STATUS "║          CUDA TEMPORARY FILE CLEANUP (Configure)           ║")
        message(STATUS "╚════════════════════════════════════════════════════════════╝")
    endif()

    # Clean /tmp files
    if(CLEANUP_QUIET)
        cuda_cleanup_tmp_files(QUIET)
    else()
        cuda_cleanup_tmp_files()
    endif()

    # Mark cleanup as performed
    set(_CUDA_CLEANUP_PERFORMED TRUE CACHE INTERNAL "Track if cleanup was done" FORCE)

    if(NOT CLEANUP_QUIET)
        message(STATUS "")
    endif()
endfunction()

# =============================================================================
# FUNCTION: cuda_create_cleanup_target
# Creates custom targets for CUDA cleanup that can be used in the build
# =============================================================================
function(cuda_create_cleanup_target)
    # Don't create targets if not a CUDA build
    if(NOT SD_CUDA)
        return()
    endif()

    # Create the cleanup script that will be run
    set(CLEANUP_SCRIPT "${CMAKE_BINARY_DIR}/cuda_cleanup.sh")

    file(WRITE "${CLEANUP_SCRIPT}"
"#!/bin/bash
# CUDA Temporary File Cleanup Script
# Generated by CMake - do not edit manually

echo '🧹 Cleaning CUDA temporary files...'

# Count files before (both /tmp and build directory)
BEFORE=\$(find /tmp -maxdepth 1 -name 'tmpxft_*' 2>/dev/null | wc -l)
BUILD_BEFORE=\$(find ${CMAKE_BINARY_DIR}/nvcc_tmp -type f 2>/dev/null | wc -l)

# Remove tmpxft files from /tmp
rm -rf /tmp/tmpxft_* 2>/dev/null

# Remove cuda-dbg directory contents
rm -rf /tmp/cuda-dbg/* 2>/dev/null

# Remove other CUDA temp files from /tmp
rm -rf /tmp/fatbin* /tmp/nvcc* 2>/dev/null

# Clean build directory nvcc temp files (NOT compiler_tmp - GCC uses it for active .s files)
rm -rf ${CMAKE_BINARY_DIR}/nvcc_tmp/* 2>/dev/null

# Count files after
AFTER=\$(find /tmp -maxdepth 1 -name 'tmpxft_*' 2>/dev/null | wc -l)
BUILD_AFTER=\$(find ${CMAKE_BINARY_DIR}/nvcc_tmp -type f 2>/dev/null | wc -l)

CLEANED=\$((BEFORE - AFTER + BUILD_BEFORE - BUILD_AFTER))
if [ \$CLEANED -gt 0 ]; then
    echo \"✅ Cleaned \$CLEANED CUDA temp files\"
else
    echo 'ℹ️  No CUDA temp files to clean'
fi
")

    # Make script executable
    file(CHMOD "${CLEANUP_SCRIPT}"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

    # Create the cleanup target
    add_custom_target(cuda_cleanup
        COMMAND "${CLEANUP_SCRIPT}"
        COMMENT "Cleaning CUDA temporary files"
        VERBATIM
    )

    # Create a pre-build cleanup target that other targets can depend on
    add_custom_target(cuda_cleanup_pre_build
        COMMAND "${CLEANUP_SCRIPT}"
        COMMENT "Pre-build CUDA temporary file cleanup"
        VERBATIM
    )

    # Create a post-build cleanup target
    add_custom_target(cuda_cleanup_post_build
        COMMAND "${CLEANUP_SCRIPT}"
        COMMENT "Post-build CUDA temporary file cleanup"
        VERBATIM
    )

    message(STATUS "✅ Created CUDA cleanup targets: cuda_cleanup, cuda_cleanup_pre_build, cuda_cleanup_post_build")
endfunction()

# =============================================================================
# FUNCTION: cuda_add_cleanup_hook
# Adds cleanup as a dependency to an existing target
# =============================================================================
function(cuda_add_cleanup_hook target_name)
    set(options PRE_BUILD POST_BUILD)
    set(oneValueArgs "")
    set(multiValueArgs "")
    cmake_parse_arguments(HOOK "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT TARGET ${target_name})
        message(WARNING "Cannot add cleanup hook: target '${target_name}' does not exist")
        return()
    endif()

    if(HOOK_PRE_BUILD)
        if(TARGET cuda_cleanup_pre_build)
            add_dependencies(${target_name} cuda_cleanup_pre_build)
            message(STATUS "   Added pre-build cleanup hook to ${target_name}")
        endif()
    endif()

    if(HOOK_POST_BUILD)
        if(TARGET cuda_cleanup_post_build)
            add_custom_command(TARGET ${target_name}
                POST_BUILD
                COMMAND "${CMAKE_BINARY_DIR}/cuda_cleanup.sh"
                COMMENT "Post-build CUDA cleanup for ${target_name}"
                VERBATIM
            )
            message(STATUS "   Added post-build cleanup hook to ${target_name}")
        endif()
    endif()
endfunction()

# =============================================================================
# FUNCTION: cuda_estimate_tmp_usage
# Estimates how much space CUDA temp files are using
# =============================================================================
function(cuda_estimate_tmp_usage)
    execute_process(
        COMMAND sh -c "du -sh /tmp/tmpxft_* /tmp/cuda-dbg 2>/dev/null | awk '{sum+=$1} END {print sum}'"
        OUTPUT_VARIABLE TMP_SIZE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    execute_process(
        COMMAND sh -c "find /tmp -maxdepth 1 -name 'tmpxft_*' 2>/dev/null | wc -l"
        OUTPUT_VARIABLE TMP_COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(TMP_COUNT GREATER 0)
        message(STATUS "📊 CUDA temp file usage: ${TMP_COUNT} files in /tmp")
    endif()
endfunction()

# =============================================================================
# AUTO-INITIALIZATION
# When this module is included, optionally perform cleanup based on option
# =============================================================================
option(SD_CUDA_AUTO_CLEANUP "Automatically clean CUDA temp files at configure time" ON)

if(SD_CUDA AND SD_CUDA_AUTO_CLEANUP)
    # Only run if not already done
    if(NOT _CUDA_CLEANUP_PERFORMED)
        cuda_cleanup_configure_time()
    endif()
endif()
