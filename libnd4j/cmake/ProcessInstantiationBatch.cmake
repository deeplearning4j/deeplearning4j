# cmake/ProcessInstantiationBatch.cmake
# Worker script for processing a batch of source files in parallel
# This script is executed as a separate CMake process for each batch

# Get batch ID from command line
if(NOT DEFINED BATCH_ID)
    message(FATAL_ERROR "BATCH_ID not defined")
endif()

message("[Batch ${BATCH_ID}] Starting batch processor")

# Determine paths
get_filename_component(CMAKE_SCRIPT_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
set(CMAKE_SOURCE_DIR "${CMAKE_SCRIPT_DIR}/..")
set(INST_DIR "${CMAKE_SOURCE_DIR}/instantiation_analysis")
set(INST_BATCH_DIR "${INST_DIR}/batches")

# Load configuration
set(BATCH_CONFIG_FILE "${INST_BATCH_DIR}/config.cmake")
if(NOT EXISTS ${BATCH_CONFIG_FILE})
    message(FATAL_ERROR "[Batch ${BATCH_ID}] Configuration file not found: ${BATCH_CONFIG_FILE}")
endif()

include(${BATCH_CONFIG_FILE})

# Include helper functions
include(${CMAKE_SCRIPT_DIR}/InstantiationHelpers.cmake)

# Read list of files to process
set(BATCH_FILE "${INST_BATCH_DIR}/batch_${BATCH_ID}.txt")
if(NOT EXISTS ${BATCH_FILE})
    message(WARNING "[Batch ${BATCH_ID}] Batch file not found: ${BATCH_FILE}")
    file(WRITE "${INST_BATCH_DIR}/batch_${BATCH_ID}.done" "")
    return()
endif()

file(STRINGS ${BATCH_FILE} FILES_TO_PROCESS)
list(LENGTH FILES_TO_PROCESS batch_size)
message("[Batch ${BATCH_ID}] Processing ${batch_size} files")

# Process each file in the batch
set(batch_results "")
set(processed_count 0)
set(failed_count 0)

foreach(src IN LISTS FILES_TO_PROCESS)
    if(NOT EXISTS ${src})
        message("[Batch ${BATCH_ID}] Warning: File not found: ${src}")
        math(EXPR failed_count "${failed_count} + 1")
        continue()
    endif()
    
    # Extract file information
    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_path ${src} PATH)
    get_filename_component(src_full_name ${src} NAME)
    file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${src_path})
    string(REPLACE "/" "_" safe_name "${rel_path}_${src_name}")
    
    # Determine if this is a generated file
    set(IS_GENERATED FALSE)
    if(src MATCHES "${CMAKE_BINARY_DIR}")
        set(IS_GENERATED TRUE)
    endif()
    
    message("[Batch ${BATCH_ID}] [${processed_count}/${batch_size}] Analyzing: ${src_full_name}")
    
    # Create a sub-temp directory for this batch
    set(BATCH_TEMP_DIR "${INST_TEMP_DIR}/batch_${BATCH_ID}")
    if(NOT EXISTS ${BATCH_TEMP_DIR})
        file(MAKE_DIRECTORY ${BATCH_TEMP_DIR})
    endif()
    set(INST_TEMP_DIR ${BATCH_TEMP_DIR})
    
    # Extract used templates
    set(used_file "${INST_USED_DIR}/${safe_name}.used")
    extract_used_templates("${src}" "${used_file}" "${safe_name}")
    
    # Extract provided templates
    set(provided_file "${INST_PROVIDED_DIR}/${safe_name}.provided")
    extract_provided_templates("${src}" "${provided_file}" "${safe_name}")
    
    # Calculate missing templates
    set(missing_templates "")
    set(missing_count 0)
    
    # Read used templates
    if(EXISTS ${used_file})
        file(STRINGS ${used_file} used_list)
    else()
        set(used_list "")
    endif()
    
    # Read provided templates
    if(EXISTS ${provided_file})
        file(STRINGS ${provided_file} provided_list)
    else()
        set(provided_list "")
    endif()
    
    # Simple missing calculation (can be enhanced with normalization)
    foreach(used_tmpl ${used_list})
        set(found FALSE)
        foreach(provided_tmpl ${provided_list})
            if(used_tmpl STREQUAL provided_tmpl)
                set(found TRUE)
                break()
            endif()
        endforeach()
        
        if(NOT found)
            list(APPEND missing_templates "${used_tmpl}")
        endif()
    endforeach()
    
    # Write missing templates
    set(missing_file "${INST_MISSING_DIR}/${safe_name}.missing")
    if(missing_templates)
        list(REMOVE_DUPLICATES missing_templates)
        string(REPLACE ";" "\n" missing_content "${missing_templates}")
        file(WRITE ${missing_file} "${missing_content}")
        list(LENGTH missing_templates missing_count)
    else()
        file(WRITE ${missing_file} "")
        set(missing_count 0)
    endif()
    
    # Count templates
    list(LENGTH used_list used_count)
    list(LENGTH provided_list provided_count)
    
    # Create per-file summary
    file(WRITE ${INST_BY_FILE_DIR}/${safe_name}.summary "File: ${src_full_name}\n")
    file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "Path: ${rel_path}\n")
    if(IS_GENERATED)
        file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "Type: generated\n")
    else()
        file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "Type: original\n")
    endif()
    file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "Used: ${used_count}\n")
    file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "Provided: ${provided_count}\n")
    file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "Missing: ${missing_count}\n")
    
    # Add details if there are templates
    if(used_count GREATER 0)
        file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "\nUsed Templates (first 10):\n")
        set(shown 0)
        foreach(tmpl ${used_list})
            if(shown LESS 10)
                file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "  - ${tmpl}\n")
                math(EXPR shown "${shown} + 1")
            else()
                break()
            endif()
        endforeach()
    endif()
    
    if(provided_count GREATER 0)
        file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "\nProvided Templates (first 10):\n")
        set(shown 0)
        foreach(tmpl ${provided_list})
            if(shown LESS 10)
                file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "  - ${tmpl}\n")
                math(EXPR shown "${shown} + 1")
            else()
                break()
            endif()
        endforeach()
    endif()
    
    if(missing_count GREATER 0)
        file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "\nMissing Templates:\n")
        foreach(tmpl ${missing_templates})
            file(APPEND ${INST_BY_FILE_DIR}/${safe_name}.summary "  - ${tmpl}\n")
        endforeach()
    endif()
    
    # Append to batch results
    string(APPEND batch_results "${src}|${safe_name}|${used_count}|${provided_count}|${missing_count}|${IS_GENERATED}\n")
    
    math(EXPR processed_count "${processed_count} + 1")
    
    # Progress reporting
    math(EXPR progress_pct "(${processed_count} * 100) / ${batch_size}")
    if(progress_pct GREATER 0 AND NOT progress_pct EQUAL last_progress)
        message("[Batch ${BATCH_ID}] Progress: ${progress_pct}% (${processed_count}/${batch_size})")
        set(last_progress ${progress_pct})
    endif()
endforeach()

# Clean up batch temp directory
if(EXISTS ${BATCH_TEMP_DIR})
    file(REMOVE_RECURSE ${BATCH_TEMP_DIR})
endif()

# Write batch completion marker with results
file(WRITE "${INST_BATCH_DIR}/batch_${BATCH_ID}.done" "${batch_results}")

message("[Batch ${BATCH_ID}] Completed: ${processed_count} processed, ${failed_count} failed")

# Write batch summary
set(BATCH_SUMMARY_FILE "${INST_BATCH_DIR}/batch_${BATCH_ID}.summary")
file(WRITE ${BATCH_SUMMARY_FILE} "Batch ${BATCH_ID} Summary\n")
file(APPEND ${BATCH_SUMMARY_FILE} "====================\n")
file(APPEND ${BATCH_SUMMARY_FILE} "Files processed: ${processed_count}\n")
file(APPEND ${BATCH_SUMMARY_FILE} "Files failed: ${failed_count}\n")
file(APPEND ${BATCH_SUMMARY_FILE} "Total in batch: ${batch_size}\n")
