# cmake/InstantiationCache.cmake
# Smart caching for instantiation analysis

function(should_process_file SOURCE_FILE SAFE_NAME OUT_VAR)
    # Get file modification time
    file(TIMESTAMP ${SOURCE_FILE} source_mtime)
    
    # Check if cached results exist
    set(cache_file "${INST_TEMP_DIR}/${SAFE_NAME}.cache")
    
    if(EXISTS ${cache_file})
        # Read cache timestamp
        file(READ ${cache_file} cache_content LIMIT 100)
        string(REGEX MATCH "timestamp:([0-9]+)" _ ${cache_content})
        set(cache_time ${CMAKE_MATCH_1})
        
        # Compare timestamps
        if(cache_time AND source_mtime EQUAL cache_time)
            # Cache is valid, skip processing
            set(${OUT_VAR} FALSE PARENT_SCOPE)
            
            # Load cached results
            if(EXISTS ${INST_USED_DIR}/${SAFE_NAME}.used.cached)
                file(COPY ${INST_USED_DIR}/${SAFE_NAME}.used.cached
                     DESTINATION ${INST_USED_DIR}
                     FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
                file(RENAME ${INST_USED_DIR}/${SAFE_NAME}.used.cached
                           ${INST_USED_DIR}/${SAFE_NAME}.used)
            endif()
            
            if(EXISTS ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided.cached)
                file(COPY ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided.cached
                     DESTINATION ${INST_PROVIDED_DIR}
                     FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
                file(RENAME ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided.cached
                           ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided)
            endif()
            
            return()
        endif()
    endif()
    
    # Need to process this file
    set(${OUT_VAR} TRUE PARENT_SCOPE)
endfunction()

function(cache_analysis_results SOURCE_FILE SAFE_NAME)
    file(TIMESTAMP ${SOURCE_FILE} source_mtime)
    
    # Write cache marker
    set(cache_file "${INST_TEMP_DIR}/${SAFE_NAME}.cache")
    file(WRITE ${cache_file} "timestamp:${source_mtime}\n")
    
    # Cache the analysis results
    if(EXISTS ${INST_USED_DIR}/${SAFE_NAME}.used)
        file(COPY ${INST_USED_DIR}/${SAFE_NAME}.used
             DESTINATION ${INST_USED_DIR}
             FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
        file(RENAME ${INST_USED_DIR}/${SAFE_NAME}.used
                   ${INST_USED_DIR}/${SAFE_NAME}.used.cached)
    endif()
    
    if(EXISTS ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided)
        file(COPY ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided
             DESTINATION ${INST_PROVIDED_DIR}
             FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
        file(RENAME ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided
                   ${INST_PROVIDED_DIR}/${SAFE_NAME}.provided.cached)
    endif()
endfunction()
