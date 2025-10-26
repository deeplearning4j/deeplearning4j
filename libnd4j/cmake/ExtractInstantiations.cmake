# cmake/ExtractInstantiations.cmake
# COMPLETE CONSOLIDATED PARALLEL TEMPLATE INSTANTIATION EXTRACTION
# All functionality in ONE file - no separate helper files needed

message(STATUS "")
message(STATUS "=====================================================")
message(STATUS "=== TEMPLATE INSTANTIATION ANALYSIS MODE ACTIVE ===")
message(STATUS "=====================================================")
message(STATUS "SD_EXTRACT_INSTANTIATIONS = ${SD_EXTRACT_INSTANTIATIONS}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

# ============================================================================
# HELPER FUNCTIONS - ALL INLINE, NO SEPARATE FILES
# ============================================================================

# Function to normalize template signatures for comparison
function(normalize_template_signature TEMPLATE OUT_VAR)
    set(normalized "${TEMPLATE}")
    
    # Remove all extra whitespace
    string(REGEX REPLACE "[ \t\r\n]+" " " normalized "${normalized}")
    string(STRIP "${normalized}" normalized)
    
    # Normalize spacing around template brackets
    string(REPLACE "< " "<" normalized "${normalized}")
    string(REPLACE " >" ">" normalized "${normalized}")
    string(REPLACE " <" "<" normalized "${normalized}")
    string(REPLACE "> " ">" normalized "${normalized}")
    
    # Normalize pointer and reference notation
    string(REPLACE " *" "*" normalized "${normalized}")
    string(REPLACE " &" "&" normalized "${normalized}")
    string(REPLACE "* " "*" normalized "${normalized}")
    string(REPLACE "& " "&" normalized "${normalized}")
    
    # Normalize const placement
    string(REGEX REPLACE "const[ ]+([^ ]+)" "\\1 const" normalized "${normalized}")
    
    # Normalize namespace separators
    string(REPLACE " ::" "::" normalized "${normalized}")
    string(REPLACE ":: " "::" normalized "${normalized}")
    
    # Remove trailing const
    string(REGEX REPLACE "[ ]+const$" "" normalized "${normalized}")
    
    set(${OUT_VAR} "${normalized}" PARENT_SCOPE)
endfunction()

# Function to check if a template should be filtered out
function(should_filter_template TEMPLATE OUT_VAR)
    set(should_filter FALSE)
    
    # Filter out all std:: templates
    if(TEMPLATE MATCHES "^std::")
        set(should_filter TRUE)
    endif()
    
    # Filter out compiler intrinsics and internal templates
    if(TEMPLATE MATCHES "^__" OR 
       TEMPLATE MATCHES "^operator" OR
       TEMPLATE MATCHES "^decltype" OR
       TEMPLATE MATCHES "^typename" OR
       TEMPLATE MATCHES "^template")
        set(should_filter TRUE)
    endif()
    
    # Filter out basic types that aren't really templates
    if(TEMPLATE MATCHES "^(bool|char|short|int|long|float|double|void|unsigned|signed)")
        set(should_filter TRUE)
    endif()
    
    set(${OUT_VAR} ${should_filter} PARENT_SCOPE)
endfunction()

# Setup compilation flags function
function(setup_instantiation_flags)
    # Setup FlatBuffers
    set(PREPROCESS_FLATBUFFERS_VERSION "25.2.10")
    set(PREPROCESS_FLATBUFFERS_URL "https://github.com/google/flatbuffers/archive/v${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz")
    set(PREPROCESS_FB_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/preprocess-flatbuffers-src")
    
    set(PREPROCESS_FB_HEADER "${PREPROCESS_FB_SOURCE_DIR}/include/flatbuffers/flatbuffers.h")
    if(NOT EXISTS ${PREPROCESS_FB_HEADER})
        message("Downloading and extracting FlatBuffers for instantiation analysis...")
        
        file(DOWNLOAD ${PREPROCESS_FLATBUFFERS_URL} 
             "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz"
             SHOW_PROGRESS
             STATUS download_status)
        
        list(GET download_status 0 download_result)
        if(NOT download_result EQUAL 0)
            message(FATAL_ERROR "Failed to download FlatBuffers for instantiation analysis")
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            RESULT_VARIABLE extract_result
        )
        
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Failed to extract FlatBuffers for instantiation analysis")
        endif()

        file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}" 
             "${PREPROCESS_FB_SOURCE_DIR}")
    endif()
    
    # Build include paths
    set(all_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")
    file(GLOB_RECURSE all_dirs LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
    foreach(item ${all_dirs})
        if(IS_DIRECTORY ${item})
            list(APPEND all_includes ${item})
        endif()
    endforeach()

    list(APPEND all_includes
            "${CMAKE_CURRENT_BINARY_DIR}/include"
            "${CMAKE_SOURCE_DIR}/include/"
            "${CMAKE_SOURCE_DIR}/include/system"
            "${CMAKE_BINARY_DIR}/compilation_units"
            "${CMAKE_BINARY_DIR}/cpu_instantiations"
            "${CMAKE_BINARY_DIR}/cuda_instantiations"
            "${CMAKE_BINARY_DIR}/include"
            "${PREPROCESS_FB_SOURCE_DIR}/include"
    )
    
    # Build include flags string
    set(include_flags "")
    foreach(dir IN LISTS all_includes)
        if(EXISTS ${dir})
            string(APPEND include_flags " -I${dir}")
        endif()
    endforeach()
    
    # Build definition flags
    set(defs_flags "")
    string(APPEND defs_flags " -D__CPUBLAS__=true")
    if(SD_CUDA)
        string(APPEND defs_flags " -D__CUDABLAS__=true -DHAVE_CUDA=1")
    endif()
    if(HAVE_OPENBLAS)
        string(APPEND defs_flags " -DHAVE_OPENBLAS=1")
    endif()

    if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
        string(APPEND defs_flags " -DSD_ALL_OPS=1")
        string(APPEND defs_flags " -DNOT_EXCLUDED(x)=1")
    else()
        foreach(OP ${SD_OPS_LIST})
            string(APPEND defs_flags " -DOP_${OP}=1")
        endforeach()
        string(APPEND defs_flags " -DNOT_EXCLUDED(x)=\\(defined\\(x\\)\\)")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        string(APPEND defs_flags " -DDEBUG=1 -D_DEBUG=1")
    else()
        string(APPEND defs_flags " -DNDEBUG=1")
    endif()
    
    if(SD_TYPES_LIST)
        string(REPLACE ";" "," types_comma_list "${SD_TYPES_LIST}")
        string(APPEND defs_flags " -DSD_TYPES_LIST=\"${types_comma_list}\"")
    endif()
    
    # Export to parent scope
    set(INST_INCLUDE_FLAGS ${include_flags} PARENT_SCOPE)
    set(INST_DEFS_FLAGS ${defs_flags} PARENT_SCOPE)
    set(INST_LANG_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}" PARENT_SCOPE)
    
    # Find required tools
    find_program(NM_TOOL NAMES nm gcc-nm llvm-nm)
    find_program(CPPFILT_TOOL NAMES c++filt llvm-cxxfilt)
    
    set(NM_TOOL ${NM_TOOL} PARENT_SCOPE)
    set(CPPFILT_TOOL ${CPPFILT_TOOL} PARENT_SCOPE)
endfunction()

# Get source list function
function(get_source_list OUT_VAR)
    if(DEFINED ALL_SOURCES AND ALL_SOURCES)
        set(${OUT_VAR} ${ALL_SOURCES} PARENT_SCOPE)
    elseif(DEFINED ALL_SOURCES_LIST AND ALL_SOURCES_LIST)
        set(${OUT_VAR} ${ALL_SOURCES_LIST} PARENT_SCOPE)
    else()
        set(${OUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()

# MAIN EXTRACTION FUNCTION - CONSOLIDATED WITH ACCURATE FILTERING
function(extract_templates_consolidated SOURCE_FILE USED_OUTPUT_FILE PROVIDED_OUTPUT_FILE SAFE_NAME)
    # Initialize empty results
    set(used_templates "")
    set(provided_templates "")
    
    # Check if source file exists
    if(NOT EXISTS "${SOURCE_FILE}")
        file(WRITE ${USED_OUTPUT_FILE} "")
        file(WRITE ${PROVIDED_OUTPUT_FILE} "")
        return()
    endif()
    
    # Determine compiler and flags
    if(SOURCE_FILE MATCHES "\\.cu$")
        set(compiler "${CMAKE_CUDA_COMPILER}")
        set(lang_flags "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}}")
    else()
        set(compiler "${CMAKE_CXX_COMPILER}")
        set(lang_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
    endif()
    
    if(NOT compiler)
        file(WRITE ${USED_OUTPUT_FILE} "")
        file(WRITE ${PROVIDED_OUTPUT_FILE} "")
        return()
    endif()
    
    # SINGLE PREPROCESSING PASS
    separate_arguments(lang_flags_list UNIX_COMMAND "${lang_flags}")
    separate_arguments(defs_flags_list UNIX_COMMAND "${INST_DEFS_FLAGS}")
    separate_arguments(include_flags_list UNIX_COMMAND "${INST_INCLUDE_FLAGS}")
    
    execute_process(
        COMMAND ${compiler} 
            -E -P -C -dD
            ${lang_flags_list} 
            ${defs_flags_list} 
            ${include_flags_list} 
            "${SOURCE_FILE}"
        OUTPUT_VARIABLE preprocessed_content
        ERROR_VARIABLE preprocess_errors
        RESULT_VARIABLE preprocess_result
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NOT preprocess_result EQUAL 0)
        file(WRITE ${USED_OUTPUT_FILE} "")
        file(WRITE ${PROVIDED_OUTPUT_FILE} "")
        return()
    endif()
    
    string(LENGTH "${preprocessed_content}" content_length)
    if(content_length GREATER 52428800)
        file(WRITE ${USED_OUTPUT_FILE} "")
        file(WRITE ${PROVIDED_OUTPUT_FILE} "")
        return()
    endif()
    
    # EXTRACT PROVIDED TEMPLATES
    string(REGEX MATCHALL "template[ \t]+(?!extern)(class|struct)[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*[ \t]*<[^;>]+>[ \t]*;" 
           explicit_instantiations "${preprocessed_content}")
    
    foreach(inst ${explicit_instantiations})
        string(REGEX MATCH "[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>" tmpl "${inst}")
        if(tmpl)
            should_filter_template("${tmpl}" should_filter)
            if(NOT should_filter)
                normalize_template_signature("${tmpl}" normalized)
                list(APPEND provided_templates "${normalized}")
            endif()
        endif()
    endforeach()
    
    # EXTRACT USED TEMPLATES
    string(REGEX MATCHALL "extern[ \t]+template[ \t]+(class|struct)[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*[ \t]*<[^;>]+>[ \t]*;" 
           extern_templates "${preprocessed_content}")
    
    foreach(ext ${extern_templates})
        string(REGEX MATCH "[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>" tmpl "${ext}")
        if(tmpl)
            should_filter_template("${tmpl}" should_filter)
            if(NOT should_filter)
                normalize_template_signature("${tmpl}" normalized)
                list(APPEND used_templates "${normalized}")
            endif()
        endif()
    endforeach()
    
    # Project-specific patterns
    set(project_template_patterns
        "NDArray<[^>]+>"
        "DataBuffer<[^>]+>"
        "ResultSet<[^>]+>"
        "ShapeList<[^>]+>"
        "LaunchContext<[^>]+>"
        "BroadcastHelper<[^>]+>"
        "BroadcastInt<[^>]+>"
        "BroadcastBool<[^>]+>"
        "Broadcast<[^>]+>"
        "PairWiseTransform<[^>]+>"
        "ScalarTransform<[^>]+>"
        "RandomFunction<[^>]+>"
        "ReduceSameFunction<[^>]+>"
        "ReduceFloatFunction<[^>]+>"
        "ReduceBoolFunction<[^>]+>"
        "ReduceLongFunction<[^>]+>"
        "Reduce3<[^>]+>"
        "IndexReduce<[^>]+>"
        "ReductionLoops<[^>]+>"
        "IndexReductionLoops<[^>]+>"
        "LoopKind::Kind<[^>]+>"
        "SpecialMethods<[^>]+>"
        "TypeCast::convertGeneric<[^>]+>"
        "DeclarableOp<[^>]+>"
        "DeclarableCustomOp<[^>]+>"
        "DeclarableReductionOp<[^>]+>"
        "GEMM<[^>]+>"
        "GEMV<[^>]+>"
        "AXPY<[^>]+>"
    )
    
    foreach(pattern ${project_template_patterns})
        string(REGEX MATCHALL "${pattern}" matches "${preprocessed_content}")
        foreach(match ${matches})
            should_filter_template("${match}" should_filter)
            if(NOT should_filter)
                normalize_template_signature("${match}" normalized)
                list(APPEND used_templates "${normalized}")
            endif()
        endforeach()
    endforeach()
    
    # Remove duplicates and write
    if(used_templates)
        list(REMOVE_DUPLICATES used_templates)
        list(SORT used_templates)
        string(REPLACE ";" "\n" used_content "${used_templates}")
        file(WRITE ${USED_OUTPUT_FILE} "${used_content}")
    else()
        file(WRITE ${USED_OUTPUT_FILE} "")
    endif()
    
    if(provided_templates)
        list(REMOVE_DUPLICATES provided_templates)
        list(SORT provided_templates)
        string(REPLACE ";" "\n" provided_content "${provided_templates}")
        file(WRITE ${PROVIDED_OUTPUT_FILE} "${provided_content}")
    else()
        file(WRITE ${PROVIDED_OUTPUT_FILE} "")
    endif()
endfunction()

# Process batch function for parallel execution
function(process_batch BATCH_ID BATCH_FILE)
    if(NOT EXISTS ${BATCH_FILE})
        file(WRITE "${INST_BATCH_DIR}/batch_${BATCH_ID}.done" "")
        return()
    endif()
    
    file(STRINGS ${BATCH_FILE} FILES_TO_PROCESS)
    list(LENGTH FILES_TO_PROCESS batch_size)
    message("[Batch ${BATCH_ID}] Processing ${batch_size} files")
    
    set(batch_results "")
    set(processed_count 0)
    
    foreach(src IN LISTS FILES_TO_PROCESS)
        if(NOT EXISTS ${src})
            continue()
        endif()
        
        get_filename_component(src_name ${src} NAME_WE)
        get_filename_component(src_path ${src} PATH)
        get_filename_component(src_full_name ${src} NAME)
        file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${src_path})
        string(REPLACE "/" "_" safe_name "${rel_path}_${src_name}")
        
        set(IS_GENERATED FALSE)
        if(src MATCHES "${CMAKE_BINARY_DIR}")
            set(IS_GENERATED TRUE)
        endif()
        
        set(used_file "${INST_USED_DIR}/${safe_name}.used")
        set(provided_file "${INST_PROVIDED_DIR}/${safe_name}.provided")
        
        extract_templates_consolidated("${src}" "${used_file}" "${provided_file}" "${safe_name}")
        
        if(EXISTS ${used_file})
            file(STRINGS ${used_file} used_list)
        else()
            set(used_list "")
        endif()
        
        if(EXISTS ${provided_file})
            file(STRINGS ${provided_file} provided_list)
        else()
            set(provided_list "")
        endif()
        
        set(missing_templates "")
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
        
        list(LENGTH used_list used_count)
        list(LENGTH provided_list provided_count)
        
        string(APPEND batch_results "${src}|${safe_name}|${used_count}|${provided_count}|${missing_count}|${IS_GENERATED}\n")
        math(EXPR processed_count "${processed_count} + 1")
    endforeach()
    
    file(WRITE "${INST_BATCH_DIR}/batch_${BATCH_ID}.done" "${batch_results}")
    message("[Batch ${BATCH_ID}] Completed: ${processed_count} files")
endfunction()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Determine parallelism
if(NOT DEFINED INST_PARALLEL_JOBS)
    include(ProcessorCount)
    ProcessorCount(PROCESSOR_COUNT)
    if(PROCESSOR_COUNT GREATER 8)
        math(EXPR INST_PARALLEL_JOBS "${PROCESSOR_COUNT} - 2")
    elseif(PROCESSOR_COUNT GREATER 0)
        set(INST_PARALLEL_JOBS ${PROCESSOR_COUNT})
    else()
        set(INST_PARALLEL_JOBS 4)
    endif()
endif()
message(STATUS "Parallel jobs: ${INST_PARALLEL_JOBS}")

# Setup directories
set(INST_DIR "${CMAKE_SOURCE_DIR}/instantiation_analysis")
set(INST_USED_DIR "${INST_DIR}/used")
set(INST_PROVIDED_DIR "${INST_DIR}/provided")
set(INST_MISSING_DIR "${INST_DIR}/missing")
set(INST_REPORTS_DIR "${INST_DIR}/reports")
set(INST_TEMP_DIR "${INST_DIR}/temp")
set(INST_BY_FILE_DIR "${INST_DIR}/by_file")
set(INST_BATCH_DIR "${INST_DIR}/batches")

foreach(dir IN ITEMS ${INST_DIR} ${INST_USED_DIR} ${INST_PROVIDED_DIR} 
                     ${INST_MISSING_DIR} ${INST_REPORTS_DIR} ${INST_TEMP_DIR} 
                     ${INST_BY_FILE_DIR} ${INST_BATCH_DIR})
    if(NOT EXISTS ${dir})
        file(MAKE_DIRECTORY ${dir})
    endif()
endforeach()

# Setup flags
setup_instantiation_flags()

# Get sources
message(STATUS "Collecting source files...")
get_source_list(SOURCE_LIST)

if(NOT SOURCE_LIST)
    file(GLOB_RECURSE SOURCE_LIST 
        "${CMAKE_SOURCE_DIR}/include/**/*.cpp"
        "${CMAKE_SOURCE_DIR}/include/**/*.cu"
    )
endif()

# Add generated sources
if(SD_CUDA)
    file(GLOB_RECURSE GEN_SOURCES 
        "${CMAKE_BINARY_DIR}/cuda_instantiations/*.cu"
        "${CMAKE_BINARY_DIR}/compilation_units/*.cu")
else()
    file(GLOB_RECURSE GEN_SOURCES 
        "${CMAKE_BINARY_DIR}/cpu_instantiations/*.cpp"
        "${CMAKE_BINARY_DIR}/compilation_units/*.cpp")
endif()
list(APPEND SOURCE_LIST ${GEN_SOURCES})

# Filter and deduplicate
set(FILTERED_SOURCE_LIST "")
foreach(src IN LISTS SOURCE_LIST)
    if(src MATCHES "\\.(cpp|cxx|cc|cu)$" AND EXISTS ${src})
        list(APPEND FILTERED_SOURCE_LIST ${src})
    endif()
endforeach()
list(REMOVE_DUPLICATES FILTERED_SOURCE_LIST)
set(SOURCE_LIST ${FILTERED_SOURCE_LIST})

list(LENGTH SOURCE_LIST source_count)
message(STATUS "Total files to analyze: ${source_count}")

# Create batches
math(EXPR FILES_PER_BATCH "${source_count} / ${INST_PARALLEL_JOBS}")
if(FILES_PER_BATCH EQUAL 0)
    set(FILES_PER_BATCH 1)
endif()

set(current_batch 0)
set(files_in_batch 0)
foreach(src IN LISTS SOURCE_LIST)
    file(APPEND "${INST_BATCH_DIR}/batch_${current_batch}.txt" "${src}\n")
    math(EXPR files_in_batch "${files_in_batch} + 1")
    if(files_in_batch GREATER_EQUAL FILES_PER_BATCH AND current_batch LESS ${INST_PARALLEL_JOBS})
        math(EXPR current_batch "${current_batch} + 1")
        set(files_in_batch 0)
    endif()
endforeach()

# Process batches
message(STATUS "Processing ${INST_PARALLEL_JOBS} batches...")
foreach(batch_id RANGE 0 ${INST_PARALLEL_JOBS})
    set(batch_file "${INST_BATCH_DIR}/batch_${batch_id}.txt")
    if(EXISTS ${batch_file})
        process_batch(${batch_id} ${batch_file})
    endif()
endforeach()

# Aggregate results
message(STATUS "Aggregating results...")
set(TOTAL_USED 0)
set(TOTAL_PROVIDED 0)
set(TOTAL_MISSING 0)

file(WRITE ${INST_REPORTS_DIR}/summary.csv "File,Path,Used,Provided,Missing,Type\n")

foreach(batch_id RANGE 0 ${INST_PARALLEL_JOBS})
    set(done_file "${INST_BATCH_DIR}/batch_${batch_id}.done")
    if(EXISTS ${done_file})
        file(STRINGS ${done_file} batch_results)
        foreach(result_line ${batch_results})
            if(result_line)
                string(REPLACE "|" ";" result_parts "${result_line}")
                list(LENGTH result_parts parts_count)
                if(parts_count EQUAL 6)
                    list(GET result_parts 0 src)
                    list(GET result_parts 2 used_count)
                    list(GET result_parts 3 provided_count)
                    list(GET result_parts 4 missing_count)
                    list(GET result_parts 5 is_generated)
                    
                    math(EXPR TOTAL_USED "${TOTAL_USED} + ${used_count}")
                    math(EXPR TOTAL_PROVIDED "${TOTAL_PROVIDED} + ${provided_count}")
                    math(EXPR TOTAL_MISSING "${TOTAL_MISSING} + ${missing_count}")
                    
                    get_filename_component(src_name ${src} NAME)
                    get_filename_component(src_path ${src} PATH)
                    file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${src_path})
                    
                    set(file_type "original")
                    if(is_generated)
                        set(file_type "generated")
                    endif()
                    
                    file(APPEND ${INST_REPORTS_DIR}/summary.csv 
                        "${src_name},${rel_path},${used_count},${provided_count},${missing_count},${file_type}\n")
                endif()
            endif()
        endforeach()
    endif()
endforeach()

# Final summary
message(STATUS "")
message(STATUS "=====================================================")
message(STATUS "=== ANALYSIS COMPLETE ===")
message(STATUS "=====================================================")
message(STATUS "Templates Used: ${TOTAL_USED}")
message(STATUS "Templates Provided: ${TOTAL_PROVIDED}")
message(STATUS "Templates Missing: ${TOTAL_MISSING}")
message(STATUS "")
message(STATUS "Reports generated in: ${INST_REPORTS_DIR}")

# Exit without continuing build
message(STATUS "Exiting after instantiation extraction (SD_EXTRACT_INSTANTIATIONS=ON)")
return()
