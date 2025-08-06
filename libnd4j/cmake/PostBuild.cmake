# cmake/PostBuild.cmake
# Configures optional post-build targets like tests and analysis tools.

# --- Configuration file ---
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Flatbuffers Header and Java Generation ---
if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
    set(FLATC_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-build/flatc")
    set(MAIN_GENERATED_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/graph/generated.h")
    add_custom_command(
            OUTPUT ${MAIN_GENERATED_HEADER}
            COMMAND ${CMAKE_COMMAND} -E env "FLATC_PATH=${FLATC_EXECUTABLE}" bash ${CMAKE_CURRENT_SOURCE_DIR}/flatc-generate.sh
            DEPENDS flatbuffers_external
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running flatc to generate C++ headers"
            VERBATIM
    )
    add_custom_target(generate_flatbuffers_headers DEPENDS ${MAIN_GENERATED_HEADER})
    add_custom_command(
            TARGET generate_flatbuffers_headers POST_BUILD
            COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/copy-flatc-java.sh
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Copying generated Java files"
            VERBATIM
    )
endif()

# --- Self-Contained Preprocessing Target ---
if(SD_PREPROCESS)
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}")
    message("Setting up self-contained preprocessing with FlatBuffers...")

    include(ExternalProject)

    # Build FlatBuffers for preprocessing (minimal build, just need headers)
    set(PREPROCESS_FLATBUFFERS_VERSION "25.2.10")
    set(PREPROCESS_FLATBUFFERS_URL "https://github.com/google/flatbuffers/archive/v${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz")
    set(PREPROCESS_FB_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/preprocess-flatbuffers-src")
    set(PREPROCESS_FB_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/preprocess-flatbuffers-build")

    # Check if we already have FlatBuffers headers
    set(PREPROCESS_FB_HEADER "${PREPROCESS_FB_SOURCE_DIR}/include/flatbuffers/flatbuffers.h")
    if(NOT EXISTS ${PREPROCESS_FB_HEADER})
        message("Downloading and extracting FlatBuffers for preprocessing...")
        
        # Download and extract FlatBuffers
        file(DOWNLOAD ${PREPROCESS_FLATBUFFERS_URL} 
             "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz"
             SHOW_PROGRESS
             STATUS download_status)
        
        list(GET download_status 0 download_result)
        if(NOT download_result EQUAL 0)
            message(FATAL_ERROR "Failed to download FlatBuffers for preprocessing")
        endif()

        # Extract the archive
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            RESULT_VARIABLE extract_result
        )
        
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Failed to extract FlatBuffers for preprocessing")
        endif()

        # Move the extracted directory to our expected location
        file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}" 
             "${PREPROCESS_FB_SOURCE_DIR}")
    endif()

    # Verify FlatBuffers headers are now available
    if(NOT EXISTS ${PREPROCESS_FB_HEADER})
        message(FATAL_ERROR "FlatBuffers headers not found for preprocessing: ${PREPROCESS_FB_HEADER}")
    endif()

    message("✅ FlatBuffers headers available for preprocessing: ${PREPROCESS_FB_SOURCE_DIR}/include")

    # Get ALL subdirectories under include/ and add the root include directory first
    set(all_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")
    file(GLOB_RECURSE all_dirs LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
    foreach(item ${all_dirs})
        if(IS_DIRECTORY ${item})
            list(APPEND all_includes ${item})
        endif()
    endforeach()

    # Add binary include directory
    list(APPEND all_includes "${CMAKE_CURRENT_BINARY_DIR}/include")

    # Also add build directories
    list(APPEND all_includes
            "${CMAKE_SOURCE_DIR}/include/"
            "${CMAKE_SOURCE_DIR}/include/system"
            "${CMAKE_BINARY_DIR}/compilation_units"
            "${CMAKE_BINARY_DIR}/cpu_instantiations"
            "${CMAKE_BINARY_DIR}/cuda_instantiations"
            "${CMAKE_BINARY_DIR}/include"
    )

    # Add our self-contained FlatBuffers include directory
    list(APPEND all_includes "${PREPROCESS_FB_SOURCE_DIR}/include")

    # Build include flags
    set(include_flags "")
    foreach(dir IN LISTS all_includes)
        if(EXISTS ${dir})
            string(APPEND include_flags " -I${dir}")
        endif()
    endforeach()

    # Build definition flags - CRITICAL: Add the same definitions as main build
    set(defs_flags "")
    
    # Add basic platform definitions
    string(APPEND defs_flags " -D__CPUBLAS__=true")
    if(SD_CUDA)
        string(APPEND defs_flags " -D__CUDABLAS__=true -DHAVE_CUDA=1")
    endif()
    if(HAVE_OPENBLAS)
        string(APPEND defs_flags " -DHAVE_OPENBLAS=1")
    endif()

    # CRITICAL: Add operation definitions that define NOT_EXCLUDED macro
    if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
        string(APPEND defs_flags " -DSD_ALL_OPS=1")
        # When SD_ALL_OPS=1, NOT_EXCLUDED should evaluate to 1 for all ops
        string(APPEND defs_flags " -DNOT_EXCLUDED(x)=1")
    else()
        # Add specific operation definitions
        foreach(OP ${SD_OPS_LIST})
            string(APPEND defs_flags " -DOP_${OP}=1")
        endforeach()
        # Define NOT_EXCLUDED macro for selective ops
        string(APPEND defs_flags " -DNOT_EXCLUDED(x)=\\(defined\\(x\\)\\)")
    endif()

    # Add build type definitions
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        string(APPEND defs_flags " -DDEBUG=1 -D_DEBUG=1")
    else()
        string(APPEND defs_flags " -DNDEBUG=1")
    endif()

    # Add any additional compile definitions from the main build
    if(compile_defs AND NOT compile_defs STREQUAL "compile_defs-NOTFOUND")
        foreach(def IN LISTS compile_defs)
            string(APPEND defs_flags " -D${def}")
        endforeach()
    endif()

    set(PREPROCESSED_DIR "${CMAKE_SOURCE_DIR}/preprocessed")
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})

    set(PREPROCESSED_FILES)
    set(PROCESSED_SOURCES "")

    # Use ALL_SOURCES from global scope
    if(DEFINED ALL_SOURCES AND ALL_SOURCES)
        set(SOURCE_LIST ${ALL_SOURCES})
    elseif(DEFINED ALL_SOURCES_LIST AND ALL_SOURCES_LIST)
        set(SOURCE_LIST ${ALL_SOURCES_LIST})
    else()
        message(WARNING "No source list available for preprocessing. Skipping.")
        add_custom_target(preprocess_sources ALL
                COMMAND ${CMAKE_COMMAND} -E echo "Preprocessing skipped - no source files available")
        return()
    endif()

    list(LENGTH SOURCE_LIST source_count)
    message("Starting preprocessing of ${source_count} source files...")
    message("Operation definitions: SD_ALL_OPS=${SD_ALL_OPS}, SD_OPS_LIST=${SD_OPS_LIST}")

    foreach(src IN LISTS SOURCE_LIST)
        if(NOT EXISTS ${src} OR NOT src MATCHES "\\.(c|cpp|cxx|cc|cu)$")
            continue()
        endif()

        if(NOT src IN_LIST PROCESSED_SOURCES)
            get_filename_component(src_name ${src} NAME_WE)
            get_filename_component(src_path ${src} PATH)
            file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${src_path})
            string(REPLACE "/" "_" src_dir_ "${rel_path}")
            set(preprocessed_file "${PREPROCESSED_DIR}/${src_dir_}_${src_name}.i")

            if(NOT EXISTS "${preprocessed_file}")
                if(src MATCHES "\\.cu$")
                    set(compiler "${CMAKE_CUDA_COMPILER}")
                    set(lang_flags "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}}")
                elseif(src MATCHES "\\.c$")
                    set(compiler "${CMAKE_C_COMPILER}")
                    set(lang_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
                else()
                    set(compiler "${CMAKE_CXX_COMPILER}")
                    set(lang_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
                endif()

                # Split the flags properly for execute_process
                separate_arguments(lang_flags_list UNIX_COMMAND "${lang_flags}")
                separate_arguments(defs_flags_list UNIX_COMMAND "${defs_flags}")
                separate_arguments(include_flags_list UNIX_COMMAND "${include_flags}")

                execute_process(
                        COMMAND ${compiler}   -E -P -C ${lang_flags_list} ${defs_flags_list} ${include_flags_list} "${src}" -o "${preprocessed_file}"
                        RESULT_VARIABLE result
                        ERROR_VARIABLE error_output
                        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                if(result EQUAL 0)
                    list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                    message(STATUS "✅ Preprocessed: ${src}")
                else()
                    message(WARNING "❌ Failed to preprocess ${src}")
                    message(WARNING "   Error: ${error_output}")
                endif()
            else()
                list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                message(STATUS "✓ Already preprocessed: ${src}")
            endif()
            list(APPEND PROCESSED_SOURCES ${src})
        endif()
    endforeach()

    list(LENGTH PREPROCESSED_FILES processed_count)
    message("✅ Preprocessing complete. Generated ${processed_count} preprocessed files in ${PREPROCESSED_DIR}")
    add_custom_target(preprocess_sources ALL DEPENDS ${PREPROCESSED_FILES})
endif()

# --- Developer Analysis Targets ---
add_custom_target(analyze_types
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/analyze_type_usage.py
        --base-path ${CMAKE_CURRENT_SOURCE_DIR}
        --output ${CMAKE_BINARY_DIR}/type_analysis.json
        --summary
        COMMENT "Analyzing type usage patterns in codebase"
        VERBATIM
)

add_custom_target(show_combinations
        COMMAND ${CMAKE_COMMAND} -E echo "=== TYPE COMBINATION STATISTICS ==="
        COMMAND ${CMAKE_COMMAND} -E echo "Type profile: ${SD_TYPE_PROFILE}"
        COMMAND ${CMAKE_COMMAND} -E echo "Selected types: ${SD_TYPES_LIST}"
        COMMAND ${CMAKE_COMMAND} -E echo "Semantic filtering: ${SD_ENABLE_SEMANTIC_FILTERING}"
        COMMENT "Display type combination configuration"
)
