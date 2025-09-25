# cmake/InstantiationHelpers.cmake
# Helper functions for template instantiation extraction

# Function to setup compilation flags for instantiation extraction
function(setup_instantiation_flags)
    # FIRST: Setup FlatBuffers EXACTLY like PostBuild.cmake does
    set(PREPROCESS_FLATBUFFERS_VERSION "25.2.10")
    set(PREPROCESS_FLATBUFFERS_URL "https://github.com/google/flatbuffers/archive/v${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz")
    set(PREPROCESS_FB_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/preprocess-flatbuffers-src")
    
    # Check if we already have FlatBuffers headers
    set(PREPROCESS_FB_HEADER "${PREPROCESS_FB_SOURCE_DIR}/include/flatbuffers/flatbuffers.h")
    if(NOT EXISTS ${PREPROCESS_FB_HEADER})
        message("Downloading and extracting FlatBuffers for instantiation analysis...")
        
        # Download and extract FlatBuffers
        file(DOWNLOAD ${PREPROCESS_FLATBUFFERS_URL} 
             "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz"
             SHOW_PROGRESS
             STATUS download_status)
        
        list(GET download_status 0 download_result)
        if(NOT download_result EQUAL 0)
            message(FATAL_ERROR "Failed to download FlatBuffers for instantiation analysis")
        endif()

        # Extract the archive
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}.tar.gz"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            RESULT_VARIABLE extract_result
        )
        
        if(NOT extract_result EQUAL 0)
            message(FATAL_ERROR "Failed to extract FlatBuffers for instantiation analysis")
        endif()

        # Move the extracted directory to our expected location
        file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers-${PREPROCESS_FLATBUFFERS_VERSION}" 
             "${PREPROCESS_FB_SOURCE_DIR}")
    endif()
    
    # NOW COPY THE EXACT INCLUDE SETUP FROM PostBuild.cmake
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

    # Also add build directories - EXACTLY AS IN PostBuild.cmake
    list(APPEND all_includes
            "${CMAKE_SOURCE_DIR}/include/"
            "${CMAKE_SOURCE_DIR}/include/system"
            "${CMAKE_BINARY_DIR}/compilation_units"
            "${CMAKE_BINARY_DIR}/cpu_instantiations"
            "${CMAKE_BINARY_DIR}/cuda_instantiations"
            "${CMAKE_BINARY_DIR}/include"
    )

    # Add our self-contained FlatBuffers include directory - EXACTLY AS IN PostBuild.cmake
    list(APPEND all_includes "${PREPROCESS_FB_SOURCE_DIR}/include")
    
    # Build include flags string
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
    
    # Datatype definitions
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

# Function to get source list
function(get_source_list OUT_VAR)
    if(DEFINED ALL_SOURCES AND ALL_SOURCES)
        set(${OUT_VAR} ${ALL_SOURCES} PARENT_SCOPE)
    elseif(DEFINED ALL_SOURCES_LIST AND ALL_SOURCES_LIST)
        set(${OUT_VAR} ${ALL_SOURCES_LIST} PARENT_SCOPE)
    else()
        set(${OUT_VAR} "" PARENT_SCOPE)
    endif()
endfunction()

# Function to extract used templates from a source file - KEEP THE ORIGINAL WORKING VERSION
function(extract_used_templates SOURCE_FILE OUTPUT_FILE SAFE_NAME)
    # Initialize empty result
    set(used_templates "")
    
    # Check if source file exists
    if(NOT EXISTS "${SOURCE_FILE}")
        message(WARNING "Source file does not exist: ${SOURCE_FILE}")
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_USED_TEMPLATES "" PARENT_SCOPE)
        return()
    endif()
    
    # Check if temp directory exists
    if(NOT EXISTS "${INST_TEMP_DIR}")
        file(MAKE_DIRECTORY "${INST_TEMP_DIR}")
    endif()
    
    # Always preprocess the file to expand all macros
    set(preprocessed_file "${INST_TEMP_DIR}/${SAFE_NAME}_preprocessed.i")
    
    # Determine compiler and flags based on file type
    if(SOURCE_FILE MATCHES "\\.cu$")
        set(compiler "${CMAKE_CUDA_COMPILER}")
        set(lang_flags "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE}}")
    elseif(SOURCE_FILE MATCHES "\\.c$")
        set(compiler "${CMAKE_C_COMPILER}")
        set(lang_flags "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE}}")
    else()
        set(compiler "${CMAKE_CXX_COMPILER}")
        set(lang_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
    endif()
    
    # Check if compiler exists
    if(NOT EXISTS "${compiler}" AND NOT compiler)
        message(WARNING "Compiler not found for ${SOURCE_FILE}")
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_USED_TEMPLATES "" PARENT_SCOPE)
        return()
    endif()
    
    # Preprocess the file to expand all macros - USING EXACT WORKING METHOD FROM PostBuild.cmake
    # Split the flags properly for execute_process
    separate_arguments(lang_flags_list UNIX_COMMAND "${lang_flags}")
    separate_arguments(defs_flags_list UNIX_COMMAND "${INST_DEFS_FLAGS}")
    separate_arguments(include_flags_list UNIX_COMMAND "${INST_INCLUDE_FLAGS}")
    
    execute_process(
        COMMAND ${compiler} -E -P -C ${lang_flags_list} ${defs_flags_list} ${include_flags_list} "${SOURCE_FILE}" -o "${preprocessed_file}"
        RESULT_VARIABLE preprocess_result
        ERROR_VARIABLE preprocess_errors
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    
    if(NOT preprocess_result EQUAL 0)
        message(WARNING "Failed to preprocess ${SOURCE_FILE}: ${preprocess_errors}")
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_USED_TEMPLATES "" PARENT_SCOPE)
        return()
    endif()
    
    if(NOT EXISTS ${preprocessed_file})
        message(WARNING "Preprocessed file not created for ${SOURCE_FILE}")
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_USED_TEMPLATES "" PARENT_SCOPE)
        return()
    endif()
    
    # Method 1: Compile the preprocessed file with -fno-implicit-templates to find required templates
    execute_process(
        COMMAND ${compiler}
            -fsyntax-only
            -fno-implicit-templates
            -ftemplate-backtrace-limit=0
            ${lang_flags}
            ${INST_DEFS_FLAGS}
            ${INST_INCLUDE_FLAGS}
            -x c++                # Force C++ mode since .i files might not be recognized
            ${preprocessed_file}
        ERROR_VARIABLE compile_errors
        OUTPUT_QUIET
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 30
    )
    
    if(compile_errors)
        # Parse undefined references from the preprocessed compilation
        string(REGEX MATCHALL "undefined reference to '[^']+'" undefined_refs "${compile_errors}")
        foreach(ref ${undefined_refs})
            string(REGEX REPLACE "undefined reference to '([^']+)'" "\\1" symbol "${ref}")
            # Filter for template instantiations
            if(symbol MATCHES ".*<.*>.*")
                # Demangle if needed
                if(symbol MATCHES "^_Z")
                    if(CPPFILT_TOOL)
                        execute_process(
                            COMMAND ${CPPFILT_TOOL} "${symbol}"
                            OUTPUT_VARIABLE demangled
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_QUIET
                        )
                        if(demangled)
                            set(symbol "${demangled}")
                        endif()
                    endif()
                endif()
                list(APPEND used_templates "${symbol}")
            endif()
        endforeach()
        
        # Parse instantiation requirements
        string(REGEX MATCHALL "instantiation of '[^']+' required" inst_reqs "${compile_errors}")
        foreach(req ${inst_reqs})
            string(REGEX REPLACE "instantiation of '([^']+)' required" "\\1" tmpl "${req}")
            if(tmpl MATCHES ".*<.*>.*")
                list(APPEND used_templates "${tmpl}")
            endif()
        endforeach()
        
        # Parse "error: explicit instantiation of '[^']+' but no definition available"
        string(REGEX MATCHALL "explicit instantiation of '[^']+' but no definition" missing_defs "${compile_errors}")
        foreach(def ${missing_defs})
            string(REGEX REPLACE "explicit instantiation of '([^']+)' but no definition" "\\1" tmpl "${def}")
            if(tmpl MATCHES ".*<.*>.*")
                list(APPEND used_templates "${tmpl}")
            endif()
        endforeach()
    endif()
    
    # Method 2: Parse the preprocessed source for template usage patterns
    file(READ ${preprocessed_file} source_content LIMIT 10485760) # Read up to 10MB
    
    # Look for explicit template instantiation declarations (extern template)
    string(REGEX MATCHALL "extern[ \t]+template[ \t]+[^;]+<[^>]+>[^;]*;" extern_templates "${source_content}")
    foreach(ext_tmpl ${extern_templates})
        string(REGEX MATCH "[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>" tmpl "${ext_tmpl}")
        if(tmpl)
            list(APPEND used_templates "${tmpl}")
        endif()
    endforeach()
    
    # Look for template instantiation patterns in expanded macros
    # These patterns commonly appear after macro expansion
    set(expanded_patterns
        "template[ \t]+class[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>"
        "template[ \t]+struct[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>"
        "template[ \t]+void[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>::[a-zA-Z_][a-zA-Z0-9_]*"
        "template[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*[ \t]+[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>::[a-zA-Z_][a-zA-Z0-9_]*"
    )
    
    foreach(pattern ${expanded_patterns})
        string(REGEX MATCHALL "${pattern}" matches "${source_content}")
        foreach(match ${matches})
            # Extract the template class/function signature
            string(REGEX MATCH "[a-zA-Z_][a-zA-Z0-9_:]*<[^>]+>" tmpl "${match}")
            if(tmpl)
                list(APPEND used_templates "${tmpl}")
            endif()
        endforeach()
    endforeach()
    
    # Method 3: Look for template function calls and instantiations
    # Common template patterns that indicate usage
    set(usage_patterns
        "NDArray<[^>]+>"
        "DataBuffer<[^>]+>"
        "BroadcastHelper<[^>]+>"
        "BroadcastInt<[^>]+>"
        "BroadcastBool<[^>]+>"
        "Broadcast<[^>]+>"
        "LoopKind::Kind<[^>]+>"
        "LaunchContext<[^>]+>"
        "ResultSet<[^>]+>"
        "ShapeList<[^>]+>"
        "SpecialMethods<[^>]+>"
        "TypeCast::convertGeneric<[^>]+>"
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
        "std::vector<[^>]+>"
        "std::unique_ptr<[^>]+>"
        "std::shared_ptr<[^>]+>"
    )
    
    foreach(pattern ${usage_patterns})
        string(REGEX MATCHALL "${pattern}" matches "${source_content}")
        list(APPEND used_templates ${matches})
    endforeach()
    
    # Method 4: Try to compile to object and extract undefined symbols
    set(temp_obj "${INST_TEMP_DIR}/${SAFE_NAME}_test.o")
    execute_process(
        COMMAND ${compiler}
            -c
            -o ${temp_obj}
            ${lang_flags}
            ${INST_DEFS_FLAGS}
            ${INST_INCLUDE_FLAGS}
            -x c++
            ${preprocessed_file}
        ERROR_VARIABLE obj_errors
        OUTPUT_QUIET
        RESULT_VARIABLE obj_result
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 30
    )
    
    # Extract undefined symbols from linker errors
    if(obj_errors)
        string(REGEX MATCHALL "undefined reference to `[^']+'" undef_symbols "${obj_errors}")
        foreach(symbol ${undef_symbols})
            string(REGEX REPLACE "undefined reference to `([^']+)'" "\\1" sym "${symbol}")
            if(sym MATCHES ".*<.*>.*")
                # Try to demangle if it's mangled
                if(sym MATCHES "^_Z" AND CPPFILT_TOOL)
                    execute_process(
                        COMMAND ${CPPFILT_TOOL} "${sym}"
                        OUTPUT_VARIABLE demangled
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET
                    )
                    if(demangled)
                        set(sym "${demangled}")
                    endif()
                endif()
                list(APPEND used_templates "${sym}")
            endif()
        endforeach()
    endif()
    
    # If object was created, also check with nm for undefined symbols
    if(obj_result EQUAL 0 AND EXISTS ${temp_obj} AND NM_TOOL)
        execute_process(
            COMMAND ${NM_TOOL} -u ${temp_obj}
            OUTPUT_VARIABLE nm_undefined
            ERROR_QUIET
        )
        
        if(nm_undefined)
            string(REGEX MATCHALL "_Z[A-Za-z0-9_]+" mangled_symbols "${nm_undefined}")
            foreach(mangled ${mangled_symbols})
                if(CPPFILT_TOOL)
                    execute_process(
                        COMMAND ${CPPFILT_TOOL} ${mangled}
                        OUTPUT_VARIABLE demangled
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET
                    )
                    if(demangled AND demangled MATCHES ".*<.*>.*")
                        list(APPEND used_templates "${demangled}")
                    endif()
                endif()
            endforeach()
        endif()
        
        file(REMOVE ${temp_obj})
    endif()
    
    # Clean up preprocessed file
    file(REMOVE ${preprocessed_file})
    
    # Remove duplicates and filter out noise
    if(used_templates)
        list(REMOVE_DUPLICATES used_templates)
        
        # Filter out standard library internals and compiler intrinsics
        set(filtered_templates "")
        foreach(tmpl ${used_templates})
            if(NOT tmpl MATCHES "^__" AND 
               NOT tmpl MATCHES "^std::__" AND
               NOT tmpl MATCHES "^__gnu" AND
               NOT tmpl MATCHES "^__builtin" AND
               NOT tmpl MATCHES "^operator")
                list(APPEND filtered_templates "${tmpl}")
            endif()
        endforeach()
        
        string(REPLACE ";" "\n" used_content "${filtered_templates}")
        file(WRITE ${OUTPUT_FILE} "${used_content}")
        set(EXTRACTED_USED_TEMPLATES ${filtered_templates} PARENT_SCOPE)
    else()
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_USED_TEMPLATES "" PARENT_SCOPE)
    endif()
endfunction()

function(extract_provided_templates SOURCE_FILE OUTPUT_FILE SAFE_NAME)
    set(provided_templates "")
    
    # Check if source file exists
    if(NOT EXISTS "${SOURCE_FILE}")
        message(WARNING "Source file does not exist: ${SOURCE_FILE}")
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_PROVIDED_TEMPLATES "" PARENT_SCOPE)
        return()
    endif()
    
    # Check if temp directory exists
    if(NOT EXISTS "${INST_TEMP_DIR}")
        file(MAKE_DIRECTORY "${INST_TEMP_DIR}")
    endif()
    
    # Method 1: Check for explicit instantiations in source
    file(READ ${SOURCE_FILE} source_content LIMIT 1048576)
    
    string(REGEX MATCHALL "template[ \t]+class[ \t]+[^;]+<[^>]+>[^;]*;" 
           explicit_class "${source_content}")
    string(REGEX MATCHALL "template[ \t]+struct[ \t]+[^;]+<[^>]+>[^;]*;" 
           explicit_struct "${source_content}")
    
    foreach(inst ${explicit_class} ${explicit_struct})
        string(REGEX MATCH "[a-zA-Z_][a-zA-Z0-9_]*<[^>]+>" tmpl "${inst}")
        if(tmpl)
            list(APPEND provided_templates "${tmpl}")
        endif()
    endforeach()
    
    # Method 2: Compile to object and extract symbols
    set(temp_obj "${INST_TEMP_DIR}/${SAFE_NAME}.o")
    
    # Determine compiler based on file type
    if(SOURCE_FILE MATCHES "\\.cu$")
        set(compiler "${CMAKE_CUDA_COMPILER}")
    elseif(SOURCE_FILE MATCHES "\\.c$")
        set(compiler "${CMAKE_C_COMPILER}")
    else()
        set(compiler "${CMAKE_CXX_COMPILER}")
    endif()
    
    # Check if compiler exists
    if(NOT EXISTS "${compiler}" AND NOT compiler)
        message(WARNING "Compiler not found for ${SOURCE_FILE}")
        # Still write what we found from method 1
        if(provided_templates)
            list(REMOVE_DUPLICATES provided_templates)
            string(REPLACE ";" "\n" provided_content "${provided_templates}")
            file(WRITE ${OUTPUT_FILE} "${provided_content}")
            set(EXTRACTED_PROVIDED_TEMPLATES ${provided_templates} PARENT_SCOPE)
        else()
            file(WRITE ${OUTPUT_FILE} "")
            set(EXTRACTED_PROVIDED_TEMPLATES "" PARENT_SCOPE)
        endif()
        return()
    endif()
    
    execute_process(
        COMMAND ${compiler}
            -c
            -o ${temp_obj}
            ${INST_LANG_FLAGS}
            ${INST_DEFS_FLAGS}
            ${INST_INCLUDE_FLAGS}
            ${SOURCE_FILE}
        ERROR_QUIET
        OUTPUT_QUIET
        RESULT_VARIABLE compile_result
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    
    if(compile_result EQUAL 0 AND EXISTS ${temp_obj})
        if(NM_TOOL AND CPPFILT_TOOL)
            # Extract defined symbols
            execute_process(
                COMMAND ${NM_TOOL} --defined-only ${temp_obj}
                OUTPUT_VARIABLE nm_output
                ERROR_QUIET
            )
            
            if(nm_output)
                # Extract and demangle template symbols
                string(REGEX MATCHALL "_Z[A-Za-z0-9_]+" mangled_symbols "${nm_output}")
                
                foreach(mangled ${mangled_symbols})
                    execute_process(
                        COMMAND ${CPPFILT_TOOL} ${mangled}
                        OUTPUT_VARIABLE demangled
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET
                    )
                    
                    if(demangled MATCHES ".*<.*>.*")
                        # Filter out std:: and internal templates
                        if(NOT demangled MATCHES "^std::" AND 
                           NOT demangled MATCHES "^__" AND
                           NOT demangled MATCHES "^operator")
                            list(APPEND provided_templates "${demangled}")
                        endif()
                    endif()
                endforeach()
            endif()
        endif()
        
        file(REMOVE ${temp_obj})
    endif()
    
    # Method 3: Parse assembly output for template instantiations
    execute_process(
        COMMAND ${compiler}
            -S
            -o -
            -fverbose-asm
            ${INST_LANG_FLAGS}
            ${INST_DEFS_FLAGS}
            ${INST_INCLUDE_FLAGS}
            ${SOURCE_FILE}
        OUTPUT_VARIABLE asm_output
        ERROR_QUIET
        RESULT_VARIABLE asm_result
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        TIMEOUT 10
    )
    
    if(asm_result EQUAL 0 AND asm_output)
        # Extract template info from assembly comments
        string(LENGTH "${asm_output}" asm_length)
        if(asm_length LESS 10000000) # Only process if under 10MB
            string(REGEX MATCHALL "#.*<[^>]+>.*" asm_templates "${asm_output}")
            foreach(comment ${asm_templates})
                string(REGEX MATCH "[a-zA-Z_][a-zA-Z0-9_]*<[^>]+>" tmpl "${comment}")
                if(tmpl AND NOT tmpl MATCHES "^std::")
                    list(APPEND provided_templates "${tmpl}")
                endif()
            endforeach()
        endif()
    endif()
    
    # Remove duplicates and write to file
    if(provided_templates)
        list(REMOVE_DUPLICATES provided_templates)
        string(REPLACE ";" "\n" provided_content "${provided_templates}")
        file(WRITE ${OUTPUT_FILE} "${provided_content}")
        set(EXTRACTED_PROVIDED_TEMPLATES ${provided_templates} PARENT_SCOPE)
    else()
        file(WRITE ${OUTPUT_FILE} "")
        set(EXTRACTED_PROVIDED_TEMPLATES "" PARENT_SCOPE)
    endif()
endfunction()

# Function to normalize template names for comparison
function(normalize_template_name TEMPLATE OUT_VAR)
    set(normalized "${TEMPLATE}")
    
    # Remove extra spaces
    string(REGEX REPLACE "[ \t]+" " " normalized "${normalized}")
    string(STRIP "${normalized}" normalized)
    
    # Normalize const placement
    string(REPLACE "const " "" normalized "${normalized}")
    string(REPLACE " const" "" normalized "${normalized}")
    
    # Remove allocator specifications from std containers
    string(REGEX REPLACE "std::allocator<[^>]+>" "" normalized "${normalized}")
    
    # Normalize pointer/reference spacing
    string(REPLACE " *" "*" normalized "${normalized}")
    string(REPLACE " &" "&" normalized "${normalized}")
    
    set(${OUT_VAR} "${normalized}" PARENT_SCOPE)
endfunction()
