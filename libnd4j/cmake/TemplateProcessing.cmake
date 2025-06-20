################################################################################
# Template Processing Functions
# Functions for processing CMake template files and generating compilation units
################################################################################

# Verifies that a generated file has been processed correctly by configure_file
function(verify_template_processing GENERATED_FILE)
    if(EXISTS "${GENERATED_FILE}")
        file(READ "${GENERATED_FILE}" FILE_CONTENT)

        if(FILE_CONTENT MATCHES "#cmakedefine[ \t]+[A-Za-z_]+")
            message(FATAL_ERROR "❌ Template processing FAILED: ${GENERATED_FILE} contains unprocessed #cmakedefine directives")
        endif()

        if(FILE_CONTENT MATCHES "@[A-Za-z_]+@")
            message(FATAL_ERROR "❌ Template processing FAILED: ${GENERATED_FILE} contains unprocessed @VAR@ tokens")
        endif()
    else()
        message(FATAL_ERROR "❌ Generated file does not exist for verification: ${GENERATED_FILE}")
    endif()
endfunction()

# Sets up the variables needed by a template file before it is processed
function(setup_unified_template_variables TEMPLATE_FILE COMB1 COMB2 COMB3)
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    set(FL_TYPE_INDEX ${COMB1} PARENT_SCOPE)
    set(TYPE_INDEX_1 ${COMB1} PARENT_SCOPE)
    set(TYPE_INDEX_2 ${COMB2} PARENT_SCOPE)
    set(TYPE_INDEX_3 ${COMB3} PARENT_SCOPE)

    set(SD_COMMON_TYPES_GEN 0 PARENT_SCOPE)
    set(SD_FLOAT_TYPES_GEN 0 PARENT_SCOPE)
    set(SD_INTEGER_TYPES_GEN 0 PARENT_SCOPE)
    set(SD_PAIRWISE_TYPES_GEN 0 PARENT_SCOPE)
    set(SD_SEMANTIC_TYPES_GEN 0 PARENT_SCOPE)

    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_COMMON_TYPES_GEN")
        set(SD_COMMON_TYPES_GEN 1 PARENT_SCOPE)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_FLOAT_TYPES_GEN")
        set(SD_FLOAT_TYPES_GEN 1 PARENT_SCOPE)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_INTEGER_TYPES_GEN")
        set(SD_INTEGER_TYPES_GEN 1 PARENT_SCOPE)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_PAIRWISE_TYPES_GEN")
        set(SD_PAIRWISE_TYPES_GEN 1 PARENT_SCOPE)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_SEMANTIC_TYPES_GEN")
        set(SD_SEMANTIC_TYPES_GEN 1 PARENT_SCOPE)
    endif()

    if(DEFINED SD_COMMON_TYPES_COUNT)
        math(EXPR max_index "${SD_COMMON_TYPES_COUNT} - 1")
        if(COMB1 GREATER max_index OR COMB2 GREATER max_index OR COMB3 GREATER max_index)
            set(SKIP_TEMPLATE TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    set(SKIP_TEMPLATE FALSE PARENT_SCOPE)
endfunction()

# Processes a single template file against a list of combinations
function(process_template_unified TEMPLATE_FILE COMBINATION_TYPE COMBINATIONS OUTPUT_DIR)
    if(NOT EXISTS "${TEMPLATE_FILE}")
        return()
    endif()

    set(GENERATED_SOURCES_LOCAL "")

    foreach(COMBINATION ${COMBINATIONS})
        string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
        list(LENGTH COMB_LIST COMB_COUNT)

        if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR
        (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
            continue()
        endif()

        if(COMBINATION_TYPE EQUAL 3)
            list(GET COMB_LIST 0 COMB1)
            list(GET COMB_LIST 1 COMB2)
            list(GET COMB_LIST 2 COMB3)
        elseif(COMBINATION_TYPE EQUAL 2)
            list(GET COMB_LIST 0 COMB1)
            list(GET COMB_LIST 1 COMB2)
            set(COMB3 ${COMB1})
        endif()

        setup_unified_template_variables("${TEMPLATE_FILE}" ${COMB1} ${COMB2} ${COMB3})

        if(SKIP_TEMPLATE)
            continue()
        endif()

        get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
        string(REPLACE "_template" "" OUTPUT_BASE_NAME "${TEMPLATE_BASE}")

        if(COMBINATION_TYPE EQUAL 3)
            set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}_${COMB3}.cpp")
        elseif(COMBINATION_TYPE EQUAL 2)
            set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}.cpp")
        endif()

        set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")
        file(MAKE_DIRECTORY "${OUTPUT_DIR}")

        configure_file("${TEMPLATE_FILE}" "${GENERATED_FILE}" @ONLY)
        verify_template_processing("${GENERATED_FILE}")

        list(APPEND GENERATED_SOURCES_LOCAL "${GENERATED_FILE}")
    endforeach()

    set(GENERATED_SOURCES ${GENERATED_SOURCES_LOCAL} PARENT_SCOPE)
endfunction()

# Detects whether a template requires 2 or 3 type combinations
function(detect_template_requirements TEMPLATE_FILE NEEDS_2_TYPE_VAR NEEDS_3_TYPE_VAR)
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    set(NEEDS_2_TYPE FALSE)
    set(NEEDS_3_TYPE FALSE)

    if(TEMPLATE_CONTENT MATCHES "TYPE_INDEX_3|COMB3|_template_3")
        set(NEEDS_3_TYPE TRUE)
    endif()

    if(TEMPLATE_CONTENT MATCHES "TYPE_INDEX_2|COMB2|_template_2")
        set(NEEDS_2_TYPE TRUE)
    endif()

    if(NOT NEEDS_2_TYPE AND NOT NEEDS_3_TYPE)
        get_filename_component(TEMPLATE_NAME "${TEMPLATE_FILE}" NAME)
        if(TEMPLATE_NAME MATCHES "_3\\.")
            set(NEEDS_3_TYPE TRUE)
        elseif(TEMPLATE_NAME MATCHES "_2\\.")
            set(NEEDS_2_TYPE TRUE)
        else()
            set(NEEDS_2_TYPE TRUE)
            set(NEEDS_3_TYPE TRUE)
        endif()
    endif()

    set(${NEEDS_2_TYPE_VAR} ${NEEDS_2_TYPE} PARENT_SCOPE)
    set(${NEEDS_3_TYPE_VAR} ${NEEDS_3_TYPE} PARENT_SCOPE)
endfunction()

# Main entry point to find and process all CPU templates
function(process_cpu_templates)
    # CRITICAL FIX: Verify combinations are available
    if(NOT DEFINED COMBINATIONS_2 OR NOT DEFINED COMBINATIONS_3)
        message(FATAL_ERROR "❌ Type combinations not initialized! Call initialize_dynamic_combinations() first.")
    endif()

    # Debug output
    list(LENGTH COMBINATIONS_2 combo2_count)
    list(LENGTH COMBINATIONS_3 combo3_count)
    message(STATUS "Processing CPU templates with ${combo2_count} 2-type and ${combo3_count} 3-type combinations")

    # Check if we have any combinations to work with
    if(combo2_count EQUAL 0 AND combo3_count EQUAL 0)
        message(FATAL_ERROR "❌ No type combinations available for template processing!")
    endif()

    file(GLOB_RECURSE ALL_TEMPLATE_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/compilation_units/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/loops/*.cpp.in"
    )

    list(LENGTH ALL_TEMPLATE_FILES template_count)
    message(STATUS "Found ${template_count} template files to process")

    set(INSTANTIATION_OUTPUT_DIR "${CMAKE_BINARY_DIR}/compilation_units")
    set(CPU_INSTANTIATION_DIR "${INSTANTIATION_OUTPUT_DIR}/cpu")
    file(MAKE_DIRECTORY "${INSTANTIATION_OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${CPU_INSTANTIATION_DIR}")

    set(ALL_GENERATED_SOURCES "")

    foreach(TEMPLATE_FILE ${ALL_TEMPLATE_FILES})
        detect_template_requirements("${TEMPLATE_FILE}" NEEDS_2_TYPE NEEDS_3_TYPE)

        get_filename_component(TEMPLATE_NAME "${TEMPLATE_FILE}" NAME)
        message(STATUS "Processing template: ${TEMPLATE_NAME} (needs_2=${NEEDS_2_TYPE}, needs_3=${NEEDS_3_TYPE})")

        set(TEMPLATE_GENERATED_SOURCES "")

        if(NEEDS_3_TYPE AND COMBINATIONS_3)
            message(STATUS "  -> Processing with 3-type combinations (${combo3_count} combinations)")
            process_template_unified("${TEMPLATE_FILE}" 3 "${COMBINATIONS_3}" "${CPU_INSTANTIATION_DIR}")
            list(APPEND TEMPLATE_GENERATED_SOURCES ${GENERATED_SOURCES})
            list(LENGTH GENERATED_SOURCES gen_count)
            message(STATUS "  -> Generated ${gen_count} 3-type instantiations")
        endif()

        if(NEEDS_2_TYPE AND COMBINATIONS_2)
            message(STATUS "  -> Processing with 2-type combinations (${combo2_count} combinations)")
            process_template_unified("${TEMPLATE_FILE}" 2 "${COMBINATIONS_2}" "${CPU_INSTANTIATION_DIR}")
            list(APPEND TEMPLATE_GENERATED_SOURCES ${GENERATED_SOURCES})
            list(LENGTH GENERATED_SOURCES gen_count)
            message(STATUS "  -> Generated ${gen_count} 2-type instantiations")
        endif()

        list(APPEND ALL_GENERATED_SOURCES ${TEMPLATE_GENERATED_SOURCES})
    endforeach()

    list(LENGTH ALL_GENERATED_SOURCES total_generated)
    message(STATUS "✅ Template processing complete. Generated ${total_generated} source files.")

    set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} PARENT_SCOPE)
endfunction()

# Enhanced template processing function with proper verification
function(genCompilation FL_ITEM)
    get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)

    set(EXTENSION "cpp")
    if(FL_ITEM MATCHES "cu.in$")
        set(EXTENSION "cu")
    endif()

    file(READ ${FL_ITEM} CONTENT_FL)

    set(SD_FLOAT_TYPES_GEN 0)
    set(SD_INTEGER_TYPES_GEN 0)
    set(SD_COMMON_TYPES_GEN 0)
    set(SD_PAIRWISE_TYPES_GEN 0)
    set(RANGE_STOP -1)

    string(REGEX MATCHALL "#cmakedefine[ \t]+SD_(INTEGER|COMMON|FLOAT|PAIRWISE)_TYPES_GEN" TYPE_MATCHES ${CONTENT_FL})

    set(SD_INTEGER_TYPES_END 7)
    set(SD_COMMON_TYPES_END 12)
    set(SD_FLOAT_TYPES_END 3)
    set(SD_PAIRWISE_TYPES_END 12)

    foreach(TYPEX ${TYPE_MATCHES})
        set(STOP -1)
        if(TYPEX MATCHES "SD_INTEGER_TYPES_GEN$")
            set(SD_INTEGER_TYPES_GEN 1)
            set(STOP ${SD_INTEGER_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_COMMON_TYPES_GEN$")
            set(SD_COMMON_TYPES_GEN 1)
            set(STOP ${SD_COMMON_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_FLOAT_TYPES_GEN$")
            set(SD_FLOAT_TYPES_GEN 1)
            set(STOP ${SD_FLOAT_TYPES_END})
        endif()
        if(TYPEX MATCHES "SD_PAIRWISE_TYPES_GEN$")
            set(SD_PAIRWISE_TYPES_GEN 1)
            set(STOP ${SD_PAIRWISE_TYPES_END})
        endif()
        if(STOP GREATER RANGE_STOP)
            set(RANGE_STOP ${STOP})
        endif()
    endforeach()

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/compilation_units")

    if(RANGE_STOP GREATER -1)
        set(CHUNK_SIZE 3)
        math(EXPR TOTAL_RANGE "${RANGE_STOP} + 1")
        math(EXPR NUM_CHUNKS "(${TOTAL_RANGE} + ${CHUNK_SIZE} - 1) / ${CHUNK_SIZE}")
        math(EXPR NUM_CHUNKS_MINUS_1 "${NUM_CHUNKS} - 1")

        foreach(CHUNK_INDEX RANGE 0 ${NUM_CHUNKS_MINUS_1})
            math(EXPR START_INDEX "${CHUNK_INDEX} * ${CHUNK_SIZE}")
            math(EXPR TEMP_END "${START_INDEX} + ${CHUNK_SIZE} - 1")
            if(TEMP_END GREATER RANGE_STOP)
                set(END_INDEX ${RANGE_STOP})
            else()
                set(END_INDEX ${TEMP_END})
            endif()

            if(START_INDEX LESS_EQUAL RANGE_STOP)
                foreach(FL_TYPE_INDEX RANGE ${START_INDEX} ${END_INDEX})
                    # Reset flags based on current type index
                    set(CURRENT_SD_FLOAT_TYPES_GEN ${SD_FLOAT_TYPES_GEN})
                    set(CURRENT_SD_INTEGER_TYPES_GEN ${SD_INTEGER_TYPES_GEN})
                    set(CURRENT_SD_COMMON_TYPES_GEN ${SD_COMMON_TYPES_GEN})
                    set(CURRENT_SD_PAIRWISE_TYPES_GEN ${SD_PAIRWISE_TYPES_GEN})

                    if(FL_TYPE_INDEX GREATER ${SD_FLOAT_TYPES_END})
                        set(CURRENT_SD_FLOAT_TYPES_GEN 0)
                    endif()
                    if(FL_TYPE_INDEX GREATER ${SD_INTEGER_TYPES_END})
                        set(CURRENT_SD_INTEGER_TYPES_GEN 0)
                    endif()
                    if(FL_TYPE_INDEX GREATER ${SD_COMMON_TYPES_END})
                        set(CURRENT_SD_COMMON_TYPES_GEN 0)
                    endif()
                    if(FL_TYPE_INDEX GREATER ${SD_PAIRWISE_TYPES_END})
                        set(CURRENT_SD_PAIRWISE_TYPES_GEN 0)
                    endif()

                    set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_chunk${CHUNK_INDEX}_${FL_TYPE_INDEX}.${EXTENSION}")

                    # CRITICAL FIX: Use configure_file instead of manual string replacement
                    # Set variables in current scope for configure_file
                    set(SD_FLOAT_TYPES_GEN ${CURRENT_SD_FLOAT_TYPES_GEN})
                    set(SD_INTEGER_TYPES_GEN ${CURRENT_SD_INTEGER_TYPES_GEN})
                    set(SD_COMMON_TYPES_GEN ${CURRENT_SD_COMMON_TYPES_GEN})
                    set(SD_PAIRWISE_TYPES_GEN ${CURRENT_SD_PAIRWISE_TYPES_GEN})

                    # Use configure_file to properly process template
                    configure_file(
                            "${FL_ITEM}"
                            "${GENERATED_SOURCE}"
                            @ONLY
                    )

                    # Verify processing worked
                    file(READ "${GENERATED_SOURCE}" VERIFICATION_CONTENT)
                    if(VERIFICATION_CONTENT MATCHES "#cmakedefine")
                        message(FATAL_ERROR "❌ genCompilation: Template processing failed! ${GENERATED_SOURCE} still contains #cmakedefine")
                    endif()
                    if(VERIFICATION_CONTENT MATCHES "@[A-Za-z_]+@")
                        message(FATAL_ERROR "❌ genCompilation: Template processing failed! ${GENERATED_SOURCE} still contains @VAR@ tokens")
                    endif()

                    list(APPEND CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE})
                endforeach()
            endif()
        endforeach()
    endif()

    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

# Enhanced partition combination function with proper template processing
function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
    list(LENGTH COMB_LIST COMB_COUNT)

    if(NOT (COMBINATION_TYPE EQUAL 3 OR COMBINATION_TYPE EQUAL 2))
        message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 3 or 2.")
    endif()

    if(NOT ((COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3) OR
    (COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2)))
        message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).")
    endif()

    if(COMBINATION_TYPE EQUAL 3)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        list(GET COMB_LIST 2 COMB3)
    elseif(COMBINATION_TYPE EQUAL 2)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
    endif()

    # CRITICAL FIX: Read template and set CMake variables for configure_file
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Detect what CMake defines the template needs
    set(SD_COMMON_TYPES_GEN 0)
    set(SD_FLOAT_TYPES_GEN 0)
    set(SD_INTEGER_TYPES_GEN 0)
    set(SD_PAIRWISE_TYPES_GEN 0)

    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_COMMON_TYPES_GEN")
        set(SD_COMMON_TYPES_GEN 1)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_FLOAT_TYPES_GEN")
        set(SD_FLOAT_TYPES_GEN 1)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_INTEGER_TYPES_GEN")
        set(SD_INTEGER_TYPES_GEN 1)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_PAIRWISE_TYPES_GEN")
        set(SD_PAIRWISE_TYPES_GEN 1)
    endif()

    # Set FL_TYPE_INDEX based on first combination value
    set(FL_TYPE_INDEX ${COMB1})

    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
    string(REPLACE "_template_" "_" OUTPUT_BASE_NAME "${TEMPLATE_BASE}")

    if(COMBINATION_TYPE EQUAL 3)
        set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}_${COMB3}.cpp")
    elseif(COMBINATION_TYPE EQUAL 2)
        set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}.cpp")
    endif()

    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")
    file(MAKE_DIRECTORY "${OUTPUT_DIR}")

    # FIXED: Use configure_file instead of manual string replacement
    configure_file(
        "${TEMPLATE_FILE}"
        "${GENERATED_FILE}"
        @ONLY
    )

    # Verify the processing worked
    file(READ "${GENERATED_FILE}" VERIFICATION_CONTENT)
    if(VERIFICATION_CONTENT MATCHES "#cmakedefine")
        message(FATAL_ERROR "❌ Template processing failed! ${GENERATED_FILE} still contains #cmakedefine directives")
    endif()
    if(VERIFICATION_CONTENT MATCHES "@[A-Za-z_]+@")
        message(FATAL_ERROR "❌ Template processing failed! ${GENERATED_FILE} still contains @VAR@ tokens")
    endif()

    list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)

    message(STATUS "✅ Generated and verified: ${GENERATED_FILE}")
endfunction()

# Function for single CUDA function generation
function(genSingleFunctionCuda TEMPLATE_FILE COMBINATION OUTPUT_DIR)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")

    list(GET COMB_LIST 0 COMB1)
    list(GET COMB_LIST 1 COMB2)
    list(GET COMB_LIST 2 COMB3)

    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)

    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    string(REGEX MATCH "([a-zA-Z0-9_:]+),[ \n\t]*::([a-zA-Z0-9_]+)" FUNCTION_MATCH "${TEMPLATE_CONTENT}")
    set(CLASS_NAME ${CMAKE_MATCH_1})
    set(METHOD_NAME ${CMAKE_MATCH_2})

    string(REGEX REPLACE "::" "_" CLASS_NAME_CLEAN "${CLASS_NAME}")

    string(REGEX MATCH "::${METHOD_NAME}\\(([^;]+)\\);" FUNC_ARGS_MATCH "${TEMPLATE_CONTENT}")
    set(FUNCTION_ARGS "${CMAKE_MATCH_1}")

    set(PARAM_COUNT 0)
    set(SIGNATURE_ID "")

    string(REPLACE "," ";" ARGS_LIST "${FUNCTION_ARGS}")
    list(LENGTH ARGS_LIST PARAM_COUNT)

    foreach(ARG ${ARGS_LIST})
        string(REGEX MATCH "^[^*& \t]+" TYPE_NAME "${ARG}")
        if(TYPE_NAME)
            string(APPEND SIGNATURE_ID "_${TYPE_NAME}")
        endif()
    endforeach()

    if(SIGNATURE_ID MATCHES ".{30,}")
        string(MD5 SIGNATURE_HASH "${SIGNATURE_ID}")
        string(SUBSTRING "${SIGNATURE_HASH}" 0 8 SIGNATURE_ID)
        set(SIGNATURE_ID "_h${SIGNATURE_ID}")
    endif()

    set(OUTPUT_FILE "${CLASS_NAME_CLEAN}_${METHOD_NAME}${SIGNATURE_ID}_${COMB1}_${COMB2}_${COMB3}.cu")
    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")

    if(EXISTS "${GENERATED_FILE}")
        list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
        set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)
        return()
    endif()

    set(START_MARKER "ITERATE_COMBINATIONS_3")
    string(FIND "${TEMPLATE_CONTENT}" "${START_MARKER}" START_POS)
    if(START_POS EQUAL -1)
        message(FATAL_ERROR "Could not find ITERATE_COMBINATIONS_3 in template file ${TEMPLATE_FILE}")
    endif()

    string(SUBSTRING "${TEMPLATE_CONTENT}" 0 ${START_POS} HEADER_CONTENT)

    set(NEW_CONTENT "${HEADER_CONTENT}\n\n// Single function instantiation for ${CLASS_NAME}::${METHOD_NAME}\n")
    string(APPEND NEW_CONTENT "template void ${CLASS_NAME}::${METHOD_NAME}<SD_SINGLE_TYPE_${COMB1}, SD_SINGLE_TYPE_${COMB2}, SD_SINGLE_TYPE_${COMB3}>(${FUNCTION_ARGS});\n")

    file(MAKE_DIRECTORY "${OUTPUT_DIR}")
    file(WRITE "${GENERATED_FILE}" "${NEW_CONTENT}")

    list(APPEND CUDA_GENERATED_SOURCES "${GENERATED_FILE}")
    set(CUDA_GENERATED_SOURCES ${CUDA_GENERATED_SOURCES} PARENT_SCOPE)

    message(STATUS "Generated: ${GENERATED_FILE}")
endfunction()

# Main template processing wrapper function
function(process_template_files)
    message(STATUS "Processing template files...")
    
    # Find all template files
    file(GLOB_RECURSE ALL_TEMPLATE_FILES
         "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.cpp.in"
         "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.cu.in"
    )
    
    # Process each template file
    foreach(template_file ${ALL_TEMPLATE_FILES})
        message(STATUS "Processing template: ${template_file}")
        genCompilation("${template_file}")
    endforeach()
    
    message(STATUS "Template processing completed")
endfunction()
