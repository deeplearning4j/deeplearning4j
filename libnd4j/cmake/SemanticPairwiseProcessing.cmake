################################################################################
# Semantic Pairwise Processing Functions
# Functions for enhanced semantic processing of pairwise template instantiations
################################################################################

function(process_pairwise_templates_semantic)
    # This feature is typically enabled by setting a CMake option like -DUSE_SEMANTIC_PAIRWISE_GENERATION=ON
    if(NOT USE_SEMANTIC_PAIRWISE_GENERATION)
        return()
    endif()

    message(STATUS "Processing pairwise templates with semantic type filtering...")

    file(GLOB PAIRWISE_TEMPLATE_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_*.cpp.in")

    set(semantic_output_dir "${CMAKE_BINARY_DIR}/semantic_pairwise_instantiations")
    file(MAKE_DIRECTORY "${semantic_output_dir}")

    set(ALL_GENERATED_SOURCES "")

    # For now, this function uses the standard combinations. A full implementation
    # would use a separate, semantically filtered list of combinations.
    foreach(TEMPLATE_FILE ${PAIRWISE_TEMPLATE_FILES})
        foreach(COMBINATION ${COMBINATIONS_2})
            string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
            list(GET COMB_LIST 0 COMB1)
            list(GET COMB_LIST 1 COMB2)

            get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
            string(REPLACE "_template" "" OUTPUT_BASE_NAME "${TEMPLATE_BASE}")
            set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}.cpp")
            set(GENERATED_FILE "${semantic_output_dir}/${OUTPUT_FILE}")

            add_custom_command(
                    OUTPUT ${GENERATED_FILE}
                    COMMAND ${CMAKE_COMMAND}
                    -DTYPE_INDEX_1=${COMB1} -DTYPE_INDEX_2=${COMB2}
                    -DTEMPLATE_INPUT=${TEMPLATE_FILE} -DTEMPLATE_OUTPUT=${GENERATED_FILE}
                    -P ${CONFIGURE_SCRIPT_PATH}
                    DEPENDS ${TEMPLATE_FILE} ${CONFIGURE_SCRIPT_PATH}
                    COMMENT "Generating semantic pairwise ${OUTPUT_FILE}"
                    VERBATIM
            )
            list(APPEND ALL_GENERATED_SOURCES ${GENERATED_FILE})
        endforeach()
    endforeach()

    # Append these generated sources to the main list to be compiled.
    list(APPEND CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES})
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)
endfunction()

# Enhanced genPartitionCombination function for pairwise semantic processing
function(genPartitionCombination_Semantic TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR)
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

    # Read template and detect semantic processing
    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    # Set standard CMake defines
    set(SD_COMMON_TYPES_GEN 0)
    set(SD_FLOAT_TYPES_GEN 0)
    set(SD_INTEGER_TYPES_GEN 0)
    set(SD_PAIRWISE_TYPES_GEN 0)
    set(SD_SEMANTIC_TYPES_GEN 0)

    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_COMMON_TYPES_GEN")
        set(SD_COMMON_TYPES_GEN 1)
    endif()
    if(TEMPLATE_CONTENT MATCHES "#cmakedefine[ \t]+SD_PAIRWISE_TYPES_GEN")
        set(SD_PAIRWISE_TYPES_GEN 1)
    endif()

    # Enable semantic processing for pairwise templates if semantic filtering is enabled
    if(SD_ENABLE_SEMANTIC_FILTERING AND TEMPLATE_FILE MATCHES "pairwise")
        set(SD_SEMANTIC_TYPES_GEN 1)
        message(STATUS "Enabling semantic processing for pairwise template: ${TEMPLATE_FILE}")

        # Generate semantic type parts if we have a workload profile
        if(SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
            generate_pairwise_type_parts("${SD_TYPE_PROFILE}" semantic_type_parts)

            # Write semantic type parts to a separate header
            set(SEMANTIC_HEADER_FILE "${OUTPUT_DIR}/pairwise_semantic_types_${COMB1}_${COMB2}.h")
            file(WRITE "${SEMANTIC_HEADER_FILE}"
                "// Auto-generated semantic type parts for pairwise operations\n"
                "// Workload profile: ${SD_TYPE_PROFILE}\n\n"
                "${semantic_type_parts}\n")

            # Include the semantic header in the template processing
            set(TEMPLATE_CONTENT "#include \"pairwise_semantic_types_${COMB1}_${COMB2}.h\"\n${TEMPLATE_CONTENT}")
        endif()
    endif()

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

    # Use configure_file for template processing
    configure_file(
        "${TEMPLATE_FILE}"
        "${GENERATED_FILE}"
        @ONLY
    )

    # Verify the processing worked
    file(READ "${GENERATED_FILE}" VERIFICATION_CONTENT)
    if(VERIFICATION_CONTENT MATCHES "#cmakedefine")
        message(FATAL_ERROR "❌ Semantic pairwise template processing failed! ${GENERATED_FILE} still contains #cmakedefine directives")
    endif()
    if(VERIFICATION_CONTENT MATCHES "@[A-Za-z_]+@")
        message(FATAL_ERROR "❌ Semantic pairwise template processing failed! ${GENERATED_FILE} still contains @VAR@ tokens")
    endif()

    list(APPEND CUSTOMOPS_GENERIC_SOURCES "${GENERATED_FILE}")
    set(CUSTOMOPS_GENERIC_SOURCES ${CUSTOMOPS_GENERIC_SOURCES} PARENT_SCOPE)

    message(STATUS "✅ Generated semantic pairwise file: ${GENERATED_FILE}")
endfunction()

# Enhanced template processing specifically for pairwise semantic types
function(process_pairwise_templates_semantic)
    if(NOT USE_SEMANTIC_PAIRWISE_GENERATION)
        message(STATUS "Semantic pairwise generation disabled - using traditional approach")
        return()
    endif()

    message(STATUS "Processing pairwise templates with semantic type filtering...")

    # Find pairwise template files
    file(GLOB PAIRWISE_TEMPLATE_3_FILES
         "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in")
    file(GLOB PAIRWISE_TEMPLATE_2_FILES
         "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in")

    set(semantic_output_dir "${CMAKE_BINARY_DIR}/semantic_pairwise_instantiations")
    file(MAKE_DIRECTORY "${semantic_output_dir}")

    # Process 3-type pairwise templates with semantic combinations
    if(DEFINED PAIRWISE_SEMANTIC_COMBINATIONS)
        # Partition semantic combinations into manageable chunks
        list(LENGTH PAIRWISE_SEMANTIC_COMBINATIONS total_semantic)
        math(EXPR chunk_size "(${total_semantic} + 8) / 9")  # Create 9 chunks (3x3 grid)

        partition_pairwise_combinations("${PAIRWISE_SEMANTIC_COMBINATIONS}" ${chunk_size} semantic_chunks)

        set(chunk_index 0)
        foreach(chunk IN LISTS semantic_chunks)
            math(EXPR comb1 "${chunk_index} / 3")
            math(EXPR comb2 "(${chunk_index} % 3)")
            math(EXPR comb3 "${chunk_index} % 3")

            foreach(template_file ${PAIRWISE_TEMPLATE_3_FILES})
                genPartitionCombination_Semantic("${template_file}" 3 "${comb1},${comb2},${comb3}" "${semantic_output_dir}")
            endforeach()

            math(EXPR chunk_index "${chunk_index} + 1")
            if(chunk_index GREATER_EQUAL 9)
                break()
            endif()
        endforeach()

        # Process 2-type pairwise templates
        set(chunk_index 0)
        foreach(chunk IN LISTS semantic_chunks)
            math(EXPR comb1 "${chunk_index} / 3")
            math(EXPR comb2 "(${chunk_index} % 3)")

            foreach(template_file ${PAIRWISE_TEMPLATE_2_FILES})
                genPartitionCombination_Semantic("${template_file}" 2 "${comb1},${comb2}" "${semantic_output_dir}")
            endforeach()

            math(EXPR chunk_index "${chunk_index} + 1")
            if(chunk_index GREATER_EQUAL 6)
                break()
            endif()
        endforeach()

        message(STATUS "✅ Processed ${chunk_index} pairwise template chunks with semantic filtering")
    endif()
endfunction()

# Function to generate semantic type part macros for pairwise operations
function(generate_pairwise_type_parts workload_profile result_var)
    generate_pairwise_semantic_combinations("${workload_profile}" pairwise_combinations)

    list(LENGTH pairwise_combinations total_combinations)
    message(STATUS "Generated ${total_combinations} semantic pairwise combinations for ${workload_profile}")

    if(total_combinations EQUAL 0)
        set(${result_var} "" PARENT_SCOPE)
        return()
    endif()

    # Partition combinations into manageable chunks (3 parts)
    math(EXPR chunk_size "(${total_combinations} + 2) / 3")
    partition_pairwise_combinations("${pairwise_combinations}" ${chunk_size} partitioned_combinations)

    set(type_parts_content "")

    # Generate SD_PAIRWISE_SEMANTIC_TYPES_PART_X macros
    set(part_index 0)
    foreach(chunk IN LISTS partitioned_combinations)
        set(part_content "#define SD_PAIRWISE_SEMANTIC_TYPES_PART_${part_index} \\\n")

        foreach(combination ${chunk})
            string(REPLACE "," ";" combo_parts ${combination})
            list(LENGTH combo_parts combo_len)

            if(combo_len EQUAL 3)
                list(GET combo_parts 0 type1)
                list(GET combo_parts 1 type2)
                list(GET combo_parts 2 type3)
                string(APPEND part_content "    (${type1}, ${type2}, ${type3}) \\\n")
            elseif(combo_len EQUAL 2)
                list(GET combo_parts 0 type1)
                list(GET combo_parts 1 type2)
                string(APPEND part_content "    (${type1}, ${type2}) \\\n")
            endif()
        endforeach()

        # Remove trailing backslash from last line
        string(REGEX REPLACE " \\\\\n$" "\n" part_content "${part_content}")
        string(APPEND type_parts_content "${part_content}\n")

        math(EXPR part_index "${part_index} + 1")
    endforeach()

    set(${result_var} "${type_parts_content}" PARENT_SCOPE)
endfunction()

# Function to override pairwise template processing with semantic approach
function(override_pairwise_processing_with_semantic)
    if(NOT SD_ENABLE_SEMANTIC_FILTERING)
        return()
    endif()

    message(STATUS "Overriding pairwise template processing with semantic generation")

    # Create semantic-aware template list
    set(INSTANTIATION_TEMPLATES_3_SEMANTIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in"
    )
    set(INSTANTIATION_TEMPLATES_2_SEMANTIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_2.cpp.in"
    )

    # Generate semantic combinations for pairwise templates
    if(DEFINED PAIRWISE_SEMANTIC_COMBINATIONS)
        # Convert semantic combinations to traditional COMBINATIONS_3 format for compatibility
        set(SEMANTIC_COMBINATIONS_3 "")
        set(SEMANTIC_COMBINATIONS_2 "")

        # Map semantic combinations to index-based format
        list(LENGTH PAIRWISE_SEMANTIC_COMBINATIONS total_semantic)
        math(EXPR max_combinations "9")  # 3x3 grid

        if(total_semantic GREATER max_combinations)
            set(semantic_subset "")
            set(subset_index 0)
            foreach(combination ${PAIRWISE_SEMANTIC_COMBINATIONS})
                list(APPEND semantic_subset ${combination})
                math(EXPR subset_index "${subset_index} + 1")
                if(subset_index GREATER_EQUAL max_combinations)
                    break()
                endif()
            endforeach()
            set(PAIRWISE_SEMANTIC_COMBINATIONS "${semantic_subset}")
        endif()

        # Generate 3x3 grid of semantic combinations
        foreach(i RANGE 0 2)
            foreach(j RANGE 0 2)
                foreach(k RANGE 0 2)
                    list(APPEND SEMANTIC_COMBINATIONS_3 "${i},${j},${k}")
                endforeach()
            endforeach()
        endforeach()

        # Generate 2x3 grid for 2-type combinations
        foreach(i RANGE 0 2)
            foreach(j RANGE 0 2)
                list(APPEND SEMANTIC_COMBINATIONS_2 "${i},${j}")
            endforeach()
        endforeach()

        # Override existing combinations for pairwise templates
        set(COMBINATIONS_3_PAIRWISE ${SEMANTIC_COMBINATIONS_3} PARENT_SCOPE)
        set(COMBINATIONS_2_PAIRWISE ${SEMANTIC_COMBINATIONS_2} PARENT_SCOPE)

        message(STATUS "Generated semantic-based combinations for pairwise templates")
        list(LENGTH SEMANTIC_COMBINATIONS_3 semantic_3_count)
        list(LENGTH SEMANTIC_COMBINATIONS_2 semantic_2_count)
        message(STATUS "  3-type combinations: ${semantic_3_count}")
        message(STATUS "  2-type combinations: ${semantic_2_count}")
    endif()
endfunction()

# Function to generate workload-specific combinations
function(generate_workload_combinations workload_profile result_var)
    message(STATUS "Generating workload-specific combinations for: ${workload_profile}")
    
    # Define workload-specific type sets
    if(workload_profile STREQUAL "quantization")
        set(core_types "int8;uint8;float32;int32")
        set(aux_types "int16;uint16;double")
    elseif(workload_profile STREQUAL "training")
        set(core_types "float16;bfloat16;float32;double")
        set(aux_types "int32;int64")
    elseif(workload_profile STREQUAL "inference")
        set(core_types "int8;uint8;float16;float32")
        set(aux_types "int32;double")
    elseif(workload_profile STREQUAL "nlp")
        set(core_types "float32;double;int32;int64")
        set(aux_types "float16;int8")
    elseif(workload_profile STREQUAL "cv")
        set(core_types "uint8;int8;float16;float32")
        set(aux_types "int32;double")
    else()
        # Generic workload
        set(core_types "float32;double;int32;int64")
        set(aux_types "int8;uint8;float16")
    endif()
    
    # Generate optimized combinations
    set(optimized_combinations "")
    
    # Core type combinations (highest priority)
    foreach(type1 ${core_types})
        foreach(type2 ${core_types})
            foreach(type3 ${core_types})
                list(APPEND optimized_combinations "${type1},${type2},${type3}")
            endforeach()
        endforeach()
    endforeach()
    
    # Mixed combinations (core + auxiliary)
    list(LENGTH core_types core_count)
    if(core_count LESS 5)  # Only add mixed if we have room
        foreach(core_type ${core_types})
            foreach(aux_type ${aux_types})
                list(GET core_types 0 output_type)
                list(APPEND optimized_combinations "${core_type},${aux_type},${output_type}")
                list(APPEND optimized_combinations "${aux_type},${core_type},${output_type}")
            endforeach()
        endforeach()
    endif()
    
    # Remove duplicates and apply semantic filtering
    list(REMOVE_DUPLICATES optimized_combinations)
    
    # Apply semantic validation to filter out invalid combinations
    set(validated_combinations "")
    foreach(combination ${optimized_combinations})
        string(REPLACE "," ";" combo_parts "${combination}")
        list(LENGTH combo_parts combo_len)
        
        if(combo_len EQUAL 3)
            list(GET combo_parts 0 type1)
            list(GET combo_parts 1 type2)
            list(GET combo_parts 2 type3)
            
            is_semantically_valid_pairwise_combination(${type1} ${type2} ${type3} is_valid)
            if(is_valid)
                apply_workload_filter_pairwise(${type1} ${type2} ${type3} ${workload_profile} workload_valid)
                if(workload_valid)
                    list(APPEND validated_combinations ${combination})
                endif()
            endif()
        endif()
    endforeach()
    
    list(LENGTH optimized_combinations initial_count)
    list(LENGTH validated_combinations final_count)
    message(STATUS "Workload combinations: ${final_count}/${initial_count} (after semantic filtering)")
    
    set(${result_var} "${validated_combinations}" PARENT_SCOPE)
endfunction()

# Function to get all available types
function(get_all_types result_var)
    if(DEFINED SD_TYPES_LIST AND SD_TYPES_LIST)
        set(${result_var} "${SD_TYPES_LIST}" PARENT_SCOPE)
    else()
        set(${result_var} "${ALL_SUPPORTED_TYPES}" PARENT_SCOPE)
    endif()
endfunction()

# Main function to setup semantic pairwise processing
function(setup_semantic_pairwise_processing)
    message(STATUS "Setting up semantic pairwise processing...")
    
    # Initialize semantic pairwise flags
    set(USE_SEMANTIC_PAIRWISE_GENERATION FALSE CACHE BOOL "Use semantic generation for pairwise templates")
    
    # Enable semantic pairwise processing if conditions are met
    if(SD_ENABLE_SEMANTIC_FILTERING AND SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        # Generate semantic combinations for the active profile
        generate_pairwise_semantic_combinations("${SD_TYPE_PROFILE}" SEMANTIC_PAIRWISE_COMBINATIONS)
        
        list(LENGTH SEMANTIC_PAIRWISE_COMBINATIONS semantic_count)
        if(semantic_count GREATER 0)
            message(STATUS "Generated ${semantic_count} semantic pairwise combinations")
            
            # Store semantic combinations for template generation
            set(PAIRWISE_SEMANTIC_COMBINATIONS "${SEMANTIC_PAIRWISE_COMBINATIONS}" CACHE INTERNAL "Semantic pairwise combinations")
            
            # Override standard combination generation for pairwise templates
            set(USE_SEMANTIC_PAIRWISE_GENERATION TRUE CACHE BOOL "Use semantic generation for pairwise templates")
            
            message(STATUS "Semantic pairwise processing enabled")
        else()
            message(STATUS "No semantic combinations generated - falling back to traditional pairwise processing")
        endif()
    else()
        message(STATUS "Semantic pairwise processing disabled - requirements not met")
    endif()
endfunction()
