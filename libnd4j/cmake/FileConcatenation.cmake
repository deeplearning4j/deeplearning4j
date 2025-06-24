# FileConcatenation.cmake
# Clean template concatenation with duplicate import elimination

# Global variables to track includes and content
set(GLOBAL_INCLUDES_SET "" CACHE INTERNAL "Set of unique includes")
set(GLOBAL_INCLUDES_LIST "" CACHE INTERNAL "Ordered list of unique includes")

# Function to extract and deduplicate includes from content
function(extract_and_deduplicate_includes content includes_var content_without_includes_var)
    # Handle empty content
    if(content STREQUAL "")
        set(${includes_var} "" PARENT_SCOPE)
        set(${content_without_includes_var} "" PARENT_SCOPE)
        return()
    endif()

    # Split content into lines
    string(REPLACE "\n" ";" content_lines "${content}")

    set(extracted_includes "")
    set(content_without_includes "")
    set(in_multiline_comment FALSE)

    foreach(line ${content_lines})
        # Handle empty lines
        if(NOT DEFINED line OR line STREQUAL "")
            string(APPEND content_without_includes "\n")
            continue()
        endif()

        # Handle multi-line comments
        if(line MATCHES "/\\*")
            set(in_multiline_comment TRUE)
        endif()
        if(line MATCHES "\\*/")
            set(in_multiline_comment FALSE)
            string(APPEND content_without_includes "${line}\n")
            continue()
        endif()
        if(in_multiline_comment)
            string(APPEND content_without_includes "${line}\n")
            continue()
        endif()

        # Skip single-line comments unless they contain includes
        if(line MATCHES "^[ \t]*//.*#include")
            # This is a commented include, treat as regular content
            string(APPEND content_without_includes "${line}\n")
        elseif(line MATCHES "^[ \t]*//")
            # Regular comment, keep as content
            string(APPEND content_without_includes "${line}\n")
        elseif(line MATCHES "^[ \t]*#include")
            # Extract include statement - use STRIP instead of REGEX REPLACE to avoid empty match issues
            string(STRIP "${line}" cleaned_include)
            if(NOT cleaned_include STREQUAL "")
                list(APPEND extracted_includes "${cleaned_include}")
            endif()
        else()
            # Regular content line
            string(APPEND content_without_includes "${line}\n")
        endif()
    endforeach()

    set(${includes_var} "${extracted_includes}" PARENT_SCOPE)
    set(${content_without_includes_var} "${content_without_includes}" PARENT_SCOPE)
endfunction()

# Function to add includes to global set maintaining order
function(add_includes_to_global_set new_includes)
    foreach(include_stmt ${new_includes})
        # Normalize the include statement
        string(STRIP "${include_stmt}" include_stmt)

        # Check if this include is already in our global set
        list(FIND GLOBAL_INCLUDES_SET "${include_stmt}" found_index)
        if(found_index EQUAL -1)
            # New include, add to both set and ordered list
            list(APPEND GLOBAL_INCLUDES_SET "${include_stmt}")
            list(APPEND GLOBAL_INCLUDES_LIST "${include_stmt}")

            # Update cache
            set(GLOBAL_INCLUDES_SET "${GLOBAL_INCLUDES_SET}" CACHE INTERNAL "Set of unique includes")
            set(GLOBAL_INCLUDES_LIST "${GLOBAL_INCLUDES_LIST}" CACHE INTERNAL "Ordered list of unique includes")
        endif()
    endforeach()
endfunction()

# Function to generate the final includes section
function(generate_includes_section result_var)
    set(includes_section "")

    if(GLOBAL_INCLUDES_LIST)
        string(APPEND includes_section "/*\n * Consolidated Include Section\n * All unique includes from processed templates\n */\n")

        # Group includes by type for better organization
        set(system_includes "")
        set(local_includes "")
        set(other_includes "")

        foreach(include_stmt ${GLOBAL_INCLUDES_LIST})
            if(include_stmt MATCHES "#include[ \t]*<.*>")
                list(APPEND system_includes "${include_stmt}")
            elseif(include_stmt MATCHES "#include[ \t]*\".*\"")
                list(APPEND local_includes "${include_stmt}")
            else()
                list(APPEND other_includes "${include_stmt}")
            endif()
        endforeach()

        # Output system includes first
        if(system_includes)
            string(APPEND includes_section "\n// System includes\n")
            foreach(include_stmt ${system_includes})
                string(APPEND includes_section "${include_stmt}\n")
            endforeach()
        endif()

        # Then local includes
        if(local_includes)
            string(APPEND includes_section "\n// Local includes\n")
            foreach(include_stmt ${local_includes})
                string(APPEND includes_section "${include_stmt}\n")
            endforeach()
        endif()

        # Finally other includes
        if(other_includes)
            string(APPEND includes_section "\n// Other includes\n")
            foreach(include_stmt ${other_includes})
                string(APPEND includes_section "${include_stmt}\n")
            endforeach()
        endif()

        string(APPEND includes_section "\n")
    endif()

    set(${result_var} "${includes_section}" PARENT_SCOPE)
endfunction()

# Process a single template instance with specific type indices
function(process_single_template_instance template_file comb1 comb2 comb3 result_var)
    file(READ "${template_file}" original_content)

    # Set template variables
    set(FL_TYPE_INDEX ${comb1})
    set(TYPE_INDEX_1 ${comb1})
    set(TYPE_INDEX_2 ${comb2})
    set(TYPE_INDEX_3 ${comb3})

    # Determine what #cmakedefine flags to enable based on template content
    set(SD_COMMON_TYPES_GEN 0)
    set(SD_FLOAT_TYPES_GEN 0)
    set(SD_INTEGER_TYPES_GEN 0)
    set(SD_PAIRWISE_TYPES_GEN 0)
    set(SD_SEMANTIC_TYPES_GEN 0)

    if(original_content MATCHES "#cmakedefine[ \t]+SD_COMMON_TYPES_GEN")
        set(SD_COMMON_TYPES_GEN 1)
    endif()
    if(original_content MATCHES "#cmakedefine[ \t]+SD_FLOAT_TYPES_GEN")
        set(SD_FLOAT_TYPES_GEN 1)
    endif()
    if(original_content MATCHES "#cmakedefine[ \t]+SD_INTEGER_TYPES_GEN")
        set(SD_INTEGER_TYPES_GEN 1)
    endif()
    if(original_content MATCHES "#cmakedefine[ \t]+SD_PAIRWISE_TYPES_GEN")
        set(SD_PAIRWISE_TYPES_GEN 1)
    endif()
    if(original_content MATCHES "#cmakedefine[ \t]+SD_SEMANTIC_TYPES_GEN")
        set(SD_SEMANTIC_TYPES_GEN 1)
    endif()

    # Manual string replacement instead of configure_file
    set(processed_content "${original_content}")

    # Replace @variable@ patterns
    string(REPLACE "@FL_TYPE_INDEX@" "${FL_TYPE_INDEX}" processed_content "${processed_content}")
    string(REPLACE "@TYPE_INDEX_1@" "${TYPE_INDEX_1}" processed_content "${processed_content}")
    string(REPLACE "@TYPE_INDEX_2@" "${TYPE_INDEX_2}" processed_content "${processed_content}")
    string(REPLACE "@TYPE_INDEX_3@" "${TYPE_INDEX_3}" processed_content "${processed_content}")

    # Process #cmakedefine directives
    if(SD_COMMON_TYPES_GEN)
        string(REPLACE "#cmakedefine SD_COMMON_TYPES_GEN" "#define SD_COMMON_TYPES_GEN" processed_content "${processed_content}")
    else()
        string(REPLACE "#cmakedefine SD_COMMON_TYPES_GEN" "/* #undef SD_COMMON_TYPES_GEN */" processed_content "${processed_content}")
    endif()

    if(SD_FLOAT_TYPES_GEN)
        string(REPLACE "#cmakedefine SD_FLOAT_TYPES_GEN" "#define SD_FLOAT_TYPES_GEN" processed_content "${processed_content}")
    else()
        string(REPLACE "#cmakedefine SD_FLOAT_TYPES_GEN" "/* #undef SD_FLOAT_TYPES_GEN */" processed_content "${processed_content}")
    endif()

    if(SD_INTEGER_TYPES_GEN)
        string(REPLACE "#cmakedefine SD_INTEGER_TYPES_GEN" "#define SD_INTEGER_TYPES_GEN" processed_content "${processed_content}")
    else()
        string(REPLACE "#cmakedefine SD_INTEGER_TYPES_GEN" "/* #undef SD_INTEGER_TYPES_GEN */" processed_content "${processed_content}")
    endif()

    if(SD_PAIRWISE_TYPES_GEN)
        string(REPLACE "#cmakedefine SD_PAIRWISE_TYPES_GEN" "#define SD_PAIRWISE_TYPES_GEN" processed_content "${processed_content}")
    else()
        string(REPLACE "#cmakedefine SD_PAIRWISE_TYPES_GEN" "/* #undef SD_PAIRWISE_TYPES_GEN */" processed_content "${processed_content}")
    endif()

    if(SD_SEMANTIC_TYPES_GEN)
        string(REPLACE "#cmakedefine SD_SEMANTIC_TYPES_GEN" "#define SD_SEMANTIC_TYPES_GEN" processed_content "${processed_content}")
    else()
        string(REPLACE "#cmakedefine SD_SEMANTIC_TYPES_GEN" "/* #undef SD_SEMANTIC_TYPES_GEN */" processed_content "${processed_content}")
    endif()

    # Validation - check that processing worked
    if(processed_content MATCHES "@[A-Za-z_]+@")
        message(WARNING "Template processing failed for ${template_file} with indices ${comb1},${comb2},${comb3} - unresolved variables remain")
        set(${result_var} "" PARENT_SCOPE)
        return()
    endif()

    if(processed_content MATCHES "#cmakedefine[ \t]+[A-Za-z_]+")
        message(WARNING "CMakeDefine processing failed for ${template_file} with indices ${comb1},${comb2},${comb3} - unresolved directives remain")
        set(${result_var} "" PARENT_SCOPE)
        return()
    endif()

    set(${result_var} "${processed_content}" PARENT_SCOPE)
endfunction()

# Process template with type combinations and include deduplication
function(process_template_with_combinations template_file combination_type combinations result_var)
    if(NOT EXISTS "${template_file}")
        set(${result_var} "" PARENT_SCOPE)
        return()
    endif()

    set(all_content "")
    get_filename_component(template_name ${template_file} NAME_WE)

    # Detect if this is a single-type template
    file(READ "${template_file}" template_content)
    set(is_single_type_template FALSE)

    if(template_content MATCHES "@FL_TYPE_INDEX@" AND
            NOT template_content MATCHES "@TYPE_INDEX_[23]@|@COMB[23]@")
        set(is_single_type_template TRUE)
        message(STATUS "📋 Detected single-type template: ${template_name}")
    endif()

    if(is_single_type_template)
        # Single-type processing: Extract all unique type indices and process each individually
        set(unique_indices "")

        foreach(combination ${combinations})
            string(REPLACE "," ";" comb_list "${combination}")
            foreach(index ${comb_list})
                list(FIND unique_indices ${index} found_pos)
                if(found_pos EQUAL -1)
                    list(APPEND unique_indices ${index})
                endif()
            endforeach()
        endforeach()

        list(SORT unique_indices COMPARE NATURAL)
        message(STATUS "🔢 Processing ${template_name} with individual indices: ${unique_indices}")

        # Process each unique index individually
        foreach(index ${unique_indices})
            # For single-type templates, use the index for all three parameters
            process_single_template_instance("${template_file}" ${index} ${index} ${index} instance_content)

            if(NOT instance_content STREQUAL "")
                # Extract includes and content separately
                extract_and_deduplicate_includes("${instance_content}" extracted_includes content_only)

                # Add includes to global set
                add_includes_to_global_set("${extracted_includes}")

                # Only append the content without includes
                string(APPEND all_content "// === ${template_name}_${index}.cpp ===\n")
                string(APPEND all_content "${content_only}")
                string(APPEND all_content "// === End ${template_name}_${index}.cpp ===\n\n")

                # Verify float16 instantiation
                if(index EQUAL 3)
                    if(instance_content MATCHES "SD_COMMON_TYPES_3")
                        message(STATUS "✅ Float16 ${template_name} instantiation generated (index ${index})")
                    else()
                        message(WARNING "⚠️  Float16 ${template_name} instantiation incomplete")
                    endif()
                endif()
            endif()
        endforeach()

    else()
        # Multi-type processing: Original logic for templates that actually use multiple types
        message(STATUS "📋 Processing multi-type template: ${template_name}")

        foreach(combination ${combinations})
            string(REPLACE "," ";" comb_list "${combination}")
            list(LENGTH comb_list comb_count)

            # Validate combination format
            if(NOT ((combination_type EQUAL 3 AND comb_count EQUAL 3) OR
            (combination_type EQUAL 2 AND comb_count EQUAL 2)))
                continue()
            endif()

            # Extract combination indices
            if(combination_type EQUAL 3)
                list(GET comb_list 0 comb1)
                list(GET comb_list 1 comb2)
                list(GET comb_list 2 comb3)
                set(file_suffix "${comb1}_${comb2}_${comb3}")
            else()
                list(GET comb_list 0 comb1)
                list(GET comb_list 1 comb2)
                set(comb3 ${comb1})
                set(file_suffix "${comb1}_${comb2}")
            endif()

            # Validate indices
            if(DEFINED SD_COMMON_TYPES_COUNT)
                math(EXPR max_index "${SD_COMMON_TYPES_COUNT} - 1")
                if(comb1 GREATER max_index OR comb2 GREATER max_index OR comb3 GREATER max_index)
                    continue()
                endif()
            endif()

            # Process the multi-type template
            process_single_template_instance("${template_file}" ${comb1} ${comb2} ${comb3} instance_content)

            if(NOT instance_content STREQUAL "")
                # Extract includes and content separately
                extract_and_deduplicate_includes("${instance_content}" extracted_includes content_only)

                # Add includes to global set
                add_includes_to_global_set("${extracted_includes}")

                # Only append the content without includes
                string(APPEND all_content "// === ${template_name}_${file_suffix}.cpp ===\n")
                string(APPEND all_content "${content_only}")
                string(APPEND all_content "// === End ${template_name}_${file_suffix}.cpp ===\n\n")
            endif()
        endforeach()
    endif()

    set(${result_var} "${all_content}" PARENT_SCOPE)
endfunction()

# Detect template requirements
function(detect_template_requirements template_file needs_2_type_var needs_3_type_var)
    file(READ "${template_file}" template_content)

    set(needs_2_type FALSE)
    set(needs_3_type FALSE)

    if(template_content MATCHES "TYPE_INDEX_3|COMB3|_template_3")
        set(needs_3_type TRUE)
    endif()

    if(template_content MATCHES "TYPE_INDEX_2|COMB2|_template_2")
        set(needs_2_type TRUE)
    endif()

    if(NOT needs_2_type AND NOT needs_3_type)
        get_filename_component(template_name "${template_file}" NAME)
        if(template_name MATCHES "_3\\.")
            set(needs_3_type TRUE)
        elseif(template_name MATCHES "_2\\.")
            set(needs_2_type TRUE)
        else()
            set(needs_2_type TRUE)
            set(needs_3_type TRUE)
        endif()
    endif()

    set(${needs_2_type_var} ${needs_2_type} PARENT_SCOPE)
    set(${needs_3_type_var} ${needs_3_type} PARENT_SCOPE)
endfunction()

# Process a batch of templates into one concatenated unit
function(process_concatenated_unit unit_index template_files)
    # Clear global includes for this unit
    set(GLOBAL_INCLUDES_SET "" CACHE INTERNAL "Set of unique includes")
    set(GLOBAL_INCLUDES_LIST "" CACHE INTERNAL "Ordered list of unique includes")

    set(output_dir "${CMAKE_BINARY_DIR}")
    set(final_unit_file "${output_dir}/concatenated_unit_${unit_index}.cpp")

    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    set(unit_content_body "")

    foreach(template_file ${template_files})
        get_filename_component(template_name ${template_file} NAME_WE)
        message(STATUS "  Processing template: ${template_name}")

        # Detect template requirements
        detect_template_requirements("${template_file}" needs_2_type needs_3_type)

        # Process with 3-type combinations if needed
        if(needs_3_type AND COMBINATIONS_3)
            process_template_with_combinations("${template_file}" 3 "${COMBINATIONS_3}" template_content)
            string(APPEND unit_content_body "${template_content}")
        endif()

        # Process with 2-type combinations if needed
        if(needs_2_type AND COMBINATIONS_2)
            process_template_with_combinations("${template_file}" 2 "${COMBINATIONS_2}" template_content)
            string(APPEND unit_content_body "${template_content}")
        endif()
    endforeach()

    # Generate the consolidated includes section
    generate_includes_section(includes_section)

    # Combine everything with proper structure
    set(final_content "/*\n * Concatenated Compilation Unit ${unit_index}\n * Generated: ${current_time}\n */\n\n")
    string(APPEND final_content "${includes_section}")
    string(APPEND final_content "/*\n * Template Implementations\n */\n\n")
    string(APPEND final_content "${unit_content_body}")

    file(WRITE "${final_unit_file}" "${final_content}")
    list(APPEND CONCATENATED_SOURCES "${final_unit_file}")
    set(CONCATENATED_SOURCES ${CONCATENATED_SOURCES} PARENT_SCOPE)

    # Report deduplication statistics
    list(LENGTH GLOBAL_INCLUDES_LIST total_unique_includes)
    message(STATUS "✅ Created concatenated_unit_${unit_index}.cpp with ${total_unique_includes} unique includes")
endfunction()

# Main function to concatenate and process compilation units
function(concatenate_and_process_compilation_units target_count)
    message(STATUS "=== CONCATENATING AND PROCESSING COMPILATION UNITS ===")

    # Find all template files
    file(GLOB_RECURSE ALL_TEMPLATE_FILES
            "${CMAKE_SOURCE_DIR}/include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in"
            "${CMAKE_SOURCE_DIR}/include/loops/cpu/compilation_units/*.cpp.in"
            "${CMAKE_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/*.cpp.in"
            "${CMAKE_SOURCE_DIR}/include/helpers/cpu/loops/*.cpp.in"
    )

    list(LENGTH ALL_TEMPLATE_FILES total_files)
    if(total_files EQUAL 0)
        message(FATAL_ERROR "❌ No template files found to process!")
    endif()

    math(EXPR files_per_unit "(${total_files} + ${target_count} - 1) / ${target_count}")
    message(STATUS "📊 Processing ${total_files} templates into ${target_count} units with include deduplication")

    # Ensure we have combinations
    if(NOT DEFINED COMBINATIONS_2 OR NOT DEFINED COMBINATIONS_3)
        message(FATAL_ERROR "❌ Type combinations not initialized!")
    endif()

    list(LENGTH COMBINATIONS_2 combo2_count)
    list(LENGTH COMBINATIONS_3 combo3_count)
    message(STATUS "Using ${combo2_count} 2-type and ${combo3_count} 3-type combinations")

    # Create concatenated units
    set(unit_index 0)
    set(files_in_current_unit 0)
    set(current_unit_files "")

    foreach(template_file ${ALL_TEMPLATE_FILES})
        list(APPEND current_unit_files ${template_file})
        math(EXPR files_in_current_unit "${files_in_current_unit} + 1")

        if(files_in_current_unit GREATER_EQUAL files_per_unit)
            process_concatenated_unit(${unit_index} "${current_unit_files}")
            set(current_unit_files "")
            set(files_in_current_unit 0)
            math(EXPR unit_index "${unit_index} + 1")
        endif()
    endforeach()

    # Handle remaining files
    if(NOT current_unit_files STREQUAL "")
        process_concatenated_unit(${unit_index} "${current_unit_files}")
    endif()

    message(STATUS "✅ Created concatenated compilation units with deduplicated includes")
    set(CONCATENATED_SOURCES ${CONCATENATED_SOURCES} PARENT_SCOPE)
endfunction()

# Initialize dynamic combinations (placeholder - should be defined elsewhere)
function(initialize_dynamic_combinations)
    # This function should be implemented in your main build system
    # Setting up basic combinations as fallback
    if(NOT DEFINED COMBINATIONS_2)
        set(COMBINATIONS_2 "0,0;0,1;1,1;2,2" PARENT_SCOPE)
    endif()
    if(NOT DEFINED COMBINATIONS_3)
        set(COMBINATIONS_3 "0,0,0;0,1,1;1,1,1;2,2,2" PARENT_SCOPE)
    endif()
    message(STATUS "Initialized fallback combinations")
endfunction()

# Main entry point
function(create_4gb_safe_concatenated_units)
    message(STATUS "🔧 Creating concatenated compilation units with include deduplication...")

    set(TARGET_UNITS 4)

    if(NOT DEFINED COMBINATIONS_2 OR NOT DEFINED COMBINATIONS_3)
        message(STATUS "Initializing dynamic combinations...")
        initialize_dynamic_combinations()
    endif()

    # Debug output
    list(LENGTH COMBINATIONS_2 combo2_count)
    list(LENGTH COMBINATIONS_3 combo3_count)
    if(combo2_count GREATER 0)
        list(GET COMBINATIONS_2 0 first_combo2)
        message(STATUS "Debug: First 2-type combination: ${first_combo2}")
    endif()
    if(combo3_count GREATER 0)
        list(GET COMBINATIONS_3 0 first_combo3)
        message(STATUS "Debug: First 3-type combination: ${first_combo3}")
    endif()

    concatenate_and_process_compilation_units(${TARGET_UNITS})

    if(DEFINED CONCATENATED_SOURCES)
        list(LENGTH CONCATENATED_SOURCES unit_count)
        message(STATUS "📊 Created ${unit_count} concatenated units with deduplicated includes")
    endif()
endfunction()

# Clean up existing units
function(clean_concatenated_units)
    file(GLOB existing_units "${CMAKE_BINARY_DIR}/concatenated_unit_*.cpp")
    if(existing_units)
        file(REMOVE ${existing_units})
        list(LENGTH existing_units removed_count)
        message(STATUS "🗑️ Cleaned up ${removed_count} existing concatenated units")
    endif()

    # Clean up any temp processing files
    file(GLOB temp_files "${CMAKE_BINARY_DIR}/temp_*.cpp*")
    if(temp_files)
        file(REMOVE ${temp_files})
    endif()
endfunction()