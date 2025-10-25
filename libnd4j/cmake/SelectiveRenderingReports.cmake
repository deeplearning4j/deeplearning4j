# ============================================================================
# SelectiveRenderingReports.cmake - Reporting and Diagnostic Functions
# Split from SelectiveRenderingCore.cmake for modularity
# ============================================================================

function(report_selective_rendering_statistics)
    message(STATUS "")
    message(STATUS "╔════════════════════════════════════════════════════════════╗")
    message(STATUS "║          SELECTIVE RENDERING STATISTICS                    ║")
    message(STATUS "╠════════════════════════════════════════════════════════════╣")

    # Get type count
    if(DEFINED UNIFIED_TYPE_COUNT)
        set(type_count ${UNIFIED_TYPE_COUNT})
    elseif(DEFINED UNIFIED_ACTIVE_TYPES)
        list(LENGTH UNIFIED_ACTIVE_TYPES type_count)
    else()
        set(type_count 0)
    endif()

    # Calculate theoretical maximums
    if(type_count GREATER 0)
        math(EXPR theoretical_1 "${type_count}")
        math(EXPR theoretical_2 "${type_count} * ${type_count}")
        math(EXPR theoretical_3 "${type_count} * ${type_count} * ${type_count}")
    else()
        set(theoretical_1 0)
        set(theoretical_2 0)
        set(theoretical_3 0)
    endif()

    # Get actual filtered counts
    if(DEFINED UNIFIED_COMBINATIONS_1)
        list(LENGTH UNIFIED_COMBINATIONS_1 actual_1)
    else()
        set(actual_1 ${type_count})
    endif()

    if(DEFINED UNIFIED_COMBINATIONS_2)
        list(LENGTH UNIFIED_COMBINATIONS_2 actual_2)
    else()
        set(actual_2 0)
    endif()

    if(DEFINED UNIFIED_COMBINATIONS_3)
        list(LENGTH UNIFIED_COMBINATIONS_3 actual_3)
    else()
        set(actual_3 0)
    endif()

    # Format type count display
    if(type_count LESS 10)
        set(type_display "  ${type_count}")
    else()
        set(type_display " ${type_count}")
    endif()

    message(STATUS "║ Active Types:       ${type_display} types                              ║")

    # Show profile if set
    if(DEFINED SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        set(profile_display "${SD_TYPE_PROFILE}")
        string(LENGTH "${profile_display}" profile_len)
        if(profile_len LESS 20)
            string(REPEAT " " ${profile_len} spaces)
            string(SUBSTRING "                    " ${profile_len} -1 padding)
            set(profile_display "${profile_display}${padding}")
        endif()
        message(STATUS "║ Active Profile:     ${profile_display}            ║")
    endif()

    message(STATUS "╠════════════════════════════════════════════════════════════╣")
    message(STATUS "║ Template Combinations:                                     ║")

    # Calculate reductions
    if(theoretical_2 GREATER 0 AND actual_2 GREATER 0)
        math(EXPR saved_2 "${theoretical_2} - ${actual_2}")
        if(saved_2 GREATER 0)
            math(EXPR reduction_2 "100 * ${saved_2} / ${theoretical_2}")
        else()
            set(reduction_2 0)
        endif()
    else()
        set(reduction_2 0)
    endif()

    if(theoretical_3 GREATER 0 AND actual_3 GREATER 0)
        math(EXPR saved_3 "${theoretical_3} - ${actual_3}")
        if(saved_3 GREATER 0)
            math(EXPR reduction_3 "100 * ${saved_3} / ${theoretical_3}")
        else()
            set(reduction_3 0)
        endif()
    else()
        set(reduction_3 0)
    endif()

    # Display statistics
    message(STATUS "║   1-type:  ${actual_1}/${theoretical_1} (no filtering needed)               ║")
    message(STATUS "║   2-type:  ${actual_2}/${theoretical_2} (${reduction_2}% reduction)              ║")
    message(STATUS "║   3-type:  ${actual_3}/${theoretical_3} (${reduction_3}% reduction)            ║")

    message(STATUS "╠════════════════════════════════════════════════════════════╣")

    # Build impact
    math(EXPR total_instantiations "${actual_2} + ${actual_3}")
    math(EXPR memory_per_instance 50)
    math(EXPR total_memory_kb "${total_instantiations} * ${memory_per_instance}")
    math(EXPR total_memory_mb "${total_memory_kb} / 1024")

    message(STATUS "║ Build Impact:                                              ║")
    message(STATUS "║   Total instantiations: ${total_instantiations}                           ║")
    message(STATUS "║   Estimated memory:     ~${total_memory_mb} MB                        ║")

    # Optimization level
    if(reduction_3 GREATER 85)
        set(optimization_level "EXCELLENT 🟢")
    elseif(reduction_3 GREATER 70)
        set(optimization_level "GOOD 🟡     ")
    else()
        set(optimization_level "POOR 🔴     ")
    endif()

    message(STATUS "║   Optimization level:   ${optimization_level}                    ║")

    # Create output directory
    set(COMBINATION_REPORT_DIR "${CMAKE_BINARY_DIR}/type_combinations")
    file(MAKE_DIRECTORY "${COMBINATION_REPORT_DIR}")

    # Dump combinations to disk
    dump_type_combinations_to_disk("${COMBINATION_REPORT_DIR}")

    message(STATUS "║   Reports written to:   type_combinations/                ║")
    message(STATUS "╚════════════════════════════════════════════════════════════╝")
    message(STATUS "")
endfunction()

function(dump_type_combinations_to_disk output_dir)
    # Get timestamp for unique filenames
    string(TIMESTAMP timestamp "%Y%m%d_%H%M%S")
    
    # Get profile name for filename
    if(DEFINED SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        set(profile_suffix "_${SD_TYPE_PROFILE}")
    else()
        set(profile_suffix "_default")
    endif()
    
    # Get type information
    _internal_get_type_statistics(type_count actual_1 actual_2 actual_3 
                                  theoretical_1 theoretical_2 theoretical_3
                                  reduction_2 reduction_3)
    
    # Create main summary file
    _internal_write_summary_file("${output_dir}" "${profile_suffix}" "${timestamp}"
                                 ${type_count} ${actual_1} ${actual_2} ${actual_3}
                                 ${theoretical_1} ${theoretical_2} ${theoretical_3}
                                 ${reduction_2} ${reduction_3})
    
    # Write 2-type combinations
    _internal_write_2type_combinations("${output_dir}" "${profile_suffix}")
    
    # Write 3-type combinations with analysis
    _internal_write_3type_combinations("${output_dir}" "${profile_suffix}")
    
    # Write CSV matrix for analysis
    _internal_write_csv_matrix("${output_dir}" "${profile_suffix}")
    
    # Write JSON format for programmatic analysis
    _internal_write_json_analysis("${output_dir}" "${profile_suffix}" "${timestamp}")
    
    # Write validation rules documentation
    _internal_write_validation_rules("${output_dir}" "${profile_suffix}")
    
    # NEW: Generate template usage analysis
    _internal_analyze_template_usage(template_data)
    set(template_report_file "${output_dir}/template_usage_report${profile_suffix}_${timestamp}.txt")
    set(template_csv_file "${output_dir}/template_usage_matrix${profile_suffix}_${timestamp}.csv")
    set(template_json_file "${output_dir}/template_usage${profile_suffix}_${timestamp}.json")
    _internal_write_template_usage_report("${template_report_file}" "${template_data}")
    _internal_write_template_usage_csv("${template_csv_file}" "${template_data}")
    _internal_write_template_usage_json("${template_json_file}" "${template_data}")
    
    # Generate visual HTML report if enabled
    if(SD_ENABLE_VISUAL_REPORTS)
        _internal_generate_html_report("${output_dir}" "${profile_suffix}" "${timestamp}")
    endif()
    
    # Enhanced logging with better formatting
    _internal_log_output_locations("${output_dir}" "${profile_suffix}" "${timestamp}")
endfunction()

function(_internal_get_type_statistics type_count_var actual_1_var actual_2_var actual_3_var
                                       theoretical_1_var theoretical_2_var theoretical_3_var
                                       reduction_2_var reduction_3_var)
    # Get type count
    if(DEFINED UNIFIED_TYPE_COUNT)
        set(type_count ${UNIFIED_TYPE_COUNT})
    elseif(DEFINED UNIFIED_ACTIVE_TYPES)
        list(LENGTH UNIFIED_ACTIVE_TYPES type_count)
    else()
        set(type_count 0)
    endif()
    
    # Calculate theoretical maximums
    if(type_count GREATER 0)
        math(EXPR theoretical_1 "${type_count}")
        math(EXPR theoretical_2 "${type_count} * ${type_count}")
        math(EXPR theoretical_3 "${type_count} * ${type_count} * ${type_count}")
    else()
        set(theoretical_1 0)
        set(theoretical_2 0)
        set(theoretical_3 0)
    endif()
    
    # Get actual filtered counts
    if(DEFINED UNIFIED_COMBINATIONS_1)
        list(LENGTH UNIFIED_COMBINATIONS_1 actual_1)
    else()
        set(actual_1 ${type_count})
    endif()
    
    if(DEFINED UNIFIED_COMBINATIONS_2)
        list(LENGTH UNIFIED_COMBINATIONS_2 actual_2)
    else()
        set(actual_2 0)
    endif()
    
    if(DEFINED UNIFIED_COMBINATIONS_3)
        list(LENGTH UNIFIED_COMBINATIONS_3 actual_3)
    else()
        set(actual_3 0)
    endif()
    
    # Calculate reductions
    if(theoretical_2 GREATER 0 AND actual_2 GREATER 0)
        math(EXPR saved_2 "${theoretical_2} - ${actual_2}")
        math(EXPR reduction_2 "100 * ${saved_2} / ${theoretical_2}")
    else()
        set(reduction_2 0)
    endif()
    
    if(theoretical_3 GREATER 0 AND actual_3 GREATER 0)
        math(EXPR saved_3 "${theoretical_3} - ${actual_3}")
        math(EXPR reduction_3 "100 * ${saved_3} / ${theoretical_3}")
    else()
        set(reduction_3 0)
    endif()
    
    # Set output variables
    set(${type_count_var} ${type_count} PARENT_SCOPE)
    set(${actual_1_var} ${actual_1} PARENT_SCOPE)
    set(${actual_2_var} ${actual_2} PARENT_SCOPE)
    set(${actual_3_var} ${actual_3} PARENT_SCOPE)
    set(${theoretical_1_var} ${theoretical_1} PARENT_SCOPE)
    set(${theoretical_2_var} ${theoretical_2} PARENT_SCOPE)
    set(${theoretical_3_var} ${theoretical_3} PARENT_SCOPE)
    set(${reduction_2_var} ${reduction_2} PARENT_SCOPE)
    set(${reduction_3_var} ${reduction_3} PARENT_SCOPE)
endfunction()

function(_internal_write_summary_file output_dir profile_suffix timestamp
                                      type_count actual_1 actual_2 actual_3
                                      theoretical_1 theoretical_2 theoretical_3
                                      reduction_2 reduction_3)
    set(summary_file "${output_dir}/combination_summary${profile_suffix}_${timestamp}.txt")
    
    file(WRITE "${summary_file}" "================================================================================\n")
    file(APPEND "${summary_file}" "                     TYPE COMBINATION ANALYSIS REPORT                           \n")
    file(APPEND "${summary_file}" "================================================================================\n")
    file(APPEND "${summary_file}" "Generated: ${timestamp}\n")
    file(APPEND "${summary_file}" "Build Directory: ${CMAKE_BINARY_DIR}\n")
    file(APPEND "${summary_file}" "Profile: ${SD_TYPE_PROFILE}\n")
    file(APPEND "${summary_file}" "Semantic Filtering: ${SD_ENABLE_SEMANTIC_FILTERING}\n")
    file(APPEND "${summary_file}" "Aggressive Filtering: ${SD_AGGRESSIVE_SEMANTIC_FILTERING}\n")
    file(APPEND "${summary_file}" "\n")
    
    file(APPEND "${summary_file}" "CONFIGURATION\n")
    file(APPEND "${summary_file}" "================================================================================\n")
    file(APPEND "${summary_file}" "Type Selection Mode: ")
    if(DEFINED SD_SELECTIVE_TYPES AND NOT SD_SELECTIVE_TYPES STREQUAL "")
        file(APPEND "${summary_file}" "SELECTIVE\n")
        file(APPEND "${summary_file}" "Selected Types: ${SD_SELECTIVE_TYPES}\n")
    else()
        file(APPEND "${summary_file}" "ALL TYPES\n")
    endif()
    file(APPEND "${summary_file}" "\n")
    
    # Active types section - display actual type names
    file(APPEND "${summary_file}" "ACTIVE TYPES (${type_count})\n")
    file(APPEND "${summary_file}" "================================================================================\n")
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(index 0)
        foreach(type ${UNIFIED_ACTIVE_TYPES})
            # The type variable already contains the actual type name
            file(APPEND "${summary_file}" "  ${type}\n")
            
            # Get additional info if available
            if(DEFINED SRCORE_TYPE_ENUM_${index})
                set(enum_val "${SRCORE_TYPE_ENUM_${index}}")
            else()
                set(enum_val "N/A")
            endif()
            if(DEFINED SRCORE_TYPE_CPP_${index})
                set(cpp_type "${SRCORE_TYPE_CPP_${index}}")
            else()
                set(cpp_type "N/A")
            endif()
            
            file(APPEND "${summary_file}" "      Enum: ${enum_val}\n")
            file(APPEND "${summary_file}" "      C++ Type: ${cpp_type}\n")
            math(EXPR index "${index} + 1")
        endforeach()
    endif()
    file(APPEND "${summary_file}" "\n")
    
    file(APPEND "${summary_file}" "STATISTICS\n")
    file(APPEND "${summary_file}" "================================================================================\n")
    file(APPEND "${summary_file}" "Template Instantiations:\n")
    file(APPEND "${summary_file}" "  1-type combinations: ${actual_1}/${theoretical_1} (100% - no filtering needed)\n")
    file(APPEND "${summary_file}" "  2-type combinations: ${actual_2}/${theoretical_2} (${reduction_2}% reduction)\n")
    file(APPEND "${summary_file}" "  3-type combinations: ${actual_3}/${theoretical_3} (${reduction_3}% reduction)\n")
    file(APPEND "${summary_file}" "\n")
    
    math(EXPR total_instantiations "${actual_2} + ${actual_3}")
    math(EXPR memory_per_instance 50)
    math(EXPR total_memory_kb "${total_instantiations} * ${memory_per_instance}")
    math(EXPR total_memory_mb "${total_memory_kb} / 1024")
    
    file(APPEND "${summary_file}" "Build Impact:\n")
    file(APPEND "${summary_file}" "  Total instantiations: ${total_instantiations}\n")
    file(APPEND "${summary_file}" "  Estimated memory: ~${total_memory_mb} MB\n")
    file(APPEND "${summary_file}" "  Estimated compile time impact: ")
    if(reduction_3 GREATER 85)
        file(APPEND "${summary_file}" "MINIMAL (>85% reduction)\n")
    elseif(reduction_3 GREATER 70)
        file(APPEND "${summary_file}" "LOW (>70% reduction)\n")
    elseif(reduction_3 GREATER 50)
        file(APPEND "${summary_file}" "MODERATE (>50% reduction)\n")
    else()
        file(APPEND "${summary_file}" "HIGH (<50% reduction)\n")
    endif()
    file(APPEND "${summary_file}" "\n")
    
    file(APPEND "${summary_file}" "FILTERING RULES APPLIED\n")
    file(APPEND "${summary_file}" "================================================================================\n")
    file(APPEND "${summary_file}" "2-Type Combinations:\n")
    file(APPEND "${summary_file}" "  ✓ Same type pairs always valid\n")
    file(APPEND "${summary_file}" "  ✓ Bool pairs with any type (masking/comparison)\n")
    file(APPEND "${summary_file}" "  ✓ Floating point mixed precision allowed\n")
    file(APPEND "${summary_file}" "  ✓ Integer pairs (including mixed signed/unsigned)\n")
    file(APPEND "${summary_file}" "  ✓ Specific int-to-float promotions\n")
    file(APPEND "${summary_file}" "\n")
    
    file(APPEND "${summary_file}" "3-Type Combinations:\n")
    file(APPEND "${summary_file}" "  ✓ Same type for all three always valid\n")
    file(APPEND "${summary_file}" "  ✓ String operations (UTF types)\n")
    file(APPEND "${summary_file}" "  ✓ Mixed precision patterns (FP16->FP32, BF16->FP32)\n")
    file(APPEND "${summary_file}" "  ✓ Quantization/Dequantization patterns\n")
    file(APPEND "${summary_file}" "  ✓ INT8/UINT8 accumulation patterns\n")
    if(SD_AGGRESSIVE_SEMANTIC_FILTERING)
        file(APPEND "${summary_file}" "  ✗ Precision downgrades blocked\n")
        file(APPEND "${summary_file}" "  ✗ Bool to float conversions blocked\n")
        file(APPEND "${summary_file}" "  ✗ Invalid string conversions blocked\n")
    endif()
    file(APPEND "${summary_file}" "\n")
    
    file(APPEND "${summary_file}" "================================================================================\n")
    file(APPEND "${summary_file}" "For detailed combinations, see accompanying files in: ${output_dir}\n")
endfunction()



function(analyze_filtered_combination type1 type2 type3 reason_var)
    set(reason "")
    
    # ENHANCED: More specific reason detection
    # Check for string issues
    if(type1 MATCHES "UTF" OR type2 MATCHES "UTF" OR type3 MATCHES "UTF")
        if(NOT (type1 MATCHES "UTF" AND type2 MATCHES "UTF" AND type3 MATCHES "UTF"))
            if(NOT (type1 MATCHES "UTF" AND type2 MATCHES "INT" AND type3 MATCHES "INT"))
                set(reason "Invalid string operation pattern - strings can only combine with strings or be indexed with integers")
            endif()
        endif()
    # Check for bool->float
    elseif(type1 STREQUAL "BOOL" AND type2 STREQUAL "BOOL")
        if(type3 MATCHES "FLOAT|DOUBLE|HALF|BFLOAT")
            set(reason "Boolean operations cannot produce floating point - logical ops should produce bool or int")
        endif()
    # Check for precision downgrades
    elseif(type1 STREQUAL "DOUBLE" AND type2 STREQUAL "DOUBLE")
        if(type3 MATCHES "FLOAT32|HALF|INT")
            set(reason "Double precision downgrade - potential precision loss")
        endif()
    elseif(type1 STREQUAL "INT64" AND type2 STREQUAL "INT64")
        if(type3 MATCHES "INT32|INT16|INT8")
            set(reason "Integer precision loss - 64-bit to smaller int conversion")
        endif()
    # NEW: Check for invalid float-to-int patterns
    elseif(type1 STREQUAL "FLOAT32" AND type2 STREQUAL "FLOAT32")
        if(type3 MATCHES "INT16|INT32|INT64")
            set(reason "Invalid float-to-int pattern - not a quantization operation")
        endif()
    else()
        set(reason "Does not match any valid operation pattern in semantic rules")
    endif()
    
    set(${reason_var} "${reason}" PARENT_SCOPE)
endfunction()

# NEW: Helper to analyze why a pair is invalid
function(_internal_analyze_invalid_pair type1 type2 reason_var)
    if(type1 MATCHES "UTF" AND NOT type2 MATCHES "UTF|INT")
        set(reason "Incompatible string operation")
    elseif(type2 MATCHES "UTF" AND NOT type1 MATCHES "UTF|INT")
        set(reason "Incompatible string operation")
    elseif(type1 MATCHES "FLOAT|DOUBLE" AND type2 MATCHES "QINT|UTF")
        set(reason "Invalid float-to-quantized conversion")
    else()
        set(reason "Type mismatch - no valid operation pattern")
    endif()
    
    set(${reason_var} "${reason}" PARENT_SCOPE)
endfunction()

# NEW: Helper to analyze why a triple is invalid
function(_internal_analyze_invalid_triple type1 type2 type3 reason_var)
    if(type1 MATCHES "UTF" OR type2 MATCHES "UTF" OR type3 MATCHES "UTF")
        if(NOT (type1 MATCHES "UTF" AND type2 MATCHES "UTF" AND type3 MATCHES "UTF"))
            if(NOT (type1 MATCHES "UTF" AND type2 MATCHES "INT" AND type3 MATCHES "INT"))
                set(reason "Invalid string operation pattern")
            else()
                set(reason "Unknown string pattern")
            endif()
        else()
            set(reason "Unknown string pattern")
        endif()
    elseif(type1 STREQUAL "BOOL" AND type2 STREQUAL "BOOL" AND type3 MATCHES "FLOAT|DOUBLE|HALF|BFLOAT")
        set(reason "Boolean operations cannot produce floating point")
    elseif(type1 STREQUAL "DOUBLE" AND type2 STREQUAL "DOUBLE" AND type3 MATCHES "FLOAT32|HALF|INT")
        set(reason "Double precision downgrade not allowed")
    elseif(type1 STREQUAL "INT64" AND type2 STREQUAL "INT64" AND type3 MATCHES "INT32|INT16|INT8")
        set(reason "Integer precision loss not allowed")
    elseif(type1 STREQUAL "FLOAT32" AND type2 STREQUAL "FLOAT32" AND type3 MATCHES "INT16|INT32|INT64")
        set(reason "Invalid float-to-int pattern (not quantization)")
    else()
        set(reason "Does not match any valid operation pattern")
    endif()
    
    set(${reason_var} "${reason}" PARENT_SCOPE)
endfunction()

# Write 2-type combinations
function(_internal_write_2type_combinations output_dir profile_suffix)
    set(valid_file "${output_dir}/valid_2type_combinations${profile_suffix}.txt")
    set(invalid_file "${output_dir}/invalid_2type_combinations${profile_suffix}.txt")
    set(analysis_file "${output_dir}/2type_combination_analysis${profile_suffix}.txt")
    
    # Headers
    file(WRITE "${valid_file}" "Valid 2-Type Combinations\n")
    file(APPEND "${valid_file}" "========================\n\n")
    
    file(WRITE "${invalid_file}" "Invalid 2-Type Combinations (Filtered Out)\n")
    file(APPEND "${invalid_file}" "=========================================\n\n")
    
    file(WRITE "${analysis_file}" "2-Type Combination Analysis\n")
    file(APPEND "${analysis_file}" "===========================\n\n")
    
    # Write valid combinations with TYPE NAMES
    if(DEFINED UNIFIED_COMBINATIONS_2)
        foreach(combo ${UNIFIED_COMBINATIONS_2})
            string(REPLACE "," ";" parts "${combo}")
            list(GET parts 0 idx1)
            list(GET parts 1 idx2)
            # Get the actual type names from the index
            list(GET UNIFIED_ACTIVE_TYPES ${idx1} type1)
            list(GET UNIFIED_ACTIVE_TYPES ${idx2} type2)
            file(APPEND "${valid_file}" "(${type1}, ${type2})\n")
        endforeach()
    endif()
    
    # Analysis summary
    list(LENGTH UNIFIED_COMBINATIONS_2 valid_count)
    file(APPEND "${analysis_file}" "Total Valid: ${valid_count}\n")
    
    # Count by type category
    set(same_type_count 0)
    set(bool_pair_count 0)
    set(float_pair_count 0)
    set(int_pair_count 0)
    set(mixed_count 0)
    
    foreach(combo ${UNIFIED_COMBINATIONS_2})
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 idx1)
        list(GET parts 1 idx2)
        list(GET UNIFIED_ACTIVE_TYPES ${idx1} type_i)
        list(GET UNIFIED_ACTIVE_TYPES ${idx2} type_j)
        
        if(idx1 EQUAL idx2)
            math(EXPR same_type_count "${same_type_count} + 1")
        elseif(type_i STREQUAL "BOOL" OR type_j STREQUAL "BOOL")
            math(EXPR bool_pair_count "${bool_pair_count} + 1")
        elseif(type_i MATCHES "FLOAT|DOUBLE|HALF|BFLOAT" AND type_j MATCHES "FLOAT|DOUBLE|HALF|BFLOAT")
            math(EXPR float_pair_count "${float_pair_count} + 1")
        elseif(type_i MATCHES "INT|UINT" AND type_j MATCHES "INT|UINT")
            math(EXPR int_pair_count "${int_pair_count} + 1")
        else()
            math(EXPR mixed_count "${mixed_count} + 1")
        endif()
    endforeach()
    
    file(APPEND "${analysis_file}" "\nBreakdown by Category:\n")
    file(APPEND "${analysis_file}" "  Same type pairs: ${same_type_count}\n")
    file(APPEND "${analysis_file}" "  Bool pairs: ${bool_pair_count}\n")
    file(APPEND "${analysis_file}" "  Float pairs: ${float_pair_count}\n")
    file(APPEND "${analysis_file}" "  Integer pairs: ${int_pair_count}\n")
    file(APPEND "${analysis_file}" "  Mixed/Other: ${mixed_count}\n")
endfunction()

function(generate_template_usage_report)
    set(COMBINATION_REPORT_DIR "${CMAKE_BINARY_DIR}/type_combinations")
    file(MAKE_DIRECTORY "${COMBINATION_REPORT_DIR}")
    
    # Get timestamp and profile
    string(TIMESTAMP timestamp "%Y%m%d_%H%M%S")
    if(DEFINED SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        set(profile_suffix "_${SD_TYPE_PROFILE}")
    else()
        set(profile_suffix "_default")
    endif()
    
    set(template_report_file "${COMBINATION_REPORT_DIR}/template_usage_report${profile_suffix}_${timestamp}.txt")
    set(template_csv_file "${COMBINATION_REPORT_DIR}/template_usage_matrix${profile_suffix}_${timestamp}.csv")
    set(template_json_file "${COMBINATION_REPORT_DIR}/template_usage${profile_suffix}_${timestamp}.json")
    
    # Analyze template usage
    _internal_analyze_template_usage(template_data)
    
    # Write reports
    _internal_write_template_usage_report("${template_report_file}" "${template_data}")
    _internal_write_template_usage_csv("${template_csv_file}" "${template_data}")
    _internal_write_template_usage_json("${template_json_file}" "${template_data}")
    
    message(STATUS "📊 Template usage reports written to: ${COMBINATION_REPORT_DIR}")
endfunction()

function(_internal_write_template_usage_json output_file template_data)
    file(WRITE "${output_file}" "{\n")
    file(APPEND "${output_file}" "  \"metadata\": {\n")
    string(TIMESTAMP timestamp "%Y-%m-%d %H:%M:%S")
    file(APPEND "${output_file}" "    \"generated\": \"${timestamp}\",\n")
    file(APPEND "${output_file}" "    \"profile\": \"${SD_TYPE_PROFILE}\"\n")
    file(APPEND "${output_file}" "  },\n")
    file(APPEND "${output_file}" "  \"templates\": [\n")
    
    set(first TRUE)
    foreach(entry ${template_data})
        if(NOT first)
            file(APPEND "${output_file}" ",\n")
        endif()
        set(first FALSE)
        
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 template_name)
        list(GET parts 1 type_count)
        list(LENGTH parts num_parts)
        
        file(APPEND "${output_file}" "    {\n")
        file(APPEND "${output_file}" "      \"name\": \"${template_name}\",\n")
        file(APPEND "${output_file}" "      \"type_count\": ${type_count},\n")
        file(APPEND "${output_file}" "      \"instantiations\": [")
        
        if(num_parts GREATER 2)
            set(first_inst TRUE)
            foreach(idx RANGE 2 ${num_parts})
                if(idx LESS num_parts)
                    list(GET parts ${idx} type_info)
                    if(NOT type_info STREQUAL "")
                        if(NOT first_inst)
                            file(APPEND "${output_file}" ", ")
                        endif()
                        set(first_inst FALSE)
                        file(APPEND "${output_file}" "\"${type_info}\"")
                    endif()
                endif()
            endforeach()
        endif()
        
        file(APPEND "${output_file}" "]\n")
        file(APPEND "${output_file}" "    }")
    endforeach()
    
    file(APPEND "${output_file}" "\n  ]\n")
    file(APPEND "${output_file}" "}\n")
endfunction()

function(_internal_write_template_usage_csv output_file template_data)
    file(WRITE "${output_file}" "Template,Type Count,Total Instantiations,Sample Types\n")
    
    foreach(entry ${template_data})
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 template_name)
        list(GET parts 1 type_count)
        list(LENGTH parts num_parts)
        
        # Count instantiations
        if(num_parts GREATER 2)
            math(EXPR instantiation_count "${num_parts} - 2")
        else()
            set(instantiation_count 0)
        endif()
        
        # Get sample types
        set(sample_types "")
        if(num_parts GREATER 2)
            set(sample_count 0)
            foreach(idx RANGE 2 ${num_parts})
                if(idx LESS num_parts AND sample_count LESS 3)
                    list(GET parts ${idx} type_info)
                    if(NOT type_info STREQUAL "")
                        if(sample_types STREQUAL "")
                            set(sample_types "${type_info}")
                        else()
                            set(sample_types "${sample_types}; ${type_info}")
                        endif()
                        math(EXPR sample_count "${sample_count} + 1")
                    endif()
                endif()
            endforeach()
        endif()
        
        file(APPEND "${output_file}" "${template_name},${type_count},${instantiation_count},\"${sample_types}\"\n")
    endforeach()
endfunction()

function(_internal_write_template_usage_report output_file template_data)
    file(WRITE "${output_file}" "================================================================================\n")
    file(APPEND "${output_file}" "                    TEMPLATE TYPE USAGE ANALYSIS REPORT                         \n")
    file(APPEND "${output_file}" "================================================================================\n")
    string(TIMESTAMP timestamp "%Y-%m-%d %H:%M:%S")
    file(APPEND "${output_file}" "Generated: ${timestamp}\n")
    file(APPEND "${output_file}" "Profile: ${SD_TYPE_PROFILE}\n")
    file(APPEND "${output_file}" "\n")
    
    # Count templates by type
    set(count_1_type 0)
    set(count_2_type 0)
    set(count_3_type 0)
    set(total_instantiations 0)
    
    foreach(entry ${template_data})
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 template_name)
        list(GET parts 1 type_count)
        
        if(type_count EQUAL 1)
            math(EXPR count_1_type "${count_1_type} + 1")
            if(DEFINED UNIFIED_COMBINATIONS_1)
                list(LENGTH UNIFIED_COMBINATIONS_1 combo_count)
            else()
                list(LENGTH UNIFIED_ACTIVE_TYPES combo_count)
            endif()
            math(EXPR total_instantiations "${total_instantiations} + ${combo_count}")
        elseif(type_count EQUAL 2)
            math(EXPR count_2_type "${count_2_type} + 1")
            list(LENGTH UNIFIED_COMBINATIONS_2 combo_count)
            math(EXPR total_instantiations "${total_instantiations} + ${combo_count}")
        elseif(type_count EQUAL 3)
            math(EXPR count_3_type "${count_3_type} + 1")
            list(LENGTH UNIFIED_COMBINATIONS_3 combo_count)
            math(EXPR total_instantiations "${total_instantiations} + ${combo_count}")
        endif()
    endforeach()
    
    file(APPEND "${output_file}" "SUMMARY\n")
    file(APPEND "${output_file}" "================================================================================\n")
    file(APPEND "${output_file}" "Templates using 1 type:  ${count_1_type}\n")
    file(APPEND "${output_file}" "Templates using 2 types: ${count_2_type}\n")
    file(APPEND "${output_file}" "Templates using 3 types: ${count_3_type}\n")
    file(APPEND "${output_file}" "Total instantiations:    ${total_instantiations}\n")
    file(APPEND "${output_file}" "\n")
    
    # Write details for each template
    file(APPEND "${output_file}" "TEMPLATE DETAILS\n")
    file(APPEND "${output_file}" "================================================================================\n")
    
    foreach(entry ${template_data})
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 template_name)
        list(GET parts 1 type_count)
        list(LENGTH parts num_parts)
        
        file(APPEND "${output_file}" "\nTemplate: ${template_name}\n")
        file(APPEND "${output_file}" "Type Parameters: ${type_count}\n")
        
        if(type_count EQUAL 1)
            file(APPEND "${output_file}" "Instantiated Types:\n")
            if(num_parts GREATER 2)
                foreach(idx RANGE 2 ${num_parts})
                    if(idx LESS num_parts)
                        list(GET parts ${idx} type_name)
                        if(NOT type_name STREQUAL "")
                            file(APPEND "${output_file}" "  - ${type_name}\n")
                        endif()
                    endif()
                endforeach()
            endif()
            
            # Add specific template patterns
            if(template_name MATCHES "random")
                file(APPEND "${output_file}" "CUDA Methods:\n")
                file(APPEND "${output_file}" "  - executeCudaSingle\n")
                file(APPEND "${output_file}" "  - executeCudaDouble\n")
                file(APPEND "${output_file}" "  - executeCudaTriple\n")
            elseif(template_name MATCHES "reduce_same")
                file(APPEND "${output_file}" "CUDA Methods:\n")
                file(APPEND "${output_file}" "  - execReduce\n")
                file(APPEND "${output_file}" "  - execReduceScalar\n")
            elseif(template_name MATCHES "specials_single")
                file(APPEND "${output_file}" "CPU Methods:\n")
                file(APPEND "${output_file}" "  - SpecialMethods<T>\n")
                file(APPEND "${output_file}" "  - concatCpuGeneric\n")
                file(APPEND "${output_file}" "  - sortGeneric\n")
                file(APPEND "${output_file}" "  - sortTadGeneric\n")
            endif()
            
        elseif(type_count EQUAL 2)
            file(APPEND "${output_file}" "Instantiated Type Pairs:\n")
            set(pair_count 0)
            if(num_parts GREATER 2)
                foreach(idx RANGE 2 ${num_parts})
                    if(idx LESS num_parts)
                        list(GET parts ${idx} type_pair)
                        if(NOT type_pair STREQUAL "")
                            file(APPEND "${output_file}" "  ${type_pair}\n")
                            math(EXPR pair_count "${pair_count} + 1")
                            if(pair_count GREATER 10)
                                file(APPEND "${output_file}" "  ... (showing first 10 of ")
                                math(EXPR remaining "${num_parts} - 2")
                                file(APPEND "${output_file}" "${remaining} combinations)\n")
                                break()
                            endif()
                        endif()
                    endif()
                endforeach()
            endif()
            
        elseif(type_count EQUAL 3)
            file(APPEND "${output_file}" "Instantiated Type Triples:\n")
            set(triple_count 0)
            if(num_parts GREATER 2)
                foreach(idx RANGE 2 ${num_parts})
                    if(idx LESS num_parts)
                        list(GET parts ${idx} type_triple)
                        if(NOT type_triple STREQUAL "")
                            file(APPEND "${output_file}" "  ${type_triple}\n")
                            math(EXPR triple_count "${triple_count} + 1")
                            if(triple_count GREATER 10)
                                file(APPEND "${output_file}" "  ... (showing first 10 of ")
                                math(EXPR remaining "${num_parts} - 2")
                                file(APPEND "${output_file}" "${remaining} combinations)\n")
                                break()
                            endif()
                        endif()
                    endif()
                endforeach()
            endif()
        endif()
        
        file(APPEND "${output_file}" "--------------------------------------------------------------------------------\n")
    endforeach()
    
    # Add missing symbol analysis section
    file(APPEND "${output_file}" "\n")
    file(APPEND "${output_file}" "POTENTIAL MISSING INSTANTIATIONS\n")
    file(APPEND "${output_file}" "================================================================================\n")
    file(APPEND "${output_file}" "Based on linker errors, check these specific instantiations:\n\n")
    
    file(APPEND "${output_file}" "SpecialMethods missing:\n")
    file(APPEND "${output_file}" "  - concatCpuGeneric for: bool, bfloat16, float, double, float16, int, long long, signed char, unsigned char\n")
    file(APPEND "${output_file}" "  - sortGeneric and sortTadGeneric for: float, unsigned char, int, long long, double\n\n")
    
    file(APPEND "${output_file}" "To fix: Ensure these types are in UNIFIED_COMBINATIONS_1 or derived from UNIFIED_COMBINATIONS_2\n")
endfunction()

function(_internal_analyze_template_usage result_var)
    set(template_usage "")
    
    # Map template names to their type requirements
    set(TEMPLATE_1_TYPE "random;reduce_same;specials_single")
    set(TEMPLATE_2_TYPE "reduce3;reduce_float;indexreduce;reduce_bool;reduce_long;specials_double;indexreduction")
    set(TEMPLATE_3_TYPE "pairwise;scalar;broadcast")
    
    # Analyze 1-type templates
    foreach(template ${TEMPLATE_1_TYPE})
        set(usage_data "${template}|1")
        if(DEFINED UNIFIED_COMBINATIONS_1)
            foreach(combo ${UNIFIED_COMBINATIONS_1})
                if(DEFINED SRCORE_TYPE_NAME_${combo})
                    string(APPEND usage_data "|${SRCORE_TYPE_NAME_${combo}}")
                else()
                    string(APPEND usage_data "|type_${combo}")
                endif()
            endforeach()
        elseif(DEFINED UNIFIED_ACTIVE_TYPES)
            foreach(type ${UNIFIED_ACTIVE_TYPES})
                string(APPEND usage_data "|${type}")
            endforeach()
        endif()
        list(APPEND template_usage "${usage_data}")
    endforeach()
    
    # Analyze 2-type templates
    foreach(template ${TEMPLATE_2_TYPE})
        set(usage_data "${template}|2")
        set(type_pairs "")
        if(DEFINED UNIFIED_COMBINATIONS_2)
            foreach(combo ${UNIFIED_COMBINATIONS_2})
                string(REPLACE "," ";" parts "${combo}")
                list(GET parts 0 i)
                list(GET parts 1 j)
                if(DEFINED SRCORE_TYPE_NAME_${i} AND DEFINED SRCORE_TYPE_NAME_${j})
                    list(APPEND type_pairs "(${SRCORE_TYPE_NAME_${i}},${SRCORE_TYPE_NAME_${j}})")
                else()
                    list(APPEND type_pairs "(${i},${j})")
                endif()
            endforeach()
        endif()
        string(REPLACE ";" "|" type_pairs_str "${type_pairs}")
        string(APPEND usage_data "|${type_pairs_str}")
        list(APPEND template_usage "${usage_data}")
    endforeach()
    
    # Analyze 3-type templates
    foreach(template ${TEMPLATE_3_TYPE})
        set(usage_data "${template}|3")
        set(type_triples "")
        if(DEFINED UNIFIED_COMBINATIONS_3)
            foreach(combo ${UNIFIED_COMBINATIONS_3})
                string(REPLACE "," ";" parts "${combo}")
                list(GET parts 0 i)
                list(GET parts 1 j)
                list(GET parts 2 k)
                if(DEFINED SRCORE_TYPE_NAME_${i} AND DEFINED SRCORE_TYPE_NAME_${j} AND DEFINED SRCORE_TYPE_NAME_${k})
                    list(APPEND type_triples "(${SRCORE_TYPE_NAME_${i}},${SRCORE_TYPE_NAME_${j}},${SRCORE_TYPE_NAME_${k}})")
                else()
                    list(APPEND type_triples "(${i},${j},${k})")
                endif()
            endforeach()
        endif()
        string(REPLACE ";" "|" type_triples_str "${type_triples}")
        string(APPEND usage_data "|${type_triples_str}")
        list(APPEND template_usage "${usage_data}")
    endforeach()
    
    set(${result_var} "${template_usage}" PARENT_SCOPE)
endfunction()

# Write 3-type combinations
function(_internal_write_3type_combinations output_dir profile_suffix)
    set(valid_file "${output_dir}/valid_3type_combinations${profile_suffix}.txt")
    set(invalid_file "${output_dir}/invalid_3type_combinations${profile_suffix}.txt")
    set(analysis_file "${output_dir}/3type_combination_analysis${profile_suffix}.txt")
    set(patterns_file "${output_dir}/3type_patterns${profile_suffix}.txt")
    
    # Headers
    file(WRITE "${valid_file}" "Valid 3-Type Combinations\n")
    file(APPEND "${valid_file}" "========================\n\n")
    
    file(WRITE "${invalid_file}" "Invalid 3-Type Combinations (Filtered Out)\n")
    file(APPEND "${invalid_file}" "=========================================\n\n")
    
    file(WRITE "${analysis_file}" "3-Type Combination Analysis\n")
    file(APPEND "${analysis_file}" "===========================\n\n")
    
    file(WRITE "${patterns_file}" "3-Type Combination Patterns\n")
    file(APPEND "${patterns_file}" "===========================\n\n")
    
    # Write valid combinations with TYPE NAMES
    if(DEFINED UNIFIED_COMBINATIONS_3)
        foreach(combo ${UNIFIED_COMBINATIONS_3})
            string(REPLACE "," ";" parts "${combo}")
            list(GET parts 0 idx1)
            list(GET parts 1 idx2)
            list(GET parts 2 idx3)
            # Get the actual type names from the indices
            list(GET UNIFIED_ACTIVE_TYPES ${idx1} type1)
            list(GET UNIFIED_ACTIVE_TYPES ${idx2} type2)
            list(GET UNIFIED_ACTIVE_TYPES ${idx3} type3)
            file(APPEND "${valid_file}" "(${type1}, ${type2}, ${type3})\n")
        endforeach()
    endif()
    
    # Pattern analysis
    set(identity_count 0)
    set(mixed_precision_count 0)
    set(quantization_count 0)
    set(accumulation_count 0)
    set(comparison_count 0)
    set(other_count 0)
    
    foreach(combo ${UNIFIED_COMBINATIONS_3})
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 idx1)
        list(GET parts 1 idx2)
        list(GET parts 2 idx3)
        list(GET UNIFIED_ACTIVE_TYPES ${idx1} type_i)
        list(GET UNIFIED_ACTIVE_TYPES ${idx2} type_j)
        list(GET UNIFIED_ACTIVE_TYPES ${idx3} type_k)
        
        if(idx1 EQUAL idx2 AND idx2 EQUAL idx3)
            math(EXPR identity_count "${identity_count} + 1")
        elseif((type_i STREQUAL "HALF" AND type_j STREQUAL "HALF" AND type_k STREQUAL "FLOAT32") OR
               (type_i STREQUAL "BFLOAT16" AND type_j STREQUAL "BFLOAT16" AND type_k STREQUAL "FLOAT32"))
            math(EXPR mixed_precision_count "${mixed_precision_count} + 1")
        elseif((type_i MATCHES "INT8|UINT8" AND type_j STREQUAL "FLOAT32") OR
               (type_i STREQUAL "FLOAT32" AND type_j MATCHES "INT8|UINT8"))
            math(EXPR quantization_count "${quantization_count} + 1")
        elseif(type_i MATCHES "INT8|UINT8" AND type_j MATCHES "INT8|UINT8" AND type_k STREQUAL "INT32")
            math(EXPR accumulation_count "${accumulation_count} + 1")
        elseif(type_k STREQUAL "BOOL")
            math(EXPR comparison_count "${comparison_count} + 1")
        else()
            math(EXPR other_count "${other_count} + 1")
        endif()
    endforeach()
    
    list(LENGTH UNIFIED_COMBINATIONS_3 total_count)
    
    file(APPEND "${analysis_file}" "Total Valid: ${total_count}\n\n")
    file(APPEND "${analysis_file}" "Pattern Breakdown:\n")
    file(APPEND "${analysis_file}" "  Identity (same type): ${identity_count}\n")
    file(APPEND "${analysis_file}" "  Mixed precision: ${mixed_precision_count}\n")
    file(APPEND "${analysis_file}" "  Quantization: ${quantization_count}\n")
    file(APPEND "${analysis_file}" "  Accumulation: ${accumulation_count}\n")
    file(APPEND "${analysis_file}" "  Comparison: ${comparison_count}\n")
    file(APPEND "${analysis_file}" "  Other: ${other_count}\n")
    
    # Write pattern examples
    file(APPEND "${patterns_file}" "Common Patterns Found:\n\n")
    file(APPEND "${patterns_file}" "1. Identity Patterns (${identity_count} instances)\n")
    file(APPEND "${patterns_file}" "   Example: (T, T, T) for same type operations\n\n")
    
    if(mixed_precision_count GREATER 0)
        file(APPEND "${patterns_file}" "2. Mixed Precision Patterns (${mixed_precision_count} instances)\n")
        file(APPEND "${patterns_file}" "   Examples: (HALF, HALF, FLOAT32), (BFLOAT16, BFLOAT16, FLOAT32)\n\n")
    endif()
    
    if(quantization_count GREATER 0)
        file(APPEND "${patterns_file}" "3. Quantization Patterns (${quantization_count} instances)\n")
        file(APPEND "${patterns_file}" "   Examples: (INT8, FLOAT32, FLOAT32), (FLOAT32, UINT8, FLOAT32)\n\n")
    endif()
    
    if(accumulation_count GREATER 0)
        file(APPEND "${patterns_file}" "4. Accumulation Patterns (${accumulation_count} instances)\n")
        file(APPEND "${patterns_file}" "   Examples: (INT8, INT8, INT32), (UINT8, UINT8, INT32)\n\n")
    endif()
    
    if(comparison_count GREATER 0)
        file(APPEND "${patterns_file}" "5. Comparison Patterns (${comparison_count} instances)\n")
        file(APPEND "${patterns_file}" "   Output type is BOOL for comparisons\n\n")
    endif()
endfunction()

# Write CSV matrix
function(_internal_write_csv_matrix output_dir profile_suffix)
    set(csv_file "${output_dir}/combination_matrix${profile_suffix}.csv")
    
    file(WRITE "${csv_file}" "Type Combination Matrix\n")
    file(APPEND "${csv_file}" "Profile: ${SD_TYPE_PROFILE}\n")
    file(APPEND "${csv_file}" "\n")
    
    # Header row
    file(APPEND "${csv_file}" "Type1,Type2")
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        foreach(type ${UNIFIED_ACTIVE_TYPES})
            file(APPEND "${csv_file}" ",${type}")
        endforeach()
    endif()
    file(APPEND "${csv_file}" "\n")
    
    # Matrix data
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(type_index_i 0)
        foreach(type_i ${UNIFIED_ACTIVE_TYPES})
            foreach(type_j ${UNIFIED_ACTIVE_TYPES})
                file(APPEND "${csv_file}" "${type_i},${type_j}")
                
                set(type_index_j 0)
                foreach(type_k ${UNIFIED_ACTIVE_TYPES})
                    # Check if this combination exists
                    set(combo_key "${type_index_i},${type_index_j},${type_index_j}")
                    list(FIND UNIFIED_COMBINATIONS_3 "${combo_key}" found_idx)
                    
                    if(found_idx GREATER_EQUAL 0)
                        file(APPEND "${csv_file}" ",✓")
                    else()
                        file(APPEND "${csv_file}" ",")
                    endif()
                    
                    math(EXPR type_index_j "${type_index_j} + 1")
                endforeach()
                
                file(APPEND "${csv_file}" "\n")
            endforeach()
            
            math(EXPR type_index_i "${type_index_i} + 1")
        endforeach()
    endif()
endfunction()

# NEW: Write JSON analysis for programmatic consumption
function(_internal_write_json_analysis output_dir profile_suffix timestamp)
    set(json_file "${output_dir}/combination_analysis${profile_suffix}.json")
    
    # Get statistics
    _internal_get_type_statistics(type_count actual_1 actual_2 actual_3 
                                  theoretical_1 theoretical_2 theoretical_3
                                  reduction_2 reduction_3)
    
    file(WRITE "${json_file}" "{\n")
    file(APPEND "${json_file}" "  \"metadata\": {\n")
    file(APPEND "${json_file}" "    \"timestamp\": \"${timestamp}\",\n")
    file(APPEND "${json_file}" "    \"profile\": \"${SD_TYPE_PROFILE}\",\n")
    file(APPEND "${json_file}" "    \"build_dir\": \"${CMAKE_BINARY_DIR}\",\n")
    file(APPEND "${json_file}" "    \"semantic_filtering\": ${SD_ENABLE_SEMANTIC_FILTERING},\n")
    file(APPEND "${json_file}" "    \"aggressive_filtering\": ${SD_AGGRESSIVE_SEMANTIC_FILTERING}\n")
    file(APPEND "${json_file}" "  },\n")
    
    file(APPEND "${json_file}" "  \"statistics\": {\n")
    file(APPEND "${json_file}" "    \"type_count\": ${type_count},\n")
    file(APPEND "${json_file}" "    \"combinations\": {\n")
    file(APPEND "${json_file}" "      \"single\": { \"actual\": ${actual_1}, \"theoretical\": ${theoretical_1} },\n")
    file(APPEND "${json_file}" "      \"pair\": { \"actual\": ${actual_2}, \"theoretical\": ${theoretical_2}, \"reduction\": ${reduction_2} },\n")
    file(APPEND "${json_file}" "      \"triple\": { \"actual\": ${actual_3}, \"theoretical\": ${theoretical_3}, \"reduction\": ${reduction_3} }\n")
    file(APPEND "${json_file}" "    }\n")
    file(APPEND "${json_file}" "  },\n")
    
    # Write active types
    file(APPEND "${json_file}" "  \"active_types\": [\n")
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(first TRUE)
        foreach(type ${UNIFIED_ACTIVE_TYPES})
            if(NOT first)
                file(APPEND "${json_file}" ",\n")
            endif()
            file(APPEND "${json_file}" "    \"${type}\"")
            set(first FALSE)
        endforeach()
        file(APPEND "${json_file}" "\n")
    endif()
    file(APPEND "${json_file}" "  ]\n")
    
    file(APPEND "${json_file}" "}\n")
endfunction()

# NEW: Write validation rules documentation
function(_internal_write_validation_rules output_dir profile_suffix)
    set(rules_file "${output_dir}/validation_rules${profile_suffix}.md")
    
    file(WRITE "${rules_file}" "# Type Combination Validation Rules\n\n")
    file(APPEND "${rules_file}" "## Overview\n")
    file(APPEND "${rules_file}" "This document describes the semantic validation rules applied to type combinations.\n\n")
    
    file(APPEND "${rules_file}" "## 2-Type Combination Rules\n\n")
    file(APPEND "${rules_file}" "### Always Valid\n")
    file(APPEND "${rules_file}" "- **Same Type**: `(T, T)` for any type T\n")
    file(APPEND "${rules_file}" "- **Bool Operations**: `(Bool, T)` or `(T, Bool)` for any type T\n")
    file(APPEND "${rules_file}" "- **Float Pairs**: Any combination of float types (mixed precision)\n")
    file(APPEND "${rules_file}" "- **Integer Pairs**: Any combination of integer types\n\n")
    
    file(APPEND "${rules_file}" "### Conditional\n")
    file(APPEND "${rules_file}" "- **Int-to-Float**: Specific promotions like `(INT32, FLOAT32)`\n\n")
    
    file(APPEND "${rules_file}" "## 3-Type Combination Rules\n\n")
    file(APPEND "${rules_file}" "### Pattern Categories\n\n")
    
    file(APPEND "${rules_file}" "#### Identity Patterns\n")
    file(APPEND "${rules_file}" "- `(T, T, T)` - Same type for all three\n\n")
    
    file(APPEND "${rules_file}" "#### Mixed Precision Patterns\n")
    file(APPEND "${rules_file}" "- `(HALF, HALF, FLOAT32)` - FP16 to FP32 accumulation\n")
    file(APPEND "${rules_file}" "- `(BFLOAT16, BFLOAT16, FLOAT32)` - BF16 to FP32 accumulation\n\n")
    
    file(APPEND "${rules_file}" "#### Quantization Patterns\n")
    file(APPEND "${rules_file}" "- `(INT8/UINT8, FLOAT32, FLOAT32)` - Dequantization\n")
    file(APPEND "${rules_file}" "- `(FLOAT32, INT8/UINT8, FLOAT32)` - Quantization scale\n")
    file(APPEND "${rules_file}" "- `(INT8, INT8, INT32)` - INT8 accumulation\n\n")
    
    file(APPEND "${rules_file}" "#### String Patterns\n")
    file(APPEND "${rules_file}" "- `(UTF*, UTF*, UTF*)` - String operations\n")
    file(APPEND "${rules_file}" "- `(UTF*, INT32/INT64, INT32/INT64)` - String indexing\n\n")
    
    if(SD_AGGRESSIVE_SEMANTIC_FILTERING)
        file(APPEND "${rules_file}" "## Aggressive Filtering Rules\n\n")
        file(APPEND "${rules_file}" "When aggressive filtering is enabled, these patterns are blocked:\n")
        file(APPEND "${rules_file}" "- Precision downgrades (e.g., DOUBLE to FLOAT32)\n")
        file(APPEND "${rules_file}" "- Bool to floating point conversions\n")
        file(APPEND "${rules_file}" "- Invalid string conversions\n")
        file(APPEND "${rules_file}" "- Integer precision loss\n\n")
    endif()
    
    file(APPEND "${rules_file}" "## Profile: ${SD_TYPE_PROFILE}\n")
    if(SD_TYPE_PROFILE STREQUAL "quantization")
        file(APPEND "${rules_file}" "Optimized for INT8/UINT8 quantized operations.\n")
    elseif(SD_TYPE_PROFILE STREQUAL "training")
        file(APPEND "${rules_file}" "Optimized for mixed precision training (FP16/BF16/FP32).\n")
    elseif(SD_TYPE_PROFILE STREQUAL "inference")
        file(APPEND "${rules_file}" "Balanced for general inference workloads.\n")
    endif()
endfunction()

# NEW: Enhanced logging function
function(_internal_log_output_locations output_dir profile_suffix timestamp)
    message(STATUS "")
    message(STATUS "╔════════════════════════════════════════════════════════════╗")
    message(STATUS "║          TYPE COMBINATION REPORTS GENERATED               ║")
    message(STATUS "╠════════════════════════════════════════════════════════════╣")
    message(STATUS "║ Output Directory: ${output_dir}")
    message(STATUS "╠════════════════════════════════════════════════════════════╣")
    message(STATUS "║ Summary Files:")
    message(STATUS "║   • combination_summary${profile_suffix}_${timestamp}.txt")
    message(STATUS "║   • validation_rules${profile_suffix}.md")
    message(STATUS "║")
    message(STATUS "║ 2-Type Combinations:")
    message(STATUS "║   • valid_2type_combinations${profile_suffix}.txt")
    message(STATUS "║   • invalid_2type_combinations${profile_suffix}.txt")
    message(STATUS "║   • 2type_combination_analysis${profile_suffix}.txt")
    message(STATUS "║")
    message(STATUS "║ 3-Type Combinations:")
    message(STATUS "║   • valid_3type_combinations${profile_suffix}.txt")
    message(STATUS "║   • invalid_3type_combinations${profile_suffix}.txt")
    message(STATUS "║   • 3type_combination_analysis${profile_suffix}.txt")
    message(STATUS "║   • 3type_patterns${profile_suffix}.txt")
    message(STATUS "║")
    message(STATUS "║ Template Usage Reports:")
    message(STATUS "║   • template_usage_report${profile_suffix}_${timestamp}.txt")
    message(STATUS "║   • template_usage_matrix${profile_suffix}_${timestamp}.csv")
    message(STATUS "║   • template_usage${profile_suffix}_${timestamp}.json")
    message(STATUS "║")
    message(STATUS "║ Analysis Files:")
    message(STATUS "║   • combination_matrix${profile_suffix}.csv")
    message(STATUS "║   • combination_analysis${profile_suffix}.json")
    if(SD_ENABLE_VISUAL_REPORTS)
        message(STATUS "║   • combination_report${profile_suffix}.html")
    endif()
    message(STATUS "╚════════════════════════════════════════════════════════════╝")
    message(STATUS "")
endfunction()

# Optional HTML report generator (stub - can be implemented if needed)
function(_internal_generate_html_report output_dir profile_suffix timestamp)
    set(html_file "${output_dir}/combination_report${profile_suffix}.html")
    
    file(WRITE "${html_file}" "<!DOCTYPE html>\n")
    file(APPEND "${html_file}" "<html>\n<head>\n")
    file(APPEND "${html_file}" "<title>Type Combination Report - ${SD_TYPE_PROFILE}</title>\n")
    file(APPEND "${html_file}" "<style>\n")
    file(APPEND "${html_file}" "body { font-family: Arial, sans-serif; margin: 20px; }\n")
    file(APPEND "${html_file}" "h1 { color: #333; }\n")
    file(APPEND "${html_file}" "table { border-collapse: collapse; width: 100%; }\n")
    file(APPEND "${html_file}" "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
    file(APPEND "${html_file}" "th { background-color: #f2f2f2; }\n")
    file(APPEND "${html_file}" ".good { color: green; }\n")
    file(APPEND "${html_file}" ".warning { color: orange; }\n")
    file(APPEND "${html_file}" ".bad { color: red; }\n")
    file(APPEND "${html_file}" "</style>\n")
    file(APPEND "${html_file}" "</head>\n<body>\n")
    
    file(APPEND "${html_file}" "<h1>Type Combination Analysis Report</h1>\n")
    file(APPEND "${html_file}" "<p>Generated: ${timestamp}</p>\n")
    file(APPEND "${html_file}" "<p>Profile: ${SD_TYPE_PROFILE}</p>\n")
    
    # Get statistics
    _internal_get_type_statistics(type_count actual_1 actual_2 actual_3 
                                  theoretical_1 theoretical_2 theoretical_3
                                  reduction_2 reduction_3)
    
    file(APPEND "${html_file}" "<h2>Statistics</h2>\n")
    file(APPEND "${html_file}" "<table>\n")
    file(APPEND "${html_file}" "<tr><th>Metric</th><th>Actual</th><th>Theoretical</th><th>Reduction</th></tr>\n")
    file(APPEND "${html_file}" "<tr><td>1-Type</td><td>${actual_1}</td><td>${theoretical_1}</td><td>0%</td></tr>\n")
    file(APPEND "${html_file}" "<tr><td>2-Type</td><td>${actual_2}</td><td>${theoretical_2}</td><td>${reduction_2}%</td></tr>\n")
    file(APPEND "${html_file}" "<tr><td>3-Type</td><td>${actual_3}</td><td>${theoretical_3}</td><td class='")
    
    if(reduction_3 GREATER 85)
        file(APPEND "${html_file}" "good")
    elseif(reduction_3 GREATER 70)
        file(APPEND "${html_file}" "warning")
    else()
        file(APPEND "${html_file}" "bad")
    endif()
    
    file(APPEND "${html_file}" "'>${reduction_3}%</td></tr>\n")
    file(APPEND "${html_file}" "</table>\n")
    
    file(APPEND "${html_file}" "</body>\n</html>\n")
endfunction()

# ADD this new quiet version that doesn't print to console
function(_internal_quiet_dump_combinations output_dir)
    # Get timestamp for unique filenames
    string(TIMESTAMP timestamp "%Y%m%d_%H%M%S")
    
    # Get profile name for filename
    if(DEFINED SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        set(profile_suffix "_${SD_TYPE_PROFILE}")
    else()
        set(profile_suffix "_default")
    endif()
    
    # Get type information
    _internal_get_type_statistics(type_count actual_1 actual_2 actual_3 
                                  theoretical_1 theoretical_2 theoretical_3
                                  reduction_2 reduction_3)
    
    # Write all the files silently
    _internal_write_summary_file("${output_dir}" "${profile_suffix}" "${timestamp}"
                                 ${type_count} ${actual_1} ${actual_2} ${actual_3}
                                 ${theoretical_1} ${theoretical_2} ${theoretical_3}
                                 ${reduction_2} ${reduction_3})
    
    _internal_write_2type_combinations("${output_dir}" "${profile_suffix}")
    _internal_write_3type_combinations("${output_dir}" "${profile_suffix}")
    _internal_write_csv_matrix("${output_dir}" "${profile_suffix}")
    _internal_write_json_analysis("${output_dir}" "${profile_suffix}" "${timestamp}")
    _internal_write_validation_rules("${output_dir}" "${profile_suffix}")
    
    # Create a simple status file
    file(WRITE "${output_dir}/status.txt" "Diagnostics generated: ${timestamp}\n")
    file(APPEND "${output_dir}/status.txt" "Profile: ${SD_TYPE_PROFILE}\n")
    file(APPEND "${output_dir}/status.txt" "Types: ${type_count}\n")
    file(APPEND "${output_dir}/status.txt" "2-type combinations: ${actual_2}/${theoretical_2}\n")
    file(APPEND "${output_dir}/status.txt" "3-type combinations: ${actual_3}/${theoretical_3}\n")
endfunction()