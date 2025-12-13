# cmake/TemplateCorrelation.cmake
# Template correlation analysis and reporting functions

# Initialize global correlation tracking variables
function(initialize_correlation_tracking)
    set(GLOBAL_USED_TEMPLATES "" CACHE INTERNAL "")
    set(GLOBAL_PROVIDED_TEMPLATES "" CACHE INTERNAL "")
    set(GLOBAL_MISSING_TEMPLATES "" CACHE INTERNAL "")
    set(CORRELATION_DATA "" CACHE INTERNAL "")
    set(FILE_ANALYSIS_DATA "" CACHE INTERNAL "")
endfunction()

include(InstantiationHelpers)
# Process a single source file for template analysis
function(process_source_file SOURCE_FILE SAFE_NAME)
    message(STATUS "Analyzing: ${SOURCE_FILE}")
    
    # Extract used templates - FIX: Added SAFE_NAME as third parameter
    set(used_file "${INST_USED_DIR}/${SAFE_NAME}.used")
    extract_used_templates(${SOURCE_FILE} ${used_file} ${SAFE_NAME})
    set(used_templates ${EXTRACTED_USED_TEMPLATES})
    
    # Extract provided templates
    set(provided_file "${INST_PROVIDED_DIR}/${SAFE_NAME}.provided")
    extract_provided_templates(${SOURCE_FILE} ${provided_file} ${SAFE_NAME})
    set(provided_templates ${EXTRACTED_PROVIDED_TEMPLATES})
    
    # Find missing templates for this file
    set(missing_templates "")
    foreach(used_tmpl ${used_templates})
        normalize_template_name("${used_tmpl}" normalized_used)
        
        set(found FALSE)
        foreach(provided_tmpl ${provided_templates})
            normalize_template_name("${provided_tmpl}" normalized_provided)
            if("${normalized_used}" STREQUAL "${normalized_provided}")
                set(found TRUE)
                break()
            endif()
        endforeach()
        
        if(NOT found)
            list(APPEND missing_templates "${used_tmpl}")
        endif()
    endforeach()
    
    # Write missing templates file
    if(missing_templates)
        string(REPLACE ";" "\n" missing_content "${missing_templates}")
        file(WRITE "${INST_MISSING_DIR}/${SAFE_NAME}.missing" "${missing_content}")
    else()
        file(WRITE "${INST_MISSING_DIR}/${SAFE_NAME}.missing" "")
    endif()
    
    # Update global lists
    set(current_global_used ${GLOBAL_USED_TEMPLATES})
    list(APPEND current_global_used ${used_templates})
    set(GLOBAL_USED_TEMPLATES ${current_global_used} CACHE INTERNAL "")
    
    set(current_global_provided ${GLOBAL_PROVIDED_TEMPLATES})
    list(APPEND current_global_provided ${provided_templates})
    set(GLOBAL_PROVIDED_TEMPLATES ${current_global_provided} CACHE INTERNAL "")
    
    set(current_global_missing ${GLOBAL_MISSING_TEMPLATES})
    list(APPEND current_global_missing ${missing_templates})
    set(GLOBAL_MISSING_TEMPLATES ${current_global_missing} CACHE INTERNAL "")
    
    # Collect analysis data for reporting
    list(LENGTH used_templates used_count)
    list(LENGTH provided_templates provided_count)
    list(LENGTH missing_templates missing_count)
    
    get_filename_component(src_name ${SOURCE_FILE} NAME)
    
    set(current_data ${FILE_ANALYSIS_DATA})
    string(APPEND current_data "    {\n")
    string(APPEND current_data "      \"file\": \"${src_name}\",\n")
    string(APPEND current_data "      \"safe_name\": \"${SAFE_NAME}\",\n")
    string(APPEND current_data "      \"used_count\": ${used_count},\n")
    string(APPEND current_data "      \"provided_count\": ${provided_count},\n")
    string(APPEND current_data "      \"missing_count\": ${missing_count}\n")
    string(APPEND current_data "    },\n")
    set(FILE_ANALYSIS_DATA ${current_data} CACHE INTERNAL "")
endfunction()

# Perform correlation analysis on all collected data
function(perform_correlation_analysis)
    message(STATUS "Performing correlation analysis...")
    
    # Remove duplicates from global lists
    set(all_used ${GLOBAL_USED_TEMPLATES})
    set(all_provided ${GLOBAL_PROVIDED_TEMPLATES})
    set(all_missing ${GLOBAL_MISSING_TEMPLATES})
    
    if(all_used)
        list(REMOVE_DUPLICATES all_used)
    endif()
    if(all_provided)
        list(REMOVE_DUPLICATES all_provided)
    endif()
    if(all_missing)
        list(REMOVE_DUPLICATES all_missing)
    endif()
    
    # Find truly missing templates (used but not provided anywhere)
    set(truly_missing "")
    foreach(used_tmpl ${all_used})
        normalize_template_name("${used_tmpl}" normalized_used)
        
        set(found FALSE)
        foreach(provided_tmpl ${all_provided})
            normalize_template_name("${provided_tmpl}" normalized_provided)
            if("${normalized_used}" STREQUAL "${normalized_provided}")
                set(found TRUE)
                break()
            endif()
        endforeach()
        
        if(NOT found)
            list(APPEND truly_missing "${used_tmpl}")
        endif()
    endforeach()
    
    # Find unused templates (provided but never used)
    set(unused_templates "")
    foreach(provided_tmpl ${all_provided})
        normalize_template_name("${provided_tmpl}" normalized_provided)
        
        set(found FALSE)
        foreach(used_tmpl ${all_used})
            normalize_template_name("${used_tmpl}" normalized_used)
            if("${normalized_provided}" STREQUAL "${normalized_used}")
                set(found TRUE)
                break()
            endif()
        endforeach()
        
        if(NOT found)
            list(APPEND unused_templates "${provided_tmpl}")
        endif()
    endforeach()
    
    # Update global variables with final results
    set(GLOBAL_USED_TEMPLATES ${all_used} CACHE INTERNAL "")
    set(GLOBAL_PROVIDED_TEMPLATES ${all_provided} CACHE INTERNAL "")
    set(GLOBAL_MISSING_TEMPLATES ${truly_missing} CACHE INTERNAL "")
    set(GLOBAL_UNUSED_TEMPLATES ${unused_templates} CACHE INTERNAL "")
    
    # Write consolidated files
    if(all_used)
        string(REPLACE ";" "\n" content "${all_used}")
        file(WRITE "${INST_DIR}/all_used.txt" "${content}")
    endif()
    
    if(all_provided)
        string(REPLACE ";" "\n" content "${all_provided}")
        file(WRITE "${INST_DIR}/all_provided.txt" "${content}")
    endif()
    
    if(truly_missing)
        string(REPLACE ";" "\n" content "${truly_missing}")
        file(WRITE "${INST_DIR}/all_missing.txt" "${content}")
    endif()
    
    if(unused_templates)
        string(REPLACE ";" "\n" content "${unused_templates}")
        file(WRITE "${INST_DIR}/all_unused.txt" "${content}")
    endif()
endfunction()

# Generate correlation reports
function(generate_correlation_reports)
    message(STATUS "Generating correlation reports...")
    
    set(all_used ${GLOBAL_USED_TEMPLATES})
    set(all_provided ${GLOBAL_PROVIDED_TEMPLATES})
    set(truly_missing ${GLOBAL_MISSING_TEMPLATES})
    set(unused ${GLOBAL_UNUSED_TEMPLATES})
    
    list(LENGTH all_used total_used)
    list(LENGTH all_provided total_provided)
    list(LENGTH truly_missing total_missing)
    list(LENGTH unused total_unused)
    
    # Generate text report
    set(report_file "${INST_REPORTS_DIR}/correlation_report.txt")
    file(WRITE ${report_file} "=== TEMPLATE INSTANTIATION CORRELATION REPORT ===\n")
    file(APPEND ${report_file} "Generated: ${CMAKE_HOST_SYSTEM_PROCESSOR} ${CMAKE_HOST_SYSTEM_NAME}\n")
    file(APPEND ${report_file} "Build Type: ${CMAKE_BUILD_TYPE}\n")
    file(APPEND ${report_file} "Datatypes: ${SD_TYPES_LIST}\n")
    file(APPEND ${report_file} "All Ops: ${SD_ALL_OPS}\n\n")
    
    file(APPEND ${report_file} "SUMMARY:\n")
    file(APPEND ${report_file} "  Total Templates Used: ${total_used}\n")
    file(APPEND ${report_file} "  Total Templates Provided: ${total_provided}\n")
    file(APPEND ${report_file} "  Total Templates Missing: ${total_missing}\n")
    file(APPEND ${report_file} "  Total Templates Unused: ${total_unused}\n\n")
    
    if(total_missing GREATER 0)
        file(APPEND ${report_file} "⚠️  MISSING TEMPLATES (WILL CAUSE LINK ERRORS):\n")
        foreach(tmpl ${truly_missing})
            file(APPEND ${report_file} "  - ${tmpl}\n")
        endforeach()
        file(APPEND ${report_file} "\n")
    endif()
    
    if(total_unused GREATER 20)
        file(APPEND ${report_file} "UNUSED TEMPLATES (showing first 20 of ${total_unused}):\n")
        set(count 0)
        foreach(tmpl ${unused})
            if(count LESS 20)
                file(APPEND ${report_file} "  - ${tmpl}\n")
                math(EXPR count "${count} + 1")
            else()
                break()
            endif()
        endforeach()
        file(APPEND ${report_file} "  ... and ${total_unused} - 20 more\n\n")
    elseif(total_unused GREATER 0)
        file(APPEND ${report_file} "UNUSED TEMPLATES:\n")
        foreach(tmpl ${unused})
            file(APPEND ${report_file} "  - ${tmpl}\n")
        endforeach()
        file(APPEND ${report_file} "\n")
    endif()
    
    # Generate JSON report
    set(json_file "${INST_REPORTS_DIR}/correlation.json")
    file(WRITE ${json_file} "{\n")
    file(APPEND ${json_file} "  \"metadata\": {\n")
    file(APPEND ${json_file} "    \"build_type\": \"${CMAKE_BUILD_TYPE}\",\n")
    file(APPEND ${json_file} "    \"datatypes\": \"${SD_TYPES_LIST}\",\n")
    file(APPEND ${json_file} "    \"all_ops\": ${SD_ALL_OPS}\n")
    file(APPEND ${json_file} "  },\n")
    file(APPEND ${json_file} "  \"summary\": {\n")
    file(APPEND ${json_file} "    \"total_used\": ${total_used},\n")
    file(APPEND ${json_file} "    \"total_provided\": ${total_provided},\n")
    file(APPEND ${json_file} "    \"total_missing\": ${total_missing},\n")
    file(APPEND ${json_file} "    \"total_unused\": ${total_unused}\n")
    file(APPEND ${json_file} "  },\n")
    file(APPEND ${json_file} "  \"files\": [\n")
    file(APPEND ${json_file} "${FILE_ANALYSIS_DATA}")
    file(APPEND ${json_file} "    {}\n")
    file(APPEND ${json_file} "  ]\n")
    file(APPEND ${json_file} "}\n")
    
    # Generate recommendations
    generate_recommendations(${total_missing} ${total_unused})
endfunction()

# Generate actionable recommendations
function(generate_recommendations MISSING_COUNT UNUSED_COUNT)
    set(recommend_file "${INST_REPORTS_DIR}/recommendations.txt")
    file(WRITE ${recommend_file} "=== ACTIONABLE RECOMMENDATIONS ===\n\n")
    
    if(MISSING_COUNT GREATER 0)
        file(APPEND ${recommend_file} "⚠️  CRITICAL: ${MISSING_COUNT} missing template instantiations detected!\n")
        file(APPEND ${recommend_file} "These WILL cause link-time errors.\n\n")
        file(APPEND ${recommend_file} "SOLUTION:\n")
        file(APPEND ${recommend_file} "1. Add the file '${INST_DIR}/fixes/missing_instantiations.cpp' to your build\n")
        file(APPEND ${recommend_file} "2. Or add explicit instantiations to the relevant source files\n\n")
        
        file(APPEND ${recommend_file} "The missing templates are listed in:\n")
        file(APPEND ${recommend_file} "  ${INST_DIR}/all_missing.txt\n\n")
    else()
        file(APPEND ${recommend_file} "✅ No missing template instantiations detected.\n\n")
    endif()
    
    if(UNUSED_COUNT GREATER 50)
        file(APPEND ${recommend_file} "💡 OPTIMIZATION OPPORTUNITY:\n")
        file(APPEND ${recommend_file} "${UNUSED_COUNT} unused template instantiations found.\n")
        file(APPEND ${recommend_file} "Removing these could significantly reduce binary size.\n\n")
        file(APPEND ${recommend_file} "Review the list in: ${INST_DIR}/all_unused.txt\n\n")
    endif()
    
    # Add type-specific recommendations
    if(SD_TYPES_LIST)
        file(APPEND ${recommend_file} "TYPE CONFIGURATION:\n")
        file(APPEND ${recommend_file} "Current types: ${SD_TYPES_LIST}\n")
        
        if(MISSING_COUNT GREATER 0)
            file(APPEND ${recommend_file} "Some missing instantiations may be due to type restrictions.\n")
            file(APPEND ${recommend_file} "Consider if all necessary types are included.\n")
        endif()
    endif()
endfunction()

# Generate fix files for missing instantiations
function(generate_instantiation_fixes)
    set(fixes_dir "${INST_DIR}/fixes")
    file(MAKE_DIRECTORY ${fixes_dir})
    
    set(missing ${GLOBAL_MISSING_TEMPLATES})
    if(NOT missing)
        return()
    endif()
    
    message(STATUS "Generating instantiation fix files...")
    
    # Generate main fix file
    set(fix_file "${fixes_dir}/missing_instantiations.cpp")
    file(WRITE ${fix_file} "// Auto-generated template instantiation fixes\n")
    file(APPEND ${fix_file} "// Generated by cmake/ExtractInstantiations.cmake\n")
    file(APPEND ${fix_file} "// Add this file to your build to resolve missing instantiations\n\n")
    
    # Add necessary includes (customize based on your project)
    file(APPEND ${fix_file} "#include <array/NDArray.h>\n")
    file(APPEND ${fix_file} "#include <array/DataBuffer.h>\n")
    file(APPEND ${fix_file} "#include <helpers/BroadcastHelper.h>\n")
    file(APPEND ${fix_file} "#include <helpers/LoopKind.h>\n")
    file(APPEND ${fix_file} "#include <execution/LaunchContext.h>\n\n")
    
    # Group templates by base class
    set(ndarray_instantiations "")
    set(databuffer_instantiations "")
    set(broadcast_instantiations "")
    set(other_instantiations "")
    
    foreach(tmpl ${missing})
        if(tmpl MATCHES "^NDArray<")
            list(APPEND ndarray_instantiations "${tmpl}")
        elseif(tmpl MATCHES "^DataBuffer<")
            list(APPEND databuffer_instantiations "${tmpl}")
        elseif(tmpl MATCHES "^BroadcastHelper<")
            list(APPEND broadcast_instantiations "${tmpl}")
        else()
            list(APPEND other_instantiations "${tmpl}")
        endif()
    endforeach()
    
    # Write grouped instantiations
    if(ndarray_instantiations)
        file(APPEND ${fix_file} "// NDArray instantiations\n")
        file(APPEND ${fix_file} "namespace sd {\n")
        foreach(tmpl ${ndarray_instantiations})
            file(APPEND ${fix_file} "template class ${tmpl};\n")
        endforeach()
        file(APPEND ${fix_file} "}\n\n")
    endif()
    
    if(databuffer_instantiations)
        file(APPEND ${fix_file} "// DataBuffer instantiations\n")
        file(APPEND ${fix_file} "namespace sd {\n")
        foreach(tmpl ${databuffer_instantiations})
            file(APPEND ${fix_file} "template class ${tmpl};\n")
        endforeach()
        file(APPEND ${fix_file} "}\n\n")
    endif()
    
    if(broadcast_instantiations)
        file(APPEND ${fix_file} "// BroadcastHelper instantiations\n")
        file(APPEND ${fix_file} "namespace sd {\n")
        file(APPEND ${fix_file} "namespace helpers {\n")
        foreach(tmpl ${broadcast_instantiations})
            file(APPEND ${fix_file} "template class ${tmpl};\n")
        endforeach()
        file(APPEND ${fix_file} "}\n}\n\n")
    endif()
    
    if(other_instantiations)
        file(APPEND ${fix_file} "// Other instantiations\n")
        foreach(tmpl ${other_instantiations})
            file(APPEND ${fix_file} "template class ${tmpl};\n")
        endforeach()
    endif()
    
    message(STATUS "Generated fix file: ${fix_file}")
endfunction()

# Display final summary
function(display_analysis_summary)
    set(all_used ${GLOBAL_USED_TEMPLATES})
    set(all_provided ${GLOBAL_PROVIDED_TEMPLATES})
    set(truly_missing ${GLOBAL_MISSING_TEMPLATES})
    set(unused ${GLOBAL_UNUSED_TEMPLATES})
    
    list(LENGTH all_used total_used)
    list(LENGTH all_provided total_provided)
    list(LENGTH truly_missing total_missing)
    list(LENGTH unused total_unused)
    
    message(STATUS "")
    message(STATUS "=== INSTANTIATION CORRELATION COMPLETE ===")
    message(STATUS "Templates Used: ${total_used}")
    message(STATUS "Templates Provided: ${total_provided}")
    
    if(total_missing GREATER 0)
        message(WARNING "Templates Missing: ${total_missing} - WILL CAUSE LINK ERRORS!")
        message(STATUS "  Fix file generated: ${INST_DIR}/fixes/missing_instantiations.cpp")
    else()
        message(STATUS "Templates Missing: 0 ✓")
    endif()
    
    if(total_unused GREATER 50)
        message(STATUS "Templates Unused: ${total_unused} (optimization opportunity)")
    else()
        message(STATUS "Templates Unused: ${total_unused}")
    endif()
    
    message(STATUS "")
    message(STATUS "Reports generated in: ${INST_REPORTS_DIR}/")
    message(STATUS "  - correlation_report.txt (human readable)")
    message(STATUS "  - correlation.json (machine readable)")
    message(STATUS "  - recommendations.txt (actionable items)")
    
    if(total_missing GREATER 0)
        message(STATUS "")
        message(STATUS "⚠️  ACTION REQUIRED: Add missing instantiations to fix link errors")
        message(STATUS "   See: ${INST_DIR}/all_missing.txt")
    endif()
    
    message(STATUS "==========================================")
endfunction()
