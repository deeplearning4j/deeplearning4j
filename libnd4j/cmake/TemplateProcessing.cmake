################################################################################
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# See the NOTICE file distributed with this work for additional
# information regarding copyright ownership.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

# TemplateProcessing.cmake - Fixed bounds calculation and template generation
# WITH PROPER VARIABLE PROPAGATION TO PARENT SCOPE

# =============================================================================
# GLOBAL VARIABLE INITIALIZATION
# =============================================================================

# Initialize the global variable that will accumulate all generated sources
set(CUSTOMOPS_GENERIC_SOURCES "" CACHE INTERNAL "Template-generated source files")

# =============================================================================
# TEMPLATE GENERATION FUNCTIONS - WITH PROPER SCOPE PROPAGATION
# =============================================================================

function(genCompilation FL_ITEM generated_sources_var)
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

    # FIXED: Proper bounds calculation with debug output
    if(DEFINED UNIFIED_TYPE_COUNT AND UNIFIED_TYPE_COUNT GREATER 0)
        math(EXPR SD_INTEGER_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        math(EXPR SD_COMMON_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        math(EXPR SD_FLOAT_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        math(EXPR SD_PAIRWISE_TYPES_END "${UNIFIED_TYPE_COUNT} - 1")
        message(STATUS "DEBUG: ${FILE_ITEM_WE} - Using unified bounds, UNIFIED_TYPE_COUNT=${UNIFIED_TYPE_COUNT}, SD_COMMON_TYPES_END=${SD_COMMON_TYPES_END}")
    else()
        set(SD_INTEGER_TYPES_END 7)
        set(SD_COMMON_TYPES_END 12)
        set(SD_FLOAT_TYPES_END 3)
        set(SD_PAIRWISE_TYPES_END 12)
        message(STATUS "DEBUG: ${FILE_ITEM_WE} - Using fallback bounds, SD_COMMON_TYPES_END=${SD_COMMON_TYPES_END}")
    endif()

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

    message(STATUS "DEBUG: ${FILE_ITEM_WE} - TYPE_MATCHES=${TYPE_MATCHES}, RANGE_STOP=${RANGE_STOP}, SD_COMMON_TYPES_GEN=${SD_COMMON_TYPES_GEN}")

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/compilation_units")

    # Local list to accumulate generated files for this template
    set(local_generated_sources ${${generated_sources_var}})

    # CRITICAL FIX: Skip templates that use broken SD_PAIRWISE_TYPES_X macro system
    if(FILE_ITEM_WE MATCHES "pairwise_p")
        message(STATUS "SKIPPING: ${FILE_ITEM_WE} - uses broken SD_PAIRWISE_TYPES_X macro system")
        set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
        return()
    endif()

    # CRITICAL FIX: Skip scalar_p templates that also use broken SD_PAIRWISE_TYPES_X macro system
    if(FILE_ITEM_WE MATCHES "scalar_p")
        message(STATUS "SKIPPING: ${FILE_ITEM_WE} - uses broken SD_PAIRWISE_TYPES_X macro system")
        set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
        return()
    endif()

    # CRITICAL FIX: Skip broadcast_*_p templates that also use broken SD_PAIRWISE_TYPES_X macro system
    if(FILE_ITEM_WE MATCHES "broadcast.*_p")
        message(STATUS "SKIPPING: ${FILE_ITEM_WE} - uses broken SD_PAIRWISE_TYPES_X macro system")
        set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
        return()
    endif()

    if(RANGE_STOP GREATER -1)
        message(STATUS "DEBUG: ${FILE_ITEM_WE} - Generating files for range 0 to ${RANGE_STOP}")
        foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP})
            # Reset flags for each iteration
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

            # Use current flags for configure_file
            set(SD_FLOAT_TYPES_GEN ${CURRENT_SD_FLOAT_TYPES_GEN})
            set(SD_INTEGER_TYPES_GEN ${CURRENT_SD_INTEGER_TYPES_GEN})
            set(SD_COMMON_TYPES_GEN ${CURRENT_SD_COMMON_TYPES_GEN})
            set(SD_PAIRWISE_TYPES_GEN ${CURRENT_SD_PAIRWISE_TYPES_GEN})

            set(GENERATED_SOURCE "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
            configure_file("${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
            list(APPEND local_generated_sources ${GENERATED_SOURCE})

            message(STATUS "DEBUG: Generated ${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}")
        endforeach()
    else()
        message(STATUS "DEBUG: ${FILE_ITEM_WE} - No generation needed, RANGE_STOP=${RANGE_STOP}")
    endif()

    # Propagate the updated list back to the caller
    set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)
endfunction()

function(genPartitionCombination TEMPLATE_FILE COMBINATION_TYPE COMBINATION OUTPUT_DIR generated_sources_var)
    string(REPLACE "," ";" COMB_LIST "${COMBINATION}")
    list(LENGTH COMB_LIST COMB_COUNT)

    if(NOT (COMBINATION_TYPE EQUAL 2 OR COMBINATION_TYPE EQUAL 3))
        message(FATAL_ERROR "Unsupported COMBINATION_TYPE: ${COMBINATION_TYPE}. Use 2 or 3.")
    endif()

    if(NOT ((COMBINATION_TYPE EQUAL 2 AND COMB_COUNT EQUAL 2) OR
    (COMBINATION_TYPE EQUAL 3 AND COMB_COUNT EQUAL 3)))
        message(FATAL_ERROR "Combination length (${COMB_COUNT}) does not match COMBINATION_TYPE (${COMBINATION_TYPE}).")
    endif()

    if(COMBINATION_TYPE EQUAL 2)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        set(PLACEHOLDER1 "@COMB1@")
        set(PLACEHOLDER2 "@COMB2@")
    elseif(COMBINATION_TYPE EQUAL 3)
        list(GET COMB_LIST 0 COMB1)
        list(GET COMB_LIST 1 COMB2)
        list(GET COMB_LIST 2 COMB3)
        set(PLACEHOLDER1 "@COMB1@")
        set(PLACEHOLDER2 "@COMB2@")
        set(PLACEHOLDER3 "@COMB3@")
    endif()

    file(READ "${TEMPLATE_FILE}" TEMPLATE_CONTENT)

    if(COMBINATION_TYPE EQUAL 2)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" FINAL_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" FINAL_CONTENT "${FINAL_CONTENT}")
    elseif(COMBINATION_TYPE EQUAL 3)
        string(REPLACE "${PLACEHOLDER1}" "${COMB1}" TEMP_CONTENT "${TEMPLATE_CONTENT}")
        string(REPLACE "${PLACEHOLDER2}" "${COMB2}" TEMP_CONTENT "${TEMP_CONTENT}")
        string(REPLACE "${PLACEHOLDER3}" "${COMB3}" FINAL_CONTENT "${TEMP_CONTENT}")
    endif()

    get_filename_component(TEMPLATE_BASE "${TEMPLATE_FILE}" NAME_WE)
    string(REPLACE "_template_" "_" OUTPUT_BASE_NAME "${TEMPLATE_BASE}")

    if(COMBINATION_TYPE EQUAL 2)
        set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}.cpp")
    elseif(COMBINATION_TYPE EQUAL 3)
        set(OUTPUT_FILE "${OUTPUT_BASE_NAME}_${COMB1}_${COMB2}_${COMB3}.cpp")
    endif()

    set(GENERATED_FILE "${OUTPUT_DIR}/${OUTPUT_FILE}")
    file(WRITE "${GENERATED_FILE}" "${FINAL_CONTENT}")

    # Add to the list being accumulated
    set(local_generated_sources ${${generated_sources_var}})
    list(APPEND local_generated_sources "${GENERATED_FILE}")
    set(${generated_sources_var} ${local_generated_sources} PARENT_SCOPE)

    message(STATUS "Generated: ${GENERATED_FILE}")
endfunction()

function(removeFileIfExcluded)
    cmake_parse_arguments(
            PARSED_ARGS
            ""
            "FILE_ITEM"
            "LIST_ITEM"
            ${ARGN}
    )
    file(READ ${PARSED_ARGS_FILE_ITEM} FILE_CONTENTS)
    string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED" NOT_EXCLUDED_IDX)

    if(${NOT_EXCLUDED_IDX} GREATER_EQUAL 0)
        set(local_list ${${PARSED_ARGS_LIST_ITEM}})
        set(file_removed FALSE)

        foreach(OP ${SD_OPS_LIST})
            string(FIND "${FILE_CONTENTS}" "NOT_EXCLUDED(OP_${OP})" NOT_EXCLUDED_OP_IDX)

            if(${NOT_EXCLUDED_OP_IDX} LESS 0)
                list(REMOVE_ITEM local_list "${PARSED_ARGS_FILE_ITEM}")
                set(file_removed TRUE)
                break()
            endif()
        endforeach()

        if(file_removed)
            set(${PARSED_ARGS_LIST_ITEM} ${local_list} PARENT_SCOPE)
        endif()
    endif()
endfunction()

# =============================================================================
# IMMEDIATE TEMPLATE GENERATION - WITH PROPER VARIABLE HANDLING
# =============================================================================

message(STATUS "🔧 FORCING IMMEDIATE TEMPLATE GENERATION")

# Debug: Show unified system status
message(STATUS "DEBUG: UNIFIED_TYPE_COUNT = ${UNIFIED_TYPE_COUNT}")
message(STATUS "DEBUG: UNIFIED_ACTIVE_TYPES = ${UNIFIED_ACTIVE_TYPES}")

# Initialize the accumulator variable
set(ALL_GENERATED_SOURCES "")

# Check if we're in CUDA or CPU mode and process accordingly
if(SD_CUDA)
    message(STATUS "Processing CUDA templates...")

    # Process CUDA compilation units
    file(GLOB_RECURSE COMPILATION_UNITS
            ./include/loops/cuda/compilation_units/*.cu.in
            ./include/ops/impl/compilation_units/*.cpp.in)

    foreach(FL_ITEM ${COMPILATION_UNITS})
        genCompilation(${FL_ITEM} ALL_GENERATED_SOURCES)
    endforeach()

else()
    message(STATUS "Processing CPU templates...")

    # Process regular CPU compilation units first
    file(GLOB_RECURSE REGULAR_COMPILATION_UNITS
            ./include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in
            ./include/loops/cpu/compilation_units/*.cpp.in
            ./include/helpers/cpu/loops/*.cpp.in)

    foreach(FL_ITEM ${REGULAR_COMPILATION_UNITS})
        message(STATUS "Processing regular template: ${FL_ITEM}")
        genCompilation(${FL_ITEM} ALL_GENERATED_SOURCES)
    endforeach()

    # Set up combinations
    if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_2 ${UNIFIED_COMBINATIONS_2})
        set(COMBINATIONS_3 ${UNIFIED_COMBINATIONS_3})
        list(LENGTH UNIFIED_COMBINATIONS_2 combo_2_count)
        list(LENGTH UNIFIED_COMBINATIONS_3 combo_3_count)
        message(STATUS "DEBUG: Using UNIFIED combinations - 2-type count: ${combo_2_count}, 3-type count: ${combo_3_count}")
    else()
        # Fallback combinations
        set(COMBINATIONS_3
                "0,0,0" "0,0,1" "0,0,2" "0,1,0" "0,1,1" "0,1,2" "0,2,0" "0,2,1" "0,2,2"
                "1,0,0" "1,0,1" "1,0,2" "1,1,0" "1,1,1" "1,1,2" "1,2,0" "1,2,1" "1,2,2"
                "2,0,0" "2,0,1" "2,0,2" "2,1,0" "2,1,1" "2,1,2" "2,2,0" "2,2,1" "2,2,2")
        set(COMBINATIONS_2 "0,0" "0,1" "1,0" "1,1" "0,2" "2,0" "1,2" "2,1" "2,2")
        message(STATUS "DEBUG: Using fallback combinations")
    endif()

    set(CPU_INST_DIR "${CMAKE_BINARY_DIR}/cpu_instantiations")
    file(MAKE_DIRECTORY "${CPU_INST_DIR}")

    # Process 3-type templates
    set(TEMPLATES_3 "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/comb_compilation_units/pairwise_instantiation_template_3.cpp.in")

    foreach(TEMPLATE_FILE ${TEMPLATES_3})
        if(EXISTS "${TEMPLATE_FILE}")
            message(STATUS "Processing 3-type template: ${TEMPLATE_FILE}")
            foreach(COMBINATION ${COMBINATIONS_3})
                genPartitionCombination(${TEMPLATE_FILE} 3 ${COMBINATION} "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
            endforeach()
        endif()
    endforeach()

    # Process 2-type templates (specials_double, specials_single)
    set(TEMPLATES_2_ACTUAL
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_double.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/impl/compilation_units/specials_single.cpp.in")

    foreach(TEMPLATE_FILE ${TEMPLATES_2_ACTUAL})
        if(EXISTS "${TEMPLATE_FILE}")
            message(STATUS "Processing 2-type template: ${TEMPLATE_FILE}")
            foreach(COMBINATION ${COMBINATIONS_2})
                genPartitionCombination(${TEMPLATE_FILE} 2 ${COMBINATION} "${CPU_INST_DIR}" ALL_GENERATED_SOURCES)
            endforeach()
        else()
            message(STATUS "❌ Template not found: ${TEMPLATE_FILE}")
        endif()
    endforeach()

    # REMOVED: OLD PAIRWISE TEMPLATE GENERATION - this was causing the compilation error
    # The pairwise_instantiation_template_2.cpp.in uses the broken SD_PAIRWISE_TYPES_X system
    message(STATUS "ℹ️ Skipping old pairwise template system - using new combination approach")
endif()

# CRITICAL: Set the global cache variable so it's available in MainBuildFlow
set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} CACHE INTERNAL "Template-generated source files" FORCE)

# Debug output
list(LENGTH ALL_GENERATED_SOURCES total_count)
message(STATUS "✅ Template processing complete: ${total_count} files generated")

# Show some examples
set(sample_count 0)
foreach(generated_file ${ALL_GENERATED_SOURCES})
    if(sample_count LESS 5)
        get_filename_component(filename ${generated_file} NAME)
        message(STATUS "   Generated: ${filename}")
        math(EXPR sample_count "${sample_count} + 1")
    endif()
endforeach()
if(total_count GREATER 5)
    math(EXPR remaining "${total_count} - 5")
    message(STATUS "   ... and ${remaining} more files")
endif()

# CRITICAL: Export variables to parent scope for immediate use
set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} PARENT_SCOPE)

message(STATUS "✅ Template-generated sources available in CUSTOMOPS_GENERIC_SOURCES")
message(STATUS "   Variable contains ${total_count} generated source files")

# =============================================================================
# LEGACY FUNCTION FOR MAINBUILDFLOW COMPATIBILITY - NOW ACTUALLY EXPORTS
# =============================================================================

function(setup_template_processing)
    # The templates were already generated above during include processing
    # This function now makes sure the variable is available in the calling scope
    
    message(STATUS "setup_template_processing() called - propagating template sources")
    
    # Get the current cached value
    get_property(cached_sources CACHE CUSTOMOPS_GENERIC_SOURCES PROPERTY VALUE)
    
    if(cached_sources)
        list(LENGTH cached_sources cached_count)
        message(STATUS "Propagating ${cached_count} template-generated sources to parent scope")
        
        # Set in parent scope so collect_all_sources_with_selective_rendering can access it
        set(CUSTOMOPS_GENERIC_SOURCES ${cached_sources} PARENT_SCOPE)
        
        # Also set as global variable for broader access
        set(CUSTOMOPS_GENERIC_SOURCES ${cached_sources} CACHE INTERNAL "Template-generated source files" FORCE)
    else()
        message(WARNING "⚠️ No template sources found in cache! Template generation may have failed.")
        set(CUSTOMOPS_GENERIC_SOURCES "" PARENT_SCOPE)
    endif()
endfunction()

message(STATUS "✅ TemplateProcessing.cmake loaded - ready for MainBuildFlow integration")