# cmake/TypeSystem.cmake
# Manages data type definitions, validation, profiles, and header generation.


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

# Main entry point to find and process all templates
function(process_all_compilation_units)
    file(GLOB_RECURSE ALL_TEMPLATE_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ops/declarable/helpers/cpu/compilation_units/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/loops/cpu/compilation_units/*.cpp.in"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/helpers/cpu/loops/*.cpp.in"
    )

    set(INSTANTIATION_OUTPUT_DIR "${CMAKE_BINARY_DIR}/compilation_units")
    set(CPU_INSTANTIATION_DIR "${INSTANTIATION_OUTPUT_DIR}/cpu")
    file(MAKE_DIRECTORY "${INSTANTIATION_OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${CPU_INSTANTIATION_DIR}")

    set(ALL_GENERATED_SOURCES "")

    foreach(TEMPLATE_FILE ${ALL_TEMPLATE_FILES})
        detect_template_requirements("${TEMPLATE_FILE}" NEEDS_2_TYPE NEEDS_3_TYPE)

        set(TEMPLATE_GENERATED_SOURCES "")

        if(NEEDS_3_TYPE AND COMBINATIONS_3)
            process_template_unified("${TEMPLATE_FILE}" 3 "${COMBINATIONS_3}" "${CPU_INSTANTIATION_DIR}")
            list(APPEND TEMPLATE_GENERATED_SOURCES ${GENERATED_SOURCES})
        endif()

        if(NEEDS_2_TYPE AND COMBINATIONS_2)
            process_template_unified("${TEMPLATE_FILE}" 2 "${COMBINATIONS_2}" "${CPU_INSTANTIATION_DIR}")
            list(APPEND TEMPLATE_GENERATED_SOURCES ${GENERATED_SOURCES})
        endif()

        list(APPEND ALL_GENERATED_SOURCES ${TEMPLATE_GENERATED_SOURCES})
    endforeach()

    set(CUSTOMOPS_GENERIC_SOURCES ${ALL_GENERATED_SOURCES} PARENT_SCOPE)
endfunction()



# In TypeSystem.cmak
function(get_type_name_from_index index result_var)
    # First check the direct global reverse lookup
    if(DEFINED TYPE_NAME_${index})
        set(${result_var} "${TYPE_NAME_${index}}" PARENT_SCOPE)
        return()
    endif()

    # Fallback: search through SD_ACTIVE_TYPES if available
    if(DEFINED SD_ACTIVE_TYPES)
        list(LENGTH SD_ACTIVE_TYPES active_count)
        if(index LESS active_count)
            list(GET SD_ACTIVE_TYPES ${index} type_name)
            set(${result_var} "${type_name}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Last resort: search through index mappings
    if(DEFINED SD_COMMON_TYPES_COUNT AND DEFINED SD_ACTIVE_TYPES)
        foreach(type_name ${SD_ACTIVE_TYPES})
            if(DEFINED SD_COMMON_TYPES_INDEX_${type_name})
                if(SD_COMMON_TYPES_INDEX_${type_name} EQUAL ${index})
                    set(${result_var} "${type_name}" PARENT_SCOPE)
                    return()
                endif()
            endif()
        endforeach()
    endif()

    message(WARNING "get_type_name_from_index: No type found for index ${index}")
    set(${result_var} "unknown_type_${index}" PARENT_SCOPE)
endfunction()

# Generates N-dimensional type combinations based on available types and semantic filtering
function(generate_dynamic_combinations combination_type workload_profile result_var)
    if(NOT DEFINED SD_COMMON_TYPES_COUNT OR SD_COMMON_TYPES_COUNT EQUAL 0)
        message(FATAL_ERROR "Dynamic type system not initialized or no types available")
    endif()

    set(combinations "")
    math(EXPR max_index "${SD_COMMON_TYPES_COUNT} - 1")

    if(combination_type EQUAL 3)
        foreach(i RANGE 0 ${max_index})
            foreach(j RANGE 0 ${max_index})
                foreach(k RANGE 0 ${max_index})
                    get_type_name_from_index(${i} type1)
                    get_type_name_from_index(${j} type2)
                    get_type_name_from_index(${k} type3)

                    set(is_valid TRUE)
                    if(SD_ENABLE_SEMANTIC_FILTERING AND workload_profile AND NOT workload_profile STREQUAL "")
                        if(COMMAND is_semantically_valid_combination)
                            is_semantically_valid_combination("${type1}" "${type2}" "${type3}" "${workload_profile}" is_valid)
                        endif()
                    endif()

                    if(is_valid)
                        list(APPEND combinations "${i},${j},${k}")
                    endif()
                endforeach()
            endforeach()
        endforeach()
    elseif(combination_type EQUAL 2)
        foreach(i RANGE 0 ${max_index})
            foreach(j RANGE 0 ${max_index})
                get_type_name_from_index(${i} type1)
                get_type_name_from_index(${j} type2)

                set(is_valid TRUE)
                if(SD_ENABLE_SEMANTIC_FILTERING AND workload_profile AND NOT workload_profile STREQUAL "")
                    if(COMMAND is_semantically_valid_combination)
                        is_semantically_valid_combination("${type1}" "${type2}" "${type1}" "${workload_profile}" is_valid)
                    endif()
                endif()

                if(is_valid)
                    list(APPEND combinations "${i},${j}")
                endif()
            endforeach()
        endforeach()
    else()
        message(FATAL_ERROR "Invalid combination_type: ${combination_type}. Must be 2 or 3.")
    endif()

    set(${result_var} "${combinations}" PARENT_SCOPE)
endfunction()


function(initialize_dynamic_combinations)
    if(NOT DEFINED SD_COMMON_TYPES_COUNT OR SD_COMMON_TYPES_COUNT EQUAL 0)
        message(WARNING "No types available for combination generation.")
        set(COMBINATIONS_3 "" PARENT_SCOPE)
        set(COMBINATIONS_2 "" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "=== Initializing Dynamic Type Combinations ===")

    set(workload_profile "")
    if(SD_ENABLE_SEMANTIC_FILTERING AND DEFINED SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        set(workload_profile "${SD_TYPE_PROFILE}")
    endif()

    generate_dynamic_combinations(3 "${workload_profile}" generated_combinations_3)
    generate_dynamic_combinations(2 "${workload_profile}" generated_combinations_2)

    list(LENGTH generated_combinations_3 count_3)
    list(LENGTH generated_combinations_2 count_2)

    if(SD_ENABLE_SEMANTIC_FILTERING AND workload_profile AND NOT workload_profile STREQUAL "")
        math(EXPR total_possible_3 "${SD_COMMON_TYPES_COUNT} * ${SD_COMMON_TYPES_COUNT} * ${SD_COMMON_TYPES_COUNT}")
        math(EXPR total_possible_2 "${SD_COMMON_TYPES_COUNT} * ${SD_COMMON_TYPES_COUNT}")
        if(total_possible_3 GREATER 0)
            math(EXPR reduction_3 "100 - (100 * ${count_3} / ${total_possible_3})")
        else()
            set(reduction_3 0)
        endif()
        if(total_possible_2 GREATER 0)
            math(EXPR reduction_2 "100 - (100 * ${count_2} / ${total_possible_2})")
        else()
            set(reduction_2 0)
        endif()
        message(STATUS "Generated combinations with semantic filtering ('${workload_profile}'):")
        message(STATUS "  3-type: ${count_3} / ${total_possible_3} (~${reduction_3}% reduction)")
        message(STATUS "  2-type: ${count_2} / ${total_possible_2} (~${reduction_2}% reduction)")
    else()
        message(STATUS "Generated combinations without semantic filtering:")
        message(STATUS "  3-type: ${count_3}")
        message(STATUS "  2-type: ${count_2}")
    endif()

    set(COMBINATIONS_3 "${generated_combinations_3}" PARENT_SCOPE)
    set(COMBINATIONS_2 "${generated_combinations_2}" PARENT_SCOPE)
endfunction()


function(setup_cpu_environment)
    set(DEFAULT_ENGINE "samediff::ENGINE_CPU")

    message("CPU BLAS")
    add_definitions(-D__CPUBLAS__=true)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
    set(CUSTOMOPS_GENERIC_SOURCES "")

    file(GLOB_RECURSE PERF_SOURCES ./include/performance/*.cpp ./include/performance/*.h)
    file(GLOB_RECURSE EXCEPTIONS_SOURCES ./include/exceptions/*.cpp ./include/exceptions/*.h)
    file(GLOB_RECURSE EXEC_SOURCES ./include/execution/*.cpp ./include/execution/*.h)
    file(GLOB_RECURSE TYPES_SOURCES ./include/types/*.cpp ./include/types/*.h)
    file(GLOB_RECURSE ARRAY_SOURCES ./include/array/*.cpp ./include/array/*.h)
    file(GLOB_RECURSE MEMORY_SOURCES ./include/memory/*.cpp ./include/memory/*.h)
    file(GLOB_RECURSE GRAPH_SOURCES ./include/graph/*.cpp ./include/graph/*.h)
    file(GLOB_RECURSE CUSTOMOPS_SOURCES ./include/ops/declarable/generic/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_IMPL_SOURCES ./include/ops/declarable/helpers/impl/*.cpp)
    file(GLOB_RECURSE CUSTOMOPS_HELPERS_CPU_SOURCES ./include/ops/declarable/helpers/cpu/*.cpp)
    file(GLOB_RECURSE OPS_SOURCES ./include/ops/impl/*.cpp ./include/ops/declarable/impl/*.cpp ./include/ops/*.h)
    file(GLOB_RECURSE INDEXING_SOURCES ./include/indexing/*.cpp ./include/indexing/*.h)
    file(GLOB_RECURSE HELPERS_SOURCES ./include/build_info.cpp ./include/ConstMessages.cpp ./include/helpers/*.cpp  ./include/helpers/cpu/*.cpp ./include/helpers/*.h)
    file(GLOB_RECURSE LEGACY_SOURCES ./include/legacy/impl/*.cpp ./include/legacy/cpu/*.cpp ./include/legacy/*.h)
    file(GLOB_RECURSE LOOPS_SOURCES ./include/loops/*.cpp ./include/loops/*.h)

    set(ALL_SOURCES "")

    set(CUSTOMOPS_ONEDNN_SOURCES "")
    if(HAVE_ONEDNN)
        message(STATUS "Including OneDNN platform sources")
        file(GLOB_RECURSE CUSTOMOPS_ONEDNN_SOURCES_TMP ./include/ops/declarable/platform/mkldnn/*.cpp ./include/ops/declarable/platform/mkldnn/mkldnnUtils.h)
        set(CUSTOMOPS_ONEDNN_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES_TMP})
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ONEDNN_SOURCES})
    else()
        message(STATUS "Skipping OneDNN platform sources (OneDNN helper disabled)")
    endif()

    set(CUSTOMOPS_ARMCOMPUTE_SOURCES "")
    if(HAVE_ARMCOMPUTE)
        message(STATUS "Including ARM Compute platform sources")
        file(GLOB_RECURSE CUSTOMOPS_ARMCOMPUTE_SOURCES_TMP ./include/ops/declarable/platform/armcompute/*.cpp ./include/ops/declarable/platform/armcompute/*.h)
        set(CUSTOMOPS_ARMCOMPUTE_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES_TMP})
        list(APPEND ALL_SOURCES ${CUSTOMOPS_ARMCOMPUTE_SOURCES})
    else()
        message(STATUS "Skipping ARM Compute platform sources (ARM Compute helper disabled)")
    endif()

    if (SD_X86_BUILD)
        set_source_files_properties(./include/helpers/impl/OpTracker.cpp PROPERTIES COMPILE_FLAGS "-march=x86-64 -mtune=generic")
    endif()

    set(STATIC_SOURCES_TO_CHECK
            ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES}
            ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
            ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}
            ${LEGACY_SOURCES} ${LOOPS_SOURCES} ${CUSTOMOPS_ONEDNN_SOURCES} ${CUSTOMOPS_ARMCOMPUTE_SOURCES}
    )

    if(NOT SD_ALL_OPS)
        message("Not all SD OPS INCLUDED - Filtering sources")
        set(FILTERED_STATIC_SOURCES "")
        foreach(SRC_FILE ${STATIC_SOURCES_TO_CHECK})
            set(temp_list "${SRC_FILE}")
            removeFileIfExcluded(FILE_ITEM "${SRC_FILE}" LIST_ITEM "temp_list")
            if(temp_list)
                list(APPEND FILTERED_STATIC_SOURCES "${SRC_FILE}")
            else()
                message("Excluding file due to op restrictions: ${SRC_FILE}")
            endif()
        endforeach()
        list(REMOVE_ITEM ALL_SOURCES ${STATIC_SOURCES_TO_CHECK})
        list(APPEND ALL_SOURCES ${FILTERED_STATIC_SOURCES})
    else()
        list(APPEND ALL_SOURCES
                ${PERF_SOURCES} ${EXCEPTIONS_SOURCES} ${EXEC_SOURCES} ${TYPES_SOURCES} ${ARRAY_SOURCES}
                ${MEMORY_SOURCES} ${GRAPH_SOURCES} ${CUSTOMOPS_SOURCES} ${CUSTOMOPS_HELPERS_IMPL_SOURCES}
                ${CUSTOMOPS_HELPERS_CPU_SOURCES} ${OPS_SOURCES} ${INDEXING_SOURCES} ${HELPERS_SOURCES}
                ${LEGACY_SOURCES} ${LOOPS_SOURCES}
        )
        list(REMOVE_DUPLICATES ALL_SOURCES)
    endif()

    list(APPEND ALL_SOURCES ${CUSTOMOPS_GENERIC_SOURCES})

    message(STATUS "=== INITIALIZING DYNAMIC TYPE SYSTEM ===")
    extract_type_definitions_from_header(AVAILABLE_TYPES)
    build_indexed_type_lists()

    if(NOT DEFINED SD_COMMON_TYPES_COUNT OR SD_COMMON_TYPES_COUNT EQUAL 0)
        message(FATAL_ERROR "Failed to build dynamic type system! SD_COMMON_TYPES_COUNT = ${SD_COMMON_TYPES_COUNT}")
    endif()

    initialize_dynamic_combinations()
    process_all_compilation_units()

    list(APPEND ALL_SOURCES ${CUSTOMOPS_GENERIC_SOURCES})
    list(REMOVE_DUPLICATES ALL_SOURCES)

    set(OBJECT_LIB_NAME "${SD_LIBRARY_NAME}_object")
    add_library(${OBJECT_LIB_NAME} OBJECT ${ALL_SOURCES})

    add_dependencies(${OBJECT_LIB_NAME} flatbuffers_interface)

    if(DEFINED ENV{GENERATE_FLATC} OR DEFINED GENERATE_FLATC)
        add_dependencies(${OBJECT_LIB_NAME} generate_flatbuffers_headers)
    endif()

    target_include_directories(${OBJECT_LIB_NAME} PUBLIC ${EXTERNAL_INCLUDE_DIRS})
    set_property(TARGET ${OBJECT_LIB_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

    add_library(${SD_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_LIB_NAME}>)
    set_target_properties(${SD_LIBRARY_NAME} PROPERTIES OUTPUT_NAME ${SD_LIBRARY_NAME})
    set_property(TARGET ${SD_LIBRARY_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "${MSVC_RT_LIB}$<$<CONFIG:Debug>:Debug>")

    if(ANDROID)
        cmake_host_system_information(RESULT _logical_cores QUERY NUMBER_OF_LOGICAL_CORES)
        if(_logical_cores LESS 4)
            set_target_properties(${SD_LIBRARY_NAME} PROPERTIES JOB_POOL_COMPILE one_jobs)
        endif()
    endif()

    target_link_libraries(${SD_LIBRARY_NAME} PUBLIC
            ${ONEDNN}
            ${ARMCOMPUTE_LIBRARIES}
            ${OPENBLAS_LIBRARIES}
            ${BLAS_LIBRARIES}
            flatbuffers_interface
    )

    install(TARGETS ${SD_LIBRARY_NAME} DESTINATION .)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "You need at least GCC 4.9")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            message(STATUS "OpenMP found, linking OpenMP::OpenMP_CXX")
            target_link_libraries(${OBJECT_LIB_NAME} PUBLIC OpenMP::OpenMP_CXX)
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC OpenMP::OpenMP_CXX)
        else()
            message(WARNING "OpenMP not found, falling back to manual configuration")
            target_compile_options(${OBJECT_LIB_NAME} INTERFACE "-fopenmp")
            target_compile_options(${OBJECT_LIB_NAME} PRIVATE "-fopenmp")
            target_link_libraries(${SD_LIBRARY_NAME} PUBLIC "-fopenmp")
        endif()
    endif()
endfunction()


# This function creates the indexed type variables and their reverse lookups.
# It's called by build_indexed_type_lists.
function(create_indexed_type_variables_fixed prefix type_list)
    list(LENGTH type_list list_length)
    message(STATUS "Creating indexed variables for ${prefix} with ${list_length} types")

    set(index 0)
    foreach(type_name ${type_list})
        if(DEFINED TYPE_TUPLE_${type_name})
            set(${prefix}_${index} "${TYPE_TUPLE_${type_name}}" CACHE INTERNAL "Indexed type tuple")
        else()
            message(WARNING "  TYPE_TUPLE_${type_name} not defined for index ${index}!")
        endif()

        set(${prefix}_INDEX_${type_name} ${index} CACHE INTERNAL "Forward type index lookup")
        set(TYPE_NAME_${index} "${type_name}" CACHE INTERNAL "Reverse type index lookup")

        math(EXPR index "${index} + 1")
    endforeach()

    set(${prefix}_COUNT ${list_length} CACHE INTERNAL "Count of indexed types")
endfunction()
#-------------------------------------------------------------------------------
# Extracts type definitions from a C++ header file.
#
# This function reads the specified header file and looks for macros matching
# the TTYPE(ENUM, CTYPE, CMAKE_TYPE) pattern. For each match, it creates a
# CMake variable `TYPE_TUPLE_<CMAKE_TYPE>` containing the full definition,
# and it populates a result variable with the list of all found CMAKE_TYPEs.
#
# @param result_var The name of the output variable that will hold the list
#                   of extracted CMake type names (e.g., "float32", "int64").
#
function(extract_type_definitions_from_header result_var)
    # Try multiple possible locations for types.h
    set(POSSIBLE_HEADERS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/system/types.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/types.h"
    )

    set(TYPES_HEADER "")
    foreach(header_path ${POSSIBLE_HEADERS})
        if(EXISTS "${header_path}")
            set(TYPES_HEADER "${header_path}")
            break()
        endif()
    endforeach()

    if(NOT TYPES_HEADER)
        message(FATAL_ERROR "Could not find types.h! Searched in:\n- ${POSSIBLE_HEADERS}\n")
    endif()

    message(STATUS "Parsing type definitions from: ${TYPES_HEADER}")
    file(READ "${TYPES_HEADER}" TYPES_CONTENT)

    set(BASE_TYPE_NAMES
            "BFLOAT16" "BOOL" "DOUBLE" "FLOAT32" "HALF" "INT16" "INT32" "INT64"
            "INT8" "UINT16" "UINT32" "UINT64" "UINT8" "UTF16" "UTF32" "UTF8"
            "BFLOAT" "FLOAT" "LONG" "UNSIGNEDLONG" "INT"
    )

    set(extracted_types "")
    message(STATUS "Searching for the following base types: ${BASE_TYPE_NAMES}")

    foreach(type_name ${BASE_TYPE_NAMES})
        # FIX: Use a more flexible regex to tolerate formatting variations.
        # This allows for an optional comma and varying whitespace between the type name and the macro body.
        string(REGEX MATCH "#define[ \t]+TTYPE_${type_name}[ \t]*,?[ \t]*\\(([^)]+)\\)" type_match "${TYPES_CONTENT}")

        if(type_match)
            string(REGEX MATCH "\\(([^)]+)\\)" tuple_match "${type_match}")
            if(tuple_match)
                string(SUBSTRING "${tuple_match}" 1 -1 type_tuple)

                # Normalize type names to lowercase for consistency
                string(TOLOWER "${type_name}" normalized_name)

                # FIX: Corrected the elseif syntax for "int"
                # Handle aliases
                if(normalized_name STREQUAL "half")
                    set(normalized_name "float16")
                    set(TYPE_ALIAS_half "float16" PARENT_SCOPE)
                elseif(normalized_name STREQUAL "long")
                    set(normalized_name "int64")
                    set(TYPE_ALIAS_long "int64" PARENT_SCOPE)
                elseif(normalized_name STREQUAL "unsignedlong")
                    set(normalized_name "uint64")
                    set(TYPE_ALIAS_unsignedlong "uint64" PARENT_SCOPE)
                elseif(normalized_name STREQUAL "bfloat")
                    set(normalized_name "bfloat16")
                    set(TYPE_ALIAS_bfloat "bfloat16" PARENT_SCOPE)
                elseif(normalized_name STREQUAL "float")
                    set(normalized_name "float32")
                    set(TYPE_ALIAS_float "float32" PARENT_SCOPE)
                elseif(normalized_name STREQUAL "int")
                    set(normalized_name "int32")
                    set(TYPE_ALIAS_int "int32" PARENT_SCOPE)
                endif()


                set(TYPE_TUPLE_${normalized_name} "(${type_tuple})" PARENT_SCOPE)
                list(APPEND extracted_types "${normalized_name}")
                message(STATUS "  [OK] Found TTYPE_${type_name} as '${normalized_name}'")
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES extracted_types)
    list(LENGTH extracted_types type_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "No TTYPE definitions found in ${TYPES_HEADER}! The regex pattern might need adjustment for your file's format.")
    else()
        message(STATUS "Successfully extracted ${type_count} base type definitions.")
    endif()

    set(${result_var} "${extracted_types}" PARENT_SCOPE)
endfunction()

function(build_indexed_type_lists)
    extract_type_definitions_from_header(AVAILABLE_TYPES)

    if(NOT AVAILABLE_TYPES)
        message(FATAL_ERROR "No types extracted from types.h!")
    endif()

    if(SD_TYPES_LIST_COUNT GREATER 0)
        set(ACTIVE_TYPES "")
        foreach(user_type ${SD_TYPES_LIST})
            normalize_type("${user_type}" normalized_type)
            if(normalized_type IN_LIST AVAILABLE_TYPES)
                list(APPEND ACTIVE_TYPES "${normalized_type}")
            endif()
        endforeach()
    else()
        set(ACTIVE_TYPES "${AVAILABLE_TYPES}")
    endif()

    if(NOT ACTIVE_TYPES)
        message(FATAL_ERROR "No active types selected!")
    endif()

    # Set as a global cached variable
    set(SD_ACTIVE_TYPES "${ACTIVE_TYPES}" CACHE INTERNAL "List of active types for the build")

    set(COMMON_TYPES_LIST "")
    set(FLOAT_TYPES_LIST "")
    set(INTEGER_TYPES_LIST "")

    foreach(type_name ${ACTIVE_TYPES})
        if(type_name MATCHES "(float|double|half|bfloat)")
            list(APPEND FLOAT_TYPES_LIST "${type_name}")
        elseif(type_name MATCHES "(int|uint|long)")
            list(APPEND INTEGER_TYPES_LIST "${type_name}")
        endif()
        list(APPEND COMMON_TYPES_LIST "${type_name}")
    endforeach()

    create_indexed_type_variables_fixed("SD_COMMON_TYPES" "${COMMON_TYPES_LIST}")
    create_indexed_type_variables_fixed("SD_FLOAT_TYPES" "${FLOAT_TYPES_LIST}")
    create_indexed_type_variables_fixed("SD_INTEGER_TYPES" "${INTEGER_TYPES_LIST}")

    list(LENGTH COMMON_TYPES_LIST common_count)
    list(LENGTH FLOAT_TYPES_LIST float_count)
    list(LENGTH INTEGER_TYPES_LIST integer_count)

    # Set counts as global cached variables
    set(SD_COMMON_TYPES_COUNT ${common_count} CACHE INTERNAL "Total count of common types")
    set(SD_FLOAT_TYPES_COUNT ${float_count} CACHE INTERNAL "Total count of float types")
    set(SD_INTEGER_TYPES_COUNT ${integer_count} CACHE INTERNAL "Total count of integer types")

    message(STATUS "Counts - Common: ${common_count}, Float: ${float_count}, Integer: ${integer_count}")
endfunction()
# Include the type validation and engine modules from the source tree
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TypeValidation.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TypeValidation.cmake)
else()
    message(WARNING "TypeValidation.cmake not found - using basic validation")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TypeCombinationEngine.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/TypeProfiles.cmake)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/loops")

# --- Type System Options ---
option(SD_STRICT_TYPE_VALIDATION "Enable strict type validation" OFF)
option(SD_DEBUG_AUTO_REDUCE "Auto-reduce types for debug builds" ON)
option(SD_SHOW_TYPE_IMPACT "Show estimated build impact" ON)
set(SD_DEBUG_TYPE_PROFILE "" CACHE STRING "Debug type profile (MINIMAL_INDEXING, ESSENTIAL, FLOATS_ONLY, QUANTIZATION, MIXED_PRECISION, NLP, etc.)")
set(SD_DEBUG_CUSTOM_TYPES "" CACHE STRING "Custom types for debug profile")

# --- Type Definitions ---
set(TYPE_ALIAS_float "float32")
set(TYPE_ALIAS_half "float16")
set(TYPE_ALIAS_long "int64")
set(TYPE_ALIAS_unsignedlong "uint64")
set(TYPE_ALIAS_int "int32")
set(TYPE_ALIAS_bfloat "bfloat16")
set(TYPE_ALIAS_float64 "double")

set(ALL_SUPPORTED_TYPES
        "bool" "int8" "uint8" "int16" "uint16" "int32" "uint32"
        "int64" "uint64" "float16" "bfloat16" "float32" "double"
        "float" "half" "long" "unsignedlong" "int" "bfloat" "float64"
        "utf8" "utf16" "utf32"
)
set(MINIMUM_REQUIRED_TYPES "int32" "int64" "float32")

# --- Debug Profiles ---
set(DEBUG_PROFILE_MINIMAL_INDEXING "float32;double;int32;int64")
set(DEBUG_PROFILE_ESSENTIAL "float32;double;int32;int64;int8;int16")
set(DEBUG_PROFILE_FLOATS_ONLY "float32;double;float16")
set(DEBUG_PROFILE_INTEGERS_ONLY "int8;int16;int32;int64;uint8;uint16;uint32;uint64")
set(DEBUG_PROFILE_SINGLE_PRECISION "float32;int32;int64")
set(DEBUG_PROFILE_DOUBLE_PRECISION "double;int32;int64")
set(DEBUG_PROFILE_QUANTIZATION "int8;uint8;float32;int32;int64")
set(DEBUG_PROFILE_MIXED_PRECISION "float16;bfloat16;float32;int32;int64")
set(DEBUG_PROFILE_NLP "std::string;float32;int32;int64")

# --- Validation and Processing ---
validate_and_process_types_failfast()

# --- Generate include_ops.h ---
set(DEFINITIONS_CONTENT "")
# Define ops
if(SD_ALL_OPS OR "${SD_OPS_LIST}" STREQUAL "")
    message("Adding all ops due to empty op list or SD_ALL_OPS definition: SD_ALL_OPS=${SD_ALL_OPS} SD_OPS_LIST=${SD_OPS_LIST}")
    add_compile_definitions(SD_ALL_OPS=1)
    string(APPEND DEFINITIONS_CONTENT "#define SD_ALL_OPS 1\n")
else()
    message("_OPS: ${SD_OPS_LIST}")
    foreach(OP ${SD_OPS_LIST})
        add_compile_definitions(OP_${OP}=1)
        message(STATUS "OP: ${OP}")
        string(APPEND DEFINITIONS_CONTENT "#define OP_${OP} 1\n")
    endforeach()
endif()

# Define types
list(LENGTH SD_TYPES_LIST SD_TYPES_LIST_COUNT)
set(USER_REQUESTED_SELECTIVE FALSE)
if(DEFINED SD_TYPES_LIST_EXPLICIT OR
(SD_DEBUG_TYPE_PROFILE AND NOT SD_DEBUG_TYPE_PROFILE STREQUAL "") OR
        SD_STRICT_TYPE_VALIDATION)
    set(USER_REQUESTED_SELECTIVE TRUE)
endif()

if(SD_TYPES_LIST_COUNT GREATER 0 AND USER_REQUESTED_SELECTIVE)
    add_compile_definitions(SD_SELECTIVE_TYPES)
    string(APPEND DEFINITIONS_CONTENT "#define SD_SELECTIVE_TYPES\n")
    print_status_colored("SUCCESS" "=== BUILDING WITH SELECTED TYPES ===")
    set(TYPE_DISPLAY_LIST "")
    foreach(SD_TYPE ${SD_TYPES_LIST})
        normalize_type("${SD_TYPE}" normalized_type)
        string(TOUPPER ${normalized_type} SD_TYPE_UPPERCASE)
        add_compile_definitions(HAS_${SD_TYPE_UPPERCASE})
        message(STATUS "✅ TYPE: ${normalized_type}")
        string(APPEND DEFINITIONS_CONTENT "#define HAS_${SD_TYPE_UPPERCASE}\n")
        list(APPEND TYPE_DISPLAY_LIST "${normalized_type}")
    endforeach()
    list(LENGTH TYPE_DISPLAY_LIST TYPE_COUNT)
    string(REPLACE ";" ", " TYPE_DISPLAY_STRING "${TYPE_DISPLAY_LIST}")
    print_status_colored("SUCCESS" "Building with ${TYPE_COUNT} data types: ${TYPE_DISPLAY_STRING}")
else()
    print_status_colored("INFO" "=== BUILDING WITH ALL TYPES ===")
    if(SD_TYPES_LIST_COUNT GREATER 0 AND NOT USER_REQUESTED_SELECTIVE)
        print_status_colored("INFO" "SD_TYPES_LIST was populated internally but selective types not requested")
        print_status_colored("INFO" "Building with ALL types (ignoring internal type list)")
    endif()
    message(STATUS "Building with all supported data types (SD_SELECTIVE_TYPES disabled)")
endif()

# Write the definitions file
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/generated")
set(INCLUDE_OPS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/generated/include_ops.h")
message("Generating include_ops.h at: ${INCLUDE_OPS_FILE}")
file(WRITE "${INCLUDE_OPS_FILE}" "#ifndef SD_DEFINITIONS_GEN_H_\n#define SD_DEFINITIONS_GEN_H_\n${DEFINITIONS_CONTENT}\n#endif\n")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/generated")