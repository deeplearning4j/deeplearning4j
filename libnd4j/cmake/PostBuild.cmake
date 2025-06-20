# cmake/PostBuild.cmake
# Configures optional post-build targets like tests and analysis tools.

# --- Configuration file ---
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Test Suite Configuration ---
if(SD_BUILD_TESTS)
    include(CTest)
    set(SD_ALL_OPS true)
    enable_testing()
    add_subdirectory(tests_cpu)
endif()

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

# --- Preprocessing Target ---
if(SD_PREPROCESS STREQUAL "ON")
    message("Preprocessing enabled: ${CMAKE_BINARY_DIR}")
    include_directories(${CMAKE_BINARY_DIR}/.././include)

    get_target_property(FINAL_ALL_SOURCES ${OBJECT_LIB_NAME} SOURCES)
    list(REMOVE_DUPLICATES FINAL_ALL_SOURCES)

    set(PREPROCESSED_DIR "${CMAKE_SOURCE_DIR}/preprocessed")
    file(MAKE_DIRECTORY ${PREPROCESSED_DIR})

    set(PREPROCESSED_FILES)
    set(PROCESSED_SOURCES "")
    foreach(src IN LISTS FINAL_ALL_SOURCES)
        if(NOT EXISTS ${src} OR NOT src MATCHES "\\.(c|cpp|cxx|cc|cu)$")
            continue()
        endif()

        if(NOT src IN_LIST PROCESSED_SOURCES)
            get_filename_component(src_name ${src} NAME_WE)
            get_filename_component(src_path ${src} PATH)
            file(RELATIVE_PATH rel_path ${CMAKE_SOURCE_DIR} ${src_path})
            string(REPLACE "/" "_" src_dir_ "${rel_path}")
            set(preprocessed_file "${PREPROCESSED_DIR}/${src_dir_}_${src_name}.i")

            message(STATUS "Processing ${src} to ${preprocessed_file}")

            if(NOT EXISTS "${preprocessed_file}")
                set(compiler "")
                set(lang_flags "")
                get_target_property(includes_list ${OBJECT_LIB_NAME} INCLUDE_DIRECTORIES)
                get_target_property(compile_defs ${OBJECT_LIB_NAME} COMPILE_DEFINITIONS)
                set(include_flags "")
                foreach(dir IN LISTS includes_list)
                    string(APPEND include_flags " -I\"${dir}\"")
                endforeach()
                set(defs_flags "")
                foreach(def IN LISTS compile_defs)
                    string(APPEND defs_flags " -D${def}")
                endforeach()

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

                execute_process(
                        COMMAND ${CMAKE_COMMAND} -E time "${compiler}" -E ${lang_flags} ${defs_flags} ${include_flags} "${src}" -o "${preprocessed_file}"
                        RESULT_VARIABLE result
                )
                if(result)
                    message(WARNING "Preprocessing failed for ${src}.")
                else()
                    list(APPEND PREPROCESSED_FILES ${preprocessed_file})
                endif()
            else()
                list(APPEND PREPROCESSED_FILES ${preprocessed_file})
            endif()
            list(APPEND PROCESSED_SOURCES ${src})
        endif()
    endforeach()
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