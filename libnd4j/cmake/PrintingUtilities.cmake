################################################################################
# Colored Status Printing and Utility Functions
# Common utility functions for enhanced CMake output and debugging
################################################################################

# Enhanced colored status printing function
function(print_status_colored type message)
    if(type STREQUAL "ERROR")
        message(FATAL_ERROR "❌ ${message}")
    elseif(type STREQUAL "WARNING")
        message(WARNING "⚠️  ${message}")
    elseif(type STREQUAL "SUCCESS")
        message(STATUS "✅ ${message}")
    elseif(type STREQUAL "INFO")
        message(STATUS "ℹ️  ${message}")
    elseif(type STREQUAL "DEBUG")
        message(STATUS "🔍 ${message}")
    elseif(type STREQUAL "NOTICE")
        message(NOTICE "📢 ${message}")
    else()
        message(STATUS "${message}")
    endif()
endfunction()

# Print all CMake variables for debugging
macro(print_all_variables)
    message(STATUS "print_all_variables------------------------------------------{")
    get_cmake_property(_variableNames VARIABLES)
    foreach(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    endforeach()
    message(STATUS "print_all_variables------------------------------------------}")
endmacro()

# Function to verify template processing worked
function(verify_template_processing GENERATED_FILE)
    if(EXISTS "${GENERATED_FILE}")
        file(READ "${GENERATED_FILE}" FILE_CONTENT)

        # Check for unprocessed #cmakedefine
        if(FILE_CONTENT MATCHES "#cmakedefine[ \t]+[A-Za-z_]+")
            message(FATAL_ERROR "❌ Template processing FAILED: ${GENERATED_FILE} contains unprocessed #cmakedefine directives")
        endif()

        # Check for unprocessed @VAR@
        if(FILE_CONTENT MATCHES "@[A-Za-z_]+@")
            message(FATAL_ERROR "❌ Template processing FAILED: ${GENERATED_FILE} contains unprocessed @VAR@ tokens")
        endif()

        message(STATUS "✅ Template processing verified: ${GENERATED_FILE}")
        return()
    else()
        message(FATAL_ERROR "❌ Generated file does not exist: ${GENERATED_FILE}")
    endif()
endfunction()

# Function to safely remove files if they match exclusion criteria
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
