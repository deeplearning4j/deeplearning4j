function(build_indexed_type_lists)
    message(STATUS "🔄 TypeSystem.cmake: build_indexed_type_lists calling unified core")

    # Ensure unified system is set up
    srcore_auto_setup()

    # Map unified results to legacy TypeSystem variables
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(SD_ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" CACHE INTERNAL "List of active types for the build")

        list(LENGTH UNIFIED_ACTIVE_TYPES type_count)
        set(SD_COMMON_TYPES_COUNT ${type_count} CACHE INTERNAL "Total count of common types")

        # Create TYPE_NAME_X mappings for legacy compatibility
        set(index 0)
        foreach(type_name ${UNIFIED_ACTIVE_TYPES})
            set(TYPE_NAME_${index} "${type_name}" CACHE INTERNAL "Legacy reverse type lookup")
            set(TYPE_NAME_${index} "${type_name}" PARENT_SCOPE)
            math(EXPR index "${index} + 1")
        endforeach()

        # Also set float/integer type counts for compatibility
        set(float_count 0)
        set(integer_count 0)
        foreach(type_name ${UNIFIED_ACTIVE_TYPES})
            if(type_name MATCHES "(float|double|half|bfloat)")
                math(EXPR float_count "${float_count} + 1")
            elseif(type_name MATCHES "(int|uint|long)")
                math(EXPR integer_count "${integer_count} + 1")
            endif()
        endforeach()

        set(SD_FLOAT_TYPES_COUNT ${float_count} CACHE INTERNAL "Total count of float types")
        set(SD_INTEGER_TYPES_COUNT ${integer_count} CACHE INTERNAL "Total count of integer types")
    endif()

    message(STATUS "✅ TypeSystem.cmake: build_indexed_type_lists complete via unified core")
endfunction()

# Ensure the critical get_type_name_from_index function works
function(get_type_name_from_index index result_var)
    # First try the unified system
    if(DEFINED SRCORE_TYPE_NAME_${index})
        set(${result_var} "${SRCORE_TYPE_NAME_${index}}" PARENT_SCOPE)
        return()
    endif()

    # Then try legacy system
    if(DEFINED TYPE_NAME_${index})
        set(${result_var} "${TYPE_NAME_${index}}" PARENT_SCOPE)
        return()
    endif()

    # Auto-setup if needed
    if(NOT DEFINED SRCORE_SETUP_COMPLETE)
        message(STATUS "🔄 get_type_name_from_index: Auto-setting up unified core")
        srcore_auto_setup()
        if(DEFINED SRCORE_TYPE_NAME_${index})
            set(${result_var} "${SRCORE_TYPE_NAME_${index}}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # Final fallback
    message(WARNING "❌ get_type_name_from_index(${index}): No type found in any system")
    set(${result_var} "unknown_type_${index}" PARENT_SCOPE)
endfunction()

# Wrapper for extract_type_definitions_from_header (used by legacy code)
function(extract_type_definitions_from_header result_var)
    message(STATUS "🔄 extract_type_definitions_from_header: Using unified core result")

    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    else()
        srcore_auto_setup()
        set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()
endfunction()