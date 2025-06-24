# Wrapper for initialize_dynamic_combinations
function(initialize_dynamic_combinations)
    message(STATUS "🔄 TypeCombinationEngine.cmake: Calling unified core system")

    # Call the unified system
    srcore_auto_setup()

    # Map results to the expected legacy variables
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" CACHE INTERNAL "2-type combinations" FORCE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" CACHE INTERNAL "3-type combinations" FORCE)
    endif()

    message(STATUS "✅ TypeCombinationEngine.cmake: Setup complete via unified core")
endfunction()

# Legacy utility functions (now thin wrappers)
function(normalize_type input_type output_var)
    srcore_normalize_type("${input_type}" result)
    set(${output_var} "${result}" PARENT_SCOPE)
endfunction()

function(get_type_name_from_index index result_var)
    # Try the unified core first
    if(DEFINED SRCORE_TYPE_NAME_${index})
        set(${result_var} "${SRCORE_TYPE_NAME_${index}}" PARENT_SCOPE)
        return()
    endif()

    # Fallback to legacy system
    if(DEFINED TYPE_NAME_${index})
        set(${result_var} "${TYPE_NAME_${index}}" PARENT_SCOPE)
        return()
    endif()

    # Emergency setup if nothing is available
    srcore_auto_setup()
    if(DEFINED SRCORE_TYPE_NAME_${index})
        set(${result_var} "${SRCORE_TYPE_NAME_${index}}" PARENT_SCOPE)
    else()
        set(${result_var} "unknown_type_${index}" PARENT_SCOPE)
    endif()
endfunction()