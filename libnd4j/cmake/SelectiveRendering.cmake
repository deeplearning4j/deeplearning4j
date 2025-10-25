include(SelectiveRenderingCore)

# Wrapper for the original setup_selective_rendering function
function(setup_selective_rendering)
    message(STATUS "🔄 SelectiveRendering.cmake: Calling unified core system")

    # Call the unified system
    setup_selective_rendering_unified_safe()

    # Map results to legacy variables that existing code expects
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()

    message(STATUS "✅ SelectiveRendering.cmake: Setup complete via unified core")
endfunction()

# Legacy wrapper functions for other entry points
function(track_combination_states active_types combinations_3)
    # This function is now handled internally by the core system
    message(STATUS "🔄 track_combination_states: Handled by unified core")
endfunction()

function(generate_selective_rendering_header)
    # This function is now handled internally by the core system  
    message(STATUS "🔄 generate_selective_rendering_header: Handled by unified core")
endfunction()

function(generate_selective_wrapper_header)
    # This function is now handled internally by the core system
    message(STATUS "🔄 generate_selective_wrapper_header: Handled by unified core")
endfunction()