function(setup_definitive_semantic_filtering_with_selective_rendering)
    message(STATUS "🔄 SelectiveRenderingIntegration.cmake: Calling unified core system")

    # Enable both semantic filtering and selective rendering
    set(SD_ENABLE_SEMANTIC_FILTERING TRUE PARENT_SCOPE)
    set(SD_ENABLE_SELECTIVE_RENDERING TRUE PARENT_SCOPE)

    # Call unified system
    setup_selective_rendering_unified_safe()

    # Legacy variable mapping
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()

    message(STATUS "✅ SelectiveRenderingIntegration.cmake: Setup complete via unified core")
endfunction()

function(enhanced_semantic_filtering_setup)
    setup_definitive_semantic_filtering_with_selective_rendering()
endfunction()

# Override the main setup function
macro(setup_definitive_semantic_filtering)
    enhanced_semantic_filtering_setup()
endmacro()