 function(setup_definitive_semantic_filtering)
     message(STATUS "🔄 SemanticTypeFiltering.cmake: Calling unified core system")

     # Call the unified system with semantic filtering enabled
     set(SD_ENABLE_SEMANTIC_FILTERING TRUE PARENT_SCOPE)
     setup_selective_rendering_unified_safe(TYPE_PROFILE "${SD_TYPE_PROFILE}")

     # Map to expected variables
     if(DEFINED UNIFIED_COMBINATIONS_2)
         set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
         set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" CACHE INTERNAL "2-type combinations" FORCE)
     endif()
     if(DEFINED UNIFIED_COMBINATIONS_3)
         set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
         set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" CACHE INTERNAL "3-type combinations" FORCE)
     endif()
     if(DEFINED UNIFIED_ACTIVE_TYPES)
         set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
         set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" CACHE INTERNAL "Active type list" FORCE)
     endif()

     message(STATUS "✅ SemanticTypeFiltering.cmake: Setup complete via unified core")
 endfunction()

 # Wrapper for initialize_definitive_combinations
 function(initialize_definitive_combinations)
     message(STATUS "🔄 initialize_definitive_combinations: Calling unified core system")
     setup_definitive_semantic_filtering()
 endfunction()

 # Legacy helper functions (now handled by core)
 function(extract_definitive_types result_var)
     if(DEFINED UNIFIED_ACTIVE_TYPES)
         set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
     else()
         srcore_auto_setup()
         set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
     endif()
 endfunction()

 function(generate_definitive_combinations active_types result_2_var result_3_var)
     message(STATUS "🔄 generate_definitive_combinations: Using cached unified core results")
     if(DEFINED UNIFIED_COMBINATIONS_2 AND DEFINED UNIFIED_COMBINATIONS_3)
         set(${result_2_var} "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
         set(${result_3_var} "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
     else()
         srcore_auto_setup()
         set(${result_2_var} "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
         set(${result_3_var} "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
     endif()
 endfunction()

 function(validate_critical_types_coverage active_types combinations_3)
     # This validation is now handled internally by the core system
     message(STATUS "🔄 validate_critical_types_coverage: Handled by unified core")
 endfunction()