# CombinationCaching.cmake
# Smart type combination reuse and caching system

# Main function to handle combination reuse
function(use_cached_type_combinations)
    # Check if combinations already exist in memory
    if(DEFINED COMBINATIONS_2 AND DEFINED COMBINATIONS_3)
        list(LENGTH COMBINATIONS_2 combo2_count)
        list(LENGTH COMBINATIONS_3 combo3_count)
        if(combo2_count GREATER 0 AND combo3_count GREATER 0)
            message(STATUS "♻️ Reusing existing combinations: 2-type=${combo2_count}, 3-type=${combo3_count}")
            return()
        endif()
    endif()

    # Check for cached combinations file
    set(COMBO_CACHE_FILE "${CMAKE_BINARY_DIR}/type_combinations.cache")
    if(EXISTS "${COMBO_CACHE_FILE}")
        message(STATUS "📁 Loading combinations from cache file...")
        include("${COMBO_CACHE_FILE}")

        # Verify cache loaded successfully
        if(DEFINED COMBINATIONS_2 AND DEFINED COMBINATIONS_3)
            list(LENGTH COMBINATIONS_2 combo2_count)
            list(LENGTH COMBINATIONS_3 combo3_count)
            message(STATUS "✅ Loaded from cache: 2-type=${combo2_count}, 3-type=${combo3_count}")
            return()
        else()
            message(STATUS "⚠️ Cache file corrupted, regenerating...")
        endif()
    endif()

    # Only generate if we absolutely have to
    message(STATUS "🔄 Generating type combinations (first time only)...")

    # Ensure type system is initialized
    if(NOT DEFINED SD_COMMON_TYPES_COUNT OR SD_COMMON_TYPES_COUNT EQUAL 0)
        validate_and_process_types_failfast()
        build_indexed_type_lists()
    endif()

    initialize_dynamic_combinations()

    # Verify combinations were created
    if(DEFINED COMBINATIONS_2 AND DEFINED COMBINATIONS_3)
        list(LENGTH COMBINATIONS_2 combo2_count)
        list(LENGTH COMBINATIONS_3 combo3_count)
        message(STATUS "✅ Generated combinations: 2-type=${combo2_count}, 3-type=${combo3_count}")

        # Cache for future use
        save_combinations_to_cache("${COMBO_CACHE_FILE}")
    else()
        message(FATAL_ERROR "❌ Failed to generate type combinations!")
    endif()
endfunction()

# Save combinations to cache file
function(save_combinations_to_cache cache_file)
    message(STATUS "💾 Saving combinations to cache...")

    set(cache_content "# Generated type combinations cache\n")
    string(APPEND cache_content "# Generated on: ${CMAKE_CURRENT_TIMESTAMP}\n\n")

    # Save 2-type combinations
    string(APPEND cache_content "set(COMBINATIONS_2 \"")
    foreach(combo ${COMBINATIONS_2})
        string(APPEND cache_content "${combo};")
    endforeach()
    string(APPEND cache_content "\" CACHE INTERNAL \"2-type combinations\")\n\n")

    # Save 3-type combinations
    string(APPEND cache_content "set(COMBINATIONS_3 \"")
    foreach(combo ${COMBINATIONS_3})
        string(APPEND cache_content "${combo};")
    endforeach()
    string(APPEND cache_content "\" CACHE INTERNAL \"3-type combinations\")\n\n")

    # Save metadata
    string(APPEND cache_content "set(COMBINATIONS_CACHE_TIMESTAMP \"${CMAKE_CURRENT_TIMESTAMP}\" CACHE INTERNAL \"Cache timestamp\")\n")
    string(APPEND cache_content "set(COMBINATIONS_CACHE_VALID TRUE CACHE INTERNAL \"Cache validity flag\")\n")

    file(WRITE "${cache_file}" "${cache_content}")
    message(STATUS "💾 Combinations cached to: ${cache_file}")
endfunction()

# Generate reusable combination strings once
function(generate_combination_strings_once)
    if(DEFINED GLOBAL_COMB2_STRING AND DEFINED GLOBAL_COMB3_STRING)
        message(STATUS "♻️ Reusing existing combination strings")
        return()
    endif()

    message(STATUS "🔧 Generating combination strings...")

    # Generate 2-type combination string
    set(COMB2_STRING "")
    foreach(combination ${COMBINATIONS_2})
        string(REPLACE "," ";" combo_parts "${combination}")
        list(GET combo_parts 0 t1)
        list(GET combo_parts 1 t2)
        string(APPEND COMB2_STRING "INSTANTIATE_2(${t1}, ${t2})")
    endforeach()

    # Generate 3-type combination string
    set(COMB3_STRING "")
    foreach(combination ${COMBINATIONS_3})
        string(REPLACE "," ";" combo_parts "${combination}")
        list(GET combo_parts 0 t1)
        list(GET combo_parts 1 t2)
        list(GET combo_parts 2 t3)
        string(APPEND COMB3_STRING "INSTANTIATE_3(${t1}, ${t2}, ${t3}); ")
    endforeach()

    # Cache globally for reuse
    set(GLOBAL_COMB2_STRING "${COMB2_STRING}" CACHE INTERNAL "Global 2-type combination string")
    set(GLOBAL_COMB3_STRING "${COMB3_STRING}" CACHE INTERNAL "Global 3-type combination string")

    list(LENGTH COMBINATIONS_2 combo2_count)
    list(LENGTH COMBINATIONS_3 combo3_count)
    message(STATUS "✅ Generated strings: 2-type=${combo2_count} instantiations, 3-type=${combo3_count} instantiations")
endfunction()

# Clear cache if needed (for development)
function(clear_combination_cache)
    set(COMBO_CACHE_FILE "${CMAKE_BINARY_DIR}/type_combinations.cache")
    if(EXISTS "${COMBO_CACHE_FILE}")
        file(REMOVE "${COMBO_CACHE_FILE}")
        message(STATUS "🗑️ Cleared combination cache")
    endif()

    # Clear memory cache
    unset(COMBINATIONS_2 CACHE)
    unset(COMBINATIONS_3 CACHE)
    unset(GLOBAL_COMB2_STRING CACHE)
    unset(GLOBAL_COMB3_STRING CACHE)
    unset(COMBINATIONS_CACHE_VALID CACHE)
endfunction()

# Validate cache integrity
function(validate_combination_cache)
    if(NOT DEFINED COMBINATIONS_CACHE_VALID OR NOT COMBINATIONS_CACHE_VALID)
        message(STATUS "⚠️ Combination cache is invalid, will regenerate")
        clear_combination_cache()
        return()
    endif()

    # Check if combinations are reasonable
    if(DEFINED COMBINATIONS_2 AND DEFINED COMBINATIONS_3)
        list(LENGTH COMBINATIONS_2 combo2_count)
        list(LENGTH COMBINATIONS_3 combo3_count)

        if(combo2_count LESS 10 OR combo3_count LESS 10)
            message(STATUS "⚠️ Combination counts seem too low, will regenerate")
            clear_combination_cache()
            return()
        endif()

        message(STATUS "✅ Cache validation passed")
    else()
        message(STATUS "⚠️ Combinations not found in cache, will regenerate")
        clear_combination_cache()
    endif()
endfunction()

# Main setup function
function(setup_smart_combination_reuse)
    message(STATUS "=== SETTING UP SMART COMBINATION REUSE ===")

    # Validate existing cache
    validate_combination_cache()

    # Use cached combinations without regeneration
    use_cached_type_combinations()

    # Create combination strings once, reuse everywhere
    generate_combination_strings_once()

    message(STATUS "✅ Smart combination reuse setup complete")
endfunction()