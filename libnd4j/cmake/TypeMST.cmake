# TypeMST.cmake - Simplified approach for selective type combinations

# Option to enable MST-based type selection
option(SD_USE_MST_TYPES "Use optimized type combinations" ON)

# Function to generate a predefined set of type combinations
# This uses a simplified approach to avoid regex issues
function(generate_mst_combinations)
    message(STATUS "TypeMST: Generating optimized type combinations")

    # Get the list of enabled types from SD_TYPES_LIST
    set(_ENABLED_TYPES)

    if(DEFINED SD_TYPES_LIST)
        # Use the selective types list if available
        foreach(TYPE ${SD_TYPES_LIST})
            string(TOUPPER ${TYPE} TYPE_UPPER)
            list(APPEND _ENABLED_TYPES ${TYPE_UPPER})
        endforeach()
        message(STATUS "TypeMST: Using types from SD_TYPES_LIST: ${_ENABLED_TYPES}")
    else()
        # Default types if not using selective types
        set(_ENABLED_TYPES "FLOAT32" "DOUBLE" "INT32" "INT64")
        message(STATUS "TypeMST: Using default types: ${_ENABLED_TYPES}")
    endif()

    # Map from type name to index
    set(TYPE_INDEX_BOOL 12)
    set(TYPE_INDEX_INT8 6)
    set(TYPE_INDEX_UINT8 8)
    set(TYPE_INDEX_INT16 7)
    set(TYPE_INDEX_UINT16 9)
    set(TYPE_INDEX_INT32 4)
    set(TYPE_INDEX_UINT32 10)
    set(TYPE_INDEX_INT64 5)
    set(TYPE_INDEX_UINT64 11)
    set(TYPE_INDEX_FLOAT16 3)
    set(TYPE_INDEX_BFLOAT16 0)
    set(TYPE_INDEX_FLOAT32 1)
    set(TYPE_INDEX_DOUBLE 2)

    # Generate homogeneous combinations only for enabled types
    set(_COMBINATIONS)
    foreach(TYPE ${_ENABLED_TYPES})
        if(DEFINED TYPE_INDEX_${TYPE})
            list(APPEND _COMBINATIONS "${TYPE_INDEX_${TYPE}},${TYPE_INDEX_${TYPE}},${TYPE_INDEX_${TYPE}}")
        endif()
    endforeach()

    # Add essential mixed-type combinations only if both types exist
    # Check if float and int32 are both enabled
    if(";${_ENABLED_TYPES};" MATCHES ";FLOAT32;" AND ";${_ENABLED_TYPES};" MATCHES ";INT32;")
        list(APPEND _COMBINATIONS "${TYPE_INDEX_FLOAT32},${TYPE_INDEX_INT32},${TYPE_INDEX_FLOAT32}")
        list(APPEND _COMBINATIONS "${TYPE_INDEX_INT32},${TYPE_INDEX_FLOAT32},${TYPE_INDEX_FLOAT32}")
    endif()

    # Check if double and int32 are both enabled
    if(";${_ENABLED_TYPES};" MATCHES ";DOUBLE;" AND ";${_ENABLED_TYPES};" MATCHES ";INT32;")
        list(APPEND _COMBINATIONS "${TYPE_INDEX_DOUBLE},${TYPE_INDEX_INT32},${TYPE_INDEX_DOUBLE}")
        list(APPEND _COMBINATIONS "${TYPE_INDEX_INT32},${TYPE_INDEX_DOUBLE},${TYPE_INDEX_DOUBLE}")
    endif()

    # You can add more combinations conditionally based on enabled types

    # If no combinations were added (should not happen), add at least float-float-float
    if("${_COMBINATIONS}" STREQUAL "")
        list(APPEND _COMBINATIONS "1,1,1")  # fall back to float at least
    endif()

    # Remove duplicates
    list(REMOVE_DUPLICATES _COMBINATIONS)

    # Set result
    set(COMBINATIONS_3 ${_COMBINATIONS} PARENT_SCOPE)

    # Report
    list(LENGTH _COMBINATIONS COMBO_COUNT)
    list(LENGTH _ENABLED_TYPES TYPE_COUNT)
    math(EXPR TOTAL_POSSIBLE "${TYPE_COUNT}*${TYPE_COUNT}*${TYPE_COUNT}")
    message(STATUS "TypeMST: Generated ${COMBO_COUNT} type combinations from ${TYPE_COUNT} types (reduced from ${TOTAL_POSSIBLE})")
endfunction()