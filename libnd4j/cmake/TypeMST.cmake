# TypeMST.cmake - Zero-indexed type combinations

# Option to enable MST-based type selection
option(SD_USE_MST_TYPES "Use optimized type combinations" ON)

# Function to generate a set of type combinations with 0-based indexing
function(generate_mst_combinations)
    message(STATUS "TypeMST: Generating zero-indexed type combinations")

    # Using 0-based indices that match the macros
    set(_COMBINATIONS)

    # Add homogeneous combinations
    list(APPEND _COMBINATIONS "0,0,0")  # BFLOAT16
    list(APPEND _COMBINATIONS "1,1,1")  # FLOAT32
    list(APPEND _COMBINATIONS "2,2,2")  # DOUBLE
    list(APPEND _COMBINATIONS "3,3,3")  # FLOAT16
    list(APPEND _COMBINATIONS "4,4,4")  # INT32
    list(APPEND _COMBINATIONS "5,5,5")  # INT64
    list(APPEND _COMBINATIONS "6,6,6")  # INT8
    list(APPEND _COMBINATIONS "7,7,7")  # INT16
    list(APPEND _COMBINATIONS "8,8,8")  # UINT8
    list(APPEND _COMBINATIONS "9,9,9")  # UINT16

    # Add critical mixed-type combinations
    list(APPEND _COMBINATIONS "1,4,1")  # float32, int32, float32
    list(APPEND _COMBINATIONS "4,1,1")  # int32, float32, float32
    list(APPEND _COMBINATIONS "2,4,2")  # double, int32, double
    list(APPEND _COMBINATIONS "4,2,2")  # int32, double, double

    # Set result
    set(COMBINATIONS_3 ${_COMBINATIONS} PARENT_SCOPE)

    # Report
    list(LENGTH _COMBINATIONS COMBO_COUNT)
    message(STATUS "TypeMST: Generated ${COMBO_COUNT} zero-indexed type combinations")
endfunction()