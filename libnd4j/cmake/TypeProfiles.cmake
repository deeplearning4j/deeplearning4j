# TypeProfiles.cmake
# Predefined ML workload configurations for type combination optimization

# Type profiles for different ML workloads
set(QUANTIZATION_TYPES "int8_t;uint8_t;float;double" CACHE INTERNAL "Types for quantization workloads")
set(TRAINING_TYPES "float16;bfloat16;float;double" CACHE INTERNAL "Types for training workloads")
set(INFERENCE_TYPES "int8_t;uint8_t;float16;bfloat16;float" CACHE INTERNAL "Types for inference workloads")
set(NLP_TYPES "std::string;std::u16string;int32_t;int64_t;float;double" CACHE INTERNAL "Types for NLP workloads")
set(CV_TYPES "uint8_t;int8_t;float16;float" CACHE INTERNAL "Types for computer vision workloads")

# Set quantization type profile - optimized for INT8/UINT8 inference
function(set_quantization_type_profile)
    set(SD_SELECTED_TYPES ${QUANTIZATION_TYPES} CACHE STRING "Selected types for quantization profile" FORCE)
    set(SD_TYPE_PROFILE "quantization" CACHE STRING "Active type profile" FORCE)
    
    message(STATUS "Applied quantization type profile")
    message(STATUS "Selected types: ${SD_SELECTED_TYPES}")
    
    # Set specific optimization flags
    set(SD_OPTIMIZE_QUANTIZATION ON CACHE BOOL "Enable quantization optimizations" FORCE)
    set(SD_PRESERVE_QUANTIZATION_PATTERNS ON CACHE BOOL "Preserve quantization patterns" FORCE)
    set(SD_ELIMINATE_PRECISION_WASTE ON CACHE BOOL "Eliminate precision waste" FORCE)
endfunction()

# Set training type profile - optimized for mixed precision training
function(set_training_type_profile)
    set(SD_SELECTED_TYPES ${TRAINING_TYPES} CACHE STRING "Selected types for training profile" FORCE)
    set(SD_TYPE_PROFILE "training" CACHE STRING "Active type profile" FORCE)
    
    message(STATUS "Applied training type profile")
    message(STATUS "Selected types: ${SD_SELECTED_TYPES}")
    
    # Set specific optimization flags
    set(SD_OPTIMIZE_MIXED_PRECISION ON CACHE BOOL "Enable mixed precision optimizations" FORCE)
    set(SD_PRESERVE_GRADIENT_TYPES ON CACHE BOOL "Preserve gradient accumulation types" FORCE)
    set(SD_ENABLE_FP16_TRAINING ON CACHE BOOL "Enable FP16 training support" FORCE)
endfunction()

# Set inference type profile - optimized for deployment
function(set_inference_type_profile)
    set(SD_SELECTED_TYPES ${INFERENCE_TYPES} CACHE STRING "Selected types for inference profile" FORCE)
    set(SD_TYPE_PROFILE "inference" CACHE STRING "Active type profile" FORCE)
    
    message(STATUS "Applied inference type profile")  
    message(STATUS "Selected types: ${SD_SELECTED_TYPES}")
    
    # Set specific optimization flags
    set(SD_OPTIMIZE_INFERENCE ON CACHE BOOL "Enable inference optimizations" FORCE)
    set(SD_ENABLE_QUANTIZED_INFERENCE ON CACHE BOOL "Enable quantized inference" FORCE)
    set(SD_MINIMIZE_MEMORY_FOOTPRINT ON CACHE BOOL "Minimize memory footprint" FORCE)
endfunction()

# Set NLP type profile - optimized for string processing + embeddings
function(set_nlp_type_profile)
    set(SD_SELECTED_TYPES ${NLP_TYPES} CACHE STRING "Selected types for NLP profile" FORCE)
    set(SD_TYPE_PROFILE "nlp" CACHE STRING "Active type profile" FORCE)
    
    message(STATUS "Applied NLP type profile")
    message(STATUS "Selected types: ${SD_SELECTED_TYPES}")
    
    # Set specific optimization flags
    set(SD_ENABLE_STRING_OPERATIONS ON CACHE BOOL "Enable string operations" FORCE)
    set(SD_OPTIMIZE_TOKENIZATION ON CACHE BOOL "Optimize tokenization operations" FORCE)
    set(SD_PRESERVE_TEXT_ENCODINGS ON CACHE BOOL "Preserve text encoding types" FORCE)
endfunction()

# Set computer vision type profile - optimized for image processing
function(set_cv_type_profile)
    set(SD_SELECTED_TYPES ${CV_TYPES} CACHE STRING "Selected types for CV profile" FORCE)
    set(SD_TYPE_PROFILE "cv" CACHE STRING "Active type profile" FORCE)
    
    message(STATUS "Applied computer vision type profile")
    message(STATUS "Selected types: ${SD_SELECTED_TYPES}")
    
    # Set specific optimization flags
    set(SD_OPTIMIZE_IMAGE_PROCESSING ON CACHE BOOL "Enable image processing optimizations" FORCE)
    set(SD_ENABLE_UINT8_IMAGES ON CACHE BOOL "Enable UINT8 image support" FORCE)
    set(SD_OPTIMIZE_CONVOLUTION ON CACHE BOOL "Optimize convolution operations" FORCE)
endfunction()

# Apply type profile based on profile name
function(apply_type_profile profile_name)
    if(profile_name STREQUAL "quantization")
        set_quantization_type_profile()
    elseif(profile_name STREQUAL "training")
        set_training_type_profile()
    elseif(profile_name STREQUAL "inference")
        set_inference_type_profile()
    elseif(profile_name STREQUAL "nlp")
        set_nlp_type_profile()
    elseif(profile_name STREQUAL "cv")
        set_cv_type_profile()
    else()
        message(WARNING "Unknown type profile: ${profile_name}")
        message(STATUS "Available profiles: quantization, training, inference, nlp, cv")
    endif()
endfunction()

# Get profile-specific type combinations
function(get_profile_type_combinations profile_name result_var)
    if(profile_name STREQUAL "quantization")
        set(profile_types ${QUANTIZATION_TYPES})
    elseif(profile_name STREQUAL "training")
        set(profile_types ${TRAINING_TYPES})
    elseif(profile_name STREQUAL "inference")
        set(profile_types ${INFERENCE_TYPES})
    elseif(profile_name STREQUAL "nlp")
        set(profile_types ${NLP_TYPES})
    elseif(profile_name STREQUAL "cv")
        set(profile_types ${CV_TYPES})
    else()
        # Default to all types
        get_all_types(profile_types)
    endif()
    
    set(${result_var} "${profile_types}" PARENT_SCOPE)
endfunction()

# Filter combinations based on active profile
function(filter_combinations_for_profile profile_name combinations result_var)
    if(NOT DEFINED profile_name OR profile_name STREQUAL "")
        set(${result_var} "${combinations}" PARENT_SCOPE)
        return()
    endif()
    
    get_profile_type_combinations(${profile_name} profile_types)
    set(filtered_combinations "")
    
    foreach(combination ${combinations})
        string(REPLACE "," ";" combo_parts ${combination})
        set(valid_combination TRUE)
        
        # Check if all types in combination are in profile
        foreach(type ${combo_parts})
            list(FIND profile_types ${type} type_index)
            if(type_index EQUAL -1)
                set(valid_combination FALSE)
                break()
            endif()
        endforeach()
        
        if(valid_combination)
            list(APPEND filtered_combinations ${combination})
        endif()
    endforeach()
    
    set(${result_var} "${filtered_combinations}" PARENT_SCOPE)
endfunction()

# Get optimized type list for profile
function(get_optimized_types_for_profile profile_name result_var)
    get_profile_type_combinations(${profile_name} profile_types)
    
    # Apply profile-specific optimizations
    if(profile_name STREQUAL "quantization")
        # Prioritize quantization-friendly types
        set(optimized_types "")
        foreach(type ${profile_types})
            if(type MATCHES "int8_t|uint8_t")
                list(INSERT optimized_types 0 ${type})  # Insert at beginning
            else()
                list(APPEND optimized_types ${type})
            endif()
        endforeach()
        set(${result_var} "${optimized_types}" PARENT_SCOPE)
    elseif(profile_name STREQUAL "training")
        # Prioritize mixed precision types
        set(optimized_types "")
        foreach(type ${profile_types})
            if(type MATCHES "float16|bfloat16")
                list(INSERT optimized_types 0 ${type})  # Insert at beginning
            else()
                list(APPEND optimized_types ${type})
            endif()
        endforeach()
        set(${result_var} "${optimized_types}" PARENT_SCOPE)
    else()
        set(${result_var} "${profile_types}" PARENT_SCOPE)
    endif()
endfunction()

# Validate profile configuration
function(validate_profile_configuration profile_name)
    if(NOT profile_name)
        message(STATUS "No type profile specified - using all available types")
        return()
    endif()
    
    get_profile_type_combinations(${profile_name} profile_types)
    list(LENGTH profile_types type_count)
    
    if(type_count EQUAL 0)
        message(FATAL_ERROR "Profile '${profile_name}' has no valid types")
    endif()
    
    message(STATUS "Profile '${profile_name}' validated with ${type_count} types")
    
    # Estimate combination count
    math(EXPR estimated_2_combinations "${type_count} * ${type_count}")
    math(EXPR estimated_3_combinations "${type_count} * ${type_count} * ${type_count}")
    
    message(STATUS "Estimated 2-type combinations: ${estimated_2_combinations}")
    message(STATUS "Estimated 3-type combinations: ${estimated_3_combinations}")
    
    # Warn about large combination counts
    if(estimated_3_combinations GREATER 1000)
        message(STATUS "Large combination count detected - consider using selective types")
    endif()
endfunction()

# Auto-detect optimal profile based on build configuration
function(auto_detect_optimal_profile result_var)
    set(detected_profile "")
    
    # Check for quantization indicators
    if(DEFINED SD_QUANTIZATION OR DEFINED SD_INT8 OR DEFINED SD_UINT8)
        set(detected_profile "quantization")
    # Check for training indicators  
    elseif(DEFINED SD_TRAINING OR DEFINED SD_MIXED_PRECISION OR DEFINED SD_FP16)
        set(detected_profile "training")
    # Check for inference indicators
    elseif(DEFINED SD_INFERENCE OR DEFINED SD_DEPLOYMENT)
        set(detected_profile "inference")
    # Check for NLP indicators
    elseif(DEFINED SD_NLP OR DEFINED SD_STRING_OPS)
        set(detected_profile "nlp")
    # Check for CV indicators
    elseif(DEFINED SD_CV OR DEFINED SD_IMAGE_PROCESSING)
        set(detected_profile "cv")
    endif()
    
    if(NOT detected_profile STREQUAL "")
        message(STATUS "Auto-detected optimal profile: ${detected_profile}")
    endif()
    
    set(${result_var} "${detected_profile}" PARENT_SCOPE)
endfunction()

# Print profile information
function(print_profile_info profile_name)
    if(NOT profile_name)
        message(STATUS "No active type profile")
        return()
    endif()
    
    message(STATUS "=== Type Profile Information ===")
    message(STATUS "Active Profile: ${profile_name}")
    
    get_profile_type_combinations(${profile_name} profile_types)
    message(STATUS "Profile Types: ${profile_types}")
    
    list(LENGTH profile_types type_count)
    message(STATUS "Type Count: ${type_count}")
    
    # Profile-specific information
    if(profile_name STREQUAL "quantization")
        message(STATUS "Optimizations: INT8/UINT8 inference, quantization patterns")
    elseif(profile_name STREQUAL "training")
        message(STATUS "Optimizations: Mixed precision, gradient accumulation")
    elseif(profile_name STREQUAL "inference")
        message(STATUS "Optimizations: Deployment, quantized inference")
    elseif(profile_name STREQUAL "nlp")
        message(STATUS "Optimizations: String operations, tokenization")
    elseif(profile_name STREQUAL "cv")
        message(STATUS "Optimizations: Image processing, convolution")
    endif()
    
    message(STATUS "=== End Profile Information ===")
endfunction()
