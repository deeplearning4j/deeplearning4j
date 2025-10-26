# ============================================================================
# SelectiveRenderingCore.cmake (v20 - Production Optimized) - FIXED
#
# Optimized version with debug profiles removed for production builds.
# Conditional diagnostics only when explicitly enabled via SD_ENABLE_DIAGNOSTICS.
# ============================================================================

# Include the reporting functions
include(SelectiveRenderingReports)

# Export type validation results for use by SelectiveRenderingCore
function(export_validated_types_for_selective_rendering)
    if(SD_TYPES_LIST_COUNT GREATER 0)
        set(SRCORE_USE_SELECTIVE_TYPES TRUE PARENT_SCOPE)
        set(SRCORE_VALIDATED_TYPES "${SD_TYPES_LIST}" PARENT_SCOPE)
        if(SD_ENABLE_DIAGNOSTICS)
            message(STATUS "Exporting SELECTIVE types for SelectiveRenderingCore: ${SD_TYPES_LIST}")
        endif()
    else()
        set(SRCORE_USE_SELECTIVE_TYPES FALSE PARENT_SCOPE)
        set(SRCORE_VALIDATED_TYPES "" PARENT_SCOPE)
        if(SD_ENABLE_DIAGNOSTICS)
            message(STATUS "Exporting ALL_TYPES mode for SelectiveRenderingCore")
        endif()
    endif()
endfunction()

# ============================================================================
# SECTION 1: SEMANTIC FILTERING LOGIC (Optimized)
# ============================================================================
function(_internal_srcore_is_type_numeric type_name output_var)
    # Include all numeric types
    set(numeric_types "BOOL;INT8;UINT8;INT16;UINT16;INT32;UINT32;INT64;UINT64;FLOAT32;DOUBLE;HALF;BFLOAT16")
    list(FIND numeric_types "${type_name}" found_index)
    if(found_index GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_is_type_floating type_name output_var)
    set(floating_types "DOUBLE;FLOAT32;HALF;BFLOAT16")
    list(FIND floating_types "${type_name}" found_index)
    if(found_index GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_is_type_integer type_name output_var)
    # Include both signed and unsigned integer types
    set(integer_types "INT8;UINT8;INT16;UINT16;INT32;UINT32;INT64;UINT64;BOOL")
    list(FIND integer_types "${type_name}" found_index)
    if(found_index GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_internal_srcore_get_type_priority type_name output_var)
    # Proper priority ordering: higher precision = higher priority
    if(type_name STREQUAL "DOUBLE")
        set(${output_var} 10 PARENT_SCOPE)
    elseif(type_name STREQUAL "INT64")
        set(${output_var} 9 PARENT_SCOPE)
    elseif(type_name STREQUAL "UINT64")
        set(${output_var} 8 PARENT_SCOPE)
    elseif(type_name STREQUAL "FLOAT32")
        set(${output_var} 7 PARENT_SCOPE)
    elseif(type_name STREQUAL "INT32")
        set(${output_var} 6 PARENT_SCOPE)
    elseif(type_name STREQUAL "UINT32")
        set(${output_var} 5 PARENT_SCOPE)
    elseif(type_name STREQUAL "BFLOAT16")
        set(${output_var} 4 PARENT_SCOPE)
    elseif(type_name STREQUAL "HALF")
        set(${output_var} 4 PARENT_SCOPE)
    elseif(type_name STREQUAL "INT16")
        set(${output_var} 3 PARENT_SCOPE)
    elseif(type_name STREQUAL "UINT16")
        set(${output_var} 3 PARENT_SCOPE)
    elseif(type_name STREQUAL "INT8")
        set(${output_var} 2 PARENT_SCOPE)
    elseif(type_name STREQUAL "UINT8")
        set(${output_var} 2 PARENT_SCOPE)
    elseif(type_name STREQUAL "BOOL")
        set(${output_var} 1 PARENT_SCOPE)
    else()
        set(${output_var} 0 PARENT_SCOPE)
    endif()
endfunction()

# SelectiveRenderingCore.cmake - Validation Functions
function(_internal_srcore_is_valid_pair type1 type2 output_var)
    # Same type pairs are always valid
    if(type1 STREQUAL type2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Define strict whitelists based on actual operations that exist
    
    # TypeCast operations (type conversions)
    set(valid_typecast_pairs
        # Quantization/Dequantization
        "INT8,FLOAT32"
        "UINT8,FLOAT32"
        "FLOAT32,INT8"
        "FLOAT32,UINT8"
        
        # Integer promotions (only necessary ones)
        "INT8,INT32"
        "UINT8,INT32"
        "INT16,INT32"
        "UINT16,INT32"
        "INT32,INT64"
        "UINT32,UINT64"
        
        # Float promotions
        "HALF,FLOAT32"
        "BFLOAT16,FLOAT32"
        "FLOAT32,DOUBLE"
        
        # Mixed precision format conversions
        "HALF,BFLOAT16"
        "BFLOAT16,HALF"
        
        # Essential int to float
        "INT32,FLOAT32"
        "INT64,DOUBLE"
        
        # Same-width mixed signedness
        "INT8,UINT8"
        "INT16,UINT16"
        "INT32,UINT32"
        "INT64,UINT64"
    )
    
    # DoubleMethods operations (sorting by key/value)
    set(valid_doublemethods_pairs
        # Sorting operations typically use same types or int indices
        "INT32,INT32"
        "INT64,INT64"
        "FLOAT32,FLOAT32"
        "DOUBLE,DOUBLE"
        
        # Index sorting (int key, any value)
        "INT32,FLOAT32"
        "INT32,DOUBLE"
        "INT64,FLOAT32"
        "INT64,DOUBLE"
    )
    
    # Special operations
    set(valid_special_pairs
        # Bool comparisons (from same type)
        "BOOL,BOOL"
        
        # String operations
        "UTF8,INT32"
        "UTF8,INT64"
        "UTF16,INT32"
        "UTF16,INT64"
        "UTF32,INT32"
        "UTF32,INT64"
        
        # Bool with core types only
        "BOOL,INT32"
        "BOOL,INT64"
        "BOOL,FLOAT32"
        "INT32,BOOL"
        "INT64,BOOL"
        "FLOAT32,BOOL"
    )
    
    # Check if pair is in any whitelist
    set(pair "${type1},${type2}")
    set(reverse "${type2},${type1}")
    
    # Check TypeCast operations
    list(FIND valid_typecast_pairs "${pair}" found_typecast)
    list(FIND valid_typecast_pairs "${reverse}" found_typecast_rev)
    if(found_typecast GREATER_EQUAL 0 OR found_typecast_rev GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Check DoubleMethods operations
    list(FIND valid_doublemethods_pairs "${pair}" found_double)
    list(FIND valid_doublemethods_pairs "${reverse}" found_double_rev)
    if(found_double GREATER_EQUAL 0 OR found_double_rev GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Check Special operations
    list(FIND valid_special_pairs "${pair}" found_special)
    list(FIND valid_special_pairs "${reverse}" found_special_rev)
    if(found_special GREATER_EQUAL 0 OR found_special_rev GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # HARD BLOCKS - patterns that should never occur
    
    # Block 8-bit types with HALF/BFLOAT16/DOUBLE
    if(type1 MATCHES "INT8|UINT8")
        if(type2 MATCHES "HALF|BFLOAT16|DOUBLE")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    if(type2 MATCHES "INT8|UINT8")
        if(type1 MATCHES "HALF|BFLOAT16|DOUBLE")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Block 16-bit types with DOUBLE
    if(type1 MATCHES "INT16|UINT16|HALF|BFLOAT16")
        if(type2 STREQUAL "DOUBLE")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    if(type2 MATCHES "INT16|UINT16|HALF|BFLOAT16")
        if(type1 STREQUAL "DOUBLE")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Block BOOL with DOUBLE/HALF/BFLOAT16
    if(type1 STREQUAL "BOOL" OR type2 STREQUAL "BOOL")
        if(type1 MATCHES "DOUBLE|HALF|BFLOAT16" OR type2 MATCHES "DOUBLE|HALF|BFLOAT16")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Block 8-bit with 64-bit (too large a jump)
    if((type1 MATCHES "INT8|UINT8" AND type2 MATCHES "INT64|UINT64|DOUBLE") OR
       (type2 MATCHES "INT8|UINT8" AND type1 MATCHES "INT64|UINT64|DOUBLE"))
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Block mixed signedness except same width
    set(signed_types "INT8;INT16;INT32;INT64")
    set(unsigned_types "UINT8;UINT16;UINT32;UINT64")
    
    list(FIND signed_types "${type1}" is_signed1)
    list(FIND unsigned_types "${type1}" is_unsigned1)
    list(FIND signed_types "${type2}" is_signed2)
    list(FIND unsigned_types "${type2}" is_unsigned2)
    
    if((is_signed1 GREATER_EQUAL 0 AND is_unsigned2 GREATER_EQUAL 0) OR
       (is_unsigned1 GREATER_EQUAL 0 AND is_signed2 GREATER_EQUAL 0))
        # Mixed signedness - already checked in whitelists for same-width
        # If not in whitelist, it's invalid
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Default: invalid - not in any whitelist
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()


# SelectiveRenderingCore.cmake - Triple Validation Function
function(_internal_srcore_is_valid_triple type1 type2 type3 output_var)
    # Same type triples are always valid
    if(type1 STREQUAL type2 AND type2 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # First validate input pairs using the strict pair validation
    _internal_srcore_is_valid_pair("${type1}" "${type2}" pair_valid)
    if(NOT pair_valid)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Define strict whitelists for triple operations
    
    # Broadcast operations (X,Y,Z where X op Y -> Z)
    set(valid_broadcast_triples
        # Same type operations
        "INT8,INT8,INT8"
        "UINT8,UINT8,UINT8"
        "INT16,INT16,INT16"
        "UINT16,UINT16,UINT16"
        "INT32,INT32,INT32"
        "UINT32,UINT32,UINT32"
        "INT64,INT64,INT64"
        "UINT64,UINT64,UINT64"
        "FLOAT32,FLOAT32,FLOAT32"
        "DOUBLE,DOUBLE,DOUBLE"
        "HALF,HALF,HALF"
        "BFLOAT16,BFLOAT16,BFLOAT16"
        
        # INT8/UINT8 accumulation to INT32
        "INT8,INT8,INT32"
        "UINT8,UINT8,INT32"
        "INT8,UINT8,INT32"
        "UINT8,INT8,INT32"
        
        # INT8/UINT8 to FLOAT32 (dequantization)
        "INT8,INT8,FLOAT32"
        "UINT8,UINT8,FLOAT32"
        "INT8,FLOAT32,FLOAT32"
        "UINT8,FLOAT32,FLOAT32"
        "FLOAT32,INT8,FLOAT32"
        "FLOAT32,UINT8,FLOAT32"
        
        # Mixed precision accumulation
        "HALF,HALF,FLOAT32"
        "BFLOAT16,BFLOAT16,FLOAT32"
        "HALF,FLOAT32,FLOAT32"
        "BFLOAT16,FLOAT32,FLOAT32"
        "FLOAT32,HALF,FLOAT32"
        "FLOAT32,BFLOAT16,FLOAT32"
        
        # Integer division to float
        "INT32,INT32,FLOAT32"
        "INT64,INT64,DOUBLE"
        
        # Float accumulation to higher precision
        "FLOAT32,FLOAT32,DOUBLE"
        "FLOAT32,DOUBLE,DOUBLE"
        "DOUBLE,FLOAT32,DOUBLE"
        
        # Safe integer promotions
        "INT16,INT16,INT32"
        "UINT16,UINT16,UINT32"
        "INT32,INT32,INT64"
        "UINT32,UINT32,UINT64"
    )
    
    # Comparison operations (X,Y,BOOL)
    set(valid_comparison_triples
        # Same type comparisons
        "INT8,INT8,BOOL"
        "UINT8,UINT8,BOOL"
        "INT16,INT16,BOOL"
        "UINT16,UINT16,BOOL"
        "INT32,INT32,BOOL"
        "UINT32,UINT32,BOOL"
        "INT64,INT64,BOOL"
        "UINT64,UINT64,BOOL"
        "FLOAT32,FLOAT32,BOOL"
        "DOUBLE,DOUBLE,BOOL"
        "HALF,HALF,BOOL"
        "BFLOAT16,BFLOAT16,BOOL"
        "BOOL,BOOL,BOOL"
        
        # Limited mixed comparisons
        "INT32,FLOAT32,BOOL"
        "FLOAT32,INT32,BOOL"
        "HALF,FLOAT32,BOOL"
        "FLOAT32,HALF,BOOL"
        "BFLOAT16,FLOAT32,BOOL"
        "FLOAT32,BFLOAT16,BOOL"
    )
    
    # Masking operations (X,BOOL,X)
    set(valid_masking_triples
        "INT32,BOOL,INT32"
        "INT64,BOOL,INT64"
        "FLOAT32,BOOL,FLOAT32"
        "DOUBLE,BOOL,DOUBLE"
        "HALF,BOOL,HALF"
        "BFLOAT16,BOOL,BFLOAT16"
    )
    
    # Scalar operations (input,scalar,output)
    set(valid_scalar_triples
        # Same type scalar ops
        "INT32,INT32,INT32"
        "INT64,INT64,INT64"
        "FLOAT32,FLOAT32,FLOAT32"
        "DOUBLE,DOUBLE,DOUBLE"
        "HALF,HALF,HALF"
        "BFLOAT16,BFLOAT16,BFLOAT16"
        
        # Mixed scalar operations
        "FLOAT32,HALF,FLOAT32"
        "FLOAT32,BFLOAT16,FLOAT32"
        "HALF,FLOAT32,FLOAT32"
        "BFLOAT16,FLOAT32,FLOAT32"
        
        # Integer scalar with float output
        "INT32,FLOAT32,FLOAT32"
        "INT64,DOUBLE,DOUBLE"
    )
    
    # PairwiseTransform operations
    set(valid_pairwise_triples
        # Same type transforms
        "FLOAT32,FLOAT32,FLOAT32"
        "DOUBLE,DOUBLE,DOUBLE"
        "INT32,INT32,INT32"
        "INT64,INT64,INT64"
        
        # Mixed precision pairwise
        "HALF,HALF,FLOAT32"
        "BFLOAT16,BFLOAT16,FLOAT32"
        
        # Quantization output
        "FLOAT32,FLOAT32,INT8"
        "FLOAT32,FLOAT32,UINT8"
        "FLOAT32,FLOAT32,HALF"
        "FLOAT32,FLOAT32,BFLOAT16"
        
        # INT8 operations
        "INT8,INT8,INT32"
        "UINT8,UINT8,INT32"
    )
    
    # Indexing operations (idx,data,data)
    set(valid_indexing_triples
        "INT32,FLOAT32,FLOAT32"
        "INT64,FLOAT32,FLOAT32"
        "INT32,DOUBLE,DOUBLE"
        "INT64,DOUBLE,DOUBLE"
        "INT32,HALF,HALF"
        "INT64,HALF,HALF"
        "INT32,BFLOAT16,BFLOAT16"
        "INT64,BFLOAT16,BFLOAT16"
        "INT32,INT32,INT32"
        "INT64,INT64,INT64"
    )
    
    # Conditional/ternary operations (BOOL,X,X)
    set(valid_conditional_triples
        "BOOL,INT32,INT32"
        "BOOL,INT64,INT64"
        "BOOL,FLOAT32,FLOAT32"
        "BOOL,DOUBLE,DOUBLE"
        "BOOL,HALF,HALF"
        "BOOL,BFLOAT16,BFLOAT16"
    )
    
    # Check if triple is in any whitelist
    set(triple "${type1},${type2},${type3}")
    
    # Check each category
    list(FIND valid_broadcast_triples "${triple}" found_broadcast)
    if(found_broadcast GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    list(FIND valid_comparison_triples "${triple}" found_comparison)
    if(found_comparison GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    list(FIND valid_masking_triples "${triple}" found_masking)
    if(found_masking GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    list(FIND valid_scalar_triples "${triple}" found_scalar)
    if(found_scalar GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    list(FIND valid_pairwise_triples "${triple}" found_pairwise)
    if(found_pairwise GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    list(FIND valid_indexing_triples "${triple}" found_indexing)
    if(found_indexing GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    list(FIND valid_conditional_triples "${triple}" found_conditional)
    if(found_conditional GREATER_EQUAL 0)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # HARD BLOCKS - patterns that should never occur in triples
    
    # Never output to 8-bit unless all inputs are 8-bit
    if(type3 MATCHES "INT8|UINT8")
        if(NOT (type1 MATCHES "INT8|UINT8" AND type2 MATCHES "INT8|UINT8"))
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Never output DOUBLE from 8/16-bit inputs
    if(type3 STREQUAL "DOUBLE")
        if(type1 MATCHES "INT8|UINT8|INT16|UINT16|HALF|BFLOAT16" OR
           type2 MATCHES "INT8|UINT8|INT16|UINT16|HALF|BFLOAT16")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Never downcast precision
    if(type3 MATCHES "HALF|BFLOAT16")
        if(type1 STREQUAL "DOUBLE" OR type2 STREQUAL "DOUBLE")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    if(type3 STREQUAL "FLOAT32")
        if(type1 STREQUAL "DOUBLE" OR type2 STREQUAL "DOUBLE")
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Block BOOL with precision types except as comparison output
    if((type1 STREQUAL "BOOL" OR type2 STREQUAL "BOOL") AND 
       type3 MATCHES "DOUBLE|HALF|BFLOAT16")
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Default: invalid - not in any whitelist
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# Helper function for integer promotion validation
function(_internal_is_valid_int_promotion from_type to_type result_var)
    set(int_hierarchy "INT8;UINT8;INT16;UINT16;INT32;UINT32;INT64;UINT64")
    list(FIND int_hierarchy "${from_type}" from_idx)
    list(FIND int_hierarchy "${to_type}" to_idx)

    if(from_idx LESS 0 OR to_idx LESS 0)
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Allow promotion to same or larger type
    if(to_idx GREATER_EQUAL from_idx)
        set(${result_var} TRUE PARENT_SCOPE)
    else()
        set(${result_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ============================================================================
# SECTION 2: OPTIMIZED COMBINATION GENERATION
# ============================================================================
function(_internal_srcore_generate_combinations active_indices type_names profile result_2_var result_3_var)
    list(LENGTH active_indices type_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types for combination generation")
    endif()

    set(combinations_2 "")
    set(combinations_3 "")

    math(EXPR max_index "${type_count} - 1")

    # Calculate theoretical maximums
    math(EXPR max_possible_2 "${type_count} * ${type_count}")
    math(EXPR max_possible_3 "${type_count} * ${type_count} * ${type_count}")

    # Generate 2-type combinations with filtering
    foreach(i RANGE ${max_index})
        list(GET type_names ${i} type_i)
        foreach(j RANGE ${max_index})
            list(GET type_names ${j} type_j)

            _internal_srcore_is_valid_pair("${type_i}" "${type_j}" is_valid)
            if(is_valid)
                list(APPEND combinations_2 "${i},${j}")
            endif()
        endforeach()
    endforeach()

    # Generate 3-type combinations with strict filtering
    foreach(i RANGE ${max_index})
        list(GET type_names ${i} type_i)
        foreach(j RANGE ${max_index})
            list(GET type_names ${j} type_j)
            foreach(k RANGE ${max_index})
                list(GET type_names ${k} type_k)

                _internal_srcore_is_valid_triple("${type_i}" "${type_j}" "${type_k}" is_valid)
                if(is_valid)
                    list(APPEND combinations_3 "${i},${j},${k}")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # Get counts before profile filtering
    list(LENGTH combinations_2 accepted_2)
    list(LENGTH combinations_3 accepted_3_before_profile)

    # Apply profile-based limits if specified
    if(DEFINED profile AND NOT profile STREQUAL "")
        list(LENGTH combinations_3 current_count)
        set(max_combinations 0)

        if(profile STREQUAL "minimal")
            set(max_combinations 50)
        elseif(profile STREQUAL "quantization")
            set(max_combinations 100)
        elseif(profile STREQUAL "inference" OR profile STREQUAL "ESSENTIAL")
            set(max_combinations 250)
        elseif(profile STREQUAL "training")
            set(max_combinations 350)
        else()
            set(max_combinations 500)
        endif()

        if(current_count GREATER max_combinations)
            list(SUBLIST combinations_3 0 ${max_combinations} combinations_3)
            message(STATUS "  Profile '${profile}' limit: ${current_count} -> ${max_combinations} combinations")
        endif()
    endif()

    # Get final count
    list(LENGTH combinations_3 accepted_3)

    # Calculate reduction percentages correctly
    if(max_possible_2 GREATER 0)
        math(EXPR reduction_2 "100 - (100 * ${accepted_2} / ${max_possible_2})")
    else()
        set(reduction_2 0)
    endif()

    if(max_possible_3 GREATER 0)
        math(EXPR reduction_3 "100 - (100 * ${accepted_3} / ${max_possible_3})")
    else()
        set(reduction_3 0)
    endif()

    # Report statistics
    message(STATUS "🎯 Selective Rendering Results:")
    message(STATUS "   - Active types: ${type_count}")
    message(STATUS "   - 2-type combinations: ${accepted_2}/${max_possible_2} (${reduction_2}% reduction)")
    message(STATUS "   - 3-type combinations: ${accepted_3}/${max_possible_3} (${reduction_3}% reduction)")

    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()


# Helper function to convert enum name to integer value
function(enum_to_int_value enum_name out_var)
    # Map enum names to their integer values
    if(enum_name STREQUAL "BOOL")
        set(${out_var} 1 PARENT_SCOPE)
    elseif(enum_name STREQUAL "FLOAT8")
        set(${out_var} 2 PARENT_SCOPE)
    elseif(enum_name STREQUAL "HALF" OR enum_name STREQUAL "FLOAT16")
        set(${out_var} 3 PARENT_SCOPE)
    elseif(enum_name STREQUAL "HALF2")
        set(${out_var} 4 PARENT_SCOPE)
    elseif(enum_name STREQUAL "FLOAT32")
        set(${out_var} 5 PARENT_SCOPE)
    elseif(enum_name STREQUAL "DOUBLE" OR enum_name STREQUAL "FLOAT64")
        set(${out_var} 6 PARENT_SCOPE)
    elseif(enum_name STREQUAL "INT8")
        set(${out_var} 7 PARENT_SCOPE)
    elseif(enum_name STREQUAL "INT16")
        set(${out_var} 8 PARENT_SCOPE)
    elseif(enum_name STREQUAL "INT32")
        set(${out_var} 9 PARENT_SCOPE)
    elseif(enum_name STREQUAL "INT64" OR enum_name STREQUAL "LONG")
        set(${out_var} 10 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UINT8")
        set(${out_var} 11 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UINT16")
        set(${out_var} 12 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UINT32")
        set(${out_var} 13 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UINT64" OR enum_name STREQUAL "ULONG")
        set(${out_var} 14 PARENT_SCOPE)
    elseif(enum_name STREQUAL "QINT8")
        set(${out_var} 15 PARENT_SCOPE)
    elseif(enum_name STREQUAL "QINT16")
        set(${out_var} 16 PARENT_SCOPE)
    elseif(enum_name STREQUAL "BFLOAT16")
        set(${out_var} 17 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UTF8")
        set(${out_var} 50 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UTF16")
        set(${out_var} 51 PARENT_SCOPE)
    elseif(enum_name STREQUAL "UTF32")
        set(${out_var} 52 PARENT_SCOPE)
    else()
        message(WARNING "Unknown enum name: ${enum_name}")
        set(${out_var} 0 PARENT_SCOPE)
    endif()
endfunction()


function(report_template_statistics combinations_2 combinations_3 type_count)
    list(LENGTH combinations_2 num_2)
    list(LENGTH combinations_3 num_3)

    # Calculate theoretical maximum
    math(EXPR max_3 "${type_count} * ${type_count} * ${type_count}")

    if(max_3 GREATER 0)
        math(EXPR reduction_percent "100 - (100 * ${num_3} / ${max_3})")
    else()
        set(reduction_percent 0)
    endif()

    message(STATUS "🎯 Template Generation Statistics:")
    message(STATUS "   - 2-type combinations: ${num_2}")
    message(STATUS "   - 3-type combinations: ${num_3}")
    message(STATUS "   - Template instantiation reduction: ${reduction_percent}%")
endfunction()

# ============================================================================
# SECTION 3: CORE HELPER FUNCTIONS (Optimized)
# ============================================================================
function(srcore_normalize_type input_type output_var)
    set(normalized_type "${input_type}")

    # Handle all common type aliases
    if(normalized_type STREQUAL "float32")
        set(normalized_type "float")
    elseif(normalized_type STREQUAL "float64")
        set(normalized_type "double")
    elseif(normalized_type STREQUAL "half")
        set(normalized_type "float16")
    elseif(normalized_type STREQUAL "long")
        set(normalized_type "int64_t")
    elseif(normalized_type STREQUAL "LongType")
        set(normalized_type "int64_t")
    elseif(normalized_type STREQUAL "int")
        set(normalized_type "int32_t")
    elseif(normalized_type STREQUAL "Int32Type")
        set(normalized_type "int32_t")
    elseif(normalized_type STREQUAL "bfloat")
        set(normalized_type "bfloat16")
    elseif(normalized_type STREQUAL "qint8")
        set(normalized_type "int8_t")
    elseif(normalized_type STREQUAL "quint8")
        set(normalized_type "uint8_t")
    elseif(normalized_type STREQUAL "qint16")
        set(normalized_type "int16_t")
    elseif(normalized_type STREQUAL "quint16")
        set(normalized_type "uint16_t")
    elseif(normalized_type STREQUAL "UnsignedLong")
        set(normalized_type "uint64_t")
    elseif(normalized_type STREQUAL "utf8")
        set(normalized_type "std::string")
    elseif(normalized_type STREQUAL "utf16")
        set(normalized_type "std::u16string")
    elseif(normalized_type STREQUAL "utf32")
        set(normalized_type "std::u32string")
    endif()

    set(${output_var} "${normalized_type}" PARENT_SCOPE)
endfunction()

function(is_semantically_valid_combination type1 type2 type3 mode result_var)
    # Normalize types first
    srcore_normalize_type("${type1}" norm_t1)
    srcore_normalize_type("${type2}" norm_t2)
    srcore_normalize_type("${type3}" norm_t3)
    
    # Rule 1: Same type combinations are ALWAYS valid
    if(norm_t1 STREQUAL norm_t2 AND norm_t2 STREQUAL norm_t3)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 2: Quantization patterns
    if((norm_t1 STREQUAL "int8_t" OR norm_t1 STREQUAL "uint8_t") AND
       (norm_t2 STREQUAL "int8_t" OR norm_t2 STREQUAL "uint8_t"))
        # INT8 accumulation or dequantization
        if(norm_t3 STREQUAL "int32_t" OR norm_t3 STREQUAL "float")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
        # Block INT8 to half precision (bad pattern)
        if(norm_t3 MATCHES "float16|bfloat16")
            set(${result_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Rule 3: Mixed precision training
    if((norm_t1 STREQUAL "float16" OR norm_t1 STREQUAL "bfloat16") AND
       (norm_t2 STREQUAL "float16" OR norm_t2 STREQUAL "bfloat16") AND
       norm_t3 STREQUAL "float")
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 4: Embedding lookups
    if((norm_t1 STREQUAL "int32_t" OR norm_t1 STREQUAL "int64_t") AND
       norm_t2 STREQUAL "float" AND norm_t3 STREQUAL "float")
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 5: Comparisons producing bool
    if(norm_t3 STREQUAL "bool")
        # Any same-type comparison is valid
        if(norm_t1 STREQUAL norm_t2)
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Rule 6: Masking operations
    if(norm_t2 STREQUAL "bool" AND norm_t1 STREQUAL norm_t3)
        set(${result_var} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 7: Block nonsensical patterns
    # Integer operations producing half precision
    if(norm_t1 MATCHES "int" AND norm_t2 MATCHES "int" AND
       norm_t3 MATCHES "float16|bfloat16")
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Double to half precision (extreme downcast)
    if(norm_t1 STREQUAL "double" AND norm_t3 MATCHES "float16|bfloat16")
        set(${result_var} FALSE PARENT_SCOPE)
        return()
    endif()
    
    # Rule 8: Standard promotions
    if(norm_t1 STREQUAL norm_t2)
        # Integer to float promotion
        if(norm_t1 MATCHES "int" AND norm_t3 MATCHES "float|double")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
        # Float to double promotion
        if(norm_t1 STREQUAL "float" AND norm_t3 STREQUAL "double")
            set(${result_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()
    
    # Default: invalid
    set(${result_var} FALSE PARENT_SCOPE)
endfunction()


function(get_all_types result_var)
    set(all_types
            "bool" "float8" "float16" "half2" "float32" "double"
            "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64"
            "qint8" "qint16" "bfloat16" "utf8" "utf16" "utf32"
    )
    set(${result_var} "${all_types}" PARENT_SCOPE)
endfunction()

function(_internal_srcore_discover_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
    if(NOT DEFINED SRCORE_USE_SELECTIVE_TYPES)
        get_property(cache_selective CACHE SRCORE_USE_SELECTIVE_TYPES PROPERTY VALUE)
        get_property(cache_types CACHE SRCORE_VALIDATED_TYPES PROPERTY VALUE)

        if(DEFINED cache_selective)
            set(SRCORE_USE_SELECTIVE_TYPES "${cache_selective}")
            set(SRCORE_VALIDATED_TYPES "${cache_types}")
        else()
            set(SRCORE_USE_SELECTIVE_TYPES FALSE)
            set(SRCORE_VALIDATED_TYPES "")
        endif()
    endif()

    if(SRCORE_USE_SELECTIVE_TYPES AND DEFINED SRCORE_VALIDATED_TYPES AND NOT SRCORE_VALIDATED_TYPES STREQUAL "")
        _internal_srcore_discover_selective_types("${SRCORE_VALIDATED_TYPES}" discovered_indices discovered_names discovered_enums discovered_cpp_types)
    else()
        _internal_srcore_discover_all_types(discovered_indices discovered_names discovered_enums discovered_cpp_types)
    endif()

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_names}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()

# Replace the entire _internal_srcore_discover_selective_types function with this:
function(_internal_srcore_discover_selective_types validated_types_list result_indices_var result_names_var result_enums_var result_cpp_types_var)
    # types.h is always at include/types/types.h
    set(types_header "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h")

    if(NOT EXISTS "${types_header}")
        message(FATAL_ERROR "Could not find types.h at expected location: ${types_header}")
    endif()

    file(READ "${types_header}" types_content)

    # Get the types from the active profile
    if(DEFINED SD_TYPE_PROFILE)
        get_profile_type_combinations("${SD_TYPE_PROFILE}" profile_types)
        set(types_to_discover "${profile_types}")
    else()
        # Use the validated types list directly
        set(types_to_discover "${validated_types_list}")
    endif()

    # Enhanced type mapping to handle C++ type names to enum names
    # Include ALL possible long type variations
    set(type_mapping_float32 "FLOAT32")
    set(type_mapping_float "FLOAT32")
    set(type_mapping_double "DOUBLE")
    set(type_mapping_int32 "INT32")
    set(type_mapping_int32_t "INT32")
    set(type_mapping_int "INT32")
    set(type_mapping_Int32Type "INT32")
    set(type_mapping_sd__Int32Type "INT32")
    set(type_mapping_signed "INT32")
    set(type_mapping_signed_int "INT32")
    
    # Critical INT64 mappings - include ALL variations
    set(type_mapping_int64 "INT64")
    set(type_mapping_int64_t "INT64")
    set(type_mapping_long_long "INT64")
    set(type_mapping_long_long_int "INT64")
    set(type_mapping_long "INT64")
    set(type_mapping_long_int "INT64")
    set(type_mapping_signed_long "INT64")
    set(type_mapping_signed_long_long "INT64")
    set(type_mapping_signed_long_int "INT64")
    set(type_mapping_LongType "INT64")
    set(type_mapping_sd__LongType "INT64")
    
    # UINT64 mappings
    set(type_mapping_uint64 "UINT64")
    set(type_mapping_uint64_t "UINT64")
    set(type_mapping_unsigned_long_long "UINT64")
    set(type_mapping_unsigned_long_long_int "UINT64")
    set(type_mapping_unsigned_long "UINT64")
    set(type_mapping_unsigned_long_int "UINT64")
    set(type_mapping_UnsignedLong "UINT64")
    set(type_mapping_sd__UnsignedLong "UINT64")
    set(type_mapping_size_t "UINT64")
    
    set(type_mapping_bool "BOOL")
    set(type_mapping_float16 "HALF")
    set(type_mapping_half "HALF")
    set(type_mapping_bfloat16 "BFLOAT16")
    set(type_mapping_bfloat "BFLOAT16")
    set(type_mapping_int8 "INT8")
    set(type_mapping_int8_t "INT8")
    set(type_mapping_signed_char "INT8")
    set(type_mapping_char "INT8")
    set(type_mapping_uint8 "UINT8")
    set(type_mapping_uint8_t "UINT8")
    set(type_mapping_unsigned_char "UINT8")
    set(type_mapping_int16 "INT16")
    set(type_mapping_int16_t "INT16")
    set(type_mapping_short "INT16")
    set(type_mapping_short_int "INT16")
    set(type_mapping_signed_short "INT16")
    set(type_mapping_uint16 "UINT16")
    set(type_mapping_uint16_t "UINT16")
    set(type_mapping_unsigned_short "UINT16")
    set(type_mapping_unsigned_short_int "UINT16")
    set(type_mapping_uint32 "UINT32")
    set(type_mapping_uint32_t "UINT32")
    set(type_mapping_unsigned_int "UINT32")
    set(type_mapping_unsigned "UINT32")

    set(discovered_types "")
    set(discovered_indices "")
    set(discovered_enums "")
    set(discovered_cpp_types "")
    set(type_index 0)

    foreach(user_type ${types_to_discover})
        string(STRIP "${user_type}" user_type)

        # Map C++ type name to enum name
        set(type_key "")
        string(REPLACE " " "_" user_type_clean "${user_type}")
        string(REPLACE "::" "__" user_type_clean "${user_type_clean}")
        
        if(DEFINED type_mapping_${user_type_clean})
            set(type_key "${type_mapping_${user_type_clean}}")
        else()
            string(TOUPPER "${user_type}" upper_type)
            set(type_key "${upper_type}")
        endif()

        if(NOT type_key)
            continue()
        endif()

        # Find the type definition in types.h
        string(REGEX MATCH "#define[ \t]+TTYPE_${type_key}[ \t]*,[ \t]*\\(([^)]+)\\)" type_match "${types_content}")
        if(type_match)
            list(APPEND discovered_types "${type_key}")
            list(APPEND discovered_indices ${type_index})

            string(REGEX MATCH "\\(([^)]+)\\)" tuple_match "${type_match}")
            string(SUBSTRING "${tuple_match}" 1 -1 type_tuple)
            string(REGEX REPLACE "^([^,]+),[ \t]*(.+)$" "\\1;\\2" tuple_parts "${type_tuple}")
            list(GET tuple_parts 0 enum_part)
            list(GET tuple_parts 1 cpp_part)
            string(STRIP "${enum_part}" enum_part)
            string(STRIP "${cpp_part}" cpp_part)
            string(REGEX REPLACE "\\)$" "" cpp_part "${cpp_part}")

            list(APPEND discovered_enums "${enum_part}")
            list(APPEND discovered_cpp_types "${cpp_part}")

            math(EXPR type_index "${type_index} + 1")
        endif()
    endforeach()

    if(type_index EQUAL 0)
        message(FATAL_ERROR "No valid types discovered from profile types: ${types_to_discover}")
    endif()

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()


function(_internal_srcore_discover_all_types result_indices_var result_names_var result_enums_var result_cpp_types_var)
    # types.h is always at include/types/types.h
    set(types_header "${CMAKE_CURRENT_SOURCE_DIR}/include/types/types.h")

    if(NOT EXISTS "${types_header}")
        message(FATAL_ERROR "Could not find types.h at expected location: ${types_header}")
    endif()

    file(READ "${types_header}" types_content)

    # FORCE usage of profile types - DO NOT discover extra types!
    if(DEFINED SD_TYPE_PROFILE AND NOT SD_TYPE_PROFILE STREQUAL "")
        # Get the types from the profile - these are C++ type names
        if(COMMAND get_profile_type_combinations)
            get_profile_type_combinations("${SD_TYPE_PROFILE}" profile_types)
        else()
            message(FATAL_ERROR "get_profile_type_combinations function not found!")
        endif()
        
        message(STATUS "FORCING profile '${SD_TYPE_PROFILE}' types: ${profile_types}")
        
        # Convert C++ type names to enum names for discovery
        set(all_types "")
        foreach(cpp_type ${profile_types})
            # Map C++ types to enum names
            if(cpp_type STREQUAL "bool")
                list(APPEND all_types "BOOL")
            elseif(cpp_type STREQUAL "int8_t")
                list(APPEND all_types "INT8")
            elseif(cpp_type STREQUAL "uint8_t")
                list(APPEND all_types "UINT8")
            elseif(cpp_type STREQUAL "int16_t")
                list(APPEND all_types "INT16")
            elseif(cpp_type STREQUAL "uint16_t")
                list(APPEND all_types "UINT16")
            elseif(cpp_type STREQUAL "int32_t")
                list(APPEND all_types "INT32")
            elseif(cpp_type STREQUAL "uint32_t")
                list(APPEND all_types "UINT32")
            elseif(cpp_type STREQUAL "int64_t")
                list(APPEND all_types "INT64")
            elseif(cpp_type STREQUAL "uint64_t")
                list(APPEND all_types "UINT64")
            elseif(cpp_type STREQUAL "float16")
                list(APPEND all_types "HALF")
            elseif(cpp_type STREQUAL "bfloat16")
                list(APPEND all_types "BFLOAT16")
            elseif(cpp_type STREQUAL "float")
                list(APPEND all_types "FLOAT32")
            elseif(cpp_type STREQUAL "double")
                list(APPEND all_types "DOUBLE")
            elseif(cpp_type STREQUAL "std::string")
                list(APPEND all_types "UTF8")
            elseif(cpp_type STREQUAL "std::u16string")
                list(APPEND all_types "UTF16")
            elseif(cpp_type STREQUAL "std::u32string")
                list(APPEND all_types "UTF32")
            else()
                message(WARNING "Unknown C++ type in profile: ${cpp_type}")
            endif()
        endforeach()
        
        list(LENGTH all_types mapped_count)
        list(LENGTH profile_types original_count)
        message(STATUS "Profile specified ${original_count} types, mapped to ${mapped_count} enum types")
        
        if(mapped_count EQUAL 0)
            message(FATAL_ERROR "No valid type mappings found for profile '${SD_TYPE_PROFILE}'")
        endif()
    else()
        # No profile - this should not happen if STANDARD_ALL_TYPES is set
        message(FATAL_ERROR "No SD_TYPE_PROFILE defined! Set SD_TYPE_PROFILE to use specific types.")
    endif()

    # Now discover ONLY the types that are in the profile
    set(discovered_types "")
    set(discovered_indices "")
    set(discovered_enums "")
    set(discovered_cpp_types "")
    set(type_index 0)

    foreach(type_key ${all_types})
        string(REGEX MATCH "#define[ \t]+TTYPE_${type_key}[ \t]*,[ \t]*\\(([^)]+)\\)" type_match "${types_content}")
        if(type_match)
            list(APPEND discovered_types "${type_key}")
            list(APPEND discovered_indices ${type_index})

            string(REGEX MATCH "\\(([^)]+)\\)" tuple_match "${type_match}")
            string(SUBSTRING "${tuple_match}" 1 -1 type_tuple)
            string(REGEX REPLACE "^([^,]+),[ \t]*(.+)$" "\\1;\\2" tuple_parts "${type_tuple}")
            list(GET tuple_parts 0 enum_part)
            list(GET tuple_parts 1 cpp_part)
            string(STRIP "${enum_part}" enum_part)
            string(STRIP "${cpp_part}" cpp_part)
            string(REGEX REPLACE "\\)$" "" cpp_part "${cpp_part}")

            list(APPEND discovered_enums "${enum_part}")
            list(APPEND discovered_cpp_types "${cpp_part}")

            math(EXPR type_index "${type_index} + 1")
        else()
            message(WARNING "Type '${type_key}' specified in profile but not found in types.h")
        endif()
    endforeach()

    if(type_index EQUAL 0)
        message(FATAL_ERROR "No types discovered from types.h using profile '${SD_TYPE_PROFILE}'")
    endif()

    # Verify we got the expected number
    list(LENGTH all_types expected_count)
    if(NOT type_index EQUAL expected_count)
        message(WARNING "Profile specifies ${expected_count} types but only ${type_index} were found in types.h")
    endif()

    message(STATUS "✅ Discovered ${type_index} types from profile '${SD_TYPE_PROFILE}'")
    message(STATUS "Types: ${discovered_types}")

    set(${result_indices_var} "${discovered_indices}" PARENT_SCOPE)
    set(${result_names_var} "${discovered_types}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()
# ============================================================================
# SECTION 4: PUBLIC API FUNCTIONS
# ============================================================================
function(srcore_discover_active_types result_var result_enums_var result_cpp_types_var)
    _internal_srcore_discover_types(active_indices active_names discovered_enums discovered_cpp_types)
    set(SRCORE_ACTIVE_TYPES "${active_names}" PARENT_SCOPE)
    list(LENGTH active_indices type_count)
    set(SRCORE_ACTIVE_TYPE_COUNT ${type_count} PARENT_SCOPE)

    # Store type mappings for later use
    set(type_index 0)
    foreach(type_enum IN LISTS discovered_enums)
        set(SRCORE_TYPE_ENUM_${type_index} "${type_enum}" PARENT_SCOPE)
        math(EXPR type_index "${type_index} + 1")
    endforeach()

    set(type_index 0)
    foreach(type_cpp IN LISTS discovered_cpp_types)
        set(SRCORE_TYPE_CPP_${type_index} "${type_cpp}" PARENT_SCOPE)
        math(EXPR type_index "${type_index} + 1")
    endforeach()

    set(type_index 0)
    foreach(type_name IN LISTS active_names)
        set(SRCORE_TYPE_NAME_${type_index} "${type_name}" PARENT_SCOPE)
        math(EXPR type_index "${type_index} + 1")
    endforeach()

    set(${result_var} "${active_indices}" PARENT_SCOPE)
    set(${result_enums_var} "${discovered_enums}" PARENT_SCOPE)
    set(${result_cpp_types_var} "${discovered_cpp_types}" PARENT_SCOPE)
endfunction()

function(srcore_generate_combinations active_indices profile result_2_var result_3_var)
    _internal_srcore_generate_combinations("${active_indices}" "${SRCORE_ACTIVE_TYPES}" "${profile}" combinations_2 combinations_3)
    set(SRCORE_COMBINATIONS_2 "${combinations_2}" PARENT_SCOPE)
    set(SRCORE_COMBINATIONS_3 "${combinations_3}" PARENT_SCOPE)
    set(${result_2_var} "${combinations_2}" PARENT_SCOPE)
    set(${result_3_var} "${combinations_3}" PARENT_SCOPE)
endfunction()

function(srcore_generate_headers active_indices combinations_2 combinations_3 output_dir type_enums type_cpp_types)
    # Generate the base validity header
    _internal_srcore_generate_validity_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")

    message(STATUS "Generated BUILD_ macro overrides: ${override_header_file}")

    # Also enhance the main selective_rendering.h with runtime dispatch
    srcore_generate_enhanced_header("${active_indices}" "${combinations_2}" "${combinations_3}" "${output_dir}" "${type_enums}" "${type_cpp_types}")
endfunction()

function(srcore_validate_output active_indices combinations_2 combinations_3)
    list(LENGTH active_indices type_count)
    list(LENGTH combinations_2 combo_2_count)
    list(LENGTH combinations_3 combo_3_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types found")
    endif()
    if(combo_2_count EQUAL 0)
        message(FATAL_ERROR "No 2-type combinations generated")
    endif()
    if(combo_3_count EQUAL 0)
        message(FATAL_ERROR "No 3-type combinations generated")
    endif()
endfunction()

function(srcore_emergency_fallback)
    set(UNIFIED_ACTIVE_TYPES "float;double;int32_t;bool" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_2 "0,0;0,1;1,0;1,1;2,2;3,3" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_3 "0,0,0;1,1,1;2,2,2;3,3,3" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT 4 PARENT_SCOPE)
        message(WARNING "Using emergency fallback type configuration")
endfunction()

function(srcore_auto_setup)
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        setup_selective_rendering_unified_safe()
    endif()
endfunction()

function(_internal_srcore_generate_validity_header active_indices type_enums type_cpp_types combinations_2 combinations_3 output_dir)
    file(MAKE_DIRECTORY "${output_dir}/system")
    set(header_file "${output_dir}/system/selective_rendering.h")

    # Helper function to convert enum value to integer
    function(enum_to_int_value enum_value output_var)
    string(REGEX REPLACE ".*::" "" datatype_name "${enum_value}")
    
    # Match exact DataType enum values from the provided header
    if(datatype_name STREQUAL "INHERIT")
        set(int_value "0")
    elseif(datatype_name STREQUAL "BOOL")
        set(int_value "1")
    elseif(datatype_name STREQUAL "FLOAT8")
        set(int_value "2")
    elseif(datatype_name STREQUAL "HALF")
        set(int_value "3")
    elseif(datatype_name STREQUAL "HALF2")
        set(int_value "4")
    elseif(datatype_name STREQUAL "FLOAT32")
        set(int_value "5")
    elseif(datatype_name STREQUAL "DOUBLE")
        set(int_value "6")
    elseif(datatype_name STREQUAL "INT8")
        set(int_value "7")
    elseif(datatype_name STREQUAL "INT16")
        set(int_value "8")
    elseif(datatype_name STREQUAL "INT32")
        set(int_value "9")
    elseif(datatype_name STREQUAL "INT64")
        set(int_value "10")
    elseif(datatype_name STREQUAL "UINT8")
        set(int_value "11")
    elseif(datatype_name STREQUAL "UINT16")
        set(int_value "12")
    elseif(datatype_name STREQUAL "UINT32")
        set(int_value "13")
    elseif(datatype_name STREQUAL "UINT64")
        set(int_value "14")
    elseif(datatype_name STREQUAL "QINT8")
        set(int_value "15")
    elseif(datatype_name STREQUAL "QINT16")
        set(int_value "16")
    elseif(datatype_name STREQUAL "BFLOAT16")
        set(int_value "17")
    elseif(datatype_name STREQUAL "UTF8")
        set(int_value "50")
    elseif(datatype_name STREQUAL "UTF16")
        set(int_value "51")
    elseif(datatype_name STREQUAL "UTF32")
        set(int_value "52")
    elseif(datatype_name STREQUAL "ANY")
        set(int_value "100")
    elseif(datatype_name STREQUAL "AUTO")
        set(int_value "200")
    elseif(datatype_name STREQUAL "UNKNOWN")
        set(int_value "255")
    else()
        set(int_value "255")  # Default to UNKNOWN
            message(WARNING "Unknown DataType enum value: ${datatype_name}")
    endif()
    set(${output_var} "${int_value}" PARENT_SCOPE)
endfunction()

    # Start building the header content
    set(header_content "/* AUTOMATICALLY GENERATED - Selective Rendering Header */\n")
    string(APPEND header_content "/* Generated by SelectiveRenderingCore.cmake */\n")
    string(APPEND header_content "#ifndef SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_SELECTIVE_RENDERING_H\n\n")

    # ============================================================================
    # SECTION 1: RAW COMPILATION FLAGS
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 1: RAW COMPILATION FLAGS\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Collect all compiled type numbers
    list(LENGTH type_enums num_types)
    set(compiled_type_numbers "")

    foreach(i RANGE 0 ${num_types})
        if(i LESS ${num_types})
            list(GET type_enums ${i} enum_value)
            enum_to_int_value("${enum_value}" int_value)
            list(FIND compiled_type_numbers "${int_value}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_value}")
            endif()
        endif()
    endforeach()

    # Generate single type compilation flags for ALL possible types
    string(APPEND header_content "// Single type compilation flags\n")
    set(all_possible_types "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;50;51;52;100;200;255")
    foreach(type_num IN LISTS all_possible_types)
        list(FIND compiled_type_numbers "${type_num}" found_idx)
        if(found_idx GREATER_EQUAL 0)
            string(APPEND header_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED 1\n")
        else()
            string(APPEND header_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED 0\n")
        endif()
    endforeach()
    string(APPEND header_content "\n")

    # Generate pair type compilation flags
    string(APPEND header_content "// Pair type compilation flags\n")
    set(all_pair_keys "")

    # Collect all valid pairs from combinations_2
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        if(i LESS ${num_types} AND j LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            list(GET type_enums ${j} enum_j)
            enum_to_int_value("${enum_i}" int_i)
            enum_to_int_value("${enum_j}" int_j)
            set(pair_key "${int_i}_${int_j}")
            list(FIND all_pair_keys "${pair_key}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND all_pair_keys "${pair_key}")
            endif()
        endif()
    endforeach()

    # Generate all pair combinations
    foreach(type1 IN LISTS all_possible_types)
        foreach(type2 IN LISTS all_possible_types)
            set(pair_key "${type1}_${type2}")
            list(FIND all_pair_keys "${pair_key}" found_idx)
            if(found_idx GREATER_EQUAL 0)
                string(APPEND header_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED 1\n")
            else()
                string(APPEND header_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED 0\n")
            endif()
        endforeach()
    endforeach()
    string(APPEND header_content "\n")

    # Generate triple type compilation flags
    string(APPEND header_content "// Triple type compilation flags\n")
    set(all_triple_keys "")

    # Collect all valid triples from combinations_3
    foreach(combo IN LISTS combinations_3)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        list(GET parts 2 k)
        if(i LESS ${num_types} AND j LESS ${num_types} AND k LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            list(GET type_enums ${j} enum_j)
            list(GET type_enums ${k} enum_k)
            enum_to_int_value("${enum_i}" int_i)
            enum_to_int_value("${enum_j}" int_j)
            enum_to_int_value("${enum_k}" int_k)
            set(triple_key "${int_i}_${int_j}_${int_k}")
            list(FIND all_triple_keys "${triple_key}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND all_triple_keys "${triple_key}")
            endif()
        endif()
    endforeach()

    # Generate all triple combinations
    foreach(type1 IN LISTS all_possible_types)
        foreach(type2 IN LISTS all_possible_types)
            foreach(type3 IN LISTS all_possible_types)
                set(triple_key "${type1}_${type2}_${type3}")
                list(FIND all_triple_keys "${triple_key}" found_idx)
                if(found_idx GREATER_EQUAL 0)
                    string(APPEND header_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED 1\n")
                else()
                    string(APPEND header_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED 0\n")
                endif()
            endforeach()
        endforeach()
    endforeach()
    string(APPEND header_content "\n")

   # ============================================================================
    # SECTION 2: MAPPING TABLES - COMPLETE VERSION
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 2: MAPPING TABLES\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Generate enum to number mappings - Complete DataType enum coverage
    string(APPEND header_content "// DataType enum to number mappings (with namespace handling)\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INHERIT 0\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_BOOL 1\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_FLOAT8 2\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_HALF 3\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_HALF2 4\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_FLOAT32 5\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_DOUBLE 6\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT8 7\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT16 8\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT32 9\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_INT64 10\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT8 11\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT16 12\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT32 13\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UINT64 14\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_QINT8 15\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_QINT16 16\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_BFLOAT16 17\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UTF8 50\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UTF16 51\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UTF32 52\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_ANY 100\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_AUTO 200\n")
    string(APPEND header_content "#define SD_ENUM_TO_NUM_UNKNOWN 255\n")
    string(APPEND header_content "\n")

    # Generate alias to number mappings
    string(APPEND header_content "// Constexpr alias to number mappings\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INHERIT 0\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_BOOL 1\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_FLOAT8 2\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_HALF 3\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_HALF2 4\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_FLOAT32 5\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_DOUBLE 6\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT8 7\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT16 8\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT32 9\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_INT64 10\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT8 11\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT16 12\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT32 13\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UINT64 14\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_QINT8 15\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_QINT16 16\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_BFLOAT16 17\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UTF8 50\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UTF16 51\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UTF32 52\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_ANY 100\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_AUTO 200\n")
    string(APPEND header_content "#define SD_ALIAS_TO_NUM_UNKNOWN 255\n")
    string(APPEND header_content "\n")

    # Generate C++ type to number mappings - COMPREHENSIVE LIST
    string(APPEND header_content "// C++ type name to number mappings\n")
    
    # Build a comprehensive type mapping dictionary
    set(type_mappings "")
    list(APPEND type_mappings "bool:1")
    list(APPEND type_mappings "float8:2")
    list(APPEND type_mappings "float16:3")
    list(APPEND type_mappings "half:3")
    list(APPEND type_mappings "half2:4")
    list(APPEND type_mappings "float:5")
    list(APPEND type_mappings "float32:5")
    list(APPEND type_mappings "double:6")
    list(APPEND type_mappings "int8_t:7")
    list(APPEND type_mappings "int16_t:8")
    list(APPEND type_mappings "int32_t:9")
    list(APPEND type_mappings "Int32Type:9")
    list(APPEND type_mappings "int:9")
    list(APPEND type_mappings "int64_t:10")
    list(APPEND type_mappings "LongType:10")
    list(APPEND type_mappings "long:10")
    list(APPEND type_mappings "long_long:10")
    list(APPEND type_mappings "uint8_t:11")
    list(APPEND type_mappings "unsigned_char:11")
    list(APPEND type_mappings "uint16_t:12")
    list(APPEND type_mappings "unsigned_short:12")
    list(APPEND type_mappings "uint32_t:13")
    list(APPEND type_mappings "unsigned_int:13")
    list(APPEND type_mappings "uint64_t:14")
    list(APPEND type_mappings "UnsignedLong:14")
    list(APPEND type_mappings "unsigned_long:14")
    list(APPEND type_mappings "qint8:15")
    list(APPEND type_mappings "qint16:16")
    list(APPEND type_mappings "bfloat16:17")
    list(APPEND type_mappings "bfloat:17")
    list(APPEND type_mappings "stdstring:50")
    list(APPEND type_mappings "std_string:50")
    list(APPEND type_mappings "utf8:50")
    list(APPEND type_mappings "u16string:51")
    list(APPEND type_mappings "std_u16string:51")
    list(APPEND type_mappings "utf16:51")
    list(APPEND type_mappings "u32string:52")
    list(APPEND type_mappings "std_u32string:52")
    list(APPEND type_mappings "utf32:52")
    list(APPEND type_mappings "SignedChar:7")
    list(APPEND type_mappings "UnsignedChar:11")
    list(APPEND type_mappings "signed_char:7")
    list(APPEND type_mappings "unsigned_char:11")
    
    # Generate all type to number mappings
    foreach(mapping IN LISTS type_mappings)
        string(REPLACE ":" ";" mapping_parts "${mapping}")
        list(GET mapping_parts 0 type_name)
        list(GET mapping_parts 1 type_num)
        string(APPEND header_content "#define SD_TYPE_TO_NUM_${type_name} ${type_num}\n")
    endforeach()
    string(APPEND header_content "\n")

    # ============================================================================
    # SECTION 3: CONDITIONAL COMPILATION MACROS
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 3: CONDITIONAL COMPILATION MACROS\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Generate SD_IF macros for all type combinations
    string(APPEND header_content "// Conditional compilation macros for type combinations\n")
    string(APPEND header_content "// These handle both direct numeric and SD_TYPE_TO_NUM_ prefixed patterns\n\n")
    
    # Extract unique type names from the mappings
    set(all_type_names "")
    foreach(mapping IN LISTS type_mappings)
        string(REPLACE ":" ";" mapping_parts "${mapping}")
        list(GET mapping_parts 0 type_name)
        list(APPEND all_type_names "${type_name}")
    endforeach()
    
    # Generate pair macros for all type name combinations
    string(APPEND header_content "// Pair type conditional compilation macros\n")
    foreach(type1_name IN LISTS all_type_names)
        # Get numeric value for type1
        set(type1_num "")
        foreach(mapping IN LISTS type_mappings)
            string(REPLACE ":" ";" mapping_parts "${mapping}")
            list(GET mapping_parts 0 mapped_name)
            list(GET mapping_parts 1 mapped_num)
            if(mapped_name STREQUAL type1_name)
                set(type1_num "${mapped_num}")
                break()
            endif()
        endforeach()
        
        if(NOT type1_num STREQUAL "")
            foreach(type2_num IN LISTS all_possible_types)
                set(pair_key "${type1_num}_${type2_num}")
                
                # Generate the macro for SD_TYPE_TO_NUM_ prefix pattern
                string(APPEND header_content "#define SD_IF_SD_PAIR_TYPE_SD_TYPE_TO_NUM_${type1_name}_${type2_num}_COMPILED(code)")
                list(FIND all_pair_keys "${pair_key}" found_idx)
                if(found_idx GREATER_EQUAL 0)
                    string(APPEND header_content " code\n")
                else()
                    string(APPEND header_content " do {} while(0)\n")
                endif()
                
                # Also generate without the prefix for direct use
                string(APPEND header_content "#define SD_IF_SD_PAIR_TYPE_${type1_name}_${type2_num}_COMPILED(code)")
                if(found_idx GREATER_EQUAL 0)
                    string(APPEND header_content " code\n")
                else()
                    string(APPEND header_content " do {} while(0)\n")
                endif()
            endforeach()
        endif()
    endforeach()
    string(APPEND header_content "\n")
    
    # Generate triple macros for all type name combinations
    string(APPEND header_content "// Triple type conditional compilation macros\n")
    foreach(type1_name IN LISTS all_type_names)
        # Get numeric value for type1
        set(type1_num "")
        foreach(mapping IN LISTS type_mappings)
            string(REPLACE ":" ";" mapping_parts "${mapping}")
            list(GET mapping_parts 0 mapped_name)
            list(GET mapping_parts 1 mapped_num)
            if(mapped_name STREQUAL type1_name)
                set(type1_num "${mapped_num}")
                break()
            endif()
        endforeach()
        
        if(NOT type1_num STREQUAL "")
            foreach(type2_num IN LISTS all_possible_types)
                foreach(type3_num IN LISTS all_possible_types)
                    set(triple_key "${type1_num}_${type2_num}_${type3_num}")
                    
                    # Generate the macro for SD_TYPE_TO_NUM_ prefix pattern
                    string(APPEND header_content "#define SD_IF_SD_TRIPLE_TYPE_SD_TYPE_TO_NUM_${type1_name}_${type2_num}_${type3_num}_COMPILED(code)")
                    list(FIND all_triple_keys "${triple_key}" found_idx)
                    if(found_idx GREATER_EQUAL 0)
                        string(APPEND header_content " code\n")
                    else()
                        string(APPEND header_content " do {} while(0)\n")
                    endif()
                endforeach()
            endforeach()
        endif()
    endforeach()
    string(APPEND header_content "\n")

    # Close the header guard
    string(APPEND header_content "#endif // SD_SELECTIVE_RENDERING_H\n")

    # Write the header file
    file(WRITE "${header_file}" "${header_content}")

    # Report generation results
        list(LENGTH all_triple_keys total_triple_combinations)
        list(LENGTH all_pair_keys total_pair_combinations)
        list(LENGTH compiled_type_numbers total_single_types)
        message(STATUS "Generated selective_rendering.h:")
        message(STATUS "  - Location: ${header_file}")
        message(STATUS "  - Single types: ${total_single_types}")
        message(STATUS "  - Pair combinations: ${total_pair_combinations}")
        message(STATUS "  - Triple combinations: ${total_triple_combinations}")
endfunction()

# ============================================================================
# NEW FUNCTION: Verify Type Compilation Status
# ============================================================================

function(verify_type_compilation_status)
    message(STATUS "")
    message(STATUS "=== Type Compilation Status ===")
    
    # Check active types count
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        list(LENGTH UNIFIED_ACTIVE_TYPES active_count)
        message(STATUS "Active types count: ${active_count}")
        message(STATUS "Active types: ${UNIFIED_ACTIVE_TYPES}")
    else()
        message(WARNING "UNIFIED_ACTIVE_TYPES not defined!")
    endif()
    
    # Check combinations count
    if(DEFINED UNIFIED_COMBINATIONS_2)
        list(LENGTH UNIFIED_COMBINATIONS_2 combo2_count)
        message(STATUS "2-type combinations: ${combo2_count}")
    endif()
    
    if(DEFINED UNIFIED_COMBINATIONS_3)
        list(LENGTH UNIFIED_COMBINATIONS_3 combo3_count)
        message(STATUS "3-type combinations: ${combo3_count}")
    endif()
    
    # Check generated header file exists and has content
    if(EXISTS "${CMAKE_BINARY_DIR}/include/system/selective_rendering.h")
        file(READ "${CMAKE_BINARY_DIR}/include/system/selective_rendering.h" header_content)
        string(LENGTH "${header_content}" header_size)
        message(STATUS "Generated header size: ${header_size} bytes")
        
        # Count how many types are actually compiled (have _COMPILED 1)
        string(REGEX MATCHALL "#define SD_SINGLE_TYPE_[0-9]+_COMPILED 1" compiled_singles "${header_content}")
        list(LENGTH compiled_singles compiled_count)
        message(STATUS "Compiled single types: ${compiled_count}")
    else()
        message(WARNING "selective_rendering.h not found at expected location!")
    endif()
    
    message(STATUS "==============================")
    message(STATUS "")
endfunction()

# ============================================================================
# SECTION 5: MAIN ORCHESTRATOR FUNCTIONS
# ============================================================================
function(setup_selective_rendering_unified)
    set(options "")
    set(one_value_args TYPE_PROFILE OUTPUT_DIR)
    set(multi_value_args "")
    cmake_parse_arguments(SRCORE "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT SRCORE_TYPE_PROFILE)
        set(SRCORE_TYPE_PROFILE "${SD_TYPE_PROFILE}")
    endif()
    if(NOT SRCORE_OUTPUT_DIR)
        set(SRCORE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")
    endif()

    srcore_discover_active_types(active_types_indices discovered_enums discovered_cpp_types)
    list(LENGTH active_types_indices type_count)

    if(type_count EQUAL 0)
        message(FATAL_ERROR "No active types discovered!")
    endif()

    srcore_generate_combinations("${active_types_indices}" "${SRCORE_TYPE_PROFILE}" combinations_2 combinations_3)
    srcore_generate_headers("${active_types_indices}" "${combinations_2}" "${combinations_3}" "${SRCORE_OUTPUT_DIR}" "${discovered_enums}" "${discovered_cpp_types}")

    set(UNIFIED_COMBINATIONS_2 "${combinations_2}" PARENT_SCOPE)
    set(UNIFIED_COMBINATIONS_3 "${combinations_3}" PARENT_SCOPE)
    set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" PARENT_SCOPE)
    set(UNIFIED_TYPE_COUNT ${type_count} PARENT_SCOPE)

    set(UNIFIED_COMBINATIONS_2 "${combinations_2}" CACHE INTERNAL "Unified 2-type combinations")
    set(UNIFIED_COMBINATIONS_3 "${combinations_3}" CACHE INTERNAL "Unified 3-type combinations")
    set(UNIFIED_ACTIVE_TYPES "${SRCORE_ACTIVE_TYPES}" CACHE INTERNAL "Active types for build")
    set(UNIFIED_TYPE_COUNT ${type_count} CACHE INTERNAL "Unified active type count")
    
    _internal_ensure_diagnostics_output()
    
    report_selective_rendering_statistics()
endfunction()

function(setup_selective_rendering_unified_safe)
    if(NOT CMAKE_CROSSCOMPILING AND NOT ANDROID)
        setup_selective_rendering_unified(${ARGN})
        srcore_map_to_legacy_variables()
  
        srcore_generate_diagnostic_report()
        
        _internal_ensure_diagnostics_output()
    else()
        setup_selective_rendering_unified(${ARGN})
        srcore_map_to_legacy_variables()
        
        _internal_ensure_diagnostics_output()
    endif()

    # Propagate variables to parent scope
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(UNIFIED_COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(UNIFIED_COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(UNIFIED_ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_TYPE_COUNT)
        set(UNIFIED_TYPE_COUNT "${UNIFIED_TYPE_COUNT}" PARENT_SCOPE)
    endif()

    # Final verification
    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        srcore_emergency_fallback()
        srcore_map_to_legacy_variables()
        
        # FIX: Ensure diagnostics even for fallback
        _internal_ensure_diagnostics_output()

        if(DEFINED UNIFIED_COMBINATIONS_3)
            set(UNIFIED_COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_COMBINATIONS_2)
            set(UNIFIED_COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_ACTIVE_TYPES)
            set(UNIFIED_ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        endif()
        if(DEFINED UNIFIED_TYPE_COUNT)
            set(UNIFIED_TYPE_COUNT "${UNIFIED_TYPE_COUNT}" PARENT_SCOPE)
        endif()
    endif()

    if(NOT DEFINED UNIFIED_COMBINATIONS_3 OR NOT UNIFIED_COMBINATIONS_3)
        message(FATAL_ERROR "Unable to establish UNIFIED_COMBINATIONS_3 even with emergency fallback!")
    endif()
    
    # FIX: Add final check for diagnostics status
    check_diagnostics_status()
endfunction()


function(_internal_ensure_diagnostics_output)
    # Always create the diagnostics output directory
    set(COMBINATION_REPORT_DIR "${CMAKE_BINARY_DIR}/type_combinations")
    file(MAKE_DIRECTORY "${COMBINATION_REPORT_DIR}")
    
    # Set as cache variable for other functions to use
    set(SD_DIAGNOSTICS_DIR "${COMBINATION_REPORT_DIR}" CACHE INTERNAL "Type combinations diagnostics directory")
    
    # Create a timestamp file to mark when the directory was created
    string(TIMESTAMP creation_time "%Y-%m-%d %H:%M:%S")
    file(WRITE "${COMBINATION_REPORT_DIR}/.created" "Directory created: ${creation_time}\n")
    
    # FIX: Add explicit message about creating directory
    message(STATUS "Creating type combinations diagnostics directory: ${COMBINATION_REPORT_DIR}")
    
    # Automatically dump combinations if they exist
    if(DEFINED UNIFIED_COMBINATIONS_2 OR DEFINED UNIFIED_COMBINATIONS_3)
        # Silently dump combinations to disk without console output
        _internal_quiet_dump_combinations("${COMBINATION_REPORT_DIR}")
        
        # FIX: Also call the main dump function for full reports
        dump_type_combinations_to_disk("${COMBINATION_REPORT_DIR}")
        
        message(STATUS "Type combination reports written to: ${COMBINATION_REPORT_DIR}")
    else()
        message(STATUS "No combinations available yet for diagnostic output")
    endif()
endfunction()

function(generate_selective_rendering_reports)
    message(STATUS "")
    message(STATUS "=== Generating Selective Rendering Reports ===")
    
    # Ensure diagnostics directory exists
    set(COMBINATION_REPORT_DIR "${CMAKE_BINARY_DIR}/type_combinations")
    file(MAKE_DIRECTORY "${COMBINATION_REPORT_DIR}")
    
    # Check if we have the necessary data
    if(NOT DEFINED UNIFIED_COMBINATIONS_2 AND NOT DEFINED UNIFIED_COMBINATIONS_3)
        message(WARNING "No type combinations available. Run setup_selective_rendering_unified_safe() first.")
        return()
    endif()
    
    # Generate all reports
    dump_type_combinations_to_disk("${COMBINATION_REPORT_DIR}")
    report_selective_rendering_statistics()
    
    # Verify files were created
    file(GLOB report_files "${COMBINATION_REPORT_DIR}/*")
    list(LENGTH report_files num_files)
    
    if(num_files GREATER 0)
        message(STATUS "Successfully generated ${num_files} report files in: ${COMBINATION_REPORT_DIR}")
        
        # List the files
        foreach(report_file ${report_files})
            get_filename_component(filename "${report_file}" NAME)
            message(STATUS "  - ${filename}")
        endforeach()
    else()
        message(WARNING "No report files were generated!")
    endif()
    
    message(STATUS "===============================================")
    message(STATUS "")
endfunction()

function(srcore_map_to_legacy_variables)
    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" CACHE INTERNAL "Legacy 2-type combinations")
    endif()

    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" CACHE INTERNAL "Legacy 3-type combinations")
    endif()

    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" CACHE INTERNAL "Legacy active types")

        list(LENGTH UNIFIED_ACTIVE_TYPES legacy_count)
        set(SD_COMMON_TYPES_COUNT ${legacy_count} PARENT_SCOPE)
        set(SD_COMMON_TYPES_COUNT ${legacy_count} CACHE INTERNAL "Legacy type count")

        set(type_index 0)
        foreach(type_name ${UNIFIED_ACTIVE_TYPES})
            set(TYPE_NAME_${type_index} "${type_name}" PARENT_SCOPE)
            set(TYPE_NAME_${type_index} "${type_name}" CACHE INTERNAL "Legacy reverse type lookup")
            math(EXPR type_index "${type_index} + 1")
        endforeach()
    endif()
endfunction()

function(srcore_generate_diagnostic_report)
    if(NOT SD_ENABLE_DIAGNOSTICS)
        return()
    endif()

    set(report_file "${CMAKE_BINARY_DIR}/selective_rendering_diagnostic_report.txt")
    set(report_content "")

    string(TIMESTAMP current_time "%Y-%m-%d %H:%M:%S")
    string(APPEND report_content "SelectiveRenderingCore Diagnostic Report\n")
    string(APPEND report_content "Generated: ${current_time}\n")
    string(APPEND report_content "========================================\n\n")

    string(APPEND report_content "Configuration:\n")
    string(APPEND report_content "- SD_ENABLE_SEMANTIC_FILTERING: ${SD_ENABLE_SEMANTIC_FILTERING}\n")
    string(APPEND report_content "- SD_TYPE_PROFILE: ${SD_TYPE_PROFILE}\n")
    string(APPEND report_content "- SD_SELECTIVE_TYPES: ${SD_SELECTIVE_TYPES}\n")
    string(APPEND report_content "\n")

    # Active types
    if(DEFINED SRCORE_ACTIVE_TYPES)
        list(LENGTH SRCORE_ACTIVE_TYPES type_count)
        string(APPEND report_content "Active Types (${type_count}):\n")
        set(index 0)
        foreach(type_name ${SRCORE_ACTIVE_TYPES})
            string(APPEND report_content "  [${index}] ${type_name}\n")
            math(EXPR index "${index} + 1")
        endforeach()
        string(APPEND report_content "\n")
    endif()

    # Combination statistics
    if(DEFINED SRCORE_COMBINATIONS_2 AND DEFINED SRCORE_COMBINATIONS_3)
        list(LENGTH SRCORE_COMBINATIONS_2 count_2)
        list(LENGTH SRCORE_COMBINATIONS_3 count_3)
        string(APPEND report_content "Combination Statistics:\n")
        string(APPEND report_content "- 2-type combinations: ${count_2}\n")
        string(APPEND report_content "- 3-type combinations: ${count_3}\n")

        if(DEFINED SRCORE_ACTIVE_TYPE_COUNT)
            math(EXPR total_possible "${SRCORE_ACTIVE_TYPE_COUNT} * ${SRCORE_ACTIVE_TYPE_COUNT} * ${SRCORE_ACTIVE_TYPE_COUNT}")
            if(total_possible GREATER 0)
                math(EXPR usage_percent "100 * ${count_3} / ${total_possible}")
                math(EXPR savings_percent "100 - ${usage_percent}")
                string(APPEND report_content "- Template usage: ${usage_percent}% (${savings_percent}% saved)\n")
            endif()
        endif()
        string(APPEND report_content "\n")
    endif()

    # Sample combinations with type names
    if(DEFINED SRCORE_COMBINATIONS_2)
        string(APPEND report_content "Sample 2-type combinations (first 10):\n")
        set(sample_count 0)
        foreach(combo ${SRCORE_COMBINATIONS_2})
            if(sample_count GREATER_EQUAL 10)
                break()
            endif()

            string(REPLACE "," ";" combo_parts "${combo}")
            list(GET combo_parts 0 i)
            list(GET combo_parts 1 j)

            if(DEFINED SRCORE_TYPE_NAME_${i} AND DEFINED SRCORE_TYPE_NAME_${j})
                string(APPEND report_content "  (${SRCORE_TYPE_NAME_${i}}, ${SRCORE_TYPE_NAME_${j}}) -> (${i},${j})\n")
            else()
                string(APPEND report_content "  (${i},${j})\n")
            endif()

            math(EXPR sample_count "${sample_count} + 1")
        endforeach()
        string(APPEND report_content "\n")
    endif()

    if(DEFINED SRCORE_COMBINATIONS_3)
        string(APPEND report_content "Sample 3-type combinations (first 10):\n")
        set(sample_count 0)
        foreach(combo ${SRCORE_COMBINATIONS_3})
            if(sample_count GREATER_EQUAL 10)
                break()
            endif()

            string(REPLACE "," ";" combo_parts "${combo}")
            list(GET combo_parts 0 i)
            list(GET combo_parts 1 j)
            list(GET combo_parts 2 k)

            if(DEFINED SRCORE_TYPE_NAME_${i} AND DEFINED SRCORE_TYPE_NAME_${j} AND DEFINED SRCORE_TYPE_NAME_${k})
                string(APPEND report_content "  (${SRCORE_TYPE_NAME_${i}}, ${SRCORE_TYPE_NAME_${j}}, ${SRCORE_TYPE_NAME_${k}}) -> (${i},${j},${k})\n")
            else()
                string(APPEND report_content "  (${i},${j},${k})\n")
            endif()

            math(EXPR sample_count "${sample_count} + 1")
        endforeach()
        string(APPEND report_content "\n")
    endif()

    # Write validation rules summary
    string(APPEND report_content "Validation Rules Applied:\n")
    string(APPEND report_content "- Numeric type pairings allowed\n")
    string(APPEND report_content "- Bool can pair with any numeric type\n")
    string(APPEND report_content "- Float types can pair together\n")
    string(APPEND report_content "- Integer types can pair together\n")
    string(APPEND report_content "- Specific int-to-float promotions allowed\n")
    string(APPEND report_content "- Triple output type must be >= input types (except bool)\n")
    string(APPEND report_content "\n")

    file(WRITE "${report_file}" "${report_content}")
    message(STATUS "Diagnostic report written to: ${report_file}")
endfunction()

# ============================================================================
# OPTIMIZED WRAPPER FUNCTIONS (Production Ready)
# ============================================================================

# Main wrapper for existing code
function(setup_selective_rendering)
    setup_selective_rendering_unified_safe()

    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(ACTIVE_TYPES "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()
endfunction()

# Legacy wrapper functions (now no-ops for performance)
function(track_combination_states active_types combinations_3)
    # Handled internally - no action needed
endfunction()

function(generate_selective_rendering_header)
    # Handled internally - no action needed
endfunction()

function(generate_selective_wrapper_header)
    # Handled internally - no action needed
endfunction()

function(setup_definitive_semantic_filtering_with_selective_rendering)
    set(SD_ENABLE_SEMANTIC_FILTERING TRUE PARENT_SCOPE)
    set(SD_ENABLE_SELECTIVE_RENDERING TRUE PARENT_SCOPE)

    setup_selective_rendering_unified_safe()

    if(DEFINED UNIFIED_COMBINATIONS_2)
        set(COMBINATIONS_2 "${UNIFIED_COMBINATIONS_2}" PARENT_SCOPE)
    endif()
    if(DEFINED UNIFIED_COMBINATIONS_3)
        set(COMBINATIONS_3 "${UNIFIED_COMBINATIONS_3}" PARENT_SCOPE)
    endif()
endfunction()

function(enhanced_semantic_filtering_setup)
    setup_definitive_semantic_filtering_with_selective_rendering()
endfunction()

function(setup_definitive_semantic_filtering)
    set(SD_ENABLE_SEMANTIC_FILTERING TRUE PARENT_SCOPE)
    setup_selective_rendering_unified_safe(TYPE_PROFILE "${SD_TYPE_PROFILE}")

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
endfunction()

function(initialize_definitive_combinations)
    setup_definitive_semantic_filtering()
endfunction()

function(extract_definitive_types result_var)
    if(DEFINED UNIFIED_ACTIVE_TYPES)
        set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    else()
        srcore_auto_setup()
        set(${result_var} "${UNIFIED_ACTIVE_TYPES}" PARENT_SCOPE)
    endif()
endfunction()

function(generate_definitive_combinations active_types result_2_var result_3_var)
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
    # Handled internally - no action needed
endfunction()

# ============================================================================
# PRODUCTION-OPTIMIZED SEMANTIC ENGINE INTEGRATION
# ============================================================================

# Simplified version without debug overhead
function(setup_enhanced_semantic_validation)
    # Core validation logic without debug output
    if(SD_ENABLE_SEMANTIC_FILTERING)
        if(NOT SD_TYPE_PROFILE OR SD_TYPE_PROFILE STREQUAL "")
            if(SD_TYPES_LIST_COUNT GREATER 0)
                set(detected_profile "")
                if("int8_t" IN_LIST SD_TYPES_LIST AND "uint8_t" IN_LIST SD_TYPES_LIST)
                    set(detected_profile "quantization")
                elseif("float16" IN_LIST SD_TYPES_LIST OR "bfloat16" IN_LIST SD_TYPES_LIST)
                    set(detected_profile "training")
                elseif(SD_TYPES_LIST MATCHES ".*string.*")
                    set(detected_profile "nlp")
                endif()

                if(NOT detected_profile STREQUAL "")
                    set(SD_TYPE_PROFILE "${detected_profile}" PARENT_SCOPE)
                else()
                    set(SD_TYPE_PROFILE "inference" PARENT_SCOPE)
                endif()
            else()
                set(SD_TYPE_PROFILE "inference" PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()

# Remove debug function calls to avoid GCC function tracing overhead
macro(print_status_colored level message)
    # Only output if diagnostics are explicitly enabled
        message(STATUS "${message}")
endmacro()

function(_internal_srcore_generate_helper_macros output_var)
    set(helper_content "")

    string(APPEND helper_content "#define SD_BUILD_TRIPLE_IF_VALID(t1, t2, t3, build_macro) \\\n")
    string(APPEND helper_content "    do { \\\n")
    string(APPEND helper_content "        if (SD_IS_TRIPLE_TYPE_COMPILED(t1, t2, t3)) { \\\n")
    string(APPEND helper_content "            SD_DISPATCH_TRIPLE_RUNTIME(t1, t2, t3, build_macro); \\\n")
    string(APPEND helper_content "        } \\\n")
    string(APPEND helper_content "    } while(0)\n\n")

    string(APPEND helper_content "#define SD_BUILD_PAIR_IF_VALID(t1, t2, build_macro) \\\n")
    string(APPEND helper_content "    do { \\\n")
    string(APPEND helper_content "        if (SD_IS_PAIR_TYPE_COMPILED(t1, t2)) { \\\n")
    string(APPEND helper_content "            SD_DISPATCH_PAIR_RUNTIME(t1, t2, build_macro); \\\n")
    string(APPEND helper_content "        } \\\n")
    string(APPEND helper_content "    } while(0)\n\n")

    string(APPEND helper_content "#define SD_BUILD_SINGLE_IF_VALID(t1, build_macro) \\\n")
    string(APPEND helper_content "    do { \\\n")
    string(APPEND helper_content "        if (SD_IS_SINGLE_TYPE_COMPILED(t1)) { \\\n")
    string(APPEND helper_content "            SD_DISPATCH_SINGLE_RUNTIME(t1, build_macro); \\\n")
    string(APPEND helper_content "        } \\\n")
    string(APPEND helper_content "    } while(0)\n\n")

    set(${output_var} "${helper_content}" PARENT_SCOPE)
endfunction()

function(srcore_generate_enhanced_header active_indices combinations_2 combinations_3 output_dir type_enums type_cpp_types)
    _internal_srcore_generate_validity_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")
    _internal_srcore_append_runtime_dispatch_to_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")
endfunction()

function(_internal_srcore_append_runtime_dispatch_to_header active_indices type_enums type_cpp_types combinations_2 combinations_3 output_dir)
    set(header_file "${output_dir}/system/selective_rendering.h")

    _internal_srcore_generate_helper_macros(helper_macros)

    if(EXISTS "${header_file}")
        file(READ "${header_file}" existing_content)

        string(REGEX REPLACE "\n#endif // SD_SELECTIVE_RENDERING_H\n?$" "" content_without_endif "${existing_content}")

        set(new_content "${content_without_endif}")
        string(APPEND new_content "\n${dispatch_macros}")
        string(APPEND new_content "${helper_macros}")
        string(APPEND new_content "#endif // SD_SELECTIVE_RENDERING_H\n")

        file(WRITE "${header_file}" "${new_content}")

            list(LENGTH combinations_3 total_triple_combinations)
            list(LENGTH combinations_2 total_pair_combinations)
            message(STATUS "Enhanced selective_rendering.h with runtime dispatch - ${total_pair_combinations} pair dispatches, ${total_triple_combinations} triple dispatches")
    else()
        message(FATAL_ERROR "Cannot append runtime dispatch - header file does not exist: ${header_file}")
    endif()
endfunction()

# ADD this new function to automatically create diagnostics during setup
function(_internal_ensure_diagnostics_output)
    # Always create the diagnostics output directory
    set(COMBINATION_REPORT_DIR "${CMAKE_BINARY_DIR}/type_combinations")
    file(MAKE_DIRECTORY "${COMBINATION_REPORT_DIR}")
    
    # Set as cache variable for other functions to use
    set(SD_DIAGNOSTICS_DIR "${COMBINATION_REPORT_DIR}" CACHE INTERNAL "Type combinations diagnostics directory")
    
    # Create a timestamp file to mark when the directory was created
    string(TIMESTAMP creation_time "%Y-%m-%d %H:%M:%S")
    file(WRITE "${COMBINATION_REPORT_DIR}/.created" "Directory created: ${creation_time}\n")
    
    # Automatically dump combinations if they exist
    if(DEFINED UNIFIED_COMBINATIONS_2 OR DEFINED UNIFIED_COMBINATIONS_3)
        # Silently dump combinations to disk without console output
        _internal_quiet_dump_combinations("${COMBINATION_REPORT_DIR}")
    endif()
endfunction()

function(check_diagnostics_status)
    if(EXISTS "${CMAKE_BINARY_DIR}/type_combinations")
        file(GLOB diagnostic_files "${CMAKE_BINARY_DIR}/type_combinations/*")
        list(LENGTH diagnostic_files num_files)
        if(num_files GREATER 0)
            message(STATUS "Type combination diagnostics available: ${CMAKE_BINARY_DIR}/type_combinations (${num_files} files)")
            return()
        endif()
    endif()
    message(STATUS "Type combination diagnostics not yet generated")
endfunction()

function(srcore_enable_runtime_dispatch)
    set(SD_ENABLE_RUNTIME_DISPATCH TRUE PARENT_SCOPE)
    set(SD_ENABLE_RUNTIME_DISPATCH TRUE CACHE BOOL "Enable runtime dispatch macro generation")

        message(STATUS "Runtime dispatch enabled - will generate SD_DISPATCH_*_RUNTIME macros")
endfunction()