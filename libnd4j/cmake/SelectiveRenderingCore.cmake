# ============================================================================
# SelectiveRenderingCore.cmake (v20 - Production Optimized) - FIXED
#
# Optimized version with debug profiles removed for production builds.
# Conditional diagnostics only when explicitly enabled via SD_ENABLE_DIAGNOSTICS.
# ============================================================================

# Include the reporting functions
include(SelectiveRenderingReports)

# ============================================================================
# HELPER FUNCTION: Write file only if content changed (preserves mtime for PCH)
# ============================================================================
function(_srcore_write_if_different filepath content)
    set(should_write TRUE)

    # Check if file already exists
    if(EXISTS "${filepath}")
        # Read existing content
        file(READ "${filepath}" existing_content)

        # Compare content
        if("${existing_content}" STREQUAL "${content}")
            set(should_write FALSE)
        endif()
    endif()

    # Only write if content changed or file doesn't exist
    if(should_write)
        file(WRITE "${filepath}" "${content}")
    endif()
endfunction()

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
    # 2-TYPE COMBINATION FILTERING FOR FUNCTRACE BUILDS
    #
    # Background: Functrace builds create ~3.3GB binaries (vs ~200MB normal) due to
    # instrumentation overhead on every template instantiation. With 196 2-type
    # combinations (14×14 with no filtering), the binary exceeds 2GB relocation limit.
    #
    # Solution: Filter rare/exotic type conversions while preserving commonly used ones.
    # This reduces binary size without impacting model execution.
    #
    # Filtering Strategy:
    # 1. KEEP: Same-type operations (X→X)
    # 2. KEEP: Common numeric conversions (float↔double, int32↔int64)
    # 3. KEEP: Numeric↔bool conversions (for comparisons)
    # 4. FILTER: Rare integer cross-conversions (uint16↔uint32, int16↔uint64)
    # 5. FILTER: Exotic combinations (bool↔float16, uint8↔bfloat16)
    # 6. FILTER: String type conversions (rarely used in hot paths)

    # Rule 1: Same-type operations always valid
    if(type1 STREQUAL type2)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Get type categories
    _internal_srcore_is_type_floating("${type1}" t1_is_float)
    _internal_srcore_is_type_floating("${type2}" t2_is_float)
    _internal_srcore_is_type_integer("${type1}" t1_is_int)
    _internal_srcore_is_type_integer("${type2}" t2_is_int)

    set(t1_is_bool FALSE)
    set(t2_is_bool FALSE)
    if(type1 STREQUAL "BOOL")
        set(t1_is_bool TRUE)
    endif()
    if(type2 STREQUAL "BOOL")
        set(t2_is_bool TRUE)
    endif()

    # Check if types are "rare" (uint16, uint32, uint64, int16)
    set(rare_types "UINT16;UINT32;UINT64;INT16")
    list(FIND rare_types "${type1}" t1_rare_idx)
    list(FIND rare_types "${type2}" t2_rare_idx)
    set(t1_is_rare FALSE)
    set(t2_is_rare FALSE)
    if(t1_rare_idx GREATER_EQUAL 0)
        set(t1_is_rare TRUE)
    endif()
    if(t2_rare_idx GREATER_EQUAL 0)
        set(t2_is_rare TRUE)
    endif()

    # Check if types are exotic floats (float16, bfloat16)
    set(exotic_float_types "HALF;BFLOAT16")
    list(FIND exotic_float_types "${type1}" t1_exotic_idx)
    list(FIND exotic_float_types "${type2}" t2_exotic_idx)
    set(t1_is_exotic FALSE)
    set(t2_is_exotic FALSE)
    if(t1_exotic_idx GREATER_EQUAL 0)
        set(t1_is_exotic TRUE)
    endif()
    if(t2_exotic_idx GREATER_EQUAL 0)
        set(t2_is_exotic TRUE)
    endif()

    # Rule 2: Filter string type conversions (rarely used)
    if(type1 MATCHES "UTF8" OR type2 MATCHES "UTF8")
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Filter rare×rare cross-type conversions
    # Examples: uint16↔uint32, int16↔uint64
    # These are rarely used in ML workloads
    if(t1_is_rare AND t2_is_rare)
        # Both are rare types and already filtered out same-type by Rule 1
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 4: DISABLED - bool↔exotic float conversions ARE needed
    # Session #320 filtered these but they're required by NativeOpExecutioner
    # Examples that failed linking: ScalarTransform<bfloat16, float16, bool>
    # if((t1_is_bool AND t2_is_exotic) OR (t1_is_exotic AND t2_is_bool))
    #     set(${output_var} FALSE PARENT_SCOPE)
    #     return()
    # endif()

    # Rule 5: DISABLED - exotic float×rare integer combinations ARE needed
    # Session #320 filtered these but they're required by PairWiseTransform
    # Examples that failed linking: PairWiseTransform<bfloat16, *, short/uint16/uint64>
    # if((t1_is_exotic AND t2_is_rare) OR (t1_is_rare AND t2_is_exotic))
    #     set(${output_var} FALSE PARENT_SCOPE)
    #     return()
    # endif()

    # Rule 6: DISABLED - exotic float cross-conversions ARE needed
    # Session #320 filtered these but they're required by NativeOpExecutioner
    # Examples that failed linking: PairWiseTransform<bfloat16, *, float16>
    # if(t1_is_exotic AND t2_is_exotic)
    #     # Both exotic and different types (same-type filtered by Rule 1)
    #     set(${output_var} FALSE PARENT_SCOPE)
    #     return()
    # endif()

    # All other combinations are valid:
    # - Common numeric conversions (float↔double, int32↔int64)
    # - Numeric↔bool (comparisons and masks)
    # - Integer promotions (int8→int32, uint8→uint32)
    # - Rare types with common types (uint16↔int32, int16↔float)
    # - Exotic floats with common types (float16↔float, bfloat16↔double)
    set(${output_var} TRUE PARENT_SCOPE)
endfunction()


# SelectiveRenderingCore.cmake - Triple Validation Function
function(_internal_srcore_is_valid_triple type1 type2 type3 output_var)
    # SEMANTIC FILTERING: Filter invalid type combinations while preserving all valid ones
    #
    # Previous approach (accept ALL) generated 2,197 combinations including many invalid ones:
    # - bool × bool → float (arithmetic on bools producing floats doesn't make sense)
    # - Excessive uint16/32/64 combinations (59% of combinations, rarely used)
    #
    # Filtering strategy (incremental):
    # - Phase 1: Filter bool × bool → numeric (13 combinations)
    # - Phase 2: Filter rare × rare → rare (cross-type) (64 combinations)
    #
    # Valid combination patterns:
    # 1. Same-type operations: (X, X, X)
    # 2. Type promotion within category: (smaller, larger, larger)
    # 3. Comparison operations: (any, any, bool)
    # 4. Type conversion/casting: (any, any, any) with restrictions
    # 5. Bool masking: (bool, numeric, numeric)
    # 6. Rare types with common types: (rare, common, any) and (common, rare, any)

    # Get type categories
    _internal_srcore_is_type_floating("${type1}" t1_is_float)
    _internal_srcore_is_type_floating("${type2}" t2_is_float)
    _internal_srcore_is_type_floating("${type3}" t3_is_float)

    _internal_srcore_is_type_integer("${type1}" t1_is_int)
    _internal_srcore_is_type_integer("${type2}" t2_is_int)
    _internal_srcore_is_type_integer("${type3}" t3_is_int)

    set(t1_is_bool FALSE)
    set(t2_is_bool FALSE)
    set(t3_is_bool FALSE)
    if(type1 STREQUAL "BOOL")
        set(t1_is_bool TRUE)
    endif()
    if(type2 STREQUAL "BOOL")
        set(t2_is_bool TRUE)
    endif()
    if(type3 STREQUAL "BOOL")
        set(t3_is_bool TRUE)
    endif()

    # Check if types are rare types (rarely used in ML workloads)
    # Rare types: UINT16, UINT32, UINT64, INT16
    set(rare_types "UINT16;UINT32;UINT64;INT16")

    set(t1_is_rare FALSE)
    set(t2_is_rare FALSE)
    set(t3_is_rare FALSE)

    list(FIND rare_types "${type1}" t1_rare_idx)
    if(t1_rare_idx GREATER_EQUAL 0)
        set(t1_is_rare TRUE)
    endif()

    list(FIND rare_types "${type2}" t2_rare_idx)
    if(t2_rare_idx GREATER_EQUAL 0)
        set(t2_is_rare TRUE)
    endif()

    list(FIND rare_types "${type3}" t3_rare_idx)
    if(t3_rare_idx GREATER_EQUAL 0)
        set(t3_is_rare TRUE)
    endif()

    # Rule 1: Same-type operations are always valid (X, X, X)
    if(type1 STREQUAL type2 AND type2 STREQUAL type3)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Comparison operations (any, any, bool) are usually valid
    # This covers ==, !=, <, >, <=, >= operations
    # However, filter cross-rare-type comparisons (both inputs different rare types)
    if(t3_is_bool)
        # If comparing two different rare types, filter it
        # Examples: (UINT16, UINT32, bool), (INT16, UINT64, bool)
        # These cross-rare-type comparisons are genuinely rare in practice
        if(t1_is_rare AND t2_is_rare AND NOT type1 STREQUAL type2)
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
        # All other comparisons are valid
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Filter INVALID bool arithmetic combinations
    # bool × bool → numeric doesn't make semantic sense (what is bool + bool → float?)
    if(t1_is_bool AND t2_is_bool AND NOT t3_is_bool)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 4: Filter RARE × RARE cross-type combinations
    # Strategy 4: Conservative rare-type filtering (Phase 3 optimization)
    #
    # Rare types (UINT16, UINT32, UINT64, INT16) are rarely used in ML workloads:
    # - Actual code usage: ~88 lines across entire codebase
    # - UINT32/UINT64 not in test suite
    # - 54% of instantiations use these types, but <1% of operations code
    #
    # Filter combinations where INPUTS are different rare types:
    #   - uint32 × int16 → int32 (different rare inputs)
    #   - uint16 × uint32 → float32 (different rare inputs)
    #   - int16 × uint64 → bool (different rare inputs, even though output is bool)
    #
    # Preserve:
    #   - Same-type rare operations: uint32 × uint32 → uint32 (already handled by Rule 1)
    #   - Same rare inputs: uint32 × uint32 → int32 (same rare inputs, different output OK)
    #   - Rare with common types: uint32 × int32 → any (one input is common)
    #   - Common with rare: int32 × uint16 → any (one input is common)
    #
    # This filtering eliminates ~180 additional combinations with no semantic meaning,
    # saving ~2.7GB total and achieving ~7% reduction with minimal risk.
    #
    # IMPORTANT: This must come BEFORE the permissive category-based rules (Rule 5, 6)
    # to prevent them from accepting these cross-rare-type combinations.

    # Check if both inputs are rare types
    if(t1_is_rare AND t2_is_rare)
        # Both inputs are rare types
        # Only allow if they're the SAME rare type (regardless of output)
        if(NOT type1 STREQUAL type2)
            # Different rare types in inputs - filter this combination
            # Examples: uint32 × int16 → any, uint16 × uint32 → any
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()

        # Strategy 5: Filter rare×rare→rare (same inputs, rare output)
        # Even when inputs are the same rare type, outputting another rare type is uncommon
        # This typically indicates unnecessary type conversions
        # Examples to filter:
        #   - uint32 × uint32 → uint16 (rare inputs → rare output)
        #   - uint32 × uint32 → uint64 (rare inputs → rare output)
        #   - int16 × int16 → int8 (rare inputs → rare output)
        # Examples to KEEP:
        #   - uint32 × uint32 → uint32 (same type operation - handled by Rule 1)
        #   - uint32 × uint32 → int32 (rare → common output)
        #   - uint32 × uint32 → float32 (rare → common output)
        #   - uint32 × uint32 → bool (comparison - handled by Rule 2)
        if(t3_is_rare AND NOT type1 STREQUAL type3)
            # Rare inputs, different rare output - filter
            # Keep only if output equals input type (Rule 1 handles this)
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
        # If type1 == type2 == type3 (all same), Rule 1 already allowed it
        # If type1 == type2 and type3 is common, continue to other rules
    endif()

    # Rule 4b: Filter float-rare type mixing (except comparisons which output bool)
    # Float arrays combined with rare integer types are uncommon in ML workloads
    # This rule filters combinations like:
    #   - (float, uint16, uint16) - float with rare int, rare int output
    #   - (uint16, float, uint16) - rare int with float, rare int output
    #   - (double, int16, int16) - float with rare int, rare int output
    # But preserves:
    #   - (float, uint16, bool) - comparison operations (already handled by Rule 2)
    #   - (float, uint16, float) - output matches float input (handled by Rule 7)
    #
    # This eliminates ~50-80 combinations with semantically unusual patterns
    if(NOT t3_is_bool)
        if((t1_is_float AND t2_is_rare) OR (t1_is_rare AND t2_is_float))
            # One input is float, other is rare int, output is NOT bool
            # Check if output is the rare type - this pattern is semantically unusual
            if((t1_is_rare AND type1 STREQUAL type3) OR (t2_is_rare AND type2 STREQUAL type3))
                # Output matches the rare type - filter this unusual pattern
                # Examples: (float, uint16, uint16), (uint16, double, uint16)
                set(${output_var} FALSE PARENT_SCOPE)
                return()
            endif()
            # If output matches float input, allow it (will be caught by Rule 7)
        endif()
    endif()

    # Rule 5: Bool masking operations - RESTRICTED to same-type masking
    # These support operations like: bool_mask ? float_x : float_y -> float_result
    # AGGRESSIVE FILTER: Only allow when output type matches the numeric input type
    # - Allow: (bool, float, float) - mask ? float : float -> float
    # - Allow: (float, bool, float) - float ? mask : float -> float
    # - Filter: (bool, int8, float) - nonsensical cross-type masking
    # - Filter: (bool, float, double) - unnecessary precision change
    if(t1_is_bool AND NOT t2_is_bool AND NOT t3_is_bool)
        # (bool, numeric, output) - only allow if output matches numeric
        if(type2 STREQUAL type3)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        else()
            # Cross-type bool masking - filter it
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(NOT t1_is_bool AND t2_is_bool AND NOT t3_is_bool)
        # (numeric, bool, output) - only allow if output matches numeric
        if(type1 STREQUAL type3)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        else()
            # Cross-type bool masking - filter it
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 6: Type promotion - inputs match, output different (X, X, Y)
    # Examples: float+float→double (accumulation), int32+int32→int64 (overflow protection)
    if(type1 STREQUAL type2 AND NOT type2 STREQUAL type3)
        # Both inputs same type, output different
        # This is valid for type promotion operations
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 7: Output matches one input (preserves type)
    # Examples: float+int→float, int+float→float, float+double→double
    if(type1 STREQUAL type3 OR type2 STREQUAL type3)
        # Output matches at least one input
        # This is valid for operations that preserve one input's type
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 8: Filter ALL other three-way type mixing
    # If we reach here, all three types are different AND none of the above rules matched
    # Examples of what we filter:
    # - (bfloat16, double, float) - three different float precisions
    # - (int8, float, int32) - completely unrelated types
    # - (float16, int32, double) - random type mixing
    #
    # Combined filtering summary:
    # - Rule 8 base: ~912 combinations (51.3%)
    # - Rule 2 refinement: ~20-30 cross-rare-type comparisons (~1-2%)
    # - Rule 4b float-rare filtering: ~50-80 combinations (~2-3%)
    # Total: ~980-1,020 combinations filtered (~55-58% reduction)
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# Scalar-specific validation function
# ScalarTransform<ArrayType, ScalarType, OutputType>
# Semantics: array op scalar -> output
function(_internal_srcore_is_valid_scalar_triple array_type scalar_type output_type output_var)
    # Get type categories for array
    _internal_srcore_is_type_floating("${array_type}" arr_is_float)
    _internal_srcore_is_type_integer("${array_type}" arr_is_int)
    set(arr_is_bool FALSE)
    if(array_type STREQUAL "BOOL")
        set(arr_is_bool TRUE)
    endif()

    # Get type categories for scalar
    _internal_srcore_is_type_floating("${scalar_type}" scal_is_float)
    _internal_srcore_is_type_integer("${scalar_type}" scal_is_int)
    set(scal_is_bool FALSE)
    if(scalar_type STREQUAL "BOOL")
        set(scal_is_bool TRUE)
    endif()

    # Get type categories for output
    _internal_srcore_is_type_floating("${output_type}" out_is_float)
    _internal_srcore_is_type_integer("${output_type}" out_is_int)
    set(out_is_bool FALSE)
    if(output_type STREQUAL "BOOL")
        set(out_is_bool TRUE)
    endif()

    # Rule 1: Same type throughout (X, X, X) - always valid
    if(array_type STREQUAL scalar_type AND scalar_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Comparison operations (X, Y, bool) - always valid
    if(out_is_bool)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Scalar type promotion (X, Y, Y) - output matches scalar type
    # Common pattern: float_array + double_scalar -> double_output
    if(scalar_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4: Array type preservation (X, Y, X) - output matches array type
    # Less common but valid: double_array + float_scalar -> double_output
    if(array_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4b: Bitwise operations with integer types
    # Pattern: (IntType, IntType, DifferentIntType) where array and scalar match
    # Common for bit manipulation: rotate/shift operations that may produce different size outputs
    # Example: (UINT64, UINT64, UINT32) for rotate operations
    if(arr_is_int AND scal_is_int AND out_is_int AND array_type STREQUAL scalar_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4c: Float type conversion operations
    # Pattern: (FloatX, FloatX, FloatY) where array and scalar match but output is different float type
    # Common for type conversions: (float, float, double), (bfloat16, bfloat16, float), etc.
    # Example: ScalarTransform<float, float, double> for accumulation with higher precision
    if(arr_is_float AND scal_is_float AND out_is_float AND array_type STREQUAL scalar_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4d: Float-to-integer type casting operations
    # Pattern: (FloatX, FloatX, IntY) where array and scalar match (both float) but output is integer
    # Common for casting operations: (float, float, int32), (double, double, int64), etc.
    # Example: ScalarTransform<bfloat16, bfloat16, Int32Type> for cast-to-int operations
    if(arr_is_float AND scal_is_float AND out_is_int AND array_type STREQUAL scalar_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4e: Integer-to-float type casting operations
    # Pattern: (IntX, IntX, FloatY) where array and scalar match (both integer) but output is float
    # Common for casting operations: (int32, int32, float), (int64, int64, double), etc.
    # Example: ScalarTransform<Int32Type, Int32Type, float> for cast-to-float operations
    if(arr_is_int AND scal_is_int AND out_is_float AND array_type STREQUAL scalar_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 5: Filter ALL other 3-way type mixing
    # Examples of what we filter:
    # - (float, uint32, int8) - completely nonsensical
    # - (double, bfloat16, float16) - random type conversions
    # - (int32, float, int8) - none of the types match
    #
    # This eliminates ~1,200 nonsensical combinations (67% reduction)
    # Note: After adding Rules 4d and 4e, the reduction is lower (~60%) due to allowing cross-type casting
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# Broadcast-specific validation function
# Broadcast<Array1Type, Array2Type, OutputType>
# Semantics: array1 op array2 -> output
function(_internal_srcore_is_valid_broadcast_triple array1_type array2_type output_type output_var)
    # Get type categories for array1
    _internal_srcore_is_type_floating("${array1_type}" arr1_is_float)
    _internal_srcore_is_type_integer("${array1_type}" arr1_is_int)
    set(arr1_is_bool FALSE)
    if(array1_type STREQUAL "BOOL")
        set(arr1_is_bool TRUE)
    endif()

    # Get type categories for array2
    _internal_srcore_is_type_floating("${array2_type}" arr2_is_float)
    _internal_srcore_is_type_integer("${array2_type}" arr2_is_int)
    set(arr2_is_bool FALSE)
    if(array2_type STREQUAL "BOOL")
        set(arr2_is_bool TRUE)
    endif()

    # Get type categories for output
    _internal_srcore_is_type_floating("${output_type}" out_is_float)
    _internal_srcore_is_type_integer("${output_type}" out_is_int)
    set(out_is_bool FALSE)
    if(output_type STREQUAL "BOOL")
        set(out_is_bool TRUE)
    endif()

    # Rule 1: Same type throughout (X, X, X) - always valid
    if(array1_type STREQUAL array2_type AND array2_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Comparison operations (X, Y, bool) - always valid
    if(out_is_bool)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2b: Bool masking operations - RESTRICTED to same-type masking
    # These support operations like: bool_array ? numeric_x : numeric_y -> numeric_result
    # AGGRESSIVE FILTER: Only allow when output type matches the numeric input type
    # - Allow: (bool, float, float) - bool_mask broadcast numeric -> numeric
    # - Allow: (float, bool, float) - numeric broadcast bool_mask -> numeric
    # - Filter: (bool, int8, float) - nonsensical cross-type masking
    # - Filter: (bool, float, double) - unnecessary precision change
    if(arr1_is_bool AND NOT arr2_is_bool AND NOT out_is_bool)
        # (bool, numeric, output) - only allow if output matches numeric array
        if(array2_type STREQUAL output_type)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        else()
            # Cross-type bool masking - filter it
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    if(NOT arr1_is_bool AND arr2_is_bool AND NOT out_is_bool)
        # (numeric, bool, output) - only allow if output matches numeric array
        if(array1_type STREQUAL output_type)
            set(${output_var} TRUE PARENT_SCOPE)
            return()
        else()
            # Cross-type bool masking - filter it
            set(${output_var} FALSE PARENT_SCOPE)
            return()
        endif()
    endif()

    # Rule 3: Type promotion (X, X, Y) - both inputs same, output is promoted type
    # Common pattern: int32_array + int32_array -> int64_output
    if(array1_type STREQUAL array2_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4: Output matches first input (X, Y, X)
    # Common in broadcasts: keep first array's type
    if(array1_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 5: Output matches second input (X, Y, Y)
    # Common in broadcasts: keep second array's type
    if(array2_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 6: Filter ALL other 3-way type mixing
    # Examples of what we filter:
    # - (float, uint32, int8) - all three different, completely nonsensical
    # - (double, bfloat16, float16) - random type conversions
    # - (int32, float, int64) - none of the types match
    #
    # Combined with Rule 2b bool masking restrictions:
    # This eliminates ~1,300-1,350 nonsensical combinations (~73-76% reduction)
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# ============================================================================
# REDUCE3 VALIDATION (Distance/Similarity Operations)
# ============================================================================
# Reduce3<InputType, OutputType> - operations like cosine similarity, euclidean distance
# InputType: the array elements being compared/measured
# OutputType: the result type (typically float for precision)
#
# Semantic rules:
# - Output should be same type as input OR a float type (for precision)
# - Float input → int output is nonsensical (losing precision on distance)
# - Cross-type int combinations rarely make sense (e.g., int8 → uint32)

function(_internal_srcore_is_valid_reduce3_pair input_type output_type output_var)
    # Get type categories
    _internal_srcore_is_type_floating("${input_type}" input_is_float)
    _internal_srcore_is_type_floating("${output_type}" output_is_float)
    _internal_srcore_is_type_integer("${input_type}" input_is_int)
    _internal_srcore_is_type_integer("${output_type}" output_is_int)

    # Rule 1: Same type is always valid (preserves precision)
    if(input_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Integer input → float output is valid (precision preserved)
    if(input_is_int AND output_is_float)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Float input → float output is valid (allow ALL float-to-float conversions)
    # Changed from precision-preserving only to support all cross-type float operations
    # Needed for operations like: Reduce3<double, float>, Reduce3<float, bfloat16>, etc.
    if(input_is_float AND output_is_float)
        # Allow all float-to-float type pairs (including precision loss)
        # The operation semantics may require flexibility in output type
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 4: Filter all other combinations
    # - Float → int (losing precision on distance measure)
    # - Cross-type int combinations (e.g., int8 → uint32)
    # - Lower precision float output (e.g., double → float16)
    set(${output_var} FALSE PARENT_SCOPE)
endfunction()

# ============================================================================
# INDEXREDUCE VALIDATION (ArgMax/ArgMin Operations)
# ============================================================================
# IndexReduce<InputType, IndexType> - find indices of max/min values
# InputType: the array elements being searched
# IndexType: MUST be int64_t (indices are always 64-bit)
#
# Semantic rules:
# - IndexType should ALWAYS be INT64 (sd::LongType)
# - Any other index type is nonsensical (indices must be 64-bit signed)

function(_internal_srcore_is_valid_indexreduce_pair input_type index_type output_var)
    # IndexType MUST be INT64
    if(index_type STREQUAL "INT64")
        set(${output_var} TRUE PARENT_SCOPE)
    else()
        # Filter all non-INT64 index types
        set(${output_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ============================================================================
# REDUCE_FLOAT VALIDATION (Reductions with Float Output)
# ============================================================================
# ReduceFloatFunction<InputType, OutputType> - sum, mean, variance, std
# InputType: any numeric type
# OutputType: MUST be float type (for precision)
#
# Semantic rules:
# - Output must be float type (already enforced in TemplateProcessing.cmake)
# - Output precision should be >= input precision
# - Float → lower precision float is nonsensical (e.g., double → float16)

function(_internal_srcore_is_valid_reduce_float_pair input_type output_type output_var)
    # Get type categories
    _internal_srcore_is_type_floating("${input_type}" input_is_float)
    _internal_srcore_is_type_floating("${output_type}" output_is_float)
    _internal_srcore_is_type_integer("${input_type}" input_is_int)

    # Output MUST be float (already checked in TemplateProcessing.cmake, but double-check)
    if(NOT output_is_float)
        set(${output_var} FALSE PARENT_SCOPE)
        return()
    endif()

    # Rule 1: Same type is always valid
    if(input_type STREQUAL output_type)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 2: Integer input → any float output is valid
    if(input_is_int)
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Rule 3: Float input → any float output (including precision changes)
    # NOTE: Originally filtered downcasts, but runtime shows these ARE needed
    # for operations that explicitly request different output precision.
    # Examples: ReduceFloatFunction<double, float>, <float, double>, etc.
    if(input_is_float)
        # Allow ALL float-to-float combinations (upcast, downcast, or same)
        # The operation implementation handles precision conversion correctly
        set(${output_var} TRUE PARENT_SCOPE)
        return()
    endif()

    # Filter all other combinations
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

    # MODIFIED: Profile-based limiting DISABLED for full matrix generation
    # Previous code limited combinations to 50-500 based on profile, causing
    # runtime undefined symbol errors. Now generating ALL combinations.
    #
    # if(DEFINED profile AND NOT profile STREQUAL "")
    #     ... profile limiting code disabled ...
    # endif()

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
    elseif(normalized_type STREQUAL "SignedChar")
        set(normalized_type "int8_t")
    elseif(normalized_type STREQUAL "UnsignedChar")
        set(normalized_type "uint8_t")
    elseif(normalized_type STREQUAL "Int16Type")
        set(normalized_type "int16_t")
    elseif(normalized_type STREQUAL "UInt16Type")
        set(normalized_type "uint16_t")
    elseif(normalized_type STREQUAL "Int32Type")
        set(normalized_type "int32_t")
    elseif(normalized_type STREQUAL "UInt32Type")
        set(normalized_type "uint32_t")
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

    # FIXED: When called with selective types, ALWAYS use the validated_types_list
    # The profile should only be used by _internal_srcore_discover_all_types()
    # This function is called when SRCORE_USE_SELECTIVE_TYPES=TRUE, meaning the
    # user explicitly provided a type list (e.g., via -Dlibnd4j.datatypes=...).
    # Using the profile here would override the user's explicit choice!
    set(types_to_discover "${validated_types_list}")

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
    set(type_mapping_Int16Type "INT16")
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

function(srcore_generate_javacpp_header combinations_2 combinations_3 output_dir type_cpp_types active_indices)
    # Generate JavaCPP-compatible header that declares only valid template instantiations
    # This ensures JavaCPP generates JNI bindings ONLY for combinations that CMake builds

    set(javacpp_header_file "${output_dir}/javacpp_instantiations.h")
    set(javacpp_content "")

    string(APPEND javacpp_content "/* AUTOMATICALLY GENERATED - JavaCPP Template Instantiation Declarations */\n")
    string(APPEND javacpp_content "/* This header limits JavaCPP to generate JNI bindings only for valid type combinations */\n")
    string(APPEND javacpp_content "/* Generated by SelectiveRenderingCore.cmake */\n\n")
    string(APPEND javacpp_content "#ifndef SD_JAVACPP_INSTANTIATIONS_H\n")
    string(APPEND javacpp_content "#define SD_JAVACPP_INSTANTIATIONS_H\n\n")

    # Include all transform headers
    string(APPEND javacpp_content "// Transform headers\n")
    string(APPEND javacpp_content "#include <loops/scalar_transform.h>\n")
    string(APPEND javacpp_content "#include <loops/scalar_bool.h>\n")
    string(APPEND javacpp_content "#include <loops/scalar_int.h>\n")
    string(APPEND javacpp_content "#include <loops/pairwise_transform.h>\n")
    string(APPEND javacpp_content "#include <loops/pairwise_bool.h>\n")
    string(APPEND javacpp_content "#include <loops/pairwise_int.h>\n")
    string(APPEND javacpp_content "#include <loops/broadcasting.h>\n")
    string(APPEND javacpp_content "#include <loops/broadcasting_bool.h>\n")
    string(APPEND javacpp_content "#include <loops/broadcasting_int.h>\n")
    string(APPEND javacpp_content "#include <loops/transform_any.h>\n")
    string(APPEND javacpp_content "#include <loops/transform_bool.h>\n")
    string(APPEND javacpp_content "#include <loops/transform_float.h>\n")
    string(APPEND javacpp_content "#include <loops/transform_same.h>\n")
    string(APPEND javacpp_content "#include <loops/transform_strict.h>\n")
    string(APPEND javacpp_content "#include <loops/reduce_float.h>\n")
    string(APPEND javacpp_content "#include <loops/reduce_same.h>\n")
    string(APPEND javacpp_content "#include <loops/reduce_bool.h>\n")
    string(APPEND javacpp_content "#include <loops/reduce_long.h>\n")
    string(APPEND javacpp_content "#include <loops/reduce3.h>\n")
    string(APPEND javacpp_content "#include <loops/indexreduce.h>\n")
    string(APPEND javacpp_content "#include <loops/summarystatsreduce.h>\n\n")

    string(APPEND javacpp_content "// Forward declarations of valid template instantiations\n")
    string(APPEND javacpp_content "// JavaCPP will only generate JNI bindings for these combinations\n\n")
    string(APPEND javacpp_content "namespace functions {\n\n")

    # Triple-type transforms (X,Y,Z)
    string(APPEND javacpp_content "// ===== TRIPLE-TYPE TRANSFORMS (X,Y,Z) =====\n\n")

    # ScalarTransform
    string(APPEND javacpp_content "namespace scalar {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class ScalarTransform<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace scalar\n\n")

    # ScalarBoolTransform
    string(APPEND javacpp_content "namespace scalar {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class ScalarBoolTransform<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace scalar\n\n")

    # ScalarIntTransform
    string(APPEND javacpp_content "namespace scalar {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class ScalarIntTransform<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace scalar\n\n")

    # PairWiseTransform
    string(APPEND javacpp_content "namespace pairwise_transforms {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class PairWiseTransform<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace pairwise_transforms\n\n")

    # PairWiseBoolTransform
    string(APPEND javacpp_content "namespace pairwise_transforms {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class PairWiseBoolTransform<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace pairwise_transforms\n\n")

    # PairWiseIntTransform
    string(APPEND javacpp_content "namespace pairwise_transforms {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class PairWiseIntTransform<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace pairwise_transforms\n\n")

    # Broadcast
    string(APPEND javacpp_content "namespace broadcast {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class Broadcast<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace broadcast\n\n")

    # BroadcastBool
    string(APPEND javacpp_content "namespace broadcast {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class BroadcastBool<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace broadcast\n\n")

    # BroadcastInt
    string(APPEND javacpp_content "namespace broadcast {\n")
    foreach(triple IN LISTS combinations_3)
        string(REPLACE "," ";" triple_list "${triple}")
        list(GET triple_list 0 t1)
        list(GET triple_list 1 t2)
        list(GET triple_list 2 t3)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        list(GET type_cpp_types ${t3} cpp_type3)
        string(APPEND javacpp_content "template class BroadcastInt<${cpp_type1}, ${cpp_type2}, ${cpp_type3}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace broadcast\n\n")

    # Pair-type transforms (X,Z)
    string(APPEND javacpp_content "// ===== PAIR-TYPE TRANSFORMS (X,Z) =====\n\n")

    foreach(pair IN LISTS combinations_2)
        string(REPLACE "," ";" pair_list "${pair}")
        list(GET pair_list 0 t1)
        list(GET pair_list 1 t2)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)

        string(APPEND javacpp_content "namespace transform {\n")
        string(APPEND javacpp_content "template class TransformAny<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class TransformBool<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class TransformFloat<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class TransformSame<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class TransformStrict<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "} // namespace transform\n\n")
    endforeach()

    # Reduce operations (X,Z)
    string(APPEND javacpp_content "// ===== REDUCE OPERATIONS (X,Z) =====\n\n")

    foreach(pair IN LISTS combinations_2)
        string(REPLACE "," ";" pair_list "${pair}")
        list(GET pair_list 0 t1)
        list(GET pair_list 1 t2)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)

        string(APPEND javacpp_content "namespace reduce {\n")
        string(APPEND javacpp_content "template class ReduceFloatFunction<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class ReduceSameFunction<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class ReduceBoolFunction<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "template class ReduceLongFunction<${cpp_type1}, ${cpp_type2}>;\n")
        string(APPEND javacpp_content "} // namespace reduce\n\n")
    endforeach()

    # Reduce3 (X,Z)
    string(APPEND javacpp_content "namespace reduce {\n")
    foreach(pair IN LISTS combinations_2)
        string(REPLACE "," ";" pair_list "${pair}")
        list(GET pair_list 0 t1)
        list(GET pair_list 1 t2)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        string(APPEND javacpp_content "template class Reduce3<${cpp_type1}, ${cpp_type2}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace reduce\n\n")

    # IndexReduce (X,Z)
    string(APPEND javacpp_content "namespace indexreduce {\n")
    foreach(pair IN LISTS combinations_2)
        string(REPLACE "," ";" pair_list "${pair}")
        list(GET pair_list 0 t1)
        list(GET pair_list 1 t2)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        string(APPEND javacpp_content "template class IndexReduce<${cpp_type1}, ${cpp_type2}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace indexreduce\n\n")

    # SummaryStatsReduce (X,Z)
    string(APPEND javacpp_content "namespace summarystats {\n")
    foreach(pair IN LISTS combinations_2)
        string(REPLACE "," ";" pair_list "${pair}")
        list(GET pair_list 0 t1)
        list(GET pair_list 1 t2)
        list(GET type_cpp_types ${t1} cpp_type1)
        list(GET type_cpp_types ${t2} cpp_type2)
        string(APPEND javacpp_content "template class SummaryStatsReduce<${cpp_type1}, ${cpp_type2}>;\n")
    endforeach()
    string(APPEND javacpp_content "} // namespace summarystats\n\n")

    # Single-type transforms (X)
    string(APPEND javacpp_content "// ===== SINGLE-TYPE OPERATIONS (X) =====\n\n")

    foreach(idx IN LISTS active_indices)
        list(GET type_cpp_types ${idx} cpp_type)
        string(APPEND javacpp_content "namespace random {\n")
        string(APPEND javacpp_content "template class RandomFunction<${cpp_type}>;\n")
        string(APPEND javacpp_content "} // namespace random\n\n")
    endforeach()

    string(APPEND javacpp_content "} // namespace functions\n\n")
    string(APPEND javacpp_content "#endif // SD_JAVACPP_INSTANTIATIONS_H\n")

    file(WRITE "${javacpp_header_file}" "${javacpp_content}")

    list(LENGTH combinations_2 total_pairs)
    list(LENGTH combinations_3 total_triples)
    list(LENGTH active_indices total_singles)
    message(STATUS "Generated JavaCPP instantiations header:")
    message(STATUS "  - File: ${javacpp_header_file}")
    message(STATUS "  - Single-type combinations: ${total_singles}")
    message(STATUS "  - Pair combinations: ${total_pairs}")
    message(STATUS "  - Triple combinations: ${total_triples}")
    message(STATUS "  - This ensures JavaCPP only generates JNI bindings for valid combinations")
endfunction()

function(srcore_generate_headers active_indices combinations_2 combinations_3 output_dir type_enums type_cpp_types)
    # Generate the base validity header
    _internal_srcore_generate_validity_header("${active_indices}" "${type_enums}" "${type_cpp_types}" "${combinations_2}" "${combinations_3}" "${output_dir}")

    message(STATUS "Generated BUILD_ macro overrides: ${override_header_file}")

    # Also enhance the main selective_rendering.h with runtime dispatch
    srcore_generate_enhanced_header("${active_indices}" "${combinations_2}" "${combinations_3}" "${output_dir}" "${type_enums}" "${type_cpp_types}")

    # Generate JavaCPP compatibility header with ALL transform types
    srcore_generate_javacpp_header("${combinations_2}" "${combinations_3}" "${output_dir}" "${type_cpp_types}" "${active_indices}")
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

    # Define type categories for partitioning
    set(bool_types "1")
    set(float_types "3;4;5;6")  # HALF, HALF2, FLOAT32, DOUBLE
    set(bfloat_types "17")
    set(int_types "7;8;9;10")   # INT8, INT16, INT32, INT64
    set(uint_types "11;12;13;14")  # UINT8, UINT16, UINT32, UINT64
    set(string_types "50;51;52")   # UTF8, UTF16, UTF32

    # Helper function to determine which category a type belongs to
    function(get_type_category type_num output_var)
        list(FIND bool_types "${type_num}" idx)
        if(idx GREATER_EQUAL 0)
            set(${output_var} "bool" PARENT_SCOPE)
            return()
        endif()
        list(FIND float_types "${type_num}" idx)
        if(idx GREATER_EQUAL 0)
            set(${output_var} "float" PARENT_SCOPE)
            return()
        endif()
        list(FIND bfloat_types "${type_num}" idx)
        if(idx GREATER_EQUAL 0)
            set(${output_var} "bfloat" PARENT_SCOPE)
            return()
        endif()
        list(FIND int_types "${type_num}" idx)
        if(idx GREATER_EQUAL 0)
            set(${output_var} "int" PARENT_SCOPE)
            return()
        endif()
        list(FIND uint_types "${type_num}" idx)
        if(idx GREATER_EQUAL 0)
            set(${output_var} "uint" PARENT_SCOPE)
            return()
        endif()
        list(FIND string_types "${type_num}" idx)
        if(idx GREATER_EQUAL 0)
            set(${output_var} "string" PARENT_SCOPE)
            return()
        endif()
        set(${output_var} "other" PARENT_SCOPE)
    endfunction()

    # Initialize separate content for each category
    set(core_content "")
    set(bool_content "")
    set(float_content "")
    set(bfloat_content "")
    set(int_content "")
    set(uint_content "")
    set(string_content "")

    # Start building core mappings header (always included)
    string(APPEND core_content "/* AUTOMATICALLY GENERATED - Core Type Mappings */\n")
    string(APPEND core_content "/* Generated by SelectiveRenderingCore.cmake */\n")
    string(APPEND core_content "#ifndef SD_SELECTIVE_RENDERING_CORE_H\n")
    string(APPEND core_content "#define SD_SELECTIVE_RENDERING_CORE_H\n")
    string(APPEND core_content "// Also define master guard so types.h recognizes selective rendering is active\n")
    string(APPEND core_content "#define SD_SELECTIVE_RENDERING_H\n\n")

    # Initialize category headers
    string(APPEND bool_content "/* BOOL type flags */\n#ifndef SD_BOOL_TYPES_H\n#define SD_BOOL_TYPES_H\n\n")
    string(APPEND float_content "/* FLOAT type flags */\n#ifndef SD_FLOAT_TYPES_H\n#define SD_FLOAT_TYPES_H\n\n")
    string(APPEND bfloat_content "/* BFLOAT type flags */\n#ifndef SD_BFLOAT_TYPES_H\n#define SD_BFLOAT_TYPES_H\n\n")
    string(APPEND int_content "/* INT type flags */\n#ifndef SD_INT_TYPES_H\n#define SD_INT_TYPES_H\n\n")
    string(APPEND uint_content "/* UINT type flags */\n#ifndef SD_UINT_TYPES_H\n#define SD_UINT_TYPES_H\n\n")
    string(APPEND string_content "/* STRING type flags */\n#ifndef SD_STRING_TYPES_H\n#define SD_STRING_TYPES_H\n\n")

    # Start building the master header content
    set(header_content "/* AUTOMATICALLY GENERATED - Selective Rendering Header */\n")
    string(APPEND header_content "/* Generated by SelectiveRenderingCore.cmake */\n")
    string(APPEND header_content "#ifndef SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_SELECTIVE_RENDERING_H\n\n")
    string(APPEND header_content "// Include all type category headers\n")
    string(APPEND header_content "#include \"selective_rendering/core.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/bool_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/float_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/bfloat_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/int_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/uint_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/string_types.h\"\n\n")

    # ============================================================================
    # SECTION 1: RAW COMPILATION FLAGS
    # ============================================================================

    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// SECTION 1: RAW COMPILATION FLAGS\n")
    string(APPEND header_content "// ============================================================================\n\n")

    # Collect all compiled type numbers from ACTUAL combinations, not from type_enums
    # This ensures SD_*_COMPILED flags match what was actually instantiated
    list(LENGTH type_enums num_types)
    set(compiled_type_numbers "")

    # Extract types from combinations_2
    foreach(combo IN LISTS combinations_2)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        if(i LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            enum_to_int_value("${enum_i}" int_i)
            list(FIND compiled_type_numbers "${int_i}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_i}")
            endif()
        endif()
        if(j LESS ${num_types})
            list(GET type_enums ${j} enum_j)
            enum_to_int_value("${enum_j}" int_j)
            list(FIND compiled_type_numbers "${int_j}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_j}")
            endif()
        endif()
    endforeach()

    # Extract types from combinations_3
    foreach(combo IN LISTS combinations_3)
        string(REPLACE "," ";" parts "${combo}")
        list(GET parts 0 i)
        list(GET parts 1 j)
        list(GET parts 2 k)
        if(i LESS ${num_types})
            list(GET type_enums ${i} enum_i)
            enum_to_int_value("${enum_i}" int_i)
            list(FIND compiled_type_numbers "${int_i}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_i}")
            endif()
        endif()
        if(j LESS ${num_types})
            list(GET type_enums ${j} enum_j)
            enum_to_int_value("${enum_j}" int_j)
            list(FIND compiled_type_numbers "${int_j}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_j}")
            endif()
        endif()
        if(k LESS ${num_types})
            list(GET type_enums ${k} enum_k)
            enum_to_int_value("${enum_k}" int_k)
            list(FIND compiled_type_numbers "${int_k}" found_idx)
            if(found_idx EQUAL -1)
                list(APPEND compiled_type_numbers "${int_k}")
            endif()
        endif()
    endforeach()

    # Generate single type compilation flags for ALL possible types
    # Append to appropriate category files
    set(all_possible_types "0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;50;51;52;100;200;255")
    foreach(type_num IN LISTS all_possible_types)
        list(FIND compiled_type_numbers "${type_num}" found_idx)
        set(flag_value "0")
        if(found_idx GREATER_EQUAL 0)
            set(flag_value "1")
        endif()

        # Determine which category this type belongs to
        get_type_category("${type_num}" category)

        # Append to appropriate category content
        if(category STREQUAL "bool")
            string(APPEND bool_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED ${flag_value}\n")
        elseif(category STREQUAL "float")
            string(APPEND float_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED ${flag_value}\n")
        elseif(category STREQUAL "bfloat")
            string(APPEND bfloat_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED ${flag_value}\n")
        elseif(category STREQUAL "int")
            string(APPEND int_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED ${flag_value}\n")
        elseif(category STREQUAL "uint")
            string(APPEND uint_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED ${flag_value}\n")
        elseif(category STREQUAL "string")
            string(APPEND string_content "#define SD_SINGLE_TYPE_${type_num}_COMPILED ${flag_value}\n")
        endif()
    endforeach()

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
    # Append to category files based on which types are involved
    foreach(type1 IN LISTS all_possible_types)
        foreach(type2 IN LISTS all_possible_types)
            set(pair_key "${type1}_${type2}")
            list(FIND all_pair_keys "${pair_key}" found_idx)
            set(flag_value "0")
            if(found_idx GREATER_EQUAL 0)
                set(flag_value "1")
            endif()

            # Get category for first type only - assign pair to exactly one file
            # This prevents duplicate definitions when multiple headers are included
            get_type_category("${type1}" cat1)

            # Append to only the category file for the first type
            if(cat1 STREQUAL "bool")
                string(APPEND bool_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            elseif(cat1 STREQUAL "float")
                string(APPEND float_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            elseif(cat1 STREQUAL "bfloat")
                string(APPEND bfloat_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            elseif(cat1 STREQUAL "int")
                string(APPEND int_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            elseif(cat1 STREQUAL "uint")
                string(APPEND uint_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            elseif(cat1 STREQUAL "string")
                string(APPEND string_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            else()
                # For "other" types, put in core.h
                string(APPEND core_content "#define SD_PAIR_TYPE_${type1}_${type2}_COMPILED ${flag_value}\n")
            endif()
        endforeach()
    endforeach()

    # Generate triple type compilation flags - append to category files
    string(APPEND bool_content "\n// Triple type compilation flags (bool-related)\n")
    string(APPEND float_content "\n// Triple type compilation flags (float-related)\n")
    string(APPEND bfloat_content "\n// Triple type compilation flags (bfloat-related)\n")
    string(APPEND int_content "\n// Triple type compilation flags (int-related)\n")
    string(APPEND uint_content "\n// Triple type compilation flags (uint-related)\n")
    string(APPEND string_content "\n// Triple type compilation flags (string-related)\n")

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

    # Generate all triple combinations - route to appropriate category files
    foreach(type1 IN LISTS all_possible_types)
        foreach(type2 IN LISTS all_possible_types)
            foreach(type3 IN LISTS all_possible_types)
                set(triple_key "${type1}_${type2}_${type3}")
                list(FIND all_triple_keys "${triple_key}" found_idx)
                if(found_idx GREATER_EQUAL 0)
                    set(flag_value "1")
                else()
                    set(flag_value "0")
                endif()

                # Get category for first type only - assign triple to exactly one file
                # This prevents duplicate definitions when multiple headers are included
                get_type_category("${type1}" cat1)

                # Append to only the category file for the first type
                if(cat1 STREQUAL "bool")
                    string(APPEND bool_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                elseif(cat1 STREQUAL "float")
                    string(APPEND float_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                elseif(cat1 STREQUAL "bfloat")
                    string(APPEND bfloat_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                elseif(cat1 STREQUAL "int")
                    string(APPEND int_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                elseif(cat1 STREQUAL "uint")
                    string(APPEND uint_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                elseif(cat1 STREQUAL "string")
                    string(APPEND string_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                else()
                    # For "other" types, put in core.h
                    string(APPEND core_content "#define SD_TRIPLE_TYPE_${type1}_${type2}_${type3}_COMPILED ${flag_value}\n")
                endif()
            endforeach()
        endforeach()
    endforeach()

    # ============================================================================
    # SECTION 2: MAPPING TABLES - COMPLETE VERSION
    # These go into core.h which is always included
    # ============================================================================

    string(APPEND core_content "// ============================================================================\n")
    string(APPEND core_content "// SECTION 2: MAPPING TABLES\n")
    string(APPEND core_content "// ============================================================================\n\n")

    # Generate enum to number mappings - Complete DataType enum coverage
    string(APPEND core_content "// DataType enum to number mappings (with namespace handling)\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_INHERIT 0\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_BOOL 1\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_FLOAT8 2\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_HALF 3\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_HALF2 4\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_FLOAT32 5\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_DOUBLE 6\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_INT8 7\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_INT16 8\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_INT32 9\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_INT64 10\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UINT8 11\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UINT16 12\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UINT32 13\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UINT64 14\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_QINT8 15\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_QINT16 16\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_BFLOAT16 17\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UTF8 50\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UTF16 51\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UTF32 52\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_ANY 100\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_AUTO 200\n")
    string(APPEND core_content "#define SD_ENUM_TO_NUM_UNKNOWN 255\n")
    string(APPEND core_content "\n")

    # Generate alias to number mappings
    string(APPEND core_content "// Constexpr alias to number mappings\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_INHERIT 0\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_BOOL 1\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_FLOAT8 2\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_HALF 3\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_HALF2 4\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_FLOAT32 5\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_DOUBLE 6\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_INT8 7\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_INT16 8\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_INT32 9\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_INT64 10\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UINT8 11\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UINT16 12\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UINT32 13\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UINT64 14\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_QINT8 15\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_QINT16 16\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_BFLOAT16 17\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UTF8 50\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UTF16 51\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UTF32 52\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_ANY 100\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_AUTO 200\n")
    string(APPEND core_content "#define SD_ALIAS_TO_NUM_UNKNOWN 255\n")
    string(APPEND core_content "\n")

    # Generate C++ type to number mappings - COMPREHENSIVE LIST
    string(APPEND core_content "// C++ type name to number mappings\n")

    # Build a comprehensive type mapping dictionary
    set(type_mappings "")
    list(APPEND type_mappings "bool:1")
    list(APPEND type_mappings "float16:3")
    list(APPEND type_mappings "half:3")
    list(APPEND type_mappings "float:5")
    list(APPEND type_mappings "float32:5")
    list(APPEND type_mappings "double:6")
    list(APPEND type_mappings "int8_t:7")
    list(APPEND type_mappings "int16_t:8")
    list(APPEND type_mappings "Int16Type:8")
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
    list(APPEND type_mappings "UInt16Type:12")
    list(APPEND type_mappings "unsigned_short:12")
    list(APPEND type_mappings "uint32_t:13")
    list(APPEND type_mappings "UInt32Type:13")
    list(APPEND type_mappings "unsigned_int:13")
    list(APPEND type_mappings "uint64_t:14")
    list(APPEND type_mappings "UInt64Type:14")
    list(APPEND type_mappings "UnsignedLong:14")
    list(APPEND type_mappings "unsigned_long:14")
    list(APPEND type_mappings "bfloat16:17")
    list(APPEND type_mappings "bfloat:17")
    list(APPEND type_mappings "stdstring:50")
    list(APPEND type_mappings "SignedChar:7")
    list(APPEND type_mappings "UnsignedChar:11")
    list(APPEND type_mappings "signed_char:7")
    list(APPEND type_mappings "unsigned_char:11")
    list(APPEND type_mappings "schar:7")
    list(APPEND type_mappings "uchar:11")
    list(APPEND type_mappings "int8:7")
    list(APPEND type_mappings "uint8:11")
    list(APPEND type_mappings "int16:8")
    list(APPEND type_mappings "uint16:12")
    list(APPEND type_mappings "int32:9")
    list(APPEND type_mappings "uint32:13")
    list(APPEND type_mappings "int64:10")
    list(APPEND type_mappings "uint64:14")
    list(APPEND type_mappings "short:8")
    list(APPEND type_mappings "ushort:12")
    list(APPEND type_mappings "uint:13")
    list(APPEND type_mappings "longlong:10")
    list(APPEND type_mappings "ulonglong:14")
    list(APPEND type_mappings "ulong:14")
    list(APPEND type_mappings "char:7")

    # Generate all type to number mappings (both SD_TYPE_TO_NUM and SD_ALIAS_TO_NUM for compatibility)
    foreach(mapping IN LISTS type_mappings)
        string(REPLACE ":" ";" mapping_parts "${mapping}")
        list(GET mapping_parts 0 type_name)
        list(GET mapping_parts 1 type_num)
        string(APPEND core_content "#define SD_TYPE_TO_NUM_${type_name} ${type_num}\n")
        string(APPEND core_content "#define SD_ALIAS_TO_NUM_${type_name} ${type_num}\n")
    endforeach()
    string(APPEND core_content "\n")

    # ============================================================================
    # SECTION 2.5: HAS_* FEATURE DETECTION MACROS
    # ============================================================================
    string(APPEND core_content "// HAS_* macros for DataTypeUtils.h scalarTypesForNDarray trait\n")
    string(APPEND core_content "// These indicate which types are compiled in selective rendering mode\n")

    # Map type names to HAS_* macro names
    set(HAS_MACRO_MAPPINGS "")
    list(APPEND HAS_MACRO_MAPPINGS "bool:BOOL")
    list(APPEND HAS_MACRO_MAPPINGS "float16:FLOAT16")
    list(APPEND HAS_MACRO_MAPPINGS "bfloat16:BFLOAT16")
    list(APPEND HAS_MACRO_MAPPINGS "float32:FLOAT32")
    list(APPEND HAS_MACRO_MAPPINGS "double:DOUBLE")
    list(APPEND HAS_MACRO_MAPPINGS "int8:INT8")
    list(APPEND HAS_MACRO_MAPPINGS "int16:INT16")
    list(APPEND HAS_MACRO_MAPPINGS "int32:INT32")
    list(APPEND HAS_MACRO_MAPPINGS "int64:INT64")
    list(APPEND HAS_MACRO_MAPPINGS "uint8:UINT8")
    list(APPEND HAS_MACRO_MAPPINGS "uint16:UINT16")
    list(APPEND HAS_MACRO_MAPPINGS "uint32:UINT32")
    list(APPEND HAS_MACRO_MAPPINGS "uint64:UNSIGNEDLONG")
    list(APPEND HAS_MACRO_MAPPINGS "utf8:UTF8")
    list(APPEND HAS_MACRO_MAPPINGS "utf16:UTF16")
    list(APPEND HAS_MACRO_MAPPINGS "utf32:UTF32")

    # Generate HAS_* macros for enabled types
    foreach(type_name IN LISTS SD_TYPES_LIST)
        # Convert type name to lowercase for matching
        string(TOLOWER "${type_name}" type_lower)

        # Find matching HAS_* macro name
        foreach(mapping IN LISTS HAS_MACRO_MAPPINGS)
            string(REPLACE ":" ";" mapping_parts "${mapping}")
            list(GET mapping_parts 0 map_type)
            list(GET mapping_parts 1 map_macro)

            if("${type_lower}" STREQUAL "${map_type}")
                string(APPEND core_content "#define HAS_${map_macro} 1\n")
                break()
            endif()
        endforeach()
    endforeach()
    string(APPEND core_content "\n")

    # ============================================================================
    # SECTION 3: CONDITIONAL COMPILATION MACROS
    # ============================================================================

    # ============================================================================
    # SECTION 3: REMOVED - SD_IF_* CONDITIONAL COMPILATION MACROS
    # ============================================================================
    # NOTE: The SD_IF_* macro generation has been completely removed.
    #
    # REASON: These macros consumed 19,344 lines (~96% of selective_rendering.h),
    # causing Clang 20.1.6 to exceed its source location limit ("ran out of source
    # locations") when compiling large translation units like NativeOpExecutioner.cpp.
    #
    # IMPACT: None. Analysis showed these macros were never used in the codebase.
    # The code directly checks SD_*_TYPE_*_COMPILED flags via preprocessor conditionals.
    #
    # RESULT: selective_rendering.h reduced from 20,152 lines to ~800 lines (96% reduction),
    # allowing compilation with Clang while maintaining full 14-type support.
    #
    # Helper macros (SD_BUILD_*_IF_VALID) are appended separately by
    # _internal_srcore_append_runtime_dispatch_to_header() around line 2171.

    # ============================================================================
    # Write partitioned header files
    # ============================================================================

    # Create selective_rendering subdirectory
    get_filename_component(header_dir "${header_file}" DIRECTORY)
    set(sr_dir "${header_dir}/selective_rendering")
    file(MAKE_DIRECTORY "${sr_dir}")

    # Close and write core.h (type mappings - always needed)
    string(APPEND core_content "\n#endif // SD_SELECTIVE_RENDERING_CORE_H\n")
    _srcore_write_if_different("${sr_dir}/core.h" "${core_content}")

    # Close and write bool_types.h
    string(APPEND bool_content "\n#endif // SD_SELECTIVE_RENDERING_BOOL_TYPES_H\n")
    _srcore_write_if_different("${sr_dir}/bool_types.h" "${bool_content}")

    # Close and write float_types.h
    string(APPEND float_content "\n#endif // SD_SELECTIVE_RENDERING_FLOAT_TYPES_H\n")
    _srcore_write_if_different("${sr_dir}/float_types.h" "${float_content}")

    # Close and write bfloat_types.h
    string(APPEND bfloat_content "\n#endif // SD_SELECTIVE_RENDERING_BFLOAT_TYPES_H\n")
    _srcore_write_if_different("${sr_dir}/bfloat_types.h" "${bfloat_content}")

    # Close and write int_types.h
    string(APPEND int_content "\n#endif // SD_SELECTIVE_RENDERING_INT_TYPES_H\n")
    _srcore_write_if_different("${sr_dir}/int_types.h" "${int_content}")

    # Close and write uint_types.h
    string(APPEND uint_content "\n#endif // SD_SELECTIVE_RENDERING_UINT_TYPES_H\n")
    _srcore_write_if_different("${sr_dir}/uint_types.h" "${uint_content}")

    # Close and write string_types.h
    string(APPEND string_content "\n#endif // SD_SELECTIVE_RENDERING_STRING_TYPES_H\n")
    _srcore_write_if_different("${sr_dir}/string_types.h" "${string_content}")

    # Replace header_content with includes to all category headers
    set(header_content "")
    string(APPEND header_content "#ifndef SD_SELECTIVE_RENDERING_H\n")
    string(APPEND header_content "#define SD_SELECTIVE_RENDERING_H\n\n")
    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// Selective Rendering Type System - Partitioned Headers\n")
    string(APPEND header_content "// ============================================================================\n")
    string(APPEND header_content "// This master header includes all type category headers.\n")
    string(APPEND header_content "// Large translation units can include only the specific category headers\n")
    string(APPEND header_content "// they need to avoid Clang source location limits.\n")
    string(APPEND header_content "// ============================================================================\n\n")
    string(APPEND header_content "// Core type mappings (always required)\n")
    string(APPEND header_content "#include \"selective_rendering/core.h\"\n\n")
    string(APPEND header_content "// Type category headers\n")
    string(APPEND header_content "#include \"selective_rendering/bool_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/float_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/bfloat_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/int_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/uint_types.h\"\n")
    string(APPEND header_content "#include \"selective_rendering/string_types.h\"\n\n")

    # Close the header guard
    string(APPEND header_content "#endif // SD_SELECTIVE_RENDERING_H\n")

    # Write the master header file
    _srcore_write_if_different("${header_file}" "${header_content}")

    # Report generation results
    list(LENGTH all_triple_keys total_triple_combinations)
    list(LENGTH all_pair_keys total_pair_combinations)
    list(LENGTH compiled_type_numbers total_single_types)
    message(STATUS "Generated selective_rendering.h (partitioned):")
    message(STATUS "  - Master header: ${header_file}")
    message(STATUS "  - Partitioned headers directory: ${sr_dir}/")
    message(STATUS "    * core.h (type mappings)")
    message(STATUS "    * bool_types.h, float_types.h, bfloat_types.h")
    message(STATUS "    * int_types.h, uint_types.h, string_types.h")
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
