//
// Created by agibsonccc on 8/6/25.
//
/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
#ifndef LIBND4J_TYPE_INSTANTIATE_BOILERPLATE_H
#define LIBND4J_TYPE_INSTANTIATE_BOILERPLATE_H
#include <system/selective_rendering.h>

// ===========================================================================
// TEMPLATE INSTANTIATION MACROS - Used for declaring template instantiations
// ===========================================================================

// Single type template instantiation using aliases only
#define SD_INSTANTIATE_SINGLE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    SD_IF_SINGLE_ALIAS_COMPILED(TYPE_ALIAS, \
        TEMPLATE_NAME<TYPE> SIGNATURE; \
    )

// Double type template instantiation using aliases only
#define SD_INSTANTIATE_DOUBLE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_A_ALIAS, TYPE_A, TYPE_B_ALIAS, TYPE_B) \
    SD_IF_PAIR_ALIAS_COMPILED(TYPE_A_ALIAS, TYPE_B_ALIAS, \
        template TEMPLATE_NAME<TYPE_A, TYPE_B> SIGNATURE; \
    )

// Triple type template instantiation using aliases only
#define SD_INSTANTIATE_TRIPLE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_A_ALIAS, TYPE_A, TYPE_B_ALIAS, TYPE_B, TYPE_C_ALIAS, TYPE_C) \
    SD_IF_TRIPLE_ALIAS_COMPILED(TYPE_A_ALIAS, TYPE_B_ALIAS, TYPE_C_ALIAS, \
        TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE; \
    )

// Template instantiation with same type used twice using aliases only
#define SD_INSTANTIATE_SINGLE_TWICE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    SD_IF_PAIR_ALIAS_COMPILED(TYPE_ALIAS, TYPE_ALIAS, \
        TEMPLATE_NAME<TYPE, TYPE> SIGNATURE; \
    )

// Template instantiation with same type used three times using aliases only
#define SD_INSTANTIATE_SINGLE_THRICE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    SD_IF_TRIPLE_ALIAS_COMPILED(TYPE_ALIAS, TYPE_ALIAS, TYPE_ALIAS, \
        TEMPLATE_NAME<TYPE, TYPE, TYPE> SIGNATURE; \
    )

// Unchained single type instantiation using aliases only
#define SD_INSTANTIATE_UNCHAINED_SINGLE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    SD_IF_SINGLE_ALIAS_COMPILED(TYPE_ALIAS, \
        TEMPLATE_NAME TYPE SIGNATURE; \
    )

// Partial single type instantiation using aliases only
#define SD_INSTANTIATE_PARTIAL_SINGLE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    SD_IF_SINGLE_ALIAS_COMPILED(TYPE_ALIAS, \
        TEMPLATE_NAME TYPE, UNPAREN2(SIGNATURE); \
    )

// Triple partial instantiation using aliases only
#define SD_INSTANTIATE_TRIPLE_PARTIAL_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_A, TYPE_B, TYPE_C_ALIAS, TYPE_C) \
    SD_IF_SINGLE_ALIAS_COMPILED(TYPE_C_ALIAS, \
        TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE; \
    )

// Special case for triple with types only (used by pairwise)
#define SD_INSTANTIATE_TRIPLE_TYPES_ONLY_DECL(TEMPLATE_NAME, TYPE_B, SIGNATURE, TYPE_A, TYPE_C) \
    TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE;

// ===========================================================================
// FUNCTION CALL MACROS - Used for calling templated functions in selectors
// ===========================================================================

// Single type function call - for use in switch cases
#define SD_INSTANTIATE_SINGLE(NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    NAME<TYPE> SIGNATURE;

// Double type function call
#define SD_INSTANTIATE_DOUBLE(TEMPLATE_NAME, SIGNATURE, TYPE_A_ALIAS, TYPE_A, TYPE_B_ALIAS, TYPE_B) \
    TEMPLATE_NAME<TYPE_A, TYPE_B> SIGNATURE

// Triple type function call
#define SD_INSTANTIATE_TRIPLE(TEMPLATE_NAME, SIGNATURE, TYPE_A_ALIAS, TYPE_A, TYPE_B_ALIAS, TYPE_B, TYPE_C_ALIAS, TYPE_C) \
    TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE

// Function call with same type used twice
#define SD_INSTANTIATE_SINGLE_TWICE(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    TEMPLATE_NAME<TYPE, TYPE> SIGNATURE

// Function call with same type used three times
#define SD_INSTANTIATE_SINGLE_THRICE(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    TEMPLATE_NAME<TYPE, TYPE, TYPE> SIGNATURE

// Unchained single type function call
#define SD_INSTANTIATE_UNCHAINED_SINGLE(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    TEMPLATE_NAME TYPE SIGNATURE

// Partial single type function call
#define SD_INSTANTIATE_PARTIAL_SINGLE(TEMPLATE_NAME, SIGNATURE, TYPE_ALIAS, TYPE) \
    TEMPLATE_NAME TYPE, UNPAREN2(SIGNATURE)

// Triple partial function call
#define SD_INSTANTIATE_TRIPLE_PARTIAL(TEMPLATE_NAME, SIGNATURE, TYPE_A, TYPE_B, TYPE_C_ALIAS, TYPE_C) \
    TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE

// Special case for triple with types only (used by pairwise)
#define SD_INSTANTIATE_TRIPLE_TYPES_ONLY(TEMPLATE_NAME, TYPE_B, SIGNATURE, TYPE_A, TYPE_C) \
    TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE


// ===========================================================================
// RAW TEMPLATE DECLARATION MACROS - Generate template syntax without conditionals
// For use inside conditional compilation wrappers
// ===========================================================================

// Raw single type template declaration
#define SD_RAW_SINGLE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE) \
    template TEMPLATE_NAME<TYPE> SIGNATURE;

// Raw double type template declaration
#define SD_RAW_DOUBLE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_A, TYPE_B) \
    template SIGNATURE<TYPE_A, TYPE_B>;

// Raw triple type template declaration
#define SD_RAW_TRIPLE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_A, TYPE_B, TYPE_C) \
    template TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE;

// Raw template declaration with same type used twice
#define SD_RAW_SINGLE_TWICE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE) \
    template TEMPLATE_NAME<TYPE, TYPE> SIGNATURE;

// Raw template declaration with same type used three times
#define SD_RAW_SINGLE_THRICE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE) \
    template TEMPLATE_NAME<TYPE, TYPE, TYPE> SIGNATURE;

// Raw unchained single type template declaration
#define SD_RAW_UNCHAINED_SINGLE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE) \
    template TEMPLATE_NAME TYPE SIGNATURE;

// Raw partial single type template declaration
#define SD_RAW_PARTIAL_SINGLE_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE) \
    template TEMPLATE_NAME TYPE, UNPAREN2(SIGNATURE);

// Raw triple partial template declaration
#define SD_RAW_TRIPLE_PARTIAL_TEMPLATE_DECL(TEMPLATE_NAME, SIGNATURE, TYPE_A, TYPE_B, TYPE_C) \
    template TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE;

// Raw triple with types only template declaration (used by pairwise)
#define SD_RAW_TRIPLE_TYPES_ONLY_TEMPLATE_DECL(TEMPLATE_NAME, TYPE_B, SIGNATURE, TYPE_A, TYPE_C) \
    template TEMPLATE_NAME<TYPE_A, TYPE_B, TYPE_C> SIGNATURE;




// ============================================================================
// SECTION 3: CONDITIONAL EXPANSION PRIMITIVES (VARIADIC)
// ============================================================================

// CRITICAL: Using variadic macros to preserve whitespace
// Statement context expansions (for use in function bodies)
#define SD_IF_1(...) __VA_ARGS__
#define SD_IF_0(...) do {} while(0);

// Declaration context expansions (for use at file/namespace scope)
#define SD_IF_DECL_1(...) __VA_ARGS__
#define SD_IF_DECL_0(...) /* filtered out */

// Expression context expansions (for use in expressions)
#define SD_IF_EXPR_1(...) __VA_ARGS__
#define SD_IF_EXPR_0(...) ((void)0)

// Special whitespace-preserving helpers
#define SD_UNPAREN(...) SD_UNPAREN_IMPL __VA_ARGS__
#define SD_UNPAREN_IMPL(...) __VA_ARGS__
#define SD_IDENTITY(...) __VA_ARGS__
#define SD_EXPAND(...) __VA_ARGS__

// ============================================================================
// SECTION 4: TOKEN MANIPULATION HELPERS
// ============================================================================

// Basic concatenation macros
#define SD_CAT(a, b) SD_CAT_I(a, b)
#define SD_CAT_I(a, b) a ## b

// Three-token concatenation
#define SD_CAT3(a, b, c) SD_CAT3_I(a, b, c)
#define SD_CAT3_I(a, b, c) a ## b ## c

// Five-token concatenation
#define SD_CAT5(a, b, c, d, e) SD_CAT5_I(a, b, c, d, e)
#define SD_CAT5_I(a, b, c, d, e) a ## b ## c ## d ## e

// Seven-token concatenation
#define SD_CAT7(a, b, c, d, e, f, g) SD_CAT7_I(a, b, c, d, e, f, g)
#define SD_CAT7_I(a, b, c, d, e, f, g) a ## b ## c ## d ## e ## f ## g

// ============================================================================
// SECTION 5: COMPILATION CHECK MACROS
// ============================================================================

// Check if type combinations are compiled (returns 0 or 1)
#define SD_IS_SINGLE_COMPILED(NUM) \
    SD_CAT3(SD_SINGLE_TYPE_, NUM, _COMPILED)

#define SD_IS_PAIR_COMPILED(NUM1, NUM2) \
    SD_CAT5(SD_PAIR_TYPE_, NUM1, _, NUM2, _COMPILED)

#define SD_IS_TRIPLE_COMPILED(NUM1, NUM2, NUM3) \
    SD_CAT7(SD_TRIPLE_TYPE_, NUM1, _, NUM2, _, NUM3, _COMPILED)

// ============================================================================
// SECTION 6: UNIFIED INTERFACE - NUMERIC (VARIADIC)
// ============================================================================


#define SD_IF_SINGLE_COMPILED(TYPE_NUM, ...) \
    SD_CAT(SD_IF_, SD_IS_SINGLE_COMPILED(TYPE_NUM))(__VA_ARGS__)

#define SD_IF_PAIR_COMPILED(TYPE1_NUM, TYPE2_NUM, ...) \
    SD_CAT(SD_IF_, SD_IS_PAIR_COMPILED(TYPE1_NUM, TYPE2_NUM))(__VA_ARGS__)

#define SD_IF_TRIPLE_COMPILED(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM, ...) \
    SD_CAT(SD_IF_, SD_IS_TRIPLE_COMPILED(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM))(__VA_ARGS__)



// Direct numeric type ID interfaces (declaration context)
#define SD_IF_SINGLE_COMPILED_DECL(TYPE_NUM, ...) \
    SD_CAT(SD_IF_DECL_, SD_IS_SINGLE_COMPILED(TYPE_NUM))(__VA_ARGS__)

#define SD_IF_PAIR_COMPILED_DECL(TYPE1_NUM, TYPE2_NUM, ...) \
    SD_CAT(SD_IF_DECL_, SD_IS_PAIR_COMPILED(TYPE1_NUM, TYPE2_NUM))(__VA_ARGS__)

#define SD_IF_TRIPLE_COMPILED_DECL(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM, ...) \
    SD_CAT(SD_IF_DECL_, SD_IS_TRIPLE_COMPILED(TYPE1_NUM, TYPE2_NUM, TYPE3_NUM))(__VA_ARGS__)

// ============================================================================
// SECTION 7: UNIFIED INTERFACE - DATATYPE ENUM (VARIADIC)
// ============================================================================

// Helper macros for forcing evaluation through indirection
#define SD_EVAL_ENUM_TO_NUM_I(x) x
#define SD_EVAL_ENUM_TO_NUM(DTYPE) SD_EVAL_ENUM_TO_NUM_I(SD_CAT(SD_ENUM_TO_NUM_, DTYPE))

// DataType enum interfaces (statement context)
#define SD_IF_SINGLE_DATATYPE_COMPILED(DTYPE, ...) \
    SD_IF_SINGLE_COMPILED(SD_EVAL_ENUM_TO_NUM(DTYPE), __VA_ARGS__)

#define SD_IF_PAIR_DATATYPE_COMPILED(DTYPE1, DTYPE2, ...) \
    SD_IF_PAIR_COMPILED( \
        SD_EVAL_ENUM_TO_NUM(DTYPE1), \
        SD_EVAL_ENUM_TO_NUM(DTYPE2), \
        __VA_ARGS__)

#define SD_IF_TRIPLE_DATATYPE_COMPILED(DTYPE1, DTYPE2, DTYPE3, ...) \
    SD_IF_TRIPLE_COMPILED( \
        SD_EVAL_ENUM_TO_NUM(DTYPE1), \
        SD_EVAL_ENUM_TO_NUM(DTYPE2), \
        SD_EVAL_ENUM_TO_NUM(DTYPE3), \
        __VA_ARGS__)

// DataType enum interfaces (declaration context)
#define SD_IF_SINGLE_DATATYPE_COMPILED_DECL(DTYPE, ...) \
    SD_IF_SINGLE_COMPILED_DECL(SD_EVAL_ENUM_TO_NUM(DTYPE), __VA_ARGS__)

#define SD_IF_PAIR_DATATYPE_COMPILED_DECL(DTYPE1, DTYPE2, ...) \
    SD_IF_PAIR_COMPILED_DECL( \
        SD_EVAL_ENUM_TO_NUM(DTYPE1), \
        SD_EVAL_ENUM_TO_NUM(DTYPE2), \
        __VA_ARGS__)

#define SD_IF_TRIPLE_DATATYPE_COMPILED_DECL(DTYPE1, DTYPE2, DTYPE3, ...) \
    SD_IF_TRIPLE_COMPILED_DECL( \
        SD_EVAL_ENUM_TO_NUM(DTYPE1), \
        SD_EVAL_ENUM_TO_NUM(DTYPE2), \
        SD_EVAL_ENUM_TO_NUM(DTYPE3), \
        __VA_ARGS__)

// ============================================================================
// SECTION 8: UNIFIED INTERFACE - CONSTEXPR ALIASES (VARIADIC)
// ============================================================================

// Constexpr alias interfaces (statement context)
#define SD_IF_SINGLE_ALIAS_COMPILED(ALIAS, ...) \
    SD_IF_SINGLE_COMPILED(SD_CAT(SD_ALIAS_TO_NUM_, ALIAS), __VA_ARGS__)

#define SD_IF_PAIR_ALIAS_COMPILED(ALIAS1, ALIAS2, ...) \
    SD_IF_PAIR_COMPILED( \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \
        __VA_ARGS__)

#define SD_IF_TRIPLE_ALIAS_COMPILED(ALIAS1, ALIAS2, ALIAS3, ...) \
    SD_IF_TRIPLE_COMPILED( \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS3), \
        __VA_ARGS__)

// Constexpr alias interfaces (declaration context)
#define SD_IF_SINGLE_ALIAS_COMPILED_DECL(ALIAS, ...) \
    SD_IF_SINGLE_COMPILED_DECL(SD_CAT(SD_ALIAS_TO_NUM_, ALIAS), __VA_ARGS__)

#define SD_IF_PAIR_ALIAS_COMPILED_DECL(ALIAS1, ALIAS2, ...) \
    SD_IF_PAIR_COMPILED_DECL( \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \
        __VA_ARGS__)

#define SD_IF_TRIPLE_ALIAS_COMPILED_DECL(ALIAS1, ALIAS2, ALIAS3, ...) \
    SD_IF_TRIPLE_COMPILED_DECL( \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS1), \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS2), \
        SD_CAT(SD_ALIAS_TO_NUM_, ALIAS3), \
        __VA_ARGS__)

// ============================================================================
// SECTION 9: UNIFIED INTERFACE - C++ TYPE NAMES (VARIADIC)
// ============================================================================

// C++ type name interfaces (statement context)
#define SD_IF_SINGLE_TYPE_COMPILED(TYPE, ...) \
    SD_IF_SINGLE_COMPILED(SD_CAT(SD_TYPE_TO_NUM_, TYPE), __VA_ARGS__)

#define SD_IF_PAIR_TYPE_COMPILED(TYPE1, TYPE2, ...) \
    SD_IF_PAIR_COMPILED( \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \
        __VA_ARGS__)

#define SD_IF_TRIPLE_TYPE_COMPILED(TYPE1, TYPE2, TYPE3, ...) \
    SD_IF_TRIPLE_COMPILED( \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE3), \
        __VA_ARGS__)

// C++ type name interfaces (declaration context)
#define SD_IF_SINGLE_TYPE_COMPILED_DECL(TYPE, ...) \
    SD_IF_SINGLE_COMPILED_DECL(SD_CAT(SD_TYPE_TO_NUM_, TYPE), __VA_ARGS__)

#define SD_IF_PAIR_TYPE_COMPILED_DECL(TYPE1, TYPE2, ...) \
    SD_IF_PAIR_COMPILED_DECL( \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \
        __VA_ARGS__)

#define SD_IF_TRIPLE_TYPE_COMPILED_DECL(TYPE1, TYPE2, TYPE3, ...) \
    SD_IF_TRIPLE_COMPILED_DECL( \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE1), \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE2), \
        SD_CAT(SD_TYPE_TO_NUM_, TYPE3), \
        __VA_ARGS__)


// Whitespace-preserving wrappers for BUILD_* macros
// These help preserve whitespace when selective rendering is used
#define SD_PRESERVE_WS(...) __VA_ARGS__
#define SD_PASS_THROUGH(...) __VA_ARGS__

// Helper for fixing parenthesized arguments
#define SD_FIX_PAREN(x) SD_FIX_PAREN_IMPL x
#define SD_FIX_PAREN_IMPL(...) __VA_ARGS__



#endif  // LIBND4J_TYPE_INSTANTIATE_BOILERPLATE_H
