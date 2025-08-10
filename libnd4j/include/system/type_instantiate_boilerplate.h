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

#endif  // LIBND4J_TYPE_INSTANTIATE_BOILERPLATE_H
