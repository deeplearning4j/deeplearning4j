/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_SAFE_TYPE_ACCESS_H
#define LIBND4J_SAFE_TYPE_ACCESS_H

#include "types.h"

// Safe type access macros that prevent out-of-bounds access when using custom type sets
// These macros gracefully handle cases where type indices don't exist

// Safe GET macro that falls back to the first type if index is out of bounds
#define SAFE_GET(INDEX, TYPE_LIST) \
    SAFE_GET_IMPL(INDEX, TYPE_LIST, COUNT_NARG(TYPE_LIST))

// Implementation helper - uses the first type as fallback
#define SAFE_GET_IMPL(INDEX, TYPE_LIST, TYPE_COUNT) \
    SAFE_GET_EXPAND(INDEX, TYPE_LIST, TYPE_COUNT)

#define SAFE_GET_EXPAND(INDEX, TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_SAFE_##INDEX(TYPE_LIST, TYPE_COUNT)

// Safe GET_ELEMENT implementations
#define GET_ELEMENT_SAFE_0(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT(0, TYPE_LIST)

#define GET_ELEMENT_SAFE_1(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_1(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_2(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_2(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_3(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_3(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_4(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_4(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_5(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_5(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_6(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_6(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_7(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_7(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_8(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_8(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_9(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_9(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_10(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_10(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_11(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_11(TYPE_LIST, TYPE_COUNT)

#define GET_ELEMENT_SAFE_12(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT_CONDITIONAL_12(TYPE_LIST, TYPE_COUNT)

// Conditional implementations that fall back to index 0
#define GET_ELEMENT_CONDITIONAL_1(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 1) ? 1 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_2(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 2) ? 2 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_3(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 3) ? 3 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_4(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 4) ? 4 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_5(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 5) ? 5 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_6(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 6) ? 6 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_7(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 7) ? 7 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_8(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 8) ? 8 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_9(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 9) ? 9 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_10(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 10) ? 10 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_11(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 11) ? 11 : 0, TYPE_LIST)

#define GET_ELEMENT_CONDITIONAL_12(TYPE_LIST, TYPE_COUNT) \
    GET_ELEMENT((TYPE_COUNT > 12) ? 12 : 0, TYPE_LIST)

#endif  // LIBND4J_SAFE_TYPE_ACCESS_H
