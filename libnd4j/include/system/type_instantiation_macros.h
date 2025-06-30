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

#ifndef SD_TYPE_INST_H
#define SD_TYPE_INST_H

#include <system/selective_rendering.h>
#include <type_traits>
#include <common.h>

// ============================================================================
// CONDITIONAL INSTANTIATION SYSTEM WITH ARGUMENT COUNT DISPATCH
// ============================================================================

// Count arguments
#define CONDITIONAL_INSTANTIATE_NARGS(...) \
    CONDITIONAL_INSTANTIATE_NARGS_IMPL(__VA_ARGS__, 16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)

#define CONDITIONAL_INSTANTIATE_NARGS_IMPL(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,N,...) N

// Main conditional instantiation macro
#define CONDITIONAL_INSTANTIATE(condition, ...) \
    EVAL(CONDITIONAL_INSTANTIATE_DISPATCH(CONDITIONAL_INSTANTIATE_NARGS(__VA_ARGS__), condition, __VA_ARGS__))

#define CONDITIONAL_INSTANTIATE_DISPATCH(N, condition, ...) \
    CONDITIONAL_INSTANTIATE_CAT(CONDITIONAL_INSTANTIATE_, N)(condition, __VA_ARGS__)

#define CONDITIONAL_INSTANTIATE_CAT(a, b) a ## b

// ============================================================================
// SOLUTION 9: FUNCTION-LIKE SELECTION - NO CONCATENATION WITH CONDITIONS
// ============================================================================

// Instead of concatenating condition results, use function-like selection
#define CONDITIONAL_INSTANTIATE_IF(condition) \
    CONDITIONAL_INSTANTIATE_EXECUTE(SIMPLE_BOOL(condition))

// Convert any condition to exactly 0 or 1 - this handles complex expressions
#define SIMPLE_BOOL(x) \
    SIMPLE_BOOL_HELPER(x, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)

#define SIMPLE_BOOL_HELPER(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,...) a16

// Use the boolean result to select execution method - NO concatenation with complex expressions
#define CONDITIONAL_INSTANTIATE_EXECUTE(clean_bool) \
    CONDITIONAL_INSTANTIATE_CAT(CONDITIONAL_INSTANTIATE_RUN_, clean_bool)

#define CONDITIONAL_INSTANTIATE_RUN_1
#define CONDITIONAL_INSTANTIATE_RUN_0 //

// Handlers for different argument counts
#define CONDITIONAL_INSTANTIATE_1(condition, arg1) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1

#define CONDITIONAL_INSTANTIATE_2(condition, arg1, arg2) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2

#define CONDITIONAL_INSTANTIATE_3(condition, arg1, arg2, arg3) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2 arg3

#define CONDITIONAL_INSTANTIATE_4(condition, arg1, arg2, arg3, arg4) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2 arg3 arg4

#define CONDITIONAL_INSTANTIATE_5(condition, arg1, arg2, arg3, arg4, arg5) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2 arg3 arg4 arg5

#define CONDITIONAL_INSTANTIATE_6(condition, arg1, arg2, arg3, arg4, arg5, arg6) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2 arg3 arg4 arg5 arg6

#define CONDITIONAL_INSTANTIATE_7(condition, arg1, arg2, arg3, arg4, arg5, arg6, arg7) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2 arg3 arg4 arg5 arg6 arg7

#define CONDITIONAL_INSTANTIATE_8(condition, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) \
    CONDITIONAL_INSTANTIATE_IF(condition) arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8

// ============================================================================
// COMMA MACRO FOR NESTED TEMPLATE ARGUMENTS
// ============================================================================


// ============================================================================
// TYPE CHECKING SYSTEM
// ============================================================================

#define EXPAND_AND_CHECK_TRIPLE(a1, b1, c1) SD_IS_TRIPLE_TYPE_COMPILED(GET_FIRST(a1),GET_FIRST(b1),GET_FIRST(c1))
#define EXPAND_AND_CHECK_PAIR(a1, b1) SD_IS_PAIR_TYPE_COMPILED(GET_FIRST(a1), GET_FIRST(b1))

// ============================================================================
// TEMPLATE INSTANTIATION MACROS
// ============================================================================
// Function template instantiation for 2 types
#define INSTANT_PROCESS_COMBINATION(a1, b1, FUNC_NAME, ARGS) \
    UNPAREN(CONDITIONAL_INSTANTIATE( \
        EXPAND_AND_CHECK_PAIR(a1, b1), \
        (template void FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1)>ARGS); \
    ))

// Function template instantiation for 3 types
#define INSTANT_PROCESS_COMBINATION_3(a1, b1, c1, FUNC_NAME, ARGS) \
    UNPAREN(CONDITIONAL_INSTANTIATE( \
        EXPAND_AND_CHECK_TRIPLE(a1, b1, c1), \
        (template void FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1),GET_SECOND(c1)> ARGS); \
    ))

// Class template instantiation for 2 types
#define INSTANT_PROCESS_CLASSCOMBINATION(a1, b1, CLASS_NAME, ARGS) \
    UNPAREN(CONDITIONAL_INSTANTIATE( \
        EXPAND_AND_CHECK_PAIR(a1, b1), \
        (template class CLASS_NAME<GET_SECOND(a1),GET_SECOND(b1)>); \
    ))

// Class template instantiation for 3 types
#define INSTANT_PROCESS_COMBINATION_CLASS_3(a1, b1, c1, CLASS_NAME, ARGS) \
    UNPAREN(CONDITIONAL_INSTANTIATE( \
        EXPAND_AND_CHECK_TRIPLE(a1, b1, c1), \
        template class CLASS_NAME<GET_SECOND(a1) COMMA GET_SECOND(b1) COMMA GET_SECOND(c1)>; \
    ))


#endif // SD_TYPE_INST_H