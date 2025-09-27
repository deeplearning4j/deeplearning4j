//
// Created by agibsonccc on 11/22/24.
//

#ifndef LIBND4J_PAIRWISE_INSTANTIATIONS_H
#define LIBND4J_PAIRWISE_INSTANTIATIONS_H
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
//
// Instantiated by genCompilation
//
#include <loops/pairwise_transform.h>
#include <system/type_boilerplate.h>
#include <types/types.h>

#include "../types/types.h"

// Note: Instantiations are generated to prevent compiler memory issues

/*
 *
 * Function Instantiation:
 * PairWiseTransform::exec instantiated for types: @TYPE1@, @TYPE2@, @TYPE3@
 */

// ============================================================================
// SAFE TYPE PARTITIONING SYSTEM 
// This system prevents the "index out of bounds" errors that occur when 
// using custom type sets with fewer types than expected by hardcoded indices.
// ============================================================================

// Helper macro to get the count of types
#define COUNT_SD_COMMON_TYPES COUNT_NARG(SD_COMMON_TYPES)
#define COUNT_SD_NUMERIC_TYPES COUNT_NARG(SD_NUMERIC_TYPES)
#define COUNT_SD_FLOAT_TYPES COUNT_NARG(SD_FLOAT_TYPES)
#define COUNT_SD_INTEGER_TYPES COUNT_NARG(SD_INTEGER_TYPES)
#define COUNT_SD_BOOL_TYPES COUNT_NARG(SD_BOOL_TYPES)
#define COUNT_SD_LONG_TYPES COUNT_NARG(SD_LONG_TYPES)

// Safe partitioning for SD_COMMON_TYPES
// These will always work regardless of how many types are actually defined

// Part 0: First available types (up to 5)
#if COUNT_SD_COMMON_TYPES >= 5
#define SD_COMMON_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_COMMON_TYPES), \
    GET_ELEMENT(1, SD_COMMON_TYPES), \
    GET_ELEMENT(2, SD_COMMON_TYPES), \
    GET_ELEMENT(3, SD_COMMON_TYPES), \
    GET_ELEMENT(4, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 4
#define SD_COMMON_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_COMMON_TYPES), \
    GET_ELEMENT(1, SD_COMMON_TYPES), \
    GET_ELEMENT(2, SD_COMMON_TYPES), \
    GET_ELEMENT(3, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 3
#define SD_COMMON_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_COMMON_TYPES), \
    GET_ELEMENT(1, SD_COMMON_TYPES), \
    GET_ELEMENT(2, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 2
#define SD_COMMON_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_COMMON_TYPES), \
    GET_ELEMENT(1, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 1
#define SD_COMMON_TYPES_PART_0 (GET_ELEMENT(0, SD_COMMON_TYPES))
#else
// Fallback for empty type list - use float as safe default
#define SD_COMMON_TYPES_PART_0 ((sd::DataType::FLOAT32, float))
#endif

// Part 1: Next types if available, otherwise reuse Part 0
#if COUNT_SD_COMMON_TYPES >= 9
#define SD_COMMON_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_COMMON_TYPES), \
    GET_ELEMENT(6, SD_COMMON_TYPES), \
    GET_ELEMENT(7, SD_COMMON_TYPES), \
    GET_ELEMENT(8, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 8
#define SD_COMMON_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_COMMON_TYPES), \
    GET_ELEMENT(6, SD_COMMON_TYPES), \
    GET_ELEMENT(7, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 7
#define SD_COMMON_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_COMMON_TYPES), \
    GET_ELEMENT(6, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 6
#define SD_COMMON_TYPES_PART_1 (GET_ELEMENT(5, SD_COMMON_TYPES))
#else
// Fallback: reuse Part 0 when we don't have enough types
#define SD_COMMON_TYPES_PART_1 SD_COMMON_TYPES_PART_0
#endif

// Part 2: Even more types if available, otherwise reuse Part 0
#if COUNT_SD_COMMON_TYPES >= 13
#define SD_COMMON_TYPES_PART_2 (\
    GET_ELEMENT(9, SD_COMMON_TYPES), \
    GET_ELEMENT(10, SD_COMMON_TYPES), \
    GET_ELEMENT(11, SD_COMMON_TYPES), \
    GET_ELEMENT(12, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 12
#define SD_COMMON_TYPES_PART_2 (\
    GET_ELEMENT(9, SD_COMMON_TYPES), \
    GET_ELEMENT(10, SD_COMMON_TYPES), \
    GET_ELEMENT(11, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 11
#define SD_COMMON_TYPES_PART_2 (\
    GET_ELEMENT(9, SD_COMMON_TYPES), \
    GET_ELEMENT(10, SD_COMMON_TYPES))
#elif COUNT_SD_COMMON_TYPES == 10
#define SD_COMMON_TYPES_PART_2 (GET_ELEMENT(9, SD_COMMON_TYPES))
#else
// Fallback: reuse Part 0 when we don't have enough types
#define SD_COMMON_TYPES_PART_2 SD_COMMON_TYPES_PART_0
#endif

// Safe partitioning for SD_NUMERIC_TYPES
// Part 0: First available numeric types (up to 5)
#if COUNT_SD_NUMERIC_TYPES >= 5
#define SD_NUMERIC_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_NUMERIC_TYPES), \
    GET_ELEMENT(1, SD_NUMERIC_TYPES), \
    GET_ELEMENT(2, SD_NUMERIC_TYPES), \
    GET_ELEMENT(3, SD_NUMERIC_TYPES), \
    GET_ELEMENT(4, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 4
#define SD_NUMERIC_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_NUMERIC_TYPES), \
    GET_ELEMENT(1, SD_NUMERIC_TYPES), \
    GET_ELEMENT(2, SD_NUMERIC_TYPES), \
    GET_ELEMENT(3, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 3
#define SD_NUMERIC_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_NUMERIC_TYPES), \
    GET_ELEMENT(1, SD_NUMERIC_TYPES), \
    GET_ELEMENT(2, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 2
#define SD_NUMERIC_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_NUMERIC_TYPES), \
    GET_ELEMENT(1, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 1
#define SD_NUMERIC_TYPES_PART_0 (GET_ELEMENT(0, SD_NUMERIC_TYPES))
#else
// Fallback for empty numeric type list
#define SD_NUMERIC_TYPES_PART_0 ((sd::DataType::FLOAT32, float))
#endif

// Part 1: Next numeric types if available, otherwise reuse Part 0
#if COUNT_SD_NUMERIC_TYPES >= 9
#define SD_NUMERIC_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_NUMERIC_TYPES), \
    GET_ELEMENT(6, SD_NUMERIC_TYPES), \
    GET_ELEMENT(7, SD_NUMERIC_TYPES), \
    GET_ELEMENT(8, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 8
#define SD_NUMERIC_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_NUMERIC_TYPES), \
    GET_ELEMENT(6, SD_NUMERIC_TYPES), \
    GET_ELEMENT(7, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 7
#define SD_NUMERIC_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_NUMERIC_TYPES), \
    GET_ELEMENT(6, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 6
#define SD_NUMERIC_TYPES_PART_1 (GET_ELEMENT(5, SD_NUMERIC_TYPES))
#else
// Fallback: reuse Part 0 when we don't have enough types
#define SD_NUMERIC_TYPES_PART_1 SD_NUMERIC_TYPES_PART_0
#endif

// Part 2: Even more numeric types if available, otherwise reuse Part 0
#if COUNT_SD_NUMERIC_TYPES >= 11
#define SD_NUMERIC_TYPES_PART_2 (\
    GET_ELEMENT(9, SD_NUMERIC_TYPES), \
    GET_ELEMENT(10, SD_NUMERIC_TYPES))
#elif COUNT_SD_NUMERIC_TYPES == 10
#define SD_NUMERIC_TYPES_PART_2 (GET_ELEMENT(9, SD_NUMERIC_TYPES))
#else
// Fallback: reuse Part 0 when we don't have enough types
#define SD_NUMERIC_TYPES_PART_2 SD_NUMERIC_TYPES_PART_0
#endif

// Safe partitioning for SD_FLOAT_TYPES
#if COUNT_SD_FLOAT_TYPES >= 3
#define SD_FLOAT_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_FLOAT_TYPES), \
    GET_ELEMENT(1, SD_FLOAT_TYPES), \
    GET_ELEMENT(2, SD_FLOAT_TYPES))
#elif COUNT_SD_FLOAT_TYPES == 2
#define SD_FLOAT_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_FLOAT_TYPES), \
    GET_ELEMENT(1, SD_FLOAT_TYPES))
#elif COUNT_SD_FLOAT_TYPES == 1
#define SD_FLOAT_TYPES_PART_0 (GET_ELEMENT(0, SD_FLOAT_TYPES))
#else
#define SD_FLOAT_TYPES_PART_0 ((sd::DataType::FLOAT32, float))
#endif

#if COUNT_SD_FLOAT_TYPES >= 4
#define SD_FLOAT_TYPES_PART_1 (GET_ELEMENT(3, SD_FLOAT_TYPES))
#else
#define SD_FLOAT_TYPES_PART_1 SD_FLOAT_TYPES_PART_0
#endif

#define SD_FLOAT_TYPES_PART_2 SD_FLOAT_TYPES_PART_0
#define SD_FLOAT_TYPES_PART_3 SD_FLOAT_TYPES_PART_0

// Safe partitioning for SD_INTEGER_TYPES
#if COUNT_SD_INTEGER_TYPES >= 5
#define SD_INTEGER_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_INTEGER_TYPES), \
    GET_ELEMENT(1, SD_INTEGER_TYPES), \
    GET_ELEMENT(2, SD_INTEGER_TYPES), \
    GET_ELEMENT(3, SD_INTEGER_TYPES), \
    GET_ELEMENT(4, SD_INTEGER_TYPES))
#elif COUNT_SD_INTEGER_TYPES == 4
#define SD_INTEGER_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_INTEGER_TYPES), \
    GET_ELEMENT(1, SD_INTEGER_TYPES), \
    GET_ELEMENT(2, SD_INTEGER_TYPES), \
    GET_ELEMENT(3, SD_INTEGER_TYPES))
#elif COUNT_SD_INTEGER_TYPES == 3
#define SD_INTEGER_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_INTEGER_TYPES), \
    GET_ELEMENT(1, SD_INTEGER_TYPES), \
    GET_ELEMENT(2, SD_INTEGER_TYPES))
#elif COUNT_SD_INTEGER_TYPES == 2
#define SD_INTEGER_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_INTEGER_TYPES), \
    GET_ELEMENT(1, SD_INTEGER_TYPES))
#elif COUNT_SD_INTEGER_TYPES == 1
#define SD_INTEGER_TYPES_PART_0 (GET_ELEMENT(0, SD_INTEGER_TYPES))
#else
#define SD_INTEGER_TYPES_PART_0 ((sd::DataType::INT32, int32_t))
#endif

#if COUNT_SD_INTEGER_TYPES >= 9
#define SD_INTEGER_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_INTEGER_TYPES), \
    GET_ELEMENT(6, SD_INTEGER_TYPES), \
    GET_ELEMENT(7, SD_INTEGER_TYPES), \
    GET_ELEMENT(8, SD_INTEGER_TYPES))
#elif COUNT_SD_INTEGER_TYPES >= 6
#define SD_INTEGER_TYPES_PART_1 (\
    GET_ELEMENT(5, SD_INTEGER_TYPES))
#else
#define SD_INTEGER_TYPES_PART_1 SD_INTEGER_TYPES_PART_0
#endif

#define SD_INTEGER_TYPES_PART_2 SD_INTEGER_TYPES_PART_0

// Safe partitioning for SD_BOOL_TYPES (typically just bool)
#define SD_BOOL_TYPES_PART_0 (SD_BOOL_TYPES)
#define SD_BOOL_TYPES_PART_1 (SD_BOOL_TYPES)
#define SD_BOOL_TYPES_PART_2 (SD_BOOL_TYPES)

// Safe partitioning for SD_LONG_TYPES
#if COUNT_SD_LONG_TYPES >= 2
#define SD_LONG_TYPES_PART_0 (\
    GET_ELEMENT(0, SD_LONG_TYPES), \
    GET_ELEMENT(1, SD_LONG_TYPES))
#elif COUNT_SD_LONG_TYPES == 1
#define SD_LONG_TYPES_PART_0 (GET_ELEMENT(0, SD_LONG_TYPES))
#else
#define SD_LONG_TYPES_PART_0 ((sd::DataType::INT64, LongType))
#endif

#define SD_LONG_TYPES_PART_1 SD_LONG_TYPES_PART_0
#define SD_LONG_TYPES_PART_2 SD_LONG_TYPES_PART_0

// Callback macros for instantiation
#define INSTANT_PROCESS_COMBINATION_1(name, suffix, tuple) \
    template name<GET_SECOND_ARG tuple>suffix
    
#define INSTANT_PROCESS_COMBINATION_2(name, suffix, tuple1, tuple2) \
    template name<GET_SECOND_ARG tuple1, GET_SECOND_ARG tuple2>suffix

// Special callback for promotion
#define CALLBACK_INSTANTIATE_PROMOTE(tuple1, tuple2, name, suffix, end) \
    template void functions::pairwise_transforms::PairWiseTransform<GET_SECOND_ARG tuple1, GET_SECOND_ARG tuple2, \
        typename TypePromotion<GET_SECOND_ARG tuple1, GET_SECOND_ARG tuple2>::type>::exec( \
        int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y, \
        const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo, \
        void *extraParams, sd::LongType start, sd::LongType stop)end

#define PRINT_NUMERIC_TYPES_PARTS \
    PRINT SD_NUMERIC_TYPES_PART_2

#endif  // LIBND4J_PAIRWISE_INSTANTIATIONS_H
