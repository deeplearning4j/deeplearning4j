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

#pragma once
#ifndef OP_TYPES_H_
#define OP_TYPES_H_

#include <array/DataTypeUtils.h>
#include <helpers/shape.h>
#include <loops/ReduceType.h>
#include <loops/summarystatsreduce.h>
#include <math/templatemath.h>
#include <system/Environment.h>
#include <system/common.h>
#include <system/op_boilerplate.h>
#include <math/templatemath.h>
#include <codecvt>
#include <vector>
#include <type_traits>

#include "helpers/unicode.h"

// =============================================================================
// ESSENTIAL MACROS AND CONSTANTS
// =============================================================================

#define COMMA ,

#define SELU_ALPHA 1.6732632423543772848170429916717
#define SELU_LAMBDA 1.0507009873554804934193349852946
#define SD_STRING_ASSIGN_TEMP_BUFFER_BYTES 256

#ifdef __CUDACC__
#define __CUDACC_ONLY__
#else
#define __CUDACC_ONLY__ template<bool = false, typename = typename std::enable_if<!false>::type>
#endif

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================

namespace functions {
namespace indexreduce {
template <typename T>
struct IndexValue {
  T value;
  sd::LongType index;
  SD_HOST_DEVICE IndexValue() = default;
  SD_HOST_DEVICE IndexValue(const T val, const sd::LongType ind) : value(val), index(ind) {}
};
}  // namespace indexreduce

namespace summarystats {
template <typename T>
class SummaryStatsData;
}
}  // namespace functions

namespace simdOps {

// =============================================================================
// SIMD SUPPORT TYPE TRAITS
// =============================================================================

/**
 * @brief Type trait to determine if a type supports SIMD operations as a return type
 */
template<typename T>
struct is_simd_unsupported_return_type {
  static constexpr bool value = std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>;
};

/**
 * @brief Type trait to determine if a type supports SIMD operations as an argument type
 */
template<typename T>
struct is_simd_unsupported_argument_type {
  static constexpr bool value = std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>;
};

/**
 * @brief Type trait to determine if any type in a parameter pack is SIMD unsupported
 */
template<typename... Types>
struct has_simd_unsupported_types {
  static constexpr bool value = (is_simd_unsupported_argument_type<Types>::value || ...);
};

/**
 * @brief Type trait for SIMD-safe types (native arithmetic types)
 */
template<typename T>
struct is_simd_native {
  static constexpr bool value = std::is_same_v<T, float> ||
                                std::is_same_v<T, double> ||
                                std::is_integral_v<T>;
};

// =============================================================================
// SIMD OPERATION DISPATCHING HELPERS
// =============================================================================

/**
 * @brief SFINAE helper for SIMD-safe operation dispatch
 */
template<typename RetType, typename... ArgTypes>
using enable_if_simd_safe = typename std::enable_if<
    !is_simd_unsupported_return_type<RetType>::value &&
        !has_simd_unsupported_types<ArgTypes...>::value,
    RetType
    >::type;

/**
 * @brief SFINAE helper for non-SIMD operation dispatch
 */
template<typename RetType, typename... ArgTypes>
using enable_if_simd_unsafe = typename std::enable_if<
    is_simd_unsupported_return_type<RetType>::value ||
        has_simd_unsupported_types<ArgTypes...>::value,
    RetType
    >::type;

// =============================================================================
// FUNCTION ATTRIBUTE MACROS
// =============================================================================

/**
 * @brief Macro for SIMD-safe function declarations
 */
#define SD_SIMD_SAFE SD_OP_DEF static SD_INLINE

/**
 * @brief Macro for non-SIMD function declarations
 */
#define SD_NON_SIMD static SD_INLINE

/**
 * @brief Macro for logic function declarations (always non-SIMD)
 */
#define SD_LOGIC_FUNC static SD_INLINE

/**
 * @brief Macro for public operation function declarations
 */
#define SD_OP_FUNC static

// =============================================================================
// CONDITIONAL SIMD DISPATCH MACROS
// =============================================================================

/**
 * @brief Conditional SIMD dispatch for unary operations
 */
#define DISPATCH_SIMD_UNARY(RetType, ArgType, simd_call, logic_call) \
  if constexpr (simdOps::is_simd_unsupported_return_type<RetType>::value || \
                simdOps::is_simd_unsupported_argument_type<ArgType>::value) \
    return logic_call; \
  else \
    return simd_call;

/**
 * @brief Conditional SIMD dispatch for binary operations
 */
#define DISPATCH_SIMD_BINARY(RetType, ArgType1, ArgType2, simd_call, logic_call) \
  if constexpr (simdOps::is_simd_unsupported_return_type<RetType>::value || \
                simdOps::is_simd_unsupported_argument_type<ArgType1>::value || \
                simdOps::is_simd_unsupported_argument_type<ArgType2>::value) \
    return logic_call; \
  else \
    return simd_call;

/**
 * @brief Conditional SIMD dispatch for ternary operations
 */
#define DISPATCH_SIMD_TERNARY(RetType, ArgType1, ArgType2, ArgType3, simd_call, logic_call) \
  if constexpr (simdOps::is_simd_unsupported_return_type<RetType>::value || \
                simdOps::is_simd_unsupported_argument_type<ArgType1>::value || \
                simdOps::is_simd_unsupported_argument_type<ArgType2>::value || \
                simdOps::is_simd_unsupported_argument_type<ArgType3>::value) \
    return logic_call; \
  else \
    return simd_call;

// =============================================================================
// AGGREGATE TYPE HELPERS
// =============================================================================

/**
 * @brief AggregateType - helper template to use desired type for aggregation expressions.
 * This way we can reduce overflow and precision issues for certain types
 */
template <typename Z>
struct AggregateType {
  using type = Z;
};

template <>
struct AggregateType<float16> {
  using type = float;
};

template <>
struct AggregateType<bfloat16> {
  using type = float;
};

// =============================================================================
// NO-OP SPECIAL EXECUTION MACROS
// =============================================================================

#define DECLARE_NO_SPECIAL_EXECUTION(X_TYPE, Z_TYPE) \
 static const bool requiresSpecial = false; \
 static void execSpecial(const X_TYPE *dx, const sd::LongType *xShapeBuffer, Z_TYPE *result, \
                         const sd::LongType *resultShapeBuffer, void *extraParams, \
                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {} \
 static void execSpecial(const X_TYPE *dx, const sd::LongType *xShapeBuffer, Z_TYPE *result, \
                         const sd::LongType *resultShapeBuffer, X_TYPE *extraParams, \
                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {} \
 template<typename U = X_TYPE, typename V = Z_TYPE> \
 static typename std::enable_if<!std::is_same_v<U, V>, void>::type execSpecial( \
     const X_TYPE *dx, const sd::LongType *xShapeBuffer, Z_TYPE *result, \
     const sd::LongType *resultShapeBuffer, Z_TYPE *extraParams, \
     const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}

#ifdef __CUDACC__
#define DECLARE_NO_SPECIAL_CUDA(X_TYPE, Z_TYPE) \
 static SD_DEVICE void execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeBuffer, Z_TYPE *result, const sd::LongType *resultShapeBuffer, \
     void *extraParams, sd::LongType *allocationPointer, Z_TYPE *reductionPointer, \
     const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {} \
 static SD_DEVICE void execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeBuffer, Z_TYPE *result, const sd::LongType *resultShapeBuffer, \
     X_TYPE *extraParams, sd::LongType *allocationPointer, Z_TYPE *reductionPointer, \
     const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {} \
 template<typename U = X_TYPE, typename V = Z_TYPE> \
 static typename std::enable_if<!std::is_same_v<U, V>, void>::type execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeBuffer, Z_TYPE *result, const sd::LongType *resultShapeBuffer, \
     Z_TYPE *extraParams, sd::LongType *allocationPointer, Z_TYPE *reductionPointer, \
     const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}

#define DECLARE_NO_SPECIAL_ACCUMULATION_CUDA(X_TYPE, Z_TYPE) \
 static SD_INLINE SD_DEVICE void execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeInfo, void *extraParams, Z_TYPE *result, \
     const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, \
     Z_TYPE *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {} \
 static SD_INLINE SD_DEVICE void execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeInfo, Z_TYPE *extraParams, Z_TYPE *result, \
     const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, \
     Z_TYPE *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {} \
 template<typename U = X_TYPE, typename V = Z_TYPE> \
 static typename std::enable_if<!std::is_same_v<U, V>, void>::type execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeInfo, X_TYPE *extraParams, Z_TYPE *result, \
     const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, \
     Z_TYPE *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {} \
 static SD_INLINE SD_DEVICE void execSpecialCuda( \
     const X_TYPE *dx, const sd::LongType *xShapeInfo, sd::LongType *extraParams, Z_TYPE *result, \
     const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength, \
     Z_TYPE *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {}
#else
#define DECLARE_NO_SPECIAL_CUDA(X_TYPE, Z_TYPE)
#define DECLARE_NO_SPECIAL_ACCUMULATION_CUDA(X_TYPE, Z_TYPE)
#endif

#define DECLARE_NO_SPECIAL_ACCUMULATION(X_TYPE, Z_TYPE) \
 static const bool requiresSpecialAccumulation = false; \
 static void execSpecial(const X_TYPE *x, const sd::LongType *xShapeInfo, void *extraParams, Z_TYPE *result, \
                         const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {} \
 static void execSpecial(const X_TYPE *x, const sd::LongType *xShapeInfo, Z_TYPE *extraParams, Z_TYPE *result, \
                         const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {} \
 template<typename U = X_TYPE, typename V = Z_TYPE> \
 static typename std::enable_if<!std::is_same_v<U, V>, void>::type execSpecial( \
     const X_TYPE *x, const sd::LongType *xShapeInfo, X_TYPE *extraParams, Z_TYPE *result, \
     const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
     const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {} \
 static void execSpecial(const X_TYPE *x, const sd::LongType *xShapeInfo, sd::LongType *extraParams, Z_TYPE *result, \
                         const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}

#define DECLARE_NO_SPECIAL_ACCUMULATION(X_TYPE, Z_TYPE) \
 static const bool requiresSpecialAccumulation = false; \
 \
 static void execSpecial(const X_TYPE *x, const sd::LongType *xShapeInfo, void *extraParams, Z_TYPE *result, \
                         const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                         const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {} \
 \
 template<typename ParamType> \
 static typename std::enable_if<!std::is_same_v<ParamType, void>, void>::type execSpecial( \
     const X_TYPE *x, const sd::LongType *xShapeInfo, ParamType *extraParams, Z_TYPE *result, \
     const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
     const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}

// Simplified macro definitions using the DECLARE_* macros above
#define no_op_exec_special_any DECLARE_NO_SPECIAL_EXECUTION(X, Z)
#define no_op_exec_special_bool DECLARE_NO_SPECIAL_EXECUTION(X, Z)
#define no_op_exec_special_same DECLARE_NO_SPECIAL_EXECUTION(X, X)
#define no_op_exec_special DECLARE_NO_SPECIAL_EXECUTION(X, Z)

#define no_op_exec_special_accumulation DECLARE_NO_SPECIAL_ACCUMULATION(X, Z)
#define no_op_exec_special_accumulation_long DECLARE_NO_SPECIAL_ACCUMULATION(X, Z)
#define no_op_exec_special_accumulation_same DECLARE_NO_SPECIAL_ACCUMULATION(X, X)

#define no_op_exec_special_any_cuda DECLARE_NO_SPECIAL_CUDA(X, Z)
#define no_op_exec_special_bool_cuda DECLARE_NO_SPECIAL_CUDA(X, Z)
#define no_op_exec_special_same_cuda DECLARE_NO_SPECIAL_CUDA(X, X)
#define no_op_exec_special_cuda DECLARE_NO_SPECIAL_CUDA(X, Z)

#define no_op_exec_special_accumulation_same_cuda DECLARE_NO_SPECIAL_ACCUMULATION_CUDA(X, X)
#define no_op_exec_special_accumulation_long_cuda DECLARE_NO_SPECIAL_ACCUMULATION_CUDA(X, Z)
#define no_op_exec_special_accumulation_cuda DECLARE_NO_SPECIAL_ACCUMULATION_CUDA(X, Z)

} // namespace simdOps

#endif // OP_TYPES_H_