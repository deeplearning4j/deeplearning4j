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
#ifndef OP_MACROS_REDUCE_H_
#define OP_MACROS_REDUCE_H_

#include "op_types.h"

namespace simdOps {

// =============================================================================
// BASIC REDUCE OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a basic reduce operation with proper SIMD handling
 */
#define DECLARE_REDUCE_OP(OP_NAME, REDUCE_TYPE_VAL, STARTING_VAL, MERGE_OP, UPDATE_OP, POST_PROCESS) \
  template <typename X>                                                                           \
  class OP_NAME {                                                                                 \
   private:                                                                                       \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* extraParams) { return d1; }                            \
    SD_HOST_DEVICE SD_INLINE static  X merge_logic(X old, X opOutput, X* extraParams) { return MERGE_OP; }       \
    SD_HOST_DEVICE SD_INLINE static  X update_logic(X old, X opOutput, X* extraParams) { return UPDATE_OP; }     \
    SD_HOST_DEVICE SD_INLINE static  X postProcess_logic(X reduction, sd::LongType n, X* extraParams) { return POST_PROCESS; } \
    static SD_HOST_DEVICE X op_simd(X d1, X* extraParams) { return op_logic(d1, extraParams); }                 \
    static SD_HOST_DEVICE X merge_simd(X old, X opOutput, X* extraParams) { return merge_logic(old, opOutput, extraParams); } \
    static SD_HOST_DEVICE X update_simd(X old, X opOutput, X* extraParams) { return update_logic(old, opOutput, extraParams); } \
    static SD_HOST_DEVICE X postProcess_simd(X reduction, sd::LongType n, X* extraParams) {                      \
      return postProcess_logic(reduction, n, extraParams);                                        \
    }                                                                                             \
                                                                                                  \
   public:                                                                                        \
    no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda;             \
    const static  functions::ReduceType reduceType = functions::ReduceType::REDUCE_TYPE_VAL;     \
    static SD_HOST_DEVICE X startingValue(const X* input) { return STARTING_VAL; }                             \
    static SD_HOST_DEVICE X merge(X old, X opOutput, X* extraParams) {                                         \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                         \
        return merge_logic(old, opOutput, extraParams);                                         \
      else                                                                                       \
        return merge_simd(old, opOutput, extraParams);                                          \
    }                                                                                            \
    static SD_HOST_DEVICE X update(X old, X opOutput, X* extraParams) {                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                         \
        return update_logic(old, opOutput, extraParams);                                        \
      else                                                                                       \
        return update_simd(old, opOutput, extraParams);                                         \
    }                                                                                            \
    static SD_HOST_DEVICE X op(X d1, X* extraParams) {                                                         \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value) return op_logic(d1, extraParams); \
      else return op_simd(d1, extraParams);                                                     \
    }                                                                                            \
    static SD_HOST_DEVICE X postProcess(X reduction, sd::LongType n, X* extraParams) {                         \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                         \
        return postProcess_logic(reduction, n, extraParams);                                    \
      else                                                                                       \
        return postProcess_simd(reduction, n, extraParams);                                     \
    }                                                                                            \
  };

// =============================================================================
// FLOAT REDUCE OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a reduce operation with float aggregation and proper SIMD handling
 */
#define DECLARE_REDUCE_FLOAT_OP(OP_NAME, REDUCE_TYPE_VAL, STARTING_VAL, MERGE_OP, UPDATE_OP, POST_PROCESS) \
  template <typename X, typename Z>                                                            \
  class OP_NAME {                                                                              \
   public:                                                                                     \
    using InterType = typename AggregateType<Z>::type;                                        \
                                                                                               \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  Z postProcess_logic(InterType reduction, sd::LongType n, Z* extraParams) { \
      return POST_PROCESS;                                                                     \
    }                                                                                          \
    SD_HOST_DEVICE SD_INLINE static  InterType merge_logic(InterType old, InterType opOutput, Z* extraParams) { return MERGE_OP; } \
    SD_HOST_DEVICE SD_INLINE static  InterType update_logic(InterType old, InterType opOutput, Z* extraParams) { return UPDATE_OP; } \
    SD_HOST_DEVICE SD_INLINE static  InterType op_logic(X d1, Z* extraParams) { return static_cast<InterType>(d1); } \
    static SD_HOST_DEVICE Z postProcess_simd(InterType reduction, sd::LongType n, Z* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                    \
    }                                                                                          \
    static SD_HOST_DEVICE InterType merge_simd(InterType old, InterType opOutput, Z* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                         \
    }                                                                                          \
    static SD_HOST_DEVICE InterType update_simd(InterType old, InterType opOutput, Z* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                        \
    }                                                                                          \
    static SD_HOST_DEVICE InterType op_simd(X d1, Z* extraParams) { return op_logic(d1, extraParams); } \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda;                    \
    const static  functions::ReduceType reduceType = functions::ReduceType::REDUCE_TYPE_VAL;  \
    static SD_HOST_DEVICE X startingValue(const X* input) { return STARTING_VAL; }                          \
    static SD_HOST_DEVICE InterType merge(InterType old, InterType opOutput, Z* extraParams) {              \
      if constexpr (simdOps::is_simd_unsupported_return_type<InterType>::value)              \
        return merge_logic(old, opOutput, extraParams);                                      \
      else                                                                                    \
        return merge_simd(old, opOutput, extraParams);                                       \
    }                                                                                         \
    static SD_HOST_DEVICE InterType update(InterType old, InterType opOutput, Z* extraParams) {             \
      if constexpr (simdOps::is_simd_unsupported_return_type<InterType>::value)              \
        return update_logic(old, opOutput, extraParams);                                     \
      else                                                                                    \
        return update_simd(old, opOutput, extraParams);                                      \
    }                                                                                         \
    template<typename T>                                                                     \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType>, InterType>::type update(T old, InterType opOutput, Z* extraParams) { \
      return update(static_cast<InterType>(old), opOutput, extraParams);                    \
    }                                                                                        \
    template<typename T>                                                                     \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType>, InterType>::type update(InterType old, T opOutput, Z* extraParams) { \
      return update(old, static_cast<InterType>(opOutput), extraParams);                    \
    }                                                                                        \
    template<typename T, typename U>                                                        \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType> && !std::is_same_v<U COMMA InterType>, InterType>::type update(T old, U opOutput, Z* extraParams) { \
      return update(static_cast<InterType>(old), static_cast<InterType>(opOutput), extraParams); \
    }                                                                                        \
    template<typename U = X, typename V = Z>                                                \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, InterType>::type update(InterType old, InterType opOutput, X* extraParams) { \
      return update(old, opOutput, reinterpret_cast<Z*>(extraParams));                     \
    }                                                                                        \
    static SD_HOST_DEVICE InterType op(X d1, Z* extraParams) {                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<InterType>::value ||           \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                   \
        return op_logic(d1, extraParams);                                                   \
      else                                                                                   \
        return op_simd(d1, extraParams);                                                    \
    }                                                                                        \
    template<typename U = X, typename V = Z>                                                \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, InterType>::type op(X d1, X* extraParams) { \
      return op(d1, reinterpret_cast<Z*>(extraParams));                                     \
    }                                                                                        \
    static SD_HOST_DEVICE Z postProcess(InterType reduction, sd::LongType n, Z* extraParams) {             \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                     \
        return postProcess_logic(reduction, n, extraParams);                                \
      else                                                                                   \
        return postProcess_simd(reduction, n, extraParams);                                 \
    }                                                                                        \
    template<typename T>                                                                     \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType>, Z>::type postProcess(T reduction, sd::LongType n, Z* extraParams) { \
      return postProcess(static_cast<InterType>(reduction), n, extraParams);               \
    }                                                                                        \
    template<typename U = X, typename V = Z>                                                \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type postProcess(InterType reduction, sd::LongType n, X* extraParams) { \
      return postProcess(reduction, n, reinterpret_cast<Z*>(extraParams));                 \
    }                                                                                        \
    template<typename T, typename U = X, typename V = Z>                                    \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType> && !std::is_same_v<U COMMA V>, Z>::type postProcess(T reduction, sd::LongType n, X* extraParams) { \
      return postProcess(static_cast<InterType>(reduction), n, reinterpret_cast<Z*>(extraParams)); \
    }                                                                                        \
  };


/**
 * @brief Fixed DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT macro
 * Uses conditional compilation to include opAtomic only when atomics are available
 */
/**
 * @brief DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT macro with conditional opAtomic
 * Two versions: one with atomics for CUDA, one without for host
 */

#ifdef __CUDACC__
#define DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT(OP_NAME, BOOL_LOGIC, NORMAL_LOGIC, EXTRA_PARAMS_LEN, STARTING_VAL, POST_PROCESS) \
  template <typename X, typename Y>                                                            \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static Y op_logic(X d1, X d2, Y* extraParamsRef) {               \
      Y* extraParams = extraParamsRef;                                                        \
      if constexpr (std::is_same_v<X COMMA bool>) {                                           \
        BOOL_LOGIC                                                                             \
      } else {                                                                                 \
        NORMAL_LOGIC                                                                           \
      }                                                                                        \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y startingValue_logic(const X* input) {                   \
      return static_cast<Y>(STARTING_VAL);                                                    \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y postProcess_logic(Y reduction, sd::LongType n, Y* extraParamsRef) { \
      Y* extraParams = extraParamsRef;                                                        \
      return POST_PROCESS;                                                                     \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y update_logic(Y old, Y opOutput, Y* extraParamsRef) {    \
      return old + opOutput;                                                                   \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y merge_logic(Y old, Y opOutput, Y* extraParamsRef) {     \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TX> op_simd(TX d1, TX d2, TY* extraParamsRef) { \
      return op_logic(d1, d2, extraParamsRef);                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TX> op_simd(TX d1, TX d2, TY* extraParamsRef) { \
      return op_logic(d1, d2, extraParamsRef);                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TX> startingValue_simd(const TX* input) { \
      return startingValue_logic(input);                                                      \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TX> startingValue_simd(const TX* input) { \
      return startingValue_logic(input);                                                      \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TY> postProcess_simd(TY reduction, sd::LongType n, TY* extraParamsRef) { \
      return postProcess_logic(reduction, n, extraParamsRef);                                 \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TY> postProcess_simd(TY reduction, sd::LongType n, TY* extraParamsRef) { \
      return postProcess_logic(reduction, n, extraParamsRef);                                 \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TY> update_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TY> update_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TY> merge_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return merge_logic(old, opOutput, extraParamsRef);                                      \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TY> merge_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return merge_logic(old, opOutput, extraParamsRef);                                      \
    }                                                                                          \
                                                                                               \
   public:                                                                                     \
    static const int extraParamsLen = EXTRA_PARAMS_LEN;                                       \
    static SD_HOST_DEVICE X *generateExtraParams() { return nullptr; }                       \
    static SD_HOST_DEVICE void finalizeExtraParams(X *extraParamsRef) {}                     \
    static SD_HOST_DEVICE void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) { \
      for (int i = 0; i < EXTRA_PARAMS_LEN; i++) {                                            \
        extraParamsTotal[i] += extraParamsLocal[i];                                           \
      }                                                                                        \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y startingValue(const X* input) {                                   \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return startingValue_logic(input);                                                    \
      else                                                                                     \
        return startingValue_simd(input);                                                     \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y postProcess(Y reduction, sd::LongType n, Y* extraParamsRef) {     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                     \
        return postProcess_logic(reduction, n, extraParamsRef);                              \
      else                                                                                     \
        return postProcess_simd(reduction, n, extraParamsRef);                               \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y op(X d1, X d2, Y* extraParamsRef) {                               \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return op_logic(d1, d2, extraParamsRef);                                              \
      else                                                                                     \
        return op_simd(d1, d2, extraParamsRef);                                               \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y update(Y old, Y opOutput, Y* extraParamsRef) {                    \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value)                       \
        return update_logic(old, opOutput, extraParamsRef);                                   \
      else                                                                                     \
        return update_simd(old, opOutput, extraParamsRef);                                    \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y merge(Y old, Y opOutput, Y* extraParamsRef) {                     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value)                       \
        return merge_logic(old, opOutput, extraParamsRef);                                    \
      else                                                                                     \
        return merge_simd(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    static SD_DEVICE Y opAtomic(X d1, X d2, Y* extraParamsRef) {                              \
      if constexpr (std::is_same_v<X COMMA bool>) {                                           \
        Y val1 = static_cast<Y>(static_cast<int>(d1) * static_cast<int>(d1));                \
        Y val2 = static_cast<Y>(static_cast<int>(d2) * static_cast<int>(d2));                \
        sd::math::atomics::sd_atomicAdd(&extraParamsRef[0], val1);                           \
        sd::math::atomics::sd_atomicAdd(&extraParamsRef[1], val2);                           \
        return static_cast<Y>(static_cast<int>(d1) * static_cast<int>(d2));                  \
      } else {                                                                                 \
        Y val1 = static_cast<Y>(sd::math::sd_abs<X COMMA X>(d1) * sd::math::sd_abs<X COMMA X>(d1)); \
        Y val2 = static_cast<Y>(sd::math::sd_abs<X COMMA X>(d2) * sd::math::sd_abs<X COMMA X>(d2)); \
        sd::math::atomics::sd_atomicAdd(&extraParamsRef[0], val1);                           \
        sd::math::atomics::sd_atomicAdd(&extraParamsRef[1], val2);                           \
        return static_cast<Y>(d1 * d2);                                                       \
      }                                                                                        \
    }                                                                                          \
  };

#else
#define DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT(OP_NAME, BOOL_LOGIC, NORMAL_LOGIC, EXTRA_PARAMS_LEN, STARTING_VAL, POST_PROCESS) \
  template <typename X, typename Y>                                                            \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static Y op_logic(X d1, X d2, Y* extraParamsRef) {               \
      Y* extraParams = extraParamsRef;                                                        \
      if constexpr (std::is_same_v<X COMMA bool>) {                                           \
        BOOL_LOGIC                                                                             \
      } else {                                                                                 \
        NORMAL_LOGIC                                                                           \
      }                                                                                        \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y startingValue_logic(const X* input) {                   \
      return static_cast<Y>(STARTING_VAL);                                                    \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y postProcess_logic(Y reduction, sd::LongType n, Y* extraParamsRef) { \
      Y* extraParams = extraParamsRef;                                                        \
      return POST_PROCESS;                                                                     \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y update_logic(Y old, Y opOutput, Y* extraParamsRef) {    \
      return old + opOutput;                                                                   \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Y merge_logic(Y old, Y opOutput, Y* extraParamsRef) {     \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TX> op_simd(TX d1, TX d2, TY* extraParamsRef) { \
      return op_logic(d1, d2, extraParamsRef);                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TX> op_simd(TX d1, TX d2, TY* extraParamsRef) { \
      return op_logic(d1, d2, extraParamsRef);                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TX> startingValue_simd(const TX* input) { \
      return startingValue_logic(input);                                                      \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TX> startingValue_simd(const TX* input) { \
      return startingValue_logic(input);                                                      \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TY> postProcess_simd(TY reduction, sd::LongType n, TY* extraParamsRef) { \
      return postProcess_logic(reduction, n, extraParamsRef);                                 \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TY> postProcess_simd(TY reduction, sd::LongType n, TY* extraParamsRef) { \
      return postProcess_logic(reduction, n, extraParamsRef);                                 \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TY> update_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TY> update_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_safe<TY COMMA TY> merge_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return merge_logic(old, opOutput, extraParamsRef);                                      \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    SD_HOST_DEVICE SD_INLINE static enable_if_simd_unsafe<TY COMMA TY> merge_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return merge_logic(old, opOutput, extraParamsRef);                                      \
    }                                                                                          \
                                                                                               \
   public:                                                                                     \
    static const int extraParamsLen = EXTRA_PARAMS_LEN;                                       \
    static SD_HOST_DEVICE X *generateExtraParams() { return nullptr; }                       \
    static SD_HOST_DEVICE void finalizeExtraParams(X *extraParamsRef) {}                     \
    static SD_HOST_DEVICE void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) { \
      for (int i = 0; i < EXTRA_PARAMS_LEN; i++) {                                            \
        extraParamsTotal[i] += extraParamsLocal[i];                                           \
      }                                                                                        \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y startingValue(const X* input) {                                   \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return startingValue_logic(input);                                                    \
      else                                                                                     \
        return startingValue_simd(input);                                                     \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y postProcess(Y reduction, sd::LongType n, Y* extraParamsRef) {     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                     \
        return postProcess_logic(reduction, n, extraParamsRef);                              \
      else                                                                                     \
        return postProcess_simd(reduction, n, extraParamsRef);                               \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y op(X d1, X d2, Y* extraParamsRef) {                               \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return op_logic(d1, d2, extraParamsRef);                                              \
      else                                                                                     \
        return op_simd(d1, d2, extraParamsRef);                                               \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y update(Y old, Y opOutput, Y* extraParamsRef) {                    \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value)                       \
        return update_logic(old, opOutput, extraParamsRef);                                   \
      else                                                                                     \
        return update_simd(old, opOutput, extraParamsRef);                                    \
    }                                                                                          \
                                                                                               \
    static SD_HOST_DEVICE Y merge(Y old, Y opOutput, Y* extraParamsRef) {                     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value)                       \
        return merge_logic(old, opOutput, extraParamsRef);                                    \
      else                                                                                     \
        return merge_simd(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
  };
#endif


#define DECLARE_REDUCE_SIMD_SAFE_OP(OP_NAME, OPERATION)                                             \
  template <typename X, typename Z>                                                             \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    SD_HOST_DEVICE SD_INLINE static  Z op_logic(X d1, Z* params) {                                             \
      OPERATION                                                                                 \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TZ COMMA TX> op_simd(TX d1, TZ* params) {             \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TZ COMMA TX> op_simd(TX d1, TZ* params) {           \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special_bool no_op_exec_special_bool_cuda                                        \
    no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda                       \
                                                                                                \
    static SD_HOST_DEVICE Z op(X d1, Z* params) {                                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                      \
        return op_logic(d1, params);                                                           \
      else                                                                                      \
        return op_simd(d1, params);                                                            \
    }                                                                                           \
                                                                                                \
    template<typename U = X, typename V = Z>                                                   \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type op(X d1, X* extraParams) { \
      return op(d1, reinterpret_cast<Z*>(extraParams));                                        \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE X startingValue(const X* input) { return static_cast<X>(0); }                       \
                                                                                                \
    static SD_HOST_DEVICE Z merge(Z old, Z opOutput, Z* extraParams) { return opOutput + old; }               \
                                                                                                \
    static SD_HOST_DEVICE Z update(Z old, Z opOutput, Z* extraParams) { return opOutput + old; }              \
                                                                                                \
    template<typename U = X, typename V = Z>                                                   \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type update(Z old, Z opOutput, X* extraParams) { \
      return update(old, opOutput, reinterpret_cast<Z*>(extraParams));                         \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE Z postProcess(Z reduction, sd::LongType n, Z* extraParams) { return reduction; }    \
                                                                                                \
    template<typename U = X, typename V = Z>                                                   \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type postProcess(Z reduction, sd::LongType n, X* extraParams) { \
      return postProcess(reduction, n, reinterpret_cast<Z*>(extraParams));                     \
    }                                                                                           \
  };

#define DECLARE_COMPLEX_ACCUMULATION_SIMD_SAFE_OP(OP_NAME, OP_LOGIC, OP_BINARY_LOGIC, OP_BINARY_NO_PARAMS_LOGIC, REDUCE_TYPE_VAL, STARTING_VAL, MERGE_OP, UPDATE_OP, POST_PROCESS) \
  template <typename X>                                                                          \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) {                                             \
      OP_LOGIC                                                                                  \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  X op_binary_logic(X d1, X d2, X* params) {                                \
      OP_BINARY_LOGIC                                                                           \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  X op_binary_no_params_logic(X d1, X d2) {                                 \
      OP_BINARY_NO_PARAMS_LOGIC                                                                 \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  X merge_logic(X old, X opOutput, X* extraParams) {                        \
      return MERGE_OP;                                                                          \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  X update_logic(X old, X opOutput, X* extraParams) {                       \
      return UPDATE_OP;                                                                         \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  X postProcess_logic(X reduction, sd::LongType n, X* extraParams) {        \
      return POST_PROCESS;                                                                      \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TX COMMA TX> op_simd(TX d1, TX* params) {             \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TX COMMA TX> op_simd(TX d1, TX* params) {           \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TX COMMA TX> op_binary_simd(TX d1, TX d2, TX* params) { \
      return op_binary_logic(d1, d2, params);                                                  \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TX COMMA TX> op_binary_simd(TX d1, TX d2, TX* params) { \
      return op_binary_logic(d1, d2, params);                                                  \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TX COMMA TX> op_binary_no_params_simd(TX d1, TX d2) { \
      return op_binary_no_params_logic(d1, d2);                                                \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TX COMMA TX> op_binary_no_params_simd(TX d1, TX d2) { \
      return op_binary_no_params_logic(d1, d2);                                                \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TX COMMA TX> merge_simd(TX old, TX opOutput, TX* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                          \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TX COMMA TX> merge_simd(TX old, TX opOutput, TX* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                          \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TX COMMA TX> update_simd(TX old, TX opOutput, TX* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TX COMMA TX> update_simd(TX old, TX opOutput, TX* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TX COMMA TX> postProcess_simd(TX reduction, sd::LongType n, TX* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                     \
    }                                                                                           \
                                                                                                \
    template<typename TX = X>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TX COMMA TX> postProcess_simd(TX reduction, sd::LongType n, TX* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                     \
    }                                                                                           \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda             \
    const static  functions::ReduceType reduceType = functions::ReduceType::REDUCE_TYPE_VAL;    \
                                                                                                \
    static SD_HOST_DEVICE X startingValue(const X* input) { return STARTING_VAL; }                            \
                                                                                                \
    static SD_HOST_DEVICE X op(X d1, X* extraParams) {                                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return op_logic(d1, extraParams);                                                      \
      else                                                                                      \
        return op_simd(d1, extraParams);                                                       \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE X op(X d1, X d2, X* params) {                                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return op_binary_logic(d1, d2, params);                                                \
      else                                                                                      \
        return op_binary_simd(d1, d2, params);                                                 \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE X op(X d1, X d2) {                                                                  \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return op_binary_no_params_logic(d1, d2);                                              \
      else                                                                                      \
        return op_binary_no_params_simd(d1, d2);                                               \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE X merge(X old, X opOutput, X* extraParams) {                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return merge_logic(old, opOutput, extraParams);                                        \
      else                                                                                      \
        return merge_simd(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE X update(X old, X opOutput, X* extraParams) {                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return update_logic(old, opOutput, extraParams);                                       \
      else                                                                                      \
        return update_simd(old, opOutput, extraParams);                                        \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE X postProcess(X reduction, sd::LongType n, X* extraParams) {                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return postProcess_logic(reduction, n, extraParams);                                   \
      else                                                                                      \
        return postProcess_simd(reduction, n, extraParams);                                    \
    }                                                                                           \
  };

// =============================================================================

/**
 * @brief DECLARE_ACCUMULATION_SIMD_SAFE_OP macro
 * Add this to op_macros_reduce.h
 *
 * This macro creates accumulation operations with proper SIMD handling and InterType support
 */
#define DECLARE_ACCUMULATION_SIMD_SAFE_OP(OP_NAME, OPERATION, REDUCE_TYPE_VAL, STARTING_VAL, MERGE_OP, UPDATE_OP, POST_PROCESS) \
  template <typename X, typename Z>                                                             \
  class OP_NAME {                                                                               \
   public:                                                                                      \
    using InterType = typename AggregateType<Z>::type;                                         \
                                                                                                \
   private:                                                                                     \
    SD_HOST_DEVICE SD_INLINE static  InterType op_logic(X d1, Z* params) {                                     \
      OPERATION                                                                                 \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  InterType merge_logic(InterType old, InterType opOutput, Z* extraParams) { \
      return MERGE_OP;                                                                          \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  InterType update_logic(InterType old, InterType opOutput, Z* extraParams) { \
      return UPDATE_OP;                                                                         \
    }                                                                                           \
                                                                                                \
    SD_HOST_DEVICE SD_INLINE static  Z postProcess_logic(InterType reduction, sd::LongType n, Z* extraParams) { \
      return static_cast<Z>(POST_PROCESS);                                                      \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<typename AggregateType<TZ>::type COMMA TX> op_simd(TX d1, TZ* params) { \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<typename AggregateType<TZ>::type COMMA TX> op_simd(TX d1, TZ* params) { \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TInterType = InterType>                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TInterType COMMA TInterType> merge_simd(TInterType old, TInterType opOutput, Z* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                          \
    }                                                                                           \
                                                                                                \
    template<typename TInterType = InterType>                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TInterType COMMA TInterType> merge_simd(TInterType old, TInterType opOutput, Z* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                          \
    }                                                                                           \
                                                                                                \
    template<typename TInterType = InterType>                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TInterType COMMA TInterType> update_simd(TInterType old, TInterType opOutput, Z* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    template<typename TInterType = InterType>                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TInterType COMMA TInterType> update_simd(TInterType old, TInterType opOutput, Z* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_safe<TZ COMMA TZ> postProcess_simd(InterType reduction, sd::LongType n, TZ* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                     \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    SD_HOST_DEVICE SD_INLINE static  enable_if_simd_unsafe<TZ COMMA TZ> postProcess_simd(InterType reduction, sd::LongType n, TZ* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                     \
    }                                                                                           \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda                       \
    const static  functions::ReduceType reduceType = functions::ReduceType::REDUCE_TYPE_VAL;    \
                                                                                                \
    static SD_HOST_DEVICE X startingValue(const X* input) { return STARTING_VAL; }                            \
                                                                                                \
    static SD_HOST_DEVICE InterType merge(InterType old, InterType opOutput, Z* extraParams) {                \
      if constexpr (simdOps::is_simd_unsupported_return_type<InterType>::value)                 \
        return merge_logic(old, opOutput, extraParams);                                        \
      else                                                                                      \
        return merge_simd(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    static SD_HOST_DEVICE InterType update(InterType old, InterType opOutput, Z* extraParams) {               \
      if constexpr (simdOps::is_simd_unsupported_return_type<InterType>::value)                 \
        return update_logic(old, opOutput, extraParams);                                       \
      else                                                                                      \
        return update_simd(old, opOutput, extraParams);                                        \
    }                                                                                           \
                                                                                                \
    template<typename T>                                                                       \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType>, InterType>::type update(T old, InterType opOutput, Z* extraParams) { \
      return update(static_cast<InterType>(old), opOutput, extraParams);                      \
    }                                                                                          \
                                                                                                \
    template<typename T>                                                                       \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType>, InterType>::type update(InterType old, T opOutput, Z* extraParams) { \
      return update(old, static_cast<InterType>(opOutput), extraParams);                      \
    }                                                                                          \
                                                                                                \
    template<typename T, typename U>                                                          \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType> && !std::is_same_v<U COMMA InterType>, InterType>::type update(T old, U opOutput, Z* extraParams) { \
      return update(static_cast<InterType>(old), static_cast<InterType>(opOutput), extraParams); \
    }                                                                                          \
                                                                                                \
    template<typename U = X, typename V = Z>                                                  \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, InterType>::type update(InterType old, InterType opOutput, X* extraParams) { \
      return update(old, opOutput, reinterpret_cast<Z*>(extraParams));                       \
    }                                                                                          \
                                                                                                \
    static SD_HOST_DEVICE InterType op(X d1, Z* extraParams) {                                               \
      if constexpr (simdOps::is_simd_unsupported_return_type<InterType>::value ||             \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return op_logic(d1, extraParams);                                                     \
      else                                                                                     \
        return op_simd(d1, extraParams);                                                      \
    }                                                                                          \
                                                                                                \
    template<typename U = X, typename V = Z>                                                  \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, InterType>::type op(X d1, X* extraParams) { \
      return op(d1, reinterpret_cast<Z*>(extraParams));                                       \
    }                                                                                          \
                                                                                                \
    static SD_HOST_DEVICE Z postProcess(InterType reduction, sd::LongType n, Z* extraParams) {               \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return postProcess_logic(reduction, n, extraParams);                                  \
      else                                                                                     \
        return postProcess_simd(reduction, n, extraParams);                                   \
    }                                                                                          \
                                                                                                \
    template<typename T>                                                                       \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType>, Z>::type postProcess(T reduction, sd::LongType n, Z* extraParams) { \
      return postProcess(static_cast<InterType>(reduction), n, extraParams);                 \
    }                                                                                          \
                                                                                                \
    template<typename U = X, typename V = Z>                                                  \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type postProcess(InterType reduction, sd::LongType n, X* extraParams) { \
      return postProcess(reduction, n, reinterpret_cast<Z*>(extraParams));                   \
    }                                                                                          \
                                                                                                \
    template<typename T, typename U = X, typename V = Z>                                      \
    static SD_HOST_DEVICE typename std::enable_if<!std::is_same_v<T COMMA InterType> && !std::is_same_v<U COMMA V>, Z>::type postProcess(T reduction, sd::LongType n, X* extraParams) { \
      return postProcess(static_cast<InterType>(reduction), n, reinterpret_cast<Z*>(extraParams)); \
    }                                                                                          \
  };

} // namespace simdOps

#endif // OP_MACROS_REDUCE_H_