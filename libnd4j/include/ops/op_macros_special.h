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
#ifndef OP_MACROS_SPECIAL_H_
#define OP_MACROS_SPECIAL_H_

#include "op_types.h"
#include <math/templatemath.h>

namespace simdOps {

// =============================================================================
// SAFE DIVISION OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a safe division operation with proper SIMD handling
 */
#define DECLARE_SAFE_DIVISION_OP(OP_NAME, CONDITION)                                           \
  template <typename X, typename Y, typename Z>                                                \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    static SD_INLINE Z op_logic(X d1, Y d2) {                                                 \
      return CONDITION ? static_cast<Z>(0) : sd::math::sd_divide<X COMMA Y COMMA Z>(d1, d2);            \
    }                                                                                          \
    static SD_INLINE Z op_logic(X d1, Y d2, Z* params) {                                      \
      return CONDITION ? static_cast<Z>(0) : sd::math::sd_divide<X COMMA Y COMMA Z>(d1, d2);            \
    }                                                                                          \
    static SD_INLINE Z op_logic(X d1) {                                                       \
      return static_cast<Z>(d1);                                                              \
    }                                                                                          \
    static SD_INLINE Z op_logic(X d1, Y* params) {                                            \
      return params[0] == static_cast<Y>(0) ? static_cast<Z>(0) : sd::math::sd_divide<X COMMA Y COMMA Z>(d1, params[0]); \
    }                                                                                          \
    static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }                                 \
    static Z op_simd(X d1, Y d2, Z* params) { return op_logic(d1, d2, params); }              \
    static Z op_simd(X d1) { return op_logic(d1); }                                           \
    static Z op_simd(X d1, Y* params) { return op_logic(d1, params); }                        \
                                                                                               \
   public:                                                                                     \
    static Z op(X d1, Y d2) {                                                                  \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1, d2);                                                              \
      else return op_simd(d1, d2);                                                            \
    }                                                                                          \
    static Z op(X d1, Y d2, Z* params) {                                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1, d2, params);                                                      \
      else return op_simd(d1, d2, params);                                                    \
    }                                                                                          \
    static Z op(X d1) {                                                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1);                                                                  \
      else return op_simd(d1);                                                                \
    }                                                                                          \
    static Z op(X d1, Y* params) {                                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1, params);                                                          \
      else return op_simd(d1, params);                                                        \
    }                                                                                          \
    SD_OP_DEF static X startingValue() { return static_cast<X>(1); }                          \
  };

// =============================================================================
// FLOOR/TRUNCATE DIVISION OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a floor division operation with proper SIMD handling
 */
#define DECLARE_FLOOR_DIVISION_OP(OP_NAME, FLOOR_FUNC)                                        \
  template <typename X, typename Y, typename Z>                                               \
  class OP_NAME {                                                                             \
   private:                                                                                   \
    static SD_INLINE Z op_logic(X d1, Y d2) {                                                \
      auto divResult = sd::math::sd_divide<X COMMA Y COMMA double>(d1, d2);                             \
      return static_cast<Z>(sd::math::FLOOR_FUNC<double COMMA Z>(divResult));                      \
    }                                                                                         \
    static SD_INLINE Z op_logic(X d1, Y d2, Z* params) {                                     \
      auto divResult = sd::math::sd_divide<X COMMA Y COMMA double>(d1, d2);                             \
      return static_cast<Z>(sd::math::FLOOR_FUNC<double COMMA Z>(divResult));                      \
    }                                                                                         \
    static SD_INLINE Z op_logic(X d1) {                                                      \
      return sd::math::FLOOR_FUNC<X COMMA Z>(d1);                                                  \
    }                                                                                         \
    static SD_INLINE Z op_logic(X d1, Y* params) {                                           \
      auto divResult = sd::math::sd_divide<X COMMA Y COMMA double>(d1, params[0]);                      \
      return static_cast<Z>(sd::math::FLOOR_FUNC<double COMMA Z>(divResult));                      \
    }                                                                                         \
    static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }                                \
    static Z op_simd(X d1, Y d2, Z* params) { return op_logic(d1, d2, params); }             \
    static Z op_simd(X d1) { return op_logic(d1); }                                          \
    static Z op_simd(X d1, Y* params) { return op_logic(d1, params); }                       \
                                                                                              \
   public:                                                                                    \
    static Z op(X d1, Y d2) {                                                                 \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1, d2);                                                             \
      else return op_simd(d1, d2);                                                           \
    }                                                                                         \
    static Z op(X d1, Y d2, Z* params) {                                                      \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1, d2, params);                                                     \
      else return op_simd(d1, d2, params);                                                   \
    }                                                                                         \
    static Z op(X d1) {                                                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1);                                                                 \
      else return op_simd(d1);                                                               \
    }                                                                                         \
    static Z op(X d1, Y* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return op_logic(d1, params);                                                         \
      else return op_simd(d1, params);                                                       \
    }                                                                                         \
    SD_OP_DEF static X startingValue() { return static_cast<X>(1); }                          \
  };

// =============================================================================
// MODULO OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a modulo operation with proper SIMD handling
 */
#define DECLARE_MODULO_OP(OP_NAME, MOD_OPERATION) \
template <typename X, typename Y, typename Z> \
class OP_NAME { \
 public: \
  SD_OP_DEF static Z op(X d1, Y d2) { return MOD_OPERATION; } \
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return MOD_OPERATION; } \
  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); } \
  SD_OP_DEF static Z op(X d1, Y *params) { return MOD_OPERATION##_PARAMS; } \
  SD_OP_DEF static X startingValue() { return static_cast<X>(0); } \
};

/**
 * @brief Declares an operation that supports both binary and unary forms with proper SIMD handling
 */
#define DECLARE_MULTI_OP_SIMD_SAFE(OP_NAME, BINARY_OP, BINARY_PARAMS_OP, UNARY_OP, UNARY_PARAMS_OP) \
  template <typename X, typename Z>                                                            \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    static SD_INLINE Z binary_op_logic(X d1, X d2) {                                          \
      BINARY_OP                                                                                \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Z binary_params_op_logic(X d1, X d2, X* params) {                        \
      BINARY_PARAMS_OP                                                                         \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Z unary_op_logic(X d1) {                                                 \
      UNARY_OP                                                                                 \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Z unary_params_op_logic(X d1, X* params) {                               \
      UNARY_PARAMS_OP                                                                          \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TX> binary_op_simd(TX d1, TX d2) {          \
      return binary_op_logic(d1, d2);                                                         \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> binary_op_simd(TX d1, TX d2) {        \
      return binary_op_logic(d1, d2);                                                         \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TX> binary_params_op_simd(TX d1, TX d2, TX* params) { \
      return binary_params_op_logic(d1, d2, params);                                          \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> binary_params_op_simd(TX d1, TX d2, TX* params) { \
      return binary_params_op_logic(d1, d2, params);                                          \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TX> unary_op_simd(TX d1) {                  \
      return unary_op_logic(d1);                                                              \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> unary_op_simd(TX d1) {                \
      return unary_op_logic(d1);                                                              \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TX> unary_params_op_simd(TX d1, TX* params) { \
      return unary_params_op_logic(d1, params);                                               \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> unary_params_op_simd(TX d1, TX* params) { \
      return unary_params_op_logic(d1, params);                                               \
    }                                                                                          \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_bool no_op_exec_special_bool_cuda;                                     \
                                                                                               \
    static Z op(X d1, X d2) {                                                                 \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return binary_op_logic(d1, d2);                                                       \
      else                                                                                     \
        return binary_op_simd(d1, d2);                                                        \
    }                                                                                          \
                                                                                               \
    static Z op(X d1, X d2, X* params) {                                                      \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return binary_params_op_logic(d1, d2, params);                                        \
      else                                                                                     \
        return binary_params_op_simd(d1, d2, params);                                         \
    }                                                                                          \
                                                                                               \
    static Z op(X d1) {                                                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return unary_op_logic(d1);                                                            \
      else                                                                                     \
        return unary_op_simd(d1);                                                             \
    }                                                                                          \
                                                                                               \
    static Z op(X d1, X* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return unary_params_op_logic(d1, params);                                             \
      else                                                                                     \
        return unary_params_op_simd(d1, params);                                              \
    }                                                                                          \
  };

#define DECLARE_BINARY_SIMD_SAFE_OP(OP_NAME, OPERATION)                                             \
  template <typename X, typename Y, typename Z>                                                 \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    static SD_INLINE Z op_logic(X d1, Y d2, Z* params) {                                       \
      OPERATION                                                                                 \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                                \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TX COMMA TY> op_simd(TX d1, TY d2, TZ* params) { \
      return op_logic(d1, d2, params);                                                         \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                                \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX COMMA TY> op_simd(TX d1, TY d2, TZ* params) { \
      return op_logic(d1, d2, params);                                                         \
    }                                                                                           \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special no_op_exec_special_cuda;                                                 \
                                                                                                \
    static Z op(X d1, Y d2, Z* params) {                                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                    \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                      \
        return op_logic(d1, d2, params);                                                       \
      else                                                                                      \
        return op_simd(d1, d2, params);                                                        \
    }                                                                                           \
  };

/**
 * @brief DECLARE_MIXED_ACCUMULATION_SIMD_SAFE_OP macro
 * Add this to op_macros_reduce.h
 *
 * This macro creates mixed accumulation operations with proper SIMD handling
 * Used for reduce operations that need custom merge/update/postProcess logic
 */
#define DECLARE_MIXED_ACCUMULATION_SIMD_SAFE_OP(OP_NAME, OP_LOGIC, REDUCE_TYPE_VAL, STARTING_VAL, MERGE_OP, UPDATE_OP, POST_PROCESS) \
  template <typename X, typename Z>                                                             \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    static SD_INLINE Z op_logic(X d1, Z* params) {                                             \
      OP_LOGIC                                                                                  \
    }                                                                                           \
                                                                                                \
    static SD_INLINE Z merge_logic(Z old, Z opOutput, Z* extraParams) {                        \
      return MERGE_OP;                                                                          \
    }                                                                                           \
                                                                                                \
    static SD_INLINE Z update_logic(Z old, Z opOutput, Z* extraParams) {                       \
      return UPDATE_OP;                                                                         \
    }                                                                                           \
                                                                                                \
    static SD_INLINE Z postProcess_logic(Z reduction, sd::LongType n, Z* extraParams) {        \
      return POST_PROCESS;                                                                      \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TZ = Z>                                                 \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TX> op_simd(TX d1, TZ* params) {             \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TX = X, typename TZ = Z>                                                 \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> op_simd(TX d1, TZ* params) {           \
      return op_logic(d1, params);                                                             \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TZ> merge_simd(TZ old, TZ opOutput, TZ* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                          \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TZ> merge_simd(TZ old, TZ opOutput, TZ* extraParams) { \
      return merge_logic(old, opOutput, extraParams);                                          \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TZ> update_simd(TZ old, TZ opOutput, TZ* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TZ> update_simd(TZ old, TZ opOutput, TZ* extraParams) { \
      return update_logic(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    static SD_INLINE enable_if_simd_safe<TZ COMMA TZ> postProcess_simd(TZ reduction, sd::LongType n, TZ* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                     \
    }                                                                                           \
                                                                                                \
    template<typename TZ = Z>                                                                  \
    static SD_INLINE enable_if_simd_unsafe<TZ COMMA TZ> postProcess_simd(TZ reduction, sd::LongType n, TZ* extraParams) { \
      return postProcess_logic(reduction, n, extraParams);                                     \
    }                                                                                           \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda                       \
    const static functions::ReduceType reduceType = functions::ReduceType::REDUCE_TYPE_VAL;    \
    using InterType = Z;                                                                        \
                                                                                                \
    static X startingValue(const X* input) { return STARTING_VAL; }                            \
                                                                                                \
    static Z op(X d1, Z* extraParams) {                                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                      \
        return op_logic(d1, extraParams);                                                      \
      else                                                                                      \
        return op_simd(d1, extraParams);                                                       \
    }                                                                                           \
                                                                                                \
    template<typename U = X, typename V = Z>                                                   \
    static typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type op(X d1, X* extraParams) { \
      return op(d1, reinterpret_cast<Z*>(extraParams));                                        \
    }                                                                                           \
                                                                                                \
    static Z merge(Z old, Z opOutput, Z* extraParams) {                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                        \
        return merge_logic(old, opOutput, extraParams);                                        \
      else                                                                                      \
        return merge_simd(old, opOutput, extraParams);                                         \
    }                                                                                           \
                                                                                                \
    static Z update(Z old, Z opOutput, Z* extraParams) {                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                        \
        return update_logic(old, opOutput, extraParams);                                       \
      else                                                                                      \
        return update_simd(old, opOutput, extraParams);                                        \
    }                                                                                           \
                                                                                                \
    template<typename T>                                                                       \
    static typename std::enable_if<!std::is_same_v<T COMMA Z>, Z>::type update(T old, Z opOutput, Z* extraParams) { \
      return update(static_cast<Z>(old), opOutput, extraParams);                              \
    }                                                                                          \
                                                                                                \
    template<typename T>                                                                       \
    static typename std::enable_if<!std::is_same_v<T COMMA Z>, Z>::type update(Z old, T opOutput, Z* extraParams) { \
      return update(old, static_cast<Z>(opOutput), extraParams);                              \
    }                                                                                          \
                                                                                                \
    template<typename T, typename U>                                                          \
    static typename std::enable_if<!std::is_same_v<T COMMA Z> && !std::is_same_v<U COMMA Z>, Z>::type update(T old, U opOutput, Z* extraParams) { \
      return update(static_cast<Z>(old), static_cast<Z>(opOutput), extraParams);              \
    }                                                                                          \
                                                                                                \
    template<typename U = X, typename V = Z>                                                  \
    static typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type update(Z old, Z opOutput, X* extraParams) { \
      return update(old, opOutput, reinterpret_cast<Z*>(extraParams));                       \
    }                                                                                          \
                                                                                                \
    static Z postProcess(Z reduction, sd::LongType n, Z* extraParams) {                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                       \
        return postProcess_logic(reduction, n, extraParams);                                  \
      else                                                                                     \
        return postProcess_simd(reduction, n, extraParams);                                   \
    }                                                                                          \
                                                                                                \
    template<typename T>                                                                       \
    static typename std::enable_if<!std::is_same_v<T COMMA Z>, Z>::type postProcess(T reduction, sd::LongType n, Z* extraParams) { \
      return postProcess(static_cast<Z>(reduction), n, extraParams);                          \
    }                                                                                          \
                                                                                                \
    template<typename U = X, typename V = Z>                                                  \
    static typename std::enable_if<!std::is_same_v<U COMMA V>, Z>::type postProcess(Z reduction, sd::LongType n, X* extraParams) { \
      return postProcess(reduction, n, reinterpret_cast<Z*>(extraParams));                    \
    }                                                                                          \
                                                                                                \
    template<typename T, typename U = X, typename V = Z>                                      \
    static typename std::enable_if<!std::is_same_v<T COMMA Z> && !std::is_same_v<U COMMA V>, Z>::type postProcess(T reduction, sd::LongType n, X* extraParams) { \
      return postProcess(static_cast<Z>(reduction), n, reinterpret_cast<Z*>(extraParams));    \
    }                                                                                          \
  };


// =============================================================================
// BOOLEAN/LOGICAL OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a boolean operation with proper SIMD handling
 */
/**
 * @brief Fixed version of DECLARE_BOOLEAN_OP_WITH_TYPE_SAFETY macro
 * Add this to op_macros_special.h or op_macros_reduce.h
 */
#define DECLARE_BOOLEAN_OP_WITH_TYPE_SAFETY(OP_NAME, OPERATION, STARTING_VAL) \
template <typename X, typename Z> \
class OP_NAME { \
 private: \
  static SD_INLINE Z op_logic(X d1, X d2, Z *extraParamsRef) { \
    X eps = static_cast<X>(extraParamsRef[2]); \
    auto result = OPERATION; \
    return static_cast<Z>(result); \
  } \
  \
  static SD_INLINE Z startingValue_logic(const X *input) { \
    return static_cast<Z>(STARTING_VAL); \
  } \
  \
  static SD_INLINE Z postProcess_logic(Z reduction, sd::LongType n, Z *extraParamsRef) { \
    return reduction; \
  } \
  \
  static SD_INLINE Z update_logic(Z old, Z opOutput, Z *extraParamsRef) { \
    return static_cast<Z>(static_cast<bool>(opOutput) && static_cast<bool>(old) ? 1 : 0); \
  } \
  \
  static SD_INLINE Z merge_logic(Z old, Z opOutput, Z *extraParamsRef) { \
    return update_logic(old, opOutput, extraParamsRef); \
  } \
  \
  template<typename TX = X, typename TZ = Z> \
  static SD_INLINE enable_if_simd_safe<TZ COMMA TX> op_simd(TX d1, TX d2, TZ *extraParamsRef) { \
    return op_logic(d1, d2, extraParamsRef); \
  } \
  \
  template<typename TX = X, typename TZ = Z> \
  static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> op_simd(TX d1, TX d2, TZ *extraParamsRef) { \
    return op_logic(d1, d2, extraParamsRef); \
  } \
  \
  template<typename TX = X, typename TZ = Z> \
  static SD_INLINE enable_if_simd_safe<TZ COMMA TX> startingValue_simd(const TX *input) { \
    return startingValue_logic(input); \
  } \
  \
  template<typename TX = X, typename TZ = Z> \
  static SD_INLINE enable_if_simd_unsafe<TZ COMMA TX> startingValue_simd(const TX *input) { \
    return startingValue_logic(input); \
  } \
  \
  template<typename TZ = Z> \
  static SD_INLINE enable_if_simd_safe<TZ COMMA TZ> postProcess_simd(TZ reduction, sd::LongType n, TZ *extraParamsRef) { \
    return postProcess_logic(reduction, n, extraParamsRef); \
  } \
  \
  template<typename TZ = Z> \
  static SD_INLINE enable_if_simd_unsafe<TZ COMMA TZ> postProcess_simd(TZ reduction, sd::LongType n, TZ *extraParamsRef) { \
    return postProcess_logic(reduction, n, extraParamsRef); \
  } \
  \
  template<typename TZ = Z> \
  static SD_INLINE enable_if_simd_safe<TZ COMMA TZ> update_simd(TZ old, TZ opOutput, TZ *extraParamsRef) { \
    return update_logic(old, opOutput, extraParamsRef); \
  } \
  \
  template<typename TZ = Z> \
  static SD_INLINE enable_if_simd_unsafe<TZ COMMA TZ> update_simd(TZ old, TZ opOutput, TZ *extraParamsRef) { \
    return update_logic(old, opOutput, extraParamsRef); \
  } \
  \
  template<typename TZ = Z> \
  static SD_INLINE enable_if_simd_safe<TZ COMMA TZ> merge_simd(TZ old, TZ opOutput, TZ *extraParamsRef) { \
    return merge_logic(old, opOutput, extraParamsRef); \
  } \
  \
  template<typename TZ = Z> \
  static SD_INLINE enable_if_simd_unsafe<TZ COMMA TZ> merge_simd(TZ old, TZ opOutput, TZ *extraParamsRef) { \
    return merge_logic(old, opOutput, extraParamsRef); \
  } \
  \
 public: \
  static const int extraParamsLen = 0; \
  static X *generateExtraParams() { return nullptr; } \
  static void finalizeExtraParams(X *extraParamsRef) {} \
  static void aggregateExtraParams(Z *extraParamsTotal, Z *extraParamsLocal) {} \
  \
  static Z startingValue(const X *input) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value || \
                  simdOps::is_simd_unsupported_argument_type<X>::value) \
      return startingValue_logic(input); \
    else \
      return startingValue_simd(input); \
  } \
  \
  static Z postProcess(Z reduction, sd::LongType n, Z *extraParamsRef) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value) \
      return postProcess_logic(reduction, n, extraParamsRef); \
    else \
      return postProcess_simd(reduction, n, extraParamsRef); \
  } \
  \
  static Z op(X d1, X d2, Z *extraParamsRef) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value || \
                  simdOps::is_simd_unsupported_argument_type<X>::value) \
      return op_logic(d1, d2, extraParamsRef); \
    else \
      return op_simd(d1, d2, extraParamsRef); \
  } \
  \
  static Z update(Z old, Z opOutput, Z *extraParamsRef) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value) \
      return update_logic(old, opOutput, extraParamsRef); \
    else \
      return update_simd(old, opOutput, extraParamsRef); \
  } \
  \
  static Z merge(Z old, Z opOutput, Z *extraParamsRef) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value) \
      return merge_logic(old, opOutput, extraParamsRef); \
    else \
      return merge_simd(old, opOutput, extraParamsRef); \
  } \
};


/**
 * @brief Declares a distance operation with bool support and proper SIMD handling
 */
/**
 * @brief Fixed version of DECLARE_DISTANCE_OP_WITH_BOOL_SUPPORT macro
 * Add this to op_macros_special.h
 */
#define DECLARE_DISTANCE_OP_WITH_BOOL_SUPPORT(OP_NAME, BOOL_LOGIC, NORMAL_LOGIC, STARTING_VAL) \
  template <typename X, typename Y>                                                            \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    static SD_INLINE Y op_logic(X d1, X d2, Y* extraParamsRef) {                              \
      if constexpr (std::is_same_v<X COMMA bool>) {                                           \
        return static_cast<Y>(BOOL_LOGIC);                                                    \
      } else {                                                                                 \
        return static_cast<Y>(NORMAL_LOGIC);                                                  \
      }                                                                                        \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Y startingValue_logic(const X* input) {                                  \
      return static_cast<Y>(STARTING_VAL);                                                    \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Y postProcess_logic(Y reduction, sd::LongType n, Y* extraParamsRef) {    \
      return reduction;                                                                        \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Y update_logic(Y old, Y opOutput, Y* extraParamsRef) {                   \
      return old + opOutput;                                                                   \
    }                                                                                          \
                                                                                               \
    static SD_INLINE Y merge_logic(Y old, Y opOutput, Y* extraParamsRef) {                    \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    static SD_INLINE enable_if_simd_safe<TY COMMA TX> op_simd(TX d1, TX d2, TY* extraParamsRef) { \
      return op_logic(d1, d2, extraParamsRef);                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    static SD_INLINE enable_if_simd_unsafe<TY COMMA TX> op_simd(TX d1, TX d2, TY* extraParamsRef) { \
      return op_logic(d1, d2, extraParamsRef);                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    static SD_INLINE enable_if_simd_safe<TY COMMA TX> startingValue_simd(const TX* input) {   \
      return startingValue_logic(input);                                                      \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y>                                                \
    static SD_INLINE enable_if_simd_unsafe<TY COMMA TX> startingValue_simd(const TX* input) { \
      return startingValue_logic(input);                                                      \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    static SD_INLINE enable_if_simd_safe<TY COMMA TY> postProcess_simd(TY reduction, sd::LongType n, TY* extraParamsRef) { \
      return postProcess_logic(reduction, n, extraParamsRef);                                 \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    static SD_INLINE enable_if_simd_unsafe<TY COMMA TY> postProcess_simd(TY reduction, sd::LongType n, TY* extraParamsRef) { \
      return postProcess_logic(reduction, n, extraParamsRef);                                 \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    static SD_INLINE enable_if_simd_safe<TY COMMA TY> update_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    static SD_INLINE enable_if_simd_unsafe<TY COMMA TY> update_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return update_logic(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    static SD_INLINE enable_if_simd_safe<TY COMMA TY> merge_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return merge_logic(old, opOutput, extraParamsRef);                                      \
    }                                                                                          \
                                                                                               \
    template<typename TY = Y>                                                                 \
    static SD_INLINE enable_if_simd_unsafe<TY COMMA TY> merge_simd(TY old, TY opOutput, TY* extraParamsRef) { \
      return merge_logic(old, opOutput, extraParamsRef);                                      \
    }                                                                                          \
                                                                                               \
   public:                                                                                     \
    static const int extraParamsLen = 0;                                                      \
    static X *generateExtraParams() { return nullptr; }                                       \
    static void finalizeExtraParams(X *extraParamsRef) {}                                     \
    static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}             \
                                                                                               \
    static Y startingValue(const X* input) {                                                  \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return startingValue_logic(input);                                                    \
      else                                                                                     \
        return startingValue_simd(input);                                                     \
    }                                                                                          \
                                                                                               \
    static Y postProcess(Y reduction, sd::LongType n, Y* extraParamsRef) {                    \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                     \
        return postProcess_logic(reduction, n, extraParamsRef);                              \
      else                                                                                     \
        return postProcess_simd(reduction, n, extraParamsRef);                               \
    }                                                                                          \
                                                                                               \
    static Y op(X d1, X d2, Y* extraParamsRef) {                                              \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value ||                     \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                     \
        return op_logic(d1, d2, extraParamsRef);                                              \
      else                                                                                     \
        return op_simd(d1, d2, extraParamsRef);                                               \
    }                                                                                          \
                                                                                               \
    static Y update(Y old, Y opOutput, Y* extraParamsRef) {                                   \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value)                       \
        return update_logic(old, opOutput, extraParamsRef);                                   \
      else                                                                                     \
        return update_simd(old, opOutput, extraParamsRef);                                    \
    }                                                                                          \
                                                                                               \
    static Y merge(Y old, Y opOutput, Y* extraParamsRef) {                                    \
      if constexpr (simdOps::is_simd_unsupported_return_type<Y>::value)                       \
        return merge_logic(old, opOutput, extraParamsRef);                                    \
      else                                                                                     \
        return merge_simd(old, opOutput, extraParamsRef);                                     \
    }                                                                                          \
  };


/**
 * @brief Declares a squared subtract operation with proper SIMD handling
 */
/**
 * @brief Declares a squared subtract operation with proper SIMD handling
 */
#define DECLARE_SQUARED_SUBTRACT_OP(OP_NAME, SUBTRACT_FUNC)                                     \
  template <typename X, typename Y, typename Z>                                                 \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    static SD_INLINE Z op_logic(X d1, Y d2) {                                                  \
      Z diff = sd::math::SUBTRACT_FUNC<X COMMA Y COMMA Z>(d1, d2);                            \
      return sd::math::sd_multiply<Z COMMA Z COMMA Z>(diff, diff);                             \
    }                                                                                          \
    static SD_INLINE Z op_logic(X d1, Y d2, Z *params) {                                       \
      Z diff = sd::math::SUBTRACT_FUNC<X COMMA Y COMMA Z>(d1, d2);                            \
      return sd::math::sd_multiply<Z COMMA Z COMMA Z>(diff, diff);                             \
    }                                                                                          \
    static SD_INLINE Z op_logic(X d1) { return static_cast<Z>(d1); }                           \
    static SD_INLINE Z op_logic(X d1, Y *params) {                                             \
      Z diff = sd::math::SUBTRACT_FUNC<X COMMA Y COMMA Z>(d1, params[0]);                     \
      return sd::math::sd_multiply<Z COMMA Z COMMA Z>(diff, diff);                             \
    }                                                                                          \
    SD_OP_DEF static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }                        \
    SD_OP_DEF static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); }     \
    SD_OP_DEF static Z op_simd(X d1) { return op_logic(d1); }                                  \
    SD_OP_DEF static Z op_simd(X d1, Y *params) { return op_logic(d1, params); }               \
                                                                                               \
   public:                                                                                     \
    static Z op(X d1, Y d2) {                                                                  \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                    \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                      \
        return op_logic(d1, d2);                                                              \
      else return op_simd(d1, d2);                                                            \
    }                                                                                          \
    static Z op(X d1, Y d2, Z *params) {                                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                    \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                      \
        return op_logic(d1, d2, params);                                                      \
      else return op_simd(d1, d2, params);                                                    \
    }                                                                                          \
    static Z op(X d1) {                                                                        \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                      \
        return op_logic(d1);                                                                  \
      else return op_simd(d1);                                                                \
    }                                                                                          \
    static Z op(X d1, Y *params) {                                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                    \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                      \
        return op_logic(d1, params);                                                          \
      else return op_simd(d1, params);                                                        \
    }                                                                                          \
    SD_OP_DEF static X startingValue() { return static_cast<X>(0.f); }                         \
  };

/**
 * @brief Declares a squared reverse subtract operation with proper SIMD handling
 */
#define DECLARE_SQUARED_REVERSE_SUBTRACT_OP(OP_NAME, SUBTRACT_FUNC) \
template <typename X, typename Y, typename Z> \
class OP_NAME { \
 private: \
  static SD_INLINE Z op_logic(X d1, Y d2) { \
    Z diff = sd::math::SUBTRACT_FUNC<Y COMMA X COMMA Z>(d2, d1); \
    return sd::math::sd_multiply<Z COMMA Z COMMA Z>(diff, diff); \
  } \
  static SD_INLINE Z op_logic(X d1, Y d2, Z *params) { \
    Z diff = sd::math::SUBTRACT_FUNC<Y COMMA X COMMA Z>(d2, d1); \
    return sd::math::sd_multiply<Z COMMA Z COMMA Z>(diff, diff); \
  } \
  static SD_INLINE Z op_logic(X d1) { return static_cast<Z>(d1); } \
  static SD_INLINE Z op_logic(X d1, Y *params) { \
    Z diff = sd::math::SUBTRACT_FUNC<Y COMMA X COMMA Z>(params[0], d1); \
    return sd::math::sd_multiply<Z COMMA Z COMMA Z>(diff, diff); \
  } \
  SD_OP_DEF static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); } \
  SD_OP_DEF static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); } \
  SD_OP_DEF static Z op_simd(X d1) { return op_logic(d1); } \
  SD_OP_DEF static Z op_simd(X d1, Y *params) { return op_logic(d1, params); } \
 \
 public: \
  static Z op(X d1, Y d2) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value || \
                  simdOps::is_simd_unsupported_argument_type<X>::value || \
                  simdOps::is_simd_unsupported_argument_type<Y>::value) \
      return op_logic(d1, d2); \
    else \
      return op_simd(d1, d2); \
  } \
  static Z op(X d1, Y d2, Z *params) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value || \
                  simdOps::is_simd_unsupported_argument_type<X>::value || \
                  simdOps::is_simd_unsupported_argument_type<Y>::value) \
      return op_logic(d1, d2, params); \
    else \
      return op_simd(d1, d2, params); \
  } \
  static Z op(X d1) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value || \
                  simdOps::is_simd_unsupported_argument_type<X>::value) \
      return op_logic(d1); \
    else \
      return op_simd(d1); \
  } \
  static Z op(X d1, Y *params) { \
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value || \
                  simdOps::is_simd_unsupported_argument_type<X>::value || \
                  simdOps::is_simd_unsupported_argument_type<Y>::value) \
      return op_logic(d1, params); \
    else \
      return op_simd(d1, params); \
  } \
  static X startingValue() { return static_cast<X>(0.f); } \
};


} // namespace simdOps

#endif // OP_MACROS_SPECIAL_H_