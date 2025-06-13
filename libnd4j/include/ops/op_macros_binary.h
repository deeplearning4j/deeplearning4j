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
#ifndef BINARY_OP_MACROS_H_
#define BINARY_OP_MACROS_H_

#include "op_types.h"
#include <math/templatemath.h>

namespace simdOps {

// =============================================================================
// STANDARD BINARY OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a standard binary operation with proper SIMD handling
 */
#define DECLARE_STANDARD_BINARY_OP(OP_NAME, OPERATION)                                         \
  template <typename X, typename Y, typename Z>                                                \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2) {                                         \
      return OPERATION;                                                                        \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2, Z* params) {                              \
      return OPERATION;                                                                        \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1) {                                               \
      return static_cast<Z>(d1);                                                              \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y* params) {                                    \
      return OPERATION##_WITH_PARAMS;                                                         \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX, TY> op_simd(TX d1, TY d2) {          \
      return op_logic(d1, d2);                                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX, TY> op_simd(TX d1, TY d2) {        \
      return op_logic(d1, d2);                                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX, TY> op_simd(TX d1, TY d2, TZ* params) { \
      return op_logic(d1, d2, params);                                                        \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX, TY> op_simd(TX d1, TY d2, TZ* params) { \
      return op_logic(d1, d2, params);                                                        \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX> op_simd(TX d1) {                     \
      return op_logic(d1);                                                                    \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX> op_simd(TX d1) {                   \
      return op_logic(d1);                                                                    \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX, TY> op_simd(TX d1, TY* params) {     \
      return op_logic(d1, params);                                                            \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX, TY> op_simd(TX d1, TY* params) {   \
      return op_logic(d1, params);                                                            \
    }                                                                                          \
                                                                                               \
   public:                                                                                     \
    SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2) {                                               \
      DISPATCH_SIMD_BINARY(Z, X, Y, op_simd(d1, d2), op_logic(d1, d2))                       \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2, Z* params) {                                    \
      DISPATCH_SIMD_BINARY(Z, X, Y, op_simd(d1, d2, params), op_logic(d1, d2, params))       \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op(X d1) {                                                     \
      DISPATCH_SIMD_UNARY(Z, X, op_simd(d1), op_logic(d1))                                   \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op(X d1, Y* params) {                                          \
      DISPATCH_SIMD_BINARY(Z, X, Y, op_simd(d1, params), op_logic(d1, params))               \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static X startingValue() { return static_cast<X>(0.f); }         \
  };

/**
 * @brief Declares a binary math operation with proper SIMD handling
 */
#define DECLARE_BINARY_MATH_OP(OP_NAME, MATH_FUNC)                                                                      \
  template <typename X, typename Y, typename Z>                                                                       \
  class OP_NAME {                                                                                                     \
   private:                                                                                                           \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2) { return sd::math::MATH_FUNC<X COMMA Y COMMA Z>(d1, d2); }           \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2, Z* params) { return sd::math::MATH_FUNC<X COMMA Y COMMA Z>(d1, d2); } \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1) { return static_cast<Z>(d1); }                                    \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y* params) { return sd::math::MATH_FUNC<X COMMA Y COMMA Z>(d1, params[0]); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }                                           \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2, Z* params) { return op_logic(d1, d2, params); }                        \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1) { return op_logic(d1); }                                                     \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y* params) { return op_logic(d1, params); }                                  \
                                                                                                                      \
   public:                                                                                                            \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2) {                                                                \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                                            \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, d2);                     \
      else return op_simd(d1, d2);                                                                                    \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2, Z* params) {                                                     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                                            \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, d2, params);             \
      else return op_simd(d1, d2, params);                                                                            \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1) {                                                                      \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value) return op_logic(d1);                         \
      else return op_simd(d1);                                                                                        \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y* params) {                                                           \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                                            \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, params);                 \
      else return op_simd(d1, params);                                                                                \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static X startingValue() { return static_cast<X>(1.f); }                                 \
  };

// =============================================================================
// COMPARISON OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a comparison operation with proper SIMD handling
 */
#define DECLARE_COMPARISON_OP(OP_NAME, COMPARISON)                                                   \
  template <typename X, typename Z>                                                                  \
  class OP_NAME {                                                                                    \
   private:                                                                                          \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, X d2) { return d1 COMPARISON d2; }             \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, X d2, X* params) { return op_logic(d1, d2); }  \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, X* params) { return d1; }                      \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, X d2) { return op_logic(d1, d2); }              \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, X d2, X* params) { return op_logic(d1, d2, params); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, X* params) { return op_logic(d1, params); }     \
                                                                                                     \
   public:                                                                                           \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, X d2) {                                              \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                             \
                    simdOps::is_simd_unsupported_argument_type<X>::value) return op_logic(d1, d2);    \
      else return op_simd(d1, d2);                                                                   \
    }                                                                                                \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, X d2, X* params) {                                   \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                             \
                    simdOps::is_simd_unsupported_argument_type<X>::value) return op_logic(d1, d2, params); \
      else return op_simd(d1, d2, params);                                                           \
    }                                                                                                \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, X* params) {                                         \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                             \
                    simdOps::is_simd_unsupported_argument_type<X>::value) return op_logic(d1, params); \
      else return op_simd(d1, params);                                                               \
    }                                                                                                \
  };

// =============================================================================
// REVERSE BINARY MATH OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a reverse binary math operation with proper SIMD handling
 */
#define DECLARE_REVERSE_BINARY_MATH_OP(OP_NAME, MATH_FUNC, START_VAL)                                                   \
  template <typename X, typename Y, typename Z>                                                                       \
  class OP_NAME {                                                                                                     \
   private:                                                                                                           \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2) { return sd::math::MATH_FUNC<Y COMMA X COMMA Z>(d2, d1); }           \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2, Z* params) { return sd::math::MATH_FUNC<Y COMMA X COMMA Z>(d2, d1); } \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1) { return static_cast<Z>(d1); }                                    \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y* params) { return sd::math::MATH_FUNC<Y COMMA X COMMA Z>(params[0], d1); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }                                           \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2, Z* params) { return op_logic(d1, d2, params); }                        \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1) { return op_logic(d1); }                                                     \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y* params) { return op_logic(d1, params); }                                  \
                                                                                                                      \
   public:                                                                                                            \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2) {                                                                \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                                            \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, d2);                     \
      else return op_simd(d1, d2);                                                                                    \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2, Z* params) {                                                     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                                            \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, d2, params);             \
      else return op_simd(d1, d2, params);                                                                            \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1) {                                                                      \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value) return op_logic(d1);                         \
      else return op_simd(d1);                                                                                        \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y* params) {                                                           \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                                              \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                                            \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, params);                 \
      else return op_simd(d1, params);                                                                                \
    }                                                                                                                 \
    SD_HOST_DEVICE SD_INLINE static X startingValue() { return static_cast<X>(START_VAL); }                           \
  };

// =============================================================================
// SQUARED OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a squared binary operation with proper SIMD handling
 */
#define DECLARE_SQUARED_BINARY_OP(OP_NAME, OPERATION)                                          \
  template <typename X, typename Y, typename Z>                                                \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2) {                                         \
      Z diff = OPERATION;                                                                      \
      return sd::math::sd_multiply<Z, Z, Z>(diff, diff);                                       \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2, Z* params) {                              \
      Z diff = OPERATION;                                                                      \
      return sd::math::sd_multiply<Z, Z, Z>(diff, diff);                                       \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1) {                                               \
      return static_cast<Z>(d1);                                                              \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y* params) {                                    \
      Z diff = OPERATION##_PARAMS;                                                            \
      return sd::math::sd_multiply<Z, Z, Z>(diff, diff);                                       \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX, TY> op_simd(TX d1, TY d2) {          \
      return op_logic(d1, d2);                                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX, TY> op_simd(TX d1, TY d2) {        \
      return op_logic(d1, d2);                                                                \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX, TY> op_simd(TX d1, TY d2, TZ* params) { \
      return op_logic(d1, d2, params);                                                        \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX, TY> op_simd(TX d1, TY d2, TZ* params) { \
      return op_logic(d1, d2, params);                                                        \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX> op_simd(TX d1) {                     \
      return op_logic(d1);                                                                    \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TZ = Z>                                                 \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX> op_simd(TX d1) {                   \
      return op_logic(d1);                                                                    \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_safe<TZ, TX, TY> op_simd(TX d1, TY* params) {     \
      return op_logic(d1, params);                                                            \
    }                                                                                          \
                                                                                               \
    template<typename TX = X, typename TY = Y, typename TZ = Z>                               \
    SD_HOST_DEVICE SD_INLINE enable_if_simd_unsafe<TZ, TX, TY> op_simd(TX d1, TY* params) {   \
      return op_logic(d1, params);                                                            \
    }                                                                                          \
                                                                                               \
   public:                                                                                     \
    SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2) {                                               \
      DISPATCH_SIMD_BINARY(Z, X, Y, op_simd(d1, d2), op_logic(d1, d2))                       \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2, Z* params) {                                    \
      DISPATCH_SIMD_BINARY(Z, X, Y, op_simd(d1, d2, params), op_logic(d1, d2, params))       \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op(X d1) {                                                     \
      DISPATCH_SIMD_UNARY(Z, X, op_simd(d1), op_logic(d1))                                   \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE Z op(X d1, Y* params) {                                          \
      DISPATCH_SIMD_BINARY(Z, X, Y, op_simd(d1, params), op_logic(d1, params))               \
    }                                                                                          \
                                                                                               \
    SD_HOST_DEVICE SD_INLINE static X startingValue() { return static_cast<X>(0.f); }         \
  };

/**
 * @brief Declares a binary copy operation with proper SIMD handling
 */
#define DECLARE_BINARY_COPY_OP(OP_NAME, BINARY_OP, BINARY_PARAM_OP, UNARY_OP, PARAM_OP)          \
  template <typename X, typename Y, typename Z>                                                   \
  class OP_NAME {                                                                                 \
   private:                                                                                       \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2) { return BINARY_OP; }                \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2, Z *params) { return BINARY_PARAM_OP; } \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1) { return UNARY_OP; }                       \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y *params) { return PARAM_OP; }            \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }          \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1) { return op_logic(d1); }                    \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y *params) { return op_logic(d1, params); } \
                                                                                                  \
   public:                                                                                        \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2) {                                          \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                       \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                         \
        return op_logic(d1, d2);                                                                  \
      else return op_simd(d1, d2);                                                               \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2, Z *params) {                               \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                       \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                         \
        return op_logic(d1, d2, params);                                                          \
      else return op_simd(d1, d2, params);                                                       \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1) {                                                \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                         \
        return op_logic(d1);                                                                      \
      else return op_simd(d1);                                                                   \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y *params) {                                     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                       \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                         \
        return op_logic(d1, params);                                                              \
      else return op_simd(d1, params);                                                           \
    }                                                                                             \
  };

// =============================================================================
// BINARY PARAMETER OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a binary operation with parameters only
 */
#define DECLARE_BINARY_PARAM_OP(OP_NAME, OPERATION, BOILERPLATE)                               \
  template <typename X, typename Y, typename Z>                                                \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2, Z* params) { return OPERATION; }  \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2, Z* params) { return op_logic(d1, d2, params); } \
                                                                                               \
   public:                                                                                     \
    BOILERPLATE;                                                                               \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2, Z* params) {                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                    \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                      \
        return op_logic(d1, d2, params);                                                      \
      else return op_simd(d1, d2, params);                                                    \
    }                                                                                          \
  };

/**
 * @brief Declares a binary math operation with different input/output types
 */
#define DECLARE_BINARY_MATH_OP_XZ(OP_NAME, MATH_FUNC)                                               \
  template <typename X, typename Y, typename Z>                                                     \
  class OP_NAME {                                                                                   \
   private:                                                                                         \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2) { return sd::math::MATH_FUNC<X COMMA Z>(d1, static_cast<X>(d2)); } \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2, Z* params) { return sd::math::MATH_FUNC<X COMMA Z>(d1, static_cast<X>(d2)); } \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1) { return static_cast<Z>(d1); }               \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y* params) { return sd::math::MATH_FUNC<X COMMA Z>(d1, static_cast<X>(params[0])); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }            \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2, Z* params) { return op_logic(d1, d2, params); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1) { return op_logic(d1); }                      \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y* params) { return op_logic(d1, params); }   \
                                                                                                    \
   public:                                                                                          \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2) {                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                           \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, d2);  \
      else return op_simd(d1, d2);                                                                 \
    }                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2, Z* params) {                                 \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                           \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, d2, params); \
      else return op_simd(d1, d2, params);                                                         \
    }                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1) {                                                  \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                           \
                    simdOps::is_simd_unsupported_argument_type<X>::value) return op_logic(d1);      \
      else return op_simd(d1);                                                                     \
    }                                                                                               \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y* params) {                                       \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                           \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<Y>::value) return op_logic(d1, params); \
      else return op_simd(d1, params);                                                             \
    }                                                                                               \
    SD_HOST_DEVICE SD_INLINE static X startingValue() { return static_cast<X>(1.f); }             \
  };

/**
 * @brief Declares a binary math operation with custom starting value
 */
#define DECLARE_BINARY_MATH_OP_WITH_STARTING(OP_NAME, BINARY_OP, UNARY_OP, PARAM_OP, STARTING_VAL) \
  template <typename X, typename Y, typename Z>                                                   \
  class OP_NAME {                                                                                 \
   private:                                                                                       \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2) { return BINARY_OP; }                \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y d2, Z *params) { return op_logic(d1, d2); } \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1) { return UNARY_OP; }                       \
    SD_HOST_DEVICE SD_INLINE static Z op_logic(X d1, Y *params) { return PARAM_OP; }            \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }          \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); } \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1) { return op_logic(d1); }                    \
    SD_HOST_DEVICE SD_INLINE static Z op_simd(X d1, Y *params) { return op_logic(d1, params); } \
                                                                                                  \
   public:                                                                                        \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2) {                                          \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                       \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                         \
        return op_logic(d1, d2);                                                                  \
      else return op_simd(d1, d2);                                                               \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y d2, Z *params) {                               \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                       \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                         \
        return op_logic(d1, d2, params);                                                          \
      else return op_simd(d1, d2, params);                                                       \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1) {                                                \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                         \
        return op_logic(d1);                                                                      \
      else return op_simd(d1);                                                                   \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static Z op(X d1, Y *params) {                                     \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||                         \
                    simdOps::is_simd_unsupported_argument_type<X>::value ||                       \
                    simdOps::is_simd_unsupported_argument_type<Y>::value)                         \
        return op_logic(d1, params);                                                              \
      else return op_simd(d1, params);                                                           \
    }                                                                                             \
    SD_HOST_DEVICE SD_INLINE static X startingValue() { return STARTING_VAL; }                  \
  };


#define DECLARE_SIMPLE_BINARY_OP(OP_NAME, OPERATION)                                           \
  template <typename X>                                                                        \
  class OP_NAME {                                                                             \
   private:                                                                                   \
    static SD_INLINE X op_logic(X d1, X d2) { return OPERATION; }                            \
    static SD_INLINE X op_logic(X d1, X d2, X *params) { return op_logic(d1, d2); }          \
    static SD_INLINE SD_HOST_DEVICE X op_simd(X d1, X d2) { return op_logic(d1, d2); }                                \
    static SD_INLINE SD_HOST_DEVICE X op_simd(X d1, X d2, X *params) { return op_logic(d1, d2, params); }             \
                                                                                              \
   public:                                                                                    \
    static SD_INLINE SD_HOST_DEVICE X op(X d1, X d2) {                                                                 \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                       \
        return op_logic(d1, d2);                                                             \
      else return op_simd(d1, d2);                                                           \
    }                                                                                         \
    static SD_INLINE SD_HOST_DEVICE X op(X d1, X d2, X *params) {                                                      \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                       \
        return op_logic(d1, d2, params);                                                     \
      else return op_simd(d1, d2, params);                                                   \
    }                                                                                         \
  };


#define DECLARE_SIMPLE_BINARY_TEMPLATE_OP(OP_NAME, OPERATION)                                  \
  template <typename X>                                                                        \
  class OP_NAME {                                                                             \
   private:                                                                                   \
    static SD_HOST_DEVICE SD_INLINE X op_logic(X d1, X d2) { return OPERATION; }                            \
    static SD_HOST_DEVICE SD_INLINE X op_logic(X d1, X d2, X *params) { return op_logic(d1, d2); }          \
    static SD_HOST_DEVICE SD_INLINE X op_simd(X d1, X d2) { return op_logic(d1, d2); }                                \
    static SD_HOST_DEVICE SD_INLINE X op_simd(X d1, X d2, X *params) { return op_logic(d1, d2, params); }             \
                                                                                              \
   public:                                                                                    \
    static SD_HOST_DEVICE SD_INLINE  X op(X d1, X d2) {                                                                 \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                       \
        return op_logic(d1, d2);                                                             \
      else return op_simd(d1, d2);                                                           \
    }                                                                                         \
    static SD_HOST_DEVICE SD_INLINE X op(X d1, X d2, X *params) {                                                      \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                       \
        return op_logic(d1, d2, params);                                                     \
      else return op_simd(d1, d2, params);                                                   \
    }                                                                                         \
  };

}
#endif