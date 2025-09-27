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
#ifndef OP_MACROS_UNARY_H_
#define OP_MACROS_UNARY_H_

#include "op_types.h"
#include <math/templatemath.h>

namespace simdOps {

// =============================================================================
// UNARY MATH OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a unary math operation with proper SIMD handling
 */
#define DECLARE_UNARY_MATH_OP(OP_NAME, MATH_FUNC)                                               \
  template <typename X>                                                                         \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) { return sd::math::MATH_FUNC<X COMMA X>(d1); }                               \
    static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }                         \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                       \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                         \
        return op_logic(d1, params);                                                           \
      else                                                                                      \
        return op_simd(d1, params);                                                            \
    }                                                                                           \
  };

/**
 * @brief Declares a unary math operation with different input/output types
 */
#define DECLARE_UNARY_MATH_OP_XZ(OP_NAME, MATH_FUNC)                                           \
  template <typename X, typename Z>                                                            \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  Z op_logic(X d1, Z* params) { return sd::math::MATH_FUNC<X COMMA Z>(d1); }             \
    static SD_HOST_DEVICE Z op_simd(X d1, Z* params) { return op_logic(d1, params); }                         \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special no_op_exec_special_cuda;                                                \
    static SD_HOST_DEVICE Z op(X d1, Z* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                        \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

// =============================================================================
// UNARY SIMPLE OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a simple unary operation with direct expression
 */
#define DECLARE_UNARY_SIMPLE_OP(OP_NAME, OPERATION)                                            \
  template <typename X>                                                                        \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) {                                            \
      if constexpr (std::is_same_v<X COMMA bool>) {                                                \
        return d1; /* Safe default for bool to avoid multiplication warnings */              \
      } else {                                                                                 \
        return OPERATION;                                                                      \
      }                                                                                        \
    }                                                                                          \
    static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }                        \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                      \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value || std::is_same_v<X COMMA bool>) \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

// =============================================================================
// UNARY CONDITIONAL OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a unary conditional operation
 */
#define DECLARE_UNARY_CONDITIONAL_OP(OP_NAME, CONDITION, TRUE_VAL, FALSE_VAL)                  \
  template <typename X>                                                                        \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) { return (CONDITION) ? (TRUE_VAL) : (FALSE_VAL); }                \
    static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }                        \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                      \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

// =============================================================================
// UNARY COMPLEX CONDITIONAL OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a unary operation with complex conditional logic (if-else chains)
 */
/**
 * @brief Declares a unary operation with complex conditional logic (if-else chains)
 */
#define DECLARE_UNARY_COMPLEX_CONDITIONAL_OP(OP_NAME, IF_CONDITION1, RETURN1, ELIF_CONDITION2, RETURN2, ELSE_RETURN) \
  template <typename X>                                                                        \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) {                                            \
      if (IF_CONDITION1)                                                                      \
        return RETURN1;                                                                        \
      else if (ELIF_CONDITION2)                                                               \
        return RETURN2;                                                                        \
      else                                                                                     \
        return ELSE_RETURN;                                                                    \
    }                                                                                          \
    SD_OP_DEF static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }              \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_same;                                                                   \
    no_op_exec_special_same_cuda;                                                              \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value ||                      \
                    simdOps::is_simd_unsupported_argument_type<X>::value)                      \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

// =============================================================================
// UNARY COMPLEX MATH EXPRESSION MACROS
// =============================================================================

/**
 * @brief Declares a unary operation with complex mathematical expressions
 */
#define DECLARE_UNARY_COMPLEX_MATH_OP(OP_NAME, MATH_EXPRESSION)                                \
  template <typename X>                                                                        \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) { return MATH_EXPRESSION; }                  \
    static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }                        \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                      \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

/**
 * @brief Declares a unary operation with complex math expression and different input/output types
 */
#define DECLARE_UNARY_COMPLEX_MATH_OP_XZ(OP_NAME, MATH_EXPRESSION)                             \
  template <typename X, typename Z>                                                            \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  Z op_logic(X d1, Z* params) { return MATH_EXPRESSION; }                  \
    static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1, Z* params) { return op_logic(d1, params); }                        \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special no_op_exec_special_cuda;                                                \
    static SD_HOST_DEVICE Z op(X d1, Z* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)                        \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

// =============================================================================
// UNARY IDENTITY OPERATION MACROS
// =============================================================================

/**
 * @brief Declares a unary identity operation (just returns input)
 */
#define DECLARE_UNARY_IDENTITY_OP(OP_NAME)                                                     \
  template <typename X>                                                                        \
  class OP_NAME {                                                                              \
   private:                                                                                    \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) { return d1; }                               \
    static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }                        \
                                                                                               \
   public:                                                                                     \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                      \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                            \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                        \
        return op_logic(d1, params);                                                          \
      else                                                                                     \
        return op_simd(d1, params);                                                           \
    }                                                                                          \
  };

/**
 * @brief DECLARE_UNARY_SIMD_SAFE_OP macro - Add this to op_macros_unary.h
 * This macro creates unary operations with proper SIMD handling
 */
#define DECLARE_UNARY_SIMD_SAFE_OP(OP_NAME, OPERATION)                                              \
  template <typename X>                                                                          \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) { OPERATION }                                 \
    static SD_HOST_DEVICE X op_simd(X d1, X* params) { return op_logic(d1, params); }                         \
                                                                                                \
   public:                                                                                      \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                       \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                         \
        return op_logic(d1, params);                                                           \
      else                                                                                      \
        return op_simd(d1, params);                                                             \
    }                                                                                           \
  };

/**
 * @brief Alternative version using the new SIMD-safe pattern for consistency
 */
#define DECLARE_UNARY_SIMD_SAFE_OP_V2(OP_NAME, OPERATION)                                       \
  template <typename X>                                                                         \
  class OP_NAME {                                                                               \
   private:                                                                                     \
    SD_HOST_DEVICE SD_INLINE static  X op_logic(X d1, X* params) { OPERATION }                                 \
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
   public:                                                                                      \
    no_op_exec_special_same no_op_exec_special_same_cuda;                                       \
    static SD_HOST_DEVICE X op(X d1, X* params) {                                                             \
      if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)                         \
        return op_logic(d1, params);                                                           \
      else                                                                                      \
        return op_simd(d1, params);                                                             \
    }                                                                                           \
  };


} // namespace simdOps

#endif // OP_MACROS_UNARY_H_