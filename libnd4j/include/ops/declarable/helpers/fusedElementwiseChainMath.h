/* ******************************************************************************
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

//
// Per-element math shared by the CPU and CUDA fused element-wise chain kernels.
//
// A fused chain must produce exactly what running its members one by one produces. Every
// member therefore runs the simdOps functor its eager op uses, with the storage type T for
// both operands and the result, so each step rounds to T once -- the rounding a materialized
// intermediate gets. The chain value is always the functor's first operand (callers map a
// member whose chain value is the right-hand operand to its swapped code), and relu, relu6
// and elu use the fixed parameter their eager op was admitted with.
//

#ifndef LIBND4J_FUSED_ELEMENTWISE_CHAIN_MATH_H
#define LIBND4J_FUSED_ELEMENTWISE_CHAIN_MATH_H

#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <ops/ops.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Apply one chain member to the chain value v. s is the member's other operand (binary codes
 * and FUSED_LEAKY_RELU's alpha); clipMin/clipMax are FUSED_CLIP's bounds, already rounded to T,
 * which gives the same result as eager clipbyvalue comparing against the unrounded bounds.
 * Codes must pass isImplementedFusedOp() on the host before a kernel runs.
 */
template <typename T>
SD_HOST_DEVICE SD_INLINE T fusedChainStep(int code, T v, T s, T clipMin, T clipMax) {
  switch (code) {
    case FUSED_ADD:         return simdOps::Add<T, T, T>::op(v, s, nullptr);
    case FUSED_SUB:         return simdOps::Subtract<T, T, T>::op(v, s, nullptr);
    case FUSED_MUL:         return simdOps::Multiply<T, T, T>::op(v, s, nullptr);
    case FUSED_DIV:         return simdOps::Divide<T, T, T>::op(v, s, nullptr);
    case FUSED_REVERSE_SUB: return simdOps::ReverseSubtract<T, T, T>::op(v, s, nullptr);
    case FUSED_REVERSE_DIV: return simdOps::ReverseDivide<T, T, T>::op(v, s, nullptr);
    case FUSED_SQUARED_SUB: return simdOps::SquaredSubtract<T, T, T>::op(v, s, nullptr);
    case FUSED_MIN:         return simdOps::MinPairwise<T, T, T>::op(v, s, nullptr);
    case FUSED_MAX:         return simdOps::MaxPairwise<T, T, T>::op(v, s, nullptr);
    case FUSED_MOD:         return simdOps::Mod<T, T, T>::op(v, s, nullptr);
    case FUSED_ATAN2:       return simdOps::Atan2<T, T, T>::op(v, s, nullptr);
    case FUSED_FLOORDIV:    return simdOps::FloorDiv<T, T, T>::op(v, s, nullptr);
    case FUSED_POW:         return simdOps::Pow<T, T, T>::op(v, s, nullptr);
    case FUSED_MUL_NO_NAN:
      return s == static_cast<T>(0) ? static_cast<T>(0) : simdOps::Multiply<T, T, T>::op(v, s, nullptr);
    case FUSED_LEAKY_RELU:  return simdOps::LeakyRELU<T, T, T>::op(v, s, nullptr);

    case FUSED_RELU:        return simdOps::RELU<T, T, T>::op(v, static_cast<T>(0), nullptr);
    case FUSED_RELU6:       return simdOps::RELU6<T, T, T>::op(v, static_cast<T>(0), nullptr);
    case FUSED_ELU:         return simdOps::ELU<T, T, T>::op(v, static_cast<T>(1), nullptr);

    case FUSED_SIGMOID:     return simdOps::Sigmoid<T>::op(v, nullptr);
    case FUSED_TANH:        return simdOps::Tanh<T>::op(v, nullptr);
    case FUSED_GELU:        return simdOps::GELU<T>::op(v, nullptr);
    case FUSED_EXP:         return simdOps::Exp<T>::op(v, nullptr);
    case FUSED_LOG:         return simdOps::Log<T>::op(v, nullptr);
    case FUSED_ABS:         return simdOps::Abs<T>::op(v, nullptr);
    case FUSED_NEG:         return simdOps::Neg<T>::op(v, nullptr);
    case FUSED_SQUARE:      return simdOps::Square<T>::op(v, nullptr);
    case FUSED_SQRT:        return simdOps::Sqrt<T, T>::op(v, nullptr);
    case FUSED_RSQRT:       return simdOps::RSqrt<T, T>::op(v, nullptr);
    case FUSED_RECIPROCAL:  return simdOps::Reciprocal<T>::op(v, nullptr);
    case FUSED_SIGN:        return simdOps::Sign<T>::op(v, nullptr);
    case FUSED_ERF:         return simdOps::Erf<T>::op(v, nullptr);
    case FUSED_ERFC:        return simdOps::Erfc<T>::op(v, nullptr);
    case FUSED_LOG1P:       return simdOps::Log1p<T>::op(v, nullptr);
    case FUSED_CEIL:        return simdOps::Ceiling<T>::op(v, nullptr);
    case FUSED_FLOOR:       return simdOps::Floor<T>::op(v, nullptr);
    case FUSED_ROUND:       return simdOps::Round<T>::op(v, nullptr);
    case FUSED_SIN:         return simdOps::Sin<T>::op(v, nullptr);
    case FUSED_COS:         return simdOps::Cosine<T>::op(v, nullptr);
    case FUSED_SELU:        return simdOps::SELU<T>::op(v, nullptr);
    case FUSED_SOFTPLUS:    return simdOps::SoftPlus<T>::op(v, nullptr);
    case FUSED_SOFTSIGN:    return simdOps::SoftSign<T>::op(v, nullptr);
    case FUSED_HARD_SIGMOID: return simdOps::HardSigmoid<T>::op(v, nullptr);
    case FUSED_HARDTANH:    return simdOps::HardTanh<T>::op(v, nullptr);
    // swish is the legacy transform (one rounding); silu is the declarable, which stores the
    // sigmoid to its output before multiplying (two roundings).
    case FUSED_SWISH:       return simdOps::Swish<T>::op(v, nullptr);
    case FUSED_SILU:
      return simdOps::Multiply<T, T, T>::op(simdOps::Sigmoid<T>::op(v, nullptr), v, nullptr);
    case FUSED_MISH:        return simdOps::Mish<T>::op(v, nullptr);

    case FUSED_CLIP:        return v > clipMax ? clipMax : (v < clipMin ? clipMin : v);

    default:                return v;  // unreachable: codes are validated on the host
  }
}

/** Host-side mirror of fusedChainStep's cases. Keep the two lists identical. */
inline bool isImplementedFusedOp(int code) {
  switch (code) {
    case FUSED_ADD: case FUSED_SUB: case FUSED_MUL: case FUSED_DIV:
    case FUSED_REVERSE_SUB: case FUSED_REVERSE_DIV: case FUSED_SQUARED_SUB:
    case FUSED_MIN: case FUSED_MAX: case FUSED_MOD: case FUSED_ATAN2: case FUSED_FLOORDIV:
    case FUSED_POW: case FUSED_MUL_NO_NAN: case FUSED_LEAKY_RELU:
    case FUSED_RELU: case FUSED_RELU6: case FUSED_ELU:
    case FUSED_SIGMOID: case FUSED_TANH: case FUSED_GELU: case FUSED_EXP: case FUSED_LOG:
    case FUSED_ABS: case FUSED_NEG: case FUSED_SQUARE: case FUSED_SQRT: case FUSED_RSQRT:
    case FUSED_RECIPROCAL: case FUSED_SIGN: case FUSED_ERF: case FUSED_ERFC: case FUSED_LOG1P:
    case FUSED_CEIL: case FUSED_FLOOR: case FUSED_ROUND: case FUSED_SIN: case FUSED_COS:
    case FUSED_SELU: case FUSED_SOFTPLUS: case FUSED_SOFTSIGN: case FUSED_HARD_SIGMOID:
    case FUSED_HARDTANH: case FUSED_SWISH: case FUSED_SILU: case FUSED_MISH:
    case FUSED_CLIP:
      return true;
    default:
      return false;
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_ELEMENTWISE_CHAIN_MATH_H
