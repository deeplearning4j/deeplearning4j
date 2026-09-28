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
 * Offset of the element of a secondary input that the output element at coords (outRank
 * coordinates) reads. The secondary is right-aligned against the output; its size-1 dimensions
 * repeat. A single-element secondary is read at offset 0 by the callers instead.
 */
SD_HOST_DEVICE SD_INLINE LongType fusedChainBroadcastOffset(const LongType* coords, int outRank,
                                                           const LongType* shape, const LongType* strides,
                                                           int rank) {
  const int lead = outRank - rank;
  LongType offset = 0;
  for (int d = 0; d < rank; d++) {
    if (shape[d] != 1) offset += coords[lead + d] * strides[d];
  }
  return offset;
}

/**
 * Apply one chain member to the chain value v. s is the member's other operand (binary codes
 * and FUSED_LEAKY_RELU's alpha); clipMin/clipMax are FUSED_CLIP's bounds, already rounded to T,
 * which gives the same result as eager clipbyvalue comparing against the unrounded bounds.
 * Codes must pass isImplementedFusedOp() (fusedElementwiseChain.h) on the host before a kernel
 * runs; every case here must be listed there.
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

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_ELEMENTWISE_CHAIN_MATH_H
