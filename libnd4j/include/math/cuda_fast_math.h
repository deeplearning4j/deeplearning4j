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

//
// cuda_fast_math.h — fast math intrinsics for CUDA hot paths.
//
// All functions are device-only and inlined.  They trade a small amount of
// IEEE-754 accuracy (~4 ULP) for a large throughput improvement (~4-5x vs
// the compliant versions) by exploiting CUDA's native hardware instructions:
//
//   __expf   → approximately 20 ns vs 100 ns for expf
//   exp2f    → maps to ex2.approx.f32 PTX instruction (~4 cycles)
//   __logf   → approximately 5 ns vs 25 ns for logf
//
// The accuracy loss is safe for softmax/attention/gating computations where
// normalisation cancels relative errors.  Do NOT use for scientific/precision
// workloads where ULP accuracy is required.
//
// exp2-based approach (fast_exp_via_exp2):
//   Uses the identity exp(x) = exp2(x * log2(e)).
//   This is the technique employed by cuLA for all gating / decay operations.
//   It maps to a single ex2.approx.f32 PTX instruction with ~4 cycle latency,
//   versus ~20 cycles for the compliant expf().
//

#ifndef LIBND4J_CUDA_FAST_MATH_H
#define LIBND4J_CUDA_FAST_MATH_H

#include <system/common.h>

#ifdef __CUDACC__

namespace sd {
namespace math {

// ---------------------------------------------------------------------------
// Exponentials
// ---------------------------------------------------------------------------

// Fast exponential using CUDA intrinsic __expf (~4 ULP error).
// Safe for softmax/attention where normalisation cancels relative error.
// ~5x faster than IEEE-compliant expf().
SD_DEVICE SD_INLINE float fast_exp(float x) {
    return __expf(x);
}

// Fast exponential via exp2f: maps to single PTX instruction ex2.approx.f32.
// Mathematically: exp(x) = exp2(x * log2(e)) = exp2(x * 1.4426950408...)
// This is the approach used by cuLA for all gating / decay computations.
// ~4 cycle latency vs ~20 for expf.
SD_DEVICE SD_INLINE float fast_exp_via_exp2(float x) {
    // log2(e) = 1 / ln(2) ≈ 1.4426950408889634
    constexpr float LOG2E = 1.4426950408889634f;
    return exp2f(x * LOG2E);
}

// Fast exp2: direct PTX ex2.approx.f32 instruction.
// Use when values are already in the log2 domain (gates, decay rates).
SD_DEVICE SD_INLINE float fast_exp2(float x) {
    return exp2f(x);
}

// ---------------------------------------------------------------------------
// Domain-conversion constants
// ---------------------------------------------------------------------------

// Reciprocal of ln(2) — converts natural-log domain to log2 domain.
// Usage: store gates as g_log2 = g_nat * RCP_LN2, then replay with exp2f(g_log2).
constexpr float RCP_LN2 = 1.4426950408889634f;

// ln(2) — converts log2 domain back to natural-log domain.
constexpr float LN2 = 0.6931471805599453f;

// ---------------------------------------------------------------------------
// Activation functions built on __expf
// ---------------------------------------------------------------------------

// Fast sigmoid using __expf intrinsic.
// sigmoid(x) = 1 / (1 + exp(-x))
SD_DEVICE SD_INLINE float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Fast softplus using numerically stable formulation.
// softplus(x) = max(0, x) + log(1 + exp(-|x|))
// Uses __expf and __logf intrinsics throughout.
// Numerically stable for all finite x (avoids overflow for large positive x).
SD_DEVICE SD_INLINE float fast_softplus(float x) {
    float ax = fabsf(x);
    return fmaxf(0.0f, x) + __logf(1.0f + __expf(-ax));
}

// Fast SiLU (Swish) activation: x * sigmoid(x).
// Equivalent to x / (1 + exp(-x)).
SD_DEVICE SD_INLINE float fast_silu(float x) {
    return x * fast_sigmoid(x);
}

// Fast GELU approximation using tanh.
// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// Matches PyTorch's gelu_new / OpenAI implementation.
SD_DEVICE SD_INLINE float fast_gelu(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2 / π)
    constexpr float COEFF          = 0.044715f;
    float x3    = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

}  // namespace math
}  // namespace sd

#endif  // __CUDACC__
#endif  // LIBND4J_CUDA_FAST_MATH_H
