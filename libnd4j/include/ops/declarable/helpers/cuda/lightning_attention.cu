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
 * distributed under the LICENSE is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Lightning Attention CUDA kernels.
//
// Implements linear attention with per-head exponential decay using the
// intra/inter-chunk decomposition from Lightning Attention-2 (TransNormerLLM).
// Algorithmic patterns based on cuLA (inclusionAI/cuLA).
//
// Reference: https://arxiv.org/abs/2405.17381
//
// Key design points (matching cuLA conventions):
//   - __exp2f in log2 domain instead of expf() for all decay computations —
//     maps to a single PTX instruction, ~5x faster than IEEE expf().
//   - FP32 accumulation for recurrent state S regardless of input type T,
//     preventing quantization error from compounding across chunks.
//   - Scale deferred: applied once at output write, not inside accumulations.
//   - Prefill kernel loops over ALL chunks in sequence within each block,
//     ensuring the recurrent state dependency is respected without any
//     synchronisation across blocks.
//   - Decode kernel handles single-token generation efficiently.
//

#include <ops/declarable/helpers/lightning_attention.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <types/float16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

// Accumulator type: double when T=double for precision, float otherwise.
// The recurrent state is always float32 by design; AccT governs Q/K/V tiles
// and the dot products that produce output (so double input -> double output).
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int LA_CHUNK_SIZE    = 64;   // Intra-chunk tile size (C)
static constexpr int LA_WARP_SIZE     = 32;
static constexpr int LA_BLOCK_PREFILL = 256;  // Threads per block for prefill
static constexpr int LA_BLOCK_DECODE  = 128;  // Threads per block for decode

// 1/ln(2): converts natural-log decay to log2 domain.
// exp2f(-s * d * RCP_LN2) == expf(-s * d)
static constexpr float LA_RCP_LN2 = 1.4426950408889634f;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Decay weight for a token at distance `distance` from the reference position.
// Uses exp2f for the fast PTX path instead of expf.
SD_DEVICE SD_INLINE float la_decay_weight(float decay_s, int distance) {
    return exp2f(-decay_s * static_cast<float>(distance) * LA_RCP_LN2);
}

// Global decay across a full chunk of `chunk_size` tokens.
SD_DEVICE SD_INLINE float la_decay_chunk(float decay_s, int chunk_size) {
    return exp2f(-decay_s * static_cast<float>(chunk_size) * LA_RCP_LN2);
}

// ---------------------------------------------------------------------------
// Kernel 1: lightningAttentionPrefillKernel
//
// Processes a full sequence by iterating over all chunks WITHIN each block.
// The sequential loop ensures the recurrent state S is consistent across
// chunk boundaries (chunk i reads the state written by chunk i-1).
//
// Grid:  (batch * numHeads, 1)   — one block per (batch, head) pair
// Block: LA_BLOCK_PREFILL threads
//
// For each chunk i of size C:
//   O_inter[pos, d] = sum_k Q[pos, k] * S[k, d]           (inter-chunk)
//   A[i, j]         = Q[i].K[j] * decay_mask[i, j]        (intra scores)
//   O_intra[pos, d] = sum_j A[pos, j] * V[j, d]           (intra output)
//   S               = decay^C * S + weighted K^T V          (state update)
//   O[pos, d]       = O_inter[pos, d] + O_intra[pos, d]   (combine)
//
// Shared memory layout:
//   smQ      [LA_CHUNK_SIZE * headDim]  AccT  — Q tile
//   smK      [LA_CHUNK_SIZE * headDim]  AccT  — K tile
//   smV      [LA_CHUNK_SIZE * headDim]  AccT  — V tile
//   smScores [LA_CHUNK_SIZE * LA_CHUNK_SIZE] AccT — intra-attn scores
//   smO      [LA_CHUNK_SIZE * headDim]  AccT  — output accumulator
//
// State is in global memory (B*H*D*D float32, too large for shared mem).
// ---------------------------------------------------------------------------
template <typename T>
SD_KERNEL __launch_bounds__(LA_BLOCK_PREFILL, 2)
void lightningAttentionPrefillKernel(
        const T*    __restrict__ query,       // [B, S, H, D]
        const T*    __restrict__ key,         // [B, S, H, D]
        const T*    __restrict__ value,       // [B, S, H, D]
        const float* __restrict__ decayRates, // [H]  per-head decay
        float*      __restrict__ state,       // [B, H, D, D]  float32, in/out
        T*          __restrict__ output,      // [B, S, H, D]
        const LongType B,
        const LongType S,
        const LongType H,
        const LongType D,
        // BSHD strides for Q
        const LongType qS0, const LongType qS1, const LongType qS2, const LongType qS3,
        // BSHD strides for K
        const LongType kS0, const LongType kS1, const LongType kS2, const LongType kS3,
        // BSHD strides for V
        const LongType vS0, const LongType vS1, const LongType vS2, const LongType vS3,
        // BSHD strides for output
        const LongType oS0, const LongType oS1, const LongType oS2, const LongType oS3,
        const LongType numChunks,
        const bool isCausal) {

    using AccT = typename AccType<T>::type;

    const LongType bh = blockIdx.x;
    if (bh >= B * H) return;

    const LongType b = bh / H;
    const LongType h = bh % H;
    const float decay_s = decayRates[h];

    // Shared memory layout — tiles in AccT (double for T=double, float otherwise).
    // State is always float32 (recurrent state stays float by design).
    extern __shared__ char smem[];
    AccT* smQ      = reinterpret_cast<AccT*>(smem);
    AccT* smK      = smQ + LA_CHUNK_SIZE * D;
    AccT* smV      = smK + LA_CHUNK_SIZE * D;
    AccT* smScores = smV + LA_CHUNK_SIZE * D;
    AccT* smO      = smScores + LA_CHUNK_SIZE * LA_CHUNK_SIZE;

    float* statePtr = state + (b * H + h) * D * D;
    const int tid = threadIdx.x;

    // Iterate over chunks sequentially — maintains recurrent state ordering.
    for (LongType ci = 0; ci < numChunks; ++ci) {
        const LongType chunkStart = ci * LA_CHUNK_SIZE;
        const LongType chunkEndRaw = chunkStart + LA_CHUNK_SIZE;
        const LongType chunkEnd = chunkEndRaw < S ? chunkEndRaw : S;
        const int C = static_cast<int>(chunkEnd - chunkStart);

        // --- Load Q, K, V tiles into shared memory as AccT ---
        for (int pos = 0; pos < C; ++pos) {
            const LongType s = chunkStart + pos;
            for (int d = tid; d < static_cast<int>(D); d += LA_BLOCK_PREFILL) {
                smQ[pos * D + d] = static_cast<AccT>(query[b * qS0 + s * qS1 + h * qS2 + d * qS3]);
                smK[pos * D + d] = static_cast<AccT>(key  [b * kS0 + s * kS1 + h * kS2 + d * kS3]);
                smV[pos * D + d] = static_cast<AccT>(value[b * vS0 + s * vS1 + h * vS2 + d * vS3]);
            }
        }
        // Zero the output accumulator for this chunk
        for (int idx = tid; idx < C * static_cast<int>(D); idx += LA_BLOCK_PREFILL)
            smO[idx] = static_cast<AccT>(0);
        __syncthreads();

        // --- Inter-chunk contribution: O_inter[pos, dv] = sum_dk Q[pos, dk] * S[dk, dv] ---
        // Each thread handles a stripe of positions; the inner loops cover all D.
        for (int pos = tid; pos < C; pos += LA_BLOCK_PREFILL) {
            for (int dv = 0; dv < static_cast<int>(D); ++dv) {
                AccT acc = static_cast<AccT>(0);
                for (int dk = 0; dk < static_cast<int>(D); ++dk)
                    acc += smQ[pos * D + dk] * static_cast<AccT>(statePtr[dk * D + dv]);
                smO[pos * D + dv] += acc;
            }
        }
        __syncthreads();

        // --- Intra-chunk attention: A[i,j] * V -> O_intra ---
        // decay_mask[i, j] = exp2f(-decay_s * (i - j) * RCP_LN2)  for j <= i
        //                   0                                       for j > i  (causal)
        // When !isCausal, all j are allowed but decay still applies to |i - j|.
        for (int i = 0; i < C; ++i) {
            // Compute score row A[i, *]
            for (int j = tid; j < C; j += LA_BLOCK_PREFILL) {
                AccT score = static_cast<AccT>(0);
                if (!isCausal || j <= i) {
                    for (int d = 0; d < static_cast<int>(D); ++d)
                        score += smQ[i * D + d] * smK[j * D + d];
                    const int dist = (i >= j) ? (i - j) : (j - i);
                    score *= static_cast<AccT>(la_decay_weight(decay_s, dist));
                }
                smScores[i * LA_CHUNK_SIZE + j] = score;
            }
            __syncthreads();

            // Accumulate O_intra[i, dv] += sum_j score[i,j] * V[j, dv]
            for (int dv = tid; dv < static_cast<int>(D); dv += LA_BLOCK_PREFILL) {
                AccT acc = static_cast<AccT>(0);
                for (int j = 0; j < C; ++j)
                    acc += smScores[i * LA_CHUNK_SIZE + j] * smV[j * D + dv];
                smO[i * D + dv] += acc;
            }
            __syncthreads();
        }

        // --- Write combined output for this chunk ---
        for (int pos = 0; pos < C; ++pos) {
            const LongType s = chunkStart + pos;
            for (int d = tid; d < static_cast<int>(D); d += LA_BLOCK_PREFILL)
                output[b * oS0 + s * oS1 + h * oS2 + d * oS3] = static_cast<T>(smO[pos * D + d]);
        }

        // --- State update: S = decay^C * S + sum_{pos} decay^(C-1-pos) * K[pos]^T * V[pos] ---
        // State is always float32 to prevent quantization drift.
        // K and V tiles are AccT; cast back to float for the float32 state accumulation.
        const float decayC = la_decay_chunk(decay_s, C);
        for (int idx = tid; idx < static_cast<int>(D * D); idx += LA_BLOCK_PREFILL) {
            const int dk = idx / static_cast<int>(D);
            const int dv = idx % static_cast<int>(D);
            float kv_acc = 0.0f;
            for (int pos = 0; pos < C; ++pos) {
                // Token at position `pos` in chunk is `C - 1 - pos` steps before
                // the end of the chunk, so its contribution is decayed accordingly.
                float w = la_decay_weight(decay_s, C - 1 - pos);
                kv_acc += w * static_cast<float>(smK[pos * D + dk]) * static_cast<float>(smV[pos * D + dv]);
            }
            statePtr[idx] = decayC * statePtr[idx] + kv_acc;
        }
        __syncthreads();  // Ensure state write is visible before next chunk reads it
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: lightningAttentionDecodeKernel
//
// Single-token decode (seqLen == 1).
// Grid:  (batch * numHeads, 1)
// Block: LA_BLOCK_DECODE threads
//
//   output = Q @ S           — matrix-vector product (all float32)
//   S      = decay * S + K ⊗ V  — rank-1 state update
//
// Shared memory:
//   smQ [D]     AccT  — Q for this token
//   smK [D]     AccT  — K for this token
//   smV [D]     AccT  — V for this token
//   smOut [D]   AccT  — output accumulator
// ---------------------------------------------------------------------------
template <typename T>
SD_KERNEL __launch_bounds__(LA_BLOCK_DECODE, 4)
void lightningAttentionDecodeKernel(
        const T*    __restrict__ query,       // [B, 1, H, D]
        const T*    __restrict__ key,         // [B, 1, H, D]
        const T*    __restrict__ value,       // [B, 1, H, D]
        const float* __restrict__ decayRates, // [H]
        float*      __restrict__ state,       // [B, H, D, D]  float32, in/out
        T*          __restrict__ output,      // [B, 1, H, D]
        const LongType B,
        const LongType H,
        const LongType D,
        // Strides: seq dim is 1 so we only need batch, head, dim strides
        const LongType qS0, const LongType qS2, const LongType qS3,
        const LongType kS0, const LongType kS2, const LongType kS3,
        const LongType vS0, const LongType vS2, const LongType vS3,
        const LongType oS0, const LongType oS2, const LongType oS3) {

    using AccT = typename AccType<T>::type;

    const LongType bh = blockIdx.x;
    if (bh >= B * H) return;

    const LongType b = bh / H;
    const LongType h = bh % H;
    const float decay_s = decayRates[h];

    // Shared memory: 4 tiles of D AccT values (Q, K, V, Out).
    extern __shared__ char smDecodeBuf[];
    AccT* smQ   = reinterpret_cast<AccT*>(smDecodeBuf);
    AccT* smK   = smQ + D;
    AccT* smV   = smK + D;
    AccT* smOut = smV + D;

    const int tid = threadIdx.x;

    // Load Q, K, V into shared memory as AccT
    for (int d = tid; d < static_cast<int>(D); d += LA_BLOCK_DECODE) {
        smQ[d]   = static_cast<AccT>(query[b * qS0 + h * qS2 + d * qS3]);
        smK[d]   = static_cast<AccT>(key  [b * kS0 + h * kS2 + d * kS3]);
        smV[d]   = static_cast<AccT>(value[b * vS0 + h * vS2 + d * vS3]);
        smOut[d] = static_cast<AccT>(0);
    }
    __syncthreads();

    float* statePtr = state + (b * H + h) * D * D;

    // Step 1: output[dv] = sum_dk Q[dk] * S[dk, dv]   (result in AccT)
    for (int dv = tid; dv < static_cast<int>(D); dv += LA_BLOCK_DECODE) {
        AccT acc = static_cast<AccT>(0);
        for (int dk = 0; dk < static_cast<int>(D); ++dk)
            acc += smQ[dk] * static_cast<AccT>(statePtr[dk * D + dv]);
        smOut[dv] = acc;
    }
    __syncthreads();

    // Write output
    for (int d = tid; d < static_cast<int>(D); d += LA_BLOCK_DECODE)
        output[b * oS0 + h * oS2 + d * oS3] = static_cast<T>(smOut[d]);

    // Step 2: S[dk, dv] = decay * S[dk, dv] + K[dk] * V[dv]
    // State is always float32; cast AccT K/V back to float for state update.
    // Each thread owns a horizontal strip (dk row) of the state matrix.
    for (int dk = tid; dk < static_cast<int>(D); dk += LA_BLOCK_DECODE) {
        const float k_val = static_cast<float>(smK[dk]);
        for (int dv = 0; dv < static_cast<int>(D); ++dv)
            statePtr[dk * D + dv] = decay_s * statePtr[dk * D + dv] + k_val * static_cast<float>(smV[dv]);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: lightningAttentionStateInitKernel
//
// Converts a T-typed initial state to float32 working buffer.
// Used when the caller passes stateIn in the model's native dtype.
// Grid: (ceil(stateElems / 256))
// Block: 256
// ---------------------------------------------------------------------------
template <typename T>
SD_KERNEL __launch_bounds__(256, 4)
void lightningAttentionStateInitKernel(
        const T*  __restrict__ src,
        float*    __restrict__ dst,
        const LongType stateElems) {
    const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < stateElems)
        dst[idx] = static_cast<float>(src[idx]);
}

// ---------------------------------------------------------------------------
// Type-dispatched implementation
// ---------------------------------------------------------------------------

template <typename T>
static void lightningAttentionImpl(
        LaunchContext* context,
        NDArray* query, NDArray* key, NDArray* value,
        NDArray* decayRates, NDArray* state, NDArray* output,
        bool isCausal) {

    auto stream = context->getCudaStream();

    const LongType B = query->sizeAt(0);
    const LongType S = query->sizeAt(1);
    const LongType H = query->sizeAt(2);
    const LongType D = query->sizeAt(3);

    const LongType qS0 = query->strideAt(0),  qS1 = query->strideAt(1),
                   qS2 = query->strideAt(2),  qS3 = query->strideAt(3);
    const LongType kS0 = key->strideAt(0),    kS1 = key->strideAt(1),
                   kS2 = key->strideAt(2),    kS3 = key->strideAt(3);
    const LongType vS0 = value->strideAt(0),  vS1 = value->strideAt(1),
                   vS2 = value->strideAt(2),  vS3 = value->strideAt(3);
    const LongType oS0 = output->strideAt(0), oS1 = output->strideAt(1),
                   oS2 = output->strideAt(2), oS3 = output->strideAt(3);

    const auto q   = reinterpret_cast<const T*>(query->specialBuffer());
    const auto k   = reinterpret_cast<const T*>(key->specialBuffer());
    const auto v   = reinterpret_cast<const T*>(value->specialBuffer());
    const auto dec = reinterpret_cast<const float*>(decayRates->specialBuffer());
    auto st  = reinterpret_cast<float*>(state->specialBuffer());
    auto out = reinterpret_cast<T*>(output->specialBuffer());

    using AccT = typename AccType<T>::type;

    if (S == 1) {
        // --- Decode path: single token ---
        dim3 grid(static_cast<unsigned>(B * H));
        dim3 block(LA_BLOCK_DECODE);
        // 4 tiles of D AccT values: smQ, smK, smV, smOut
        const size_t smem = 4 * static_cast<size_t>(D) * sizeof(AccT);

        lightningAttentionDecodeKernel<T><<<grid, block, smem, *stream>>>(
            q, k, v, dec, st, out,
            B, H, D,
            qS0, qS2, qS3,
            kS0, kS2, kS3,
            vS0, vS2, vS3,
            oS0, oS2, oS3);
        DebugHelper::checkGlobalErrorCode("lightningAttentionDecodeKernel failed");

    } else {
        // --- Prefill path: full sequence chunked ---
        const LongType numChunks = (S + LA_CHUNK_SIZE - 1) / LA_CHUNK_SIZE;

        dim3 grid(static_cast<unsigned>(B * H));
        dim3 block(LA_BLOCK_PREFILL);

        // Shared memory (all tiles in AccT — double for T=double, float otherwise):
        //   3 tiles of [LA_CHUNK_SIZE * D] for Q/K/V
        //   1 tile  of [LA_CHUNK_SIZE * LA_CHUNK_SIZE] for scores
        //   1 tile  of [LA_CHUNK_SIZE * D] for output accumulator
        const size_t smem =
            (static_cast<size_t>(3) * LA_CHUNK_SIZE * D +
             static_cast<size_t>(LA_CHUNK_SIZE) * LA_CHUNK_SIZE +
             static_cast<size_t>(LA_CHUNK_SIZE) * D)
            * sizeof(AccT);

        lightningAttentionPrefillKernel<T><<<grid, block, smem, *stream>>>(
            q, k, v, dec, st, out,
            B, S, H, D,
            qS0, qS1, qS2, qS3,
            kS0, kS1, kS2, kS3,
            vS0, vS1, vS2, vS3,
            oS0, oS1, oS2, oS3,
            numChunks,
            isCausal);
        DebugHelper::checkGlobalErrorCode("lightningAttentionPrefillKernel failed");
    }
}

// ---------------------------------------------------------------------------
// BUILD_SINGLE_TEMPLATE dispatch trampoline
// ---------------------------------------------------------------------------
template <typename T>
static void lightningAttentionLauncher(
        LaunchContext* context,
        NDArray* query, NDArray* key, NDArray* value,
        NDArray* decayRates, NDArray* state, NDArray* output,
        bool isCausal) {
    lightningAttentionImpl<T>(context, query, key, value, decayRates, state, output, isCausal);
}

BUILD_SINGLE_TEMPLATE(void lightningAttentionLauncher,
    (LaunchContext* context,
     NDArray* query, NDArray* key, NDArray* value,
     NDArray* decayRates, NDArray* state, NDArray* output,
     bool isCausal),
    SD_FLOAT_TYPES);

// ---------------------------------------------------------------------------
// Public CUDA entry point
// ---------------------------------------------------------------------------
void lightningAttention(LaunchContext* context,
                             NDArray* query,
                             NDArray* key,
                             NDArray* value,
                             NDArray* decayRates,
                             NDArray* state,
                             NDArray* output,
                             bool isCausal) {
    // state is float32 and is both read (as prior state) and written (updated state).
    // decayRates is always float32.
    NDArray::prepareSpecialUse({output, state}, {query, key, value, decayRates, state});

    BUILD_SINGLE_SELECTOR(query->dataType(), lightningAttentionLauncher,
        (context, query, key, value, decayRates, state, output, isCausal),
        SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output, state}, {query, key, value, decayRates, state});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
