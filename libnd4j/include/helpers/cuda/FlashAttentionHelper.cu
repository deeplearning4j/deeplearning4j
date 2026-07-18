/* ******************************************************************************
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* See the NOTICE file distributed with this work for additional
* information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// Fused Scaled Dot-Product Attention CUDA kernel
// Based on Flash Attention algorithm with online softmax
// Reference: https://arxiv.org/abs/2205.14135
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <helpers/DebugHelper.h>
#include <graph/DspDiagnostics.h>
#include <helpers/PointersManager.h>
#include <helpers/FlashAttentionHelper.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <execution/cuda/LaunchDims.h>
#include <math/templatemath.h>

// Fast exponential for softmax hot paths.
// __expf has ~4 ULP error (vs ~1 ULP for expf), which is irrelevant for softmax
// because the normalization cancels relative error. This maps to a single PTX
// instruction and is ~5x faster than IEEE expf().
// Reference: cuLA (inclusionAI/cuLA) uses exp2f throughout for the same reason.
SD_DEVICE SD_INLINE float sd_fast_exp(float x) {
    return __expf(x);
}

namespace sd {

// Block sizes for tiling - tuned for modern GPUs (Ada Lovelace / Ampere)
// RTX 4090: 128 SMs, 100KB shared memory per SM, 1 TB/s memory bandwidth
constexpr int TILE_SIZE_Q = 64;   // Query tile size (increased from 32)
constexpr int TILE_SIZE_KV = 64;  // Key/Value tile size (increased from 32)
constexpr int WARP_SIZE = 32;
constexpr int DEFAULT_BLOCK_SIZE = 512;  // Increased from 256 for better occupancy

// Attention accumulators follow the same convention as other CUDA transformer
// kernels: double inputs accumulate in double, all other floating types use FP32.
template <typename T>
struct FlashAccType {
  using type = float;
};
template <>
struct FlashAccType<double> {
  using type = double;
};

template <typename AccT>
SD_DEVICE SD_INLINE AccT flashExp(AccT x) {
  return sd::math::sd_exp<AccT, AccT>(x);
}
template <>
SD_DEVICE SD_INLINE float flashExp<float>(float x) {
  return sd_fast_exp(x);
}

//////////////////////////////////////////////////////////////////////////////
// In-place causal mask kernel - sets scores[b, i, j] = -inf where j > i
// This replaces: create mask array + nullify + fillAsTriangular + broadcast add
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(256, 4) void applyCausalMaskInPlaceKernel(
   T* __restrict__ scores,  // [batch, seqQ, seqKV]
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV) {

 const LongType totalElements = batch * seqQ * seqKV;
 const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;
 const LongType causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;

 for (LongType idx = tid; idx < totalElements; idx += blockDim.x * gridDim.x) {
   // Convert linear index to (b, i, j)
   const LongType j = idx % seqKV;
   const LongType i = (idx / seqKV) % seqQ;
   // const LongType b = idx / (seqQ * seqKV);  // not needed

   // Decode-aware causal mask:
   // prefill (seqQ == seqKV): allow j <= i
   // decode  (seqQ == 1): allow all past keys via offset.
   if (j > (i + causalOffset)) {
     scores[idx] = static_cast<T>(-1.0e9f);
   }
 }
}

template <typename T>
static void applyCausalMaskInPlaceLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                          const cudaStream_t* stream, void* vScores,
                                          LongType batch, LongType seqQ, LongType seqKV) {
 auto scores = reinterpret_cast<T*>(vScores);
 applyCausalMaskInPlaceKernel<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(scores, batch, seqQ, seqKV);
 DebugHelper::checkGlobalErrorCode("applyCausalMaskInPlace failed");
}

BUILD_SINGLE_TEMPLATE(void applyCausalMaskInPlaceLauncher,
                     (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream,
                      void* vScores, LongType batch, LongType seqQ, LongType seqKV),
                     SD_FLOAT_TYPES);

// Public interface
void applyCausalMaskCuda(NDArray* scores, LaunchContext* context) {
 auto stream = context->getCudaStream();
 const auto batch = scores->sizeAt(0);
 const auto seqQ = scores->sizeAt(1);
 const auto seqKV = scores->sizeAt(2);

 const LongType totalElements = batch * seqQ * seqKV;
 const int blockSize = 256;
 const int numBlocks = (totalElements + blockSize - 1) / blockSize;

 NDArray::prepareSpecialUse({scores}, {scores});

 BUILD_SINGLE_SELECTOR(scores->dataType(), applyCausalMaskInPlaceLauncher,
                       (numBlocks, blockSize, stream, scores->specialBuffer(), batch, seqQ, seqKV),
                       SD_FLOAT_TYPES);

 NDArray::registerSpecialUse({scores}, {scores});
}

//////////////////////////////////////////////////////////////////////////////
// Fused causal mask + softmax kernel
// Each block handles one row (batch*seqQ rows total, each of length seqKV)
// Fuses: causal mask application + row-wise softmax
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(1024, 2) void fusedCausalMaskSoftmaxKernel(
   const T* __restrict__ input,   // [batch, seqQ, seqKV] - logits from Q@K^T
   T* __restrict__ output,        // [batch, seqQ, seqKV] - softmax output
   T* __restrict__ logitsOut,     // [batch, seqQ, seqKV] - masked logits (optional)
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV,
   const bool isCausal) {

 using AccT = typename FlashAccType<T>::type;
 const LongType row = blockIdx.x;  // which row (batch*seqQ rows)
 if (row >= batch * seqQ) return;

 const LongType queryIdx = row % seqQ;
 const LongType rowStart = row * seqKV;
 const LongType causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;
 const LongType queryPos = queryIdx + causalOffset;

 // Shared memory for warp reductions in accumulator precision.
 extern __shared__ char sharedMem[];
 AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

 // Pass 1: Apply causal mask and find max.
 AccT threadMax = -DataTypeUtils::infOrMax<AccT>();
 const LongType maxKV = isCausal ? min(queryPos + 1, seqKV) : seqKV;

 for (LongType j = threadIdx.x; j < seqKV; j += blockDim.x) {
   AccT val;
   if (isCausal && j > queryPos) {
     val = -DataTypeUtils::infOrMax<AccT>();
   } else {
     val = static_cast<AccT>(input[rowStart + j]);
   }
   // Store masked logits if requested
   if (logitsOut != nullptr) {
     logitsOut[rowStart + j] = static_cast<T>(val);
   }
   threadMax = sd::math::sd_max<AccT>(threadMax, val);
 }

 // Warp reduce max.
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadMax = sd::math::sd_max<AccT>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
 }

 const int lane = threadIdx.x % WARP_SIZE;
 const int wid = threadIdx.x / WARP_SIZE;
 const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

 if (lane == 0) sdata[wid] = threadMax;
 __syncthreads();

 AccT rowMax = -DataTypeUtils::infOrMax<AccT>();
 if (threadIdx.x < numWarps) rowMax = sdata[threadIdx.x];
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   rowMax = sd::math::sd_max<AccT>(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
 }

 __shared__ AccT sharedMax;
 if (threadIdx.x == 0) sharedMax = rowMax;
 __syncthreads();
 rowMax = sharedMax;

 // Pass 2: Compute sum of exp(x - max) — NO output writes.
 // This kernel is called with input == output (in-place). Writing exp values
 // to output here would clobber the original logits that later iterations
 // of the same loop (or other threads) still need to read. Instead, we only
 // accumulate the sum and defer all output writes to Pass 3.
 AccT threadSum = static_cast<AccT>(0);
 for (LongType j = threadIdx.x; j < seqKV; j += blockDim.x) {
   AccT val;
   if (logitsOut != nullptr) {
     val = static_cast<AccT>(logitsOut[rowStart + j]);
   } else if (isCausal && j > queryPos) {
     val = -DataTypeUtils::infOrMax<AccT>();
   } else {
     val = static_cast<AccT>(input[rowStart + j]);
   }
   threadSum += flashExp<AccT>(val - rowMax);
 }

 // Warp reduce sum
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
 }
 if (lane == 0) sdata[wid] = threadSum;
 __syncthreads();

 AccT rowSum = static_cast<AccT>(0);
 if (threadIdx.x < numWarps) rowSum = sdata[threadIdx.x];
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
 }

 __shared__ AccT sharedSum;
 if (threadIdx.x == 0) sharedSum = rowSum;
 __syncthreads();
 AccT invSum = (sharedSum > static_cast<AccT>(0)) ? (static_cast<AccT>(1) / sharedSum) : static_cast<AccT>(0);

 // Pass 3: Compute exp and normalize in one pass, write to output.
 // Safe for in-place (input == output): each thread reads input[j] then writes
 // output[j] at the same index. Threads handle non-overlapping j values
 // (stride = blockDim.x), so no thread reads a location another thread has
 // already written in this pass.
 for (LongType j = threadIdx.x; j < seqKV; j += blockDim.x) {
   AccT val;
   if (logitsOut != nullptr) {
     val = static_cast<AccT>(logitsOut[rowStart + j]);
   } else if (isCausal && j > queryPos) {
     val = -DataTypeUtils::infOrMax<AccT>();
   } else {
     val = static_cast<AccT>(input[rowStart + j]);
   }
   AccT expVal = flashExp<AccT>(val - rowMax);
   output[rowStart + j] = static_cast<T>(expVal * invSum);
 }
}

template <typename T>
static void fusedCausalMaskSoftmaxLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                          const int numWarps, const cudaStream_t* stream,
                                          const void* vInput, void* vOutput, void* vLogitsOut,
                                          LongType batch, LongType seqQ, LongType seqKV, bool isCausal) {
 auto input = reinterpret_cast<const T*>(vInput);
 auto output = reinterpret_cast<T*>(vOutput);
 auto logitsOut = vLogitsOut != nullptr ? reinterpret_cast<T*>(vLogitsOut) : nullptr;
 using AccT = typename FlashAccType<T>::type;
 const size_t sharedMemSize = static_cast<size_t>(numWarps) * sizeof(AccT);
 fusedCausalMaskSoftmaxKernel<T><<<blocksPerGrid, threadsPerBlock, sharedMemSize, *stream>>>(
     input, output, logitsOut, batch, seqQ, seqKV, isCausal);
 DebugHelper::checkGlobalErrorCode("fusedCausalMaskSoftmax failed");
}

BUILD_SINGLE_TEMPLATE(void fusedCausalMaskSoftmaxLauncher,
                     (const int blocksPerGrid, const int threadsPerBlock, const int numWarps,
                      const cudaStream_t* stream, const void* vInput, void* vOutput, void* vLogitsOut,
                      LongType batch, LongType seqQ, LongType seqKV, bool isCausal),
                     SD_FLOAT_TYPES);

// Public interface for fused causal mask + softmax
void fusedCausalMaskSoftmaxCuda(NDArray* input, NDArray* output, NDArray* logitsOut,
                               bool isCausal, LaunchContext* context) {
 auto stream = context->getCudaStream();
 const auto batch = input->sizeAt(0);
 const auto seqQ = input->sizeAt(1);
 const auto seqKV = input->sizeAt(2);

 const LongType numRows = batch * seqQ;
 int threadsPerBlock = 256;
 if (seqKV > 256) threadsPerBlock = 512;
 if (seqKV > 512) threadsPerBlock = 1024;
 if (seqKV < threadsPerBlock) {
   threadsPerBlock = ((seqKV + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
   if (threadsPerBlock < WARP_SIZE) threadsPerBlock = WARP_SIZE;
 }

 int numWarps = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
 // Shared memory size is computed inside fusedCausalMaskSoftmaxLauncher using
 // the accumulator type: double for double inputs, FP32 otherwise.

 if (logitsOut != nullptr) {
   NDArray::prepareSpecialUse({output, logitsOut}, {input});
 } else {
   NDArray::prepareSpecialUse({output}, {input});
 }

 void* logitsPtr = logitsOut != nullptr ? logitsOut->specialBuffer() : nullptr;

 BUILD_SINGLE_SELECTOR(input->dataType(), fusedCausalMaskSoftmaxLauncher,
                       (numRows, threadsPerBlock, numWarps, stream,
                        input->specialBuffer(), output->specialBuffer(), logitsPtr,
                        batch, seqQ, seqKV, isCausal),
                       SD_FLOAT_TYPES);

 if (logitsOut != nullptr) {
   NDArray::registerSpecialUse({output, logitsOut}, {input});
 } else {
   NDArray::registerSpecialUse({output}, {input});
 }
}

//////////////////////////////////////////////////////////////////////////////
// Fused attention kernel for 3D inputs [batch, seqLen, dim]
// Uses online softmax to avoid materializing full attention matrix
// Each block handles one (batch, query_position) pair
// Supports optional additive attention bias for ONNX compatibility
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(512, 1) void fusedAttention3DKernel(
   const T* __restrict__ query,    // [batch, seqQ, dim]
   const T* __restrict__ key,      // [batch, seqKV, dim]
   const T* __restrict__ value,    // [batch, seqKV, dim]
   const T* __restrict__ attnBias, // [batch, seqQ, seqKV] or [batch, 1, seqQ, seqKV] or nullptr
   T* __restrict__ output,         // [batch, seqQ, dim]
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV,
   const LongType dim,
   const double scale,
   const bool isCausal,
   const int biasRank,             // 0=no bias, 3=[batch,seqQ,seqKV], 4=[batch,1,seqQ,seqKV]
   const LongType biasStride0,     // Stride for batch dimension
   const LongType biasStride1,     // Stride for seqQ (or heads) dimension
   const LongType biasStride2) {   // Stride for seqKV dimension

 using AccT = typename FlashAccType<T>::type;

 // Each block handles one query position for one batch
 const LongType batchIdx = blockIdx.y;
 const LongType queryIdx = blockIdx.x;

 if (batchIdx >= batch || queryIdx >= seqQ) return;

 // Keep tile scores and the running output accumulator in AccT; only the
 // boundary tensors remain T-typed.
 extern __shared__ char sharedMem[];
 AccT* sharedScores  = reinterpret_cast<AccT*>(sharedMem);
 AccT* sharedOutput  = sharedScores + TILE_SIZE_KV;
 __shared__ AccT warpMaxesBuf[32];
 __shared__ AccT warpSumsBuf[32];

 // Initialize output accumulator to zero
 for (int d = threadIdx.x; d < dim; d += blockDim.x) {
   sharedOutput[d] = static_cast<AccT>(0);
 }
 __syncthreads();

 // Pointers to current batch
 const T* Q = query + batchIdx * seqQ * dim + queryIdx * dim;
 const T* K = key + batchIdx * seqKV * dim;
 const T* V = value + batchIdx * seqKV * dim;
 T* O = output + batchIdx * seqQ * dim + queryIdx * dim;

 // Pointer to attention bias for this (batch, query) position
 const T* biasRow = nullptr;
 if (attnBias != nullptr && biasRank > 0) {
   // For rank 3: [batch, seqQ, seqKV] -> offset = batch*biasStride0 + queryIdx*biasStride1
   // For rank 4: [batch, 1, seqQ, seqKV] -> offset = batch*biasStride0 + queryIdx*biasStride1
   biasRow = attnBias + batchIdx * biasStride0 + queryIdx * biasStride1;
 }

 // Global max and sum for this query position in accumulator precision.
 __shared__ AccT globalMax;
 __shared__ AccT globalSum;
 __shared__ AccT newMax;
 if (threadIdx.x == 0) {
   globalMax = -DataTypeUtils::infOrMax<AccT>();
   globalSum = static_cast<AccT>(0);
 }
 __syncthreads();

 // Process key/value positions in tiles
 const LongType causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;
 const LongType queryPos = queryIdx + causalOffset;
 const LongType maxKV = isCausal ? min(queryPos + 1, seqKV) : seqKV;

 // Defensive: ensure maxKV is valid
 if (maxKV <= 0 || dim <= 0) return;

 for (LongType kvStart = 0; kvStart < maxKV; kvStart += TILE_SIZE_KV) {
   const LongType kvEnd = min(kvStart + TILE_SIZE_KV, maxKV);
   const int tileSize = static_cast<int>(kvEnd - kvStart);

   // Defensive: ensure tileSize is valid
   if (tileSize <= 0) continue;

   // Step 1: Compute Q @ K^T for this tile + add attention bias.
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     const LongType kvIdx = kvStart + k;
     const T* Krow = K + kvIdx * dim;

     AccT score = static_cast<AccT>(0);
     for (LongType d = 0; d < dim; d++) {
       score += static_cast<AccT>(Q[d]) * static_cast<AccT>(Krow[d]);
     }
     score *= static_cast<AccT>(scale);

     // Add attention bias if present
     if (biasRow != nullptr) {
       score += static_cast<AccT>(biasRow[kvIdx * biasStride2]);
     }

     // Apply causal mask
     if (isCausal && kvIdx > queryPos) {
       score = -DataTypeUtils::infOrMax<AccT>();
     }

     sharedScores[k] = score;
   }
   __syncthreads();

   // Step 2: Find max in this tile (for numerical stability).
   AccT tileMax = -DataTypeUtils::infOrMax<AccT>();
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     tileMax = sd::math::sd_max<AccT>(tileMax, sharedScores[k]);
   }

   // Warp reduce to find max.
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = sd::math::sd_max<AccT>(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   // First thread in each warp writes to shared memory
   if (threadIdx.x % WARP_SIZE == 0) {
     warpMaxesBuf[threadIdx.x / WARP_SIZE] = tileMax;
   }
   __syncthreads();

   // First warp reduces across all warps
   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileMax = warpMaxesBuf[threadIdx.x];
   } else {
     tileMax = -DataTypeUtils::infOrMax<AccT>();
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = sd::math::sd_max<AccT>(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   if (threadIdx.x == 0) {
     newMax = sd::math::sd_max<AccT>(globalMax, tileMax);
   }
   __syncthreads();

   // Step 3: Rescale previous output if max changed
   if (newMax > globalMax) {
     AccT rescale = flashExp<AccT>(globalMax - newMax);
     for (int d = threadIdx.x; d < dim; d += blockDim.x) {
       sharedOutput[d] *= rescale;
     }
     if (threadIdx.x == 0) {
       globalSum *= rescale;
       globalMax = newMax;
     }
   }
   __syncthreads();

   // Step 4: Compute exp(score - max) and accumulate sum.
   AccT tileSum = static_cast<AccT>(0);
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     AccT expScore = flashExp<AccT>(sharedScores[k] - globalMax);
     sharedScores[k] = expScore;
     tileSum += expScore;
   }

   // Reduce sum across threads
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   if (threadIdx.x % WARP_SIZE == 0) {
     warpSumsBuf[threadIdx.x / WARP_SIZE] = tileSum;
   }
   __syncthreads();

   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileSum = warpSumsBuf[threadIdx.x];
   } else {
     tileSum = static_cast<AccT>(0);
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   if (threadIdx.x == 0) {
     globalSum += tileSum;
   }
   __syncthreads();

   // Step 5: Accumulate weighted values: output += exp_scores @ V.
   for (int d = threadIdx.x; d < dim; d += blockDim.x) {
     AccT acc = static_cast<AccT>(0);
     for (int k = 0; k < tileSize; k++) {
       const LongType kvIdx = kvStart + k;
       acc += sharedScores[k] * static_cast<AccT>(V[kvIdx * dim + d]);
     }
     sharedOutput[d] += acc;
   }
   __syncthreads();
 }

 // Step 6: Normalize by sum and write output.
 AccT invSum3d = (globalSum > static_cast<AccT>(0)) ? (static_cast<AccT>(1) / globalSum) : static_cast<AccT>(0);
 for (int d = threadIdx.x; d < dim; d += blockDim.x) {
   O[d] = static_cast<T>(sharedOutput[d] * invSum3d);
 }
}

//////////////////////////////////////////////////////////////////////////////
// Fused attention kernel WITH scores output - for cases where we need
// to return attention logits and/or attention scores
// This version materializes the full attention row for each query position
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedAttentionWithScores3DKernel(
   const T* __restrict__ query,         // [batch, seqQ, dim]
   const T* __restrict__ key,           // [batch, seqKV, dim]
   const T* __restrict__ value,         // [batch, seqKV, dim]
   T* __restrict__ output,              // [batch, seqQ, dim]
   T* __restrict__ attentionLogits,     // [batch, seqQ, seqKV] or nullptr
   T* __restrict__ attentionScores,     // [batch, seqQ, seqKV] or nullptr
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV,
   const LongType dim,
   const double scale,
   const bool isCausal) {

 using AccT = typename FlashAccType<T>::type;

 // Each block handles one query position for one batch
 const LongType batchIdx = blockIdx.y;
 const LongType queryIdx = blockIdx.x;

 if (batchIdx >= batch || queryIdx >= seqQ) return;

 // Pointers to current batch
 const T* Q = query + batchIdx * seqQ * dim + queryIdx * dim;
 const T* K = key + batchIdx * seqKV * dim;
 const T* V = value + batchIdx * seqKV * dim;
 T* O = output + batchIdx * seqQ * dim + queryIdx * dim;
 T* logitsRow = attentionLogits != nullptr ?
                                           attentionLogits + batchIdx * seqQ * seqKV + queryIdx * seqKV : nullptr;
 T* scoresRow = attentionScores != nullptr ?
                                           attentionScores + batchIdx * seqQ * seqKV + queryIdx * seqKV : nullptr;

 // Shared memory for reductions in accumulator precision.
 extern __shared__ char sharedMem[];
 AccT* sharedMax = reinterpret_cast<AccT*>(sharedMem);   // [32] for warp maxes
 AccT* sharedSum = sharedMax + 32;                        // [32] for warp sums

 // Step 1: Compute all logits for this query row and find max.
 AccT threadMax = -DataTypeUtils::infOrMax<AccT>();
 const LongType causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;
 const LongType queryPos = queryIdx + causalOffset;
 const LongType maxKV = isCausal ? min(queryPos + 1, seqKV) : seqKV;

 for (LongType k = threadIdx.x; k < seqKV; k += blockDim.x) {
   AccT score;
   if (k < maxKV) {
     // Compute dot product Q[queryIdx] . K[k].
     const T* Krow = K + k * dim;
     score = static_cast<AccT>(0);
     for (LongType d = 0; d < dim; d++) {
       score += static_cast<AccT>(Q[d]) * static_cast<AccT>(Krow[d]);
     }
     score *= static_cast<AccT>(scale);
   } else {
     // Causal mask: future positions get -inf
     score = -DataTypeUtils::infOrMax<AccT>();
   }

   // Write logits if requested
   if (logitsRow != nullptr) {
     logitsRow[k] = static_cast<T>(score);
   }

   threadMax = sd::math::sd_max<AccT>(threadMax, score);
 }

 // Reduce max across threads.
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadMax = sd::math::sd_max<AccT>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
 }
 if (threadIdx.x % WARP_SIZE == 0) {
   sharedMax[threadIdx.x / WARP_SIZE] = threadMax;
 }
 __syncthreads();

 // First warp reduces across all warps
 AccT globalMax = -DataTypeUtils::infOrMax<AccT>();
 if (threadIdx.x < 32) {
   AccT val = (threadIdx.x < blockDim.x / WARP_SIZE) ? sharedMax[threadIdx.x] : -DataTypeUtils::infOrMax<AccT>();
   for (int offset = 16; offset > 0; offset /= 2) {
     val = sd::math::sd_max<AccT>(val, __shfl_down_sync(0xffffffff, val, offset));
   }
   if (threadIdx.x == 0) {
     sharedMax[0] = val;
   }
 }
 __syncthreads();
 globalMax = sharedMax[0];

 // Step 2: Compute exp(score - max) and sum, also write scores.
 AccT threadSum = static_cast<AccT>(0);
 for (LongType k = threadIdx.x; k < seqKV; k += blockDim.x) {
   AccT score;
   if (logitsRow != nullptr) {
     score = static_cast<AccT>(logitsRow[k]);
   } else if (k < maxKV) {
     // Recompute score if logits not stored
     const T* Krow = K + k * dim;
     score = static_cast<AccT>(0);
     for (LongType d = 0; d < dim; d++) {
       score += static_cast<AccT>(Q[d]) * static_cast<AccT>(Krow[d]);
     }
     score *= static_cast<AccT>(scale);
   } else {
     score = -DataTypeUtils::infOrMax<AccT>();
   }

   AccT expScore = flashExp<AccT>(score - globalMax);
   threadSum += expScore;

   // Temporarily store exp score (will normalize after we have sum)
   if (scoresRow != nullptr) {
     scoresRow[k] = static_cast<T>(expScore);
   }
 }

 // Reduce sum across threads
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
 }
 if (threadIdx.x % WARP_SIZE == 0) {
   sharedSum[threadIdx.x / WARP_SIZE] = threadSum;
 }
 __syncthreads();

 AccT globalSum = static_cast<AccT>(0);
 if (threadIdx.x < 32) {
   AccT val = (threadIdx.x < blockDim.x / WARP_SIZE) ? sharedSum[threadIdx.x] : static_cast<AccT>(0);
   for (int offset = 16; offset > 0; offset /= 2) {
     val += __shfl_down_sync(0xffffffff, val, offset);
   }
   if (threadIdx.x == 0) {
     sharedSum[0] = val;
   }
 }
 __syncthreads();
 globalSum = sharedSum[0];
 AccT invSum = (globalSum > static_cast<AccT>(0)) ? (static_cast<AccT>(1) / globalSum) : static_cast<AccT>(0);

 // Step 3: Normalize scores (write to scoresRow if needed).
 if (scoresRow != nullptr) {
   for (LongType k = threadIdx.x; k < seqKV; k += blockDim.x) {
     scoresRow[k] = static_cast<T>(static_cast<AccT>(scoresRow[k]) * invSum);
   }
 }
 __syncthreads();

 // Step 4: Compute output - each thread handles a subset of output dimensions
 // This avoids atomicAdd contention by having each thread own its dimensions
 for (int d = threadIdx.x; d < dim; d += blockDim.x) {
   AccT acc = static_cast<AccT>(0);
   for (LongType k = 0; k < seqKV; k++) {
     AccT attnWeight;
     if (scoresRow != nullptr) {
       attnWeight = static_cast<AccT>(scoresRow[k]);
     } else if (logitsRow != nullptr) {
       AccT score = static_cast<AccT>(logitsRow[k]);
       attnWeight = flashExp<AccT>(score - globalMax) * invSum;
     } else if (k < maxKV) {
       // Recompute score
       const T* Krow = K + k * dim;
       AccT score = static_cast<AccT>(0);
       for (LongType dd = 0; dd < dim; dd++) {
         score += static_cast<AccT>(Q[dd]) * static_cast<AccT>(Krow[dd]);
       }
       score *= static_cast<AccT>(scale);
       attnWeight = flashExp<AccT>(score - globalMax) * invSum;
     } else {
       attnWeight = static_cast<AccT>(0);
     }
     acc += attnWeight * static_cast<AccT>(V[k * dim + d]);
   }
   O[d] = static_cast<T>(acc);
 }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for 3D fused attention with scores output
//////////////////////////////////////////////////////////////////////////////
template <typename T>
void launchFusedAttention3DWithScores(
   const T* query,
   const T* key,
   const T* value,
   T* output,
   T* attentionLogits,
   T* attentionScores,
   LongType batch,
   LongType seqQ,
   LongType seqKV,
   LongType dim,
   double scale,
   bool isCausal,
   cudaStream_t stream) {

 // Grid: one block per (query_position, batch) pair
 dim3 grid(seqQ, batch);
 dim3 block(256);

 using AccT = typename FlashAccType<T>::type;
 size_t sharedMem = 32 * sizeof(AccT) + 32 * sizeof(AccT);

 fusedAttentionWithScores3DKernel<T><<<grid, block, sharedMem, stream>>>(
     query, key, value, output, attentionLogits, attentionScores,
     batch, seqQ, seqKV, dim, scale, isCausal);

 DebugHelper::checkGlobalErrorCode("fusedAttention3DWithScores failed");
}

//////////////////////////////////////////////////////////////////////////////
// Void*-based launcher wrapper for 3D fused attention with scores output
// (follows applyCausalMaskInPlaceLauncher / fusedGQADecodeLauncher pattern)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void fusedAttention3DWithScoresLauncher(
   const void* vQuery, const void* vKey, const void* vValue,
   void* vOutput, void* vLogits, void* vScores,
   LongType batch, LongType seqQ, LongType seqKV, LongType dim,
   double scale, bool isCausal, cudaStream_t stream) {

 launchFusedAttention3DWithScores<T>(
     reinterpret_cast<const T*>(vQuery),
     reinterpret_cast<const T*>(vKey),
     reinterpret_cast<const T*>(vValue),
     reinterpret_cast<T*>(vOutput),
     reinterpret_cast<T*>(vLogits),
     reinterpret_cast<T*>(vScores),
     batch, seqQ, seqKV, dim, scale, isCausal, stream);
}

BUILD_SINGLE_TEMPLATE(void fusedAttention3DWithScoresLauncher,
                      (const void*, const void*, const void*,
                       void*, void*, void*,
                       LongType, LongType, LongType, LongType,
                       double, bool, cudaStream_t),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////////
// Fused rank-4 GQA attention with materialized logits and scores.
//
// Inputs stay in BSHD layout. Each block owns one (batch, query head,
// query position) row and maps that query head to its shared KV head via
// kvHead = qHead / headsPerKvHead. This removes the Q/K/V permute copies and
// the headsPerKvHead-wide K/V materialization used by the workspace fallback.
//////////////////////////////////////////////////////////////////////////////
struct GQAAttentionStrides4D {
  LongType q[4];
  LongType k[4];
  LongType v[4];
  LongType currentK[4];
  LongType currentV[4];
  LongType o[4];
  LongType logits[4];
  LongType scores[4];
  LongType bias[4];
};

template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedGQAAttentionWithScores4DKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ currentKeyWindow,
    const T* __restrict__ currentValueWindow,
    const LongType* __restrict__ currentKvPosition,
    LongType currentSeq,
    const T* __restrict__ attentionBias,
    T* __restrict__ output,
    T* __restrict__ attentionLogits,
    T* __restrict__ attentionScores,
    LongType batch,
    LongType seqQ,
    LongType seqKV,
    LongType numQHeads,
    LongType numKvHeads,
    LongType headDim,
    LongType headsPerKvHead,
    double scale,
    bool isCausal,
    GQAAttentionStrides4D strides) {
  using AccT = typename FlashAccType<T>::type;

  const LongType queryIdx = blockIdx.x;
  const LongType qHead = blockIdx.y;
  const LongType batchIdx = blockIdx.z;
  if (batchIdx >= batch || qHead >= numQHeads || queryIdx >= seqQ) return;

  const LongType kvHead = qHead / headsPerKvHead;
  if (kvHead >= numKvHeads) return;

  const T* qRow = query
      + batchIdx * strides.q[0]
      + queryIdx * strides.q[1]
      + qHead * strides.q[2];
  const T* kBase = key
      + batchIdx * strides.k[0]
      + kvHead * strides.k[2];
  const T* vBase = value
      + batchIdx * strides.v[0]
      + kvHead * strides.v[2];
  const bool hasCurrentWindow =
      currentKeyWindow != nullptr && currentValueWindow != nullptr
      && currentKvPosition != nullptr && currentSeq > 0;
  const LongType currentStart = hasCurrentWindow ? currentKvPosition[0] : -1;
  const bool validCurrentWindow =
      hasCurrentWindow && currentStart >= 0 && currentStart < seqKV;
  const T* currentKBase = validCurrentWindow
      ? currentKeyWindow
          + batchIdx * strides.currentK[0]
          + kvHead * strides.currentK[2]
      : nullptr;
  const T* currentVBase = validCurrentWindow
      ? currentValueWindow
          + batchIdx * strides.currentV[0]
          + kvHead * strides.currentV[2]
      : nullptr;
  T* outRow = output
      + batchIdx * strides.o[0]
      + queryIdx * strides.o[1]
      + qHead * strides.o[2];
  T* logitsRow = attentionLogits
      + batchIdx * strides.logits[0]
      + qHead * strides.logits[1]
      + queryIdx * strides.logits[2];
  T* scoresRow = attentionScores
      + batchIdx * strides.scores[0]
      + qHead * strides.scores[1]
      + queryIdx * strides.scores[2];

  __shared__ AccT warpMaxes[32];
  __shared__ AccT warpSums[32];
  __shared__ AccT globalMax;
  __shared__ AccT globalSum;

  const LongType causalOffset = seqKV > seqQ ? seqKV - seqQ : 0;
  const LongType queryPosition = validCurrentWindow
      ? currentStart + queryIdx
      : queryIdx + causalOffset;
  const LongType maxKV = isCausal ? min(queryPosition + 1, seqKV) : seqKV;

  AccT threadMax = -DataTypeUtils::infOrMax<AccT>();
  for (LongType kv = threadIdx.x; kv < seqKV; kv += blockDim.x) {
    AccT logit = -DataTypeUtils::infOrMax<AccT>();
    if (kv < maxKV) {
      const LongType currentIndex = kv - currentStart;
      const bool useCurrent =
          validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq;
      const T* kRow = useCurrent
          ? currentKBase + currentIndex * strides.currentK[1]
          : kBase + kv * strides.k[1];
      const LongType kDimStride =
          useCurrent ? strides.currentK[3] : strides.k[3];
      logit = static_cast<AccT>(0);
      for (LongType d = 0; d < headDim; d++) {
        logit += static_cast<AccT>(qRow[d * strides.q[3]])
            * static_cast<AccT>(kRow[d * kDimStride]);
      }
      logit *= static_cast<AccT>(scale);
      if (attentionBias != nullptr) {
        const LongType biasOffset =
            batchIdx * strides.bias[0]
            + qHead * strides.bias[1]
            + queryIdx * strides.bias[2]
            + kv * strides.bias[3];
        logit += static_cast<AccT>(attentionBias[biasOffset]);
      }
    }
    logitsRow[kv * strides.logits[3]] = static_cast<T>(logit);
    threadMax = sd::math::sd_max<AccT>(threadMax, logit);
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    threadMax = sd::math::sd_max<AccT>(
        threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
  }
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int warpCount = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  if (lane == 0) warpMaxes[warp] = threadMax;
  __syncthreads();

  if (warp == 0) {
    AccT blockMax = lane < warpCount
        ? warpMaxes[lane]
        : -DataTypeUtils::infOrMax<AccT>();
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      blockMax = sd::math::sd_max<AccT>(
          blockMax, __shfl_down_sync(0xffffffff, blockMax, offset));
    }
    if (lane == 0) globalMax = blockMax;
  }
  __syncthreads();

  AccT threadSum = static_cast<AccT>(0);
  for (LongType kv = threadIdx.x; kv < seqKV; kv += blockDim.x) {
    const AccT logit = static_cast<AccT>(
        logitsRow[kv * strides.logits[3]]);
    const AccT probability = kv < maxKV
        ? flashExp<AccT>(logit - globalMax)
        : static_cast<AccT>(0);
    scoresRow[kv * strides.scores[3]] = static_cast<T>(probability);
    threadSum += probability;
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
  }
  if (lane == 0) warpSums[warp] = threadSum;
  __syncthreads();

  if (warp == 0) {
    AccT blockSum = lane < warpCount
        ? warpSums[lane]
        : static_cast<AccT>(0);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      blockSum += __shfl_down_sync(0xffffffff, blockSum, offset);
    }
    if (lane == 0) globalSum = blockSum;
  }
  __syncthreads();

  const AccT invSum = globalSum > static_cast<AccT>(0)
      ? static_cast<AccT>(1) / globalSum
      : static_cast<AccT>(0);
  for (LongType kv = threadIdx.x; kv < seqKV; kv += blockDim.x) {
    const LongType scoreOffset = kv * strides.scores[3];
    scoresRow[scoreOffset] = static_cast<T>(
        static_cast<AccT>(scoresRow[scoreOffset]) * invSum);
  }
  __syncthreads();

  for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
    AccT accumulated = static_cast<AccT>(0);
    for (LongType kv = 0; kv < seqKV; kv++) {
      const AccT probability = static_cast<AccT>(
          scoresRow[kv * strides.scores[3]]);
      const LongType currentIndex = kv - currentStart;
      const bool useCurrent =
          validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq;
      const T* vRow = useCurrent
          ? currentVBase + currentIndex * strides.currentV[1]
          : vBase + kv * strides.v[1];
      const LongType vDimStride =
          useCurrent ? strides.currentV[3] : strides.v[3];
      accumulated += probability * static_cast<AccT>(vRow[d * vDimStride]);
    }
    outRow[d * strides.o[3]] = static_cast<T>(accumulated);
  }
}

template <typename T>
static void fusedGQAAttentionWithScores4DLauncher(
    const cudaStream_t* stream,
    const void* query,
    const void* key,
    const void* value,
    const void* currentKeyWindow,
    const void* currentValueWindow,
    const void* currentKvPosition,
    LongType currentSeq,
    const void* attentionBias,
    void* output,
    void* attentionLogits,
    void* attentionScores,
    LongType batch,
    LongType seqQ,
    LongType seqKV,
    LongType numQHeads,
    LongType numKvHeads,
    LongType headDim,
    LongType headsPerKvHead,
    double scale,
    bool isCausal,
    GQAAttentionStrides4D strides) {
  dim3 grid(static_cast<unsigned int>(seqQ),
            static_cast<unsigned int>(numQHeads),
            static_cast<unsigned int>(batch));
  dim3 block(256);
  fusedGQAAttentionWithScores4DKernel<T><<<grid, block, 0, *stream>>>(
      reinterpret_cast<const T*>(query),
      reinterpret_cast<const T*>(key),
      reinterpret_cast<const T*>(value),
      reinterpret_cast<const T*>(currentKeyWindow),
      reinterpret_cast<const T*>(currentValueWindow),
      reinterpret_cast<const LongType*>(currentKvPosition),
      currentSeq,
      reinterpret_cast<const T*>(attentionBias),
      reinterpret_cast<T*>(output),
      reinterpret_cast<T*>(attentionLogits),
      reinterpret_cast<T*>(attentionScores),
      batch, seqQ, seqKV, numQHeads, numKvHeads, headDim,
      headsPerKvHead, scale, isCausal, strides);
  DebugHelper::checkGlobalErrorCode("fusedGQAAttentionWithScores4D failed");
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for 3D fused attention with optional attention bias
//////////////////////////////////////////////////////////////////////////////
template <typename T>
void launchFusedAttention3D(
   const T* query,
   const T* key,
   const T* value,
   const T* attnBias,
   T* output,
   LongType batch,
   LongType seqQ,
   LongType seqKV,
   LongType dim,
   double scale,
   bool isCausal,
   int biasRank,
   LongType biasStride0,
   LongType biasStride1,
   LongType biasStride2,
   cudaStream_t stream) {

 // Grid: one block per (query_position, batch) pair
 dim3 grid(seqQ, batch);

 // Optimize block size based on sequence length and dimension
 // Use larger blocks for better occupancy on modern GPUs
 int blockSize = DEFAULT_BLOCK_SIZE;  // 512 for RTX 4090
 // Cap at 512: fusedAttention3DKernel uses __launch_bounds__(512, 1)
 if (seqKV < 64 && dim < 128) blockSize = 256;  // Smaller blocks for tiny inputs
 dim3 block(blockSize);

 // Dynamic shared memory holds AccT tile scores/output; reduction staging is static
 // shared AccT inside the kernel.
 // [TILE_SIZE_KV * sizeof(AccT)] sharedScores
 // [dim * sizeof(AccT)]          sharedOutput
 using AccT = typename FlashAccType<T>::type;
 size_t sharedMem = (TILE_SIZE_KV + dim) * sizeof(AccT);

 fusedAttention3DKernel<T><<<grid, block, sharedMem, stream>>>(
     query, key, value, attnBias, output,
     batch, seqQ, seqKV, dim,
     scale, isCausal,
     biasRank, biasStride0, biasStride1, biasStride2);

 DebugHelper::checkGlobalErrorCode("fusedAttention3D failed");
}

//////////////////////////////////////////////////////////////////////////////
// Void*-based launcher wrapper for 3D fused attention with bias
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void fusedAttention3DLauncher(
   const void* vQuery, const void* vKey, const void* vValue,
   const void* vAttnBias, void* vOutput,
   LongType batch, LongType seqQ, LongType seqKV, LongType dim,
   double scale, bool isCausal,
   int biasRank, LongType biasStride0, LongType biasStride1, LongType biasStride2,
   cudaStream_t stream) {

 launchFusedAttention3D<T>(
     reinterpret_cast<const T*>(vQuery),
     reinterpret_cast<const T*>(vKey),
     reinterpret_cast<const T*>(vValue),
     reinterpret_cast<const T*>(vAttnBias),
     reinterpret_cast<T*>(vOutput),
     batch, seqQ, seqKV, dim, scale, isCausal,
     biasRank, biasStride0, biasStride1, biasStride2, stream);
}

BUILD_SINGLE_TEMPLATE(void fusedAttention3DLauncher,
                      (const void*, const void*, const void*,
                       const void*, void*,
                       LongType, LongType, LongType, LongType,
                       double, bool,
                       int, LongType, LongType, LongType,
                       cudaStream_t),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////////
// Public interface - called from FlashAttentionHelper
// Supports optional attention bias for ONNX MultiHeadAttention compatibility
//////////////////////////////////////////////////////////////////////////////
void fusedAttentionCuda(
   NDArray* query,
   NDArray* key,
   NDArray* value,
   NDArray* output,
   double scale,
   bool isCausal,
   LaunchContext* context,
   NDArray* attentionBias) {

 auto stream = context->getCudaStream();

 const auto batch = query->sizeAt(0);
 const auto seqQ = query->sizeAt(1);
 const auto seqKV = key->sizeAt(1);
 const auto dim = query->sizeAt(2);

 // Compute bias strides if bias is provided
 int biasRank = 0;
 LongType biasStride0 = 0, biasStride1 = 0, biasStride2 = 0;
 const void* biasPtr = nullptr;

 if (attentionBias != nullptr && !attentionBias->isEmpty()) {
   biasRank = attentionBias->rankOf();

   if (biasRank == 3) {
     // [batch, seqQ, seqKV] - use broadcast-safe strides (0 for size-1 dims)
     biasStride0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride1 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
     biasStride2 = attentionBias->sizeAt(2) > 1 ? attentionBias->strideAt(2) : 0;
   } else if (biasRank == 4) {
     // [batch, numHeads, seqQ, seqKV] — for 3D attention, skip heads dim
     biasStride0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride1 = attentionBias->sizeAt(2) > 1 ? attentionBias->strideAt(2) : 0;
     biasStride2 = attentionBias->sizeAt(3) > 1 ? attentionBias->strideAt(3) : 0;
   }
   // IMPORTANT: prepareSpecialUse BEFORE reading specialBuffer().
   // attentionBias may be host-only when first created (specialBuffer() returns host ptr).
   // prepareSpecialUse calls syncToDevice(), which allocates the device buffer and copies
   // data to it. Reading specialBuffer() AFTER ensures we get the valid device pointer.
   NDArray::prepareSpecialUse({output}, {query, key, value, attentionBias});
   biasPtr = attentionBias->specialBuffer();
 } else {
   NDArray::prepareSpecialUse({output}, {query, key, value});
 }

 BUILD_SINGLE_SELECTOR(query->dataType(), fusedAttention3DLauncher,
                       (query->specialBuffer(), key->specialBuffer(),
                        value->specialBuffer(), biasPtr,
                        output->specialBuffer(),
                        batch, seqQ, seqKV, dim, scale, isCausal,
                        biasRank, biasStride0, biasStride1, biasStride2, *stream),
                       SD_FLOAT_TYPES);

 if (attentionBias != nullptr && !attentionBias->isEmpty()) {
   NDArray::registerSpecialUse({output}, {query, key, value, attentionBias});
 } else {
   NDArray::registerSpecialUse({output}, {query, key, value});
 }
}

//////////////////////////////////////////////////////////////////////////////
// Direct GQA attention kernel — 4D BSHD inputs, tiled online softmax.
// Each block handles one (batch, qHead, queryIdx) tuple.
// K/V are indexed via kvHead = qHead / headsPerKvHead, so multi-row GQA
// avoids both K/V head materialization and Q/K/V permutation round-trips.
// NO atomicAdd — each thread owns output dimensions.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL __launch_bounds__(512, 1) void fusedGQADecodeKernel(
   const T* __restrict__ query,      // [batch, seqQ, numQHeads, headDim] BSHD
   const T* __restrict__ key,        // [batch, seqKV, numKvHeads, headDim] BSHD
   const T* __restrict__ value,      // [batch, seqKV, numKvHeads, headDim] BSHD
   const T* __restrict__ currentKeyWindow,
   const T* __restrict__ currentValueWindow,
   const LongType* __restrict__ currentKvPosition,
   const LongType currentSeq,
   const T* __restrict__ attnBias,   // [batch, numQHeads, seqQ, seqKV] or nullptr
   T* __restrict__ output,           // [batch, seqQ, numQHeads, headDim] BSHD
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV,
   const LongType numQHeads,
   const LongType numKvHeads,
   const LongType headDim,
   const LongType headsPerKvHead,
   const double scale,
   const bool isCausal,
   // Strides for Q [batch, seqQ, numQHeads, headDim]
   const LongType qStride0, const LongType qStride1,
   const LongType qStride2, const LongType qStride3,
   // Strides for K [batch, seqKV, numKvHeads, headDim]
   const LongType kStride0, const LongType kStride1, const LongType kStride2, const LongType kStride3,
   // Strides for V [batch, seqKV, numKvHeads, headDim]
   const LongType vStride0, const LongType vStride1, const LongType vStride2, const LongType vStride3,
   // Strides for the current K/V producer window [batch, currentSeq, numKvHeads, headDim]
   const LongType currentKStride0, const LongType currentKStride1,
   const LongType currentKStride2, const LongType currentKStride3,
   const LongType currentVStride0, const LongType currentVStride1,
   const LongType currentVStride2, const LongType currentVStride3,
   // Strides for output [batch, seqQ, numQHeads, headDim]
   const LongType oStride0, const LongType oStride1,
   const LongType oStride2, const LongType oStride3,
   // Broadcast-safe strides for bias
   const LongType biasStride0,
   const LongType biasStride1,
   const LongType biasStride2,
   const LongType biasStride3) {

 using AccT = typename FlashAccType<T>::type;

 const LongType qHead = blockIdx.x;
 const LongType batchIdx = blockIdx.y;
 const LongType queryIdx = blockIdx.z;
 if (batchIdx >= batch || qHead >= numQHeads || queryIdx >= seqQ) return;

 const LongType kvHead = qHead / headsPerKvHead;
 if (kvHead >= numKvHeads) return;

 // Shared memory layout: scores tile [TILE_SIZE_KV] + output accumulator [headDim]
 // in accumulator precision.
 extern __shared__ char sharedMem[];
 AccT* sharedScores = reinterpret_cast<AccT*>(sharedMem);
 AccT* sharedOutput = sharedScores + TILE_SIZE_KV;

 // Q pointer: query[batchIdx, queryIdx, qHead, :] — stride-based indexing
 const T* Q = query + batchIdx * qStride0 + queryIdx * qStride1 + qHead * qStride2;

 // K/V base: key[batchIdx, :, kvHead, :] — stride-based indexing
 const T* Kbase = key + batchIdx * kStride0 + kvHead * kStride2;
 const T* Vbase = value + batchIdx * vStride0 + kvHead * vStride2;
 const bool hasCurrentWindow =
     currentKeyWindow != nullptr && currentValueWindow != nullptr
     && currentKvPosition != nullptr && currentSeq > 0;
 const LongType currentStart = hasCurrentWindow ? currentKvPosition[0] : -1;
 const bool validCurrentWindow =
     hasCurrentWindow && currentStart >= 0 && currentStart < seqKV;
 const T* currentKBase = validCurrentWindow
     ? currentKeyWindow + batchIdx * currentKStride0 + kvHead * currentKStride2
     : nullptr;
 const T* currentVBase = validCurrentWindow
     ? currentValueWindow + batchIdx * currentVStride0 + kvHead * currentVStride2
     : nullptr;

 // Output: output[batchIdx, queryIdx, qHead, :]
 T* O = output + batchIdx * oStride0 + queryIdx * oStride1 + qHead * oStride2;

 // Bias row: attnBias[batchIdx, qHead, queryIdx, :]
 const T* biasRow = nullptr;
 if (attnBias != nullptr) {
   biasRow = attnBias + batchIdx * biasStride0 + qHead * biasStride1
       + queryIdx * biasStride2;
 }

 // When a cache-form op supplies the current producer window, query rows are
 // anchored at that device-resident cache position. Otherwise retain the
 // right-aligned semantics used by direct non-cache callers.
 const LongType causalOffset = seqKV > seqQ ? seqKV - seqQ : 0;
 const LongType queryPosition = validCurrentWindow
     ? currentStart + queryIdx
     : queryIdx + causalOffset;
 const LongType maxKV = isCausal ? min(queryPosition + 1, seqKV) : seqKV;

 // Online softmax state (block-wide via shared memory)
 __shared__ AccT globalMax;
 __shared__ AccT globalSum;
 if (threadIdx.x == 0) {
   globalMax = -DataTypeUtils::infOrMax<AccT>();
   globalSum = static_cast<AccT>(0);
 }

 // Initialize output accumulator
 for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
   sharedOutput[d] = static_cast<AccT>(0);
 }
 __syncthreads();

 if (headDim <= 0 || seqKV <= 0) return;

 // Tile over KV positions — same structure as fusedAttention3DKernel
 for (LongType kvStart = 0; kvStart < seqKV; kvStart += TILE_SIZE_KV) {
   const LongType kvEnd = min(kvStart + TILE_SIZE_KV, seqKV);
   const int tileSize = static_cast<int>(kvEnd - kvStart);
   if (tileSize <= 0) continue;

   // Step 1: Compute Q @ K^T scores for this tile + add bias.
   // Positions beyond this query row's causal boundary remain -inf.
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     const LongType kvIdx = kvStart + k;
     AccT score = -DataTypeUtils::infOrMax<AccT>();
     if (kvIdx < maxKV) {
       const LongType currentIndex = kvIdx - currentStart;
       const bool useCurrent =
           validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq;
       const T* Krow = useCurrent
           ? currentKBase + currentIndex * currentKStride1
           : Kbase + kvIdx * kStride1;
       const LongType kDimStride = useCurrent ? currentKStride3 : kStride3;
       score = static_cast<AccT>(0);
       for (LongType d = 0; d < headDim; d++) {
         score += static_cast<AccT>(Q[d * qStride3])
             * static_cast<AccT>(Krow[d * kDimStride]);
       }
       score *= static_cast<AccT>(scale);

       if (biasRow != nullptr) {
         score += static_cast<AccT>(biasRow[kvIdx * biasStride3]);
       }
     }

     sharedScores[k] = score;
   }
   __syncthreads();

   // Step 2: Find max in this tile
   AccT tileMax = -DataTypeUtils::infOrMax<AccT>();
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     tileMax = sd::math::sd_max<AccT>(tileMax, sharedScores[k]);
   }

   // Warp reduce max
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = sd::math::sd_max<AccT>(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   __shared__ AccT warpMaxes[32];
   if (threadIdx.x % WARP_SIZE == 0) {
     warpMaxes[threadIdx.x / WARP_SIZE] = tileMax;
   }
   __syncthreads();

   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileMax = warpMaxes[threadIdx.x];
   } else {
     tileMax = -DataTypeUtils::infOrMax<AccT>();
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = sd::math::sd_max<AccT>(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   __shared__ AccT newMax;
   if (threadIdx.x == 0) {
     newMax = sd::math::sd_max<AccT>(globalMax, tileMax);
   }
   __syncthreads();

   // Step 3: Rescale previous output accumulator if max changed
   if (newMax > globalMax) {
     AccT rescale = flashExp<AccT>(globalMax - newMax);
     for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
       sharedOutput[d] *= rescale;
     }
     if (threadIdx.x == 0) {
       globalSum *= rescale;
       globalMax = newMax;
     }
   }
   __syncthreads();

   // Step 4: Compute exp(score - max) and accumulate sum
   AccT tileSum = static_cast<AccT>(0);
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     AccT expScore = flashExp<AccT>(sharedScores[k] - globalMax);
     sharedScores[k] = expScore;
     tileSum += expScore;
   }

   // Reduce sum across threads
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   __shared__ AccT warpSums[32];
   if (threadIdx.x % WARP_SIZE == 0) {
     warpSums[threadIdx.x / WARP_SIZE] = tileSum;
   }
   __syncthreads();

   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileSum = warpSums[threadIdx.x];
   } else {
     tileSum = static_cast<AccT>(0);
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   if (threadIdx.x == 0) {
     globalSum += tileSum;
   }
   __syncthreads();

   // Step 5: Accumulate weighted V — each thread owns a subset of output dims
   for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
     AccT acc = static_cast<AccT>(0);
     for (int k = 0; k < tileSize; k++) {
       const LongType kvIdx = kvStart + k;
       const LongType currentIndex = kvIdx - currentStart;
       const bool useCurrent =
           validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq;
       const T* Vrow = useCurrent
           ? currentVBase + currentIndex * currentVStride1
           : Vbase + kvIdx * vStride1;
       const LongType vDimStride = useCurrent ? currentVStride3 : vStride3;
       acc += sharedScores[k] * static_cast<AccT>(Vrow[d * vDimStride]);
     }
     sharedOutput[d] += acc;
   }
   __syncthreads();
 }

 // Step 6: Normalize by sum and write output
 // Guard against globalSum == 0 (all positions masked → exp sums to 0).
 // Output zeros when nothing is attended to, matching PyTorch behavior.
 AccT invSum = (globalSum > static_cast<AccT>(0)) ? (static_cast<AccT>(1) / globalSum) : static_cast<AccT>(0);
 for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
   O[d * oStride3] = static_cast<T>(sharedOutput[d] * invSum);
 }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for fused GQA decode — uses void* params + BUILD_SINGLE_SELECTOR
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void fusedGQADecodeLauncher(
   const int blocksPerGrid, const int threadsPerBlock,
   const int sharedMem, const cudaStream_t* stream,
   const void* vQuery, const void* vKey, const void* vValue,
   const void* vCurrentKeyWindow, const void* vCurrentValueWindow,
   const void* vCurrentKvPosition, LongType currentSeq,
   const void* vAttnBias, void* vOutput,
   LongType batch, LongType seqQ, LongType seqKV,
   LongType numQHeads, LongType numKvHeads,
   LongType headDim, LongType headsPerKvHead, double scale, bool isCausal,
   LongType qStride0, LongType qStride1, LongType qStride2, LongType qStride3,
   LongType kStride0, LongType kStride1, LongType kStride2, LongType kStride3,
   LongType vStride0, LongType vStride1, LongType vStride2, LongType vStride3,
   LongType currentKStride0, LongType currentKStride1,
   LongType currentKStride2, LongType currentKStride3,
   LongType currentVStride0, LongType currentVStride1,
   LongType currentVStride2, LongType currentVStride3,
   LongType oStride0, LongType oStride1, LongType oStride2, LongType oStride3,
   LongType biasStride0, LongType biasStride1,
   LongType biasStride2, LongType biasStride3) {

 auto query = reinterpret_cast<const T*>(vQuery);
 auto key = reinterpret_cast<const T*>(vKey);
 auto value = reinterpret_cast<const T*>(vValue);
 auto currentKeyWindow = vCurrentKeyWindow != nullptr
     ? reinterpret_cast<const T*>(vCurrentKeyWindow) : nullptr;
 auto currentValueWindow = vCurrentValueWindow != nullptr
     ? reinterpret_cast<const T*>(vCurrentValueWindow) : nullptr;
 auto currentKvPosition = vCurrentKvPosition != nullptr
     ? reinterpret_cast<const LongType*>(vCurrentKvPosition) : nullptr;
 auto attnBias = vAttnBias != nullptr ? reinterpret_cast<const T*>(vAttnBias) : nullptr;
 auto output = reinterpret_cast<T*>(vOutput);

 // Grid: one block per (qHead, batch, queryIdx) tuple.
 dim3 grid(static_cast<unsigned int>(numQHeads),
           static_cast<unsigned int>(batch),
           static_cast<unsigned int>(seqQ));
 dim3 block(threadsPerBlock);

 using AccT = typename FlashAccType<T>::type;
 size_t smem = sharedMem > 0
     ? static_cast<size_t>(sharedMem)
     : static_cast<size_t>(TILE_SIZE_KV + headDim) * sizeof(AccT);

 fusedGQADecodeKernel<T><<<grid, block, smem, *stream>>>(
     query, key, value,
     currentKeyWindow, currentValueWindow, currentKvPosition, currentSeq,
     attnBias, output,
     batch, seqQ, seqKV, numQHeads, numKvHeads, headDim,
     headsPerKvHead, scale, isCausal,
     qStride0, qStride1, qStride2, qStride3,
     kStride0, kStride1, kStride2, kStride3,
     vStride0, vStride1, vStride2, vStride3,
     currentKStride0, currentKStride1, currentKStride2, currentKStride3,
     currentVStride0, currentVStride1, currentVStride2, currentVStride3,
     oStride0, oStride1, oStride2, oStride3,
     biasStride0, biasStride1, biasStride2, biasStride3);
 DebugHelper::checkGlobalErrorCode("fusedGQADecode failed");
}

//////////////////////////////////////////////////////////////////////////////
// Public interface for fused GQA decode attention
//////////////////////////////////////////////////////////////////////////////
void fusedGQADecodeCuda(
   NDArray* query, NDArray* key, NDArray* value,
   NDArray* output, double scale, bool isCausal,
   LaunchContext* context, NDArray* attentionBias,
   NDArray* currentKeyWindow, NDArray* currentValueWindow,
   const void* currentKvPosition) {

 auto stream = context->getCudaStream();

 // Input layout: BSHD — [batch, seq, heads, dim]
 const auto batch = query->sizeAt(0);
 const auto seqQ = query->sizeAt(1);
 const auto numQHeads = query->sizeAt(2);
 const auto headDim = query->sizeAt(3);
 const auto seqKV = key->sizeAt(1);
 const auto numKvHeads = key->sizeAt(2);
 const auto headsPerKvHead = numQHeads / numKvHeads;
 const bool useCurrentWindow =
     currentKeyWindow != nullptr && currentValueWindow != nullptr
     && currentKvPosition != nullptr;
 const LongType currentSeq =
     useCurrentWindow ? currentKeyWindow->sizeAt(1) : 0;

 // Extract actual strides — kernel uses stride-based indexing so it works
 // correctly with non-contiguous views (e.g. BHSD→BSHD permuted arrays
 // from DSP pre-allocation or KV concat in onnx_mha.cpp).
 const LongType qStride0 = query->strideAt(0);
 const LongType qStride1 = query->strideAt(1);
 const LongType qStride2 = query->strideAt(2);
 const LongType qStride3 = query->strideAt(3);

 const LongType kStride0 = key->strideAt(0);
 const LongType kStride1 = key->strideAt(1);
 const LongType kStride2 = key->strideAt(2);
 const LongType kStride3 = key->strideAt(3);

 const LongType vStride0 = value->strideAt(0);
 const LongType vStride1 = value->strideAt(1);
 const LongType vStride2 = value->strideAt(2);
 const LongType vStride3 = value->strideAt(3);

 LongType currentKStride0 = 0, currentKStride1 = 0;
 LongType currentKStride2 = 0, currentKStride3 = 0;
 LongType currentVStride0 = 0, currentVStride1 = 0;
 LongType currentVStride2 = 0, currentVStride3 = 0;
 if (useCurrentWindow) {
   currentKStride0 = currentKeyWindow->strideAt(0);
   currentKStride1 = currentKeyWindow->strideAt(1);
   currentKStride2 = currentKeyWindow->strideAt(2);
   currentKStride3 = currentKeyWindow->strideAt(3);
   currentVStride0 = currentValueWindow->strideAt(0);
   currentVStride1 = currentValueWindow->strideAt(1);
   currentVStride2 = currentValueWindow->strideAt(2);
   currentVStride3 = currentValueWindow->strideAt(3);
 }

 const LongType oStride0 = output->strideAt(0);
 const LongType oStride1 = output->strideAt(1);
 const LongType oStride2 = output->strideAt(2);
 const LongType oStride3 = output->strideAt(3);

 LongType biasStride0 = 0, biasStride1 = 0, biasStride2 = 0, biasStride3 = 0;
 const void* biasPtr = nullptr;
 std::vector<NDArray*> inputs = {query, key, value};
 if (useCurrentWindow) {
   inputs.push_back(currentKeyWindow);
   inputs.push_back(currentValueWindow);
 }

 if (attentionBias != nullptr && !attentionBias->isEmpty()) {
   inputs.push_back(attentionBias);
   biasPtr = attentionBias->specialBuffer();
   // Normalize rank-2/3/4 masks to logical [batch, head, query, key]
   // broadcast-safe strides. Dimensions of size one intentionally use stride zero.
   const int biasRank = attentionBias->rankOf();
   if (biasRank == 4) {
     biasStride0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride1 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
     biasStride2 = attentionBias->sizeAt(2) > 1 ? attentionBias->strideAt(2) : 0;
     biasStride3 = attentionBias->sizeAt(3) > 1 ? attentionBias->strideAt(3) : 0;
   } else if (biasRank == 3) {
     biasStride0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride1 = 0;
     biasStride2 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
     biasStride3 = attentionBias->sizeAt(2) > 1 ? attentionBias->strideAt(2) : 0;
   } else {
     biasStride0 = 0;
     biasStride1 = 0;
     biasStride2 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride3 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
   }
 }
 NDArray::prepareSpecialUse({output}, inputs);

 // Centralized launch dimensions computation
 int dtypeSize = query->sizeOfT();
 dim3 launchDims = getFusedGQADecodeDims(numQHeads, batch, seqKV, headDim, dtypeSize);

 BUILD_SINGLE_SELECTOR(query->dataType(), fusedGQADecodeLauncher,
                       (launchDims.x, launchDims.y, launchDims.z, stream,
                        query->specialBuffer(), key->specialBuffer(),
                        value->specialBuffer(),
                        useCurrentWindow ? currentKeyWindow->specialBuffer() : nullptr,
                        useCurrentWindow ? currentValueWindow->specialBuffer() : nullptr,
                        useCurrentWindow ? currentKvPosition : nullptr,
                        currentSeq, biasPtr, output->specialBuffer(),
                        batch, seqQ, seqKV, numQHeads, numKvHeads,
                        headDim, headsPerKvHead, scale, isCausal,
                        qStride0, qStride1, qStride2, qStride3,
                        kStride0, kStride1, kStride2, kStride3,
                        vStride0, vStride1, vStride2, vStride3,
                        currentKStride0, currentKStride1,
                        currentKStride2, currentKStride3,
                        currentVStride0, currentVStride1,
                        currentVStride2, currentVStride3,
                        oStride0, oStride1, oStride2, oStride3,
                        biasStride0, biasStride1, biasStride2, biasStride3),
                       SD_FLOAT_TYPES);

 NDArray::registerSpecialUse({output}, inputs);
}

//////////////////////////////////////////////////////////////////////////////
// V2: fusedGQADecodeQuantisedKernel
//
// GQA decode attention with inline INT8 K/V dequantisation.
// One block per (qHead, batch) pair — same grid as fusedGQADecodeKernel.
// Inner loop: kval = float(keyQ[idx]) * keyScale[head_pos_idx]
//             vval = float(valQ[idx]) * valScale[head_pos_idx]
//
// Accepts the ADR-0106 substrate mask (attentionBias) with shape
// [B, 1_or_qH, 1, seqKV] (broadcast-safe via zero strides for size-1 dims).
//////////////////////////////////////////////////////////////////////////////
SD_KERNEL __launch_bounds__(512, 1) void fusedGQADecodeQuantisedKernel(
    const float* __restrict__ query,       // [batch, 1, numQHeads, headDim]
    const int8_t* __restrict__ keyQ,       // [batch, seqKV, numKvHeads, headDim]
    const float*  __restrict__ keyScale,   // [batch, seqKV, numKvHeads]
    const int8_t* __restrict__ valQ,       // [batch, seqKV, numKvHeads, headDim]
    const float*  __restrict__ valScale,   // [batch, seqKV, numKvHeads]
    const float*  __restrict__ attnBias,   // [batch, numQHeads, 1, seqKV] or nullptr
    float* __restrict__ output,            // [batch, 1, numQHeads, headDim]
    const LongType batch,
    const LongType seqKV,
    const LongType numQHeads,
    const LongType numKvHeads,
    const LongType headDim,
    const LongType headsPerKvHead,
    const double scale,
    // Q strides [batch, 1, numQHeads, headDim]
    const LongType qStride0, const LongType qStride2, const LongType qStride3,
    // K/V int8 cache strides [batch, seqKV, numKvHeads, headDim] — assumed contiguous
    const LongType kvS0, const LongType kvS1, const LongType kvS2,
    // Scale strides [batch, seqKV, numKvHeads] — assumed contiguous
    const LongType ksS0, const LongType ksS1,
    // Output strides [batch, 1, numQHeads, headDim]
    const LongType oStride0, const LongType oStride2, const LongType oStride3,
    // Bias strides (broadcast-safe)
    const LongType biasStride0, const LongType biasStride1,
    const LongType biasStride2, const LongType biasStride3) {

    const LongType qHead    = blockIdx.x;
    const LongType batchIdx = blockIdx.y;
    if (batchIdx >= batch || qHead >= numQHeads) return;

    const LongType kvHead = qHead / headsPerKvHead;

    // Shared memory: scores tile [TILE_SIZE_KV] + output accumulator [headDim]
    extern __shared__ char sharedMemQ2[];
    float* sharedScores = reinterpret_cast<float*>(sharedMemQ2);
    float* sharedOutput = sharedScores + TILE_SIZE_KV;

    // Q pointer
    const float* Q = query + batchIdx * qStride0 + qHead * qStride2;

    // K/V base pointers for this batch × kvHead
    const int8_t* Kbase = keyQ   + batchIdx * kvS0 + kvHead * kvS2;
    const float*  KsBase= keyScale+ batchIdx * ksS0;          // seqKV × kvHeads; kvHead added per-token
    const int8_t* Vbase = valQ   + batchIdx * kvS0 + kvHead * kvS2;
    const float*  VsBase= valScale+ batchIdx * ksS0;

    // Output
    float* O = output + batchIdx * oStride0 + qHead * oStride2;

    // Bias row
    const float* biasRow = nullptr;
    if (attnBias != nullptr) {
        biasRow = attnBias + batchIdx * biasStride0 + qHead * biasStride1;
    }

    // Online softmax state
    __shared__ float globalMax;
    __shared__ float globalSum;
    if (threadIdx.x == 0) {
        globalMax = -DataTypeUtils::infOrMax<float>();
        globalSum = 0.0f;
    }

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        sharedOutput[d] = 0.0f;
    }
    __syncthreads();

    if (headDim <= 0 || seqKV <= 0) return;

    for (LongType kvStart = 0; kvStart < seqKV; kvStart += TILE_SIZE_KV) {
        const LongType kvEnd  = min(kvStart + TILE_SIZE_KV, seqKV);
        const int tileSize = static_cast<int>(kvEnd - kvStart);
        if (tileSize <= 0) continue;

        // Step 1: Q @ K^T scores (inline dequant)
        // ADR 0107 V2 ROW-INLINE: null keyScale → the cache last dim is headDim+4 and each row's
        // float32 scale sits at Krow+headDim (inside the logical tensor — staging-proof).
        for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
            const LongType kvIdx = kvStart + k;
            const int8_t* Krow  = Kbase + kvIdx * kvS1;
            const float   ksc   = keyScale != nullptr
                ? KsBase[kvIdx * ksS1 + kvHead]
                : *reinterpret_cast<const float*>(Krow + headDim);

            float score = 0.0f;
            for (LongType d = 0; d < headDim; d++) {
                float kval = static_cast<float>(Krow[d]) * ksc;
                score += Q[d * qStride3] * kval;
            }
            score *= static_cast<float>(scale);
            if (biasRow != nullptr) {
                score += biasRow[kvIdx * biasStride3];
            }
            sharedScores[k] = score;
        }
        __syncthreads();

        // Step 2: tile max
        float tileMax = -DataTypeUtils::infOrMax<float>();
        for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
            tileMax = sd::math::sd_max<float>(tileMax, sharedScores[k]);
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            tileMax = sd::math::sd_max<float>(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
        }
        __shared__ float warpMaxes[32];
        if (threadIdx.x % WARP_SIZE == 0) warpMaxes[threadIdx.x / WARP_SIZE] = tileMax;
        __syncthreads();
        if (threadIdx.x < blockDim.x / WARP_SIZE) tileMax = warpMaxes[threadIdx.x];
        else tileMax = -DataTypeUtils::infOrMax<float>();
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            tileMax = sd::math::sd_max<float>(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
        }

        __shared__ float newMax;
        if (threadIdx.x == 0) newMax = sd::math::sd_max<float>(globalMax, tileMax);
        __syncthreads();

        // Step 3: rescale previous accumulator
        if (newMax > globalMax) {
            float rescale = flashExp<float>(globalMax - newMax);
            for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
                sharedOutput[d] *= rescale;
            }
            if (threadIdx.x == 0) {
                globalSum *= rescale;
                globalMax = newMax;
            }
        }
        __syncthreads();

        // Step 4: softmax weights
        float tileSum = 0.0f;
        for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
            float expScore = flashExp<float>(sharedScores[k] - globalMax);
            sharedScores[k] = expScore;
            tileSum += expScore;
        }
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
        }
        __shared__ float warpSums[32];
        if (threadIdx.x % WARP_SIZE == 0) warpSums[threadIdx.x / WARP_SIZE] = tileSum;
        __syncthreads();
        if (threadIdx.x < blockDim.x / WARP_SIZE) tileSum = warpSums[threadIdx.x];
        else tileSum = 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
        }
        if (threadIdx.x == 0) globalSum += tileSum;
        __syncthreads();

        // Step 5: weighted V accumulation (inline dequant; row-inline scale when valScale null)
        for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
            float acc = 0.0f;
            for (int k = 0; k < tileSize; k++) {
                const LongType kvIdx = kvStart + k;
                const int8_t* Vrow = Vbase + kvIdx * kvS1;
                const float   vsc  = valScale != nullptr
                    ? VsBase[kvIdx * ksS1 + kvHead]
                    : *reinterpret_cast<const float*>(Vrow + headDim);
                float vval = static_cast<float>(Vrow[d]) * vsc;
                acc += sharedScores[k] * vval;
            }
            sharedOutput[d] += acc;
        }
        __syncthreads();
    }

    // Step 6: normalize and write
    float invSum = (globalSum > 0.0f) ? (1.0f / globalSum) : 0.0f;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        O[d * oStride3] = sharedOutput[d] * invSum;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for fusedGQADecodeQuantised
//////////////////////////////////////////////////////////////////////////////
static void fusedGQADecodeQuantisedLauncher(
    const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
    const cudaStream_t* stream,
    const void* vQuery,
    const void* vKeyQ, const void* vKeyScale,
    const void* vValQ, const void* vValScale,
    const void* vAttnBias,
    void* vOutput,
    LongType batch, LongType seqKV, LongType numQHeads, LongType numKvHeads,
    LongType headDim, LongType headsPerKvHead, double scale,
    LongType qStride0, LongType qStride2, LongType qStride3,
    LongType kvS0, LongType kvS1, LongType kvS2,
    LongType ksS0, LongType ksS1,
    LongType oStride0, LongType oStride2, LongType oStride3,
    LongType biasStride0, LongType biasStride1,
    LongType biasStride2, LongType biasStride3) {

    auto query    = reinterpret_cast<const float*>(vQuery);
    auto keyQ     = reinterpret_cast<const int8_t*>(vKeyQ);
    auto keyScale = reinterpret_cast<const float*>(vKeyScale);
    auto valQ     = reinterpret_cast<const int8_t*>(vValQ);
    auto valScale = reinterpret_cast<const float*>(vValScale);
    auto attnBias = vAttnBias != nullptr ? reinterpret_cast<const float*>(vAttnBias) : nullptr;
    auto output   = reinterpret_cast<float*>(vOutput);

    dim3 grid(numQHeads, batch);
    dim3 block(threadsPerBlock);

    size_t smem = sharedMem > 0
        ? static_cast<size_t>(sharedMem)
        : static_cast<size_t>(TILE_SIZE_KV + headDim) * sizeof(float);

    fusedGQADecodeQuantisedKernel<<<grid, block, smem, *stream>>>(
        query,
        keyQ, keyScale, valQ, valScale,
        attnBias, output,
        batch, seqKV, numQHeads, numKvHeads,
        headDim, headsPerKvHead, scale,
        qStride0, qStride2, qStride3,
        kvS0, kvS1, kvS2,
        ksS0, ksS1,
        oStride0, oStride2, oStride3,
        biasStride0, biasStride1, biasStride2, biasStride3);
    DebugHelper::checkGlobalErrorCode("fusedGQADecodeQuantised failed");
}

//////////////////////////////////////////////////////////////////////////////
// Public interface: fusedGQADecodeQuantisedCuda
//////////////////////////////////////////////////////////////////////////////
void fusedGQADecodeQuantisedCuda(
    NDArray* query,
    NDArray* quantKeyCache,
    NDArray* keyScaleCache,
    NDArray* quantValCache,
    NDArray* valScaleCache,
    NDArray* output,
    double scale,
    LaunchContext* context,
    NDArray* attentionBias) {

    auto stream = context->getCudaStream();

    const auto batch        = query->sizeAt(0);
    const auto numQHeads    = query->sizeAt(2);
    const auto headDim      = query->sizeAt(3);
    const auto seqKV        = quantKeyCache->sizeAt(1);
    const auto numKvHeads   = quantKeyCache->sizeAt(2);
    const auto headsPerKvH  = numQHeads / numKvHeads;

    // Q strides
    const LongType qStride0 = query->strideAt(0);
    const LongType qStride2 = query->strideAt(2);
    const LongType qStride3 = query->strideAt(3);

    // INT8 K/V cache strides [batch, seqKV, kvHeads, headDim] — typically contiguous
    const LongType kvS0 = quantKeyCache->strideAt(0);
    const LongType kvS1 = quantKeyCache->strideAt(1);
    const LongType kvS2 = quantKeyCache->strideAt(2);

    // ADR 0107 V2 ROW-INLINE: when the scale caches are null the INT8 caches are row-inline
    // tensors [batch, seqKV, kvHeads, headDim+4] — each row carries its own float32 scale at
    // row+headDim, INSIDE the logical tensor (survives DSP ext-input staging by construction).
    // The kernel derives the per-row scale from the row pointer when its scale pointer is null.
    const bool inlineKeyScale = (keyScaleCache == nullptr);
    const bool inlineValScale = (valScaleCache == nullptr);
    if (inlineKeyScale && quantKeyCache->sizeAt(3) != headDim + 4) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: row-inline key cache last dim must equal headDim+4");
    }
    if (inlineValScale && quantValCache->sizeAt(3) != headDim + 4) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: row-inline value cache last dim must equal headDim+4");
    }
    const void* keyScalePtr = inlineKeyScale ? nullptr : keyScaleCache->specialBuffer();
    const void* valScalePtr = inlineValScale ? nullptr : valScaleCache->specialBuffer();

    // Scale strides [batch, seqKV, kvHeads] — unused (0) in row-inline mode.
    const LongType ksS0 = inlineKeyScale ? 0 : keyScaleCache->strideAt(0);
    const LongType ksS1 = inlineKeyScale ? 0 : keyScaleCache->strideAt(1);

    // Output strides
    const LongType oStride0 = output->strideAt(0);
    const LongType oStride2 = output->strideAt(2);
    const LongType oStride3 = output->strideAt(3);

    LongType biasStride0 = 0, biasStride1 = 0, biasStride2 = 0, biasStride3 = 0;
    const void* biasPtr = nullptr;

    // Build the input special-use list; the inline scale tail is covered by the cache buffers, so
    // only include the scale NDArrays when they are genuinely separate allocations.
    std::vector<NDArray*> inArrs = {query, quantKeyCache, quantValCache};
    if (!inlineKeyScale) inArrs.push_back(keyScaleCache);
    if (!inlineValScale) inArrs.push_back(valScaleCache);
    if (attentionBias != nullptr && !attentionBias->isEmpty()) {
        inArrs.push_back(attentionBias);
        NDArray::prepareSpecialUse({output}, inArrs);
        biasPtr = attentionBias->specialBuffer();
        biasStride0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
        biasStride1 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
        biasStride2 = attentionBias->sizeAt(2) > 1 ? attentionBias->strideAt(2) : 0;
        biasStride3 = attentionBias->sizeAt(3) > 1 ? attentionBias->strideAt(3) : 0;
    } else {
        NDArray::prepareSpecialUse({output}, inArrs);
    }

    // Reuse the existing dims function — same grid structure as float GQA decode
    int dtypeSize = static_cast<int>(sizeof(float));  // output is always float
    dim3 launchDims = getFusedGQADecodeDims(
        static_cast<int>(numQHeads), static_cast<int>(batch),
        static_cast<int>(seqKV), static_cast<int>(headDim), dtypeSize);

    fusedGQADecodeQuantisedLauncher(
        launchDims.x, launchDims.y, launchDims.z, stream,
        query->specialBuffer(),
        quantKeyCache->specialBuffer(), keyScalePtr,
        quantValCache->specialBuffer(), valScalePtr,
        biasPtr, output->specialBuffer(),
        batch, seqKV, numQHeads, numKvHeads,
        headDim, headsPerKvH, scale,
        qStride0, qStride2, qStride3,
        kvS0, kvS1, kvS2,
        ksS0, ksS1,
        oStride0, oStride2, oStride3,
        biasStride0, biasStride1, biasStride2, biasStride3);

    // inArrs already excludes null (inline) scale caches and includes attentionBias when present.
    NDArray::registerSpecialUse({output}, inArrs);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface for direct rank-4 GQA attention with scores and logits.
//////////////////////////////////////////////////////////////////////////////
void fusedGQAAttentionCudaWithScores(
    NDArray* query,
    NDArray* key,
    NDArray* value,
    NDArray* output,
    NDArray* attentionLogits,
    NDArray* attentionScores,
    double scale,
    bool isCausal,
    LaunchContext* context,
    NDArray* attentionBias,
    NDArray* currentKeyWindow,
    NDArray* currentValueWindow,
    const void* currentKvPosition) {
  auto stream = context->getCudaStream();

  const LongType batch = query->sizeAt(0);
  const LongType seqQ = query->sizeAt(1);
  const LongType numQHeads = query->sizeAt(2);
  const LongType headDim = query->sizeAt(3);
  const LongType seqKV = key->sizeAt(1);
  const LongType numKvHeads = key->sizeAt(2);
  const LongType headsPerKvHead = numQHeads / numKvHeads;
  const bool useCurrentWindow =
      currentKeyWindow != nullptr && currentValueWindow != nullptr
      && currentKvPosition != nullptr;
  const LongType currentSeq =
      useCurrentWindow ? currentKeyWindow->sizeAt(1) : 0;

  GQAAttentionStrides4D strides{};
  for (int i = 0; i < 4; i++) {
    strides.q[i] = query->strideAt(i);
    strides.k[i] = key->strideAt(i);
    strides.v[i] = value->strideAt(i);
    strides.o[i] = output->strideAt(i);
    strides.logits[i] = attentionLogits->strideAt(i);
    strides.scores[i] = attentionScores->strideAt(i);
    if (useCurrentWindow) {
      strides.currentK[i] = currentKeyWindow->strideAt(i);
      strides.currentV[i] = currentValueWindow->strideAt(i);
    }
  }

  const void* biasPtr = nullptr;
  std::vector<NDArray*> inputs = {query, key, value};
  if (useCurrentWindow) {
    inputs.push_back(currentKeyWindow);
    inputs.push_back(currentValueWindow);
  }
  if (attentionBias != nullptr && !attentionBias->isEmpty()) {
    biasPtr = attentionBias->specialBuffer();
    inputs.push_back(attentionBias);
    const int biasRank = attentionBias->rankOf();
    if (biasRank == 4) {
      for (int i = 0; i < 4; i++) {
        strides.bias[i] = attentionBias->sizeAt(i) > 1
            ? attentionBias->strideAt(i)
            : 0;
      }
    } else if (biasRank == 3) {
      strides.bias[0] = attentionBias->sizeAt(0) > 1
          ? attentionBias->strideAt(0)
          : 0;
      strides.bias[1] = 0;
      strides.bias[2] = attentionBias->sizeAt(1) > 1
          ? attentionBias->strideAt(1)
          : 0;
      strides.bias[3] = attentionBias->sizeAt(2) > 1
          ? attentionBias->strideAt(2)
          : 0;
    } else {
      strides.bias[0] = 0;
      strides.bias[1] = 0;
      strides.bias[2] = attentionBias->sizeAt(0) > 1
          ? attentionBias->strideAt(0)
          : 0;
      strides.bias[3] = attentionBias->sizeAt(1) > 1
          ? attentionBias->strideAt(1)
          : 0;
    }
  }

  std::vector<NDArray*> outputs = {
      output, attentionLogits, attentionScores};
  NDArray::prepareSpecialUse(outputs, inputs);

  BUILD_SINGLE_SELECTOR(
      query->dataType(), fusedGQAAttentionWithScores4DLauncher,
      (stream,
       query->specialBuffer(),
       key->specialBuffer(),
       value->specialBuffer(),
       useCurrentWindow ? currentKeyWindow->specialBuffer() : nullptr,
       useCurrentWindow ? currentValueWindow->specialBuffer() : nullptr,
       useCurrentWindow ? currentKvPosition : nullptr,
       currentSeq,
       biasPtr,
       output->specialBuffer(),
       attentionLogits->specialBuffer(),
       attentionScores->specialBuffer(),
       batch, seqQ, seqKV, numQHeads, numKvHeads, headDim,
       headsPerKvHead, scale, isCausal, strides),
      SD_FLOAT_TYPES);

  NDArray::registerSpecialUse(outputs, inputs);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface for fused attention with scores output
//////////////////////////////////////////////////////////////////////////////
void fusedAttentionCudaWithScores(
   NDArray* query,
   NDArray* key,
   NDArray* value,
   NDArray* output,
   NDArray* attentionLogits,
   NDArray* attentionScores,
   double scale,
   bool isCausal,
   LaunchContext* context) {

 auto stream = context->getCudaStream();

 const auto batch = query->sizeAt(0);
 const auto seqQ = query->sizeAt(1);
 const auto seqKV = key->sizeAt(1);
 const auto dim = query->sizeAt(2);

 // Prepare all arrays that will be used
 std::vector<NDArray*> outputs = {output};
 if (attentionLogits != nullptr) outputs.push_back(attentionLogits);
 if (attentionScores != nullptr) outputs.push_back(attentionScores);
 NDArray::prepareSpecialUse(outputs, {query, key, value});

 // Get raw pointers (nullptr if array is null)
 void* logitsPtr = attentionLogits != nullptr ? attentionLogits->specialBuffer() : nullptr;
 void* scoresPtr = attentionScores != nullptr ? attentionScores->specialBuffer() : nullptr;

 BUILD_SINGLE_SELECTOR(query->dataType(), fusedAttention3DWithScoresLauncher,
                       (query->specialBuffer(), key->specialBuffer(),
                        value->specialBuffer(),
                        output->specialBuffer(), logitsPtr, scoresPtr,
                        batch, seqQ, seqKV, dim, scale, isCausal, *stream),
                       SD_FLOAT_TYPES);

 NDArray::registerSpecialUse(outputs, {query, key, value});
}

}  // namespace sd
