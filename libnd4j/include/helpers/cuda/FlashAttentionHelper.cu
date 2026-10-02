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
#include <helpers/AttentionWorkspace.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <execution/cuda/LaunchDims.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <string>
#include <type_traits>

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
// Scalar GQA decode uses a 256-thread block. Match one KV score to each thread so the
// block is fully occupied and the online-softmax loop needs 4x fewer barriers.
constexpr int GQA_DECODE_TILE_SIZE_KV = 256;
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
   const int biasRank,             // 0=no bias, 1=[seqKV], 2=[seqQ,seqKV], 3=[batch,seqQ,seqKV], 4=[batch,1,seqQ,seqKV]
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

   // Step 3: Rescale previous output if max changed. Every thread reads the
   // previous max before thread 0 publishes the new one; updating globalMax
   // while other threads still compare against it is a WAR race that leaves
   // a scheduling-dependent subset of output dimensions unrescaled.
   const AccT previousMax = globalMax;
   const bool maxChanged = newMax > previousMax;
   const AccT rescale = maxChanged ? flashExp<AccT>(previousMax - newMax) : static_cast<AccT>(1);
   if (maxChanged) {
     for (int d = threadIdx.x; d < dim; d += blockDim.x) {
       sharedOutput[d] *= rescale;
     }
   }
   __syncthreads();
   if (threadIdx.x == 0 && maxChanged) {
     globalSum *= rescale;
     globalMax = newMax;
   }
   __syncthreads();

   // Step 4: Compute exp(score - max) and accumulate sum.
   AccT tileSum = static_cast<AccT>(0);
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     // A masked score has zero weight even before any finite tile is seen.
     // Otherwise an all-masked leading tile evaluates -inf - -inf and poisons
     // the online sum/output before later, valid sliding-window tiles arrive.
     AccT expScore = sharedScores[k] == -DataTypeUtils::infOrMax<AccT>()
         ? static_cast<AccT>(0)
         : flashExp<AccT>(sharedScores[k] - globalMax);
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
    typename FlashAccType<T>::type* __restrict__ accumulatorScratch,
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

  // Public HALF/BFLOAT16 auxiliary outputs are observations, not computation
  // scratch. Reloading them quantizes logits and probabilities before P*V.
  // Keep those intermediates in AccT until the final output stores instead.
  const LongType scratchRow = (batchIdx * numQHeads + qHead) * seqQ + queryIdx;
  AccT* logitsAcc = accumulatorScratch != nullptr
      ? accumulatorScratch + scratchRow * 2 * seqKV : nullptr;
  AccT* scoresAcc = logitsAcc != nullptr ? logitsAcc + seqKV : nullptr;

  __shared__ AccT warpMaxes[32];
  __shared__ AccT warpSums[32];
  __shared__ AccT globalMax;
  __shared__ AccT globalSum;

  const LongType causalOffset = seqKV > seqQ ? seqKV - seqQ : 0;
  const LongType queryPosition = validCurrentWindow
      ? currentStart + queryIdx
      : queryIdx + causalOffset;
  // A current window ends the written cache prefix at currentStart + currentSeq; rows past it are
  // unwritten or stale, so they are never attended whatever the bias holds.
  const LongType writtenKV = validCurrentWindow ? min(currentStart + currentSeq, seqKV) : seqKV;
  const LongType maxKV = isCausal ? min(queryPosition + 1, writtenKV) : writtenKV;

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
    if (logitsAcc != nullptr) logitsAcc[kv] = logit;
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

  // Empty-reduction identity (accumulation contract): a row with no
  // attendable position at all — every kv either beyond maxKV or masked by
  // an additive bias so negative it saturates to -infinity (e.g. a window
  // substrate row entirely marked inactive/rejected) — reduces globalMax to
  // -infinity. flashExp(logit - globalMax) would then evaluate
  // (-infinity) - (-infinity) = NaN and poison this row's scores/output.
  // Define that case as zero probability everywhere instead: harmless,
  // finite, and confined to this row (this block only writes its own
  // queryIdx's logits/scores/output slices).
  const bool rowHasFiniteMax = globalMax > -DataTypeUtils::infOrMax<AccT>();

  AccT threadSum = static_cast<AccT>(0);
  for (LongType kv = threadIdx.x; kv < seqKV; kv += blockDim.x) {
    const AccT logit = logitsAcc != nullptr ? logitsAcc[kv]
        : static_cast<AccT>(logitsRow[kv * strides.logits[3]]);
    const AccT probability = (kv < maxKV && rowHasFiniteMax)
        ? flashExp<AccT>(logit - globalMax)
        : static_cast<AccT>(0);
    scoresRow[kv * strides.scores[3]] = static_cast<T>(probability);
    if (scoresAcc != nullptr) scoresAcc[kv] = probability;
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
  // BFLOAT16 has FLOAT32's exponent range: even two finite V values can
  // overflow an unnormalized FLOAT32 numerator. Normalize in AccT first.
  constexpr bool normalizeBeforePv = std::is_same<T, bfloat16>::value;
  for (LongType kv = threadIdx.x; kv < seqKV; kv += blockDim.x) {
    const LongType scoreOffset = kv * strides.scores[3];
    scoresRow[scoreOffset] = static_cast<T>(
        (scoresAcc != nullptr ? scoresAcc[kv]
                              : static_cast<AccT>(scoresRow[scoreOffset])) * invSum);
    if (normalizeBeforePv && scoresAcc != nullptr) scoresAcc[kv] *= invSum;
  }
  __syncthreads();

  for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
    AccT accumulated = static_cast<AccT>(0);
    for (LongType kv = 0; kv < seqKV; kv++) {
      const AccT probability = scoresAcc != nullptr ? scoresAcc[kv]
          : static_cast<AccT>(scoresRow[kv * strides.scores[3]]);
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
    // HALF's bounded range permits the final reciprocal; BFLOAT16's full
    // exponent range requires the normalized AccT weights above.
    outRow[d * strides.o[3]] = static_cast<T>(
        scoresAcc != nullptr && !normalizeBeforePv ? accumulated * invSum : accumulated);
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
    void* accumulatorScratch,
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
      reinterpret_cast<typename FlashAccType<T>::type*>(accumulatorScratch),
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
   } else if (biasRank == 2) {
     // [seqQ, seqKV], shared by every batch
     biasStride1 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride2 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
   } else if (biasRank == 1) {
     // [seqKV], shared by every batch and query
     biasStride2 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
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

// Loads each thread of a GQA decode block issues per staging pass before it stores any, so a
// pass costs about one memory latency instead of one per element.
constexpr int GQA_DECODE_STAGE_LOADS = 16;
// Dynamic shared memory every CUDA device grants a block without an opt-in.
constexpr size_t GQA_DECODE_SHARED_LIMIT = 48 * 1024;

// Copies rows [firstKv, firstKv + rows) of one KV head into staged[row * pitch + d] as AccT,
// with coalesced loads by the whole block. Rows inside the current producer window come from
// that window, the others from the cache. The conversion to AccT is exact, so a consumer reads
// the same values it would read from global memory.
template <typename T, typename AccT>
static SD_DEVICE SD_INLINE void stageGqaDecodeRows(
    AccT* staged, const LongType pitch, const int rows, const LongType firstKv, const LongType headDim,
    const T* cacheBase, const LongType cacheRowStride, const LongType cacheDimStride,
    const T* windowBase, const LongType windowRowStride, const LongType windowDimStride,
    const bool validWindow, const LongType windowStart, const LongType windowSeq) {
  const int width = static_cast<int>(headDim);
  const int total = rows * width;
  const int passElements = static_cast<int>(blockDim.x) * GQA_DECODE_STAGE_LOADS;
  for (int passStart = 0; passStart < total; passStart += passElements) {
    AccT loaded[GQA_DECODE_STAGE_LOADS];
#pragma unroll
    for (int j = 0; j < GQA_DECODE_STAGE_LOADS; j++) {
      const int element = passStart + static_cast<int>(threadIdx.x) + j * static_cast<int>(blockDim.x);
      if (element < total) {
        const int row = element / width;
        const LongType d = element - row * width;
        const LongType kvIdx = firstKv + row;
        const LongType windowIndex = kvIdx - windowStart;
        const bool fromWindow = validWindow && windowIndex >= 0 && windowIndex < windowSeq;
        const T* source = fromWindow ? windowBase + windowIndex * windowRowStride + d * windowDimStride
                                     : cacheBase + kvIdx * cacheRowStride + d * cacheDimStride;
        loaded[j] = static_cast<AccT>(*source);
      }
    }
#pragma unroll
    for (int j = 0; j < GQA_DECODE_STAGE_LOADS; j++) {
      const int element = passStart + static_cast<int>(threadIdx.x) + j * static_cast<int>(blockDim.x);
      if (element < total) {
        const int row = element / width;
        staged[row * pitch + (element - row * width)] = loaded[j];
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Direct GQA attention kernel — 4D BSHD inputs, tiled online softmax.
// Each block handles one (batch, qHead, queryIdx) tuple.
// K/V are indexed via kvHead = qHead / headsPerKvHead, so multi-row GQA
// avoids both K/V head materialization and Q/K/V permutation round-trips.
// NO atomicAdd — each thread owns output dimensions.
// K and V rows pass through shared memory stageRows at a time (stageGqaDecodeRows); every
// score and output sum is still the same sequential chain over the same values, so the
// result does not depend on stageRows and matches the compiled Triton recipe
// (emitNativeOrderedGqaRowAttention).
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
   const LongType biasStride3,
   // K/V rows staged per pass
   const int stageRows) {

 using AccT = typename FlashAccType<T>::type;

 // Legacy decode launches (seqQ == 1) use the 2D grid from getFusedGQADecodeDims
 // where blockIdx.x spans numQHeads*batch and blockIdx.y is unused. Multi-row
 // verification launches (seqQ > 1) use the 3D decomposition
 // (blockIdx.x = qHead, blockIdx.y = batch, blockIdx.z = queryIdx) set up by
 // fusedGQADecodeLauncher, so every query row is an independent block instead of
 // all rows aliasing batch row 0.
 const LongType qHead = seqQ <= 1 ? blockIdx.x % numQHeads : blockIdx.x;
 const LongType batchIdx = seqQ <= 1 ? blockIdx.x / numQHeads : blockIdx.y;
 const LongType queryIdx = seqQ <= 1 ? 0 : blockIdx.z;
 if (batchIdx >= batch || qHead >= numQHeads || queryIdx >= seqQ) return;

 const LongType kvHead = qHead / headsPerKvHead;
 if (kvHead >= numKvHeads) return;

 // Shared memory layout, all in accumulator precision: the tile's scores, the output
 // accumulator [headDim], this tile's P*V subtotals [headDim], the query row [headDim], and
 // stageRows K or V rows at a pitch of headDim + 1, so a warp reading one row per thread and a
 // warp reading one dimension per thread both hit distinct banks.
 extern __shared__ char sharedMem[];
 AccT* sharedScores = reinterpret_cast<AccT*>(sharedMem);
 AccT* sharedOutput = sharedScores + GQA_DECODE_TILE_SIZE_KV;
 AccT* tileOutput = sharedOutput + headDim;
 AccT* queryRow = tileOutput + headDim;
 AccT* stagedRows = queryRow + headDim;
 const LongType stagePitch = headDim + 1;

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
 // anchored at that device-resident cache position and the written cache prefix
 // ends at currentStart + currentSeq: rows past it are unwritten or stale, so they
 // are never attended whatever the bias holds (the INT8 decode uses the same
 // bound). Otherwise retain the right-aligned semantics used by direct non-cache
 // callers.
 const LongType causalOffset = seqKV > seqQ ? seqKV - seqQ : 0;
 const LongType queryPosition = validCurrentWindow
     ? currentStart + queryIdx
     : queryIdx + causalOffset;
 const LongType writtenKV = validCurrentWindow ? min(currentStart + currentSeq, seqKV) : seqKV;
 const LongType maxKV = isCausal ? min(queryPosition + 1, writtenKV) : writtenKV;

 // Online softmax state (block-wide via shared memory)
 __shared__ AccT globalMax;
 __shared__ AccT globalSum;
 if (threadIdx.x == 0) {
   globalMax = -DataTypeUtils::infOrMax<AccT>();
   globalSum = static_cast<AccT>(0);
 }

 // Initialize output accumulator and stage the query row
 for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
   sharedOutput[d] = static_cast<AccT>(0);
   queryRow[d] = static_cast<AccT>(Q[d * qStride3]);
 }
 __syncthreads();

 if (headDim <= 0 || seqKV <= 0) return;

 // Tile only over the device-resident logical prefix. The cache tensor remains max-sized
 // (and therefore capture-stable), while cache_position controls the actual work per replay.
 for (LongType kvStart = 0; kvStart < maxKV; kvStart += GQA_DECODE_TILE_SIZE_KV) {
   const LongType kvEnd = min(kvStart + GQA_DECODE_TILE_SIZE_KV, maxKV);
   const int tileSize = static_cast<int>(kvEnd - kvStart);
   if (tileSize <= 0) continue;

   // Step 1: Compute Q @ K^T scores for this tile + add bias, from K rows staged
   // stageRows at a time. Each score is one sequential chain over d.
   // Positions beyond this query row's causal boundary remain -inf.
   for (int stageStart = 0; stageStart < tileSize; stageStart += stageRows) {
     const int rows = min(stageRows, tileSize - stageStart);
     stageGqaDecodeRows<T, AccT>(stagedRows, stagePitch, rows, kvStart + stageStart, headDim,
                                 Kbase, kStride1, kStride3,
                                 currentKBase, currentKStride1, currentKStride3,
                                 validCurrentWindow, currentStart, currentSeq);
     __syncthreads();

     for (int r = threadIdx.x; r < rows; r += blockDim.x) {
       const int k = stageStart + r;
       const LongType kvIdx = kvStart + k;
       AccT score = -DataTypeUtils::infOrMax<AccT>();
       if (kvIdx < maxKV) {
         const AccT* Krow = stagedRows + r * stagePitch;
         score = static_cast<AccT>(0);
         for (LongType d = 0; d < headDim; d++) {
           score += queryRow[d] * Krow[d];
         }
         score *= static_cast<AccT>(scale);

         if (biasRow != nullptr) {
           score += static_cast<AccT>(biasRow[kvIdx * biasStride3]);
         }
       }

       sharedScores[k] = score;
     }
     __syncthreads();
   }

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

   // Step 3: Rescale previous output accumulator if max changed. Every thread
   // reads the previous max before thread 0 publishes the new one (see the
   // same WAR hazard note in the 4D kernel above).
   const AccT previousMax = globalMax;
   const bool maxChanged = newMax > previousMax;
   const AccT rescale = maxChanged ? flashExp<AccT>(previousMax - newMax) : static_cast<AccT>(1);
   if (maxChanged) {
     for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
       sharedOutput[d] *= rescale;
     }
   }
   __syncthreads();
   if (threadIdx.x == 0 && maxChanged) {
     globalSum *= rescale;
     globalMax = newMax;
   }
   __syncthreads();

   // Step 4: Compute exp(score - max) and accumulate sum
   AccT tileSum = static_cast<AccT>(0);
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     // A masked score has zero weight even before any finite tile is seen.
     // Otherwise an all-masked leading tile evaluates -inf - -inf and poisons
     // the online sum/output before later, valid sliding-window tiles arrive.
     AccT expScore = sharedScores[k] == -DataTypeUtils::infOrMax<AccT>()
         ? static_cast<AccT>(0)
         : flashExp<AccT>(sharedScores[k] - globalMax);
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

   // Step 5: Accumulate weighted V — each thread owns a subset of output dims. A dim's
   // tile subtotal is one chain over k in order, carried in tileOutput across the V rows
   // staged stageRows at a time, and then added to the output accumulator.
   for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
     tileOutput[d] = static_cast<AccT>(0);
   }
   for (int stageStart = 0; stageStart < tileSize; stageStart += stageRows) {
     const int rows = min(stageRows, tileSize - stageStart);
     stageGqaDecodeRows<T, AccT>(stagedRows, stagePitch, rows, kvStart + stageStart, headDim,
                                 Vbase, vStride1, vStride3,
                                 currentVBase, currentVStride1, currentVStride3,
                                 validCurrentWindow, currentStart, currentSeq);
     __syncthreads();

     for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
       AccT acc = tileOutput[d];
       for (int r = 0; r < rows; r++) {
         acc += sharedScores[stageStart + r] * stagedRows[r * stagePitch + d];
       }
       tileOutput[d] = acc;
     }
     __syncthreads();
   }
   for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
     sharedOutput[d] += tileOutput[d];
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
 // getFusedGQADecodeDims() was built for the seqQ==1 decode contract and returns
 // grid.x = numQHeads*batch. For seqQ == 1 that product equals the legacy 2D
 // grid and blockIdx.z (always 0) is unused by callers that read blockIdx.y as
 // the batch index. For seqQ > 1 (W-wide MTP verification), the product form is
 // AMBIGUOUS: the kernel reads blockIdx.y as batchIdx, so batch >= 2 collapses
 // every query row onto batch row 0. Decompose the product explicitly into
 // (qHead, batch, queryIdx) triples so every verification row gets its own
 // blockIdx.z while the seqQ == 1 decode path stays bit-identical.
 dim3 grid;
 if (seqQ <= 1) {
   grid = dim3(static_cast<unsigned int>(numQHeads * batch), 1u, 1u);
 } else {
   grid = dim3(static_cast<unsigned int>(numQHeads),
               static_cast<unsigned int>(batch),
               static_cast<unsigned int>(seqQ));
 }
 dim3 block(threadsPerBlock);

 using AccT = typename FlashAccType<T>::type;
 // Stage as many K/V rows as the block loads in one pass, at most a tile, and fewer when the
 // layout (scores, three [headDim] rows, staged rows) would not fit a block's shared memory.
 auto layoutBytes = [&](LongType rows) -> LongType {
   return (GQA_DECODE_TILE_SIZE_KV + 3 * headDim + rows * (headDim + 1)) * static_cast<LongType>(sizeof(AccT));
 };
 LongType stageRows = sd::math::sd_max<LongType>(
     1, sd::math::sd_min<LongType>(GQA_DECODE_TILE_SIZE_KV,
                                   static_cast<LongType>(threadsPerBlock) * GQA_DECODE_STAGE_LOADS /
                                       sd::math::sd_max<LongType>(1, headDim)));
 while (stageRows > 1 && layoutBytes(stageRows) > static_cast<LongType>(GQA_DECODE_SHARED_LIMIT)) stageRows /= 2;
 const size_t smem = static_cast<size_t>(
     sd::math::sd_max<LongType>(layoutBytes(stageRows), static_cast<LongType>(sharedMem)));

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
     biasStride0, biasStride1, biasStride2, biasStride3,
     static_cast<int>(stageRows));
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
   // Normalize rank-1/2/3/4 masks to logical [batch, head, query, key]
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
   } else if (biasRank == 2) {
     biasStride2 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
     biasStride3 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
   } else {
     // Rank 1: [seqKV], or a scalar
     biasStride3 = attentionBias->lengthOf() > 1 ? attentionBias->strideAt(0) : 0;
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
// Single-token GQA decode over INT8 K/V caches, templated on the model dtype T: the query, the
// current K/V window, the bias and the output are T. Each cache row carries one FLOAT32 scale,
// held in separate [batch, seqKV, kvHeads] caches or inline in bytes [headDim, headDim + 4) of a
// [.., headDim + 4] row (ADR 0107 V2). The K row scale folds into the dot product and the V row
// scale into the softmax weight; scores, softmax state and the output accumulator are AccT.
// Grid and shared memory follow fusedGQADecodeKernel's seqQ == 1 contract: one block per
// (qHead, batch) with blockIdx.x = batch * numQHeads + qHead.
//
// Rows inside the current window [currentStart, currentStart + currentSeq) are read from the T
// window, so this call never reads back the INT8 rows it just wrote and the current token is not
// quantized twice. Attention stops at the end of that window, which for the single decode query
// is also its causal bound, so unwritten or stale rows are never read whatever the bias holds.
// Without a window the bias alone masks.
//
// The optional score/logit outputs get the softmax weights and pre-softmax scores of that same
// computation; rows past the attended prefix get the masked logit and a zero score.
//////////////////////////////////////////////////////////////////////////////
static SD_DEVICE SD_INLINE float int8KvRowScale(const int8_t* row, LongType headDim, const float* separateScale) {
  if (separateScale != nullptr) return *separateScale;
  // A row-inline scale sits at an arbitrary byte offset, so read it without assuming alignment.
  float rowScale;
  memcpy(&rowScale, row + headDim, sizeof(float));
  return rowScale;
}

template <typename T>
SD_KERNEL __launch_bounds__(512, 1) void fusedGQADecodeQuantisedKernel(
    const T* query,               // [batch, 1, numQHeads, headDim]
    const int8_t* keyCache,       // [batch, seqKV, numKvHeads, headDim] or [.., headDim + 4]
    const float* keyScales,       // [batch, seqKV, numKvHeads], nullptr when row-inline
    const int8_t* valueCache,     // [batch, seqKV, numKvHeads, headDim] or [.., headDim + 4]
    const float* valueScales,     // [batch, seqKV, numKvHeads], nullptr when row-inline
    const T* currentKeyWindow,    // [batch, currentSeq, numKvHeads, headDim] or nullptr
    const T* currentValueWindow,  // [batch, currentSeq, numKvHeads, headDim] or nullptr
    const LongType* currentKvPosition,
    const LongType currentSeq,
    const T* attnBias,            // logical [batch, numQHeads, 1, seqKV] or nullptr
    T* output,                    // [batch, 1, numQHeads, headDim]
    T* attentionScores,           // [batch, numQHeads, 1, seqKV] or nullptr
    T* attentionLogits,           // [batch, numQHeads, 1, seqKV] or nullptr
    const LongType batch,
    const LongType seqKV,
    const LongType numQHeads,
    const LongType headDim,
    const LongType headsPerKvHead,
    const double scale,
    const LongType qStride0, const LongType qStride2, const LongType qStride3,
    const LongType kStride0, const LongType kStride1, const LongType kStride2, const LongType kStride3,
    const LongType vStride0, const LongType vStride1, const LongType vStride2, const LongType vStride3,
    const LongType kScaleStride0, const LongType kScaleStride1, const LongType kScaleStride2,
    const LongType vScaleStride0, const LongType vScaleStride1, const LongType vScaleStride2,
    const LongType currentKStride0, const LongType currentKStride1,
    const LongType currentKStride2, const LongType currentKStride3,
    const LongType currentVStride0, const LongType currentVStride1,
    const LongType currentVStride2, const LongType currentVStride3,
    const LongType oStride0, const LongType oStride2, const LongType oStride3,
    // Broadcast-safe bias strides (zero on size-1 dims); seqQ == 1 needs no query stride.
    const LongType biasStride0, const LongType biasStride1, const LongType biasStride3,
    const LongType scoresStride0, const LongType scoresStride1, const LongType scoresStride3,
    const LongType logitsStride0, const LongType logitsStride1, const LongType logitsStride3) {
  using AccT = typename FlashAccType<T>::type;

  const LongType qHead = blockIdx.x % numQHeads;
  const LongType batchIdx = blockIdx.x / numQHeads;
  if (batchIdx >= batch) return;
  const LongType kvHead = qHead / headsPerKvHead;

  // Shared memory layout: scores tile + output accumulator [headDim], both AccT.
  extern __shared__ char sharedMem[];
  AccT* sharedScores = reinterpret_cast<AccT*>(sharedMem);
  AccT* sharedOutput = sharedScores + GQA_DECODE_TILE_SIZE_KV;
  __shared__ AccT reduceScratch[WARP_SIZE];
  __shared__ AccT globalMax;
  __shared__ AccT globalSum;
  __shared__ AccT tileRescale;

  const T* Q = query + batchIdx * qStride0 + qHead * qStride2;
  T* O = output + batchIdx * oStride0 + qHead * oStride2;
  const T* biasRow = attnBias != nullptr ? attnBias + batchIdx * biasStride0 + qHead * biasStride1 : nullptr;
  T* scoresRow = attentionScores != nullptr ? attentionScores + batchIdx * scoresStride0 + qHead * scoresStride1
                                            : nullptr;
  T* logitsRow = attentionLogits != nullptr ? attentionLogits + batchIdx * logitsStride0 + qHead * logitsStride1
                                            : nullptr;

  const int8_t* kBase = keyCache + batchIdx * kStride0 + kvHead * kStride2;
  const int8_t* vBase = valueCache + batchIdx * vStride0 + kvHead * vStride2;
  const float* kScaleBase =
      keyScales != nullptr ? keyScales + batchIdx * kScaleStride0 + kvHead * kScaleStride2 : nullptr;
  const float* vScaleBase =
      valueScales != nullptr ? valueScales + batchIdx * vScaleStride0 + kvHead * vScaleStride2 : nullptr;

  const bool hasCurrentWindow = currentKeyWindow != nullptr && currentValueWindow != nullptr &&
                                currentKvPosition != nullptr && currentSeq > 0;
  const LongType currentStart = hasCurrentWindow ? currentKvPosition[0] : -1;
  const bool validCurrentWindow = hasCurrentWindow && currentStart >= 0 && currentStart < seqKV;
  const T* currentKBase =
      validCurrentWindow ? currentKeyWindow + batchIdx * currentKStride0 + kvHead * currentKStride2 : nullptr;
  const T* currentVBase =
      validCurrentWindow ? currentValueWindow + batchIdx * currentVStride0 + kvHead * currentVStride2 : nullptr;
  const LongType maxKV =
      validCurrentWindow ? sd::math::sd_min<LongType>(currentStart + currentSeq, seqKV) : seqKV;

  const AccT masked = -DataTypeUtils::infOrMax<AccT>();
  const AccT scaleAcc = static_cast<AccT>(scale);

  // Q.K * scale + bias for one attended row. Step 1 and the score/logit outputs share it.
  auto scoreAt = [&](LongType kvIdx) -> AccT {
    const LongType currentIndex = kvIdx - currentStart;
    AccT dot = static_cast<AccT>(0);
    if (validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq) {
      const T* kRow = currentKBase + currentIndex * currentKStride1;
      for (LongType d = 0; d < headDim; d++) {
        dot += static_cast<AccT>(Q[d * qStride3]) * static_cast<AccT>(kRow[d * currentKStride3]);
      }
    } else {
      const int8_t* kRow = kBase + kvIdx * kStride1;
      for (LongType d = 0; d < headDim; d++) {
        dot += static_cast<AccT>(Q[d * qStride3]) * static_cast<AccT>(kRow[d * kStride3]);
      }
      dot *= static_cast<AccT>(
          int8KvRowScale(kRow, headDim, kScaleBase != nullptr ? kScaleBase + kvIdx * kScaleStride1 : nullptr));
    }
    AccT score = dot * scaleAcc;
    if (biasRow != nullptr) score += static_cast<AccT>(biasRow[kvIdx * biasStride3]);
    return score;
  };

  if (threadIdx.x == 0) {
    globalMax = masked;
    globalSum = static_cast<AccT>(0);
  }
  for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
    sharedOutput[d] = static_cast<AccT>(0);
  }
  __syncthreads();

  for (LongType kvStart = 0; kvStart < maxKV; kvStart += GQA_DECODE_TILE_SIZE_KV) {
    const int tileSize = static_cast<int>(sd::math::sd_min<LongType>(GQA_DECODE_TILE_SIZE_KV, maxKV - kvStart));

    // Step 1: scores = Q.K * scale + bias, plus a per-thread max.
    AccT localMax = masked;
    for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
      const AccT score = scoreAt(kvStart + k);
      sharedScores[k] = score;
      localMax = sd::math::sd_max<AccT>(localMax, score);
    }

    // Step 2: tile max, valid in thread 0.
    const AccT tileMax = sd::device::blockReduceMax<AccT>(localMax, reduceScratch);

    // Step 3: only thread 0 updates the running max and sum, and the other threads read the
    // rescale factor after the barrier, so no thread reads globalMax while it changes. Before any
    // finite score there is nothing to rescale, which also keeps a fully masked leading tile from
    // evaluating -inf - -inf.
    if (threadIdx.x == 0) {
      const AccT newMax = sd::math::sd_max<AccT>(globalMax, tileMax);
      tileRescale = (globalMax == masked || newMax == globalMax) ? static_cast<AccT>(1)
                                                                  : flashExp<AccT>(globalMax - newMax);
      globalSum *= tileRescale;
      globalMax = newMax;
    }
    __syncthreads();
    if (tileRescale != static_cast<AccT>(1)) {
      for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        sharedOutput[d] *= tileRescale;
      }
    }

    // Step 4: softmax numerators; a masked score has zero weight. The running sum takes the plain
    // weight, while the stored weight of a cache row also carries that row's V scale.
    AccT localSum = static_cast<AccT>(0);
    for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
      const AccT score = sharedScores[k];
      const AccT weight = score == masked ? static_cast<AccT>(0) : flashExp<AccT>(score - globalMax);
      localSum += weight;
      const LongType kvIdx = kvStart + k;
      const LongType currentIndex = kvIdx - currentStart;
      if (validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq) {
        sharedScores[k] = weight;
      } else {
        sharedScores[k] = weight * static_cast<AccT>(int8KvRowScale(
            vBase + kvIdx * vStride1, headDim, vScaleBase != nullptr ? vScaleBase + kvIdx * vScaleStride1 : nullptr));
      }
    }
    // The barrier inside the reduction also publishes the weights to Step 5.
    const AccT tileSum = sd::device::blockReduceSum<AccT>(localSum, reduceScratch);
    if (threadIdx.x == 0) globalSum += tileSum;

    // Step 5: weighted V; each thread owns a disjoint set of output dimensions.
    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
      AccT acc = static_cast<AccT>(0);
      for (int k = 0; k < tileSize; k++) {
        const LongType kvIdx = kvStart + k;
        const LongType currentIndex = kvIdx - currentStart;
        const AccT value = validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq
                               ? static_cast<AccT>(currentVBase[currentIndex * currentVStride1 + d * currentVStride3])
                               : static_cast<AccT>(vBase[kvIdx * vStride1 + d * vStride3]);
        acc += sharedScores[k] * value;
      }
      sharedOutput[d] += acc;
    }
    __syncthreads();
  }

  // Step 6: normalize. If nothing was attended (every score masked, or an empty prefix) the
  // output is zero.
  const AccT invSum = globalSum > static_cast<AccT>(0) ? static_cast<AccT>(1) / globalSum : static_cast<AccT>(0);
  for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
    O[d * oStride3] = static_cast<T>(sharedOutput[d] * invSum);
  }

  // Step 7: the optional score/logit outputs. The shared tile only ever holds one tile, so the
  // scores are recomputed against the final max and sum. The last tile barrier published both.
  if (scoresRow != nullptr || logitsRow != nullptr) {
    for (LongType kvIdx = threadIdx.x; kvIdx < seqKV; kvIdx += blockDim.x) {
      const AccT score = kvIdx < maxKV ? scoreAt(kvIdx) : masked;
      if (logitsRow != nullptr) logitsRow[kvIdx * logitsStride3] = static_cast<T>(score);
      if (scoresRow != nullptr) {
        const AccT weight = score == masked ? static_cast<AccT>(0) : flashExp<AccT>(score - globalMax);
        scoresRow[kvIdx * scoresStride3] = static_cast<T>(weight * invSum);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for fusedGQADecodeQuantised
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void fusedGQADecodeQuantisedLauncher(
    const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t* stream,
    const void* vQuery, const void* vKeyCache, const void* vKeyScales,
    const void* vValueCache, const void* vValueScales,
    const void* vCurrentKeyWindow, const void* vCurrentValueWindow,
    const void* vCurrentKvPosition, LongType currentSeq,
    const void* vAttnBias, void* vOutput, void* vAttentionScores, void* vAttentionLogits,
    LongType batch, LongType seqKV, LongType numQHeads, LongType headDim, LongType headsPerKvHead, double scale,
    LongType qStride0, LongType qStride2, LongType qStride3,
    LongType kStride0, LongType kStride1, LongType kStride2, LongType kStride3,
    LongType vStride0, LongType vStride1, LongType vStride2, LongType vStride3,
    LongType kScaleStride0, LongType kScaleStride1, LongType kScaleStride2,
    LongType vScaleStride0, LongType vScaleStride1, LongType vScaleStride2,
    LongType currentKStride0, LongType currentKStride1, LongType currentKStride2, LongType currentKStride3,
    LongType currentVStride0, LongType currentVStride1, LongType currentVStride2, LongType currentVStride3,
    LongType oStride0, LongType oStride2, LongType oStride3,
    LongType biasStride0, LongType biasStride1, LongType biasStride3,
    LongType scoresStride0, LongType scoresStride1, LongType scoresStride3,
    LongType logitsStride0, LongType logitsStride1, LongType logitsStride3) {
  using AccT = typename FlashAccType<T>::type;
  // Never launch with less than the score tile plus the headDim accumulator in AccT.
  const size_t required = static_cast<size_t>(GQA_DECODE_TILE_SIZE_KV + headDim) * sizeof(AccT);
  const size_t smem = static_cast<size_t>(sharedMem) > required ? static_cast<size_t>(sharedMem) : required;

  fusedGQADecodeQuantisedKernel<T><<<blocksPerGrid, threadsPerBlock, smem, *stream>>>(
      reinterpret_cast<const T*>(vQuery),
      reinterpret_cast<const int8_t*>(vKeyCache), reinterpret_cast<const float*>(vKeyScales),
      reinterpret_cast<const int8_t*>(vValueCache), reinterpret_cast<const float*>(vValueScales),
      reinterpret_cast<const T*>(vCurrentKeyWindow), reinterpret_cast<const T*>(vCurrentValueWindow),
      reinterpret_cast<const LongType*>(vCurrentKvPosition), currentSeq,
      reinterpret_cast<const T*>(vAttnBias), reinterpret_cast<T*>(vOutput),
      reinterpret_cast<T*>(vAttentionScores), reinterpret_cast<T*>(vAttentionLogits),
      batch, seqKV, numQHeads, headDim, headsPerKvHead, scale,
      qStride0, qStride2, qStride3,
      kStride0, kStride1, kStride2, kStride3,
      vStride0, vStride1, vStride2, vStride3,
      kScaleStride0, kScaleStride1, kScaleStride2,
      vScaleStride0, vScaleStride1, vScaleStride2,
      currentKStride0, currentKStride1, currentKStride2, currentKStride3,
      currentVStride0, currentVStride1, currentVStride2, currentVStride3,
      oStride0, oStride2, oStride3,
      biasStride0, biasStride1, biasStride3,
      scoresStride0, scoresStride1, scoresStride3,
      logitsStride0, logitsStride1, logitsStride3);
  DebugHelper::checkGlobalErrorCode("fusedGQADecodeQuantised failed");
}

//////////////////////////////////////////////////////////////////////////////
// Public interface: fusedGQADecodeQuantisedCuda. Same validation as fusedGQADecodeQuantisedCpu.
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
    NDArray* attentionBias,
    NDArray* currentKeyWindow,
    NDArray* currentValueWindow,
    const void* currentKvPosition,
    NDArray* attentionScores,
    NDArray* attentionLogits) {

  const DataType dtype = query->dataType();
  if (query->rankOf() != 4 || query->sizeAt(1) != 1) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: query must be [batch, 1, qHeads, headDim]");
  }
  const LongType batch = query->sizeAt(0);
  const LongType numQHeads = query->sizeAt(2);
  const LongType headDim = query->sizeAt(3);
  if (output->dataType() != dtype || output->rankOf() != 4 || output->sizeAt(0) != batch ||
      output->sizeAt(1) != 1 || output->sizeAt(2) != numQHeads || output->sizeAt(3) != headDim) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: output must match the query shape and dtype");
  }
  if (quantKeyCache->dataType() != DataType::INT8 || quantValCache->dataType() != DataType::INT8 ||
      quantKeyCache->rankOf() != 4 || !quantKeyCache->isSameShape(quantValCache)) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: key/value caches must be INT8 rank-4 with the same shape");
  }
  if (quantKeyCache->sizeAt(0) != batch) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: cache batch must match the query batch");
  }
  const LongType seqKV = quantKeyCache->sizeAt(1);
  const LongType numKvHeads = quantKeyCache->sizeAt(2);
  if (numKvHeads <= 0 || numQHeads % numKvHeads != 0) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: qHeads must be a multiple of kvHeads");
  }

  // ADR 0107 V2 ROW-INLINE: null scale caches mean each cache row carries its FLOAT32 scale at
  // row + headDim, inside the logical tensor. Non-null scales are the separate [batch, seqKV, kvHeads]
  // layout.
  const bool inlineScales = (keyScaleCache == nullptr);
  if (inlineScales != (valScaleCache == nullptr)) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: key and value scale caches must both be set or both be null");
  }
  LongType kScaleStride0 = 0, kScaleStride1 = 0, kScaleStride2 = 0;
  LongType vScaleStride0 = 0, vScaleStride1 = 0, vScaleStride2 = 0;
  if (inlineScales) {
    if (quantKeyCache->sizeAt(3) != headDim + 4 || quantKeyCache->strideAt(3) != 1 ||
        quantValCache->strideAt(3) != 1) {
      THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: row-inline caches must be [batch, seqKV, kvHeads, headDim+4] "
                      "with a unit last-dimension stride");
    }
  } else {
    if (quantKeyCache->sizeAt(3) != headDim) {
      THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: separate-scale cache last dim must equal headDim");
    }
    const std::vector<LongType> scaleShape = {batch, seqKV, numKvHeads};
    if (keyScaleCache->dataType() != DataType::FLOAT32 || valScaleCache->dataType() != DataType::FLOAT32 ||
        !keyScaleCache->isSameShape(scaleShape) || !valScaleCache->isSameShape(scaleShape)) {
      THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: scale caches must be FLOAT32 [batch, seqKV, kvHeads]");
    }
    kScaleStride0 = keyScaleCache->strideAt(0);
    kScaleStride1 = keyScaleCache->strideAt(1);
    kScaleStride2 = keyScaleCache->strideAt(2);
    vScaleStride0 = valScaleCache->strideAt(0);
    vScaleStride1 = valScaleCache->strideAt(1);
    vScaleStride2 = valScaleCache->strideAt(2);
  }

  const bool hasWindow = currentKeyWindow != nullptr && currentValueWindow != nullptr;
  if (hasWindow) {
    if (currentKeyWindow->dataType() != dtype || currentValueWindow->dataType() != dtype ||
        currentKeyWindow->rankOf() != 4 || !currentKeyWindow->isSameShape(currentValueWindow) ||
        currentKeyWindow->sizeAt(0) != batch || currentKeyWindow->sizeAt(2) != numKvHeads ||
        currentKeyWindow->sizeAt(3) != headDim) {
      THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: current K/V windows must be [batch, seq, kvHeads, headDim] "
                      "in the query dtype");
    }
  }
  const bool hasBias = attentionBias != nullptr && !attentionBias->isEmpty();
  if (hasBias && attentionBias->dataType() != dtype) {
    THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: attentionBias must be in the query dtype");
  }
  // The kernel reads the bias at every kvIdx below the device-resident written-prefix bound, so a
  // wide last dim must cover the whole cache; the other dims broadcast or match exactly.
  if (hasBias) {
    const int biasRank = attentionBias->rankOf();
    auto broadcastsTo = [attentionBias](int dim, LongType full) {
      return attentionBias->sizeAt(dim) == 1 || attentionBias->sizeAt(dim) == full;
    };
    const LongType biasKv = biasRank == 0 ? 1 : attentionBias->sizeAt(biasRank - 1);
    bool biasFits = biasRank <= 4 && (biasKv == 1 || biasKv >= seqKV);
    if (biasRank == 4) {
      biasFits = biasFits && broadcastsTo(0, batch) && broadcastsTo(1, numQHeads) && attentionBias->sizeAt(2) == 1;
    } else if (biasRank == 3) {
      biasFits = biasFits && broadcastsTo(0, batch) && attentionBias->sizeAt(1) == 1;
    } else if (biasRank == 2) {
      biasFits = biasFits && attentionBias->sizeAt(0) == 1;
    }
    if (!biasFits) {
      THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: attentionBias must broadcast to [batch, qHeads, 1, seqKV]");
    }
  }
  // Null or empty aux outputs are not requested (the DSP executor passes empty placeholders for
  // dead outputs); requested ones must be the dpa_v2 score layout in the query dtype.
  NDArray* scoresOut = attentionScores != nullptr && !attentionScores->isEmpty() ? attentionScores : nullptr;
  NDArray* logitsOut = attentionLogits != nullptr && !attentionLogits->isEmpty() ? attentionLogits : nullptr;
  const std::vector<LongType> auxShape = {batch, numQHeads, 1, seqKV};
  for (NDArray* aux : {scoresOut, logitsOut}) {
    if (aux != nullptr && (aux->dataType() != dtype || !aux->isSameShape(auxShape))) {
      THROW_EXCEPTION("fusedGQADecodeQuantisedCuda: attention scores/logits must be [batch, qHeads, 1, seqKV] "
                      "in the query dtype");
    }
  }

  if (batch == 0 || numQHeads == 0 || headDim == 0) return;

  // "Current window" contract (mirrors fusedGQADecodeCuda): the device-resident cache position
  // anchors the pre-quantization window rows inside the cache.
  const bool useCurrentWindow = hasWindow && currentKvPosition != nullptr;
  const LongType currentSeq = useCurrentWindow ? currentKeyWindow->sizeAt(1) : 0;
  LongType currentKStride0 = 0, currentKStride1 = 0, currentKStride2 = 0, currentKStride3 = 0;
  LongType currentVStride0 = 0, currentVStride1 = 0, currentVStride2 = 0, currentVStride3 = 0;
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

  // Additive bias, broadcast through zero strides on size-1 dims. seqQ == 1, so the query
  // dimension never contributes an offset. Rank 3 is [batch, seqQ, seqKV], rank 2 [seqQ, seqKV].
  LongType biasStride0 = 0, biasStride1 = 0, biasStride3 = 0;
  if (hasBias) {
    auto strideIfWide = [attentionBias](int dim) -> LongType {
      return attentionBias->sizeAt(dim) > 1 ? attentionBias->strideAt(dim) : 0;
    };
    const int biasRank = attentionBias->rankOf();
    if (biasRank == 4) {
      biasStride0 = strideIfWide(0);
      biasStride1 = strideIfWide(1);
      biasStride3 = strideIfWide(3);
    } else if (biasRank == 3) {
      biasStride0 = strideIfWide(0);
      biasStride3 = strideIfWide(2);
    } else if (biasRank == 2) {
      biasStride3 = strideIfWide(1);
    } else {
      biasStride3 = attentionBias->lengthOf() > 1 ? attentionBias->strideAt(0) : 0;
    }
  }

  // Row-inline scales live inside the cache buffers, so separate scale caches are only listed
  // when they exist.
  std::vector<NDArray*> inputs = {query, quantKeyCache, quantValCache};
  if (!inlineScales) {
    inputs.push_back(keyScaleCache);
    inputs.push_back(valScaleCache);
  }
  if (useCurrentWindow) {
    inputs.push_back(currentKeyWindow);
    inputs.push_back(currentValueWindow);
  }
  if (hasBias) inputs.push_back(attentionBias);
  std::vector<NDArray*> outputs = {output};
  if (scoresOut != nullptr) outputs.push_back(scoresOut);
  if (logitsOut != nullptr) outputs.push_back(logitsOut);
  NDArray::prepareSpecialUse(outputs, inputs);

  const dim3 launchDims = getFusedGQADecodeDims(static_cast<int>(numQHeads), static_cast<int>(batch),
                                                static_cast<int>(seqKV), static_cast<int>(headDim),
                                                static_cast<int>(query->sizeOfT()));

  BUILD_SINGLE_SELECTOR(dtype, fusedGQADecodeQuantisedLauncher,
                        (launchDims.x, launchDims.y, launchDims.z, context->getCudaStream(),
                         query->specialBuffer(),
                         quantKeyCache->specialBuffer(), inlineScales ? nullptr : keyScaleCache->specialBuffer(),
                         quantValCache->specialBuffer(), inlineScales ? nullptr : valScaleCache->specialBuffer(),
                         useCurrentWindow ? currentKeyWindow->specialBuffer() : nullptr,
                         useCurrentWindow ? currentValueWindow->specialBuffer() : nullptr,
                         useCurrentWindow ? currentKvPosition : nullptr, currentSeq,
                         hasBias ? attentionBias->specialBuffer() : nullptr, output->specialBuffer(),
                         scoresOut != nullptr ? scoresOut->specialBuffer() : nullptr,
                         logitsOut != nullptr ? logitsOut->specialBuffer() : nullptr,
                         batch, seqKV, numQHeads, headDim, numQHeads / numKvHeads, scale,
                         query->strideAt(0), query->strideAt(2), query->strideAt(3),
                         quantKeyCache->strideAt(0), quantKeyCache->strideAt(1),
                         quantKeyCache->strideAt(2), quantKeyCache->strideAt(3),
                         quantValCache->strideAt(0), quantValCache->strideAt(1),
                         quantValCache->strideAt(2), quantValCache->strideAt(3),
                         kScaleStride0, kScaleStride1, kScaleStride2,
                         vScaleStride0, vScaleStride1, vScaleStride2,
                         currentKStride0, currentKStride1, currentKStride2, currentKStride3,
                         currentVStride0, currentVStride1, currentVStride2, currentVStride3,
                         output->strideAt(0), output->strideAt(2), output->strideAt(3),
                         biasStride0, biasStride1, biasStride3,
                         scoresOut != nullptr ? scoresOut->strideAt(0) : 0,
                         scoresOut != nullptr ? scoresOut->strideAt(1) : 0,
                         scoresOut != nullptr ? scoresOut->strideAt(3) : 0,
                         logitsOut != nullptr ? logitsOut->strideAt(0) : 0,
                         logitsOut != nullptr ? logitsOut->strideAt(1) : 0,
                         logitsOut != nullptr ? logitsOut->strideAt(3) : 0),
                        SD_FLOAT_TYPES);

  NDArray::registerSpecialUse(outputs, inputs);
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
  NDArray* accumulatorScratch = nullptr;
  if (query->dataType() == DataType::HALF || query->dataType() == DataType::BFLOAT16) {
    // Reuse the plan/stream-scoped attention workspace. Include the shape in
    // the key so another attention slot cannot evict storage captured here.
    const std::string scratchKey = "gqa_with_scores_acc_" + std::to_string(batch)
        + "_" + std::to_string(numQHeads) + "_" + std::to_string(seqQ)
        + "_" + std::to_string(seqKV);
    accumulatorScratch = AttentionWorkspace::getInstance()->getBuffer(
        scratchKey, {batch, numQHeads, seqQ, 2, seqKV}, DataType::FLOAT32, context);
    outputs.push_back(accumulatorScratch);
  }
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
       accumulatorScratch != nullptr ? accumulatorScratch->specialBuffer() : nullptr,
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
