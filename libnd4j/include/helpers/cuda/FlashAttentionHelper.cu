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

//////////////////////////////////////////////////////////////////////////////
// In-place causal mask kernel - sets scores[b, i, j] = -inf where j > i
// This replaces: create mask array + nullify + fillAsTriangular + broadcast add
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ __launch_bounds__(256, 4) void applyCausalMaskInPlaceKernel(
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
__global__ __launch_bounds__(1024, 2) void fusedCausalMaskSoftmaxKernel(
   const T* __restrict__ input,   // [batch, seqQ, seqKV] - logits from Q@K^T
   T* __restrict__ output,        // [batch, seqQ, seqKV] - softmax output
   T* __restrict__ logitsOut,     // [batch, seqQ, seqKV] - masked logits (optional)
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV,
   const bool isCausal) {

 const LongType row = blockIdx.x;  // which row (batch*seqQ rows)
 if (row >= batch * seqQ) return;

 const LongType queryIdx = row % seqQ;
 const LongType rowStart = row * seqKV;
 const LongType causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;
 const LongType queryPos = queryIdx + causalOffset;

 extern __shared__ char sharedMem[];
 float* sdata = reinterpret_cast<float*>(sharedMem);

 // Pass 1: Apply causal mask and find max
 float threadMax = -INFINITY;
 const LongType maxKV = isCausal ? min(queryPos + 1, seqKV) : seqKV;

 for (LongType j = threadIdx.x; j < seqKV; j += blockDim.x) {
   float val;
   if (isCausal && j > queryPos) {
     val = -1.0e9f;
   } else {
     val = static_cast<float>(input[rowStart + j]);
   }
   // Store masked logits if requested
   if (logitsOut != nullptr) {
     logitsOut[rowStart + j] = static_cast<T>(val);
   }
   threadMax = fmaxf(threadMax, val);
 }

 // Warp reduce max
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadMax = fmaxf(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
 }

 const int lane = threadIdx.x % WARP_SIZE;
 const int wid = threadIdx.x / WARP_SIZE;
 const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

 if (lane == 0) sdata[wid] = threadMax;
 __syncthreads();

 float rowMax = -INFINITY;
 if (threadIdx.x < numWarps) rowMax = sdata[threadIdx.x];
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   rowMax = fmaxf(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
 }

 __shared__ float sharedMax;
 if (threadIdx.x == 0) sharedMax = rowMax;
 __syncthreads();
 rowMax = sharedMax;

 // Pass 2: Compute sum of exp(x - max) — NO output writes.
 // This kernel is called with input == output (in-place). Writing exp values
 // to output here would clobber the original logits that later iterations
 // of the same loop (or other threads) still need to read. Instead, we only
 // accumulate the sum and defer all output writes to Pass 3.
 float threadSum = 0.0f;
 for (LongType j = threadIdx.x; j < seqKV; j += blockDim.x) {
   float val;
   if (logitsOut != nullptr) {
     val = static_cast<float>(logitsOut[rowStart + j]);
   } else if (isCausal && j > queryPos) {
     val = -1.0e9f;
   } else {
     val = static_cast<float>(input[rowStart + j]);
   }
   threadSum += sd_fast_exp(val - rowMax);
 }

 // Warp reduce sum
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
 }
 if (lane == 0) sdata[wid] = threadSum;
 __syncthreads();

 float rowSum = 0.0f;
 if (threadIdx.x < numWarps) rowSum = sdata[threadIdx.x];
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
 }

 __shared__ float sharedSum;
 if (threadIdx.x == 0) sharedSum = rowSum;
 __syncthreads();
 float invSum = 1.0f / sharedSum;

 // Pass 3: Compute exp and normalize in one pass, write to output.
 // Safe for in-place (input == output): each thread reads input[j] then writes
 // output[j] at the same index. Threads handle non-overlapping j values
 // (stride = blockDim.x), so no thread reads a location another thread has
 // already written in this pass.
 for (LongType j = threadIdx.x; j < seqKV; j += blockDim.x) {
   float val;
   if (logitsOut != nullptr) {
     val = static_cast<float>(logitsOut[rowStart + j]);
   } else if (isCausal && j > queryPos) {
     val = -1.0e9f;
   } else {
     val = static_cast<float>(input[rowStart + j]);
   }
   float expVal = sd_fast_exp(val - rowMax);
   output[rowStart + j] = static_cast<T>(expVal * invSum);
 }
}

template <typename T>
static void fusedCausalMaskSoftmaxLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                          const int sharedMem, const cudaStream_t* stream,
                                          const void* vInput, void* vOutput, void* vLogitsOut,
                                          LongType batch, LongType seqQ, LongType seqKV, bool isCausal) {
 auto input = reinterpret_cast<const T*>(vInput);
 auto output = reinterpret_cast<T*>(vOutput);
 auto logitsOut = vLogitsOut != nullptr ? reinterpret_cast<T*>(vLogitsOut) : nullptr;
 fusedCausalMaskSoftmaxKernel<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
     input, output, logitsOut, batch, seqQ, seqKV, isCausal);
 DebugHelper::checkGlobalErrorCode("fusedCausalMaskSoftmax failed");
}

BUILD_SINGLE_TEMPLATE(void fusedCausalMaskSoftmaxLauncher,
                     (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
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
 size_t sharedMemSize = numWarps * sizeof(float);

 if (logitsOut != nullptr) {
   NDArray::prepareSpecialUse({output, logitsOut}, {input});
 } else {
   NDArray::prepareSpecialUse({output}, {input});
 }

 void* logitsPtr = logitsOut != nullptr ? logitsOut->specialBuffer() : nullptr;

 BUILD_SINGLE_SELECTOR(input->dataType(), fusedCausalMaskSoftmaxLauncher,
                       (numRows, threadsPerBlock, sharedMemSize, stream,
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
__global__ __launch_bounds__(512, 1) void fusedAttention3DKernel(
   const T* __restrict__ query,    // [batch, seqQ, dim]
   const T* __restrict__ key,      // [batch, seqKV, dim]
   const T* __restrict__ value,    // [batch, seqKV, dim]
   const T* __restrict__ attnBias, // [batch, seqQ, seqKV] or [batch, 1, seqQ, seqKV] or nullptr
   T* __restrict__ output,         // [batch, seqQ, dim]
   const LongType batch,
   const LongType seqQ,
   const LongType seqKV,
   const LongType dim,
   const float scale,
   const bool isCausal,
   const int biasRank,             // 0=no bias, 3=[batch,seqQ,seqKV], 4=[batch,1,seqQ,seqKV]
   const LongType biasStride0,     // Stride for batch dimension
   const LongType biasStride1,     // Stride for seqQ (or heads) dimension
   const LongType biasStride2) {   // Stride for seqKV dimension

 // Each block handles one query position for one batch
 const LongType batchIdx = blockIdx.y;
 const LongType queryIdx = blockIdx.x;

 if (batchIdx >= batch || queryIdx >= seqQ) return;

 // Shared memory for partial results
 extern __shared__ char sharedMem[];
 T* sharedScores = reinterpret_cast<T*>(sharedMem);  // [TILE_SIZE_KV]
 T* sharedOutput = sharedScores + TILE_SIZE_KV;      // [dim]

 // Thread-local accumulators for online softmax
 float threadMax = -INFINITY;
 float threadSum = 0.0f;

 // Initialize output accumulator to zero
 for (int d = threadIdx.x; d < dim; d += blockDim.x) {
   sharedOutput[d] = static_cast<T>(0);
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

 // Global max and sum for this query position
 __shared__ float globalMax;
 __shared__ float globalSum;
 if (threadIdx.x == 0) {
   globalMax = -INFINITY;
   globalSum = 0.0f;
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

   // Step 1: Compute Q @ K^T for this tile + add attention bias
   // Each thread computes one or more dot products
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     const LongType kvIdx = kvStart + k;
     const T* Krow = K + kvIdx * dim;

     float score = 0.0f;
     for (LongType d = 0; d < dim; d++) {
       score += static_cast<float>(Q[d]) * static_cast<float>(Krow[d]);
     }
     score *= scale;

     // Add attention bias if present
     if (biasRow != nullptr) {
       score += static_cast<float>(biasRow[kvIdx * biasStride2]);
     }

     // Apply causal mask
     if (isCausal && kvIdx > queryPos) {
       score = -INFINITY;
     }

     sharedScores[k] = static_cast<T>(score);
   }
   __syncthreads();

   // Step 2: Find max in this tile (for numerical stability)
   float tileMax = -INFINITY;
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     tileMax = fmaxf(tileMax, static_cast<float>(sharedScores[k]));
   }

   // Warp reduce to find max
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = fmaxf(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   // First thread in each warp writes to shared memory
   __shared__ float warpMaxes[32];
   if (threadIdx.x % WARP_SIZE == 0) {
     warpMaxes[threadIdx.x / WARP_SIZE] = tileMax;
   }
   __syncthreads();

   // First warp reduces across all warps
   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileMax = warpMaxes[threadIdx.x];
   } else {
     tileMax = -INFINITY;
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = fmaxf(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   __shared__ float newMax;
   if (threadIdx.x == 0) {
     newMax = fmaxf(globalMax, tileMax);
   }
   __syncthreads();

   // Step 3: Rescale previous output if max changed
   if (newMax > globalMax) {
     float rescale = sd_fast_exp(globalMax - newMax);
     for (int d = threadIdx.x; d < dim; d += blockDim.x) {
       sharedOutput[d] = static_cast<T>(static_cast<float>(sharedOutput[d]) * rescale);
     }
     if (threadIdx.x == 0) {
       globalSum *= rescale;
       globalMax = newMax;
     }
   }
   __syncthreads();

   // Step 4: Compute exp(score - max) and accumulate sum
   float tileSum = 0.0f;
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     float score = static_cast<float>(sharedScores[k]);
     float expScore = sd_fast_exp(score - globalMax);
     sharedScores[k] = static_cast<T>(expScore);
     tileSum += expScore;
   }

   // Reduce sum across threads
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   __shared__ float warpSums[32];
   if (threadIdx.x % WARP_SIZE == 0) {
     warpSums[threadIdx.x / WARP_SIZE] = tileSum;
   }
   __syncthreads();

   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileSum = warpSums[threadIdx.x];
   } else {
     tileSum = 0.0f;
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   if (threadIdx.x == 0) {
     globalSum += tileSum;
   }
   __syncthreads();

   // Step 5: Accumulate weighted values: output += exp_scores @ V
   for (int d = threadIdx.x; d < dim; d += blockDim.x) {
     float acc = 0.0f;
     for (int k = 0; k < tileSize; k++) {
       const LongType kvIdx = kvStart + k;
       acc += static_cast<float>(sharedScores[k]) * static_cast<float>(V[kvIdx * dim + d]);
     }
     sharedOutput[d] = static_cast<T>(static_cast<float>(sharedOutput[d]) + acc);
   }
   __syncthreads();
 }

 // Step 6: Normalize by sum and write output
 float invSum3d = (globalSum > 0.0f) ? (1.0f / globalSum) : 0.0f;
 for (int d = threadIdx.x; d < dim; d += blockDim.x) {
   O[d] = static_cast<T>(static_cast<float>(sharedOutput[d]) * invSum3d);
 }
}

//////////////////////////////////////////////////////////////////////////////
// Fused attention kernel WITH scores output - for cases where we need
// to return attention logits and/or attention scores
// This version materializes the full attention row for each query position
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ __launch_bounds__(256, 2) void fusedAttentionWithScores3DKernel(
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
   const float scale,
   const bool isCausal) {

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

 // Shared memory for reductions
 extern __shared__ char sharedMem[];
 float* sharedMax = reinterpret_cast<float*>(sharedMem);  // [32] for warp maxes
 float* sharedSum = sharedMax + 32;                        // [32] for warp sums

 // Step 1: Compute all logits for this query row and find max
 float threadMax = -INFINITY;
 const LongType causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;
 const LongType queryPos = queryIdx + causalOffset;
 const LongType maxKV = isCausal ? min(queryPos + 1, seqKV) : seqKV;

 for (LongType k = threadIdx.x; k < seqKV; k += blockDim.x) {
   float score;
   if (k < maxKV) {
     // Compute dot product Q[queryIdx] . K[k]
     const T* Krow = K + k * dim;
     score = 0.0f;
     for (LongType d = 0; d < dim; d++) {
       score += static_cast<float>(Q[d]) * static_cast<float>(Krow[d]);
     }
     score *= scale;
   } else {
     // Causal mask: future positions get -inf
     score = -INFINITY;
   }

   // Write logits if requested
   if (logitsRow != nullptr) {
     logitsRow[k] = static_cast<T>(score);
   }

   threadMax = fmaxf(threadMax, score);
 }

 // Reduce max across threads
 for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
   threadMax = fmaxf(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
 }
 if (threadIdx.x % WARP_SIZE == 0) {
   sharedMax[threadIdx.x / WARP_SIZE] = threadMax;
 }
 __syncthreads();

 // First warp reduces across all warps
 float globalMax = -INFINITY;
 if (threadIdx.x < 32) {
   float val = (threadIdx.x < blockDim.x / WARP_SIZE) ? sharedMax[threadIdx.x] : -INFINITY;
   for (int offset = 16; offset > 0; offset /= 2) {
     val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
   }
   if (threadIdx.x == 0) {
     sharedMax[0] = val;
   }
 }
 __syncthreads();
 globalMax = sharedMax[0];

 // Step 2: Compute exp(score - max) and sum, also write scores
 float threadSum = 0.0f;
 for (LongType k = threadIdx.x; k < seqKV; k += blockDim.x) {
   float score;
   if (logitsRow != nullptr) {
     score = static_cast<float>(logitsRow[k]);
   } else if (k < maxKV) {
     // Recompute score if logits not stored
     const T* Krow = K + k * dim;
     score = 0.0f;
     for (LongType d = 0; d < dim; d++) {
       score += static_cast<float>(Q[d]) * static_cast<float>(Krow[d]);
     }
     score *= scale;
   } else {
     score = -INFINITY;
   }

   float expScore = sd_fast_exp(score - globalMax);
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

 float globalSum = 0.0f;
 if (threadIdx.x < 32) {
   float val = (threadIdx.x < blockDim.x / WARP_SIZE) ? sharedSum[threadIdx.x] : 0.0f;
   for (int offset = 16; offset > 0; offset /= 2) {
     val += __shfl_down_sync(0xffffffff, val, offset);
   }
   if (threadIdx.x == 0) {
     sharedSum[0] = val;
   }
 }
 __syncthreads();
 globalSum = sharedSum[0];
 float invSum = (globalSum > 0.0f) ? (1.0f / globalSum) : 0.0f;

 // Step 3: Normalize scores (write to scoresRow if needed)
 if (scoresRow != nullptr) {
   for (LongType k = threadIdx.x; k < seqKV; k += blockDim.x) {
     float expScore = static_cast<float>(scoresRow[k]);
     scoresRow[k] = static_cast<T>(expScore * invSum);
   }
 }
 __syncthreads();

 // Step 4: Compute output - each thread handles a subset of output dimensions
 // This avoids atomicAdd contention by having each thread own its dimensions
 for (int d = threadIdx.x; d < dim; d += blockDim.x) {
   float acc = 0.0f;
   for (LongType k = 0; k < seqKV; k++) {
     float attnWeight;
     if (scoresRow != nullptr) {
       attnWeight = static_cast<float>(scoresRow[k]);
     } else if (logitsRow != nullptr) {
       float score = static_cast<float>(logitsRow[k]);
       attnWeight = sd_fast_exp(score - globalMax) * invSum;
     } else if (k < maxKV) {
       // Recompute score
       const T* Krow = K + k * dim;
       float score = 0.0f;
       for (LongType dd = 0; dd < dim; dd++) {
         score += static_cast<float>(Q[dd]) * static_cast<float>(Krow[dd]);
       }
       score *= scale;
       attnWeight = sd_fast_exp(score - globalMax) * invSum;
     } else {
       attnWeight = 0.0f;
     }
     acc += attnWeight * static_cast<float>(V[k * dim + d]);
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
   float scale,
   bool isCausal,
   cudaStream_t stream) {

 // Grid: one block per (query_position, batch) pair
 dim3 grid(seqQ, batch);
 dim3 block(256);

 // Shared memory: max array + sum array
 size_t sharedMem = 32 * sizeof(float) + 32 * sizeof(float);

 fusedAttentionWithScores3DKernel<T><<<grid, block, sharedMem, stream>>>(
     query, key, value, output, attentionLogits, attentionScores,
     batch, seqQ, seqKV, dim, scale, isCausal);

 DebugHelper::checkGlobalErrorCode("fusedAttention3DWithScores failed");
}

// Explicit instantiations for with-scores variant
template void launchFusedAttention3DWithScores<float>(
   const float*, const float*, const float*, float*, float*, float*,
   LongType, LongType, LongType, LongType, float, bool, cudaStream_t);

template void launchFusedAttention3DWithScores<double>(
   const double*, const double*, const double*, double*, double*, double*,
   LongType, LongType, LongType, LongType, float, bool, cudaStream_t);

template void launchFusedAttention3DWithScores<float16>(
   const float16*, const float16*, const float16*, float16*, float16*, float16*,
   LongType, LongType, LongType, LongType, float, bool, cudaStream_t);

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
   float scale,
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

 // Shared memory: scores tile + output accumulator + warp reduction arrays
 size_t sharedMem = (TILE_SIZE_KV + dim) * sizeof(T) + 64 * sizeof(float);

 fusedAttention3DKernel<T><<<grid, block, sharedMem, stream>>>(
     query, key, value, attnBias, output,
     batch, seqQ, seqKV, dim,
     scale, isCausal,
     biasRank, biasStride0, biasStride1, biasStride2);

 DebugHelper::checkGlobalErrorCode("fusedAttention3D failed");
}

// Explicit instantiations
template void launchFusedAttention3D<float>(
   const float*, const float*, const float*, const float*, float*,
   LongType, LongType, LongType, LongType, float, bool,
   int, LongType, LongType, LongType, cudaStream_t);

template void launchFusedAttention3D<double>(
   const double*, const double*, const double*, const double*, double*,
   LongType, LongType, LongType, LongType, float, bool,
   int, LongType, LongType, LongType, cudaStream_t);

template void launchFusedAttention3D<float16>(
   const float16*, const float16*, const float16*, const float16*, float16*,
   LongType, LongType, LongType, LongType, float, bool,
   int, LongType, LongType, LongType, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public interface - called from FlashAttentionHelper
// Supports optional attention bias for ONNX MultiHeadAttention compatibility
//////////////////////////////////////////////////////////////////////////////
void fusedAttentionCuda(
   NDArray* query,
   NDArray* key,
   NDArray* value,
   NDArray* output,
   float scale,
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

 auto dtype = query->dataType();
 if (dtype == DataType::FLOAT32) {
   launchFusedAttention3D<float>(
       reinterpret_cast<const float*>(query->specialBuffer()),
       reinterpret_cast<const float*>(key->specialBuffer()),
       reinterpret_cast<const float*>(value->specialBuffer()),
       reinterpret_cast<const float*>(biasPtr),
       reinterpret_cast<float*>(output->specialBuffer()),
       batch, seqQ, seqKV, dim, scale, isCausal,
       biasRank, biasStride0, biasStride1, biasStride2, *stream);
 } else if (dtype == DataType::DOUBLE) {
   launchFusedAttention3D<double>(
       reinterpret_cast<const double*>(query->specialBuffer()),
       reinterpret_cast<const double*>(key->specialBuffer()),
       reinterpret_cast<const double*>(value->specialBuffer()),
       reinterpret_cast<const double*>(biasPtr),
       reinterpret_cast<double*>(output->specialBuffer()),
       batch, seqQ, seqKV, dim, scale, isCausal,
       biasRank, biasStride0, biasStride1, biasStride2, *stream);
 } else if (dtype == DataType::HALF) {
   launchFusedAttention3D<float16>(
       reinterpret_cast<const float16*>(query->specialBuffer()),
       reinterpret_cast<const float16*>(key->specialBuffer()),
       reinterpret_cast<const float16*>(value->specialBuffer()),
       reinterpret_cast<const float16*>(biasPtr),
       reinterpret_cast<float16*>(output->specialBuffer()),
       batch, seqQ, seqKV, dim, scale, isCausal,
       biasRank, biasStride0, biasStride1, biasStride2, *stream);
 } else {
   THROW_EXCEPTION("fusedAttentionCuda: Unsupported data type");
 }

 if (attentionBias != nullptr && !attentionBias->isEmpty()) {
   NDArray::registerSpecialUse({output}, {query, key, value, attentionBias});
 } else {
   NDArray::registerSpecialUse({output}, {query, key, value});
 }
}

//////////////////////////////////////////////////////////////////////////////
// Fused GQA decode attention kernel — 4D BSHD inputs, tiled online softmax.
// Each block handles one (batch, qHead) pair.
// K/V indexed via kvHead = qHead / headsPerKvHead for GQA head mapping.
// Tiles over KV positions with online softmax + rescaling (same pattern as
// fusedAttention3DKernel). NO atomicAdd — each thread owns output dimensions.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ __launch_bounds__(512, 1) void fusedGQADecodeKernel(
   const T* __restrict__ query,      // [batch, 1, numQHeads, headDim] BSHD
   const T* __restrict__ key,        // [batch, seqKV, numKvHeads, headDim] BSHD
   const T* __restrict__ value,      // [batch, seqKV, numKvHeads, headDim] BSHD
   const T* __restrict__ attnBias,   // [batch, numQHeads, 1, seqKV] or nullptr
   T* __restrict__ output,           // [batch, 1, numQHeads, headDim] BSHD
   const LongType batch,
   const LongType seqKV,
   const LongType numQHeads,
   const LongType numKvHeads,
   const LongType headDim,
   const LongType headsPerKvHead,
   const float scale,
   // Strides for Q [batch, 1, numQHeads, headDim]
   const LongType qStride0, const LongType qStride2, const LongType qStride3,
   // Strides for K [batch, seqKV, numKvHeads, headDim]
   const LongType kStride0, const LongType kStride1, const LongType kStride2, const LongType kStride3,
   // Strides for V [batch, seqKV, numKvHeads, headDim]
   const LongType vStride0, const LongType vStride1, const LongType vStride2, const LongType vStride3,
   // Strides for output [batch, 1, numQHeads, headDim]
   const LongType oStride0, const LongType oStride2, const LongType oStride3,
   // Strides for bias
   const LongType biasStride0,
   const LongType biasStride1,
   const LongType biasStride2,
   const LongType biasStride3) {

 const LongType qHead = blockIdx.x;
 const LongType batchIdx = blockIdx.y;
 if (batchIdx >= batch || qHead >= numQHeads) return;

 const LongType kvHead = qHead / headsPerKvHead;

 // Shared memory layout: scores tile [TILE_SIZE_KV] + output accumulator [headDim]
 extern __shared__ char sharedMem[];
 T* sharedScores = reinterpret_cast<T*>(sharedMem);
 T* sharedOutput = sharedScores + TILE_SIZE_KV;

 // Q pointer: query[batchIdx, 0, qHead, :] — stride-based indexing
 const T* Q = query + batchIdx * qStride0 + qHead * qStride2;

 // K/V base: key[batchIdx, :, kvHead, :] — stride-based indexing
 const T* Kbase = key + batchIdx * kStride0 + kvHead * kStride2;
 const T* Vbase = value + batchIdx * vStride0 + kvHead * vStride2;

 // Output: output[batchIdx, 0, qHead, :]
 T* O = output + batchIdx * oStride0 + qHead * oStride2;

 // Bias row: attnBias[batchIdx, qHead, 0, :]
 const T* biasRow = nullptr;
 if (attnBias != nullptr) {
   biasRow = attnBias + batchIdx * biasStride0 + qHead * biasStride1;
 }

 // Online softmax state (block-wide via shared memory)
 __shared__ float globalMax;
 __shared__ float globalSum;
 if (threadIdx.x == 0) {
   globalMax = -INFINITY;
   globalSum = 0.0f;
 }

 // Initialize output accumulator
 for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
   sharedOutput[d] = static_cast<T>(0);
 }
 __syncthreads();

 if (headDim <= 0 || seqKV <= 0) return;

 // Tile over KV positions — same structure as fusedAttention3DKernel
 for (LongType kvStart = 0; kvStart < seqKV; kvStart += TILE_SIZE_KV) {
   const LongType kvEnd = min(kvStart + TILE_SIZE_KV, seqKV);
   const int tileSize = static_cast<int>(kvEnd - kvStart);
   if (tileSize <= 0) continue;

   // Step 1: Compute Q @ K^T scores for this tile + add bias
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     const LongType kvIdx = kvStart + k;
     // K[batchIdx, kvIdx, kvHead, d] — use stride-based offset
     const T* Krow = Kbase + kvIdx * kStride1;

     float score = 0.0f;
     for (LongType d = 0; d < headDim; d++) {
       score += static_cast<float>(Q[d * qStride3]) * static_cast<float>(Krow[d * kStride3]);
     }
     score *= scale;

     if (biasRow != nullptr) {
       score += static_cast<float>(biasRow[kvIdx * biasStride3]);
     }

     sharedScores[k] = static_cast<T>(score);
   }
   __syncthreads();

   // Step 2: Find max in this tile
   float tileMax = -INFINITY;
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     tileMax = fmaxf(tileMax, static_cast<float>(sharedScores[k]));
   }

   // Warp reduce max
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = fmaxf(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   __shared__ float warpMaxes[32];
   if (threadIdx.x % WARP_SIZE == 0) {
     warpMaxes[threadIdx.x / WARP_SIZE] = tileMax;
   }
   __syncthreads();

   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileMax = warpMaxes[threadIdx.x];
   } else {
     tileMax = -INFINITY;
   }
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileMax = fmaxf(tileMax, __shfl_down_sync(0xffffffff, tileMax, offset));
   }

   __shared__ float newMax;
   if (threadIdx.x == 0) {
     newMax = fmaxf(globalMax, tileMax);
   }
   __syncthreads();

   // Step 3: Rescale previous output accumulator if max changed
   if (newMax > globalMax) {
     float rescale = sd_fast_exp(globalMax - newMax);
     for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
       sharedOutput[d] = static_cast<T>(static_cast<float>(sharedOutput[d]) * rescale);
     }
     if (threadIdx.x == 0) {
       globalSum *= rescale;
       globalMax = newMax;
     }
   }
   __syncthreads();

   // Step 4: Compute exp(score - max) and accumulate sum
   float tileSum = 0.0f;
   for (int k = threadIdx.x; k < tileSize; k += blockDim.x) {
     float score = static_cast<float>(sharedScores[k]);
     float expScore = sd_fast_exp(score - globalMax);
     sharedScores[k] = static_cast<T>(expScore);
     tileSum += expScore;
   }

   // Reduce sum across threads
   for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
     tileSum += __shfl_down_sync(0xffffffff, tileSum, offset);
   }

   __shared__ float warpSums[32];
   if (threadIdx.x % WARP_SIZE == 0) {
     warpSums[threadIdx.x / WARP_SIZE] = tileSum;
   }
   __syncthreads();

   if (threadIdx.x < blockDim.x / WARP_SIZE) {
     tileSum = warpSums[threadIdx.x];
   } else {
     tileSum = 0.0f;
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
     float acc = 0.0f;
     for (int k = 0; k < tileSize; k++) {
       const LongType kvIdx = kvStart + k;
       // V[batchIdx, kvIdx, kvHead, d] — use stride-based offset
       acc += static_cast<float>(sharedScores[k]) * static_cast<float>(Vbase[kvIdx * vStride1 + d * vStride3]);
     }
     sharedOutput[d] = static_cast<T>(static_cast<float>(sharedOutput[d]) + acc);
   }
   __syncthreads();
 }

 // Step 6: Normalize by sum and write output
 // Guard against globalSum == 0 (all positions masked → exp sums to 0).
 // Output zeros when nothing is attended to, matching PyTorch behavior.
 float invSum = (globalSum > 0.0f) ? (1.0f / globalSum) : 0.0f;
 for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
   O[d * oStride3] = static_cast<T>(static_cast<float>(sharedOutput[d]) * invSum);
 }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for fused GQA decode — uses void* params + BUILD_SINGLE_SELECTOR
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void fusedGQADecodeLauncher(
   const int blocksPerGrid, const int threadsPerBlock,
   const int sharedMem, const cudaStream_t* stream,
   const void* vQuery, const void* vKey, const void* vValue, const void* vAttnBias,
   void* vOutput,
   LongType batch, LongType seqKV, LongType numQHeads, LongType numKvHeads,
   LongType headDim, LongType headsPerKvHead, float scale,
   LongType qStride0, LongType qStride2, LongType qStride3,
   LongType kStride0, LongType kStride1, LongType kStride2, LongType kStride3,
   LongType vStride0, LongType vStride1, LongType vStride2, LongType vStride3,
   LongType oStride0, LongType oStride2, LongType oStride3,
   LongType biasStride0, LongType biasStride1,
   LongType biasStride2, LongType biasStride3) {

 auto query = reinterpret_cast<const T*>(vQuery);
 auto key = reinterpret_cast<const T*>(vKey);
 auto value = reinterpret_cast<const T*>(vValue);
 auto attnBias = vAttnBias != nullptr ? reinterpret_cast<const T*>(vAttnBias) : nullptr;
 auto output = reinterpret_cast<T*>(vOutput);

 // Grid: one block per (qHead, batch) pair
 dim3 grid(numQHeads, batch);
 dim3 block(threadsPerBlock);

 // Shared memory: scores tile + output accumulator + warp reduction arrays
 size_t smem = (TILE_SIZE_KV + headDim) * sizeof(T) + 64 * sizeof(float);

 fusedGQADecodeKernel<T><<<grid, block, smem, *stream>>>(
     query, key, value, attnBias, output,
     batch, seqKV, numQHeads, numKvHeads, headDim, headsPerKvHead, scale,
     qStride0, qStride2, qStride3,
     kStride0, kStride1, kStride2, kStride3,
     vStride0, vStride1, vStride2, vStride3,
     oStride0, oStride2, oStride3,
     biasStride0, biasStride1, biasStride2, biasStride3);
 DebugHelper::checkGlobalErrorCode("fusedGQADecode failed");
}

//////////////////////////////////////////////////////////////////////////////
// Public interface for fused GQA decode attention
//////////////////////////////////////////////////////////////////////////////
void fusedGQADecodeCuda(
   NDArray* query, NDArray* key, NDArray* value,
   NDArray* output, float scale,
   LaunchContext* context, NDArray* attentionBias) {

 auto stream = context->getCudaStream();

 // Input layout: BSHD — [batch, seq, heads, dim]
 const auto batch = query->sizeAt(0);
 const auto numQHeads = query->sizeAt(2);
 const auto headDim = query->sizeAt(3);
 const auto seqKV = key->sizeAt(1);
 const auto numKvHeads = key->sizeAt(2);
 const auto headsPerKvHead = numQHeads / numKvHeads;

 // Extract actual strides — kernel uses stride-based indexing so it works
 // correctly with non-contiguous views (e.g. BHSD→BSHD permuted arrays
 // from DSP pre-allocation or KV concat in onnx_mha.cpp).
 const LongType qStride0 = query->strideAt(0);
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

 const LongType oStride0 = output->strideAt(0);
 const LongType oStride2 = output->strideAt(2);
 const LongType oStride3 = output->strideAt(3);

 LongType biasStride0 = 0, biasStride1 = 0, biasStride2 = 0, biasStride3 = 0;
 const void* biasPtr = nullptr;

 if (attentionBias != nullptr && !attentionBias->isEmpty()) {
   biasPtr = attentionBias->specialBuffer();
   // Use broadcast-safe strides: zero out stride for dimensions of size 1
   // so the kernel reuses the same data (broadcast semantics).
   biasStride0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
   biasStride1 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
   biasStride2 = attentionBias->sizeAt(2) > 1 ? attentionBias->strideAt(2) : 0;
   biasStride3 = attentionBias->sizeAt(3) > 1 ? attentionBias->strideAt(3) : 0;
   NDArray::prepareSpecialUse({output}, {query, key, value, attentionBias});
 } else {
   NDArray::prepareSpecialUse({output}, {query, key, value});
 }

 // Centralized launch dimensions computation
 int dtypeSize = query->sizeOfT();
 dim3 launchDims = getFusedGQADecodeDims(numQHeads, batch, seqKV, headDim, dtypeSize);

 BUILD_SINGLE_SELECTOR(query->dataType(), fusedGQADecodeLauncher,
                       (launchDims.x, launchDims.y, launchDims.z, stream,
                        query->specialBuffer(), key->specialBuffer(),
                        value->specialBuffer(), biasPtr,
                        output->specialBuffer(),
                        batch, seqKV, numQHeads, numKvHeads,
                        headDim, headsPerKvHead, scale,
                        qStride0, qStride2, qStride3,
                        kStride0, kStride1, kStride2, kStride3,
                        vStride0, vStride1, vStride2, vStride3,
                        oStride0, oStride2, oStride3,
                        biasStride0, biasStride1, biasStride2, biasStride3),
                       SD_FLOAT_TYPES);

 if (attentionBias != nullptr && !attentionBias->isEmpty()) {
   NDArray::registerSpecialUse({output}, {query, key, value, attentionBias});
 } else {
   NDArray::registerSpecialUse({output}, {query, key, value});
 }
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
   float scale,
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

 auto dtype = query->dataType();

 // Get raw pointers (nullptr if array is null)
 void* logitsPtr = attentionLogits != nullptr ? attentionLogits->specialBuffer() : nullptr;
 void* scoresPtr = attentionScores != nullptr ? attentionScores->specialBuffer() : nullptr;

 if (dtype == DataType::FLOAT32) {
   launchFusedAttention3DWithScores<float>(
       reinterpret_cast<const float*>(query->specialBuffer()),
       reinterpret_cast<const float*>(key->specialBuffer()),
       reinterpret_cast<const float*>(value->specialBuffer()),
       reinterpret_cast<float*>(output->specialBuffer()),
       reinterpret_cast<float*>(logitsPtr),
       reinterpret_cast<float*>(scoresPtr),
       batch, seqQ, seqKV, dim, scale, isCausal, *stream);
 } else if (dtype == DataType::DOUBLE) {
   launchFusedAttention3DWithScores<double>(
       reinterpret_cast<const double*>(query->specialBuffer()),
       reinterpret_cast<const double*>(key->specialBuffer()),
       reinterpret_cast<const double*>(value->specialBuffer()),
       reinterpret_cast<double*>(output->specialBuffer()),
       reinterpret_cast<double*>(logitsPtr),
       reinterpret_cast<double*>(scoresPtr),
       batch, seqQ, seqKV, dim, scale, isCausal, *stream);
 } else if (dtype == DataType::HALF) {
   launchFusedAttention3DWithScores<float16>(
       reinterpret_cast<const float16*>(query->specialBuffer()),
       reinterpret_cast<const float16*>(key->specialBuffer()),
       reinterpret_cast<const float16*>(value->specialBuffer()),
       reinterpret_cast<float16*>(output->specialBuffer()),
       reinterpret_cast<float16*>(logitsPtr),
       reinterpret_cast<float16*>(scoresPtr),
       batch, seqQ, seqKV, dim, scale, isCausal, *stream);
 } else {
   THROW_EXCEPTION("fusedAttentionCudaWithScores: Unsupported data type");
 }

 NDArray::registerSpecialUse(outputs, {query, key, value});
}

}  // namespace sd
