
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
// @author Adam Gibson
//
// CUDA implementations of fused LLM operations.
// These mega-kernels optimize memory bandwidth by fusing multiple operations.
//

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <ops/op_types.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <types/float16.h>
#include <ops/declarable/helpers/fused_llm_ops.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

constexpr int WARP_SIZE = 32;

// Every kernel here accumulates in the framework's aggregate type (simdOps::AggregateType, ops/op_types.h): float for
// HALF and BFLOAT16, the type itself for FLOAT and DOUBLE. The kernels take SD_FLOAT_TYPES only, where this is the
// policy of the file-local accumulator type it replaced (float for everything but double).

//////////////////////////////////////////////////////////////////////////////
// Utility device functions
//////////////////////////////////////////////////////////////////////////////

// (Unused local warp/block sum reductions removed; the RMSNorm kernel below
//  uses sd::device::blockReduceSum from device_primitives.cuh.)

// (Dead fastSigmoid/silu inline helpers removed — they duplicated sd::math::sd_sigmoid
//  and were unused after the GELU/activation paths were converted to simdOps::AggregateType.)

//////////////////////////////////////////////////////////////////////////////
// Fused GELU Kernel - x * sigmoid(1.702 * x)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void fusedGELUKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const LongType totalElements) {

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  AccT x = static_cast<AccT>(input[idx]);
  // Fast GELU approximation: x * sigmoid(1.702 * x)
  AccT scale = static_cast<AccT>(1.702);
  AccT expVal = sd::math::sd_exp<AccT, AccT>(-(scale * x));
  AccT sig = static_cast<AccT>(1) / (static_cast<AccT>(1) + expVal);
  AccT result = x * sig;
  output[idx] = static_cast<T>(result);
}

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void fusedGELUBackwardKernel(
    const T* __restrict__ input,
    const T* __restrict__ gradOut,
    T* __restrict__ gradIn,
    const LongType totalElements) {

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  AccT x = static_cast<AccT>(input[idx]);
  AccT dout = static_cast<AccT>(gradOut[idx]);

  // d/dx[x * sigmoid(1.702*x)] = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
  AccT scale = static_cast<AccT>(1.702);
  AccT expVal = sd::math::sd_exp<AccT, AccT>(-(scale * x));
  AccT sig = static_cast<AccT>(1) / (static_cast<AccT>(1) + expVal);
  AccT grad = sig + x * scale * sig * (static_cast<AccT>(1) - sig);
  gradIn[idx] = static_cast<T>(dout * grad);
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm Kernel with Welford's algorithm
//////////////////////////////////////////////////////////////////////////////

template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedLayerNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ gain,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const LongType numRows,
    const LongType rowLen,
    const float epsilon) {

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char sharedMem[];
  AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

  const T* inputRow = input + row * rowLen;
  T* outputRow = output + row * rowLen;

  // Welford's online algorithm for mean and variance
  AccT mean = static_cast<AccT>(0);
  AccT M2 = static_cast<AccT>(0);
  AccT count = static_cast<AccT>(0);

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    AccT val = static_cast<AccT>(inputRow[i]);
    count += static_cast<AccT>(1);
    AccT delta = val - mean;
    mean += delta / count;
    AccT delta2 = val - mean;
    M2 += delta * delta2;
  }

  // Parallel reduction for Welford's algorithm
  AccT* sMean = sdata;
  AccT* sM2 = sdata + blockDim.x;
  AccT* sCount = sdata + 2 * blockDim.x;

  sMean[threadIdx.x] = mean;
  sM2[threadIdx.x] = M2;
  sCount[threadIdx.x] = count;
  __syncthreads();

  // Combine Welford results in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      AccT na = sCount[threadIdx.x];
      AccT nb = sCount[threadIdx.x + s];
      AccT delta = sMean[threadIdx.x + s] - sMean[threadIdx.x];
      AccT nab = na + nb;
      if (nab > static_cast<AccT>(0)) {
        sMean[threadIdx.x] = (na * sMean[threadIdx.x] + nb * sMean[threadIdx.x + s]) / nab;
        sM2[threadIdx.x] = sM2[threadIdx.x] + sM2[threadIdx.x + s] + delta * delta * na * nb / nab;
        sCount[threadIdx.x] = nab;
      }
    }
    __syncthreads();
  }

  __shared__ AccT finalMean;
  __shared__ AccT finalInvStd;

  if (threadIdx.x == 0) {
    finalMean = sMean[0];
    AccT variance = sM2[0] / sCount[0];
    finalInvStd = static_cast<AccT>(1) / sd::math::sd_sqrt<AccT, AccT>(variance + static_cast<AccT>(epsilon));
  }
  __syncthreads();

  // Normalize, scale and shift
  if (bias != nullptr) {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      AccT val = static_cast<AccT>(inputRow[i]);
      AccT normalized = (val - finalMean) * finalInvStd;
      AccT g = static_cast<AccT>(gain[i]);
      AccT b = static_cast<AccT>(bias[i]);
      outputRow[i] = static_cast<T>(normalized * g + b);
    }
  } else {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      AccT val = static_cast<AccT>(inputRow[i]);
      AccT normalized = (val - finalMean) * finalInvStd;
      AccT g = static_cast<AccT>(gain[i]);
      outputRow[i] = static_cast<T>(normalized * g);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE Kernel
//////////////////////////////////////////////////////////////////////////////

template <typename T, typename P>
SD_KERNEL __launch_bounds__(256, 2) void fusedRoPEKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const LongType batch,
    const LongType seqLen,
    const LongType numHeads,
    const LongType headDim,
    const P* __restrict__ positionPtr,
    const float freqBase,
    const float freqScale,
    const int ropeType,
    const LongType rotateDims) {

  // Each thread handles one element pair for rotation
  const LongType halfRotate = rotateDims / 2;
  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  const LongType totalPairs = batch * seqLen * numHeads * halfRotate;
  if (idx >= totalPairs) return;

  // Decode index
  const LongType pairIdx = idx % halfRotate;
  LongType rem = idx / halfRotate;
  const LongType h = rem % numHeads;
  rem /= numHeads;
  const LongType s = rem % seqLen;
  const LongType b = rem / seqLen;

  using AccT = typename simdOps::AggregateType<T>::type;

  // Read position from device pointer — capture-safe (no host sync).
  const LongType pos = static_cast<LongType>(positionPtr[0]) + s;

  // Compute theta using rotateDims for frequency spacing
  AccT theta = static_cast<AccT>(pos) * static_cast<AccT>(freqScale) /
               sd::math::sd_pow<AccT, AccT, AccT>(static_cast<AccT>(freqBase),
                   static_cast<AccT>(2) * static_cast<AccT>(pairIdx) / static_cast<AccT>(rotateDims));
  AccT cosTheta = sd::math::sd_cos<AccT, AccT>(theta);
  AccT sinTheta = sd::math::sd_sin<AccT, AccT>(theta);

  // Calculate indices based on RoPE type
  LongType base = ((b * seqLen + s) * numHeads + h) * headDim;
  LongType idx1, idx2;
  if (ropeType == 0) {  // Standard (LLaMA)
    idx1 = base + pairIdx;
    idx2 = base + pairIdx + halfRotate;
  } else if (ropeType == 1) {  // NeoX
    idx1 = base + pairIdx * 2;
    idx2 = base + pairIdx * 2 + 1;
  } else {  // GPT-J
    idx1 = base + pairIdx;
    idx2 = base + pairIdx + halfRotate;
  }

  AccT x1 = static_cast<AccT>(input[idx1]);
  AccT x2 = static_cast<AccT>(input[idx2]);

  output[idx1] = static_cast<T>(x1 * cosTheta - x2 * sinTheta);
  output[idx2] = static_cast<T>(x1 * sinTheta + x2 * cosTheta);
}

template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedRoPEBackwardKernel(
    const T* __restrict__ gradOut,
    T* __restrict__ gradIn,
    const LongType batch,
    const LongType seqLen,
    const LongType numHeads,
    const LongType headDim,
    const int positionOffset,
    const float freqBase,
    const float freqScale,
    const int ropeType,
    const LongType rotateDims) {

  const LongType halfRotate = rotateDims / 2;
  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  const LongType totalPairs = batch * seqLen * numHeads * halfRotate;
  if (idx >= totalPairs) return;

  const LongType pairIdx = idx % halfRotate;
  LongType rem = idx / halfRotate;
  const LongType h = rem % numHeads;
  rem /= numHeads;
  const LongType s = rem % seqLen;
  const LongType b = rem / seqLen;

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType pos = positionOffset + s;

  AccT theta = static_cast<AccT>(pos) * static_cast<AccT>(freqScale) /
               sd::math::sd_pow<AccT, AccT, AccT>(static_cast<AccT>(freqBase),
                   static_cast<AccT>(2) * static_cast<AccT>(pairIdx) / static_cast<AccT>(rotateDims));
  AccT cosTheta = sd::math::sd_cos<AccT, AccT>(theta);
  AccT sinTheta = sd::math::sd_sin<AccT, AccT>(theta);

  LongType base = ((b * seqLen + s) * numHeads + h) * headDim;
  LongType idx1, idx2;
  if (ropeType == 0) {
    idx1 = base + pairIdx;
    idx2 = base + pairIdx + halfRotate;
  } else if (ropeType == 1) {
    idx1 = base + pairIdx * 2;
    idx2 = base + pairIdx * 2 + 1;
  } else {
    idx1 = base + pairIdx;
    idx2 = base + pairIdx + halfRotate;
  }

  AccT g1 = static_cast<AccT>(gradOut[idx1]);
  AccT g2 = static_cast<AccT>(gradOut[idx2]);

  // Inverse rotation
  gradIn[idx1] = static_cast<T>(g1 * cosTheta + g2 * sinTheta);
  gradIn[idx2] = static_cast<T>(-g1 * sinTheta + g2 * cosTheta);
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE with pre-computed cos/sin (cached variant)
//////////////////////////////////////////////////////////////////////////////

template <typename T, typename CS = T>
SD_KERNEL __launch_bounds__(256, 2) void fusedRoPECachedKernel(
    const T* __restrict__ input,
    const CS* __restrict__ cosValues,
    const CS* __restrict__ sinValues,
    T* __restrict__ output,
    const LongType batch,
    const LongType seqLen,
    const LongType numHeads,
    const LongType headDim,
    const LongType cosStride0,
    const LongType cosStride1,
    const LongType cosStride2,
    const int ropeType) {

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  const LongType halfDim = headDim / 2;
  const LongType totalPairs = batch * seqLen * numHeads * halfDim;
  if (idx >= totalPairs) return;

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType pairIdx = idx % halfDim;
  LongType rem = idx / halfDim;
  const LongType h = rem % numHeads;
  rem /= numHeads;
  const LongType s = rem % seqLen;
  const LongType b = rem / seqLen;

  // Index into cos/sin using their actual strides (handles 2D, 3D, or 4D with broadcast)
  const LongType csIdx = b * cosStride0 + s * cosStride1 + pairIdx * cosStride2;
  AccT cosVal = static_cast<AccT>(cosValues[csIdx]);
  AccT sinVal = static_cast<AccT>(sinValues[csIdx]);

  LongType idx1, idx2;
  if (ropeType == 0) {  // Standard (LLaMA)
    idx1 = ((b * seqLen + s) * numHeads + h) * headDim + pairIdx;
    idx2 = ((b * seqLen + s) * numHeads + h) * headDim + pairIdx + halfDim;
  } else if (ropeType == 1) {  // NeoX
    idx1 = ((b * seqLen + s) * numHeads + h) * headDim + pairIdx * 2;
    idx2 = ((b * seqLen + s) * numHeads + h) * headDim + pairIdx * 2 + 1;
  } else {  // GPT-J
    idx1 = ((b * seqLen + s) * numHeads + h) * headDim + pairIdx;
    idx2 = ((b * seqLen + s) * numHeads + h) * headDim + pairIdx + halfDim;
  }

  AccT x1 = static_cast<AccT>(input[idx1]);
  AccT x2 = static_cast<AccT>(input[idx2]);

  output[idx1] = static_cast<T>(x1 * cosVal - x2 * sinVal);
  output[idx2] = static_cast<T>(x1 * sinVal + x2 * cosVal);
}

//////////////////////////////////////////////////////////////////////////////
// Fused Bias + Dropout + Residual Kernel
//////////////////////////////////////////////////////////////////////////////

template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedBiasDropoutResidualKernel(
    const T* __restrict__ input,
    const T* __restrict__ bias,
    const T* __restrict__ residual,
    T* __restrict__ output,
    const LongType totalElements,
    const LongType biasLen,
    const float dropoutProb,
    const LongType seed,
    const bool training) {

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  float val = static_cast<float>(input[idx]);

  // Add bias (broadcast along last dimension)
  if (bias != nullptr) {
    val += static_cast<float>(bias[idx % biasLen]);
  }

  // Apply dropout if training
  if (training && dropoutProb > 0.0f) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    float rand = curand_uniform(&state);
    if (rand < dropoutProb) {
      val = 0.0f;
    } else {
      val /= (1.0f - dropoutProb);
    }
  }

  // Add residual
  if (residual != nullptr) {
    val += static_cast<float>(residual[idx]);
  }

  output[idx] = static_cast<T>(val);
}

//////////////////////////////////////////////////////////////////////////////
// Launcher functions
//////////////////////////////////////////////////////////////////////////////

template <typename T>
void launchFusedGELU(
    const T* input,
    T* output,
    LongType totalElements,
    cudaStream_t stream) {

  int threadsPerBlock = 256;
  int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

  fusedGELUKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
      input, output, totalElements);
  DebugHelper::checkGlobalErrorCode("fusedGELUKernel failed");
}

template <typename T>
void launchFusedGELUBackward(
    const T* input,
    const T* gradOut,
    T* gradIn,
    LongType totalElements,
    cudaStream_t stream) {

  int threadsPerBlock = 256;
  int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

  fusedGELUBackwardKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
      input, gradOut, gradIn, totalElements);
  DebugHelper::checkGlobalErrorCode("fusedGELUBackwardKernel failed");
}

template <typename T>
void launchFusedLayerNorm(
    const T* input,
    const T* gain,
    const T* bias,
    T* output,
    LongType numRows,
    LongType rowLen,
    float epsilon,
    cudaStream_t stream) {

  // a power of two (the Welford merge halves the block) of at most 256 threads (the kernel's __launch_bounds__: a
  // longer row is strided): 512 or 1024 threads failed to launch, and a multiple of 32 such as 96 or 224 dropped
  // partial statistics in the merge
  int threadsPerBlock = WARP_SIZE;
  while (threadsPerBlock < 256 && threadsPerBlock < rowLen) threadsPerBlock *= 2;

  // 3 arrays of AccT per thread (mean, M2, count for Welford)
  size_t sharedMemSize = 3 * threadsPerBlock * sizeof(typename simdOps::AggregateType<T>::type);

  fusedLayerNormKernel<T><<<numRows, threadsPerBlock, sharedMemSize, stream>>>(
      input, gain, bias, output, numRows, rowLen, epsilon);
  if (!DebugHelper::inGraphCapture(&stream)) {
    DebugHelper::checkGlobalErrorCode("fusedLayerNormKernel failed");
  }
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm backward: one block per row for the input gradient, one thread per column for the gain and bias
// gradients
//////////////////////////////////////////////////////////////////////////////

// Each block takes rows blockIdx.x, blockIdx.x + gridDim.x, ...: the row's Welford statistics (merged as the forward
// kernel merges them), then sums of dnorm and dnorm * xhat (dnorm = dy * gain, xhat the normalized input), then
// dx = invStd * (dnorm - mean(dnorm) - xhat * mean(dnorm * xhat)). The row's mean and 1 / std go to stats for the
// column kernel. blockDim.x is a power of two (the merges halve the block).
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedLayerNormBackwardRowsKernel(
    const T* input, const T* gain, const T* gradOut, T* gradInput, typename simdOps::AggregateType<T>::type* stats,
    const LongType numRows, const LongType rowLen, const float epsilon) {
  using AccT = typename simdOps::AggregateType<T>::type;

  extern __shared__ char sharedMem[];
  AccT* sFirst = reinterpret_cast<AccT*>(sharedMem);
  AccT* sSecond = sFirst + blockDim.x;
  AccT* sCount = sFirst + 2 * blockDim.x;
  __shared__ AccT rowMean;
  __shared__ AccT rowInvStd;
  __shared__ AccT rowSumDnorm;
  __shared__ AccT rowSumDnormXhat;

  for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
    const T* x = input + row * rowLen;
    const T* dy = gradOut + row * rowLen;
    T* dx = gradInput + row * rowLen;

    AccT mean = static_cast<AccT>(0);
    AccT m2 = static_cast<AccT>(0);
    AccT count = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      const AccT value = static_cast<AccT>(x[i]);
      count += static_cast<AccT>(1);
      const AccT delta = value - mean;
      mean += delta / count;
      m2 += delta * (value - mean);
    }
    sFirst[threadIdx.x] = mean;
    sSecond[threadIdx.x] = m2;
    sCount[threadIdx.x] = count;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const AccT na = sCount[threadIdx.x];
        const AccT nb = sCount[threadIdx.x + s];
        const AccT nab = na + nb;
        if (nab > static_cast<AccT>(0)) {
          const AccT delta = sFirst[threadIdx.x + s] - sFirst[threadIdx.x];
          sFirst[threadIdx.x] = (na * sFirst[threadIdx.x] + nb * sFirst[threadIdx.x + s]) / nab;
          sSecond[threadIdx.x] = sSecond[threadIdx.x] + sSecond[threadIdx.x + s] + delta * delta * na * nb / nab;
          sCount[threadIdx.x] = nab;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      rowMean = sFirst[0];
      rowInvStd = static_cast<AccT>(1) /
                  sd::math::sd_sqrt<AccT, AccT>(sSecond[0] / sCount[0] + static_cast<AccT>(epsilon));
      stats[2 * row] = rowMean;
      stats[2 * row + 1] = rowInvStd;
    }
    __syncthreads();

    AccT sumDnorm = static_cast<AccT>(0);
    AccT sumDnormXhat = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      const AccT xhat = (static_cast<AccT>(x[i]) - rowMean) * rowInvStd;
      const AccT dnorm = static_cast<AccT>(dy[i]) * static_cast<AccT>(gain[i]);
      sumDnorm += dnorm;
      sumDnormXhat += dnorm * xhat;
    }
    sFirst[threadIdx.x] = sumDnorm;
    sSecond[threadIdx.x] = sumDnormXhat;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        sFirst[threadIdx.x] += sFirst[threadIdx.x + s];
        sSecond[threadIdx.x] += sSecond[threadIdx.x + s];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      rowSumDnorm = sFirst[0];
      rowSumDnormXhat = sSecond[0];
    }
    __syncthreads();

    const AccT n = static_cast<AccT>(rowLen);
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      const AccT xhat = (static_cast<AccT>(x[i]) - rowMean) * rowInvStd;
      const AccT dnorm = static_cast<AccT>(dy[i]) * static_cast<AccT>(gain[i]);
      dx[i] = static_cast<T>(rowInvStd * (dnorm - rowSumDnorm / n - xhat * rowSumDnormXhat / n));
    }
    // the shared row values and partials belong to the block's next row from here
    __syncthreads();
  }
}

// One thread per column: the gain gradient sums dy * xhat and the bias gradient dy over the rows, in row order (no
// atomics: the order, and so the result, is fixed).
template <typename T>
SD_KERNEL void fusedLayerNormBackwardColumnsKernel(
    const T* input, const T* gradOut, const typename simdOps::AggregateType<T>::type* stats, T* gradGain, T* gradBias,
    const LongType numRows, const LongType rowLen) {
  using AccT = typename simdOps::AggregateType<T>::type;
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < rowLen;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    AccT gainSum = static_cast<AccT>(0);
    AccT biasSum = static_cast<AccT>(0);
    for (LongType row = 0; row < numRows; row++) {
      const AccT dy = static_cast<AccT>(gradOut[row * rowLen + i]);
      gainSum += dy * (static_cast<AccT>(input[row * rowLen + i]) - stats[2 * row]) * stats[2 * row + 1];
      biasSum += dy;
    }
    gradGain[i] = static_cast<T>(gainSum);
    if (gradBias != nullptr) gradBias[i] = static_cast<T>(biasSum);
  }
}

template <typename T>
void launchFusedLayerNormBackward(const T* input, const T* gain, const T* gradOut, T* gradInput, T* gradGain,
                                  T* gradBias, typename simdOps::AggregateType<T>::type* stats, LongType numRows,
                                  LongType rowLen, float epsilon, cudaStream_t* stream) {
  using AccT = typename simdOps::AggregateType<T>::type;
  // the forward kernel's block: a power of two of at most 256 threads (__launch_bounds__)
  int threadsPerBlock = WARP_SIZE;
  while (threadsPerBlock < 256 && threadsPerBlock < rowLen) threadsPerBlock *= 2;
  const size_t sharedMemSize = 3 * threadsPerBlock * sizeof(AccT);
  fusedLayerNormBackwardRowsKernel<T><<<numRows, threadsPerBlock, sharedMemSize, *stream>>>(
      input, gain, gradOut, gradInput, stats, numRows, rowLen, epsilon);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("fusedLayerNormBackwardRowsKernel failed");
  }

  const int columnThreads = 256;
  const LongType columnBlocks = (rowLen + columnThreads - 1) / columnThreads;
  fusedLayerNormBackwardColumnsKernel<T><<<columnBlocks, columnThreads, 0, *stream>>>(
      input, gradOut, stats, gradGain, gradBias, numRows, rowLen);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("fusedLayerNormBackwardColumnsKernel failed");
  }
}

template <typename T, typename P>
void launchFusedRoPE(
    const T* input,
    T* output,
    LongType batch,
    LongType seqLen,
    LongType numHeads,
    LongType headDim,
    const P* positionPtr,
    float freqBase,
    float freqScale,
    int ropeType,
    cudaStream_t stream,
    int rotaryDims = 0) {

  LongType rotateDims = (rotaryDims > 0 && rotaryDims < headDim) ? rotaryDims : headDim;
  LongType totalPairs = batch * seqLen * numHeads * (rotateDims / 2);
  LongType totalElements = batch * seqLen * numHeads * headDim;

  // No pairs to rotate — just copy
  if (totalPairs == 0 || rotateDims < 2) {
    LongType totalBytes = totalElements * sizeof(T);
    cudaMemcpyAsync(output, input, totalBytes, cudaMemcpyDeviceToDevice, stream);
    return;
  }

  // When partial rotation, copy input to output first to preserve unrotated dims
  if (rotateDims < headDim) {
    LongType totalBytes = totalElements * sizeof(T);
    cudaMemcpyAsync(output, input, totalBytes, cudaMemcpyDeviceToDevice, stream);
  }

  int threadsPerBlock = 256;
  int numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

  fusedRoPEKernel<T, P><<<numBlocks, threadsPerBlock, 0, stream>>>(
      input, output, batch, seqLen, numHeads, headDim,
      positionPtr, freqBase, freqScale, ropeType, rotateDims);
  DebugHelper::checkGlobalErrorCode("fusedRoPEKernel failed");
}

template <typename T>
void launchFusedRoPEBackward(
    const T* gradOut,
    T* gradIn,
    LongType batch,
    LongType seqLen,
    LongType numHeads,
    LongType headDim,
    int positionOffset,
    float freqBase,
    float freqScale,
    int ropeType,
    cudaStream_t stream,
    int rotaryDims = 0) {

  LongType rotateDims = (rotaryDims > 0 && rotaryDims < headDim) ? rotaryDims : headDim;
  LongType totalPairs = batch * seqLen * numHeads * (rotateDims / 2);
  LongType totalElements = batch * seqLen * numHeads * headDim;

  if (totalPairs == 0 || rotateDims < 2) {
    LongType totalBytes = totalElements * sizeof(T);
    cudaMemcpyAsync(gradIn, gradOut, totalBytes, cudaMemcpyDeviceToDevice, stream);
    return;
  }

  // When partial rotation, copy gradOut to gradIn first to preserve unrotated dims
  if (rotateDims < headDim) {
    LongType totalBytes = totalElements * sizeof(T);
    cudaMemcpyAsync(gradIn, gradOut, totalBytes, cudaMemcpyDeviceToDevice, stream);
  }

  int threadsPerBlock = 256;
  int numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

  fusedRoPEBackwardKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
      gradOut, gradIn, batch, seqLen, numHeads, headDim,
      positionOffset, freqBase, freqScale, ropeType, rotateDims);
  DebugHelper::checkGlobalErrorCode("fusedRoPEBackwardKernel failed");
}

template <typename T>
void launchFusedBiasDropoutResidual(
    const T* input,
    const T* bias,
    const T* residual,
    T* output,
    LongType totalElements,
    LongType biasLen,
    float dropoutProb,
    LongType seed,
    bool training,
    cudaStream_t stream) {

  int threadsPerBlock = 256;
  int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

  fusedBiasDropoutResidualKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
      input, bias, residual, output, totalElements, biasLen,
      dropoutProb, seed, training);
  DebugHelper::checkGlobalErrorCode("fusedBiasDropoutResidualKernel failed");
}

// Explicit instantiations
template void launchFusedGELU<float>(const float*, float*, LongType, cudaStream_t);
template void launchFusedGELU<double>(const double*, double*, LongType, cudaStream_t);
template void launchFusedGELU<float16>(const float16*, float16*, LongType, cudaStream_t);

template void launchFusedGELUBackward<float>(const float*, const float*, float*, LongType, cudaStream_t);
template void launchFusedGELUBackward<double>(const double*, const double*, double*, LongType, cudaStream_t);
template void launchFusedGELUBackward<float16>(const float16*, const float16*, float16*, LongType, cudaStream_t);

template void launchFusedLayerNorm<float>(const float*, const float*, const float*, float*,
    LongType, LongType, float, cudaStream_t);
template void launchFusedLayerNorm<double>(const double*, const double*, const double*, double*,
    LongType, LongType, float, cudaStream_t);
template void launchFusedLayerNorm<float16>(const float16*, const float16*, const float16*, float16*,
    LongType, LongType, float, cudaStream_t);

// launchFusedRoPE: implicitly instantiated via BUILD_SINGLE_SELECTOR in fusedRoPE().

template void launchFusedRoPEBackward<float>(const float*, float*, LongType, LongType, LongType, LongType,
    int, float, float, int, cudaStream_t, int);
template void launchFusedRoPEBackward<double>(const double*, double*, LongType, LongType, LongType, LongType,
    int, float, float, int, cudaStream_t, int);
template void launchFusedRoPEBackward<float16>(const float16*, float16*, LongType, LongType, LongType, LongType,
    int, float, float, int, cudaStream_t, int);

template void launchFusedBiasDropoutResidual<float>(const float*, const float*, const float*, float*,
    LongType, LongType, float, LongType, bool, cudaStream_t);
template void launchFusedBiasDropoutResidual<double>(const double*, const double*, const double*, double*,
    LongType, LongType, float, LongType, bool, cudaStream_t);
template void launchFusedBiasDropoutResidual<float16>(const float16*, const float16*, const float16*, float16*,
    LongType, LongType, float, LongType, bool, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public API implementations
//////////////////////////////////////////////////////////////////////////////

void fusedGELU(NDArray* input, NDArray* output, LaunchContext* context) {
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  auto dtype = input->dataType();
  auto totalElements = input->lengthOf();

  if (dtype == DataType::FLOAT32) {
    launchFusedGELU<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<float*>(output->specialBuffer()),
        totalElements, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchFusedGELU<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<double*>(output->specialBuffer()),
        totalElements, *stream);
  } else if (dtype == DataType::HALF) {
    launchFusedGELU<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<float16*>(output->specialBuffer()),
        totalElements, *stream);
  } else {
    THROW_EXCEPTION("fusedGELU: Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input});
}

void fusedGELUBackward(NDArray* input, NDArray* gradOut, NDArray* gradIn, LaunchContext* context) {
  NDArray::prepareSpecialUse({gradIn}, {input, gradOut});
  auto stream = context->getCudaStream();
  auto dtype = input->dataType();
  auto totalElements = input->lengthOf();

  if (dtype == DataType::FLOAT32) {
    launchFusedGELUBackward<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float*>(gradOut->specialBuffer()),
        reinterpret_cast<float*>(gradIn->specialBuffer()),
        totalElements, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchFusedGELUBackward<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const double*>(gradOut->specialBuffer()),
        reinterpret_cast<double*>(gradIn->specialBuffer()),
        totalElements, *stream);
  } else if (dtype == DataType::HALF) {
    launchFusedGELUBackward<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(gradOut->specialBuffer()),
        reinterpret_cast<float16*>(gradIn->specialBuffer()),
        totalElements, *stream);
  } else {
    THROW_EXCEPTION("fusedGELUBackward: Unsupported data type");
  }

  NDArray::registerSpecialUse({gradIn}, {input, gradOut});
}

template <typename T>
static void fusedLayerNorm_(NDArray* input, NDArray* gain, NDArray* bias, NDArray* output, float epsilon,
                            LaunchContext* context) {
  const LongType rowLen = input->sizeAt(-1);
  const LongType numRows = input->lengthOf() / rowLen;
  launchFusedLayerNorm<T>(reinterpret_cast<const T*>(input->specialBuffer()),
                          reinterpret_cast<const T*>(gain->specialBuffer()),
                          bias != nullptr ? reinterpret_cast<const T*>(bias->specialBuffer()) : nullptr,
                          reinterpret_cast<T*>(output->specialBuffer()), numRows, rowLen, epsilon,
                          *context->getCudaStream());
}

void fusedLayerNorm(NDArray* originalInput, NDArray* originalGain, NDArray* originalBias, NDArray* originalOutput,
                    float epsilon, LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  // The kernel reads and writes dense C-order rows and vectors in the input's type: another layout (an F-ordered or
  // permuted array, a stepped view), or a gain or bias of another type (the op takes any float type for each), goes
  // through a copy. The strides decide whether a layout is dense row-major (a view's offset is already in the
  // buffer); the order flag and the element-wise stride do not.
  auto readable = [&](NDArray* a) -> NDArray* {
    if (a == nullptr) return nullptr;
    NDArray* typed = a->dataType() == dataType ? a : a->cast(dataType);
    if (shape::isDenseRowMajor(typed->shapeInfo())) return typed;
    NDArray* dense = typed->dup('c');
    if (typed != a) delete typed;
    return dense;
  };
  NDArray* input = readable(originalInput);
  NDArray* gain = readable(originalGain);
  NDArray* bias = readable(originalBias);
  // An output that is not dense, or not in the input's type, is written dense and assigned to the caller's array.
  std::vector<LongType> outputShape(originalOutput->shapeOf(), originalOutput->shapeOf() + originalOutput->rankOf());
  NDArray* output = originalOutput->dataType() == dataType && shape::isDenseRowMajor(originalOutput->shapeInfo())
                        ? originalOutput
                        : new NDArray('c', outputShape, dataType, context);

  NDArray::prepareSpecialUse({output}, {input, gain, bias});
  BUILD_SINGLE_SELECTOR(dataType, fusedLayerNorm_, (input, gain, bias, output, epsilon, context), SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {input, gain, bias});

  if (output != originalOutput) originalOutput->assign(output);
  const bool staged =
      input != originalInput || gain != originalGain || bias != originalBias || output != originalOutput;
  if (staged) {
    // the copies go once the stream is past the kernel and the copy back
    PointersManager(context, "fusedLayerNorm").synchronize();
    if (input != originalInput) delete input;
    if (gain != originalGain) delete gain;
    if (bias != originalBias) delete bias;
    if (output != originalOutput) delete output;
  }
}

template <typename T, typename P>
void fusedRoPE_(NDArray* input, NDArray* output, NDArray* positionArr,
                LongType batch, LongType seqLen, LongType numHeads, LongType headDim,
                float freqBase, float freqScale, int ropeType,
                cudaStream_t stream, int rotaryDims) {
  launchFusedRoPE<T, P>(
      reinterpret_cast<const T*>(input->specialBuffer()),
      reinterpret_cast<T*>(output->specialBuffer()),
      batch, seqLen, numHeads, headDim,
      reinterpret_cast<const P*>(positionArr->specialBuffer()),
      freqBase, freqScale, ropeType, stream, rotaryDims);
}

void fusedRoPE(NDArray* input, NDArray* output, NDArray* positionArr,
               float freqBase, float freqScale, int ropeType, LaunchContext* context,
               int rotaryDims) {
  const int rank = input->rankOf();
  auto batch = input->sizeAt(0);
  auto seqLen = input->sizeAt(1);
  auto numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);

  NDArray::prepareSpecialUse({output}, {input, positionArr});
  auto stream = context->getCudaStream();

  BUILD_DOUBLE_SELECTOR(input->dataType(), positionArr->dataType(), fusedRoPE_,
      (input, output, positionArr, batch, seqLen, numHeads, headDim,
       freqBase, freqScale, ropeType, *stream, rotaryDims), SD_FLOAT_TYPES, SD_COMMON_TYPES);

  NDArray::registerSpecialUse({output}, {input, positionArr});
}

void fusedRoPECached(NDArray* input, NDArray* cosValues, NDArray* sinValues,
                     NDArray* output, int ropeType, LaunchContext* context) {
  const int rank = input->rankOf();
  auto batch = input->sizeAt(0);
  auto seqLen = input->sizeAt(1);
  auto numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);

  // cos/sin can be 2D [S, halfDim], 3D [B, S, halfDim], or 4D [B, S, 1, halfDim]
  // Compute strides for the batch, seq, and halfDim dimensions
  auto cosRank = cosValues->rankOf();
  LongType cosStride0 = 0;  // batch stride
  LongType cosStride1 = 0;  // seq stride
  LongType cosStride2 = 1;  // halfDim stride (innermost)
  if (cosRank == 2) {
    // [S, halfDim] - no batch dim, broadcast across batch
    cosStride0 = 0;
    cosStride1 = cosValues->strideAt(0);
    cosStride2 = cosValues->strideAt(1);
  } else if (cosRank == 3) {
    // [B, S, halfDim]
    cosStride0 = cosValues->strideAt(0);
    cosStride1 = cosValues->strideAt(1);
    cosStride2 = cosValues->strideAt(2);
  } else if (cosRank == 4) {
    // [B, S, 1, halfDim] - skip the broadcast head dim
    cosStride0 = cosValues->strideAt(0);
    cosStride1 = cosValues->strideAt(1);
    cosStride2 = cosValues->strideAt(3);
  }

  auto stream = context->getCudaStream();
  auto dtype = input->dataType();

  LongType totalPairs = batch * seqLen * numHeads * (headDim / 2);

  // headDim < 2 means no pairs to rotate — RoPE is a no-op, just copy input
  if (totalPairs == 0) {
    NDArray::prepareSpecialUse({output}, {input, cosValues, sinValues});
    output->assign(input);
    NDArray::registerSpecialUse({output}, {input, cosValues, sinValues});
    return;
  }

  // The kernel uses separate template types for input (T) and cos/sin (CS), reading
  // cos/sin in their native dtype and casting to float in-register. This eliminates
  // temporary NDArray allocations from ->cast() which are unsafe during CUDA graph
  // capture (the cast allocates device memory + launches a transform kernel, and the
  // delete frees/nulls the buffer — on replay the baked-in kernel reads stale addresses).
  NDArray::prepareSpecialUse({output}, {input, cosValues, sinValues});

  dim3 launchDims = getLaunchDims("fusedRopeCached");
  int threadsPerBlock = launchDims.y;
  int numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

  auto csDtype = cosValues->dataType();

  // Dispatch: <T=input type, CS=cos/sin type>
  // The kernel reads cos/sin as CS and casts to float in-register (lines 360-361).
  if (dtype == DataType::FLOAT32 && csDtype == DataType::FLOAT32) {
    fusedRoPECachedKernel<float, float><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float*>(cosValues->specialBuffer()),
        reinterpret_cast<const float*>(sinValues->specialBuffer()),
        reinterpret_cast<float*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::FLOAT32 && csDtype == DataType::HALF) {
    fusedRoPECachedKernel<float, float16><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(cosValues->specialBuffer()),
        reinterpret_cast<const float16*>(sinValues->specialBuffer()),
        reinterpret_cast<float*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::HALF && csDtype == DataType::HALF) {
    fusedRoPECachedKernel<float16, float16><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(cosValues->specialBuffer()),
        reinterpret_cast<const float16*>(sinValues->specialBuffer()),
        reinterpret_cast<float16*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::HALF && csDtype == DataType::FLOAT32) {
    fusedRoPECachedKernel<float16, float><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<const float*>(cosValues->specialBuffer()),
        reinterpret_cast<const float*>(sinValues->specialBuffer()),
        reinterpret_cast<float16*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::DOUBLE && csDtype == DataType::DOUBLE) {
    fusedRoPECachedKernel<double, double><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const double*>(cosValues->specialBuffer()),
        reinterpret_cast<const double*>(sinValues->specialBuffer()),
        reinterpret_cast<double*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::DOUBLE && csDtype == DataType::FLOAT32) {
    fusedRoPECachedKernel<double, float><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const float*>(cosValues->specialBuffer()),
        reinterpret_cast<const float*>(sinValues->specialBuffer()),
        reinterpret_cast<double*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::DOUBLE && csDtype == DataType::HALF) {
    fusedRoPECachedKernel<double, float16><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(cosValues->specialBuffer()),
        reinterpret_cast<const float16*>(sinValues->specialBuffer()),
        reinterpret_cast<double*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else {
    THROW_EXCEPTION("fusedRoPECached: Unsupported data type combination");
  }

  DebugHelper::checkGlobalErrorCode("fusedRoPECachedKernel failed");
  NDArray::registerSpecialUse({output}, {input, cosValues, sinValues});
}

void fusedRoPEBackward(NDArray* gradOut, NDArray* gradIn, int positionOffset,
                       float freqBase, float freqScale, int ropeType, LaunchContext* context,
                       int rotaryDims) {
  const int rank = gradOut->rankOf();
  auto batch = gradOut->sizeAt(0);
  auto seqLen = gradOut->sizeAt(1);
  auto numHeads = (rank >= 4) ? gradOut->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? gradOut->sizeAt(3) : gradOut->sizeAt(2);

  NDArray::prepareSpecialUse({gradIn}, {gradOut});
  auto stream = context->getCudaStream();
  auto dtype = gradOut->dataType();

  if (dtype == DataType::FLOAT32) {
    launchFusedRoPEBackward<float>(
        reinterpret_cast<const float*>(gradOut->specialBuffer()),
        reinterpret_cast<float*>(gradIn->specialBuffer()),
        batch, seqLen, numHeads, headDim,
        positionOffset, freqBase, freqScale, ropeType, *stream, rotaryDims);
  } else if (dtype == DataType::DOUBLE) {
    launchFusedRoPEBackward<double>(
        reinterpret_cast<const double*>(gradOut->specialBuffer()),
        reinterpret_cast<double*>(gradIn->specialBuffer()),
        batch, seqLen, numHeads, headDim,
        positionOffset, freqBase, freqScale, ropeType, *stream, rotaryDims);
  } else if (dtype == DataType::HALF) {
    launchFusedRoPEBackward<float16>(
        reinterpret_cast<const float16*>(gradOut->specialBuffer()),
        reinterpret_cast<float16*>(gradIn->specialBuffer()),
        batch, seqLen, numHeads, headDim,
        positionOffset, freqBase, freqScale, ropeType, *stream, rotaryDims);
  } else {
    THROW_EXCEPTION("fusedRoPEBackward: Unsupported data type");
  }

  NDArray::registerSpecialUse({gradIn}, {gradOut});
}

void fusedBiasDropoutResidual(NDArray* input, NDArray* bias, NDArray* residual,
                              NDArray* output, float dropoutProb, LongType seed,
                              bool training, LaunchContext* context) {
  auto totalElements = input->lengthOf();
  auto biasLen = bias != nullptr ? bias->lengthOf() : 1;

  NDArray::prepareSpecialUse({output}, {input, bias, residual});
  auto stream = context->getCudaStream();
  auto dtype = input->dataType();

  if (dtype == DataType::FLOAT32) {
    launchFusedBiasDropoutResidual<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const float*>(bias->specialBuffer()) : nullptr,
        residual != nullptr ? reinterpret_cast<const float*>(residual->specialBuffer()) : nullptr,
        reinterpret_cast<float*>(output->specialBuffer()),
        totalElements, biasLen, dropoutProb, seed, training, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchFusedBiasDropoutResidual<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const double*>(bias->specialBuffer()) : nullptr,
        residual != nullptr ? reinterpret_cast<const double*>(residual->specialBuffer()) : nullptr,
        reinterpret_cast<double*>(output->specialBuffer()),
        totalElements, biasLen, dropoutProb, seed, training, *stream);
  } else if (dtype == DataType::HALF) {
    launchFusedBiasDropoutResidual<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const float16*>(bias->specialBuffer()) : nullptr,
        residual != nullptr ? reinterpret_cast<const float16*>(residual->specialBuffer()) : nullptr,
        reinterpret_cast<float16*>(output->specialBuffer()),
        totalElements, biasLen, dropoutProb, seed, training, *stream);
  } else {
    THROW_EXCEPTION("fusedBiasDropoutResidual: Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input, bias, residual});
}

//////////////////////////////////////////////////////////////////////////////
// Fused RMS Norm + SwiGLU
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL __launch_bounds__(512, 1) void rmsNormGammaKernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const LongType numRows,
    const LongType rowLen,
    const float epsilon) {

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char shmem[];
  AccT* sdata = reinterpret_cast<AccT*>(shmem);

  const T* inputRow = input + row * rowLen;
  T* outputRow = output + row * rowLen;

  // Compute sum of squares for RMS norm
  AccT sumSq = static_cast<AccT>(0);
  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    AccT val = static_cast<AccT>(inputRow[i]);
    sumSq += val * val;
  }

  // Block-level sum of squares (result on thread 0 -> sdata[0])
  AccT blockSumSq = sd::device::blockReduceSum(sumSq, sdata);
  if (threadIdx.x == 0) sdata[0] = blockSumSq;
  __syncthreads();

  // Compute RMS norm scale and apply gamma
  AccT rms = static_cast<AccT>(1) / sd::math::sd_sqrt<AccT, AccT>(
      sdata[0] / static_cast<AccT>(rowLen) + static_cast<AccT>(epsilon));

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    AccT val = static_cast<AccT>(inputRow[i]);
    AccT g = static_cast<AccT>(gamma[i]);
    outputRow[i] = static_cast<T>(val * rms * g);
  }
}

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void siluMultiplyKernel(
    T* __restrict__ output,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    const LongType totalElements) {

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  using AccT = typename simdOps::AggregateType<T>::type;
  AccT g = static_cast<AccT>(gate[idx]);
  AccT u = static_cast<AccT>(up[idx]);

  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  AccT siluG = g / (static_cast<AccT>(1) + sd::math::sd_exp<AccT, AccT>(-g));

  output[idx] = static_cast<T>(siluG * u);
}

void fusedRmsNormSwiGLU(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                        NDArray* output, float epsilon, LaunchContext* context) {
  // Fused RMS Norm + SwiGLU for LLaMA-style MLP
  // Computes: silu(rms_norm(x) @ W_gate) * (rms_norm(x) @ W_up)

  const auto batchSize = input->sizeAt(0);
  const auto seqLen = input->sizeAt(1);
  const auto hiddenDim = input->sizeAt(2);
  const auto intermediateDim = wGate->sizeAt(1);

  const LongType numRows = batchSize * seqLen;
  const LongType totalElements = numRows * intermediateDim;

  auto stream = context->getCudaStream();
  auto dtype = input->dataType();

  // Allocate temporary for normalized input
  std::vector<LongType> normShape = {batchSize, seqLen, hiddenDim};
  NDArray normalized('c', normShape, input->dataType(), context);

  // Step 1: RMS norm + gamma scaling
  // sharedMem must match the size of simdOps::AggregateType<T>::type (double when T=double, float otherwise)
  dim3 block(512);
  dim3 grid(static_cast<unsigned int>(numRows));
  size_t sharedMem = block.x * (dtype == DataType::DOUBLE ? sizeof(double) : sizeof(float));

  if (dtype == DataType::FLOAT32) {
    rmsNormGammaKernel<float><<<grid, block, sharedMem, *stream>>>(
        reinterpret_cast<float*>(normalized.specialBuffer()),
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float*>(gamma->specialBuffer()),
        numRows, hiddenDim, epsilon);
  } else if (dtype == DataType::DOUBLE) {
    rmsNormGammaKernel<double><<<grid, block, sharedMem, *stream>>>(
        reinterpret_cast<double*>(normalized.specialBuffer()),
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const double*>(gamma->specialBuffer()),
        numRows, hiddenDim, epsilon);
  } else if (dtype == DataType::HALF) {
    rmsNormGammaKernel<float16><<<grid, block, sharedMem, *stream>>>(
        reinterpret_cast<float16*>(normalized.specialBuffer()),
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(gamma->specialBuffer()),
        numRows, hiddenDim, epsilon);
  } else {
    THROW_EXCEPTION("fusedRmsNormSwiGLU: Unsupported data type");
  }
  DebugHelper::checkGlobalErrorCode("rmsNormGammaKernel failed");

  // Step 2: Matmul normalized @ W_gate -> gate
  std::vector<LongType> gateShape = {batchSize, seqLen, intermediateDim};
  NDArray gate('c', gateShape, input->dataType(), context);
  MmulHelper::mmul(&normalized, wGate, &gate);

  // Step 3: Matmul normalized @ W_up -> up
  std::vector<LongType> upShape = {batchSize, seqLen, intermediateDim};
  NDArray up('c', upShape, input->dataType(), context);
  MmulHelper::mmul(&normalized, wUp, &up);

  // Step 4: Fused SiLU(gate) * up -> output
  dim3 siluBlock(256);
  dim3 siluGrid((static_cast<unsigned long long>(totalElements) + siluBlock.x - 1) / siluBlock.x);

  if (dtype == DataType::FLOAT32) {
    siluMultiplyKernel<float><<<siluGrid, siluBlock, 0, *stream>>>(
        reinterpret_cast<float*>(output->specialBuffer()),
        reinterpret_cast<const float*>(gate.specialBuffer()),
        reinterpret_cast<const float*>(up.specialBuffer()),
        totalElements);
  } else if (dtype == DataType::DOUBLE) {
    siluMultiplyKernel<double><<<siluGrid, siluBlock, 0, *stream>>>(
        reinterpret_cast<double*>(output->specialBuffer()),
        reinterpret_cast<const double*>(gate.specialBuffer()),
        reinterpret_cast<const double*>(up.specialBuffer()),
        totalElements);
  } else if (dtype == DataType::HALF) {
    siluMultiplyKernel<float16><<<siluGrid, siluBlock, 0, *stream>>>(
        reinterpret_cast<float16*>(output->specialBuffer()),
        reinterpret_cast<const float16*>(gate.specialBuffer()),
        reinterpret_cast<const float16*>(up.specialBuffer()),
        totalElements);
  } else {
    THROW_EXCEPTION("fusedRmsNormSwiGLU: Unsupported data type");
  }
  DebugHelper::checkGlobalErrorCode("siluMultiplyKernel failed");

  NDArray::registerSpecialUse({output}, {&normalized, &gate, &up});
}

void fusedRmsNormSwiGLUBackward(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* gradOut, NDArray* gradInput, NDArray* gradGamma,
                                 NDArray* gradWGate, NDArray* gradWUp, float epsilon,
                                 LaunchContext* context) {
  THROW_EXCEPTION("fusedRmsNormSwiGLUBackward: Full kernel not yet implemented");
}

template <typename T>
static void fusedLayerNormBackward_(NDArray* input, NDArray* gain, NDArray* gradOut, NDArray* gradInput,
                                    NDArray* gradGain, NDArray* gradBias, float epsilon, LaunchContext* context) {
  using AccT = typename simdOps::AggregateType<T>::type;
  const LongType rowLen = input->sizeAt(-1);
  const LongType numRows = input->lengthOf() / rowLen;
  PointersManager manager(context, "fusedLayerNormBackward");
  // each row's mean and 1 / std; freed stream-ordered with the manager
  auto stats = reinterpret_cast<AccT*>(manager.allocateDevMem(2 * numRows * sizeof(AccT)));
  launchFusedLayerNormBackward<T>(
      reinterpret_cast<const T*>(input->specialBuffer()), reinterpret_cast<const T*>(gain->specialBuffer()),
      reinterpret_cast<const T*>(gradOut->specialBuffer()), reinterpret_cast<T*>(gradInput->specialBuffer()),
      reinterpret_cast<T*>(gradGain->specialBuffer()),
      gradBias != nullptr ? reinterpret_cast<T*>(gradBias->specialBuffer()) : nullptr, stats, numRows, rowLen,
      epsilon, context->getCudaStream());
}

void fusedLayerNormBackward(NDArray* originalInput, NDArray* originalGain, NDArray* originalGradOut,
                             NDArray* originalGradInput, NDArray* originalGradGain, NDArray* originalGradBias,
                             float epsilon, LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  // The kernels read and write dense C-order rows and vectors in the input's type: another layout (an F-ordered or
  // permuted array, a stepped view) or type goes through a copy, as in the forward pass (the gain and bias gradients
  // keep their parameters' types: they are written dense in the input's type and assigned back).
  auto readable = [&](NDArray* a) -> NDArray* {
    NDArray* typed = a->dataType() == dataType ? a : a->cast(dataType);
    if (shape::isDenseRowMajor(typed->shapeInfo())) return typed;
    NDArray* dense = typed->dup('c');
    if (typed != a) delete typed;
    return dense;
  };
  auto writable = [&](NDArray* a) -> NDArray* {
    if (a == nullptr || (a->dataType() == dataType && shape::isDenseRowMajor(a->shapeInfo()))) return a;
    std::vector<LongType> dims(a->shapeOf(), a->shapeOf() + a->rankOf());
    return new NDArray('c', dims, dataType, context);
  };
  NDArray* input = readable(originalInput);
  NDArray* gain = readable(originalGain);
  NDArray* gradOut = readable(originalGradOut);
  NDArray* gradInput = writable(originalGradInput);
  NDArray* gradGain = writable(originalGradGain);
  NDArray* gradBias = writable(originalGradBias);

  NDArray::prepareSpecialUse({gradInput, gradGain, gradBias}, {input, gain, gradOut});
  BUILD_SINGLE_SELECTOR(dataType, fusedLayerNormBackward_,
                        (input, gain, gradOut, gradInput, gradGain, gradBias, epsilon, context),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({gradInput, gradGain, gradBias}, {input, gain, gradOut});

  if (gradInput != originalGradInput) originalGradInput->assign(gradInput);
  if (gradGain != originalGradGain) originalGradGain->assign(gradGain);
  if (gradBias != originalGradBias) originalGradBias->assign(gradBias);
  const bool staged = input != originalInput || gain != originalGain || gradOut != originalGradOut ||
                      gradInput != originalGradInput || gradGain != originalGradGain ||
                      gradBias != originalGradBias;
  if (staged) {
    // the copies go once the stream is past the kernels and the copies back
    PointersManager(context, "fusedLayerNormBackward").synchronize();
    if (input != originalInput) delete input;
    if (gain != originalGain) delete gain;
    if (gradOut != originalGradOut) delete gradOut;
    if (gradInput != originalGradInput) delete gradInput;
    if (gradGain != originalGradGain) delete gradGain;
    if (gradBias != originalGradBias) delete gradBias;
  }
}

//////////////////////////////////////////////////////////////////////////////
// Fused bias-add kernel (applied after cuBLAS mmul)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void biasAddKernel(
    T* __restrict__ output,
    const T* __restrict__ bias,
    const LongType totalRows,
    const LongType outDim) {

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalRows * outDim) return;

  const LongType col = idx % outDim;
  output[idx] = static_cast<T>(
      static_cast<float>(output[idx]) + static_cast<float>(bias[col]));
}

//////////////////////////////////////////////////////////////////////////////
// Fused attention output projection
// output = reshape(attentionOutput, [B*S, hidden_dim]) @ Wo  [+ bias]
//////////////////////////////////////////////////////////////////////////////

void fusedAttentionProjection(NDArray* attentionOutput, NDArray* Wo, NDArray* bias,
                               NDArray* output, LaunchContext* context) {
  const int rank        = attentionOutput->rankOf();
  const LongType batch  = attentionOutput->sizeAt(0);
  const LongType seqLen = attentionOutput->sizeAt(1);

  LongType hiddenDim;
  if (rank == 4) {
    hiddenDim = attentionOutput->sizeAt(2) * attentionOutput->sizeAt(3);
  } else {
    hiddenDim = attentionOutput->sizeAt(rank - 1);
  }
  const LongType outDim = Wo->sizeAt(1);

  NDArray::prepareSpecialUse({output}, {attentionOutput, Wo, bias});

  // Step 1: reshape attention output to 2D [B*S, hidden_dim]
  // copyToNewBuff=false: create a view sharing the same DataBuffer.
  // This avoids allocating new device memory + launching a copy kernel,
  // which is unsafe during CUDA graph capture (baked-in addresses from temporary
  // allocations become stale on replay). reshape() verifies contiguity internally
  // and only copies when strides are incompatible.
  std::vector<LongType> flatShape = {batch * seqLen, hiddenDim};
  NDArray* attnFlat = attentionOutput->reshape('c', flatShape, false);

  // Step 2: reshape output to 2D [B*S, out_dim]
  std::vector<LongType> outFlat2D = {batch * seqLen, outDim};
  NDArray* outFlat = output->reshape('c', outFlat2D, false);

  // Step 3: cuBLAS-backed matmul
  MmulHelper::mmul(attnFlat, Wo, outFlat, 1.0, 0.0);

  delete attnFlat;
  delete outFlat;

  // Step 4: fused bias add if bias is provided
  if (bias != nullptr) {
    auto stream = context->getCudaStream();
    auto dtype  = output->dataType();
    const LongType totalRows = batch * seqLen;

    dim3 block(256);
    dim3 grid(static_cast<unsigned int>(
        (totalRows * outDim + block.x - 1) / block.x));

    if (dtype == DataType::FLOAT32) {
      biasAddKernel<float><<<grid, block, 0, *stream>>>(
          reinterpret_cast<float*>(output->specialBuffer()),
          reinterpret_cast<const float*>(bias->specialBuffer()),
          totalRows, outDim);
    } else if (dtype == DataType::DOUBLE) {
      biasAddKernel<double><<<grid, block, 0, *stream>>>(
          reinterpret_cast<double*>(output->specialBuffer()),
          reinterpret_cast<const double*>(bias->specialBuffer()),
          totalRows, outDim);
    } else if (dtype == DataType::HALF) {
      biasAddKernel<float16><<<grid, block, 0, *stream>>>(
          reinterpret_cast<float16*>(output->specialBuffer()),
          reinterpret_cast<const float16*>(bias->specialBuffer()),
          totalRows, outDim);
    } else {
      THROW_EXCEPTION("fusedAttentionProjection: Unsupported data type for bias add");
    }
    DebugHelper::checkGlobalErrorCode("biasAddKernel failed");
  }

  NDArray::registerSpecialUse({output}, {attentionOutput, Wo, bias});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
