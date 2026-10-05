
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
// Operands of the kernels below. They index their operands as dense rows of one type, so an operand that is a stepped
// view, an F-ordered or permuted array, or of another type goes through a dense copy (the strides decide whether
// an array is dense row-major: the order flag does not, and a view's offset is already in specialBuffer()).
//////////////////////////////////////////////////////////////////////////////

// `a` as a dense row-major array of the given type: `a` itself when it is one, else a copy the caller retires with
// retireTemporary.
static NDArray* denseInType(NDArray* a, DataType dataType) {
  if (a == nullptr) return nullptr;
  NDArray* typed = a->dataType() == dataType ? a : a->cast(dataType);
  if (shape::isDenseRowMajor(typed->shapeInfo())) return typed;
  NDArray* dense = typed->dup('c');
  if (typed != a) MmulHelper::deleteTemporary(typed);
  return dense;
}

// The array a kernel writes in place of `a`: `a` itself when it is dense row-major and of the given type, else a dense
// temporary the caller assigns to `a` and retires with retireTemporary.
static NDArray* denseOutputInType(NDArray* a, DataType dataType, LaunchContext* context) {
  if (a == nullptr || (a->dataType() == dataType && shape::isDenseRowMajor(a->shapeInfo()))) return a;
  std::vector<LongType> dims(a->shapeOf(), a->shapeOf() + a->rankOf());
  return new NDArray('c', dims, dataType, context);
}

// `a` in the given type whatever its layout (a matmul operand: the matmuls deal with layouts themselves): `a` itself,
// or a cast the caller retires with retireTemporary.
static NDArray* asType(NDArray* a, DataType dataType) {
  return a == nullptr || a->dataType() == dataType ? a : a->cast(dataType);
}

// Retires a temporary of a call (an array that is not the caller's own) behind the stream that still reads it: the
// kernels and matmuls that consume it are asynchronous, and the pool must not recycle its storage before the last of
// them.
static void retireTemporary(NDArray* temporary, NDArray* original) {
  if (temporary != original) MmulHelper::deleteTemporary(temporary);
}

// The threads of a block that works on one row of rowLen elements: a power of two of at most 512, at least a warp
// (the block reductions need whole warps).
static int rowBlockThreads(LongType rowLen) {
  int threads = WARP_SIZE;
  while (threads < 512 && threads < rowLen) threads *= 2;
  return threads;
}

// The blocks that cover `items` work items of `perBlock` each with a grid-stride loop, in the range of a launch.
static unsigned int blocksFor(LongType items, int perBlock) {
  const LongType needed = (items + perBlock - 1) / perBlock;
  const LongType limit = 2147483647;
  return static_cast<unsigned int>(needed < limit ? (needed > 0 ? needed : 1) : limit);
}

//////////////////////////////////////////////////////////////////////////////
// Fused GELU Kernel - x * sigmoid(1.702 * x)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void fusedGELUKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const LongType totalElements) {

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
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

  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
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
  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
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
  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
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
    const LongType sinStride0,
    const LongType sinStride1,
    const LongType sinStride2,
    const int ropeType) {

  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
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

  // Index into cos and sin, each through its own strides (handles 2D, 3D, or 4D with broadcast, and a sin table that
  // is laid out differently from the cos table)
  AccT cosVal = static_cast<AccT>(cosValues[b * cosStride0 + s * cosStride1 + pairIdx * cosStride2]);
  AccT sinVal = static_cast<AccT>(sinValues[b * sinStride0 + s * sinStride1 + pairIdx * sinStride2]);

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

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  AccT val = static_cast<AccT>(input[idx]);

  // Add bias (broadcast along last dimension)
  if (bias != nullptr) {
    val += static_cast<AccT>(bias[idx % biasLen]);
  }

  // Apply dropout if training
  if (training && dropoutProb > 0.0f) {
    curandState state;
    curand_init(seed, idx, 0, &state);
    float rand = curand_uniform(&state);
    if (rand < dropoutProb) {
      val = static_cast<AccT>(0);
    } else {
      val /= static_cast<AccT>(1.0f - dropoutProb);
    }
  }

  // Add residual
  if (residual != nullptr) {
    val += static_cast<AccT>(residual[idx]);
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

template <typename T>
static void fusedGELU_(NDArray* input, NDArray* output, LaunchContext* context) {
  NDArray::prepareSpecialUse({output}, {input});
  auto stream = context->getCudaStream();
  launchFusedGELU<T>(
      reinterpret_cast<const T*>(input->specialBuffer()),
      reinterpret_cast<T*>(output->specialBuffer()),
      input->lengthOf(), *stream);
  NDArray::registerSpecialUse({output}, {input});
}

void fusedGELU(NDArray* originalInput, NDArray* originalOutput, LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  // The kernel pairs the elements of the input and the output by their offsets from the start of their buffers: both go
  // through dense row-major arrays of the input's type where they are not already (a stepped view, an F-ordered or
  // permuted array, an output of another type). In place on a dense array, the input and the output are one array.
  const auto dataType = originalInput->dataType();
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedGELU_, (input, output, context), SD_FLOAT_TYPES);

  if (output != originalOutput) originalOutput->assign(output);
  retireTemporary(input, originalInput);
  retireTemporary(output, originalOutput);
}

template <typename T>
static void fusedGELUBackward_(NDArray* input, NDArray* gradOut, NDArray* gradIn, LaunchContext* context) {
  NDArray::prepareSpecialUse({gradIn}, {input, gradOut});
  auto stream = context->getCudaStream();
  launchFusedGELUBackward<T>(
      reinterpret_cast<const T*>(input->specialBuffer()),
      reinterpret_cast<const T*>(gradOut->specialBuffer()),
      reinterpret_cast<T*>(gradIn->specialBuffer()),
      input->lengthOf(), *stream);
  NDArray::registerSpecialUse({gradIn}, {input, gradOut});
}

void fusedGELUBackward(NDArray* originalInput, NDArray* originalGradOut, NDArray* originalGradIn,
                       LaunchContext* context) {
  if (originalInput->lengthOf() == 0) return;
  // As the forward pass: dense row-major arrays of the input's type for the kernel.
  const auto dataType = originalInput->dataType();
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* gradOut = denseInType(originalGradOut, dataType);
  NDArray* gradIn = denseOutputInType(originalGradIn, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedGELUBackward_, (input, gradOut, gradIn, context), SD_FLOAT_TYPES);

  if (gradIn != originalGradIn) originalGradIn->assign(gradIn);
  retireTemporary(input, originalInput);
  retireTemporary(gradOut, originalGradOut);
  retireTemporary(gradIn, originalGradIn);
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
    // the copy above is still reading the cast
    if (typed != a) MmulHelper::deleteTemporary(typed);
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

void fusedRoPE(NDArray* originalInput, NDArray* originalOutput, NDArray* positionArr,
               float freqBase, float freqScale, int ropeType, LaunchContext* context,
               int rotaryDims) {
  // The kernel indexes the input and the output as dense row-major [batch, seq, heads, head_dim] arrays of the input's
  // type: any other layout (a stepped view, an F-ordered or permuted array) or an output of another type goes through
  // a dense copy.
  const auto dataType = originalInput->dataType();
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

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

  if (output != originalOutput) originalOutput->assign(output);
  retireTemporary(input, originalInput);
  retireTemporary(output, originalOutput);
}

// The launch of the cached RoPE kernel for an input of type T and cos/sin tables of type CS.
template <typename T, typename CS>
static void fusedRoPECachedLaunch_(NDArray* input, NDArray* cosValues, NDArray* sinValues, NDArray* output,
                                   LongType batch, LongType seqLen, LongType numHeads, LongType headDim,
                                   LongType cosStride0, LongType cosStride1, LongType cosStride2,
                                   LongType sinStride0, LongType sinStride1, LongType sinStride2, int ropeType,
                                   int numBlocks, int threadsPerBlock, cudaStream_t stream) {
  fusedRoPECachedKernel<T, CS><<<numBlocks, threadsPerBlock, 0, stream>>>(
      reinterpret_cast<const T*>(input->specialBuffer()),
      reinterpret_cast<const CS*>(cosValues->specialBuffer()),
      reinterpret_cast<const CS*>(sinValues->specialBuffer()),
      reinterpret_cast<T*>(output->specialBuffer()),
      batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, sinStride0, sinStride1, sinStride2,
      ropeType);
}

// The cached RoPE of dense row-major input and output arrays of one type (fusedRoPECached stages the caller's).
static void fusedRoPECachedDense(NDArray* input, NDArray* cosValues, NDArray* sinValues,
                                 NDArray* output, int ropeType, LaunchContext* context) {
  const int rank = input->rankOf();
  auto batch = input->sizeAt(0);
  auto seqLen = input->sizeAt(1);
  auto numHeads = (rank >= 4) ? input->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? input->sizeAt(3) : input->sizeAt(2);

  // cos/sin can be 2D [S, halfDim], 3D [B, S, halfDim], or 4D [B, S, 1, halfDim]
  // Compute strides for the batch, seq, and halfDim dimensions, of each table (the tables have one shape, and their
  // layouts may differ)
  auto tableStrides = [](NDArray* table, LongType& batchStride, LongType& seqStride, LongType& halfDimStride) {
    const int tableRank = table->rankOf();
    batchStride = 0;     // batch stride
    seqStride = 0;       // seq stride
    halfDimStride = 1;   // halfDim stride (innermost)
    if (tableRank == 2) {
      // [S, halfDim] - no batch dim, broadcast across batch
      seqStride = table->strideAt(0);
      halfDimStride = table->strideAt(1);
    } else if (tableRank == 3) {
      // [B, S, halfDim]
      batchStride = table->strideAt(0);
      seqStride = table->strideAt(1);
      halfDimStride = table->strideAt(2);
    } else if (tableRank == 4) {
      // [B, S, 1, halfDim] - skip the broadcast head dim
      batchStride = table->strideAt(0);
      seqStride = table->strideAt(1);
      halfDimStride = table->strideAt(3);
    }
  };
  LongType cosStride0, cosStride1, cosStride2, sinStride0, sinStride1, sinStride2;
  tableStrides(cosValues, cosStride0, cosStride1, cosStride2);
  tableStrides(sinValues, sinStride0, sinStride1, sinStride2);

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

  // Dispatch: <T = input type, CS = cos/sin type>, any pair of the float types (the BFLOAT16 pairs threw as an
  // unsupported combination, though the op takes it and the CPU helper rotates it). The kernel reads cos/sin as CS and
  // casts to its aggregate type in-register.
  BUILD_DOUBLE_SELECTOR(dtype, cosValues->dataType(), fusedRoPECachedLaunch_,
                        (input, cosValues, sinValues, output, batch, seqLen, numHeads, headDim, cosStride0,
                         cosStride1, cosStride2, sinStride0, sinStride1, sinStride2, ropeType, numBlocks,
                         threadsPerBlock, *stream),
                        SD_FLOAT_TYPES, SD_FLOAT_TYPES);

  DebugHelper::checkGlobalErrorCode("fusedRoPECachedKernel failed");
  NDArray::registerSpecialUse({output}, {input, cosValues, sinValues});
}

void fusedRoPECached(NDArray* originalInput, NDArray* cosValues, NDArray* originalSinValues,
                     NDArray* originalOutput, int ropeType, LaunchContext* context) {
  // The kernel indexes the input and the output as dense row-major arrays of the input's type (the cos and sin tables
  // are read through their own strides): another layout or an output of another type goes through a dense copy. It
  // reads both tables as one type, the cos table's: a sin table of another type is read in a copy of that type (the
  // kernel took its bytes for the cos table's type before).
  const auto dataType = originalInput->dataType();
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* sinValues = asType(originalSinValues, cosValues->dataType());
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  fusedRoPECachedDense(input, cosValues, sinValues, output, ropeType, context);

  if (output != originalOutput) originalOutput->assign(output);
  retireTemporary(input, originalInput);
  retireTemporary(sinValues, originalSinValues);
  retireTemporary(output, originalOutput);
}

template <typename T>
static void fusedRoPEBackward_(NDArray* gradOut, NDArray* gradIn, int positionOffset,
                               float freqBase, float freqScale, int ropeType, LaunchContext* context,
                               int rotaryDims) {
  const int rank = gradOut->rankOf();
  auto batch = gradOut->sizeAt(0);
  auto seqLen = gradOut->sizeAt(1);
  auto numHeads = (rank >= 4) ? gradOut->sizeAt(2) : static_cast<LongType>(1);
  auto headDim = (rank >= 4) ? gradOut->sizeAt(3) : gradOut->sizeAt(2);

  NDArray::prepareSpecialUse({gradIn}, {gradOut});
  auto stream = context->getCudaStream();

  launchFusedRoPEBackward<T>(
      reinterpret_cast<const T*>(gradOut->specialBuffer()),
      reinterpret_cast<T*>(gradIn->specialBuffer()),
      batch, seqLen, numHeads, headDim,
      positionOffset, freqBase, freqScale, ropeType, *stream, rotaryDims);

  NDArray::registerSpecialUse({gradIn}, {gradOut});
}

void fusedRoPEBackward(NDArray* originalGradOut, NDArray* originalGradIn, int positionOffset,
                       float freqBase, float freqScale, int ropeType, LaunchContext* context,
                       int rotaryDims) {
  // The kernel indexes the gradient and its result as dense row-major arrays of the gradient's type: another layout
  // or a result of another type goes through a dense copy.
  const auto dataType = originalGradOut->dataType();
  NDArray* gradOut = denseInType(originalGradOut, dataType);
  NDArray* gradIn = denseOutputInType(originalGradIn, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedRoPEBackward_,
                        (gradOut, gradIn, positionOffset, freqBase, freqScale, ropeType, context, rotaryDims),
                        SD_FLOAT_TYPES);

  if (gradIn != originalGradIn) originalGradIn->assign(gradIn);
  retireTemporary(gradOut, originalGradOut);
  retireTemporary(gradIn, originalGradIn);
}

template <typename T>
static void fusedBiasDropoutResidual_(NDArray* input, NDArray* bias, NDArray* residual,
                                      NDArray* output, float dropoutProb, LongType seed,
                                      bool training, LaunchContext* context) {
  auto totalElements = input->lengthOf();
  auto biasLen = bias != nullptr ? bias->lengthOf() : 1;

  NDArray::prepareSpecialUse({output}, {input, bias, residual});
  auto stream = context->getCudaStream();

  launchFusedBiasDropoutResidual<T>(
      reinterpret_cast<const T*>(input->specialBuffer()),
      bias != nullptr ? reinterpret_cast<const T*>(bias->specialBuffer()) : nullptr,
      residual != nullptr ? reinterpret_cast<const T*>(residual->specialBuffer()) : nullptr,
      reinterpret_cast<T*>(output->specialBuffer()),
      totalElements, biasLen, dropoutProb, seed, training, *stream);

  NDArray::registerSpecialUse({output}, {input, bias, residual});
}

void fusedBiasDropoutResidual(NDArray* originalInput, NDArray* originalBias, NDArray* originalResidual,
                              NDArray* originalOutput, float dropoutProb, LongType seed,
                              bool training, LaunchContext* context) {
  // The kernel indexes the input, the residual and the output as dense row-major arrays and the bias as a dense vector,
  // all of the input's type: another layout or type goes through a dense copy. An element's random draw depends on
  // its position in the dense order, so the result does not depend on the layout.
  const auto dataType = originalInput->dataType();
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* bias = denseInType(originalBias, dataType);
  NDArray* residual = denseInType(originalResidual, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedBiasDropoutResidual_,
                        (input, bias, residual, output, dropoutProb, seed, training, context), SD_FLOAT_TYPES);

  if (output != originalOutput) originalOutput->assign(output);
  retireTemporary(input, originalInput);
  retireTemporary(bias, originalBias);
  retireTemporary(residual, originalResidual);
  retireTemporary(output, originalOutput);
}

//////////////////////////////////////////////////////////////////////////////
// Fused RMS Norm + SwiGLU
//////////////////////////////////////////////////////////////////////////////

// The rows of input [numRows, rowLen] scaled by their root mean square and by gamma:
//   output = input * (1 / sqrt(mean(input^2) + epsilon)) * gamma
// and, when invRmsOut is given, each row's 1 / rms for the backward pass. One block per row (a grid-stride loop over
// the rows); blockDim.x is a multiple of the warp size.
template <typename T>
static SD_KERNEL __launch_bounds__(512, 1) void rmsNormGammaKernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    typename simdOps::AggregateType<T>::type* __restrict__ invRmsOut,
    const LongType numRows,
    const LongType rowLen,
    const float epsilon) {

  using AccT = typename simdOps::AggregateType<T>::type;

  extern __shared__ char shmem[];
  AccT* sdata = reinterpret_cast<AccT*>(shmem);

  for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
    const T* inputRow = input + row * rowLen;
    T* outputRow = output + row * rowLen;

    // Sum of squares of the row for RMS norm, on every thread (the reduction ends with a barrier: sdata is free for
    // the block's next row)
    AccT sumSq = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      const AccT val = static_cast<AccT>(inputRow[i]);
      sumSq += val * val;
    }
    const AccT total = sd::device::blockAllReduceSum<AccT>(sumSq, sdata);

    // Compute RMS norm scale and apply gamma
    const AccT rms = static_cast<AccT>(1) / sd::math::sd_sqrt<AccT, AccT>(
        total / static_cast<AccT>(rowLen) + static_cast<AccT>(epsilon));
    if (invRmsOut != nullptr && threadIdx.x == 0) invRmsOut[row] = rms;

    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      const AccT val = static_cast<AccT>(inputRow[i]);
      const AccT g = static_cast<AccT>(gamma[i]);
      outputRow[i] = static_cast<T>(val * rms * g);
    }
  }
}

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void siluMultiplyKernel(
    T* __restrict__ output,
    const T* __restrict__ gate,
    const T* __restrict__ up,
    const LongType totalElements) {

  using AccT = typename simdOps::AggregateType<T>::type;

  for (LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; idx < totalElements;
       idx += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const AccT g = static_cast<AccT>(gate[idx]);
    const AccT u = static_cast<AccT>(up[idx]);

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const AccT siluG = g / (static_cast<AccT>(1) + sd::math::sd_exp<AccT, AccT>(-g));

    output[idx] = static_cast<T>(siluG * u);
  }
}

// The gradients of the gate and up projections of y = silu(gate) * up, which replace gate and up (each element
// depends on its own values only), from the gradient of y:
//   s = sigmoid(gate)   dUp = dy * gate * s   dGate = dy * up * s * (1 + gate * (1 - s))
template <typename A>
static SD_KERNEL __launch_bounds__(256, 2) void swigluBackwardKernel(
    A* __restrict__ gateToGradient,
    A* __restrict__ upToGradient,
    const A* __restrict__ gradOut,
    const LongType totalElements) {

  for (LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; idx < totalElements;
       idx += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const A g = gateToGradient[idx];
    const A u = upToGradient[idx];
    const A dy = gradOut[idx];
    const A s = static_cast<A>(1) / (static_cast<A>(1) + sd::math::sd_exp<A, A>(-g));
    upToGradient[idx] = dy * g * s;
    gateToGradient[idx] = dy * u * s * (static_cast<A>(1) + g * (static_cast<A>(1) - s));
  }
}

// The gradient of the rows of the input of the RMS norm, from the gradient of their normalized rows: with
// dz = dNormalized * gamma, dx = invRms * (dz - x * invRms^2 * mean(dz * x)). One block per row (a grid-stride loop
// over the rows); blockDim.x is a multiple of the warp size.
template <typename A>
static SD_KERNEL __launch_bounds__(512, 1) void rmsNormBackwardRowsKernel(
    A* __restrict__ gradInput,
    const A* __restrict__ input,
    const A* __restrict__ gamma,
    const A* __restrict__ gradNormalized,
    const A* __restrict__ invRms,
    const LongType numRows,
    const LongType rowLen) {

  extern __shared__ char shmem[];
  A* scratch = reinterpret_cast<A*>(shmem);

  for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
    const A* x = input + row * rowLen;
    const A* dn = gradNormalized + row * rowLen;
    A* dx = gradInput + row * rowLen;
    const A inv = invRms[row];

    A partial = static_cast<A>(0);
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) partial += dn[i] * gamma[i] * x[i];
    // on every thread; the reduction ends with a barrier: scratch is free for the block's next row
    const A total = sd::device::blockAllReduceSum<A>(partial, scratch);

    const A coefficient = inv * inv * total / static_cast<A>(rowLen);
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) dx[i] = inv * (dn[i] * gamma[i] - x[i] * coefficient);
  }
}

// One thread per column: the gradient of gamma sums dNormalized * x * invRms over the rows, in row order (no atomics:
// the order, and so the result, is fixed).
template <typename A>
static SD_KERNEL void rmsNormGammaGradientKernel(
    A* __restrict__ gradGamma,
    const A* __restrict__ input,
    const A* __restrict__ gradNormalized,
    const A* __restrict__ invRms,
    const LongType numRows,
    const LongType rowLen) {

  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < rowLen;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    A sum = static_cast<A>(0);
    for (LongType row = 0; row < numRows; row++) {
      sum += gradNormalized[row * rowLen + i] * input[row * rowLen + i] * invRms[row];
    }
    gradGamma[i] = sum;
  }
}

// Fused RMS Norm + SwiGLU for LLaMA-style MLP
// Computes: silu(rms_norm(x) @ W_gate) * (rms_norm(x) @ W_up)
//
// input [batch, seq_len, hidden_dim], gamma [hidden_dim] and output [batch, seq_len, intermediate_dim] are dense
// row-major arrays of the input's type here; wGate and wUp [hidden_dim, intermediate_dim] go to the matmuls in whatever
// layout and type they have. The normalized rows, the gate and the up projection are stored in the input's type, and
// every sum and the SiLU in its aggregate type.
template <typename T>
static void fusedRmsNormSwiGLU_(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp, NDArray* output,
                                float epsilon, LaunchContext* context) {
  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType hiddenDim = input->sizeAt(2);
  const LongType intermediateDim = wGate->sizeAt(1);
  const LongType numRows = input->sizeAt(0) * input->sizeAt(1);
  const LongType totalElements = numRows * intermediateDim;
  const DataType dataType = input->dataType();

  auto stream = context->getCudaStream();

  // Allocate temporary for normalized input [numRows, hiddenDim]
  std::vector<LongType> normShape = {numRows, hiddenDim};
  NDArray* normalized = new NDArray('c', normShape, dataType, context);

  // Step 1: RMS norm + gamma scaling
  const int rowThreads = rowBlockThreads(hiddenDim);
  const size_t sharedMem = rowThreads * sizeof(AccT);
  NDArray::prepareSpecialUse({normalized}, {input, gamma});
  rmsNormGammaKernel<T><<<blocksFor(numRows, 1), rowThreads, sharedMem, *stream>>>(
      reinterpret_cast<T*>(normalized->specialBuffer()),
      reinterpret_cast<const T*>(input->specialBuffer()),
      reinterpret_cast<const T*>(gamma->specialBuffer()),
      nullptr, numRows, hiddenDim, epsilon);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("rmsNormGammaKernel failed");
  }
  NDArray::registerSpecialUse({normalized}, {input, gamma});

  // Steps 2 and 3: gate = normalized @ W_gate and up = normalized @ W_up, [numRows, intermediateDim] each
  std::vector<LongType> projectedShape = {numRows, intermediateDim};
  NDArray* gate = new NDArray('c', projectedShape, dataType, context);
  NDArray* up = new NDArray('c', projectedShape, dataType, context);
  MmulHelper::mmul(normalized, wGate, gate, 1.0, 0.0);
  MmulHelper::mmul(normalized, wUp, up, 1.0, 0.0);

  // Step 4: Fused SiLU(gate) * up -> output
  NDArray::prepareSpecialUse({output}, {gate, up});
  siluMultiplyKernel<T><<<blocksFor(totalElements, 256), 256, 0, *stream>>>(
      reinterpret_cast<T*>(output->specialBuffer()),
      reinterpret_cast<const T*>(gate->specialBuffer()),
      reinterpret_cast<const T*>(up->specialBuffer()),
      totalElements);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("siluMultiplyKernel failed");
  }
  NDArray::registerSpecialUse({output}, {gate, up});

  // the temporaries are still read by the kernels and matmuls above
  MmulHelper::deleteTemporary(normalized);
  MmulHelper::deleteTemporary(gate);
  MmulHelper::deleteTemporary(up);
}

void fusedRmsNormSwiGLU(NDArray* originalInput, NDArray* originalGamma, NDArray* originalWGate,
                        NDArray* originalWUp, NDArray* originalOutput, float epsilon, LaunchContext* context) {
  if (originalInput->lengthOf() == 0 || originalOutput->lengthOf() == 0) return;
  const auto dataType = originalInput->dataType();

  // The kernels read and write dense row-major arrays in the input's type (the op takes any float type for the gamma
  // and the output): another layout or type goes through a copy. The weights go to the matmuls as they are, whatever
  // their layout and type: MmulHelper::mmul handles both (it upcasts HALF weights of FLOAT activations through its
  // persistent cast cache, which a per-call copy here would bypass).
  NDArray* input = denseInType(originalInput, dataType);
  NDArray* gamma = denseInType(originalGamma, dataType);
  NDArray* output = denseOutputInType(originalOutput, dataType, context);

  BUILD_SINGLE_SELECTOR(dataType, fusedRmsNormSwiGLU_,
                        (input, gamma, originalWGate, originalWUp, output, epsilon, context), SD_FLOAT_TYPES);

  if (output != originalOutput) originalOutput->assign(output);
  retireTemporary(input, originalInput);
  retireTemporary(gamma, originalGamma);
  retireTemporary(output, originalOutput);
}

// Backward of the fused RMS norm + SwiGLU. The forward pass is recomputed from the input (normalized rows, gate and
// up projections) and the gradients follow in closed form: with y = silu(gate) * up, gate = n @ wGate, up = n @ wUp,
// n = x * invRms * gamma and dy the gradient of y,
//   dUp = dy * silu(gate)   dGate = dy * up * silu'(gate)
//   dWGate = n^T @ dGate    dWUp = n^T @ dUp    dn = dGate @ wGate^T + dUp @ wUp^T
//   dGamma = sum over rows of dn * x * invRms    dx = invRms * (dn * gamma - x * invRms^2 * mean(dn * gamma * x))
// Everything is computed in the aggregate type of the input's (float for HALF and BFLOAT16, the type itself
// otherwise): the operands are dense copies in that type where they are not already, and each gradient is rounded to
// its own type once, as it is assigned back.
template <typename T>
static void fusedRmsNormSwiGLUBackward_(NDArray* originalInput, NDArray* originalGamma, NDArray* originalWGate,
                                        NDArray* originalWUp, NDArray* originalGradOut, NDArray* originalGradInput,
                                        NDArray* originalGradGamma, NDArray* originalGradWGate,
                                        NDArray* originalGradWUp, float epsilon, LaunchContext* context) {
  using AccT = typename simdOps::AggregateType<T>::type;
  const DataType accType = DataTypeUtils::fromT<AccT>();

  const LongType hiddenDim = originalInput->sizeAt(2);
  const LongType intermediateDim = originalWGate->sizeAt(1);
  const LongType numRows = originalInput->sizeAt(0) * originalInput->sizeAt(1);
  const LongType totalElements = numRows * intermediateDim;

  auto stream = context->getCudaStream();

  NDArray* input = denseInType(originalInput, accType);
  NDArray* gamma = denseInType(originalGamma, accType);
  NDArray* gradOut = denseInType(originalGradOut, accType);
  NDArray* wGate = asType(originalWGate, accType);
  NDArray* wUp = asType(originalWUp, accType);
  NDArray* gradInput = denseOutputInType(originalGradInput, accType, context);
  NDArray* gradGamma = denseOutputInType(originalGradGamma, accType, context);
  NDArray* gradWGate = denseOutputInType(originalGradWGate, accType, context);
  NDArray* gradWUp = denseOutputInType(originalGradWUp, accType, context);

  std::vector<LongType> rowsShape = {numRows, hiddenDim};
  std::vector<LongType> projectedShape = {numRows, intermediateDim};
  std::vector<LongType> invRmsShape = {numRows};
  NDArray* normalized = new NDArray('c', rowsShape, accType, context);
  NDArray* invRms = new NDArray('c', invRmsShape, accType, context);
  NDArray* gate = new NDArray('c', projectedShape, accType, context);
  NDArray* up = new NDArray('c', projectedShape, accType, context);
  NDArray* gradNormalized = new NDArray('c', rowsShape, accType, context);

  // the normalized rows and each row's 1 / rms
  const int rowThreads = rowBlockThreads(hiddenDim);
  const size_t sharedMem = rowThreads * sizeof(AccT);
  NDArray::prepareSpecialUse({normalized, invRms}, {input, gamma});
  rmsNormGammaKernel<AccT><<<blocksFor(numRows, 1), rowThreads, sharedMem, *stream>>>(
      reinterpret_cast<AccT*>(normalized->specialBuffer()),
      reinterpret_cast<const AccT*>(input->specialBuffer()),
      reinterpret_cast<const AccT*>(gamma->specialBuffer()),
      reinterpret_cast<AccT*>(invRms->specialBuffer()),
      numRows, hiddenDim, epsilon);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("rmsNormGammaKernel failed");
  }
  NDArray::registerSpecialUse({normalized, invRms}, {input, gamma});

  // gate = normalized @ W_gate, up = normalized @ W_up
  MmulHelper::mmul(normalized, wGate, gate, 1.0, 0.0);
  MmulHelper::mmul(normalized, wUp, up, 1.0, 0.0);

  // gate and up become dGate and dUp
  NDArray::prepareSpecialUse({gate, up}, {gate, up, gradOut});
  swigluBackwardKernel<AccT><<<blocksFor(totalElements, 256), 256, 0, *stream>>>(
      reinterpret_cast<AccT*>(gate->specialBuffer()),
      reinterpret_cast<AccT*>(up->specialBuffer()),
      reinterpret_cast<const AccT*>(gradOut->specialBuffer()),
      totalElements);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("swigluBackwardKernel failed");
  }
  NDArray::registerSpecialUse({gate, up}, {gradOut});

  // dWGate = normalized^T @ dGate, dWUp = normalized^T @ dUp
  MmulHelper::matmul(normalized, gate, gradWGate, true, false, 1.0, 0.0);
  MmulHelper::matmul(normalized, up, gradWUp, true, false, 1.0, 0.0);

  // dn = dGate @ W_gate^T + dUp @ W_up^T
  MmulHelper::matmul(gate, wGate, gradNormalized, false, true, 1.0, 0.0);
  MmulHelper::matmul(up, wUp, gradNormalized, false, true, 1.0, 1.0);

  // dx and dGamma
  NDArray::prepareSpecialUse({gradInput, gradGamma}, {input, gamma, gradNormalized, invRms});
  rmsNormBackwardRowsKernel<AccT><<<blocksFor(numRows, 1), rowThreads, sharedMem, *stream>>>(
      reinterpret_cast<AccT*>(gradInput->specialBuffer()),
      reinterpret_cast<const AccT*>(input->specialBuffer()),
      reinterpret_cast<const AccT*>(gamma->specialBuffer()),
      reinterpret_cast<const AccT*>(gradNormalized->specialBuffer()),
      reinterpret_cast<const AccT*>(invRms->specialBuffer()),
      numRows, hiddenDim);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("rmsNormBackwardRowsKernel failed");
  }
  rmsNormGammaGradientKernel<AccT><<<blocksFor(hiddenDim, 256), 256, 0, *stream>>>(
      reinterpret_cast<AccT*>(gradGamma->specialBuffer()),
      reinterpret_cast<const AccT*>(input->specialBuffer()),
      reinterpret_cast<const AccT*>(gradNormalized->specialBuffer()),
      reinterpret_cast<const AccT*>(invRms->specialBuffer()),
      numRows, hiddenDim);
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode("rmsNormGammaGradientKernel failed");
  }
  NDArray::registerSpecialUse({gradInput, gradGamma}, {input, gamma, gradNormalized, invRms});

  // the gradients that were computed in a dense copy, in the input's aggregate type, go to their own arrays
  if (gradInput != originalGradInput) originalGradInput->assign(gradInput);
  if (gradGamma != originalGradGamma) originalGradGamma->assign(gradGamma);
  if (gradWGate != originalGradWGate) originalGradWGate->assign(gradWGate);
  if (gradWUp != originalGradWUp) originalGradWUp->assign(gradWUp);

  // everything above is still being read by kernels and matmuls on the stream
  MmulHelper::deleteTemporary(normalized);
  MmulHelper::deleteTemporary(invRms);
  MmulHelper::deleteTemporary(gate);
  MmulHelper::deleteTemporary(up);
  MmulHelper::deleteTemporary(gradNormalized);
  retireTemporary(input, originalInput);
  retireTemporary(gamma, originalGamma);
  retireTemporary(gradOut, originalGradOut);
  retireTemporary(wGate, originalWGate);
  retireTemporary(wUp, originalWUp);
  retireTemporary(gradInput, originalGradInput);
  retireTemporary(gradGamma, originalGradGamma);
  retireTemporary(gradWGate, originalGradWGate);
  retireTemporary(gradWUp, originalGradWUp);
}

void fusedRmsNormSwiGLUBackward(NDArray* input, NDArray* gamma, NDArray* wGate, NDArray* wUp,
                                 NDArray* gradOut, NDArray* gradInput, NDArray* gradGamma,
                                 NDArray* gradWGate, NDArray* gradWUp, float epsilon,
                                 LaunchContext* context) {
  if (input->lengthOf() == 0 || gradOut->lengthOf() == 0) return;
  BUILD_SINGLE_SELECTOR(input->dataType(), fusedRmsNormSwiGLUBackward_,
                        (input, gamma, wGate, wUp, gradOut, gradInput, gradGamma, gradWGate, gradWUp, epsilon,
                         context),
                        SD_FLOAT_TYPES);
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
    // the copy above is still reading the cast
    if (typed != a) MmulHelper::deleteTemporary(typed);
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

  using AccT = typename simdOps::AggregateType<T>::type;

  const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= totalRows * outDim) return;

  const LongType col = idx % outDim;
  output[idx] = static_cast<T>(static_cast<AccT>(output[idx]) + static_cast<AccT>(bias[col]));
}

// Adds the bias [outDim] to every row of the dense output [totalRows, outDim], both of type T.
template <typename T>
static void biasAdd_(NDArray* output, NDArray* bias, LongType totalRows, LongType outDim, LaunchContext* context) {
  auto stream = context->getCudaStream();
  biasAddKernel<T><<<blocksFor(totalRows * outDim, 256), 256, 0, *stream>>>(
      reinterpret_cast<T*>(output->specialBuffer()),
      reinterpret_cast<const T*>(bias->specialBuffer()),
      totalRows, outDim);
  DebugHelper::checkGlobalErrorCode("biasAddKernel failed");
}

//////////////////////////////////////////////////////////////////////////////
// Fused attention output projection
// output = reshape(attentionOutput, [B*S, hidden_dim]) @ Wo  [+ bias]
//////////////////////////////////////////////////////////////////////////////

void fusedAttentionProjection(NDArray* originalAttentionOutput, NDArray* Wo, NDArray* originalBias,
                               NDArray* originalOutput, LaunchContext* context) {
  // The reshapes below are views of dense arrays. The reshape of a layout that is not dense returns a copy instead: of
  // the attention output, whose delete is not retired behind the matmul that reads it, and of the output, in which the
  // product would be lost. The bias kernel indexes the output and the bias as dense arrays of the output's type. So
  // an attention output or an output of any other layout (a stepped view, an F-ordered or permuted array), and a bias of
  // another layout or type, go through dense copies; arrays that are dense already (the usual case) are used as they are.
  NDArray* attentionOutput = denseInType(originalAttentionOutput, originalAttentionOutput->dataType());
  NDArray* output = denseOutputInType(originalOutput, originalOutput->dataType(), context);
  NDArray* bias = denseInType(originalBias, originalOutput->dataType());

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
  // allocations become stale on replay). The attention output is dense here, so this is a view.
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
    BUILD_SINGLE_SELECTOR(output->dataType(), biasAdd_, (output, bias, batch * seqLen, outDim, context),
                          SD_FLOAT_TYPES);
  }

  NDArray::registerSpecialUse({output}, {attentionOutput, Wo, bias});

  if (output != originalOutput) originalOutput->assign(output);
  retireTemporary(attentionOutput, originalAttentionOutput);
  retireTemporary(bias, originalBias);
  retireTemporary(output, originalOutput);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
