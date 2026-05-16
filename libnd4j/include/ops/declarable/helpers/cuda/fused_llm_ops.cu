
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
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <types/float16.h>
#include <ops/declarable/helpers/fused_llm_ops.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Utility device functions
//////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* sharedMem) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0) {
    sharedMem[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : static_cast<T>(0);
  if (wid == 0) {
    val = warpReduceSum(val);
  }
  return val;
}

__device__ __forceinline__ float fastSigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu(float x) {
  return x * fastSigmoid(x);
}

//////////////////////////////////////////////////////////////////////////////
// Fused GELU Kernel - x * sigmoid(1.702 * x)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void fusedGELUKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const LongType totalElements) {

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  float x = static_cast<float>(input[idx]);
  // Fast GELU approximation: x * sigmoid(1.702 * x)
  float result = x * fastSigmoid(1.702f * x);
  output[idx] = static_cast<T>(result);
}

template <typename T>
static SD_KERNEL __launch_bounds__(256, 2) void fusedGELUBackwardKernel(
    const T* __restrict__ input,
    const T* __restrict__ gradOut,
    T* __restrict__ gradIn,
    const LongType totalElements) {

  const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= totalElements) return;

  float x = static_cast<float>(input[idx]);
  float dout = static_cast<float>(gradOut[idx]);

  // d/dx[x * sigmoid(1.702*x)] = sigmoid(1.702*x) + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
  float sig = fastSigmoid(1.702f * x);
  float grad = sig + x * 1.702f * sig * (1.0f - sig);
  gradIn[idx] = static_cast<T>(dout * grad);
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm Kernel with Welford's algorithm
//////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ __launch_bounds__(256, 2) void fusedLayerNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ gain,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const LongType numRows,
    const LongType rowLen,
    const float epsilon) {

  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char sharedMem[];
  float* sdata = reinterpret_cast<float*>(sharedMem);

  const T* inputRow = input + row * rowLen;
  T* outputRow = output + row * rowLen;

  // Welford's online algorithm for mean and variance
  float mean = 0.0f;
  float M2 = 0.0f;
  float count = 0.0f;

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    float val = static_cast<float>(inputRow[i]);
    count += 1.0f;
    float delta = val - mean;
    mean += delta / count;
    float delta2 = val - mean;
    M2 += delta * delta2;
  }

  // Parallel reduction for Welford's algorithm
  float* sMean = sdata;
  float* sM2 = sdata + blockDim.x;
  float* sCount = sdata + 2 * blockDim.x;

  sMean[threadIdx.x] = mean;
  sM2[threadIdx.x] = M2;
  sCount[threadIdx.x] = count;
  __syncthreads();

  // Combine Welford results in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      float na = sCount[threadIdx.x];
      float nb = sCount[threadIdx.x + s];
      float delta = sMean[threadIdx.x + s] - sMean[threadIdx.x];
      float nab = na + nb;
      if (nab > 0) {
        sMean[threadIdx.x] = (na * sMean[threadIdx.x] + nb * sMean[threadIdx.x + s]) / nab;
        sM2[threadIdx.x] = sM2[threadIdx.x] + sM2[threadIdx.x + s] + delta * delta * na * nb / nab;
        sCount[threadIdx.x] = nab;
      }
    }
    __syncthreads();
  }

  __shared__ float finalMean;
  __shared__ float finalInvStd;

  if (threadIdx.x == 0) {
    finalMean = sMean[0];
    float variance = sM2[0] / sCount[0];
    finalInvStd = rsqrtf(variance + epsilon);
  }
  __syncthreads();

  // Normalize, scale and shift
  if (bias != nullptr) {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      float val = static_cast<float>(inputRow[i]);
      float normalized = (val - finalMean) * finalInvStd;
      float g = static_cast<float>(gain[i]);
      float b = static_cast<float>(bias[i]);
      outputRow[i] = static_cast<T>(normalized * g + b);
    }
  } else {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      float val = static_cast<float>(inputRow[i]);
      float normalized = (val - finalMean) * finalInvStd;
      float g = static_cast<float>(gain[i]);
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

  // Read position from device pointer — capture-safe (no host sync).
  const LongType pos = static_cast<LongType>(positionPtr[0]) + s;

  // Compute theta using rotateDims for frequency spacing
  float theta = static_cast<float>(pos) * freqScale /
                powf(freqBase, (2.0f * pairIdx) / rotateDims);
  float cosTheta = cosf(theta);
  float sinTheta = sinf(theta);

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

  float x1 = static_cast<float>(input[idx1]);
  float x2 = static_cast<float>(input[idx2]);

  output[idx1] = static_cast<T>(x1 * cosTheta - x2 * sinTheta);
  output[idx2] = static_cast<T>(x1 * sinTheta + x2 * cosTheta);
}

template <typename T>
__global__ __launch_bounds__(256, 2) void fusedRoPEBackwardKernel(
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

  const LongType pos = positionOffset + s;

  float theta = static_cast<float>(pos) * freqScale /
                powf(freqBase, (2.0f * pairIdx) / rotateDims);
  float cosTheta = cosf(theta);
  float sinTheta = sinf(theta);

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

  float g1 = static_cast<float>(gradOut[idx1]);
  float g2 = static_cast<float>(gradOut[idx2]);

  // Inverse rotation
  gradIn[idx1] = static_cast<T>(g1 * cosTheta + g2 * sinTheta);
  gradIn[idx2] = static_cast<T>(-g1 * sinTheta + g2 * cosTheta);
}

//////////////////////////////////////////////////////////////////////////////
// Fused RoPE with pre-computed cos/sin (cached variant)
//////////////////////////////////////////////////////////////////////////////

template <typename T>
SD_KERNEL __launch_bounds__(256, 2) void fusedRoPECachedKernel(
    const T* __restrict__ input,
    const T* __restrict__ cosValues,
    const T* __restrict__ sinValues,
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

  const LongType pairIdx = idx % halfDim;
  LongType rem = idx / halfDim;
  const LongType h = rem % numHeads;
  rem /= numHeads;
  const LongType s = rem % seqLen;
  const LongType b = rem / seqLen;

  // Index into cos/sin using their actual strides (handles 2D, 3D, or 4D with broadcast)
  const LongType csIdx = b * cosStride0 + s * cosStride1 + pairIdx * cosStride2;
  float cosVal = static_cast<float>(cosValues[csIdx]);
  float sinVal = static_cast<float>(sinValues[csIdx]);

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

  float x1 = static_cast<float>(input[idx1]);
  float x2 = static_cast<float>(input[idx2]);

  output[idx1] = static_cast<T>(x1 * cosVal - x2 * sinVal);
  output[idx2] = static_cast<T>(x1 * sinVal + x2 * cosVal);
}

//////////////////////////////////////////////////////////////////////////////
// Fused Bias + Dropout + Residual Kernel
//////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ __launch_bounds__(256, 2) void fusedBiasDropoutResidualKernel(
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

  int threadsPerBlock = 256;
  if (rowLen > 256) threadsPerBlock = 512;
  if (rowLen > 512) threadsPerBlock = 1024;

  if (rowLen < threadsPerBlock) {
    threadsPerBlock = ((rowLen + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (threadsPerBlock < WARP_SIZE) threadsPerBlock = WARP_SIZE;
  }

  size_t sharedMemSize = 3 * threadsPerBlock * sizeof(float);

  fusedLayerNormKernel<T><<<numRows, threadsPerBlock, sharedMemSize, stream>>>(
      input, gain, bias, output, numRows, rowLen, epsilon);
  DebugHelper::checkGlobalErrorCode("fusedLayerNormKernel failed");
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

void fusedLayerNorm(NDArray* input, NDArray* gain, NDArray* bias, NDArray* output,
                    float epsilon, LaunchContext* context) {
  const int rank = input->rankOf();
  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen = input->sizeAt(-1);

  NDArray::prepareSpecialUse({output}, {input, gain, bias});
  auto stream = context->getCudaStream();
  auto dtype = input->dataType();

  if (dtype == DataType::FLOAT32) {
    launchFusedLayerNorm<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float*>(gain->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const float*>(bias->specialBuffer()) : nullptr,
        reinterpret_cast<float*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchFusedLayerNorm<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const double*>(gain->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const double*>(bias->specialBuffer()) : nullptr,
        reinterpret_cast<double*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::HALF) {
    launchFusedLayerNorm<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(gain->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const float16*>(bias->specialBuffer()) : nullptr,
        reinterpret_cast<float16*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else {
    THROW_EXCEPTION("fusedLayerNorm: Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input, gain, bias});
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

  // Cast cos/sin to match input dtype if they differ (e.g., cos/sin are FP16 but input is FP32).
  // The kernel uses a single template type T for all buffers — mismatched dtypes cause
  // reinterpret_cast to read garbage (e.g., two FP16 values combined into one FP32).
  NDArray* cosWork = cosValues;
  NDArray* sinWork = sinValues;
  NDArray* cosCast = nullptr;
  NDArray* sinCast = nullptr;
  if (cosValues->dataType() != dtype) {
    cosCast = cosValues->cast(dtype);
    cosWork = cosCast;
    // Recompute strides for the casted array (cast always produces contiguous)
    cosStride0 = 0;
    cosStride1 = 0;
    cosStride2 = 1;
    auto castRank = cosWork->rankOf();
    if (castRank == 2) {
      cosStride0 = 0;
      cosStride1 = cosWork->strideAt(0);
      cosStride2 = cosWork->strideAt(1);
    } else if (castRank == 3) {
      cosStride0 = cosWork->strideAt(0);
      cosStride1 = cosWork->strideAt(1);
      cosStride2 = cosWork->strideAt(2);
    } else if (castRank == 4) {
      cosStride0 = cosWork->strideAt(0);
      cosStride1 = cosWork->strideAt(1);
      cosStride2 = cosWork->strideAt(3);
    }
  }
  if (sinValues->dataType() != dtype) {
    sinCast = sinValues->cast(dtype);
    sinWork = sinCast;
  }

  NDArray::prepareSpecialUse({output}, {input, cosWork, sinWork});

  dim3 launchDims = getLaunchDims("fusedRopeCached");
  int threadsPerBlock = launchDims.y;
  int numBlocks = (totalPairs + threadsPerBlock - 1) / threadsPerBlock;

  if (dtype == DataType::FLOAT32) {
    fusedRoPECachedKernel<float><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float*>(cosWork->specialBuffer()),
        reinterpret_cast<const float*>(sinWork->specialBuffer()),
        reinterpret_cast<float*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::DOUBLE) {
    fusedRoPECachedKernel<double><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const double*>(cosWork->specialBuffer()),
        reinterpret_cast<const double*>(sinWork->specialBuffer()),
        reinterpret_cast<double*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else if (dtype == DataType::HALF) {
    fusedRoPECachedKernel<float16><<<numBlocks, threadsPerBlock, 0, *stream>>>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        reinterpret_cast<const float16*>(cosWork->specialBuffer()),
        reinterpret_cast<const float16*>(sinWork->specialBuffer()),
        reinterpret_cast<float16*>(output->specialBuffer()),
        batch, seqLen, numHeads, headDim, cosStride0, cosStride1, cosStride2, ropeType);
  } else {
    THROW_EXCEPTION("fusedRoPECached: Unsupported data type");
  }

  DebugHelper::checkGlobalErrorCode("fusedRoPECachedKernel failed");
  NDArray::registerSpecialUse({output}, {input, cosWork, sinWork});

  if (cosCast != nullptr) delete cosCast;
  if (sinCast != nullptr) delete sinCast;
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

  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char shmem[];
  float* sdata = reinterpret_cast<float*>(shmem);

  const T* inputRow = input + row * rowLen;
  T* outputRow = output + row * rowLen;

  // Compute sum of squares for RMS norm
  float sumSq = 0.0f;
  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    float val = static_cast<float>(inputRow[i]);
    sumSq += val * val;
  }

  // Warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sumSq += __shfl_down_sync(0xffffffff, sumSq, offset);
  }

  // Block-level reduction
  if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
    sdata[threadIdx.x / WARP_SIZE] = sumSq;
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
    sumSq = sdata[threadIdx.x];
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      sumSq += __shfl_down_sync(0xffffffff, sumSq, offset);
    }
    if (threadIdx.x == 0) {
      sdata[0] = sumSq;
    }
  }
  __syncthreads();

  // Compute RMS norm scale and apply gamma
  float rms = rsqrtf(sdata[0] / static_cast<float>(rowLen) + epsilon);

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    float val = static_cast<float>(inputRow[i]);
    float g = static_cast<float>(gamma[i]);
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

  float g = static_cast<float>(gate[idx]);
  float u = static_cast<float>(up[idx]);
  
  // SiLU(x) = x * sigmoid(x)
  float siluG = g / (1.0f + __expf(-g));
  
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
  dim3 block(512);
  dim3 grid(static_cast<unsigned int>(numRows));
  size_t sharedMem = block.x * sizeof(float);

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

void fusedLayerNormBackward(NDArray* input, NDArray* gain, NDArray* gradOut,
                             NDArray* gradInput, NDArray* gradGain, NDArray* gradBias,
                             float epsilon, LaunchContext* context) {
  THROW_EXCEPTION("fusedLayerNormBackward: Full kernel not yet implemented");
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
  std::vector<LongType> flatShape = {batch * seqLen, hiddenDim};
  NDArray* attnFlat = attentionOutput->reshape('c', flatShape);

  // Step 2: reshape output to 2D [B*S, out_dim]
  std::vector<LongType> outFlat2D = {batch * seqLen, outDim};
  NDArray* outFlat = output->reshape('c', outFlat2D);

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
