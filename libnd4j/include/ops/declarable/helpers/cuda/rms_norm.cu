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
// Fused RMS Normalization CUDA kernel
// Computes: output = input / sqrt(mean(input^2) + epsilon) * gamma
// in a single fused kernel for optimal performance.
//
// RMS norm is simpler than layer norm: no mean subtraction needed,
// only one reduction pass (sum of squares) instead of two (sum + sum of squares).
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <types/float16.h>
#include <ops/declarable/helpers/rms_norm.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int RMS_WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level reduction for sum
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ T rmsWarpReduceSum(T val) {
  for (int offset = RMS_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduction for sum using shared memory
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ T rmsBlockReduceSum(T val, T* sharedMem) {
  const int lane = threadIdx.x % RMS_WARP_SIZE;
  const int wid = threadIdx.x / RMS_WARP_SIZE;
  const int numWarps = (blockDim.x + RMS_WARP_SIZE - 1) / RMS_WARP_SIZE;

  // Warp-level reduction
  val = rmsWarpReduceSum(val);

  // Write reduced value from each warp to shared memory
  if (lane == 0) {
    sharedMem[wid] = val;
  }
  __syncthreads();

  // First warp reduces across all warps
  val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : static_cast<T>(0);
  if (wid == 0) {
    val = rmsWarpReduceSum(val);
  }

  return val;
}

//////////////////////////////////////////////////////////////////////////////
// Fused RMS Norm Kernel - handles one row per block
// Each row is normalized independently
//
// RMS norm formula:
//   rms = sqrt(mean(x^2) + eps)
//   output = (x / rms) * gamma
//
// Only ONE reduction pass needed (sum of squares), vs TWO for layer norm.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ void rmsNormKernel(
    const T* __restrict__ input,    // [numRows, rowLen]
    const T* __restrict__ gamma,    // [rowLen] or nullptr
    T* __restrict__ output,         // [numRows, rowLen]
    const LongType numRows,
    const LongType rowLen,
    const float epsilon) {

  // Each block handles one row
  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char sharedMem[];
  float* sdata = reinterpret_cast<float*>(sharedMem);

  const T* inputRow = input + row * rowLen;
  T* outputRow = output + row * rowLen;

  // Pass 1: Compute sum of squares in parallel
  float threadSumSq = 0.0f;

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    float val = static_cast<float>(inputRow[i]);
    threadSumSq += val * val;
  }

  // Block-level reduction for sum of squares
  float totalSumSq = rmsBlockReduceSum(threadSumSq, sdata);

  // Compute inverse RMS
  __shared__ float invRms;

  if (threadIdx.x == 0) {
    float meanSq = totalSumSq / static_cast<float>(rowLen);
    invRms = rsqrtf(meanSq + epsilon);
  }
  __syncthreads();

  // Pass 2: Normalize and scale
  if (gamma != nullptr) {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      float val = static_cast<float>(inputRow[i]);
      float g = static_cast<float>(gamma[i]);
      outputRow[i] = static_cast<T>(val * invRms * g);
    }
  } else {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      float val = static_cast<float>(inputRow[i]);
      outputRow[i] = static_cast<T>(val * invRms);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher function
//////////////////////////////////////////////////////////////////////////////
template <typename T>
void launchRmsNormKernel(
    const T* input,
    const T* gamma,
    T* output,
    LongType numRows,
    LongType rowLen,
    float epsilon,
    cudaStream_t stream) {

  // One block per row, scale threads to row length
  int threadsPerBlock = 256;
  if (rowLen > 256) threadsPerBlock = 512;
  if (rowLen > 512) threadsPerBlock = 1024;

  // Limit to actual row length if smaller
  if (rowLen < threadsPerBlock) {
    threadsPerBlock = ((rowLen + RMS_WARP_SIZE - 1) / RMS_WARP_SIZE) * RMS_WARP_SIZE;
    if (threadsPerBlock < RMS_WARP_SIZE) threadsPerBlock = RMS_WARP_SIZE;
  }

  int numBlocks = numRows;

  // Shared memory for reductions (need space for warp results)
  int numWarps = (threadsPerBlock + RMS_WARP_SIZE - 1) / RMS_WARP_SIZE;
  size_t sharedMemSize = numWarps * sizeof(float);

  rmsNormKernel<T><<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
      input, gamma, output, numRows, rowLen, epsilon);

  DebugHelper::checkGlobalErrorCode("rmsNormKernel failed");
}

// Explicit instantiations
template void launchRmsNormKernel<float>(
    const float*, const float*, float*,
    LongType, LongType, float, cudaStream_t);

template void launchRmsNormKernel<double>(
    const double*, const double*, double*,
    LongType, LongType, float, cudaStream_t);

template void launchRmsNormKernel<float16>(
    const float16*, const float16*, float16*,
    LongType, LongType, float, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public interface called from rms_norm op
//////////////////////////////////////////////////////////////////////////////
void rmsNorm(
    LaunchContext* context,
    NDArray* input,
    NDArray* gamma,
    NDArray* output,
    float epsilon) {

  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen = input->sizeAt(-1);

  NDArray::prepareSpecialUse({output}, {input, gamma});

  auto stream = context->getCudaStream();
  auto dtype = input->dataType();

  if (dtype == DataType::FLOAT32) {
    launchRmsNormKernel<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        gamma != nullptr ? reinterpret_cast<const float*>(gamma->specialBuffer()) : nullptr,
        reinterpret_cast<float*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchRmsNormKernel<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        gamma != nullptr ? reinterpret_cast<const double*>(gamma->specialBuffer()) : nullptr,
        reinterpret_cast<double*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::HALF) {
    launchRmsNormKernel<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        gamma != nullptr ? reinterpret_cast<const float16*>(gamma->specialBuffer()) : nullptr,
        reinterpret_cast<float16*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else {
    THROW_EXCEPTION("rmsNormCuda: Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input, gamma});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
