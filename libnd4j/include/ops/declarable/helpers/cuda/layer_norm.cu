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
// Fused Layer Normalization CUDA kernel
// Computes: output = (input - mean) / sqrt(variance + epsilon) * gain + bias
// in a single fused kernel for optimal performance
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <system/common.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level reduction for sum
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Load a parameter as AccT precision (double when AccT=double, float otherwise)
template <typename AccT>
SD_DEVICE SD_INLINE AccT loadParamAs(const void* ptr, int dtype, LongType index) {
  if (ptr == nullptr) {
    return static_cast<AccT>(0);
  }
  if (dtype == static_cast<int>(DataType::FLOAT32)) {
    return static_cast<AccT>(reinterpret_cast<const float*>(ptr)[index]);
  }
  if (dtype == static_cast<int>(DataType::DOUBLE)) {
    return static_cast<AccT>(reinterpret_cast<const double*>(ptr)[index]);
  }
  if (dtype == static_cast<int>(DataType::HALF)) {
    return static_cast<AccT>(static_cast<float>(reinterpret_cast<const float16*>(ptr)[index]));
  }
  return static_cast<AccT>(0);
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduction for sum using shared memory
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE T blockReduceSum(T val, T* sharedMem) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  // Warp-level reduction
  val = warpReduceSum(val);

  // Write reduced value from each warp to shared memory
  if (lane == 0) {
    sharedMem[wid] = val;
  }
  __syncthreads();

  // First warp reduces across all warps
  val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : static_cast<T>(0);
  if (wid == 0) {
    val = warpReduceSum(val);
  }

  return val;
}

//////////////////////////////////////////////////////////////////////////////
// Fused Layer Norm Kernel - handles one row per block
// Each row is normalized independently
// Accumulator type: use double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void layerNormKernel(
    const T* __restrict__ input,    // [numRows, rowLen]
    const void* __restrict__ gain,  // [rowLen], may differ from T
    const void* __restrict__ bias,  // [rowLen] or nullptr, may differ from T
    T* __restrict__ output,         // [numRows, rowLen]
    const LongType numRows,
    const LongType rowLen,
    const int gainDtype,
    const int biasDtype,
    const float epsilon) {

  using AccT = typename AccType<T>::type;

  // Each block handles one row
  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char sharedMem[];
  AccT* sdata = reinterpret_cast<AccT*>(sharedMem);

  const T* inputRow = input + row * rowLen;
  T* outputRow = output + row * rowLen;

  // Pass 1: Compute sum and sum of squares in parallel
  AccT threadSum = static_cast<AccT>(0);
  AccT threadSumSq = static_cast<AccT>(0);

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    AccT val = static_cast<AccT>(inputRow[i]);
    threadSum += val;
    threadSumSq += val * val;
  }

  // Block-level reduction for sum
  AccT totalSum = blockReduceSum(threadSum, sdata);
  __syncthreads();

  // Block-level reduction for sum of squares
  AccT totalSumSq = blockReduceSum(threadSumSq, sdata);

  // Compute mean and inverse standard deviation
  __shared__ AccT mean;
  __shared__ AccT invStd;

  if (threadIdx.x == 0) {
    mean = totalSum / static_cast<AccT>(rowLen);
    AccT variance = (totalSumSq / static_cast<AccT>(rowLen)) - (mean * mean);
    // rsqrt: CUDA provides rsqrtf (float) and rsqrt (double)
    invStd = rsqrt(variance + static_cast<AccT>(epsilon));
  }
  __syncthreads();

  // Pass 2: Normalize, scale, and shift
  if (bias != nullptr) {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      AccT val = static_cast<AccT>(inputRow[i]);
      AccT normalized = (val - mean) * invStd;
      AccT g = loadParamAs<AccT>(gain, gainDtype, i);
      AccT b = loadParamAs<AccT>(bias, biasDtype, i);
      outputRow[i] = static_cast<T>(normalized * g + b);
    }
  } else {
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
      AccT val = static_cast<AccT>(inputRow[i]);
      AccT normalized = (val - mean) * invStd;
      AccT g = loadParamAs<AccT>(gain, gainDtype, i);
      outputRow[i] = static_cast<T>(normalized * g);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher function
//////////////////////////////////////////////////////////////////////////////
template <typename T>
void launchLayerNormKernel(
    const T* input,
    const void* gain,
    const void* bias,
    T* output,
    LongType numRows,
    LongType rowLen,
    int gainDtype,
    int biasDtype,
    float epsilon,
    cudaStream_t stream) {

  // One block per row, up to 256 or 512 threads per block
  int threadsPerBlock = 256;
  if (rowLen > 256) threadsPerBlock = 512;
  if (rowLen > 512) threadsPerBlock = 1024;

  // Limit to actual row length if smaller
  if (rowLen < threadsPerBlock) {
    threadsPerBlock = ((rowLen + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (threadsPerBlock < WARP_SIZE) threadsPerBlock = WARP_SIZE;
  }

  int numBlocks = numRows;

  // Shared memory for reductions (need space for warp results).
  // Use sizeof AccT: double when T=double, float otherwise.
  int numWarps = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
  size_t sharedMemSize = numWarps * sizeof(typename AccType<T>::type);

  layerNormKernel<T><<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
      input, gain, bias, output, numRows, rowLen, gainDtype, biasDtype, epsilon);

  DebugHelper::checkGlobalErrorCode("layerNormKernel failed");
}

// Explicit instantiations
template void launchLayerNormKernel<float>(
    const float*, const void*, const void*, float*,
    LongType, LongType, int, int, float, cudaStream_t);

template void launchLayerNormKernel<double>(
    const double*, const void*, const void*, double*,
    LongType, LongType, int, int, float, cudaStream_t);

template void launchLayerNormKernel<float16>(
    const float16*, const void*, const void*, float16*,
    LongType, LongType, int, int, float, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public interface called from layer_norm op
//////////////////////////////////////////////////////////////////////////////
void layerNorm(
    NDArray* input,
    NDArray* gain,
    NDArray* bias,
    NDArray* output,
    const std::vector<LongType>& axis,
    float epsilon,
    LaunchContext* context) {

  // This kernel handles the common case: normalizing over the last dimension
  // with contiguous data
  const int rank = input->rankOf();
  const bool lastDimNorm = (axis.size() == 1 && axis[0] == rank - 1);

  if (!lastDimNorm) {
    // Fall back to general implementation for non-last-dimension normalization
    THROW_EXCEPTION("layerNorm (CUDA): Only last dimension normalization is supported");
  }

  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen = input->sizeAt(-1);

  NDArray::prepareSpecialUse({output}, {input, gain, bias});

  auto stream = context->getCudaStream();
  auto dtype = input->dataType();
  auto gainDtype = gain->dataType();
  auto biasDtype = bias != nullptr ? bias->dataType() : gainDtype;

  if (dtype == DataType::FLOAT32) {
    launchLayerNormKernel<float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        gain->specialBuffer(),
        bias != nullptr ? bias->specialBuffer() : nullptr,
        reinterpret_cast<float*>(output->specialBuffer()),
        numRows, rowLen, static_cast<int>(gainDtype), static_cast<int>(biasDtype), epsilon, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchLayerNormKernel<double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        gain->specialBuffer(),
        bias != nullptr ? bias->specialBuffer() : nullptr,
        reinterpret_cast<double*>(output->specialBuffer()),
        numRows, rowLen, static_cast<int>(gainDtype), static_cast<int>(biasDtype), epsilon, *stream);
  } else if (dtype == DataType::HALF) {
    launchLayerNormKernel<float16>(
        reinterpret_cast<const float16*>(input->specialBuffer()),
        gain->specialBuffer(),
        bias != nullptr ? bias->specialBuffer() : nullptr,
        reinterpret_cast<float16*>(output->specialBuffer()),
        numRows, rowLen, static_cast<int>(gainDtype), static_cast<int>(biasDtype), epsilon, *stream);
  } else {
    THROW_EXCEPTION("layerNorm (CUDA): Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input, gain, bias});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
