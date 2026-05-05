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
#include <array/NDArrayFactory.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/MmulHelper.h>
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
template <typename T, typename G>
__global__ void rmsNormKernel(
    const T* __restrict__ input,    // [numRows, rowLen]
    const G* __restrict__ gamma,    // [rowLen] or nullptr (may differ from T)
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
template <typename T, typename G>
void launchRmsNormKernel(
    const T* input,
    const G* gamma,
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

  rmsNormKernel<T, G><<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
      input, gamma, output, numRows, rowLen, epsilon);

  DebugHelper::checkGlobalErrorCode("rmsNormKernel failed");
}

// Explicit instantiations — same-type gamma
template void launchRmsNormKernel<float, float>(
    const float*, const float*, float*,
    LongType, LongType, float, cudaStream_t);

template void launchRmsNormKernel<double, double>(
    const double*, const double*, double*,
    LongType, LongType, float, cudaStream_t);

template void launchRmsNormKernel<float16, float16>(
    const float16*, const float16*, float16*,
    LongType, LongType, float, cudaStream_t);

// Mixed-type gamma — F16 input with F32 gamma (common in transformer decode)
template void launchRmsNormKernel<float16, float>(
    const float16*, const float*, float16*,
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
  auto gammaDtype = gamma != nullptr ? gamma->dataType() : dtype;

  if (dtype == DataType::FLOAT32) {
    launchRmsNormKernel<float, float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        gamma != nullptr ? reinterpret_cast<const float*>(gamma->specialBuffer()) : nullptr,
        reinterpret_cast<float*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchRmsNormKernel<double, double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        gamma != nullptr ? reinterpret_cast<const double*>(gamma->specialBuffer()) : nullptr,
        reinterpret_cast<double*>(output->specialBuffer()),
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::HALF) {
    if (gamma != nullptr && gammaDtype == DataType::FLOAT32) {
      // Mixed-type: F16 input, F32 gamma — pass gamma directly without casting
      launchRmsNormKernel<float16, float>(
          reinterpret_cast<const float16*>(input->specialBuffer()),
          reinterpret_cast<const float*>(gamma->specialBuffer()),
          reinterpret_cast<float16*>(output->specialBuffer()),
          numRows, rowLen, epsilon, *stream);
    } else {
      launchRmsNormKernel<float16, float16>(
          reinterpret_cast<const float16*>(input->specialBuffer()),
          gamma != nullptr ? reinterpret_cast<const float16*>(gamma->specialBuffer()) : nullptr,
          reinterpret_cast<float16*>(output->specialBuffer()),
          numRows, rowLen, epsilon, *stream);
    }
  } else {
    THROW_EXCEPTION("rmsNormCuda: Unsupported data type");
  }

  NDArray::registerSpecialUse({output}, {input, gamma});
}

//////////////////////////////////////////////////////////////////////////////
// Fused Skip (Residual Add) + RMS Norm Kernel
// Combines: hidden = input + skip [+ bias], then RMS normalize hidden
// Saves one kernel launch per transformer layer by eliminating separate add.
//////////////////////////////////////////////////////////////////////////////
template <typename T, typename G>
__global__ void skipRmsNormKernel(
    const T* __restrict__ input,     // [numRows, rowLen]
    const T* __restrict__ skip,      // [numRows, rowLen]
    const G* __restrict__ gamma,     // [rowLen] (may differ from T)
    const T* __restrict__ bias,      // [rowLen] or nullptr
    T* __restrict__ output,          // [numRows, rowLen]
    T* __restrict__ hiddenOut,       // [numRows, rowLen] or nullptr
    const LongType numRows,
    const LongType rowLen,
    const float epsilon) {

  const LongType row = blockIdx.x;
  if (row >= numRows) return;

  extern __shared__ char sharedMem[];
  float* sdata = reinterpret_cast<float*>(sharedMem);

  const T* inputRow = input + row * rowLen;
  const T* skipRow = skip + row * rowLen;
  T* outputRow = output + row * rowLen;
  T* hiddenRow = hiddenOut != nullptr ? hiddenOut + row * rowLen : nullptr;

  // Pass 1: compute hidden = input + skip [+ bias], accumulate sum of squares
  float threadSumSq = 0.0f;

  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    float val = static_cast<float>(inputRow[i]) + static_cast<float>(skipRow[i]);
    if (bias != nullptr) val += static_cast<float>(bias[i]);
    if (hiddenRow != nullptr) hiddenRow[i] = static_cast<T>(val);
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

  // Pass 2: recompute hidden, normalize and scale
  for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
    float val = static_cast<float>(inputRow[i]) + static_cast<float>(skipRow[i]);
    if (bias != nullptr) val += static_cast<float>(bias[i]);
    float g = static_cast<float>(gamma[i]);
    outputRow[i] = static_cast<T>(val * invRms * g);
  }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher for skip_rms_norm
//////////////////////////////////////////////////////////////////////////////
template <typename T, typename G>
void launchSkipRmsNormKernel(
    const T* input,
    const T* skip,
    const G* gamma,
    const T* bias,
    T* output,
    T* hiddenOut,
    LongType numRows,
    LongType rowLen,
    float epsilon,
    cudaStream_t stream) {

  int threadsPerBlock = 256;
  if (rowLen > 256) threadsPerBlock = 512;
  if (rowLen > 512) threadsPerBlock = 1024;

  if (rowLen < threadsPerBlock) {
    threadsPerBlock = ((rowLen + RMS_WARP_SIZE - 1) / RMS_WARP_SIZE) * RMS_WARP_SIZE;
    if (threadsPerBlock < RMS_WARP_SIZE) threadsPerBlock = RMS_WARP_SIZE;
  }

  int numBlocks = numRows;
  int numWarps = (threadsPerBlock + RMS_WARP_SIZE - 1) / RMS_WARP_SIZE;
  size_t sharedMemSize = numWarps * sizeof(float);

  skipRmsNormKernel<T, G><<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
      input, skip, gamma, bias, output, hiddenOut, numRows, rowLen, epsilon);

  DebugHelper::checkGlobalErrorCode("skipRmsNormKernel failed");
}

// Same-type gamma instantiations
template void launchSkipRmsNormKernel<float, float>(
    const float*, const float*, const float*, const float*,
    float*, float*, LongType, LongType, float, cudaStream_t);

template void launchSkipRmsNormKernel<double, double>(
    const double*, const double*, const double*, const double*,
    double*, double*, LongType, LongType, float, cudaStream_t);

template void launchSkipRmsNormKernel<float16, float16>(
    const float16*, const float16*, const float16*, const float16*,
    float16*, float16*, LongType, LongType, float, cudaStream_t);

// Mixed-type gamma — F16 input with F32 gamma (common in transformer decode)
template void launchSkipRmsNormKernel<float16, float>(
    const float16*, const float16*, const float*, const float16*,
    float16*, float16*, LongType, LongType, float, cudaStream_t);

//////////////////////////////////////////////////////////////////////////////
// Public interface for skip_rms_norm
//////////////////////////////////////////////////////////////////////////////
void skipRmsNorm(
    LaunchContext* context,
    NDArray* input,
    NDArray* skip,
    NDArray* gamma,
    NDArray* bias,
    NDArray* output,
    NDArray* hiddenOut,
    float epsilon) {

  const LongType numRows = input->lengthOf() / input->sizeAt(-1);
  const LongType rowLen = input->sizeAt(-1);

  NDArray::prepareSpecialUse({output, hiddenOut}, {input, skip, gamma, bias});

  auto stream = context->getCudaStream();
  auto dtype = input->dataType();
  auto gammaDtype = gamma->dataType();

  if (dtype == DataType::FLOAT32) {
    launchSkipRmsNormKernel<float, float>(
        reinterpret_cast<const float*>(input->specialBuffer()),
        reinterpret_cast<const float*>(skip->specialBuffer()),
        reinterpret_cast<const float*>(gamma->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const float*>(bias->specialBuffer()) : nullptr,
        reinterpret_cast<float*>(output->specialBuffer()),
        hiddenOut != nullptr ? reinterpret_cast<float*>(hiddenOut->specialBuffer()) : nullptr,
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::DOUBLE) {
    launchSkipRmsNormKernel<double, double>(
        reinterpret_cast<const double*>(input->specialBuffer()),
        reinterpret_cast<const double*>(skip->specialBuffer()),
        reinterpret_cast<const double*>(gamma->specialBuffer()),
        bias != nullptr ? reinterpret_cast<const double*>(bias->specialBuffer()) : nullptr,
        reinterpret_cast<double*>(output->specialBuffer()),
        hiddenOut != nullptr ? reinterpret_cast<double*>(hiddenOut->specialBuffer()) : nullptr,
        numRows, rowLen, epsilon, *stream);
  } else if (dtype == DataType::HALF) {
    if (gammaDtype == DataType::FLOAT32) {
      // Mixed-type: F16 input, F32 gamma — pass gamma directly without casting
      launchSkipRmsNormKernel<float16, float>(
          reinterpret_cast<const float16*>(input->specialBuffer()),
          reinterpret_cast<const float16*>(skip->specialBuffer()),
          reinterpret_cast<const float*>(gamma->specialBuffer()),
          bias != nullptr ? reinterpret_cast<const float16*>(bias->specialBuffer()) : nullptr,
          reinterpret_cast<float16*>(output->specialBuffer()),
          hiddenOut != nullptr ? reinterpret_cast<float16*>(hiddenOut->specialBuffer()) : nullptr,
          numRows, rowLen, epsilon, *stream);
    } else {
      launchSkipRmsNormKernel<float16, float16>(
          reinterpret_cast<const float16*>(input->specialBuffer()),
          reinterpret_cast<const float16*>(skip->specialBuffer()),
          reinterpret_cast<const float16*>(gamma->specialBuffer()),
          bias != nullptr ? reinterpret_cast<const float16*>(bias->specialBuffer()) : nullptr,
          reinterpret_cast<float16*>(output->specialBuffer()),
          hiddenOut != nullptr ? reinterpret_cast<float16*>(hiddenOut->specialBuffer()) : nullptr,
          numRows, rowLen, epsilon, *stream);
    }
  } else {
    THROW_EXCEPTION("skipRmsNormCuda: Unsupported data type");
  }

  NDArray::registerSpecialUse({output, hiddenOut}, {input, skip, gamma, bias});
}

///////////////////////////////////////////////////////////////////////////////
// Fused RMSNorm + Linear kernel for M=1 (decode hot path)
//
// Single kernel computes output[j] = dot(x * invRMS * gamma, W[:, j])
// for all j in [0, N). Eliminates the intermediate normalized tensor.
//
// Phase 1: cooperative warp-reduce sum-of-squares to get invRMS
// Phase 2: write normalized x * gamma into shared memory
// Phase 3: each block computes dot products for its assigned output columns
///////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void rmsNormLinearFusedKernel(
    const T* __restrict__ x,       // [K]
    const T* __restrict__ gamma,   // [K] or nullptr
    const T* __restrict__ W,       // [K, N] with arbitrary strides
    T* __restrict__ output,        // [N]
    const LongType K,
    const LongType N,
    const LongType wStride0,       // stride along K dimension (row stride)
    const LongType wStride1,       // stride along N dimension (col stride)
    const float epsilon) {

  extern __shared__ char smem[];
  // Layout: [K floats for normalized x] [numWarps floats for reduction]
  float* normX = reinterpret_cast<float*>(smem);
  float* reduceSpace = normX + K;

  // Phase 1: compute invRMS via block reduction
  float threadSumSq = 0.0f;
  for (LongType i = threadIdx.x; i < K; i += blockDim.x) {
    float val = static_cast<float>(x[i]);
    threadSumSq += val * val;
  }

  float totalSumSq = rmsBlockReduceSum(threadSumSq, reduceSpace);

  __shared__ float invRms;
  if (threadIdx.x == 0) {
    invRms = rsqrtf(totalSumSq / static_cast<float>(K) + epsilon);
  }
  __syncthreads();

  // Phase 2: write normalized x * gamma to shared memory
  if (gamma != nullptr) {
    for (LongType i = threadIdx.x; i < K; i += blockDim.x) {
      normX[i] = static_cast<float>(x[i]) * invRms * static_cast<float>(gamma[i]);
    }
  } else {
    for (LongType i = threadIdx.x; i < K; i += blockDim.x) {
      normX[i] = static_cast<float>(x[i]) * invRms;
    }
  }
  __syncthreads();

  // Phase 3: each thread computes dot products for its assigned output columns
  // Grid-stride across output dimension N
  // Uses stride-based indexing so both C-contiguous [K,N] (strides [N,1]) and
  // transposed views (strides [1,K]) work without a per-step dup('c') copy.
  LongType jStart = blockIdx.x * blockDim.x + threadIdx.x;
  LongType jStride = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType j = jStart; j < N; j += jStride) {
    float acc = 0.0f;
    for (LongType k = 0; k < K; ++k) {
      acc += normX[k] * static_cast<float>(W[k * wStride0 + j * wStride1]);
    }
    output[j] = static_cast<T>(acc);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Typed launcher for fused M=1 path
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void rmsNormLinearFusedLauncher(
    const cudaStream_t* stream,
    const void* vX,
    const void* vGamma,
    const void* vW,
    void* vOutput,
    LongType K,
    LongType N,
    LongType wStride0,
    LongType wStride1,
    float epsilon) {

  auto x = reinterpret_cast<const T*>(vX);
  auto gamma = (vGamma != nullptr) ? reinterpret_cast<const T*>(vGamma) : nullptr;
  auto W = reinterpret_cast<const T*>(vW);
  auto output = reinterpret_cast<T*>(vOutput);

  dim3 launchDims = getLaunchDims("rms_norm_linear");
  // Shared memory: K floats for normX + warp reduction scratch
  int numWarps = (launchDims.y + RMS_WARP_SIZE - 1) / RMS_WARP_SIZE;
  size_t requiredSmem = (K + numWarps) * sizeof(float);
  // Use configured default or actual requirement, whichever is larger
  size_t sharedMemSize = requiredSmem > launchDims.z ? requiredSmem : launchDims.z;

  // Grid covers the N output columns
  auto gridSize = static_cast<int>((N + launchDims.y - 1) / launchDims.y);
  if (gridSize > static_cast<int>(launchDims.x))
    gridSize = launchDims.x;
  if (gridSize < 1) gridSize = 1;

  rmsNormLinearFusedKernel<T><<<gridSize, launchDims.y, sharedMemSize, *stream>>>(
      x, gamma, W, output, K, N, wStride0, wStride1, epsilon);

  DebugHelper::checkGlobalErrorCode("rmsNormLinearFusedKernel failed");
}

BUILD_SINGLE_TEMPLATE(void rmsNormLinearFusedLauncher,
                       (const cudaStream_t* stream,
                        const void* vX, const void* vGamma,
                        const void* vW, void* vOutput,
                        LongType K, LongType N,
                        LongType wStride0, LongType wStride1,
                        float epsilon),
                       SD_FLOAT_TYPES);

///////////////////////////////////////////////////////////////////////////////
// Typed launcher for general M>1 path: fused rmsNorm + MmulHelper
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void rmsNormLinearGeneralLauncher(
    LaunchContext* context,
    NDArray* input,
    NDArray* gamma,
    NDArray* weight,
    NDArray* output,
    float epsilon) {

  // Allocate temp for normalized input (same shape as input)
  auto shapeVec = *input->getShapeAsVector();
  NDArray normalized(input->ordering(), shapeVec, input->dataType(), context);

  // Step 1: fused rmsNorm kernel (single kernel launch)
  rmsNorm(context, input, gamma, &normalized, epsilon);

  // Step 2: matmul normalized @ weight -> output (single cuBLAS call)
  // Type mismatch is handled by the public rmsNormLinear() before calling this launcher.
  MmulHelper::mmul(&normalized, weight, output, 1.0, 0.0);
}

BUILD_SINGLE_TEMPLATE(void rmsNormLinearGeneralLauncher,
                       (LaunchContext* context,
                        NDArray* input, NDArray* gamma,
                        NDArray* weight, NDArray* output,
                        float epsilon),
                       SD_FLOAT_TYPES);

///////////////////////////////////////////////////////////////////////////////
// Public interface — uses BUILD_SINGLE_SELECTOR for runtime dispatch
///////////////////////////////////////////////////////////////////////////////
void rmsNormLinear(
    LaunchContext* context,
    NDArray* input,
    NDArray* gamma,
    NDArray* weight,
    NDArray* output,
    float epsilon) {

  const LongType K = input->sizeAt(-1);
  const LongType M = input->lengthOf() / K;
  const LongType N = weight->sizeAt(1);

  // Handle type mismatch: graph optimizer may strip casts between rms_norm and matmul,
  // leaving weight in a different dtype (e.g., FLOAT32 weight with HALF input).
  // Cast weight to match input so all paths get uniform types.
  NDArray* effWeight = const_cast<NDArray*>(weight);
  NDArray* castWeight = nullptr;
  if (weight->dataType() != input->dataType()) {
    castWeight = weight->cast(input->dataType());
    effWeight = castWeight;
  }

  // For M=1 (decode hot path): use fused single-kernel path
  // Requires K fits in shared memory (48KB = 12288 floats)
  // The fused kernel uses stride-based weight indexing, so it handles both
  // C-contiguous [K,N] (strides [N,1]) and transposed views (strides [1,K])
  // without needing a per-step dup('c') copy.
  if (M == 1 && K <= 8192) {
    NDArray::prepareSpecialUse({output}, {input, gamma, effWeight});

    auto stream = context->getCudaStream();
    LongType wStride0 = effWeight->strideAt(0);  // stride along K dimension
    LongType wStride1 = effWeight->strideAt(1);  // stride along N dimension

    BUILD_SINGLE_SELECTOR(input->dataType(), rmsNormLinearFusedLauncher,
                           (stream,
                            input->specialBuffer(),
                            gamma != nullptr ? gamma->specialBuffer() : nullptr,
                            effWeight->specialBuffer(),
                            output->specialBuffer(),
                            K, N, wStride0, wStride1, epsilon),
                           SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {input, gamma, effWeight});
    delete castWeight;
    return;
  }

  // General M>1 path: fused rmsNorm + cuBLAS GEMM (2 kernel launches)
  BUILD_SINGLE_SELECTOR(input->dataType(), rmsNormLinearGeneralLauncher,
                         (context, input, gamma, effWeight, output, epsilon),
                         SD_FLOAT_TYPES);
  delete castWeight;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
