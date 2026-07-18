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
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
// CUDA implementations for mixed precision training helpers
//

#include <ops/declarable/helpers/mixedPrecision.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <cuda_runtime.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// scaleGradients: elementwise multiply every element by scale
// ---------------------------------------------------------------------------

template <typename T>
SD_KERNEL void scaleGradientsKernel(T* buf, LongType length, T scale) {
  for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += blockDim.x * gridDim.x) {
    buf[idx] = buf[idx] * scale;
  }
}

template <typename T>
static void scaleGradientsTyped(sd::LaunchContext* context, NDArray* gradients, double scale) {
  auto stream = context->getCudaStream();
  auto length = gradients->lengthOf();

  NDArray::prepareSpecialUse({gradients}, {gradients});

  dim3 dims = getLaunchDims("updater");
  scaleGradientsKernel<T><<<dims.y, dims.x, dims.z, *stream>>>(
      reinterpret_cast<T*>(gradients->specialBuffer()), length, static_cast<T>(scale));
  DebugHelper::checkErrorCode(stream, "scaleGradientsKernel");

  NDArray::registerSpecialUse({gradients}, {gradients});
}

BUILD_SINGLE_TEMPLATE(void scaleGradientsTyped,
                      (sd::LaunchContext* context, NDArray* gradients, double scale),
                      SD_FLOAT_TYPES);

void scaleGradients(sd::LaunchContext* context, NDArray* gradients, double scale) {
  if (gradients == nullptr || gradients->isEmpty()) return;
  BUILD_SINGLE_SELECTOR(gradients->dataType(), scaleGradientsTyped, (context, gradients, scale), SD_FLOAT_TYPES);
}

// ---------------------------------------------------------------------------
// hasInfOrNan: returns true if any element is inf or nan.
// Uses an int flag reduced atomically; result copied back to host.
// ---------------------------------------------------------------------------

template <typename T>
SD_KERNEL void hasInfOrNanKernel(const T* buf, LongType length, int* flag) {
  for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += blockDim.x * gridDim.x) {
    if (sd::math::sd_isnan<T>(buf[idx]) || sd::math::sd_isinf<T>(buf[idx])) {
      atomicOr(flag, 1);
      return;
    }
  }
}

template <typename T>
static bool hasInfOrNanTyped(sd::LaunchContext* context, NDArray* arr) {
  auto stream = context->getCudaStream();
  auto length = arr->lengthOf();

  NDArray::prepareSpecialUse({}, {const_cast<NDArray*>(arr)});

  int* dFlag = nullptr;
  cudaMalloc(&dFlag, sizeof(int));
  cudaMemsetAsync(dFlag, 0, sizeof(int), *stream);

  dim3 dims = getLaunchDims("updater");
  hasInfOrNanKernel<T><<<dims.y, dims.x, dims.z, *stream>>>(
      reinterpret_cast<const T*>(arr->specialBuffer()), length, dFlag);
  DebugHelper::checkErrorCode(stream, "hasInfOrNanKernel");

  int hFlag = 0;
  cudaMemcpyAsync(&hFlag, dFlag, sizeof(int), cudaMemcpyDeviceToHost, *stream);
  cudaStreamSynchronize(*stream);
  cudaFree(dFlag);

  NDArray::registerSpecialUse({}, {const_cast<NDArray*>(arr)});
  return hFlag != 0;
}

BUILD_SINGLE_TEMPLATE(bool hasInfOrNanTyped,
                      (sd::LaunchContext* context, NDArray* arr),
                      SD_FLOAT_TYPES);

bool hasInfOrNan(sd::LaunchContext* context, const NDArray* arr) {
  auto arrNonConst = const_cast<NDArray*>(arr);
  if (arr == nullptr || arrNonConst->isEmpty()) return false;
  BUILD_SINGLE_SELECTOR(arrNonConst->dataType(), return hasInfOrNanTyped, (context, arrNonConst), SD_FLOAT_TYPES);
}

// ---------------------------------------------------------------------------
// castAndScale: cast input to output type and multiply by scale in one pass.
// Handles same-type and cross-type (e.g. FP16 compute -> FP32 cast + scale).
// ---------------------------------------------------------------------------

template <typename Tsrc, typename Tdst>
SD_KERNEL void castAndScaleKernel(const Tsrc* src, Tdst* dst, LongType length, double scale) {
  for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += blockDim.x * gridDim.x) {
    dst[idx] = static_cast<Tdst>(static_cast<double>(src[idx]) * scale);
  }
}

template <typename Tsrc, typename Tdst>
static void castAndScaleImpl(sd::LaunchContext* context, NDArray* input, NDArray* output, double scale) {
  auto stream = context->getCudaStream();
  auto length = input->lengthOf();

  NDArray::prepareSpecialUse({output}, {const_cast<NDArray*>(input)});

  dim3 dims = getLaunchDims("updater");
  castAndScaleKernel<Tsrc, Tdst><<<dims.y, dims.x, dims.z, *stream>>>(
      reinterpret_cast<const Tsrc*>(input->specialBuffer()),
      reinterpret_cast<Tdst*>(output->specialBuffer()),
      length, scale);
  DebugHelper::checkErrorCode(stream, "castAndScaleKernel");

  NDArray::registerSpecialUse({output}, {const_cast<NDArray*>(input)});
}

BUILD_DOUBLE_TEMPLATE(void castAndScaleImpl,
                      (sd::LaunchContext* context, NDArray* input, NDArray* output, double scale),
                      SD_FLOAT_TYPES, SD_FLOAT_TYPES);

void castAndScale(sd::LaunchContext* context, const NDArray* input, NDArray* output, double scale) {
  if (input == nullptr || output == nullptr) return;
  auto inputNonConst = const_cast<NDArray*>(input);
  if (inputNonConst->isEmpty()) return;

  BUILD_DOUBLE_SELECTOR(inputNonConst->dataType(), output->dataType(), castAndScaleImpl,
                        (context, inputNonConst, output, scale),
                        SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}

// ---------------------------------------------------------------------------
// unscaleGradientsAndCheck: divide each element by lossScale and write back,
// setting the overflow flag if any result is inf or nan.
// Returns true if all elements are finite after scaling (valid), false on overflow.
// Matches CPU semantics: overflow-detected elements are NOT written back.
// ---------------------------------------------------------------------------

template <typename T>
SD_KERNEL void unscaleAndCheckKernel(T* buf, LongType length, T invScale, int* flag) {
  for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += blockDim.x * gridDim.x) {
    T val = buf[idx] * invScale;
    if (sd::math::sd_isnan<T>(val) || sd::math::sd_isinf<T>(val)) {
      atomicOr(flag, 1);
    } else {
      buf[idx] = val;
    }
  }
}

template <typename T>
static bool unscaleAndCheckTyped(sd::LaunchContext* context, NDArray* gradients, double lossScale) {
  auto stream = context->getCudaStream();
  auto length = gradients->lengthOf();

  if (lossScale == 1.0) {
    return !hasInfOrNanTyped<T>(context, gradients);
  }

  NDArray::prepareSpecialUse({gradients}, {gradients});

  int* dFlag = nullptr;
  cudaMalloc(&dFlag, sizeof(int));
  cudaMemsetAsync(dFlag, 0, sizeof(int), *stream);

  T invScale = static_cast<T>(1.0 / lossScale);
  dim3 dims = getLaunchDims("updater");
  unscaleAndCheckKernel<T><<<dims.y, dims.x, dims.z, *stream>>>(
      reinterpret_cast<T*>(gradients->specialBuffer()), length, invScale, dFlag);
  DebugHelper::checkErrorCode(stream, "unscaleAndCheckKernel");

  int hFlag = 0;
  cudaMemcpyAsync(&hFlag, dFlag, sizeof(int), cudaMemcpyDeviceToHost, *stream);
  cudaStreamSynchronize(*stream);
  cudaFree(dFlag);

  NDArray::registerSpecialUse({gradients}, {gradients});

  return hFlag == 0;
}

BUILD_SINGLE_TEMPLATE(bool unscaleAndCheckTyped,
                      (sd::LaunchContext* context, NDArray* gradients, double lossScale),
                      SD_FLOAT_TYPES);

bool unscaleGradientsAndCheck(sd::LaunchContext* context, NDArray* gradients, double lossScale) {
  if (gradients == nullptr || gradients->isEmpty()) return true;
  BUILD_SINGLE_SELECTOR(gradients->dataType(), return unscaleAndCheckTyped, (context, gradients, lossScale), SD_FLOAT_TYPES);
}

// ---------------------------------------------------------------------------
// updateMasterWeight: masterWeight -= (learningRate / lossScale) * gradient
// masterWeight is typically FP32; gradient may be FP16/BF16/FP32.
// ---------------------------------------------------------------------------

template <typename Tmaster, typename Tgrad>
SD_KERNEL void updateMasterWeightKernel(Tmaster* master, const Tgrad* grad, LongType length, double scaledLr) {
  for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += blockDim.x * gridDim.x) {
    master[idx] = master[idx] - static_cast<Tmaster>(static_cast<double>(grad[idx]) * scaledLr);
  }
}

template <typename Tmaster, typename Tgrad>
static void updateMasterWeightImpl(sd::LaunchContext* context, NDArray* masterWeight,
                                   NDArray* gradient, double learningRate, double lossScale) {
  auto stream = context->getCudaStream();
  auto length = masterWeight->lengthOf();
  double scaledLr = learningRate / lossScale;

  NDArray::prepareSpecialUse({masterWeight}, {masterWeight, const_cast<NDArray*>(gradient)});

  dim3 dims = getLaunchDims("updater");
  updateMasterWeightKernel<Tmaster, Tgrad><<<dims.y, dims.x, dims.z, *stream>>>(
      reinterpret_cast<Tmaster*>(masterWeight->specialBuffer()),
      reinterpret_cast<const Tgrad*>(gradient->specialBuffer()),
      length, scaledLr);
  DebugHelper::checkErrorCode(stream, "updateMasterWeightKernel");

  NDArray::registerSpecialUse({masterWeight}, {masterWeight, const_cast<NDArray*>(gradient)});
}

BUILD_DOUBLE_TEMPLATE(void updateMasterWeightImpl,
                      (sd::LaunchContext* context, NDArray* masterWeight,
                       NDArray* gradient, double learningRate, double lossScale),
                      SD_FLOAT_TYPES, SD_FLOAT_TYPES);

void updateMasterWeight(sd::LaunchContext* context, NDArray* masterWeight,
                        const NDArray* gradient, double learningRate, double lossScale) {
  if (masterWeight == nullptr || gradient == nullptr) return;
  auto gradNonConst = const_cast<NDArray*>(gradient);
  if (masterWeight->isEmpty() || gradNonConst->isEmpty()) return;

  BUILD_DOUBLE_SELECTOR(masterWeight->dataType(), gradNonConst->dataType(), updateMasterWeightImpl,
                        (context, masterWeight, gradNonConst, learningRate, lossScale),
                        SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}

// ---------------------------------------------------------------------------
// syncMasterToCompute: copy master weights to compute weights with type cast.
// Typically FP32 master -> FP16/BF16 compute copy.
// ---------------------------------------------------------------------------

template <typename Tsrc, typename Tdst>
SD_KERNEL void syncMasterToComputeKernel(const Tsrc* src, Tdst* dst, LongType length) {
  for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x; idx < length; idx += blockDim.x * gridDim.x) {
    dst[idx] = static_cast<Tdst>(src[idx]);
  }
}

template <typename Tsrc, typename Tdst>
static void syncMasterToComputeImpl(sd::LaunchContext* context, NDArray* masterWeight, NDArray* computeWeight) {
  auto stream = context->getCudaStream();
  auto length = masterWeight->lengthOf();

  NDArray::prepareSpecialUse({computeWeight}, {const_cast<NDArray*>(masterWeight)});

  dim3 dims = getLaunchDims("updater");
  syncMasterToComputeKernel<Tsrc, Tdst><<<dims.y, dims.x, dims.z, *stream>>>(
      reinterpret_cast<const Tsrc*>(masterWeight->specialBuffer()),
      reinterpret_cast<Tdst*>(computeWeight->specialBuffer()),
      length);
  DebugHelper::checkErrorCode(stream, "syncMasterToComputeKernel");

  NDArray::registerSpecialUse({computeWeight}, {const_cast<NDArray*>(masterWeight)});
}

BUILD_DOUBLE_TEMPLATE(void syncMasterToComputeImpl,
                      (sd::LaunchContext* context, NDArray* masterWeight, NDArray* computeWeight),
                      SD_FLOAT_TYPES, SD_FLOAT_TYPES);

void syncMasterToCompute(sd::LaunchContext* context, const NDArray* masterWeight, NDArray* computeWeight) {
  if (masterWeight == nullptr || computeWeight == nullptr) return;
  auto masterNonConst = const_cast<NDArray*>(masterWeight);
  if (masterNonConst->isEmpty()) return;

  BUILD_DOUBLE_SELECTOR(masterNonConst->dataType(), computeWeight->dataType(), syncMasterToComputeImpl,
                        (context, masterNonConst, computeWeight),
                        SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
