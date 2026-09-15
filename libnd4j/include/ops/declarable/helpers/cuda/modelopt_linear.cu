/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/helpers/modelopt_linear.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace sd {
namespace ops {
namespace helpers {

// Runtime tensors cannot be synchronously value-validated during capture. Keep
// validation on the SAME stream, before the dot kernel; it is captured and checked
// on EVERY replay, not cached by pointer. The CUDA device trap intrinsic remains
// active in release builds (plain assert() disappears under NDEBUG). Invalid
// device-only scales fail at the caller's normal stream completion boundary.
SD_KERNEL static void modelOptValidateScalesKernel(const void* scale, const float* second,
                                                   const LongType* shapeInfo,
                                                   LongType count, bool nvfp4) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && !modelOptValidScale(second[0]))
    __trap();
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < count; linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    float value;
    if (nvfp4) {
      const LongType cols = shape::sizeAt(shapeInfo, 1);
      const auto* strides = shape::stride(shapeInfo);
      value = static_cast<float>(static_cast<const float8*>(scale)[
          (linear / cols) * strides[0] + (linear % cols) * strides[1]]);
    } else {
      value = static_cast<const float*>(scale)[0];
    }
    if (!modelOptValidScale(value))
      __trap();
  }
}

// Correctness-first packed FMA implementation; NOT a tensor-core kernel. Each
// thread owns a complete output dot product (FP32), avoiding output scratch,
// atomic accumulation, and whole-weight dequantization. No contiguity assumption.
template <typename X>
SD_KERNEL static void modelOptLinearKernel(const X* x, const void* w, const void* scale,
                                          const float* secondScale, void* z,
                                          const LongType* xShape, const LongType* wShape,
                                          const LongType* sShape, const LongType* zShape,
                                          LongType length, bool nvfp4, bool floatOutput) {
  const int rank = shape::rank(xShape);
  const LongType kLength = shape::sizeAt(xShape, rank - 1);
  const auto* xs = shape::stride(xShape);
  const auto* ws = shape::stride(wShape);
  const auto* ss = shape::stride(sShape);
  const auto* zs = shape::stride(zShape);
  const auto* zd = shape::shapeOf(zShape);
  const float second = secondScale[0];
  const float weightScale = nvfp4 ? 1.0f : static_cast<const float*>(scale)[0];
  for (LongType linear = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < length; linear += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linear, rank, zd, coords);
    const LongType n = coords[rank - 1];
    LongType xo = 0, zo = 0;
    COORDS2INDEX(rank, zs, coords, zo);
    coords[rank - 1] = 0;
    COORDS2INDEX(rank, xs, coords, xo);
    using AccT = float;
    AccT sum = 0.0f;
    for (LongType k = 0; k < kLength; ++k) {
      float a = static_cast<float>(x[xo + k * xs[rank - 1]]);
      float b;
      if (nvfp4) {
        const auto byte = static_cast<const uint8_t*>(w)[n * ws[0] + (k / 2) * ws[1]];
        const auto nibble = static_cast<uint8_t>((byte >> ((k & 1) * 4)) & 15);
        b = modelOptNvfp4Weight<X>(nibble,
            static_cast<float>(static_cast<const float8*>(scale)[n * ss[0] + (k / 16) * ss[1]]), second);
      } else {
        a = modelOptFp8Activation(a, second) * second;
        b = static_cast<float>(static_cast<const float8*>(w)[n * ws[0] + k * ws[1]]) * weightScale;
      }
      sum = fmaf(a, b, sum);
    }
    if (floatOutput) static_cast<float*>(z)[zo] = sum;
    else static_cast<X*>(z)[zo] = static_cast<X>(sum);
  }
}

// Host-current checkpoint scales can fail before any launch without a device
// transfer. Device-current scales are checked by the ordered validation kernel.
static void validateHostScales(NDArray* scale, NDArray* secondScale, bool nvfp4) {
  if (secondScale->isActualOnHostSide() && !modelOptValidScale(secondScale->bufferAsT<float>()[0]))
    throw std::invalid_argument("ModelOpt linear: global/input scale must be positive and finite");
  if (scale->isEmpty() || !scale->isActualOnHostSide()) return;
  if (!nvfp4) {
    if (!modelOptValidScale(scale->bufferAsT<float>()[0]))
      throw std::invalid_argument("ModelOpt FP8 linear: weight scale must be positive and finite");
    return;
  }
  const auto* data = scale->bufferAsT<float8>();
  const auto* strides = scale->stridesOf();
  const LongType rows = scale->sizeAt(0), cols = scale->sizeAt(1);
  for (LongType row = 0; row < rows; ++row)
    for (LongType col = 0; col < cols; ++col)
      if (!modelOptValidScale(static_cast<float>(data[row * strides[0] + col * strides[1]])))
        throw std::invalid_argument("ModelOpt NVFP4 linear: every block scale must be positive and finite");
}

template <typename X>
static void modelOptLinear_(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                            NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput,
                            dim3 dims) {
  auto* stream = context->getCudaStream();
  const auto* xShape = x->specialShapeInfo();
  const auto* wShape = w->specialShapeInfo();
  const auto* sShape = scale->specialShapeInfo();
  const auto* zShape = z->specialShapeInfo();
  const LongType needed = (z->lengthOf() - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  modelOptLinearKernel<X><<<blocks, dims.y, dims.z, *stream>>>(
      x->isEmpty() ? nullptr : static_cast<const X*>(x->specialBuffer()),
      w->isEmpty() ? nullptr : w->specialBuffer(), scale->isEmpty() ? nullptr : scale->specialBuffer(),
      static_cast<const float*>(secondScale->specialBuffer()), z->specialBuffer(),
      xShape, wShape, sShape, zShape, z->lengthOf(), nvfp4, floatOutput);
}

BUILD_SINGLE_TEMPLATE(void modelOptLinear_,
    (LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale,
     NDArray* z, bool nvfp4, bool floatOutput, dim3 dims), SD_MODELOPT_LINEAR_TYPES);

void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                    NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput) {
  auto* stream = context->getCudaStream();
  const bool capturing = DebugHelper::inGraphCapture(stream);
  if (!capturing) validateHostScales(scale, secondScale, nvfp4);
  const dim3 dims = getLaunchDims("modelopt_linear");
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, context->getDeviceID()) != cudaSuccess)
    throw std::runtime_error("ModelOpt linear: unable to query launch limits");
  if (dims.x == 0 || dims.y == 0 || dims.x > static_cast<unsigned int>(prop.maxGridSize[0]) ||
      dims.y > static_cast<unsigned int>(prop.maxThreadsPerBlock) ||
      dims.y > static_cast<unsigned int>(prop.maxThreadsDim[0]) || dims.z > prop.sharedMemPerBlock)
    throw std::invalid_argument("ModelOpt linear: invalid launch dimensions");

  NDArray::prepareSpecialUse({}, {scale, secondScale});
  const auto* sShape = scale->specialShapeInfo();
  // At least one thread must validate the scalar even if the block-scale tensor
  // is empty. This is validation work, not an empty-output compute launch.
  const LongType count = scale->lengthOf();
  const LongType needed = count == 0 ? 1 : (count - 1) / dims.y + 1;
  const unsigned int blocks = needed < dims.x ? static_cast<unsigned int>(needed) : dims.x;
  modelOptValidateScalesKernel<<<blocks, dims.y, 0, *stream>>>(
      scale->isEmpty() ? nullptr : scale->specialBuffer(),
      static_cast<const float*>(secondScale->specialBuffer()), sShape, count, nvfp4);
  NDArray::registerSpecialUse({}, {scale, secondScale});
  if (!z->isEmpty()) {
    NDArray::prepareSpecialUse({z}, {x, w, scale, secondScale});
    BUILD_SINGLE_SELECTOR(x->dataType(), modelOptLinear_,
        (context, x, w, scale, secondScale, z, nvfp4, floatOutput, dims), SD_MODELOPT_LINEAR_TYPES);
    NDArray::registerSpecialUse({z}, {x, w, scale, secondScale});
  }
  if (!capturing) DebugHelper::checkGlobalErrorCode("ModelOpt linear launch failed");
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
