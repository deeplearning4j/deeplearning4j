/* SPDX-License-Identifier: Apache-2.0 */
#ifndef LIBND4J_MODELOPT_LINEAR_H
#define LIBND4J_MODELOPT_LINEAR_H

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <array/DataTypeUtils.h>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/reproducible_math.h>

#include <stdexcept>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_HOST_DEVICE SD_INLINE bool modelOptValidScale(T value) {
  return value > static_cast<T>(0) && math::sd_isfin<T>(value);
}

// Per-element scale validation fused into the device kernels that read the
// scales: the single on-stream source of truth (captured, replayed, and observed
// at the caller's stream completion boundary). The device trap remains active in
// release builds (plain assert() disappears under NDEBUG). Host callers validate
// every scale, with exceptions, before reading any.
template <typename T>
SD_HOST_DEVICE SD_INLINE T modelOptScaleChecked(T value) {
#if defined(__CUDA_ARCH__)
  if (!modelOptValidScale<T>(value)) __trap();
#endif
  return value;
}

// ModelOpt NVFP4QTensor.dequantize: S(blockScale * globalScale), then
// S(e2m1 * scale), THEN one conversion to Z (the activation dtype, or its
// tensor-core element). Every E2M1 code is exact in S.
// Source: NVIDIA/Model-Optimizer modelopt/torch/quantization/qtensor/nvfp4_tensor.py.
template <typename Z, typename S>
SD_HOST_DEVICE SD_INLINE Z modelOptNvfp4Weight(float4_e2m1 code, S blockScale, S globalScale) {
  const S scale = reproducible::multiply<S>(blockScale, globalScale);
  return static_cast<Z>(reproducible::multiply<S>(static_cast<S>(code), scale));
}

// ModelOpt tensor_quant.py _fp8_eager: x / inputScale clamped to Q's finite
// range, then converted to Q (sd_saturate: saturating round to nearest even,
// NaN stays NaN). inputScale is the exported dequantization scale, NOT amax or
// its reciprocal.
template <typename Q, typename AccT>
SD_HOST_DEVICE SD_INLINE Q modelOptQuantize(AccT x, AccT inputScale) {
  return math::sd_saturate<AccT, Q>(reproducible::divide<AccT>(x, inputScale));
}

// The quantized activation re-expanded: Q(x) * inputScale.
template <typename Q, typename AccT>
SD_HOST_DEVICE SD_INLINE AccT modelOptFakeQuantize(AccT x, AccT inputScale) {
  return reproducible::multiply<AccT>(static_cast<AccT>(modelOptQuantize<Q, AccT>(x, inputScale)), inputScale);
}

// Format policies: each format's storage types plus the per-element arithmetic
// every native kernel shares (the general and tiled kernels here, the CPU
// kernel, and WeightOnlyGemm):
//  - weightScale(scale, second): the per-tensor weight factor;
//  - activation<AccT>(x, second): the activation as the format consumes it;
//  - weight<X, AccT>(...): the dequantized weight (n, k) of a strided view.
// second is the format's scalar FP32 scale: the NVFP4 global scale, or the FP8
// static input scale.

// W [N, K/2] packed E2M1 (even k in the low nibble); block scales [N, K/16]
// E4M3; FP32 global scale.
struct ModelOptNvfp4 {
  using Weight = float4_e2m1;
  using Storage = uint8_t;
  using ScaleStorage = float8_e4m3;
  using Scale = float;
  static constexpr int kWeightsPerStorage = Weight::codesPer<Storage>();
  static constexpr int kBlockLength = 16;
  static_assert(kBlockLength % kWeightsPerStorage == 0, "an NVFP4 block covers whole storage words");
  static constexpr const char* kName = "NVFP4";
  static constexpr const char* kInvalidScale = "ModelOpt NVFP4 linear: every block scale must be positive and finite";

  SD_HOST_DEVICE SD_INLINE static Scale weightScale(const ScaleStorage*, Scale globalScale) { return globalScale; }

  template <typename AccT>
  SD_HOST_DEVICE SD_INLINE static AccT activation(AccT value, Scale) {
    return value;
  }

  template <typename X, typename AccT>
  SD_HOST_DEVICE SD_INLINE static AccT weight(const Storage* w, const LongType* ws, const ScaleStorage* scale,
                                              const LongType* ss, Scale globalScale, LongType n, LongType k) {
    const Weight code = Weight::unpack(w[n * ws[0] + (k / kWeightsPerStorage) * ws[1]],
                                       static_cast<int>(k % kWeightsPerStorage));
    const Scale blockScale = modelOptScaleChecked(static_cast<Scale>(scale[n * ss[0] + (k / kBlockLength) * ss[1]]));
    return static_cast<AccT>(modelOptNvfp4Weight<X, Scale>(code, blockScale, globalScale));
  }
};

// W [N, K] E4M3; FP32 per-tensor weight scale; FP32 static input scale, the
// activations quantized to E4M3 per element.
struct ModelOptFp8 {
  using Weight = float8_e4m3;
  using Activation = float8_e4m3;
  using Storage = float8_e4m3;
  using ScaleStorage = float;
  using Scale = float;
  static constexpr const char* kName = "FP8";
  static constexpr const char* kInvalidScale = "ModelOpt FP8 linear: weight scale must be positive and finite";

  SD_HOST_DEVICE SD_INLINE static Scale weightScale(const ScaleStorage* scale, Scale) {
    return modelOptScaleChecked(static_cast<Scale>(scale[0]));
  }

  template <typename AccT>
  SD_HOST_DEVICE SD_INLINE static AccT activation(AccT value, Scale inputScale) {
    return modelOptFakeQuantize<Activation, AccT>(value, static_cast<AccT>(inputScale));
  }

  template <typename X, typename AccT>
  SD_HOST_DEVICE SD_INLINE static AccT weight(const Storage* w, const LongType* ws, const ScaleStorage*,
                                              const LongType*, Scale weightScale, LongType n, LongType k) {
    return reproducible::multiply<AccT>(static_cast<AccT>(w[n * ws[0] + k * ws[1]]), static_cast<AccT>(weightScale));
  }
};

// Host validation of host-current scales, throwing std::invalid_argument: the
// format's scalar scale, and every element of its scale tensor (any rank or view).
template <typename Format>
void modelOptCheckSecondScale(NDArray* secondScale) {
  using Scale = typename Format::Scale;
  if (!modelOptValidScale<Scale>(secondScale->bufferAsT<Scale>()[0]))
    throw std::invalid_argument("ModelOpt linear: global/input scale must be positive and finite");
}

template <typename Format>
void modelOptCheckScaleTensor(NDArray* scale) {
  using Scale = typename Format::Scale;
  if (scale->isEmpty()) return;
  const auto* values = scale->bufferAsT<typename Format::ScaleStorage>();
  const int rank = scale->rankOf();
  const LongType* shapeOf = scale->shapeOf();
  const LongType* strides = scale->stridesOf();
  const LongType length = scale->lengthOf();
  for (LongType linear = 0; linear < length; ++linear) {
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(linear, rank, shapeOf, coords);
    LongType offset = 0;
    COORDS2INDEX(rank, strides, coords, offset);
    if (!modelOptValidScale<Scale>(static_cast<Scale>(values[offset]))) throw std::invalid_argument(Format::kInvalidScale);
  }
}

// z = x . dequantize(w)^T in Format. x and z dispatch over the float types
// (SD_FLOAT_TYPES); z's dtype (x's, or FLOAT32) is the output type. scale is the
// format's scale tensor and secondScale its scalar scale.
template <typename Format>
SD_LIB_HIDDEN void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                  NDArray* secondScale, NDArray* z);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
#endif
