/* SPDX-License-Identifier: Apache-2.0 */
#ifndef LIBND4J_MODELOPT_LINEAR_H
#define LIBND4J_MODELOPT_LINEAR_H

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/helpers/helpers.h>
#include <math/templatemath.h>
#include <types/float8.h>

namespace sd {
namespace ops {
namespace helpers {

// X dtype is the only independently dispatched type. W/scales have fixed storage
// and the output is either X's type or FLOAT32. Preserve selective type gates.
#define SD_MODELOPT_LINEAR_TYPES SKIP_FIRST_COMMA(TTYPE_FLOAT32 TTYPE_HALF TTYPE_BFLOAT)

SD_LIB_HIDDEN void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w,
                                 NDArray* scale, NDArray* secondScale, NDArray* z,
                                 bool nvfp4, bool floatOutput);

SD_HOST_DEVICE SD_INLINE bool modelOptValidScale(float value) {
  return value > 0.0f && math::sd_isfin<float>(value);
}

#if defined(__CUDACC__)
// Per-element scale validation fused into the device kernels that read the
// scales: the single on-stream source of truth (captured, replayed, and observed
// at the caller's stream completion boundary). The device trap remains active in
// release builds (plain assert() disappears under NDEBUG).
SD_DEVICE SD_INLINE float modelOptScaleChecked(float value) {
  if (!modelOptValidScale(value))
    asm("trap;");
  return value;
}
#endif

// ModelOpt NVFP4QTensor.dequantize: FP32(blockScale * globalScale), then
// FP32(e2m1 * scale), THEN conversion to the original activation dtype.
// Source: NVIDIA/Model-Optimizer modelopt/torch/quantization/qtensor/nvfp4_tensor.py.
//
// E2M1 codes are exact FP32 values, built from bits without branches (this runs
// once per weight in every NVFP4 kernel): magnitudes 2..7 (1, 1.5, 2, 3, 4, 6)
// are the FP32 patterns (magnitude << 22) + bits(0.5); 1 is 0.5 and 0 is +0;
// bit 3 is the sign (so code 8 is -0).
SD_HOST_DEVICE SD_INLINE float modelOptE2M1(unsigned char nibble) {
  constexpr uint32_t kHalfBits = 0x3F000000u;  // FP32 0.5
  const uint32_t magnitude = nibble & 7u;
  const uint32_t magnitudeBits = magnitude >= 2u ? (magnitude << 22) + kHalfBits
                                                 : (magnitude == 1u ? kHalfBits : 0u);
  const uint32_t bits = magnitudeBits | (static_cast<uint32_t>(nibble & 8u) << 28);
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

// The FP32 weight before the conversion to the activation dtype. Callers that
// convert with a hardware round-to-nearest-even into an X-equivalent type (the
// tensor-core element) get the same bits as modelOptNvfp4Weight<X>.
SD_HOST_DEVICE SD_INLINE float modelOptNvfp4WeightFp32(unsigned char nibble, float blockScale,
                                                      float globalScale) {
#if defined(__CUDA_ARCH__)
  // Keep both roundings explicit even under CUDA fast-math compilation.
  const float scale = __fmul_rn(blockScale, globalScale);
  return __fmul_rn(modelOptE2M1(nibble), scale);
#else
  const float scale = blockScale * globalScale;
  return modelOptE2M1(nibble) * scale;
#endif
}

template <typename X>
SD_HOST_DEVICE SD_INLINE float modelOptNvfp4Weight(unsigned char nibble, float blockScale,
                                                   float globalScale) {
  return static_cast<float>(static_cast<X>(modelOptNvfp4WeightFp32(nibble, blockScale, globalScale)));
}

// ModelOpt tensor_quant.py _fp8_eager clamps before its E4M3FN cast. Explicit
// comparisons preserve NaN and saturate infinities (the raw float8 cast does not).
// inputScale is the exported dequantization scale, NOT amax or its reciprocal.
SD_HOST_DEVICE SD_INLINE float8 modelOptFp8Quantize(float x, float inputScale) {
#if defined(__CUDA_ARCH__)
  float scaled = __fdiv_rn(x, inputScale);
#else
  float scaled = x / inputScale;
#endif
  if (scaled > 448.0f) scaled = 448.0f;
  if (scaled < -448.0f) scaled = -448.0f;
  return float8(scaled);
}

// The quantized activation re-expanded to FP32 (the E4M3 value, unscaled).
SD_HOST_DEVICE SD_INLINE float modelOptFp8Activation(float x, float inputScale) {
  return static_cast<float>(modelOptFp8Quantize(x, inputScale));
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
#endif
