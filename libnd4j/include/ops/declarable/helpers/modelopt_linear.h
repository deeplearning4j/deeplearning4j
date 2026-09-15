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

// ModelOpt NVFP4QTensor.dequantize: FP32(blockScale * globalScale), then
// FP32(e2m1 * scale), THEN conversion to the original activation dtype.
// Source: NVIDIA/Model-Optimizer modelopt/torch/quantization/qtensor/nvfp4_tensor.py.
SD_HOST_DEVICE SD_INLINE float modelOptE2M1(unsigned char nibble) {
  const unsigned char magnitude = nibble & 7;
  float value = magnitude < 4 ? 0.5f * magnitude :
                (magnitude == 4 ? 2.0f : magnitude == 5 ? 3.0f : magnitude == 6 ? 4.0f : 6.0f);
  return (nibble & 8) ? -value : value;
}

template <typename X>
SD_HOST_DEVICE SD_INLINE float modelOptNvfp4Weight(unsigned char nibble, float blockScale,
                                                   float globalScale) {
#if defined(__CUDA_ARCH__)
  // Keep both roundings explicit even under CUDA fast-math compilation.
  const float scale = __fmul_rn(blockScale, globalScale);
  const float weight = __fmul_rn(modelOptE2M1(nibble), scale);
#else
  const float scale = blockScale * globalScale;
  const float weight = modelOptE2M1(nibble) * scale;
#endif
  return static_cast<float>(static_cast<X>(weight));
}

// ModelOpt tensor_quant.py _fp8_eager clamps before its E4M3FN cast. Explicit
// comparisons preserve NaN and saturate infinities (the raw float8 cast does not).
// inputScale is the exported dequantization scale, NOT amax or its reciprocal.
SD_HOST_DEVICE SD_INLINE float modelOptFp8Activation(float x, float inputScale) {
#if defined(__CUDA_ARCH__)
  float scaled = __fdiv_rn(x, inputScale);
#else
  float scaled = x / inputScale;
#endif
  if (scaled > 448.0f) scaled = 448.0f;
  if (scaled < -448.0f) scaled = -448.0f;
  return static_cast<float>(float8(scaled));
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
#endif
