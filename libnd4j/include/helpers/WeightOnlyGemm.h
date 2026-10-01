/* SPDX-License-Identifier: Apache-2.0 */
#ifndef LIBND4J_WEIGHT_ONLY_GEMM_H
#define LIBND4J_WEIGHT_ONLY_GEMM_H

#include <array/NDArray.h>
#include <execution/LaunchContext.h>

#if defined(SD_CUDA) && !defined(__JAVACPP_HACK__)
namespace sd {

/**
 * Weight-only quantized GEMM on tensor cores (ADR 0123):
 *   z[rows, N] = x[rows, K] . dequant(W)[N, K]^T
 * W stays in its quantized storage and is dequantized in registers with the
 * format's exact arithmetic; products accumulate in FP32 in a fixed order that
 * does not depend on the row count, so results are bit-reproducible under
 * graph capture/replay and a row's result does not depend on how many rows the
 * call carries.
 */
enum class WeightOnlyFormat {
  // ModelOpt NVFP4 (ADR 0122): W [N, K/2] packed E2M1 (even K in the low
  // nibble); scale: the block scales [N, K/16] E4M3; secondScale: the FP32
  // global scale. Each weight is rounded to the activation dtype before the
  // product.
  MODELOPT_NVFP4,
  // ModelOpt FP8 W8A8 (ADR 0122): W [N, K] E4M3; scale: the FP32 per-tensor
  // weight scale; secondScale: the FP32 static input scale. Activations are
  // quantized to E4M3 exactly as ops::helpers::modelOptQuantize<
  // ModelOptFp8::Activation>, the exact E4M3 x E4M3 products accumulate in
  // FP32, and the product of the two scales multiplies each output's sum once.
  MODELOPT_FP8,
};

class SD_LIB_HIDDEN WeightOnlyGemm {
 public:
  /**
   * Whether the tensor-core path accepts these operands. Depends only on the
   * format, dtypes, shapes, layouts and base alignment — never on runtime
   * state — so a given linear always takes the same path.
   */
  static bool isAdmitted(WeightOnlyFormat format, NDArray* x, NDArray* w, NDArray* scale, NDArray* z);

  /**
   * Enqueues the GEMM on the context stream. The operands must be admitted.
   * x and z dispatch over the float types; z's dtype is the output type. scale
   * and secondScale are the format's (WeightOnlyFormat).
   */
  static void run(LaunchContext* context, WeightOnlyFormat format, NDArray* x, NDArray* w, NDArray* scale,
                  NDArray* secondScale, NDArray* z);
};

}  // namespace sd
#endif

#endif  // LIBND4J_WEIGHT_ONLY_GEMM_H
