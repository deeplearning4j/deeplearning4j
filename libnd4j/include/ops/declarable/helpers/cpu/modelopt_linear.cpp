/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/helpers/modelopt_linear.h>
#include <execution/Threads.h>
#include <helpers/shape.h>
#include <stdexcept>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

static void validateScales(NDArray* scale, NDArray* secondScale, bool nvfp4) {
  if (!modelOptValidScale(secondScale->bufferAsT<float>()[0]))
    throw std::invalid_argument("ModelOpt linear: global/input scale must be positive and finite");
  if (!nvfp4) {
    if (!modelOptValidScale(scale->bufferAsT<float>()[0]))
      throw std::invalid_argument("ModelOpt FP8 linear: weight scale must be positive and finite");
    return;
  }
  if (scale->isEmpty()) return;
  const auto* values = scale->bufferAsT<float8>();
  const auto* strides = scale->stridesOf();
  const LongType rows = scale->sizeAt(0), cols = scale->sizeAt(1);
  for (LongType row = 0; row < rows; ++row)
    for (LongType col = 0; col < cols; ++col)
      if (!modelOptValidScale(static_cast<float>(values[row * strides[0] + col * strides[1]])))
        throw std::invalid_argument("ModelOpt NVFP4 linear: every block scale must be positive and finite");
}

// One independent output per worker. We deliberately do not use dense MmulHelper:
// its operand ABI cannot express nibble decoding and per-block scales in registers.
// No intermediate grows with N*K; all indexing is relative to the shifted view bases.
template <typename X>
static void modelOptLinear_(NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale,
                            NDArray* z, bool nvfp4, bool floatOutput) {
  const X* input = x->isEmpty() ? nullptr : x->bufferAsT<X>();
  const auto* packed = nvfp4 && !w->isEmpty() ? w->bufferAsT<uint8_t>() : nullptr;
  const auto* fp8 = !nvfp4 && !w->isEmpty() ? w->bufferAsT<float8>() : nullptr;
  const auto* blockScales = nvfp4 && !scale->isEmpty() ? scale->bufferAsT<float8>() : nullptr;
  const float second = secondScale->bufferAsT<float>()[0];
  const float weightScale = nvfp4 ? 1.0f : scale->bufferAsT<float>()[0];
  void* output = z->buffer();
  const int rank = x->rankOf();
  const LongType kLength = x->sizeAt(-1), length = z->lengthOf();
  const auto* xs = x->stridesOf();
  const auto* ws = w->stridesOf();
  const auto* ss = scale->stridesOf();
  const auto* zs = z->stridesOf();
  const auto* zshape = z->shapeOf();
  auto work = PRAGMA_THREADS_FOR {
    for (LongType linear = start; linear < stop; linear += increment) {
      LongType coords[SD_MAX_RANK];
      INDEX2COORDS(linear, rank, zshape, coords);
      const LongType n = coords[rank - 1];
      LongType zo = 0, xo = 0;
      COORDS2INDEX(rank, zs, coords, zo);
      coords[rank - 1] = 0;
      COORDS2INDEX(rank, xs, coords, xo);
      using AccT = float;
      AccT sum = 0.0f;
      for (LongType k = 0; k < kLength; ++k) {
        float a = static_cast<float>(input[xo + k * xs[rank - 1]]);
        float b;
        if (nvfp4) {
          const uint8_t byte = packed[n * ws[0] + (k / 2) * ws[1]];
          const uint8_t nibble = (byte >> ((k & 1) * 4)) & 15;
          b = modelOptNvfp4Weight<X>(nibble,
              static_cast<float>(blockScales[n * ss[0] + (k / 16) * ss[1]]), second);
        } else {
          a = modelOptFp8Activation(a, second) * second;
          b = static_cast<float>(fp8[n * ws[0] + k * ws[1]]) * weightScale;
        }
        sum = std::fma(a, b, sum);
      }
      if (floatOutput) static_cast<float*>(output)[zo] = sum;
      else static_cast<X*>(output)[zo] = static_cast<X>(sum);
    }
  };
  samediff::Threads::parallel_for(work, 0, length);
}

BUILD_SINGLE_TEMPLATE(void modelOptLinear_,
    (NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale, NDArray* z,
     bool nvfp4, bool floatOutput), SD_MODELOPT_LINEAR_TYPES);

void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                    NDArray* secondScale, NDArray* z, bool nvfp4, bool floatOutput) {
  NDArray::preparePrimaryUse({}, {scale, secondScale});
  validateScales(scale, secondScale, nvfp4);
  NDArray::registerPrimaryUse({}, {scale, secondScale});
  if (z->isEmpty()) return;
  NDArray::preparePrimaryUse({z}, {x, w, scale, secondScale});
  BUILD_SINGLE_SELECTOR(x->dataType(), modelOptLinear_,
      (x, w, scale, secondScale, z, nvfp4, floatOutput), SD_MODELOPT_LINEAR_TYPES);
  NDArray::registerPrimaryUse({z}, {x, w, scale, secondScale});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
