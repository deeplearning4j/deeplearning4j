/* SPDX-License-Identifier: Apache-2.0 */
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_modelopt_nvfp4_linear) || NOT_EXCLUDED(OP_modelopt_fp8_linear)
#include <ops/declarable/helpers/modelopt_linear.h>
#include <execution/Threads.h>
#include <helpers/shape.h>
#include <ops/op_types.h>

namespace sd {
namespace ops {
namespace helpers {

// One independent output per worker. We deliberately do not use dense MmulHelper:
// its operand ABI cannot express nibble decoding and per-block scales in registers.
// No intermediate grows with N*K; all indexing is relative to the shifted view bases.
template <typename X, typename Z, typename Format>
static void modelOptLinear_(Format, NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale, NDArray* z) {
  using AccT = typename simdOps::AggregateType<X>::type;
  const X* input = x->isEmpty() ? nullptr : x->bufferAsT<X>();
  const auto* weights = w->isEmpty() ? nullptr : w->bufferAsT<typename Format::Storage>();
  const auto* scales = scale->isEmpty() ? nullptr : scale->bufferAsT<typename Format::ScaleStorage>();
  const typename Format::Scale second = secondScale->bufferAsT<typename Format::Scale>()[0];
  const auto weightScale = Format::weightScale(scales, second);
  Z* output = z->bufferAsT<Z>();
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
      AccT sum = static_cast<AccT>(0);
      for (LongType k = 0; k < kLength; ++k) {
        const AccT a = Format::template activation<AccT>(static_cast<AccT>(input[xo + k * xs[rank - 1]]), second);
        const AccT b = Format::template weight<X, AccT>(weights, ws, scales, ss, weightScale, n, k);
        sum = math::sd_fma<AccT>(a, b, sum);
      }
      output[zo] = static_cast<Z>(sum);
    }
  };
  samediff::Threads::parallel_for(work, 0, length);
}

template <typename Format>
void modelOptLinear(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale, NDArray* secondScale,
                    NDArray* z) {
  NDArray::preparePrimaryUse({}, {scale, secondScale});
  modelOptCheckSecondScale<Format>(secondScale);
  modelOptCheckScaleTensor<Format>(scale);
  NDArray::registerPrimaryUse({}, {scale, secondScale});
  if (z->isEmpty()) return;
  NDArray::preparePrimaryUse({z}, {x, w, scale, secondScale});
  BUILD_DOUBLE_SELECTOR(x->dataType(), z->dataType(), modelOptLinear_, (Format{}, x, w, scale, secondScale, z),
                        SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({z}, {x, w, scale, secondScale});
}

template void modelOptLinear<ModelOptNvfp4>(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                            NDArray* secondScale, NDArray* z);
template void modelOptLinear<ModelOptFp8>(LaunchContext* context, NDArray* x, NDArray* w, NDArray* scale,
                                          NDArray* secondScale, NDArray* z);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
