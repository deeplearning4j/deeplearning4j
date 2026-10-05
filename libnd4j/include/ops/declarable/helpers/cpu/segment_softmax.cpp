/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// CPU implementation of segment_softmax and segment_softmax_bp.
//
// Segment s is the set of rows whose id is s (the ids are sorted, validated by the op, so a segment is one run of
// rows); the classes are 0 .. K-1 and a row with another id keeps the zero the op wrote.
//
// Forward, for every segment s and feature f (a column of a row; a row is the trailing dimensions of the logits in C
// order):
//     1. max_s = max(logits[i, f] for i in segment s)
//     2. expSum_s = sum(exp(logits[i, f] - max_s) for i in segment s)
//     3. out[i, f] = exp(logits[i, f] - max_s) / expSum_s
//
// Backward:
//     dLogits[i, f] = out[i, f] * (gradOut[i, f] - dotProd_s)
//     where dotProd_s = sum_{j in segment s} gradOut[j, f] * out[j, f]
//
// The ids may have any integer dtype, rank and strides; the arrays are read and written through their own shapes and
// strides (any rank, order, offset or stepped view). The arithmetic is done in FLOAT (DOUBLE for DOUBLE).
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/segment_semantics.h>
#include <ops/declarable/helpers/segment_softmax.h>
#include <system/op_boilerplate.h>

#include <cmath>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

namespace segment_softmax_cpu {

using namespace segment_sem;

template <typename I>
static void readIds_(NDArray& indices, LongType* dst) {
  const I* ids = indices.bufferAsT<I>();
  const LongType n = indices.lengthOf();
  const LongType rank = indices.rankOf();
  const LongType* shp = shape::shapeOf(indices.shapeInfo());
  const LongType* str = shape::stride(indices.shapeInfo());
  for (LongType i = 0; i < n; ++i) dst[i] = idToLong<I>(ids[logicalOffset(i, rank, shp, str)]);
}

static void readIds(NDArray& indices, std::vector<LongType>& out) {
  out.resize(static_cast<size_t>(indices.lengthOf()));
  if (out.empty()) return;
  BUILD_SINGLE_SELECTOR(indices.dataType(), readIds_, (indices, out.data()), SD_INTEGER_TYPES);
}

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static void segmentSoftmax_(LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out) {
  using A = typename AccFloat<X, X>::type;
  std::vector<LongType> ids;
  readIds(segmentIds, ids);
  const LongType N = static_cast<LongType>(ids.size());
  const SegMat lm = makeSegMat(logits.shapeInfo(), logits.shapeInfo());
  const SegMat om = makeSegMat(out.shapeInfo(), out.shapeInfo());
  const LongType inner = lm.inner;
  if (K <= 0 || N == 0 || inner == 0) return;

  std::vector<LongType> begin(static_cast<size_t>(K)), end(static_cast<size_t>(K));
  host::buildRanges(ids.data(), N, K, begin.data(), end.data());
  const X* x = logits.bufferAsT<X>();
  X* z = out.bufferAsT<X>();

  auto func = PRAGMA_THREADS_FOR {
    for (LongType p = start; p < stop; ++p) {
      const LongType s = p / inner;
      const LongType f = p - s * inner;
      const LongType first = begin[s];
      const LongType last = end[s];
      if (last <= first) continue;
      const LongType lc = lm.colOffset(f);
      const LongType oc = om.colOffset(f);
      A maxVal = -DataTypeUtils::max<A>();
      for (LongType q = first; q < last; ++q) {
        const A v = static_cast<A>(x[q * lm.rowStride + lc]);
        if (v > maxVal) maxVal = v;
      }
      A expSum = static_cast<A>(0);
      for (LongType q = first; q < last; ++q)
        expSum += sd::math::sd_exp<A, A>(static_cast<A>(x[q * lm.rowStride + lc]) - maxVal);
      for (LongType q = first; q < last; ++q) {
        const A e = sd::math::sd_exp<A, A>(static_cast<A>(x[q * lm.rowStride + lc]) - maxVal);
        z[q * om.rowStride + oc] = static_cast<X>(e / expSum);
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, K * inner);
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static void segmentSoftmaxBp_(LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out, NDArray& gradOut,
                              NDArray& dLogits) {
  using A = typename AccFloat<X, X>::type;
  std::vector<LongType> ids;
  readIds(segmentIds, ids);
  const LongType N = static_cast<LongType>(ids.size());
  const SegMat fm = makeSegMat(out.shapeInfo(), out.shapeInfo());
  const SegMat gm = makeSegMat(gradOut.shapeInfo(), gradOut.shapeInfo());
  const SegMat dm = makeSegMat(dLogits.shapeInfo(), dLogits.shapeInfo());
  const LongType inner = dm.inner;
  if (K <= 0 || N == 0 || inner == 0) return;

  std::vector<LongType> begin(static_cast<size_t>(K)), end(static_cast<size_t>(K));
  host::buildRanges(ids.data(), N, K, begin.data(), end.data());
  const X* fwd = out.bufferAsT<X>();
  const X* go = gradOut.bufferAsT<X>();
  X* dl = dLogits.bufferAsT<X>();

  auto func = PRAGMA_THREADS_FOR {
    for (LongType p = start; p < stop; ++p) {
      const LongType s = p / inner;
      const LongType f = p - s * inner;
      const LongType first = begin[s];
      const LongType last = end[s];
      if (last <= first) continue;
      const LongType fc = fm.colOffset(f);
      const LongType gc = gm.colOffset(f);
      const LongType dc = dm.colOffset(f);
      A dot = static_cast<A>(0);
      for (LongType q = first; q < last; ++q)
        dot += static_cast<A>(go[q * gm.rowStride + gc]) * static_cast<A>(fwd[q * fm.rowStride + fc]);
      for (LongType q = first; q < last; ++q) {
        const A o = static_cast<A>(fwd[q * fm.rowStride + fc]);
        const A g = static_cast<A>(go[q * gm.rowStride + gc]);
        dl[q * dm.rowStride + dc] = static_cast<X>(o * (g - dot));
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, K * inner);
}

}  // namespace segment_softmax_cpu

// ────────────────────────────────────────────────────────────────────────────
// Public interface
// ────────────────────────────────────────────────────────────────────────────

void segmentSoftmax(sd::LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out) {
  NDArray::preparePrimaryUse({&out}, {&logits, &segmentIds});
  BUILD_SINGLE_SELECTOR(logits.dataType(), segment_softmax_cpu::segmentSoftmax_, (K, logits, segmentIds, out),
                        SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({&out}, {&logits, &segmentIds});
}

void segmentSoftmaxBp(sd::LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out, NDArray& gradOut,
                      NDArray& dLogits) {
  NDArray::preparePrimaryUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
  BUILD_SINGLE_SELECTOR(logits.dataType(), segment_softmax_cpu::segmentSoftmaxBp_,
                        (K, logits, segmentIds, out, gradOut, dLogits), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
