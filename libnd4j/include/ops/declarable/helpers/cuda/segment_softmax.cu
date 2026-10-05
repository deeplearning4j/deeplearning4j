/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// CUDA implementation of segment_softmax and segment_softmax_bp.
//
// Segment s is the set of rows whose id is s (the ids are sorted, validated by the op, so a segment is one run of
// rows); the classes are 0 .. K-1 and a row with another id keeps the zero the op wrote. One block per (segment,
// feature) pair, grid-stride over the pairs (no pair count is limited by a grid dimension), the threads of a block
// stride over the rows of the segment:
//   pass 1: max (block reduction)   pass 2: sum of exp(x - max) (block reduction)   pass 3: out = exp(x - max) / sum.
// The ids of any integer dtype are first read into one dense int64 sequence and cut into per-segment row ranges with
// the same kernels as the other segment ops (segmentReadIds / segmentBuildRanges: O(N), no per-block scan). The
// logits, the output and the gradients are read and written through their own shapes and strides, so any rank, order,
// offset or stepped view is correct. The arithmetic is done in FLOAT (DOUBLE for DOUBLE).
//
#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <cuda_runtime.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>
#include <ops/declarable/helpers/segment_common.h>
#include <ops/declarable/helpers/segment_semantics.h>
#include <ops/declarable/helpers/segment_softmax.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

namespace {
SD_INLINE int softmaxClampInt(LongType value) {
  if (value < 0) return 0;
  if (value > static_cast<LongType>(2147483647)) return 2147483647;
  return static_cast<int>(value);
}
}  // namespace

// ────────────────────────────────────────────────────────────────────────────
// Forward kernel: one block per (segment, feature) pair
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void segSoftmaxKernel(const X* logits, segment_sem::SegMat lm, X* out, segment_sem::SegMat om,
                                       const LongType* begin, const LongType* end, LongType K, LongType inner) {
  using A = typename segment_sem::AccFloat<X, X>::type;
  __shared__ A scratch[32];  // one slot per warp of a block of at most 1024 threads
  const LongType pairs = K * inner;
  for (LongType p = blockIdx.x; p < pairs; p += gridDim.x) {
    const LongType s = p / inner;
    const LongType f = p - s * inner;
    const LongType first = begin[s];
    const LongType last = end[s];
    if (last <= first) continue;  // the same for every thread of the block
    const LongType lc = lm.colOffset(f);
    const LongType oc = om.colOffset(f);

    // pass 1: max
    A localMax = -DataTypeUtils::max<A>();
    for (LongType q = first + threadIdx.x; q < last; q += blockDim.x) {
      const A v = static_cast<A>(logits[q * lm.rowStride + lc]);
      if (v > localMax) localMax = v;
    }
    const A maxVal = sd::device::blockAllReduceMax<A>(localMax, scratch);

    // pass 2: sum of exp
    A localSum = static_cast<A>(0);
    for (LongType q = first + threadIdx.x; q < last; q += blockDim.x)
      localSum += sd::math::sd_exp<A, A>(static_cast<A>(logits[q * lm.rowStride + lc]) - maxVal);
    const A expSum = sd::device::blockAllReduceSum<A>(localSum, scratch);

    // pass 3: normalize
    for (LongType q = first + threadIdx.x; q < last; q += blockDim.x) {
      const A e = sd::math::sd_exp<A, A>(static_cast<A>(logits[q * lm.rowStride + lc]) - maxVal);
      out[q * om.rowStride + oc] = static_cast<X>(e / expSum);
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel: dLogits = out * (gradOut - sum_{segment} gradOut * out)
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void segSoftmaxBpKernel(const X* fwd, segment_sem::SegMat fm, const X* gradOut,
                                         segment_sem::SegMat gm, X* dLogits, segment_sem::SegMat dm,
                                         const LongType* begin, const LongType* end, LongType K, LongType inner) {
  using A = typename segment_sem::AccFloat<X, X>::type;
  __shared__ A scratch[32];
  const LongType pairs = K * inner;
  for (LongType p = blockIdx.x; p < pairs; p += gridDim.x) {
    const LongType s = p / inner;
    const LongType f = p - s * inner;
    const LongType first = begin[s];
    const LongType last = end[s];
    if (last <= first) continue;
    const LongType fc = fm.colOffset(f);
    const LongType gc = gm.colOffset(f);
    const LongType dc = dm.colOffset(f);

    A localDot = static_cast<A>(0);
    for (LongType q = first + threadIdx.x; q < last; q += blockDim.x)
      localDot += static_cast<A>(gradOut[q * gm.rowStride + gc]) * static_cast<A>(fwd[q * fm.rowStride + fc]);
    const A dot = sd::device::blockAllReduceSum<A>(localDot, scratch);

    for (LongType q = first + threadIdx.x; q < last; q += blockDim.x) {
      const A o = static_cast<A>(fwd[q * fm.rowStride + fc]);
      const A g = static_cast<A>(gradOut[q * gm.rowStride + gc]);
      dLogits[q * dm.rowStride + dc] = static_cast<X>(o * (g - dot));
    }
  }
}

// the launch: a power of two of at least one warp (the block reductions use full-warp shuffles)
static dim3 softmaxDims(LongType K, LongType inner, LongType N) {
  dim3 dims = segmentDims(softmaxClampInt(K * inner), softmaxClampInt(N * inner));
  if (dims.y < 32) dims.y = 32;
  return dims;
}

// ────────────────────────────────────────────────────────────────────────────
// Forward dispatch
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static void segmentSoftmaxCuda_(LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out) {
  auto* context = logits.getContext();
  auto* stream = context->getCudaStream();
  const LongType N = segmentIds.lengthOf();
  const segment_sem::SegMat lm = segment_sem::makeSegMat(logits.shapeInfo(), logits.specialShapeInfo());
  const segment_sem::SegMat om = segment_sem::makeSegMat(out.shapeInfo(), out.specialShapeInfo());
  const LongType inner = lm.inner;
  if (K <= 0 || N == 0 || inner == 0) return;

  PointersManager manager(context, "segmentSoftmax");
  auto* ids = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(N) * sizeof(LongType)));
  auto* begin = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(K) * sizeof(LongType)));
  auto* end = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(K) * sizeof(LongType)));
  segmentReadIds(context, &segmentIds, ids);
  segmentBuildRanges(context, ids, N, K, begin, end);

  dim3 dims = softmaxDims(K, inner, N);
  segSoftmaxKernel<X><<<dims.x, dims.y, dims.z, *stream>>>(reinterpret_cast<const X*>(logits.specialBuffer()), lm,
                                                          reinterpret_cast<X*>(out.specialBuffer()), om, begin, end, K,
                                                          inner);
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("segSoftmaxKernel failed");
}

// ────────────────────────────────────────────────────────────────────────────
// Backward dispatch
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static void segmentSoftmaxBpCuda_(LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out, NDArray& gradOut,
                                  NDArray& dLogits) {
  auto* context = logits.getContext();
  auto* stream = context->getCudaStream();
  const LongType N = segmentIds.lengthOf();
  const segment_sem::SegMat fm = segment_sem::makeSegMat(out.shapeInfo(), out.specialShapeInfo());
  const segment_sem::SegMat gm = segment_sem::makeSegMat(gradOut.shapeInfo(), gradOut.specialShapeInfo());
  const segment_sem::SegMat dm = segment_sem::makeSegMat(dLogits.shapeInfo(), dLogits.specialShapeInfo());
  const LongType inner = dm.inner;
  if (K <= 0 || N == 0 || inner == 0) return;

  PointersManager manager(context, "segmentSoftmaxBp");
  auto* ids = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(N) * sizeof(LongType)));
  auto* begin = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(K) * sizeof(LongType)));
  auto* end = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(K) * sizeof(LongType)));
  segmentReadIds(context, &segmentIds, ids);
  segmentBuildRanges(context, ids, N, K, begin, end);

  dim3 dims = softmaxDims(K, inner, N);
  segSoftmaxBpKernel<X><<<dims.x, dims.y, dims.z, *stream>>>(
      reinterpret_cast<const X*>(out.specialBuffer()), fm, reinterpret_cast<const X*>(gradOut.specialBuffer()), gm,
      reinterpret_cast<X*>(dLogits.specialBuffer()), dm, begin, end, K, inner);
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("segSoftmaxBpKernel failed");
}

// ────────────────────────────────────────────────────────────────────────────
// Public interface
// ────────────────────────────────────────────────────────────────────────────

void segmentSoftmax(sd::LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out) {
  NDArray::prepareSpecialUse({&out}, {&logits, &segmentIds});
  BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmaxCuda_, (K, logits, segmentIds, out), SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&out}, {&logits, &segmentIds});
}

void segmentSoftmaxBp(sd::LongType K, NDArray& logits, NDArray& segmentIds, NDArray& out, NDArray& gradOut,
                      NDArray& dLogits) {
  NDArray::prepareSpecialUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
  BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmaxBpCuda_, (K, logits, segmentIds, out, gradOut, dLogits),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
