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
// We use a two-pass approach per segment:
//   Pass 1: compute per-segment max (reduction within block, then serial scan across segments)
//   Pass 2: compute exp(x - max) and sum (reduction)
//   Pass 3: normalize
//
// For GNN batch sizes (K=8..64, N_k=10..5000 per graph), a segment typically has
// 10..5000 elements. We launch one CUDA block per (segment, feature) pair.
// Within each block, warp shuffles reduce max and sum.
//
// For simplicity we launch a single kernel for all K segments sequentially
// (correct), with a second dedicated segment_reduce kernel for large segments.
// A production impl would use CUB DeviceSegmentedReduce.
//

#include <cuda_runtime.h>
#include <array/DataTypeUtils.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/segment_softmax.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Warp-level max reduction
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
SD_DEVICE SD_INLINE X warpReduceMax(X val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = sd::math::sd_max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Warp-level sum reduction
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
SD_DEVICE SD_INLINE X warpReduceSum(X val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ────────────────────────────────────────────────────────────────────────────
// Forward kernel: one block per (segment, inner_feature).
// segStart[s], segEnd[s] passed as flat arrays.
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void segSoftmaxKernel(
    const X*   logBuf,    // [N, inner]  or 1D [N]
    X*         outBuf,    // same shape
    const int* sBuf,      // segmentIds [N] sorted
    LongType   N,
    LongType   inner,     // trailing size (1 for 1D)
    LongType   ls0, LongType ls1,  // strides of logits
    LongType   os0, LongType os1)  // strides of out
{
    // blockIdx.x = segment index s
    // blockIdx.y = feature index f
    const int s = blockIdx.x;
    const LongType f = blockIdx.y;

    // Find segment range by scanning (OK for small K):
    // Segments are sorted contiguous runs; scan from beginning.
    // For better perf a segStart/segEnd array would be better, but this is correct.
    LongType segStart = 0, segEnd = 0;
    {
        int cur = 0;
        LongType i = 0;
        while (i < N && cur < s) {
            int seg = sBuf[i];
            while (i < N && sBuf[i] == seg) ++i;
            ++cur;
        }
        segStart = i;
        if (i < N) {
            int seg = sBuf[i];
            while (i < N && sBuf[i] == seg) ++i;
        }
        segEnd = i;
    }

    const LongType segLen = segEnd - segStart;
    if (segLen == 0 || f >= inner) return;

    // Each thread handles a stride-spaced subset of the segment elements.
    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    // Pass 1: block-level max
    X localMax = -DataTypeUtils::max<X>();
    for (LongType q = tid; q < segLen; q += bsz) {
        X v = logBuf[(segStart + q) * ls0 + f * ls1];
        if (v > localMax) localMax = v;
    }
    localMax = warpReduceMax<X>(localMax);
    // Share across warps via shared memory
    extern __shared__ char smem[];
    X* shared = reinterpret_cast<X*>(smem);
    int warp = tid / 32;
    int lane = tid % 32;
    if (lane == 0) shared[warp] = localMax;
    __syncthreads();
    if (warp == 0) {
        localMax = (lane < (bsz + 31) / 32) ? shared[lane] : -DataTypeUtils::max<X>();
        localMax = warpReduceMax<X>(localMax);
        if (lane == 0) shared[0] = localMax;
    }
    __syncthreads();
    X maxVal = shared[0];

    // Pass 2: exp sum
    X localSum = static_cast<X>(0);
    for (LongType q = tid; q < segLen; q += bsz) {
        X e = sd::math::sd_exp<X, X>(logBuf[(segStart + q) * ls0 + f * ls1] - maxVal);
        outBuf[(segStart + q) * os0 + f * os1] = e;
        localSum += e;
    }
    localSum = warpReduceSum<X>(localSum);
    if (lane == 0) shared[warp] = localSum;
    __syncthreads();
    if (warp == 0) {
        localSum = (lane < (bsz + 31) / 32) ? shared[lane] : static_cast<X>(0);
        localSum = warpReduceSum<X>(localSum);
        if (lane == 0) shared[0] = localSum;
    }
    __syncthreads();
    X expSum = shared[0];

    // Pass 3: normalize
    for (LongType q = tid; q < segLen; q += bsz) {
        outBuf[(segStart + q) * os0 + f * os1] /= expSum;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel: one block per (segment, feature)
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void segSoftmaxBpKernel(
    const X*   outBuf,     // forward output [N, inner]
    const X*   grBuf,      // gradOut [N, inner]
    X*         dlBuf,      // dLogits [N, inner]
    const int* sBuf,       // segmentIds [N]
    LongType   N,
    LongType   inner,
    LongType   os0, LongType os1,
    LongType   gs0, LongType gs1,
    LongType   ds0, LongType ds1)
{
    const int s = blockIdx.x;
    const LongType f = blockIdx.y;

    LongType segStart = 0, segEnd = 0;
    {
        int cur = 0;
        LongType i = 0;
        while (i < N && cur < s) {
            int seg = sBuf[i];
            while (i < N && sBuf[i] == seg) ++i;
            ++cur;
        }
        segStart = i;
        if (i < N) {
            int seg = sBuf[i];
            while (i < N && sBuf[i] == seg) ++i;
        }
        segEnd = i;
    }

    const LongType segLen = segEnd - segStart;
    if (segLen == 0 || f >= inner) return;

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    // Compute dot product sum_{j in seg} gradOut[j]*out[j]
    X localDot = static_cast<X>(0);
    for (LongType q = tid; q < segLen; q += bsz) {
        localDot += grBuf[(segStart + q) * gs0 + f * gs1]
                  * outBuf[(segStart + q) * os0 + f * os1];
    }
    localDot = warpReduceSum<X>(localDot);
    extern __shared__ char smem[];
    X* shared = reinterpret_cast<X*>(smem);
    int warp = tid / 32;
    int lane = tid % 32;
    if (lane == 0) shared[warp] = localDot;
    __syncthreads();
    if (warp == 0) {
        localDot = (lane < (bsz + 31) / 32) ? shared[lane] : static_cast<X>(0);
        localDot = warpReduceSum<X>(localDot);
        if (lane == 0) shared[0] = localDot;
    }
    __syncthreads();
    X dotProd = shared[0];

    for (LongType q = tid; q < segLen; q += bsz) {
        X o = outBuf[(segStart + q) * os0 + f * os1];
        X g = grBuf[(segStart + q) * gs0 + f * gs1];
        dlBuf[(segStart + q) * ds0 + f * ds1] = o * (g - dotProd);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Forward dispatch
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static void segmentSoftmaxCuda_(
    sd::LongType K,
    NDArray& logits,
    NDArray& segmentIds,
    NDArray& out)
{
    auto* stream = logits.getContext()->getCudaStream();
    const X*   logBuf = reinterpret_cast<const X*>(logits.specialBuffer());
    X*         outBuf = reinterpret_cast<X*>(out.specialBuffer());
    const int* sBuf   = reinterpret_cast<const int*>(segmentIds.specialBuffer());

    const LongType N     = logits.sizeAt(0);
    const LongType inner = logits.lengthOf() / N;

    // Grid: K segments x inner features; block: min(256, N) threads
    const int bsz = 256;
    const int nWarps = (bsz + 31) / 32;
    const size_t smem = nWarps * sizeof(X);

    dim3 grid(static_cast<unsigned>(K), static_cast<unsigned>(inner));
    dim3 block(bsz);

    LongType ls0 = (logits.rankOf() >= 1) ? logits.strideAt(0) : 1;
    LongType ls1 = (logits.rankOf() >= 2) ? logits.strideAt(1) : 1;
    LongType os0 = (out.rankOf() >= 1) ? out.strideAt(0) : 1;
    LongType os1 = (out.rankOf() >= 2) ? out.strideAt(1) : 1;

    segSoftmaxKernel<X><<<grid, block, smem, *stream>>>(
        logBuf, outBuf, sBuf, N, inner, ls0, ls1, os0, os1);
}

// ────────────────────────────────────────────────────────────────────────────
// Backward dispatch
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static void segmentSoftmaxBpCuda_(
    sd::LongType K,
    NDArray& logits,
    NDArray& segmentIds,
    NDArray& out,
    NDArray& gradOut,
    NDArray& dLogits)
{
    auto* stream = logits.getContext()->getCudaStream();
    const X*   outBuf = reinterpret_cast<const X*>(out.specialBuffer());
    const X*   grBuf  = reinterpret_cast<const X*>(gradOut.specialBuffer());
    X*         dlBuf  = reinterpret_cast<X*>(dLogits.specialBuffer());
    const int* sBuf   = reinterpret_cast<const int*>(segmentIds.specialBuffer());

    const LongType N     = logits.sizeAt(0);
    const LongType inner = logits.lengthOf() / N;

    const int bsz = 256;
    const int nWarps = (bsz + 31) / 32;
    const size_t smem = nWarps * sizeof(X);

    dim3 grid(static_cast<unsigned>(K), static_cast<unsigned>(inner));
    dim3 block(bsz);

    LongType os0 = out.strideAt(0);
    LongType os1 = (out.rankOf() >= 2) ? out.strideAt(1) : 1;
    LongType gs0 = gradOut.strideAt(0);
    LongType gs1 = (gradOut.rankOf() >= 2) ? gradOut.strideAt(1) : 1;
    LongType ds0 = dLogits.strideAt(0);
    LongType ds1 = (dLogits.rankOf() >= 2) ? dLogits.strideAt(1) : 1;

    segSoftmaxBpKernel<X><<<grid, block, smem, *stream>>>(
        outBuf, grBuf, dlBuf, sBuf, N, inner, os0, os1, gs0, gs1, ds0, ds1);
}

// ────────────────────────────────────────────────────────────────────────────
// Public interface
// ────────────────────────────────────────────────────────────────────────────

void segmentSoftmax(sd::LongType K,
                    NDArray& logits,
                    NDArray& segmentIds,
                    NDArray& out) {
    NDArray::prepareSpecialUse({&out}, {&logits, &segmentIds});
    BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmaxCuda_,
                          (K, logits, segmentIds, out),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({&out}, {&logits, &segmentIds});
}

void segmentSoftmaxBp(sd::LongType K,
                      NDArray& logits,
                      NDArray& segmentIds,
                      NDArray& out,
                      NDArray& gradOut,
                      NDArray& dLogits) {
    NDArray::prepareSpecialUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
    BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmaxBpCuda_,
                          (K, logits, segmentIds, out, gradOut, dLogits),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
