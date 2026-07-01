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
// Algorithm for forward (1D logits):
//   For each segment s (a contiguous run of rows with the same segmentId):
//     1. max_s = max(logits[i] for i in seg(s))
//     2. expSum_s = sum(exp(logits[i] - max_s) for i in seg(s))
//     3. out[i] = exp(logits[i] - max_s) / expSum_s
//
// For higher-rank logits [N, ...] the softmax is applied independently to each
// "feature vector" across the node dimension (axis 0).
//
// Backward:
//   dLogits[i] = out[i] * (gradOut[i] - dotProd_s)
//   where dotProd_s = sum_{j in seg(s)} gradOut[j] * out[j]
//

#include <ops/declarable/helpers/segment_softmax.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>

#include <cstring>
#include <vector>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// 1D forward helper
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static void segmentSoftmax1D_(
    sd::LongType K,
    NDArray& logits,
    NDArray& segmentIds,
    NDArray& out)
{
    const X*   lBuf  = logits.bufferAsT<X>();
    const int* sBuf  = segmentIds.bufferAsT<int>();
    X*         oBuf  = out.bufferAsT<X>();

    const sd::LongType N   = logits.lengthOf();
    const sd::LongType ls0 = logits.strideAt(0);
    const sd::LongType os0 = out.strideAt(0);

    // Find segment start/end by scanning sorted segmentIds
    sd::LongType i = 0;
    while (i < N) {
        const int seg = sBuf[i];
        // find end of this segment
        sd::LongType j = i;
        while (j < N && sBuf[j] == seg) ++j;
        // [i, j) belongs to segment seg

        // Pass 1: max
        X maxVal = lBuf[i * ls0];
        for (sd::LongType q = i + 1; q < j; ++q) {
            X v = lBuf[q * ls0];
            if (v > maxVal) maxVal = v;
        }
        // Pass 2: exp sum
        X expSum = static_cast<X>(0);
        for (sd::LongType q = i; q < j; ++q) {
            X e = sd::math::sd_exp<X, X>(lBuf[q * ls0] - maxVal);
            oBuf[q * os0] = e;
            expSum += e;
        }
        // Pass 3: normalize
        for (sd::LongType q = i; q < j; ++q) {
            oBuf[q * os0] /= expSum;
        }
        i = j;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ND forward helper  (logits [N, d1, d2, ...])
// For each position in the trailing dims, do 1D segment softmax over dim 0.
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static void segmentSoftmaxND_(
    sd::LongType K,
    NDArray& logits,
    NDArray& segmentIds,
    NDArray& out)
{
    const int*   sBuf = segmentIds.bufferAsT<int>();
    const sd::LongType N     = logits.sizeAt(0);
    const sd::LongType inner = logits.lengthOf() / N;  // product of trailing dims

    for (sd::LongType f = 0; f < inner; ++f) {
        sd::LongType i = 0;
        while (i < N) {
            const int seg = sBuf[i];
            sd::LongType j = i;
            while (j < N && sBuf[j] == seg) ++j;

            // compute offset of element (row, f) for general strides
            // logits is [N, inner], but inner may be multi-dim
            // Use NDArray::e<X>(lin_idx) for correctness with arbitrary strides
            X maxVal = logits.e<X>(i * inner + f);
            for (sd::LongType q = i + 1; q < j; ++q) {
                X v = logits.e<X>(q * inner + f);
                if (v > maxVal) maxVal = v;
            }
            X expSum = static_cast<X>(0);
            for (sd::LongType q = i; q < j; ++q) {
                X e = sd::math::sd_exp<X, X>(logits.e<X>(q * inner + f) - maxVal);
                out.p<X>(q * inner + f, e);
                expSum += e;
            }
            for (sd::LongType q = i; q < j; ++q) {
                out.p<X>(q * inner + f, out.e<X>(q * inner + f) / expSum);
            }
            i = j;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

template <typename X>
static void segmentSoftmaxBp_(
    sd::LongType K,
    NDArray& logits,
    NDArray& segmentIds,
    NDArray& out,
    NDArray& gradOut,
    NDArray& dLogits)
{
    const int*   sBuf  = segmentIds.bufferAsT<int>();
    const sd::LongType N     = logits.sizeAt(0);
    const sd::LongType inner = logits.lengthOf() / N;

    // For each feature position and each segment:
    //   dotProd = sum_{j in seg} gradOut[j,f] * out[j,f]
    //   dLogits[i,f] = out[i,f] * (gradOut[i,f] - dotProd)

    for (sd::LongType f = 0; f < inner; ++f) {
        sd::LongType i = 0;
        while (i < N) {
            const int seg = sBuf[i];
            sd::LongType j = i;
            while (j < N && sBuf[j] == seg) ++j;

            X dotProd = static_cast<X>(0);
            for (sd::LongType q = i; q < j; ++q) {
                dotProd += gradOut.e<X>(q * inner + f) * out.e<X>(q * inner + f);
            }
            for (sd::LongType q = i; q < j; ++q) {
                X o = out.e<X>(q * inner + f);
                X g = gradOut.e<X>(q * inner + f);
                dLogits.p<X>(q * inner + f, o * (g - dotProd));
            }
            i = j;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public interface
// ────────────────────────────────────────────────────────────────────────────

void segmentSoftmax(sd::LongType K,
                    NDArray& logits,
                    NDArray& segmentIds,
                    NDArray& out) {
    NDArray::preparePrimaryUse({&out}, {&logits, &segmentIds});

    if (logits.rankOf() == 1) {
        BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmax1D_,
                              (K, logits, segmentIds, out),
                              SD_FLOAT_TYPES);
    } else {
        BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmaxND_,
                              (K, logits, segmentIds, out),
                              SD_FLOAT_TYPES);
    }

    NDArray::registerPrimaryUse({&out}, {&logits, &segmentIds});
}

void segmentSoftmaxBp(sd::LongType K,
                      NDArray& logits,
                      NDArray& segmentIds,
                      NDArray& out,
                      NDArray& gradOut,
                      NDArray& dLogits) {
    NDArray::preparePrimaryUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});

    BUILD_SINGLE_SELECTOR(logits.dataType(), segmentSoftmaxBp_,
                          (K, logits, segmentIds, out, gradOut, dLogits),
                          SD_FLOAT_TYPES);

    NDArray::registerPrimaryUse({&dLogits}, {&logits, &segmentIds, &out, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
