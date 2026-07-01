/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// segment_softmax — softmax within segments (for batched GAT attention).
//
// Inputs:
//   [0] logits     [N] or [N, d1, ...]  float  — attention logits
//   [1] segmentIds [N]                  INT32  — sorted segment IDs (0-based, in [0,K))
// IArgs:
//   [0] K — number of segments
// Output:
//   [0] out  — same shape as logits
//
// segment_softmax_bp — backward.
//
// Inputs:
//   [0] logits     [N, ...]  (shape only needed)
//   [1] segmentIds [N]       INT32
//   [2] out        [N, ...]  forward output
//   [3] gradOut    [N, ...]  upstream gradient
// IArgs:
//   [0] K
// Output:
//   [0] dLogits [N, ...]
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_segment_softmax)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/segment_softmax.h>

namespace sd {
namespace ops {

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_IMPL(segment_softmax, 2, 1, false, 0, 1) {
    auto logits     = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto out        = OUTPUT_NULLIFIED(0);
    const LongType K = INT_ARG(0);

    REQUIRE_TRUE(segmentIds->isVector(), 0,
                 "segment_softmax: segmentIds must be 1D, got rank %d", segmentIds->rankOf());
    REQUIRE_TRUE(segmentIds->lengthOf() == logits->sizeAt(0), 0,
                 "segment_softmax: segmentIds length (%lld) must equal logits.shape[0] (%lld)",
                 (long long)segmentIds->lengthOf(), (long long)logits->sizeAt(0));
    REQUIRE_TRUE(K >= 1, 0, "segment_softmax: K must be >= 1, got %lld", (long long)K);

    helpers::segmentSoftmax(K, *logits, *segmentIds, *out);
    return Status::OK;
}

DECLARE_SHAPE_FN(segment_softmax) {
    return SHAPELIST(CONSTANT(inputShape->at(0)));
}

DECLARE_TYPES(segment_softmax) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->setSameMode(false);
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

CUSTOM_OP_IMPL(segment_softmax_bp, 4, 1, false, 0, 1) {
    auto logits     = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto fwdOut     = INPUT_VARIABLE(2);
    auto gradOut    = INPUT_VARIABLE(3);
    auto dLogits    = OUTPUT_NULLIFIED(0);
    const LongType K = INT_ARG(0);

    helpers::segmentSoftmaxBp(K, *logits, *segmentIds, *fwdOut, *gradOut, *dLogits);
    return Status::OK;
}

DECLARE_SHAPE_FN(segment_softmax_bp) {
    return SHAPELIST(CONSTANT(inputShape->at(0)));
}

DECLARE_TYPES(segment_softmax_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})
        ->setAllowedInputTypes(3, {ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->setSameMode(false);
}

}  // namespace ops
}  // namespace sd
#endif  // NOT_EXCLUDED(OP_segment_softmax)
