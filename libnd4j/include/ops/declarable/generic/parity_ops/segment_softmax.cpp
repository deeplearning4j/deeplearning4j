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
#include <ops/declarable/helpers/segment.h>
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

    REQUIRE_TRUE(logits->rankOf() >= 1, 0, "segment_softmax: logits must have rank >= 1, got a scalar");
    REQUIRE_TRUE(segmentIds->isVector(), 0,
                 "segment_softmax: segmentIds must be 1D, got rank %d", segmentIds->rankOf());
    REQUIRE_TRUE(segmentIds->lengthOf() == logits->sizeAt(0), 0,
                 "segment_softmax: segmentIds length (%lld) must equal logits.shape[0] (%lld)",
                 (long long)segmentIds->lengthOf(), (long long)logits->sizeAt(0));
    REQUIRE_TRUE(K >= 1, 0, "segment_softmax: K must be >= 1, got %lld", (long long)K);
    REQUIRE_TRUE(out->dataType() == logits->dataType(), 0,
                 "segment_softmax: the output type (%s) must be the logits type (%s)",
                 DataTypeUtils::asString(out->dataType()).c_str(), DataTypeUtils::asString(logits->dataType()).c_str());

    // sorted ids in [0, K): a segment is one run of rows. Both checks read the ids where they live and are skipped
    // inside a CUDA graph capture (a host read of the last id would synchronize the captured stream); the kernels
    // ignore an id outside [0, K).
    LongType previous = 0;
    LongType offending = 0;
    REQUIRE_TRUE(helpers::segmentIndicesValidate(block.launchContext(), segmentIds, previous, offending), 0,
                 "segment_softmax: segmentIds must be non-negative and sorted, but the id %lld follows the id %lld",
                 (long long)offending, (long long)previous);
    LongType wrong = K;
    REQUIRE_TRUE(helpers::unsortedSegmentIndicesValidate(block.launchContext(), segmentIds, K, wrong), 0,
                 "segment_softmax: segmentIds must be in [0, K = %lld), but the id %lld is not", (long long)K,
                 (long long)wrong);

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

    REQUIRE_TRUE(logits->rankOf() >= 1, 0, "segment_softmax_bp: logits must have rank >= 1, got a scalar");
    REQUIRE_TRUE(segmentIds->isVector(), 0,
                 "segment_softmax_bp: segmentIds must be 1D, got rank %d", segmentIds->rankOf());
    REQUIRE_TRUE(segmentIds->lengthOf() == logits->sizeAt(0), 0,
                 "segment_softmax_bp: segmentIds length (%lld) must equal logits.shape[0] (%lld)",
                 (long long)segmentIds->lengthOf(), (long long)logits->sizeAt(0));
    REQUIRE_TRUE(K >= 1, 0, "segment_softmax_bp: K must be >= 1, got %lld", (long long)K);
    REQUIRE_TRUE(fwdOut->isSameShape(logits) && gradOut->isSameShape(logits), 0,
                 "segment_softmax_bp: the forward output and the gradient must have the shape of the logits");
    REQUIRE_TRUE(fwdOut->dataType() == logits->dataType() && gradOut->dataType() == logits->dataType() &&
                     dLogits->dataType() == logits->dataType(),
                 0, "segment_softmax_bp: the logits, the forward output, the gradient and the output must share one type");

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
