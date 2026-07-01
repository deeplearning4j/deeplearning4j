/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// segment_softmax: softmax within each segment (for batched GAT attention).
//
// Inputs:
//   logits      [N] or [N, ...] float — values to softmax
//   segmentIds  [N] int32 — sorted segment assignment (0-based, values in [0, K))
// IArgs:
//   [0] K — number of segments
// Output:
//   out [N] or [N, ...] — softmax(logits) within each segment
//
// Backward (segment_softmax_bp):
//   dLogits[i] = out[i] * (gradOut[i] - sum_{j in seg(i)} gradOut[j]*out[j])
//

#ifndef SAMEDIFF_SEGMENT_SOFTMAX_H
#define SAMEDIFF_SEGMENT_SOFTMAX_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Forward: softmax within each segment.
// segmentIds must be sorted non-decreasing.
SD_LIB_HIDDEN void segmentSoftmax(sd::LongType K,
                                   NDArray& logits,
                                   NDArray& segmentIds,
                                   NDArray& out);

// Backward: gradient of segment softmax.
// out is the forward output (needed for the Jacobian-vector product).
SD_LIB_HIDDEN void segmentSoftmaxBp(sd::LongType K,
                                     NDArray& logits,
                                     NDArray& segmentIds,
                                     NDArray& out,
                                     NDArray& gradOut,
                                     NDArray& dLogits);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SEGMENT_SOFTMAX_H
