/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_segment_gemm)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/segment_gemm.h>

namespace sd {
namespace ops {

/**
 * Segmented GEMM — variable-length batched matrix multiply for MoE dispatch.
 *
 * Inputs:
 *   0: input          [totalTokens, inDim] — concatenated token embeddings
 *   1: weights        [numExperts, inDim, outDim] — per-expert weight matrices
 *   2: segmentOffsets [numExperts] INT64 — start index of each expert's tokens
 *   3: segmentSizes   [numExperts] INT64 — token count for each expert
 *
 * Output:
 *   0: output [totalTokens, outDim]
 */
CUSTOM_OP_IMPL(segment_gemm, 4, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto segmentOffsets = INPUT_VARIABLE(2);
    auto segmentSizes = INPUT_VARIABLE(3);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(input->rankOf() == 2, 0,
                 "segment_gemm: input must be rank 2 [totalTokens, inDim], got %lld",
                 (long long)input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 3, 0,
                 "segment_gemm: weights must be rank 3 [numExperts, inDim, outDim], got %lld",
                 (long long)weights->rankOf());
    REQUIRE_TRUE(segmentOffsets->rankOf() == 1, 0,
                 "segment_gemm: segmentOffsets must be rank 1, got %lld",
                 (long long)segmentOffsets->rankOf());
    REQUIRE_TRUE(segmentSizes->rankOf() == 1, 0,
                 "segment_gemm: segmentSizes must be rank 1, got %lld",
                 (long long)segmentSizes->rankOf());

    helpers::segmentGemm(block.launchContext(), input, weights, segmentOffsets, segmentSizes, output);

    return sd::Status::OK;
}

DECLARE_TYPES(segment_gemm) {
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedInputTypes(1, {ALL_FLOATS});
    getOpDescriptor()->setAllowedInputTypes(2, {INT64, INT32});
    getOpDescriptor()->setAllowedInputTypes(3, {INT64, INT32});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(segment_gemm) {
    // Output: [totalTokens, outDim]
    auto inShape = inputShape->at(0);
    auto wShape = inputShape->at(1);

    auto totalTokens = shape::sizeAt(inShape, static_cast<LongType>(0));
    auto outDim = shape::sizeAt(wShape, static_cast<LongType>(2));

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inShape), 'c', {totalTokens, outDim}));
}

}  // namespace ops
}  // namespace sd

#endif
