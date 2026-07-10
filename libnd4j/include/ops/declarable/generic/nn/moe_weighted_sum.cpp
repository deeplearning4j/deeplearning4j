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

//
// @author Eclipse Deeplearning4j
//
// moe_weighted_sum — fused top-K expert accumulation without holding [T,topK,D] downstream.
//
// Two layouts are supported (iArg 0 selects the mode):
//
//  Mode 0 — Dense:   expertOutputs [T, topK, D],  weights [T, topK]
//  Mode 1 — Flat:    expertOutputs [totalRows, D], weights [T, topK],
//                    rowIndex [totalRows, 2] INT32/INT64 = {token_idx, k_idx}
//
// The flat layout composes directly with segment_gemm, whose output is
// [totalTokens, outDim] sorted by expert index (see segment_gemm_ops.cpp).
// To use it: run segment_gemm, build rowIndex mapping each output row back
// to (token, k) slot, then call moe_weighted_sum with mode=1.
//
// Output: [T, D] — always fp32 accumulated, cast to input element type.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_moe_weighted_sum)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/moe_weighted_sum.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(moe_weighted_sum, 2, 1, false, 0, 1) {
    // iArg[0]: layout mode — 0 = dense [T,topK,D], 1 = flat [totalRows,D]
    const int mode = INT_ARG(0);

    auto expertOutputs = INPUT_VARIABLE(0);
    auto weights       = INPUT_VARIABLE(1);
    auto output        = OUTPUT_VARIABLE(0);

    if (mode == 0) {
        // Dense: expertOutputs [T, topK, D], weights [T, topK]
        REQUIRE_TRUE(expertOutputs->rankOf() == 3, 0,
            "moe_weighted_sum (dense): expertOutputs must be rank 3 [T, topK, D], got %i",
            expertOutputs->rankOf());
        REQUIRE_TRUE(weights->rankOf() == 2, 0,
            "moe_weighted_sum (dense): weights must be rank 2 [T, topK], got %i",
            weights->rankOf());
        REQUIRE_TRUE(expertOutputs->sizeAt(0) == weights->sizeAt(0), 0,
            "moe_weighted_sum (dense): T mismatch: expertOutputs T=%lld, weights T=%lld",
            (long long)expertOutputs->sizeAt(0), (long long)weights->sizeAt(0));
        REQUIRE_TRUE(expertOutputs->sizeAt(1) == weights->sizeAt(1), 0,
            "moe_weighted_sum (dense): topK mismatch: expertOutputs topK=%lld, weights topK=%lld",
            (long long)expertOutputs->sizeAt(1), (long long)weights->sizeAt(1));

        helpers::moeWeightedSumDense(block.launchContext(), expertOutputs, weights, output);

    } else {
        // Flat: expertOutputs [totalRows, D], weights [T, topK], rowIndex [totalRows, 2]
        REQUIRE_TRUE(block.width() >= 3, 0,
            "moe_weighted_sum (flat, mode=1): requires 3 inputs: expertOutputs, weights, rowIndex");

        auto rowIndex = INPUT_VARIABLE(2);

        REQUIRE_TRUE(expertOutputs->rankOf() == 2, 0,
            "moe_weighted_sum (flat): expertOutputs must be rank 2 [totalRows, D], got %i",
            expertOutputs->rankOf());
        REQUIRE_TRUE(weights->rankOf() == 2, 0,
            "moe_weighted_sum (flat): weights must be rank 2 [T, topK], got %i",
            weights->rankOf());
        REQUIRE_TRUE(rowIndex->rankOf() == 2 && rowIndex->sizeAt(1) == 2, 0,
            "moe_weighted_sum (flat): rowIndex must be rank 2 with shape [totalRows, 2], got rank=%i sizeAt(1)=%lld",
            rowIndex->rankOf(), (long long)rowIndex->sizeAt(1));
        REQUIRE_TRUE(expertOutputs->sizeAt(0) == rowIndex->sizeAt(0), 0,
            "moe_weighted_sum (flat): totalRows mismatch: expertOutputs=%lld, rowIndex=%lld",
            (long long)expertOutputs->sizeAt(0), (long long)rowIndex->sizeAt(0));

        helpers::moeWeightedSumFlat(block.launchContext(), expertOutputs, weights, rowIndex, output);
    }

    return sd::Status::OK;
}

DECLARE_TYPES(moe_weighted_sum) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedInputTypes(2, {INT32, INT64})  // rowIndex (flat mode)
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(moe_weighted_sum) {
    const int mode = INT_ARG(0);

    auto eoShape  = inputShape->at(0);   // expert outputs
    auto wShape   = inputShape->at(1);   // weights

    // Output: [T, D]
    sd::LongType T, D;

    if (mode == 0) {
        // Dense: expertOutputs [T, topK, D]
        T = shape::sizeAt(eoShape, static_cast<sd::LongType>(0));
        D = shape::sizeAt(eoShape, static_cast<sd::LongType>(2));
    } else {
        // Flat: expertOutputs [totalRows, D], weights [T, topK]
        T = shape::sizeAt(wShape, static_cast<sd::LongType>(0));
        D = shape::sizeAt(eoShape, static_cast<sd::LongType>(1));
    }

    auto outType = sd::ArrayOptions::dataType(eoShape);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(outType, 'c', {T, D}));
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_moe_weighted_sum)
