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
#if NOT_EXCLUDED(OP_top_k_renorm)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/top_k_renorm.h>

namespace sd {
namespace ops {

/**
 * Top-K filtering with renormalization.
 *
 * Keeps only the top-K highest-probability tokens, zeros the rest,
 * then renormalizes so the kept probabilities sum to 1.
 *
 * Inputs:
 *   0: logits [batch, vocabSize] or [vocabSize] — input logits (pre-softmax)
 *
 * Output:
 *   0: renormalized probabilities (same shape as input)
 *
 * Int args:
 *   0: k — number of top tokens to keep
 */
CUSTOM_OP_IMPL(top_k_renorm, 1, 1, false, 0, 1) {
    auto logits = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    auto logitsRank = logits->rankOf();
    REQUIRE_TRUE(logitsRank == 1 || logitsRank == 2, 0,
                 "top_k_renorm: logits must be rank 1 or 2, got %lld",
                 (long long)logitsRank);

    int k = INT_ARG(0);
    REQUIRE_TRUE(k > 0, 0, "top_k_renorm: k must be positive, got %d", k);

    helpers::topKRenorm(block.launchContext(), logits, output, k);

    return sd::Status::OK;
}

DECLARE_TYPES(top_k_renorm) {
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(top_k_renorm) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_top_p_renorm)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/top_k_renorm.h>

namespace sd {
namespace ops {

/**
 * Top-P (nucleus) filtering with renormalization.
 *
 * Sorts tokens by descending probability, accumulates until cumulative
 * probability >= p, zeros the rest, then renormalizes.
 *
 * Inputs:
 *   0: logits [batch, vocabSize] or [vocabSize] — input logits (pre-softmax)
 *
 * Output:
 *   0: renormalized probabilities (same shape as input)
 *
 * Float args:
 *   0: p — cumulative probability threshold (0.0-1.0)
 */
CUSTOM_OP_IMPL(top_p_renorm, 1, 1, false, 1, 0) {
    auto logits = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    auto logitsRank = logits->rankOf();
    REQUIRE_TRUE(logitsRank == 1 || logitsRank == 2, 0,
                 "top_p_renorm: logits must be rank 1 or 2, got %lld",
                 (long long)logitsRank);

    double p = T_ARG(0);
    REQUIRE_TRUE(p > 0.0 && p <= 1.0, 0,
                 "top_p_renorm: p must be in (0.0, 1.0], got %f", p);

    helpers::topPRenorm(block.launchContext(), logits, output, p);

    return sd::Status::OK;
}

DECLARE_TYPES(top_p_renorm) {
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(top_p_renorm) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif
