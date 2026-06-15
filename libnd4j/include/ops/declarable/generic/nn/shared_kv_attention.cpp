/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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
// Shared KV Attention (Gemma 4 pattern): grouped-query attention where
// K/V come from a donor layer rather than the current hidden state.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_shared_kv_attention)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/shared_kv_attention.h>

#include <cmath>

namespace sd {
namespace ops {

/**
 * shared_kv_attention
 *
 * Inputs:
 *   0: query       [batch, numHeads,   qSeqLen,  headDim]
 *   1: shared_key  [batch, numKvHeads, kvSeqLen, headDim]
 *   2: shared_value[batch, numKvHeads, kvSeqLen, headDim]
 *   3: mask        [batch, 1, qSeqLen, kvSeqLen]  (optional, additive)
 *
 * Int args:
 *   0: numHeads           — total query heads
 *   1: numKvHeads         — KV heads (numHeads must be divisible)
 *   2: causal             — 1 for causal masking, 0 for bidirectional
 *   3: slidingWindowSize  — sliding window size (0 = disabled)
 *
 * Float args:
 *   0: scale  — attention scale factor (0 = auto = 1/sqrt(headDim))
 *
 * Output:
 *   0: [batch, numHeads, qSeqLen, headDim]
 */
CUSTOM_OP_IMPL(shared_kv_attention, 3, 1, false, 0, 0) {
    auto query      = INPUT_VARIABLE(0);
    auto sharedKey  = INPUT_VARIABLE(1);
    auto sharedValue = INPUT_VARIABLE(2);
    auto mask       = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    auto output     = OUTPUT_VARIABLE(0);

    // Integer args with defaults
    int numHeads          = block.numI() > 0 ? INT_ARG(0) : static_cast<int>(query->sizeAt(1));
    int numKvHeads        = block.numI() > 1 ? INT_ARG(1) : static_cast<int>(sharedKey->sizeAt(1));
    bool causal           = block.numI() > 2 ? (INT_ARG(2) != 0) : false;
    int slidingWindowSize = block.numI() > 3 ? INT_ARG(3) : 0;

    // Scale: 0 or missing means auto = 1/sqrt(headDim)
    double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;
    if (scale == 0.0) {
        auto headDim = query->sizeAt(3);
        scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    }

    // Validate ranks
    REQUIRE_TRUE(query->rankOf() == 4, 0,
                 "shared_kv_attention: query must be rank 4 [batch, heads, qSeq, dim], got %i",
                 query->rankOf());
    REQUIRE_TRUE(sharedKey->rankOf() == 4, 0,
                 "shared_kv_attention: shared_key must be rank 4 [batch, kvHeads, kvSeq, dim], got %i",
                 sharedKey->rankOf());
    REQUIRE_TRUE(sharedValue->rankOf() == 4, 0,
                 "shared_kv_attention: shared_value must be rank 4 [batch, kvHeads, kvSeq, dim], got %i",
                 sharedValue->rankOf());

    // Validate GQA divisibility
    REQUIRE_TRUE(numHeads % numKvHeads == 0, 0,
                 "shared_kv_attention: numHeads (%d) must be divisible by numKvHeads (%d)",
                 numHeads, numKvHeads);

    // Validate headDim consistency
    REQUIRE_TRUE(query->sizeAt(3) == sharedKey->sizeAt(3), 0,
                 "shared_kv_attention: query headDim (%lld) must match key headDim (%lld)",
                 query->sizeAt(3), sharedKey->sizeAt(3));
    REQUIRE_TRUE(query->sizeAt(3) == sharedValue->sizeAt(3), 0,
                 "shared_kv_attention: query headDim (%lld) must match value headDim (%lld)",
                 query->sizeAt(3), sharedValue->sizeAt(3));

    // Validate KV sequence lengths match
    REQUIRE_TRUE(sharedKey->sizeAt(2) == sharedValue->sizeAt(2), 0,
                 "shared_kv_attention: key kvSeqLen (%lld) must match value kvSeqLen (%lld)",
                 sharedKey->sizeAt(2), sharedValue->sizeAt(2));

    // Handle empty optional mask
    if (mask != nullptr && mask->isEmpty()) {
        mask = nullptr;
    }

    helpers::sharedKvAttention(block.launchContext(),
                                query, sharedKey, sharedValue, mask, output,
                                numHeads, numKvHeads, causal, slidingWindowSize, scale);

    return sd::Status::OK;
}

DECLARE_TYPES(shared_kv_attention) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(shared_kv_attention) {
    // Output has the same shape as the query input
    return SHAPELIST(CONSTANT(inputShape->at(0)));
}

}  // namespace ops
}  // namespace sd

#endif
