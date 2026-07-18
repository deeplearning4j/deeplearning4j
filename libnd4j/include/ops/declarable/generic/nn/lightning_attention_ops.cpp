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
#if NOT_EXCLUDED(OP_lightning_attention)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/lightning_attention.h>

namespace sd {
namespace ops {

/**
 * Lightning Attention — O(N) linear attention with per-head exponential decay
 * via intra/inter-chunk decomposition.
 *
 * Inputs:
 *   0: query      [batch, seqLen, numHeads, headDim]
 *   1: key        [batch, seqLen, numHeads, headDim]
 *   2: value      [batch, seqLen, numHeads, headDim]
 *   3: decayRates [numHeads]  (per-head scalar decay rate, float32)
 *   4: state      [batch, numHeads, headDim, headDim] (recurrent state, float32, in/out)
 *
 * Output:
 *   0: attention output [batch, seqLen, numHeads, headDim]
 *
 * Int args:
 *   0: isCausal (0 or 1, default 1)
 */
CUSTOM_OP_IMPL(lightning_attention, 5, 1, false, 0, 0) {
    auto query = INPUT_VARIABLE(0);
    auto key = INPUT_VARIABLE(1);
    auto value = INPUT_VARIABLE(2);
    auto decayRates = INPUT_VARIABLE(3);
    auto state = INPUT_VARIABLE(4);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(query->rankOf() == 4, 0,
                 "lightning_attention: query must be rank 4 [batch, seqLen, numHeads, headDim], got %lld",
                 (long long)query->rankOf());
    REQUIRE_TRUE(key->rankOf() == 4, 0,
                 "lightning_attention: key must be rank 4, got %lld",
                 (long long)key->rankOf());
    REQUIRE_TRUE(value->rankOf() == 4, 0,
                 "lightning_attention: value must be rank 4, got %lld",
                 (long long)value->rankOf());
    REQUIRE_TRUE(decayRates->rankOf() == 1, 0,
                 "lightning_attention: decayRates must be rank 1 [numHeads], got %lld",
                 (long long)decayRates->rankOf());
    REQUIRE_TRUE(state->rankOf() == 4, 0,
                 "lightning_attention: state must be rank 4 [batch, numHeads, headDim, headDim], got %lld",
                 (long long)state->rankOf());

    bool isCausal = block.getIArguments()->size() > 0 ? (I_ARG(0) != 0) : true;

    helpers::lightningAttention(block.launchContext(), query, key, value, decayRates, state, output, isCausal);

    return sd::Status::OK;
}

DECLARE_TYPES(lightning_attention) {
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(lightning_attention) {
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inShape)->primary());
}

}  // namespace ops
}  // namespace sd

#endif
