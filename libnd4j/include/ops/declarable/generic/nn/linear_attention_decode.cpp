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
// linear_attention_decode — single-token decode step for linear attention.
//
// Implements the recurrent form of linear attention:
//   state_new = exp(-decay[h]) * state_old + k ⊗ v   (rank-1 outer-product update)
//   output    = q @ state_new                          (matrix-vector product)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_linear_attention_decode)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/linear_attention_decode.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(linear_attention_decode, 5, 1, false, 0, 0) {
    auto query      = INPUT_VARIABLE(0);  // [batch, 1, numHeads, headDimK]
    auto key        = INPUT_VARIABLE(1);  // [batch, 1, numHeads, headDimK]
    auto value      = INPUT_VARIABLE(2);  // [batch, 1, numHeads, headDimV]
    auto decayRates = INPUT_VARIABLE(3);  // [numHeads]
    auto state      = INPUT_VARIABLE(4);  // [batch, numHeads, headDimV, headDimK]
    auto output     = OUTPUT_VARIABLE(0); // [batch, 1, numHeads, headDimV]

    REQUIRE_TRUE(query->rankOf() == 4, 0,
                 "linear_attention_decode: query must be rank 4 [batch, 1, numHeads, headDimK], got %lld",
                 (long long)query->rankOf());
    REQUIRE_TRUE(key->rankOf() == 4, 0,
                 "linear_attention_decode: key must be rank 4, got %lld",
                 (long long)key->rankOf());
    REQUIRE_TRUE(value->rankOf() == 4, 0,
                 "linear_attention_decode: value must be rank 4, got %lld",
                 (long long)value->rankOf());
    REQUIRE_TRUE(decayRates->rankOf() == 1, 0,
                 "linear_attention_decode: decayRates must be rank 1 [numHeads], got %lld",
                 (long long)decayRates->rankOf());
    REQUIRE_TRUE(state->rankOf() == 4, 0,
                 "linear_attention_decode: state must be rank 4 [batch, numHeads, headDimV, headDimK], got %lld",
                 (long long)state->rankOf());
    REQUIRE_TRUE(query->sizeAt(1) == 1, 0,
                 "linear_attention_decode: seqLen dimension must be 1 for decode, got %lld",
                 (long long)query->sizeAt(1));

    helpers::linearAttentionDecode(block.launchContext(), query, key, value, decayRates, state, output);

    return sd::Status::OK;
}

DECLARE_TYPES(linear_attention_decode) {
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(linear_attention_decode) {
    auto queryShape = inputShape->at(0);  // [batch, 1, numHeads, headDimK]
    auto valueShape = inputShape->at(2);  // [batch, 1, numHeads, headDimV]

    auto batch    = shape::sizeAt(queryShape, 0);
    auto numHeads = shape::sizeAt(queryShape, 2);
    auto headDimV = shape::sizeAt(valueShape, 3);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(queryShape), 'c', {batch, 1, numHeads, headDimV});

    return SHAPELIST(outputShape);
}

}  // namespace ops
}  // namespace sd

#endif
