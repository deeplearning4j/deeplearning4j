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

#include <system/op_boilerplate.h>

#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/gated_linear_attn.h>

#include <cmath>

namespace sd {
namespace ops {

#if NOT_EXCLUDED(OP_gated_linear_attn)
CUSTOM_OP_IMPL(gated_linear_attn, 3, 1, false, -2, 0) {
    auto q = INPUT_VARIABLE(0);
    auto k = INPUT_VARIABLE(1);
    auto v = INPUT_VARIABLE(2);
    NDArray* gate = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(q->rankOf() == 4, 0, "gated_linear_attn: q must be rank 4 [B,T,H,S], got %i", q->rankOf());
    REQUIRE_TRUE(k->isSameShape(q) && v->isSameShape(q), 0,
                 "gated_linear_attn: q, k, v must all be [B,T,H,S]");
    REQUIRE_TRUE(gate == nullptr || gate->isSameShape(q), 0,
                 "gated_linear_attn: gate (if present) must be [B,T,H,S]");
    if (q->isEmpty()) return Status::OK;

    const LongType S = q->sizeAt(3);
    const double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0 / std::sqrt((double)S);

    // GLA starts from zero recurrent state (matches ggml_gated_linear_attn).
    std::vector<LongType> stateShape = {q->sizeAt(0), q->sizeAt(2), S, S};
    NDArray state('c', stateShape, q->dataType(), block.launchContext());
    state.nullify();

    helpers::gatedLinearAttn(block.launchContext(), q, k, v, gate, &state, output, scale);
    return Status::OK;
}
DECLARE_TYPES(gated_linear_attn) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(gated_linear_attn) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

}  // namespace ops
}  // namespace sd
