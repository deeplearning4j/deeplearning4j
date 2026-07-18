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
#include <ops/declarable/helpers/rwkv_wkv.h>

namespace sd {
namespace ops {

#if NOT_EXCLUDED(OP_rwkv_wkv6)
CUSTOM_OP_IMPL(rwkv_wkv6, 6, 1, false, 0, 0) {
    auto k = INPUT_VARIABLE(0);
    auto v = INPUT_VARIABLE(1);
    auto r = INPUT_VARIABLE(2);
    auto tf = INPUT_VARIABLE(3);
    auto td = INPUT_VARIABLE(4);
    auto state = INPUT_VARIABLE(5);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(k->rankOf() == 4, 0, "rwkv_wkv6: k must be rank 4 [B,T,H,S], got %i", k->rankOf());
    REQUIRE_TRUE(v->isSameShape(k) && r->isSameShape(k) && td->isSameShape(k), 0,
                 "rwkv_wkv6: k, v, r, td must all be [B,T,H,S]");
    REQUIRE_TRUE(tf->rankOf() == 2 && tf->sizeAt(0) == k->sizeAt(2) && tf->sizeAt(1) == k->sizeAt(3), 0,
                 "rwkv_wkv6: tf must be [H,S]");
    REQUIRE_TRUE(state->rankOf() == 4 && state->sizeAt(0) == k->sizeAt(0) &&
                     state->sizeAt(1) == k->sizeAt(2) && state->sizeAt(2) == k->sizeAt(3) &&
                     state->sizeAt(3) == k->sizeAt(3), 0,
                 "rwkv_wkv6: state must be [B,H,S,S]");
    if (k->isEmpty()) return Status::OK;

    helpers::rwkvWkv6(block.launchContext(), k, v, r, tf, td, state, output);
    return Status::OK;
}
DECLARE_TYPES(rwkv_wkv6) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(rwkv_wkv6) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

#if NOT_EXCLUDED(OP_rwkv_wkv7)
CUSTOM_OP_IMPL(rwkv_wkv7, 7, 1, false, 0, 0) {
    auto r = INPUT_VARIABLE(0);
    auto w = INPUT_VARIABLE(1);
    auto k = INPUT_VARIABLE(2);
    auto v = INPUT_VARIABLE(3);
    auto a = INPUT_VARIABLE(4);
    auto b = INPUT_VARIABLE(5);
    auto state = INPUT_VARIABLE(6);
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(k->rankOf() == 4, 0, "rwkv_wkv7: k must be rank 4 [B,T,H,S], got %i", k->rankOf());
    REQUIRE_TRUE(r->isSameShape(k) && w->isSameShape(k) && v->isSameShape(k) &&
                     a->isSameShape(k) && b->isSameShape(k), 0,
                 "rwkv_wkv7: r, w, k, v, a, b must all be [B,T,H,S]");
    REQUIRE_TRUE(state->rankOf() == 4 && state->sizeAt(0) == k->sizeAt(0) &&
                     state->sizeAt(1) == k->sizeAt(2) && state->sizeAt(2) == k->sizeAt(3) &&
                     state->sizeAt(3) == k->sizeAt(3), 0,
                 "rwkv_wkv7: state must be [B,H,S,S]");
    if (k->isEmpty()) return Status::OK;

    helpers::rwkvWkv7(block.launchContext(), r, w, k, v, a, b, state, output);
    return Status::OK;
}
DECLARE_TYPES(rwkv_wkv7) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(rwkv_wkv7) {
    // output has k's shape; k is input index 2
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(2))->primary());
}
#endif

}  // namespace ops
}  // namespace sd
