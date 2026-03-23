/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//
// gated_delta_rule - Gated Delta Network recurrent state update
//
// Implements the gated delta rule from arXiv:2412.06464 (ICLR 2025, NVIDIA Research):
//   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
//   output_t = S_t^T * q_t
//
// State shape: [batch, heads, d_k, d_v]
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gated_delta_rule)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/gated_delta_rule.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(gated_delta_rule, 5, 2, false, 0, 0) {
    auto Q = INPUT_VARIABLE(0);        // [B, L, H, D_k]
    auto K = INPUT_VARIABLE(1);        // [B, L, H, D_k]
    auto V = INPUT_VARIABLE(2);        // [B, L, H, D_v]
    auto beta = INPUT_VARIABLE(3);     // [B, L, H]
    auto gate = INPUT_VARIABLE(4);     // [B, L, H]

    auto output = OUTPUT_VARIABLE(0);     // [B, L, H, D_v]
    auto stateOut = OUTPUT_VARIABLE(1);   // [B, H, D_k, D_v]

    NDArray* stateIn = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;

    helpers::gatedDeltaRule(block.launchContext(), Q, K, V, beta, gate, stateIn, output, stateOut);

    return sd::Status::OK;
}

DECLARE_TYPES(gated_delta_rule) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(gated_delta_rule) {
    auto qShape = inputShape->at(0);  // [B, L, H, D_k]
    auto vShape = inputShape->at(2);  // [B, L, H, D_v]

    auto B = shape::sizeAt(qShape, 0);
    auto L = shape::sizeAt(qShape, 1);
    auto H = shape::sizeAt(qShape, 2);
    auto D_k = shape::sizeAt(qShape, 3);
    auto D_v = shape::sizeAt(vShape, 3);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(qShape), 'c', {B, L, H, D_v});

    auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(qShape), 'c', {B, H, D_k, D_v});

    return SHAPELIST(outputShape, stateShape);
}

}  // namespace ops
}  // namespace sd

#endif
