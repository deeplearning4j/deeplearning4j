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

    NDArray* stateIn = nullptr;
    NDArray* actualLen = nullptr;
    for (int i = 5; i < block.width(); ++i) {
        auto input = INPUT_VARIABLE(i);
        if (input->rankOf() == 0) {
            REQUIRE_TRUE(actualLen == nullptr, 0,
                         "gated_delta_rule: multiple scalar actualLen inputs are not allowed");
            actualLen = input;
        } else {
            REQUIRE_TRUE(stateIn == nullptr, 0,
                         "gated_delta_rule: multiple recurrent state inputs are not allowed");
            stateIn = input;
        }
    }
    REQUIRE_TRUE(actualLen == nullptr || actualLen->dataType() == DataType::INT64, 0,
                 "gated_delta_rule: actualLen input must be INT64 scalar");
    const auto dataType = Q->dataType();
    REQUIRE_TRUE(K->dataType() == dataType && V->dataType() == dataType &&
                     beta->dataType() == dataType && gate->dataType() == dataType,
                 0, "gated_delta_rule: Q, K, V, beta, and gate must have the same floating dtype");
    REQUIRE_TRUE(stateIn == nullptr || stateIn->dataType() == dataType, 0,
                 "gated_delta_rule: stateIn dtype must match Q dtype");
    REQUIRE_TRUE(stateIn == nullptr ||
                     (stateIn->rankOf() == 4 &&
                      stateIn->sizeAt(0) == Q->sizeAt(0) &&
                      stateIn->sizeAt(1) == Q->sizeAt(2) &&
                      stateIn->sizeAt(2) == Q->sizeAt(3) &&
                      stateIn->sizeAt(3) == V->sizeAt(3)), 0,
                 "gated_delta_rule: stateIn must have shape [B,H,D_k,D_v]");

    helpers::gatedDeltaRule(block.launchContext(), Q, K, V, beta, gate, stateIn, actualLen,
                            output, stateOut);

    return sd::Status::OK;
}

DECLARE_TYPES(gated_delta_rule) {
    // CUDA execution allocates per-invocation recurrent/chunk scratch buffers.
    // Keep this op live between captured Triton islands so graph replay never
    // retains pointers to scratch storage returned to the memory pool.
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING | OP_TRAIT_EXTERNAL_WORKSPACE);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
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
