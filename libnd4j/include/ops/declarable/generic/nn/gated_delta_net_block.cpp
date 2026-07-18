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
// gated_delta_net_block - Full Gated Delta Network layer
//
// Fuses: linear projection -> causal_conv1d + SiLU -> gated_delta_rule
//        -> RMSNorm + Swish gate -> output projection
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gated_delta_net_block)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/gated_delta_net_block.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(gated_delta_net_block, 7, 3, false, 0, 3) {
    auto x = INPUT_VARIABLE(0);           // [B, L, D]
    auto Wqkv = INPUT_VARIABLE(1);        // [D, qkv_dim]
    auto Wbeta = INPUT_VARIABLE(2);       // [D, H]
    auto Wgate = INPUT_VARIABLE(3);       // [D, H]
    auto Wout = INPUT_VARIABLE(4);        // [H*D_v, D]
    auto convWeight = INPUT_VARIABLE(5);  // [D, K]
    auto convBias = INPUT_VARIABLE(6);    // [D]

    auto output = OUTPUT_VARIABLE(0);            // [B, L, D]
    auto recurrentStateOut = OUTPUT_VARIABLE(1);  // [B, H, D_k, D_v]
    auto convStateOut = OUTPUT_VARIABLE(2);       // [B, D, K-1]

    NDArray* stateIn = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;

    int numHeads = INT_ARG(0);
    int headDimK = INT_ARG(1);
    int headDimV = INT_ARG(2);
    float rmsEps = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5f;

    helpers::gatedDeltaNetBlock(block.launchContext(),
                                x, Wqkv, Wbeta, Wgate, Wout, convWeight, convBias,
                                stateIn, output, recurrentStateOut, convStateOut,
                                numHeads, headDimK, headDimV, rmsEps);

    return sd::Status::OK;
}

DECLARE_TYPES(gated_delta_net_block) {
    getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(gated_delta_net_block) {
    auto xShape = inputShape->at(0);           // [B, L, D]
    auto convWeightShape = inputShape->at(5);  // [D, K]

    auto B = shape::sizeAt(xShape, 0);
    auto L = shape::sizeAt(xShape, 1);
    auto D = shape::sizeAt(xShape, 2);
    auto K = shape::sizeAt(convWeightShape, 1);

    int numHeads = INT_ARG(0);
    int headDimK = INT_ARG(1);
    int headDimV = INT_ARG(2);

    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, L, D});

    auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, (sd::LongType)numHeads, (sd::LongType)headDimK, (sd::LongType)headDimV});

    auto convStateShape = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(xShape), 'c', {B, D, K - 1});

    return new ShapeList(std::vector<LongType*>{outputShape, stateShape, convStateShape});
}

}  // namespace ops
}  // namespace sd

#endif
