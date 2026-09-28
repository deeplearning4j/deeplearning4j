/* ******************************************************************************
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
// Fused element-wise chain op: executes a sequence of element-wise ops
// in a single kernel pass, keeping intermediates in registers.
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_fused_elementwise_chain)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(fused_elementwise_chain, 1, 1, false, 0, 1) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // iArgs contain the FusedElemOp codes for the chain
    const int numOps = static_cast<int>(block.getIArguments()->size());
    REQUIRE_TRUE(numOps >= 1 && numOps <= helpers::FUSED_CHAIN_MAX_OPS, 0,
                 "fused_elementwise_chain: chain length %i is outside 1..%i", numOps,
                 helpers::FUSED_CHAIN_MAX_OPS);

    helpers::FusedElemOp ops[helpers::FUSED_CHAIN_MAX_OPS];
    int numBinary = 0;
    bool hasClip = false;
    for (int i = 0; i < numOps; i++) {
        const LongType code = INT_ARG(i);
        REQUIRE_TRUE(code >= 0 && code <= 255 && helpers::isImplementedFusedOp(static_cast<int>(code)), 0,
                     "fused_elementwise_chain: op code %lld at member %i is not implemented",
                     static_cast<long long>(code), i);
        ops[i] = static_cast<helpers::FusedElemOp>(code);
        if (helpers::isBinaryFusedOp(ops[i])) numBinary++;
        if (ops[i] == helpers::FUSED_CLIP) hasClip = true;
    }

    // Secondary inputs follow input 0, one per binary member in chain order.
    REQUIRE_TRUE(block.width() == static_cast<size_t>(1 + numBinary), 0,
                 "fused_elementwise_chain: %i binary members need %i inputs, got %i", numBinary, 1 + numBinary,
                 static_cast<int>(block.width()));
    NDArray* secondaryInputs[helpers::FUSED_CHAIN_MAX_OPS] = {nullptr};
    int secondaryIdx = 1;
    for (int i = 0; i < numOps; i++) {
        if (helpers::isBinaryFusedOp(ops[i])) secondaryInputs[i] = INPUT_VARIABLE(secondaryIdx++);
    }

    // FUSED_CLIP bounds: tArgs [clipMin, clipMax]
    const double* clipMin = nullptr;
    const double* clipMax = nullptr;
    double clipMinVal = 0, clipMaxVal = 0;
    if (hasClip) {
        REQUIRE_TRUE(block.getTArguments()->size() >= 2, 0,
                     "fused_elementwise_chain: FUSED_CLIP needs tArgs [clipMin, clipMax]");
        clipMinVal = T_ARG(0);
        clipMaxVal = T_ARG(1);
        clipMin = &clipMinVal;
        clipMax = &clipMaxVal;
    }

    helpers::fusedElementwiseChain(input, output, ops, numOps,
                                   secondaryInputs, clipMin, clipMax,
                                   block.launchContext());

    return Status::OK;
}

DECLARE_SHAPE_FN(fused_elementwise_chain) {
    // Output shape = input 0 shape
    auto inShape = inputShape->at(0);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inShape), inShape));
}

DECLARE_TYPES(fused_elementwise_chain) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
    getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    // Variadic fused chain (1..9 inputs). No arity-accurate elementwise trait
    // exists; UNARY_ELEMENTWISE keeps it in the elementwise family for
    // classification while wiring/arity checks gate chaining decisions.
    getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

}  // namespace ops
}  // namespace sd

#endif
