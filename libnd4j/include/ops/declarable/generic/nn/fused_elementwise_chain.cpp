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
    int numOps = block.getIArguments()->size();
    if (numOps <= 0) {
        return Status::BAD_ARGUMENTS;
    }
    if (numOps > 8) {
        numOps = 8;  // Match helper limit
    }

    // Convert iArgs to FusedElemOp array
    helpers::FusedElemOp ops[8];
    for (int i = 0; i < numOps; i++) {
        ops[i] = static_cast<helpers::FusedElemOp>(INT_ARG(i));
    }

    // Gather secondary inputs for binary ops.
    // Secondary inputs start at input index 1.
    // They are matched to binary ops in chain order.
    NDArray* secondaryInputs[8] = {nullptr};
    int secondaryIdx = 1;  // First secondary input is at index 1
    for (int i = 0; i < numOps; i++) {
        if (helpers::isBinaryFusedOp(ops[i])) {
            if (secondaryIdx < block.width()) {
                secondaryInputs[i] = INPUT_VARIABLE(secondaryIdx);
                secondaryIdx++;
            }
        }
    }

    // Optional clip parameters from tArgs
    const double* clipMin = nullptr;
    const double* clipMax = nullptr;
    double clipMinVal = 0, clipMaxVal = 0;
    if (block.getTArguments()->size() >= 2) {
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
}

}  // namespace ops
}  // namespace sd

#endif
