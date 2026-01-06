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
// @author Eclipse Deeplearning4j Development Team
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dft)

#include <ops/declarable/headers/signal.h>
#include <ops/declarable/helpers/signal.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dft, 1, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);  // [..., N, 2] complex input
    auto output = OUTPUT_VARIABLE(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : -2;  // Second to last by default
    bool inverse = block.numI() > 1 ? INT_ARG(1) != 0 : false;
    bool onesided = block.numI() > 2 ? INT_ARG(2) != 0 : false;

    // Validate input has complex format (last dim = 2)
    REQUIRE_TRUE(input->sizeAt(-1) == 2, 0,
                 "dft: input must have last dimension = 2 for [real, imag], got %lld",
                 input->sizeAt(-1));

    helpers::dft(block.launchContext(), input, axis, inverse, onesided, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(dft) {
    auto inputShapeInfo = inputShape->at(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : -2;
    bool onesided = block.numI() > 2 ? INT_ARG(2) != 0 : false;

    auto inputRank = shape::rank(inputShapeInfo);

    // Normalize axis
    if (axis < 0) axis += inputRank;

    std::vector<sd::LongType> outputShapeVec;
    for (int i = 0; i < inputRank; i++) {
        auto dim = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(i));
        if (i == axis && onesided) {
            // For onesided, output is N/2 + 1
            outputShapeVec.push_back(dim / 2 + 1);
        } else {
            outputShapeVec.push_back(dim);
        }
    }

    auto desc = new ShapeDescriptor(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
}

DECLARE_TYPES(dft) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
