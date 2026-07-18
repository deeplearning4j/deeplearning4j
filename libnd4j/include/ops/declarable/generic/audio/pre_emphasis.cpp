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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_pre_emphasis)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(pre_emphasis, 1, 1, false, -2, 0) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    double coefficient = block.numT() > 0 ? T_ARG(0) : 0.97;

    REQUIRE_TRUE(input->rankOf() >= 1, 0,
                 "pre_emphasis: input must be at least rank 1, got %d", input->rankOf());

    helpers::preEmphasis(block.launchContext(), input, coefficient, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(pre_emphasis) {
    auto inputShapeInfo = inputShape->at(0);

    auto inputRank = shape::rank(inputShapeInfo);
    std::vector<sd::LongType> outputShapeVec;
    for (int i = 0; i < inputRank; i++) {
        outputShapeVec.push_back(shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(i)));
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(pre_emphasis) {
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
