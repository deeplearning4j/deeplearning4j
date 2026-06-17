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
#if NOT_EXCLUDED(OP_audio_normalize)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(audio_normalize, 1, 1, false, -2, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    double targetLevel = block.numT() > 0 ? T_ARG(0) : 1.0;
    int useRms = block.numI() > 0 ? INT_ARG(0) : 0;

    REQUIRE_TRUE(input->rankOf() >= 1, 0,
                 "audio_normalize: input must be at least rank 1, got %d", input->rankOf());
    REQUIRE_TRUE(targetLevel > 0.0, 0, "audio_normalize: targetLevel must be positive");

    helpers::audioNormalize(block.launchContext(), input, targetLevel, useRms, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(audio_normalize) {
    auto inputShapeInfo = inputShape->at(0);

    auto inputRank = shape::rank(inputShapeInfo);
    std::vector<sd::LongType> outputShapeVec;
    for (int i = 0; i < inputRank; i++) {
        outputShapeVec.push_back(shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(i)));
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(audio_normalize) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
