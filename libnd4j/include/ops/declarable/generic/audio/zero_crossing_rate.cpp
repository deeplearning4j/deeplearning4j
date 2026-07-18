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
#if NOT_EXCLUDED(OP_zero_crossing_rate)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(zero_crossing_rate, 1, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int frameLength = block.numI() > 0 ? INT_ARG(0) : 2048;
    int hopLength = block.numI() > 1 ? INT_ARG(1) : 512;

    REQUIRE_TRUE(input->rankOf() == 1 || input->rankOf() == 2, 0,
                 "zero_crossing_rate: input must be rank 1 or 2, got %d", input->rankOf());
    REQUIRE_TRUE(frameLength > 0, 0, "zero_crossing_rate: frameLength must be positive, got %d", frameLength);
    REQUIRE_TRUE(hopLength > 0, 0, "zero_crossing_rate: hopLength must be positive, got %d", hopLength);

    helpers::zeroCrossingRate(block.launchContext(), input, frameLength, hopLength, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(zero_crossing_rate) {
    auto inputShapeInfo = inputShape->at(0);
    auto inputRank = shape::rank(inputShapeInfo);

    int frameLength = block.numI() > 0 ? INT_ARG(0) : 2048;
    int hopLength = block.numI() > 1 ? INT_ARG(1) : 512;

    sd::LongType batch = 1;
    sd::LongType numSamples;

    if (inputRank == 1) {
        numSamples = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    } else {
        batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
        numSamples = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(1));
    }

    sd::LongType numFrames = (numSamples - frameLength) / hopLength + 1;

    std::vector<sd::LongType> outputShapeVec;
    if (inputRank == 2) {
        outputShapeVec = {batch, numFrames};
    } else {
        outputShapeVec = {numFrames};
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(zero_crossing_rate) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
