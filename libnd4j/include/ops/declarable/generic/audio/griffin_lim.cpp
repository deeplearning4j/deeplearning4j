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
#if NOT_EXCLUDED(OP_griffin_lim)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(griffin_lim, 1, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int fftSize = block.numI() > 0 ? INT_ARG(0) : 2048;
    int hopLength = block.numI() > 1 ? INT_ARG(1) : 512;
    int numIterations = block.numI() > 2 ? INT_ARG(2) : 32;

    REQUIRE_TRUE(input->rankOf() == 3, 0,
                 "griffin_lim: input must be rank 3 [batch, freqBins, numFrames], got %d", input->rankOf());
    REQUIRE_TRUE(fftSize > 0, 0, "griffin_lim: fftSize must be positive, got %d", fftSize);
    REQUIRE_TRUE(hopLength > 0, 0, "griffin_lim: hopLength must be positive, got %d", hopLength);
    REQUIRE_TRUE(numIterations > 0, 0, "griffin_lim: numIterations must be positive, got %d", numIterations);

    helpers::griffinLim(block.launchContext(), input, fftSize, hopLength, numIterations, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(griffin_lim) {
    auto inputShapeInfo = inputShape->at(0);

    int fftSize = block.numI() > 0 ? INT_ARG(0) : 2048;
    int hopLength = block.numI() > 1 ? INT_ARG(1) : 512;

    sd::LongType batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    sd::LongType numFrames = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(2));
    sd::LongType numSamples = (numFrames - 1) * hopLength + fftSize;

    std::vector<sd::LongType> outputShapeVec = {batch, numSamples};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(griffin_lim) {
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
