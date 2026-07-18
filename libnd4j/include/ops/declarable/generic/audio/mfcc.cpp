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
#if NOT_EXCLUDED(OP_mfcc)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(mfcc, 1, 1, false, -2, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int sampleRate = block.numI() > 0 ? INT_ARG(0) : 22050;
    int fftSize = block.numI() > 1 ? INT_ARG(1) : 2048;
    int hopLength = block.numI() > 2 ? INT_ARG(2) : 512;
    int numMelBins = block.numI() > 3 ? INT_ARG(3) : 128;
    int numMfcc = block.numI() > 4 ? INT_ARG(4) : 13;
    double lowerEdgeHz = block.numT() > 0 ? T_ARG(0) : 0.0;
    double upperEdgeHz = block.numT() > 1 ? T_ARG(1) : 8000.0;

    REQUIRE_TRUE(input->rankOf() == 1 || input->rankOf() == 2, 0,
                 "mfcc: input must be rank 1 or 2, got %d", input->rankOf());
    REQUIRE_TRUE(fftSize > 0, 0, "mfcc: fftSize must be positive, got %d", fftSize);
    REQUIRE_TRUE(hopLength > 0, 0, "mfcc: hopLength must be positive, got %d", hopLength);
    REQUIRE_TRUE(numMelBins > 0, 0, "mfcc: numMelBins must be positive, got %d", numMelBins);
    REQUIRE_TRUE(numMfcc > 0 && numMfcc <= numMelBins, 0,
                 "mfcc: numMfcc must be in [1, numMelBins], got %d", numMfcc);

    helpers::mfcc(block.launchContext(), input, sampleRate, fftSize, hopLength, numMelBins, numMfcc,
                  lowerEdgeHz, upperEdgeHz, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(mfcc) {
    auto inputShapeInfo = inputShape->at(0);
    auto inputRank = shape::rank(inputShapeInfo);

    int fftSize = block.numI() > 1 ? INT_ARG(1) : 2048;
    int hopLength = block.numI() > 2 ? INT_ARG(2) : 512;
    int numMfcc = block.numI() > 4 ? INT_ARG(4) : 13;

    sd::LongType batch = 1;
    sd::LongType numSamples;

    if (inputRank == 1) {
        numSamples = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    } else {
        batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
        numSamples = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(1));
    }

    sd::LongType numFrames = (numSamples - fftSize) / hopLength + 1;

    std::vector<sd::LongType> outputShapeVec;
    if (inputRank == 2) {
        outputShapeVec = {batch, numMfcc, numFrames};
    } else {
        outputShapeVec = {numMfcc, numFrames};
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(mfcc) {
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
