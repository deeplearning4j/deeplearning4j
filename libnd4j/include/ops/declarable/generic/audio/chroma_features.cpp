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
#if NOT_EXCLUDED(OP_chroma_features)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(chroma_features, 1, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int sampleRate = block.numI() > 0 ? INT_ARG(0) : 22050;
    int fftSize = block.numI() > 1 ? INT_ARG(1) : 2048;
    int numChroma = block.numI() > 2 ? INT_ARG(2) : 12;

    REQUIRE_TRUE(input->rankOf() == 3, 0,
                 "chroma_features: input must be rank 3 [batch, freqBins, numFrames], got %d", input->rankOf());
    REQUIRE_TRUE(sampleRate > 0, 0, "chroma_features: sampleRate must be positive, got %d", sampleRate);
    REQUIRE_TRUE(fftSize > 0, 0, "chroma_features: fftSize must be positive, got %d", fftSize);
    REQUIRE_TRUE(numChroma > 0, 0, "chroma_features: numChroma must be positive, got %d", numChroma);

    helpers::chromaFeatures(block.launchContext(), input, sampleRate, fftSize, numChroma, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(chroma_features) {
    auto inputShapeInfo = inputShape->at(0);

    int numChroma = block.numI() > 2 ? INT_ARG(2) : 12;

    sd::LongType batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    sd::LongType numFrames = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(2));

    std::vector<sd::LongType> outputShapeVec = {batch, numChroma, numFrames};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(chroma_features) {
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
