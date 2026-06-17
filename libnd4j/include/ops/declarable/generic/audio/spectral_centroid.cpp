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
#if NOT_EXCLUDED(OP_spectral_centroid)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(spectral_centroid, 1, 1, false, 0, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int sampleRate = block.numI() > 0 ? INT_ARG(0) : 22050;
    int fftSize = block.numI() > 1 ? INT_ARG(1) : 2048;

    REQUIRE_TRUE(input->rankOf() == 3, 0,
                 "spectral_centroid: input must be rank 3 [batch, freqBins, numFrames], got %d", input->rankOf());
    REQUIRE_TRUE(sampleRate > 0, 0, "spectral_centroid: sampleRate must be positive, got %d", sampleRate);
    REQUIRE_TRUE(fftSize > 0, 0, "spectral_centroid: fftSize must be positive, got %d", fftSize);

    helpers::spectralCentroid(block.launchContext(), input, sampleRate, fftSize, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(spectral_centroid) {
    auto inputShapeInfo = inputShape->at(0);

    sd::LongType batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    sd::LongType numFrames = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(2));

    std::vector<sd::LongType> outputShapeVec = {batch, numFrames};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(spectral_centroid) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
