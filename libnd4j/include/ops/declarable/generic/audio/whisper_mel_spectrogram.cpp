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
#if NOT_EXCLUDED(OP_whisper_mel_spectrogram)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(whisper_mel_spectrogram, 1, 1, false, -2, -2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int sampleRate = block.numI() > 0 ? INT_ARG(0) : 16000;
    int fftSize = block.numI() > 1 ? INT_ARG(1) : 400;
    int hopLength = block.numI() > 2 ? INT_ARG(2) : 160;
    int numMelBins = block.numI() > 3 ? INT_ARG(3) : 80;
    int targetFrames = block.numI() > 4 ? INT_ARG(4) : 3000;
    double lowerEdgeHz = block.numT() > 0 ? T_ARG(0) : 0.0;
    double upperEdgeHz = block.numT() > 1 ? T_ARG(1) : 8000.0;

    REQUIRE_TRUE(input->rankOf() == 1 || input->rankOf() == 2, 0,
                 "whisper_mel_spectrogram: input must be rank 1 or 2, got %d", input->rankOf());
    REQUIRE_TRUE(fftSize > 0, 0, "whisper_mel_spectrogram: fftSize must be positive, got %d", fftSize);
    REQUIRE_TRUE(hopLength > 0, 0, "whisper_mel_spectrogram: hopLength must be positive, got %d", hopLength);
    REQUIRE_TRUE(numMelBins > 0, 0, "whisper_mel_spectrogram: numMelBins must be positive, got %d", numMelBins);
    REQUIRE_TRUE(targetFrames > 0, 0, "whisper_mel_spectrogram: targetFrames must be positive, got %d", targetFrames);

    helpers::whisperMelSpectrogram(block.launchContext(), input, sampleRate, fftSize, hopLength,
                                    numMelBins, targetFrames, lowerEdgeHz, upperEdgeHz, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(whisper_mel_spectrogram) {
    auto inputShapeInfo = inputShape->at(0);
    auto inputRank = shape::rank(inputShapeInfo);

    int numMelBins = block.numI() > 3 ? INT_ARG(3) : 80;
    int targetFrames = block.numI() > 4 ? INT_ARG(4) : 3000;

    sd::LongType batch = 1;
    if (inputRank == 2) {
        batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    }

    // Output is always [batch, numMelBins, targetFrames] (or [numMelBins, targetFrames] for rank-1 input)
    std::vector<sd::LongType> outputShapeVec;
    if (inputRank == 2) {
        outputShapeVec = {batch, static_cast<sd::LongType>(numMelBins), static_cast<sd::LongType>(targetFrames)};
    } else {
        outputShapeVec = {static_cast<sd::LongType>(numMelBins), static_cast<sd::LongType>(targetFrames)};
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(whisper_mel_spectrogram) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
