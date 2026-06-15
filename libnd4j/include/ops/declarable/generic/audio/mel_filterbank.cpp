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
#if NOT_EXCLUDED(OP_mel_filterbank)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(mel_filterbank, 0, 1, false, -2, -2) {
    auto output = OUTPUT_VARIABLE(0);

    int numMelBins = INT_ARG(0);
    int fftSize = INT_ARG(1);
    int sampleRate = INT_ARG(2);
    double lowerEdgeHz = block.numT() > 0 ? T_ARG(0) : 0.0;
    double upperEdgeHz = block.numT() > 1 ? T_ARG(1) : 8000.0;

    REQUIRE_TRUE(numMelBins > 0, 0, "mel_filterbank: numMelBins must be positive, got %d", numMelBins);
    REQUIRE_TRUE(fftSize > 0, 0, "mel_filterbank: fftSize must be positive, got %d", fftSize);
    REQUIRE_TRUE(sampleRate > 0, 0, "mel_filterbank: sampleRate must be positive, got %d", sampleRate);
    REQUIRE_TRUE(upperEdgeHz > lowerEdgeHz, 0, "mel_filterbank: upperEdgeHz must be > lowerEdgeHz");

    helpers::melFilterbank(block.launchContext(), numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(mel_filterbank) {
    int numMelBins = INT_ARG(0);
    int fftSize = INT_ARG(1);

    std::vector<sd::LongType> outputShapeVec = {numMelBins, fftSize / 2 + 1};
    auto desc = new ShapeDescriptor(sd::DataType::FLOAT32, 'c', outputShapeVec);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
}

DECLARE_TYPES(mel_filterbank) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
