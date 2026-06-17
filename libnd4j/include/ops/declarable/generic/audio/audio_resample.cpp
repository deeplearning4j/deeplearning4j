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
#if NOT_EXCLUDED(OP_audio_resample)

#include <ops/declarable/headers/audio.h>
#include <ops/declarable/helpers/audio.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(audio_resample, 1, 1, false, 0, 2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int origSampleRate = INT_ARG(0);
    int targetSampleRate = INT_ARG(1);

    REQUIRE_TRUE(input->rankOf() == 1 || input->rankOf() == 2, 0,
                 "audio_resample: input must be rank 1 or 2, got %d", input->rankOf());
    REQUIRE_TRUE(origSampleRate > 0, 0, "audio_resample: origSampleRate must be positive, got %d", origSampleRate);
    REQUIRE_TRUE(targetSampleRate > 0, 0, "audio_resample: targetSampleRate must be positive, got %d", targetSampleRate);

    helpers::audioResample(block.launchContext(), input, origSampleRate, targetSampleRate, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(audio_resample) {
    auto inputShapeInfo = inputShape->at(0);
    auto inputRank = shape::rank(inputShapeInfo);

    int origSampleRate = INT_ARG(0);
    int targetSampleRate = INT_ARG(1);

    sd::LongType batch = 1;
    sd::LongType origSamples;

    if (inputRank == 1) {
        origSamples = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
    } else {
        batch = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(0));
        origSamples = shape::sizeAt(inputShapeInfo, static_cast<sd::LongType>(1));
    }

    sd::LongType targetSamples = origSamples * targetSampleRate / origSampleRate;

    std::vector<sd::LongType> outputShapeVec;
    if (inputRank == 2) {
        outputShapeVec = {batch, targetSamples};
    } else {
        outputShapeVec = {targetSamples};
    }

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outputShapeVec));
}

DECLARE_TYPES(audio_resample) {
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
