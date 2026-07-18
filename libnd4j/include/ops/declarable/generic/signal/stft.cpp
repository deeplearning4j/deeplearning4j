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
#if NOT_EXCLUDED(OP_stft)

#include <ops/declarable/headers/signal.h>
#include <ops/declarable/helpers/signal.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(stft, 2, 1, false, 0, -2) {
    auto signal = INPUT_VARIABLE(0);     // [batch, signal_length] or [batch, signal_length, 1]
    auto frameStep = INPUT_VARIABLE(1);  // scalar
    auto output = OUTPUT_VARIABLE(0);

    NDArray* window = nullptr;
    int frameLength = 256;  // Default frame length

    if (block.width() > 2) {
        window = INPUT_VARIABLE(2);
    }
    if (block.width() > 3) {
        auto frameLengthTensor = INPUT_VARIABLE(3);
        frameLength = frameLengthTensor->e<int>(0);
    } else if (window != nullptr) {
        frameLength = static_cast<int>(window->lengthOf());
    }

    int frameStepVal = frameStep->e<int>(0);
    bool onesided = block.numI() > 0 ? INT_ARG(0) != 0 : true;

    REQUIRE_TRUE(signal->rankOf() >= 1 && signal->rankOf() <= 3, 0,
                 "stft: signal must be rank 1-3, got rank %i", signal->rankOf());

    helpers::stft(block.launchContext(), signal, frameStepVal, window, frameLength, onesided, output);

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(stft) {
    auto signalShape = inputShape->at(0);
    auto frameStep = INPUT_VARIABLE(1);

    NDArray* window = nullptr;
    int frameLength = 256;

    if (block.width() > 2) {
        window = INPUT_VARIABLE(2);
    }
    if (block.width() > 3) {
        auto frameLengthTensor = INPUT_VARIABLE(3);
        frameLength = frameLengthTensor->e<int>(0);
    } else if (window != nullptr) {
        frameLength = static_cast<int>(window->lengthOf());
    }

    int frameStepVal = frameStep->e<int>(0);
    bool onesided = block.numI() > 0 ? INT_ARG(0) != 0 : true;

    auto signalRank = shape::rank(signalShape);
    sd::LongType batchSize = signalRank > 1 ? shape::sizeAt(signalShape, static_cast<sd::LongType>(0)) : 1;
    sd::LongType signalLength = signalRank > 1 ?
        shape::sizeAt(signalShape, static_cast<sd::LongType>(1)) :
        shape::sizeAt(signalShape, static_cast<sd::LongType>(0));

    sd::LongType numFrames = (signalLength - frameLength) / frameStepVal + 1;
    sd::LongType numBins = onesided ? frameLength / 2 + 1 : frameLength;

    // Output: [batch, num_frames, num_bins, 2]
    std::vector<sd::LongType> outputShapeVec = {batchSize, numFrames, numBins, 2};

    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(sd::DataType::FLOAT32, 'c', outputShapeVec));
}

DECLARE_TYPES(stft) {
    getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})  // window (optional)
        ->setAllowedInputTypes(3, {ALL_INTS})    // frame_length (optional)
        ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
