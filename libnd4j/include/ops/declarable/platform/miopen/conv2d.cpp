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
// MIOpen Conv2D Implementation for ZLUDA AMD GPU Support
//

#include <system/common.h>

#if defined(HAVE_MIOPEN)

#include "miopenUtils.h"
#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void conv2dMIOpen(const LaunchContext* context,
                         NDArray* input, NDArray* weights,
                         NDArray* bias, NDArray* output,
                         const int kH, const int kW,
                         const int sH, const int sW,
                         const int pH, const int pW,
                         const int dH, const int dW,
                         const int paddingMode, const int isNCHW) {
    if (!isNCHW) {
        THROW_EXCEPTION("MIOpen conv2d requires NCHW input");
    }
    (void)paddingMode;

    LongType bS, iC, iH, iW, oC, oH, oW;
    LongType indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH;
    ConvolutionUtils::getSizesAndIndexesConv2d(
        isNCHW, 0, *input, *output,
        bS, iC, iH, iW, oC, oH, oW,
        indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    const auto inputTensor = miopenTensor4D(
        input->dataType(), static_cast<int>(bS), static_cast<int>(iC),
        static_cast<int>(iH), static_cast<int>(iW));
    const auto weightsTensor = miopenTensor4D(
        weights->dataType(), static_cast<int>(oC), static_cast<int>(iC),
        kH, kW);
    const auto outputTensor = miopenTensor4D(
        output->dataType(), static_cast<int>(bS), static_cast<int>(oC),
        static_cast<int>(oH), static_cast<int>(oW));
    const miopen_bridge::Convolution2D convolution{
        pH, pW, sH, sW, dH, dW};

    miopen_bridge::Tensor4D biasTensor{};
    const miopen_bridge::Tensor4D* biasTensorPointer = nullptr;
    const void* biasBuffer = nullptr;
    if (bias != nullptr && bias->lengthOf() > 0) {
        biasTensor = miopenTensor4D(bias->dataType(), 1, oC, 1, 1);
        biasTensorPointer = &biasTensor;
        biasBuffer = bias->specialBuffer();
        NDArray::prepareSpecialUse({output}, {input, weights, bias});
    } else {
        NDArray::prepareSpecialUse({output}, {input, weights});
    }

    synchronizeZludaForMIOpen(context);
    checkMIOpenBridge(
        miopen_bridge::convolutionForward(
            context->getDeviceID(), inputTensor, weightsTensor, outputTensor,
            convolution, input->specialBuffer(), weights->specialBuffer(),
            output->specialBuffer(), biasTensorPointer, biasBuffer),
        "convolutionForward");

    if (biasTensorPointer != nullptr) {
        NDArray::registerSpecialUse({output}, {input, weights, bias});
    } else {
        NDArray::registerSpecialUse({output}, {input, weights});
    }
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    // Extract convolution parameters
    int kH = INT_ARG(0);
    int kW = INT_ARG(1);
    int sH = INT_ARG(2);
    int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);
    int paddingMode = INT_ARG(8);
    int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;

    // Handle SAME padding
    if (paddingMode == 1) {
        LongType oH, oW;
        ConvolutionUtils::calcOutSizePool2D(oH, oW,
            kH, kW, sH, sW, pH, pW, dH, dW,
            isNCHW ? input->sizeAt(2) : input->sizeAt(1),
            isNCHW ? input->sizeAt(3) : input->sizeAt(2),
            1);  // SAME padding
    }

    conv2dMIOpen(block.launchContext(), input, weights, bias, output,
                 kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW);

    return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv2d, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    int paddingMode = INT_ARG(8);
    int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;

    Requirements req("MIOPEN CONV2D OP");

    // Check for supported data types
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, HALF, BFLOAT16});

    // MIOpen works best with NCHW format
    req.expectTrue(makeInfoVariable(isNCHW, "isNCHW format"));

    // Check that padding mode is not CAUSAL (not supported)
    req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2);

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Backpropagation implementation
static void conv2dBpMIOpen(const LaunchContext* context,
                           NDArray* input, NDArray* weights,
                           NDArray* gradO, NDArray* gradI,
                           NDArray* gradW, NDArray* gradB,
                           const int kH, const int kW,
                           const int sH, const int sW,
                           const int pH, const int pW,
                           const int dH, const int dW,
                           const int paddingMode, const int isNCHW) {
    if (!isNCHW) {
        THROW_EXCEPTION("MIOpen conv2d_bp requires NCHW input");
    }
    (void)paddingMode;

    LongType bS, iC, iH, iW, oC, oH, oW;
    LongType indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH;
    ConvolutionUtils::getSizesAndIndexesConv2d(
        isNCHW, 0, *input, *gradO,
        bS, iC, iH, iW, oC, oH, oW,
        indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    const auto inputTensor = miopenTensor4D(
        input->dataType(), static_cast<int>(bS), static_cast<int>(iC),
        static_cast<int>(iH), static_cast<int>(iW));
    const auto weightsTensor = miopenTensor4D(
        weights->dataType(), static_cast<int>(oC), static_cast<int>(iC),
        kH, kW);
    const auto gradOutputTensor = miopenTensor4D(
        gradO->dataType(), static_cast<int>(bS), static_cast<int>(oC),
        static_cast<int>(oH), static_cast<int>(oW));
    const miopen_bridge::Convolution2D convolution{
        pH, pW, sH, sW, dH, dW};

    miopen_bridge::Tensor4D gradBiasTensor{};
    const miopen_bridge::Tensor4D* gradBiasTensorPointer = nullptr;
    if (gradB != nullptr) {
        gradBiasTensor =
            miopenTensor4D(gradB->dataType(), 1, oC, 1, 1);
        gradBiasTensorPointer = &gradBiasTensor;
    }

    NDArray::prepareSpecialUse({gradI, gradW, gradB},
                               {input, weights, gradO});
    synchronizeZludaForMIOpen(context);
    checkMIOpenBridge(
        miopen_bridge::convolutionBackward(
            context->getDeviceID(), inputTensor, weightsTensor,
            gradOutputTensor, convolution,
            input->specialBuffer(), weights->specialBuffer(),
            gradO->specialBuffer(),
            gradI != nullptr ? gradI->specialBuffer() : nullptr,
            gradW != nullptr ? gradW->specialBuffer() : nullptr,
            gradBiasTensorPointer,
            gradB != nullptr ? gradB->specialBuffer() : nullptr),
        "convolutionBackward");
    NDArray::registerSpecialUse({gradI, gradW, gradB},
                                {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
    auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

    auto gradI = OUTPUT_NULLIFIED(0);
    auto gradW = OUTPUT_NULLIFIED(1);
    auto gradB = block.width() > 3 ? OUTPUT_NULLIFIED(2) : nullptr;

    int kH = INT_ARG(0);
    int kW = INT_ARG(1);
    int sH = INT_ARG(2);
    int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);
    int paddingMode = INT_ARG(8);
    int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;

    conv2dBpMIOpen(block.launchContext(), input, weights, gradO,
                   gradI, gradW, gradB,
                   kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW);

    return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv2d_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

    int paddingMode = INT_ARG(8);
    int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;

    Requirements req("MIOPEN CONV2D_BP OP");

    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(gradO->dataType(), "gradO type"),
                 {FLOAT32, HALF, BFLOAT16});

    req.expectTrue(makeInfoVariable(isNCHW, "isNCHW format"));
    req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN
