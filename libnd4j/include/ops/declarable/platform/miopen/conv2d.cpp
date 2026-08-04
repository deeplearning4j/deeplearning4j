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
                         const NDArray* input, const NDArray* weights, const NDArray* bias,
                         NDArray* output,
                         const int kH, const int kW,
                         const int sH, const int sW,
                         const int pH, const int pW,
                         const int dH, const int dW,
                         const int paddingMode, const int isNCHW) {

    // Get dimensions
    int bS, iC, iH, iW, oC, oH, oW;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;

    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output,
        bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    // Get MIOpen handle
    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceID()));

    // Create tensor descriptors
    MIOpenTensor xDesc, yDesc, wDesc;

    auto xType = miopenDataType(input->dataType());
    auto yType = miopenDataType(output->dataType());
    auto wType = miopenDataType(weights->dataType());

    if (isNCHW) {
        xDesc.set4D(xType, bS, iC, iH, iW);
        yDesc.set4D(yType, bS, oC, oH, oW);
    } else {
        // NHWC format - MIOpen requires NCHW, so we need to handle this
        // For now, we'll use NCHW and let the caller handle any needed transposition
        xDesc.set4D(xType, bS, iC, iH, iW);
        yDesc.set4D(yType, bS, oC, oH, oW);
    }

    // Filter descriptor: MIOpen uses NCHW for filters (oC, iC, kH, kW)
    wDesc.set4D(wType, oC, iC, kH, kW);

    // Convolution descriptor
    MIOpenConvolution convDesc;
    convDesc.set2D(pH, pW, sH, sW, dH, dW, miopenConvolution);

    // Find best algorithm
    int returnedAlgoCount;
    miopenConvAlgoPerf_t perfResults;
    size_t wsSize = 0;

    CHECK_MIOPEN(miopenConvolutionForwardGetWorkSpaceSize(
        handle, wDesc, xDesc, convDesc, yDesc, &wsSize));

    HipWorkspace workspace(wsSize);

    CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(
        handle, xDesc, input->specialBuffer(),
        wDesc, weights->specialBuffer(),
        convDesc, yDesc, output->specialBuffer(),
        1, &returnedAlgoCount, &perfResults,
        workspace, wsSize, false));

    miopenConvFwdAlgorithm_t algo = perfResults.fwd_algo;

    // Execute convolution
    float alpha = 1.0f;
    float beta = 0.0f;

    NDArray::prepareSpecialUse({output}, {input, weights, bias});

    CHECK_MIOPEN(miopenConvolutionForward(
        handle, &alpha,
        xDesc, input->specialBuffer(),
        wDesc, weights->specialBuffer(),
        convDesc, algo,
        &beta,
        yDesc, output->specialBuffer(),
        workspace, wsSize));

    // Add bias if present
    if (bias != nullptr && bias->lengthOf() > 0) {
        MIOpenTensor biasDesc;
        biasDesc.set4D(miopenDataType(bias->dataType()), 1, oC, 1, 1);

        float biasAlpha = 1.0f;
        float biasBeta = 1.0f;

        CHECK_MIOPEN(miopenOpTensor(
            handle, miopenTensorOpAdd,
            &biasAlpha, yDesc, output->specialBuffer(),
            &biasAlpha, biasDesc, bias->specialBuffer(),
            &biasBeta, yDesc, output->specialBuffer()));
    }

    NDArray::registerSpecialUse({output}, {input, weights, bias});
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
        int oH, oW;
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
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});

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
                           const NDArray* input, const NDArray* weights,
                           const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB,
                           const int kH, const int kW,
                           const int sH, const int sW,
                           const int pH, const int pW,
                           const int dH, const int dW,
                           const int paddingMode, const int isNCHW) {

    int bS, iC, iH, iW, oC, oH, oW;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;

    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO,
        bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceID()));

    // Create descriptors
    MIOpenTensor xDesc, dyDesc, dxDesc, dwDesc;
    auto dtype = miopenDataType(input->dataType());

    xDesc.set4D(dtype, bS, iC, iH, iW);
    dyDesc.set4D(dtype, bS, oC, oH, oW);

    if (gradI != nullptr) {
        dxDesc.set4D(dtype, bS, iC, iH, iW);
    }
    dwDesc.set4D(dtype, oC, iC, kH, kW);

    MIOpenConvolution convDesc;
    convDesc.set2D(pH, pW, sH, sW, dH, dW, miopenConvolution);

    NDArray::prepareSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});

    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute gradient with respect to input
    if (gradI != nullptr) {
        size_t wsSize = 0;
        CHECK_MIOPEN(miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, dyDesc, dwDesc, convDesc, dxDesc, &wsSize));

        HipWorkspace workspace(wsSize);

        int returnedAlgoCount;
        miopenConvAlgoPerf_t perfResults;
        CHECK_MIOPEN(miopenFindConvolutionBackwardDataAlgorithm(
            handle, dyDesc, gradO->specialBuffer(),
            dwDesc, weights->specialBuffer(),
            convDesc, dxDesc, gradI->specialBuffer(),
            1, &returnedAlgoCount, &perfResults,
            workspace, wsSize, false));

        CHECK_MIOPEN(miopenConvolutionBackwardData(
            handle, &alpha,
            dyDesc, gradO->specialBuffer(),
            dwDesc, weights->specialBuffer(),
            convDesc, perfResults.bwd_data_algo,
            &beta,
            dxDesc, gradI->specialBuffer(),
            workspace, wsSize));
    }

    // Compute gradient with respect to weights
    if (gradW != nullptr) {
        size_t wsSize = 0;
        CHECK_MIOPEN(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle, dyDesc, xDesc, convDesc, dwDesc, &wsSize));

        HipWorkspace workspace(wsSize);

        int returnedAlgoCount;
        miopenConvAlgoPerf_t perfResults;
        CHECK_MIOPEN(miopenFindConvolutionBackwardWeightsAlgorithm(
            handle, dyDesc, gradO->specialBuffer(),
            xDesc, input->specialBuffer(),
            convDesc, dwDesc, gradW->specialBuffer(),
            1, &returnedAlgoCount, &perfResults,
            workspace, wsSize, false));

        CHECK_MIOPEN(miopenConvolutionBackwardWeights(
            handle, &alpha,
            dyDesc, gradO->specialBuffer(),
            xDesc, input->specialBuffer(),
            convDesc, perfResults.bwd_weights_algo,
            &beta,
            dwDesc, gradW->specialBuffer(),
            workspace, wsSize));
    }

    // Compute gradient with respect to bias
    if (gradB != nullptr) {
        MIOpenTensor dbDesc;
        dbDesc.set4D(dtype, 1, oC, 1, 1);

        CHECK_MIOPEN(miopenConvolutionBackwardBias(
            handle, &alpha,
            dyDesc, gradO->specialBuffer(),
            &beta,
            dbDesc, gradB->specialBuffer()));
    }

    NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
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
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(gradO->dataType(), "gradO type"),
                 {FLOAT32, FLOAT16, BFLOAT16});

    req.expectTrue(makeInfoVariable(isNCHW, "isNCHW format"));
    req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2);

    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN
