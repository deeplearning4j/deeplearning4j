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
// Apple Accelerate framework - 2D Convolution via BNNS
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>
#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_ACCELERATE

PLATFORM_IMPL(conv2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);    // [bS, iH, iW, iC] or [bS, iC, iH, iW]
    auto weights = INPUT_VARIABLE(1);  // [kH, kW, iC, oC] or [oC, iC, kH, kW]
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

    auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, oC] or [bS, oC, oH, oW]

    // Get convolution parameters
    LongType kH = INT_ARG(0);  // kernel height
    LongType kW = INT_ARG(1);  // kernel width
    LongType sH = INT_ARG(2);  // stride height
    LongType sW = INT_ARG(3);  // stride width
    LongType pH = INT_ARG(4);  // padding height
    LongType pW = INT_ARG(5);  // padding width
    LongType dH = INT_ARG(6);  // dilation height
    LongType dW = INT_ARG(7);  // dilation width
    int isNCHW = block.getIArguments()->size() > 9 ? INT_ARG(9) : 1;  // 0=NHWC, 1=NCHW

    // Get dimensions
    LongType bS = input->sizeAt(0);
    LongType iC, iH, iW, oC, oH, oW;

    if (isNCHW) {
        iC = input->sizeAt(1);
        iH = input->sizeAt(2);
        iW = input->sizeAt(3);
        oC = output->sizeAt(1);
        oH = output->sizeAt(2);
        oW = output->sizeAt(3);
    } else {
        iH = input->sizeAt(1);
        iW = input->sizeAt(2);
        iC = input->sizeAt(3);
        oH = output->sizeAt(1);
        oW = output->sizeAt(2);
        oC = output->sizeAt(3);
    }

    // BNNS requires NHWC format and float32
    // For now, we only support this case directly
    if (!isNCHW && input->dataType() == DataType::FLOAT32 && dH == 1 && dW == 1) {
        // Create BNNS descriptors for input
        BNNSNDArrayDescriptor inputDesc;
        memset(&inputDesc, 0, sizeof(inputDesc));
        inputDesc.layout = BNNSDataLayoutImageCHW;  // Actually using HWC but BNNS calls it this
        inputDesc.size[0] = static_cast<size_t>(iW);
        inputDesc.size[1] = static_cast<size_t>(iH);
        inputDesc.size[2] = static_cast<size_t>(iC);
        inputDesc.stride[0] = 1;
        inputDesc.stride[1] = static_cast<size_t>(iW);
        inputDesc.stride[2] = static_cast<size_t>(iH * iW);
        inputDesc.data_type = BNNSDataTypeFloat32;

        // Create BNNS descriptors for output
        BNNSNDArrayDescriptor outputDesc;
        memset(&outputDesc, 0, sizeof(outputDesc));
        outputDesc.layout = BNNSDataLayoutImageCHW;
        outputDesc.size[0] = static_cast<size_t>(oW);
        outputDesc.size[1] = static_cast<size_t>(oH);
        outputDesc.size[2] = static_cast<size_t>(oC);
        outputDesc.stride[0] = 1;
        outputDesc.stride[1] = static_cast<size_t>(oW);
        outputDesc.stride[2] = static_cast<size_t>(oH * oW);
        outputDesc.data_type = BNNSDataTypeFloat32;

        // Create BNNS descriptors for weights
        BNNSNDArrayDescriptor weightsDesc;
        memset(&weightsDesc, 0, sizeof(weightsDesc));
        weightsDesc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
        weightsDesc.size[0] = static_cast<size_t>(kW);
        weightsDesc.size[1] = static_cast<size_t>(kH);
        weightsDesc.size[2] = static_cast<size_t>(iC);
        weightsDesc.size[3] = static_cast<size_t>(oC);
        weightsDesc.data_type = BNNSDataTypeFloat32;
        weightsDesc.data = const_cast<void*>(weights->buffer());

        // Create convolution layer parameters
        BNNSLayerParametersConvolution convParams;
        memset(&convParams, 0, sizeof(convParams));
        convParams.i_desc = inputDesc;
        convParams.o_desc = outputDesc;
        convParams.w_desc = weightsDesc;
        convParams.x_stride = static_cast<size_t>(sW);
        convParams.y_stride = static_cast<size_t>(sH);
        convParams.x_padding = static_cast<size_t>(pW);
        convParams.y_padding = static_cast<size_t>(pH);
        convParams.groups = 1;

        // Set bias if provided
        if (bias != nullptr) {
            BNNSNDArrayDescriptor biasDesc;
            memset(&biasDesc, 0, sizeof(biasDesc));
            biasDesc.layout = BNNSDataLayoutVector;
            biasDesc.size[0] = static_cast<size_t>(oC);
            biasDesc.data_type = BNNSDataTypeFloat32;
            biasDesc.data = const_cast<void*>(bias->buffer());
            convParams.bias = biasDesc;
        }

        // Create the filter
        BNNSFilter filter = BNNSFilterCreateLayerConvolution(&convParams, nullptr);

        if (filter != nullptr) {
            // Apply convolution for each sample in batch
            for (LongType b = 0; b < bS; b++) {
                const float* inPtr = input->bufferAsT<float>() + b * iH * iW * iC;
                float* outPtr = output->bufferAsT<float>() + b * oH * oW * oC;

                BNNSFilterApply(filter, inPtr, outPtr);
            }

            // Destroy the filter
            BNNSFilterDestroy(filter);

            return sd::Status::OK;
        }
    }

    // Fall back to generic implementation for unsupported cases
    return sd::Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(conv2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);

    int isNCHW = block.getIArguments()->size() > 9 ? INT_ARG(9) : 1;
    LongType dH = INT_ARG(6);
    LongType dW = INT_ARG(7);

    Requirements req("ACCELERATE CONV2D OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);

    // BNNS only supports float32
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported by BNNS convolution");

    // Currently only NHWC format without dilation
    req.expectTrue(isNCHW == 0, "Only NHWC format is currently supported by BNNS");
    req.expectTrue(dH == 1 && dW == 1, "Dilation is not supported by BNNS convolution");

    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectTrue(accelerateUtils::isContiguous(*weights), "Weights must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

// Backward pass placeholder - BNNS supports training but implementation is more complex
PLATFORM_IMPL(conv2d_bp, ENGINE_CPU) {
    // For now, fall back to generic implementation
    return sd::Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(conv2d_bp, ENGINE_CPU) {
    Requirements req("ACCELERATE CONV2D_BP OP");
    // Disable for now - training requires more complex BNNS setup
    req.expectFalse(true, "BNNS backward pass not yet implemented");
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
