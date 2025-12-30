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
// Apple Accelerate framework - 2D Pooling via BNNS
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_ACCELERATE

/**
 * Helper function to perform pooling using BNNS
 */
static sd::Status poolingBNNS(const NDArray* input, NDArray* output,
                               LongType kH, LongType kW,
                               LongType sH, LongType sW,
                               LongType pH, LongType pW,
                               bool isNCHW, BNNSPoolingFunction poolingType) {

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

    // BNNS works best with NHWC format and float32
    if (!isNCHW && input->dataType() == DataType::FLOAT32) {
        // Create BNNS descriptors for input
        BNNSNDArrayDescriptor inputDesc;
        memset(&inputDesc, 0, sizeof(inputDesc));
        inputDesc.layout = BNNSDataLayoutImageCHW;
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

        // Create pooling layer parameters
        BNNSLayerParametersPooling poolParams;
        memset(&poolParams, 0, sizeof(poolParams));
        poolParams.i_desc = inputDesc;
        poolParams.o_desc = outputDesc;
        poolParams.pooling_function = poolingType;
        poolParams.k_width = static_cast<size_t>(kW);
        poolParams.k_height = static_cast<size_t>(kH);
        poolParams.x_stride = static_cast<size_t>(sW);
        poolParams.y_stride = static_cast<size_t>(sH);
        poolParams.x_padding = static_cast<size_t>(pW);
        poolParams.y_padding = static_cast<size_t>(pH);

        // Create the filter
        BNNSFilter filter = BNNSFilterCreateLayerPooling(&poolParams, nullptr);

        if (filter != nullptr) {
            // Apply pooling for each sample in batch
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

    return sd::Status::KERNEL_FAILURE;
}

//////////////////////////////////////////////////////////////////////////
// Max Pooling 2D
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(maxpool2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get pooling parameters
    LongType kH = INT_ARG(0);  // kernel height
    LongType kW = INT_ARG(1);  // kernel width
    LongType sH = INT_ARG(2);  // stride height
    LongType sW = INT_ARG(3);  // stride width
    LongType pH = INT_ARG(4);  // padding height
    LongType pW = INT_ARG(5);  // padding width
    // INT_ARG(6) - dilation height (not supported by BNNS)
    // INT_ARG(7) - dilation width (not supported by BNNS)
    int isNCHW = block.getIArguments()->size() > 10 ? INT_ARG(10) : 1;

    return poolingBNNS(input, output, kH, kW, sH, sW, pH, pW, isNCHW != 0, BNNSPoolingFunctionMax);
}

PLATFORM_CHECK(maxpool2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    int isNCHW = block.getIArguments()->size() > 10 ? INT_ARG(10) : 1;
    LongType dH = INT_ARG(6);
    LongType dW = INT_ARG(7);

    Requirements req("ACCELERATE MAXPOOL2D OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported by BNNS pooling");
    req.expectTrue(isNCHW == 0, "Only NHWC format is currently supported by BNNS");
    req.expectTrue(dH == 1 && dW == 1, "Dilation is not supported by BNNS pooling");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Average Pooling 2D
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(avgpool2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // Get pooling parameters
    LongType kH = INT_ARG(0);
    LongType kW = INT_ARG(1);
    LongType sH = INT_ARG(2);
    LongType sW = INT_ARG(3);
    LongType pH = INT_ARG(4);
    LongType pW = INT_ARG(5);
    int isNCHW = block.getIArguments()->size() > 10 ? INT_ARG(10) : 1;

    return poolingBNNS(input, output, kH, kW, sH, sW, pH, pW, isNCHW != 0, BNNSPoolingFunctionAverage);
}

PLATFORM_CHECK(avgpool2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    int isNCHW = block.getIArguments()->size() > 10 ? INT_ARG(10) : 1;
    LongType dH = INT_ARG(6);
    LongType dW = INT_ARG(7);

    Requirements req("ACCELERATE AVGPOOL2D OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported by BNNS pooling");
    req.expectTrue(isNCHW == 0, "Only NHWC format is currently supported by BNNS");
    req.expectTrue(dH == 1 && dW == 1, "Dilation is not supported by BNNS pooling");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
