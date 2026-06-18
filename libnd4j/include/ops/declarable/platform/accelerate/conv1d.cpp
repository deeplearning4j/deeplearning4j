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
// Apple Accelerate framework - 1D Convolution operations
// Uses vDSP for optimized 1D convolution and correlation
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include "accelerateUtils.h"

#ifdef HAVE_ACCELERATE

namespace sd {
namespace ops {
namespace platforms {

//==============================================================================
// CONV1D - 1D Convolution
// Uses vDSP_conv for efficient convolution
//==============================================================================

PLATFORM_IMPL(conv1d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);    // [batch, channels, length] or [batch, length, channels]
    auto weights = INPUT_VARIABLE(1);  // [kernelSize, inChannels, outChannels] or similar
    auto output = OUTPUT_VARIABLE(0);

    // Get convolution parameters
    int kW = INT_ARG(0);  // kernel width
    int sW = INT_ARG(1);  // stride
    int pW = INT_ARG(2);  // padding
    int dW = INT_ARG(3);  // dilation
    int paddingMode = INT_ARG(4);  // 0: valid, 1: same, 2: causal
    int isNCW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 1;  // data format

    // For simple case: 1D signal convolution (batch=1, channels=1)
    if (input->rankOf() == 1 && weights->rankOf() == 1) {
        const vDSP_Length inputLen = static_cast<vDSP_Length>(input->lengthOf());
        const vDSP_Length kernelLen = static_cast<vDSP_Length>(weights->lengthOf());
        const vDSP_Length outputLen = static_cast<vDSP_Length>(output->lengthOf());

        if (input->dataType() == DataType::FLOAT32) {
            // vDSP_conv computes correlation, need to reverse kernel for convolution
            std::vector<float> reversedKernel(kernelLen);
            const float* kernelPtr = weights->bufferAsT<float>();
            for (vDSP_Length i = 0; i < kernelLen; i++) {
                reversedKernel[i] = kernelPtr[kernelLen - 1 - i];
            }

            vDSP_conv(input->bufferAsT<float>(), 1,
                      reversedKernel.data(), 1,
                      output->bufferAsT<float>(), 1,
                      outputLen, kernelLen);
        } else if (input->dataType() == DataType::DOUBLE) {
            std::vector<double> reversedKernel(kernelLen);
            const double* kernelPtr = weights->bufferAsT<double>();
            for (vDSP_Length i = 0; i < kernelLen; i++) {
                reversedKernel[i] = kernelPtr[kernelLen - 1 - i];
            }

            vDSP_convD(input->bufferAsT<double>(), 1,
                       reversedKernel.data(), 1,
                       output->bufferAsT<double>(), 1,
                       outputLen, kernelLen);
        }

        return Status::OK;
    }

    // For more complex cases with batches and channels, fall back
    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(conv1d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("CONV1D ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*weights), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    // Currently only support simple 1D case
    reqs.expectTrue(input->rankOf() == 1 && weights->rankOf() == 1,
                    "Only simple 1D convolution (rank-1 arrays) currently supported");

    return reqs.ok();
}

//==============================================================================
// CORRELATE - 1D Cross-correlation
// Uses vDSP_conv directly (which computes correlation)
//==============================================================================

PLATFORM_IMPL(cross_correlate, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    // Simple 1D cross-correlation
    if (x->rankOf() == 1 && y->rankOf() == 1) {
        const vDSP_Length xLen = static_cast<vDSP_Length>(x->lengthOf());
        const vDSP_Length yLen = static_cast<vDSP_Length>(y->lengthOf());
        const vDSP_Length zLen = static_cast<vDSP_Length>(z->lengthOf());

        if (x->dataType() == DataType::FLOAT32) {
            // vDSP_conv computes correlation directly
            vDSP_conv(x->bufferAsT<float>(), 1,
                      y->bufferAsT<float>(), 1,
                      z->bufferAsT<float>(), 1,
                      zLen, yLen);
        } else if (x->dataType() == DataType::DOUBLE) {
            vDSP_convD(x->bufferAsT<double>(), 1,
                       y->bufferAsT<double>(), 1,
                       z->bufferAsT<double>(), 1,
                       zLen, yLen);
        }

        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(cross_correlate, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("CROSS_CORRELATE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->rankOf() == 1 && y->rankOf() == 1,
                    "Only 1D cross-correlation currently supported");

    return reqs.ok();
}

//==============================================================================
// DEPTHWISE_CONV2D - Depthwise 2D Convolution via BNNS
//==============================================================================

PLATFORM_IMPL(depthwise_conv2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    // Get parameters
    int kH = INT_ARG(0);
    int kW = INT_ARG(1);
    int sH = INT_ARG(2);
    int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);
    int paddingMode = INT_ARG(8);
    int isNCHW = block.getIArguments()->size() > 9 ? INT_ARG(9) : 1;

    // For BNNS depthwise convolution
    // This is a simplified implementation - full implementation would need
    // proper BNNS filter setup similar to conv2d

    // Currently only support basic cases via BNNS
    if (input->dataType() == DataType::FLOAT32 && dH == 1 && dW == 1) {
        const LongType batchSize = isNCHW ? input->sizeAt(0) : input->sizeAt(0);
        const LongType inChannels = isNCHW ? input->sizeAt(1) : input->sizeAt(3);
        const LongType inH = isNCHW ? input->sizeAt(2) : input->sizeAt(1);
        const LongType inW = isNCHW ? input->sizeAt(3) : input->sizeAt(2);

        const LongType outH = isNCHW ? output->sizeAt(2) : output->sizeAt(1);
        const LongType outW = isNCHW ? output->sizeAt(3) : output->sizeAt(2);

        // Setup BNNS descriptors for depthwise convolution
        BNNSLayerParametersConvolution convParams;
        memset(&convParams, 0, sizeof(convParams));

        convParams.i_desc.layout = isNCHW ? BNNSDataLayoutImageCHW : BNNSDataLayoutRowMajorMatrix;
        convParams.i_desc.size[0] = inW;
        convParams.i_desc.size[1] = inH;
        convParams.i_desc.size[2] = inChannels;
        convParams.i_desc.data_type = BNNSDataTypeFloat32;

        convParams.o_desc.layout = isNCHW ? BNNSDataLayoutImageCHW : BNNSDataLayoutRowMajorMatrix;
        convParams.o_desc.size[0] = outW;
        convParams.o_desc.size[1] = outH;
        convParams.o_desc.size[2] = inChannels;  // depthwise: out_channels = in_channels
        convParams.o_desc.data_type = BNNSDataTypeFloat32;

        convParams.w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
        convParams.w_desc.size[0] = kW;
        convParams.w_desc.size[1] = kH;
        convParams.w_desc.size[2] = 1;  // depthwise: 1 input channel per group
        convParams.w_desc.size[3] = inChannels;
        convParams.w_desc.data_type = BNNSDataTypeFloat32;
        convParams.w_desc.data = weights->buffer();

        if (bias != nullptr) {
            convParams.bias.data_type = BNNSDataTypeFloat32;
            convParams.bias.data = bias->buffer();
        }

        convParams.x_stride = sW;
        convParams.y_stride = sH;
        convParams.x_padding = pW;
        convParams.y_padding = pH;
        convParams.groups = inChannels;  // depthwise: groups = in_channels

        BNNSFilter filter = BNNSFilterCreateLayerConvolution(&convParams, nullptr);
        if (filter == nullptr) {
            return Status::KERNEL_FAILURE;
        }

        // Apply filter for each batch
        for (LongType b = 0; b < batchSize; b++) {
            const void* inPtr = input->bufferAsT<float>() + b * inChannels * inH * inW;
            void* outPtr = output->bufferAsT<float>() + b * inChannels * outH * outW;

            int result = BNNSFilterApply(filter, inPtr, outPtr);
            if (result != 0) {
                BNNSFilterDestroy(filter);
                return Status::KERNEL_FAILURE;
            }
        }

        BNNSFilterDestroy(filter);
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(depthwise_conv2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    int dH = INT_ARG(6);
    int dW = INT_ARG(7);

    Requirements reqs("DEPTHWISE_CONV2D ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*weights), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(input->dataType() == DataType::FLOAT32, TYPE_MSG_INPUT0);
    reqs.expectTrue(dH == 1 && dW == 1, "Dilation must be 1 for BNNS");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
