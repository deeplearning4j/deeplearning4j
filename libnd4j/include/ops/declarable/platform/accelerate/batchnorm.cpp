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
// Apple Accelerate framework - Batch Normalization via BNNS
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

PLATFORM_IMPL(batchnorm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);     // [bS, iH, iW, iC] or [bS, iC, iH, iW]
    auto mean = INPUT_VARIABLE(1);      // [iC]
    auto variance = INPUT_VARIABLE(2);  // [iC]
    NDArray* gamma = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;  // [iC]
    NDArray* beta = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;   // [iC]

    auto output = OUTPUT_VARIABLE(0);

    // Get parameters
    double epsilon = T_ARG(0);
    int isNCHW = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;

    // Get dimensions
    LongType bS = input->sizeAt(0);
    LongType iC, iH, iW;

    if (isNCHW) {
        iC = input->sizeAt(1);
        iH = input->sizeAt(2);
        iW = input->sizeAt(3);
    } else {
        iH = input->sizeAt(1);
        iW = input->sizeAt(2);
        iC = input->sizeAt(3);
    }

    // For NHWC format and float32, we can use BNNS normalization
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

        // Create normalization layer parameters
        BNNSLayerParametersNormalization normParams;
        memset(&normParams, 0, sizeof(normParams));
        normParams.i_desc = inputDesc;
        normParams.o_desc = inputDesc;  // Same as input
        normParams.normalization_function = BNNSNormalizationFunctionBatchNorm;
        normParams.epsilon = static_cast<float>(epsilon);

        // Set beta (bias) if provided
        if (beta != nullptr) {
            BNNSNDArrayDescriptor betaDesc;
            memset(&betaDesc, 0, sizeof(betaDesc));
            betaDesc.layout = BNNSDataLayoutVector;
            betaDesc.size[0] = static_cast<size_t>(iC);
            betaDesc.data_type = BNNSDataTypeFloat32;
            betaDesc.data = const_cast<void*>(beta->buffer());
            normParams.beta = betaDesc;
        }

        // Set gamma (scale) if provided
        if (gamma != nullptr) {
            BNNSNDArrayDescriptor gammaDesc;
            memset(&gammaDesc, 0, sizeof(gammaDesc));
            gammaDesc.layout = BNNSDataLayoutVector;
            gammaDesc.size[0] = static_cast<size_t>(iC);
            gammaDesc.data_type = BNNSDataTypeFloat32;
            gammaDesc.data = const_cast<void*>(gamma->buffer());
            normParams.gamma = gammaDesc;
        }

        // Set moving mean
        BNNSNDArrayDescriptor meanDesc;
        memset(&meanDesc, 0, sizeof(meanDesc));
        meanDesc.layout = BNNSDataLayoutVector;
        meanDesc.size[0] = static_cast<size_t>(iC);
        meanDesc.data_type = BNNSDataTypeFloat32;
        meanDesc.data = const_cast<void*>(mean->buffer());
        normParams.moving_mean = meanDesc;

        // Set moving variance
        BNNSNDArrayDescriptor varDesc;
        memset(&varDesc, 0, sizeof(varDesc));
        varDesc.layout = BNNSDataLayoutVector;
        varDesc.size[0] = static_cast<size_t>(iC);
        varDesc.data_type = BNNSDataTypeFloat32;
        varDesc.data = const_cast<void*>(variance->buffer());
        normParams.moving_variance = varDesc;

        // Create the filter
        BNNSFilter filter = BNNSFilterCreateLayerNormalization(&normParams, nullptr);

        if (filter != nullptr) {
            // Apply batch normalization for each sample in batch
            for (LongType b = 0; b < bS; b++) {
                const float* inPtr = input->bufferAsT<float>() + b * iH * iW * iC;
                float* outPtr = output->bufferAsT<float>() + b * iH * iW * iC;

                BNNSFilterApply(filter, inPtr, outPtr);
            }

            // Destroy the filter
            BNNSFilterDestroy(filter);

            return sd::Status::OK;
        }
    }

    // Fall back to generic implementation
    return sd::Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(batchnorm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    int isNCHW = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;

    Requirements req("ACCELERATE BATCHNORM OP");

    req.expectTrue(block.isUseAccelerate(), IS_USE_ACCELERATE_MSG);
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported by BNNS normalization");
    req.expectTrue(isNCHW == 0, "Only NHWC format is currently supported by BNNS");
    req.expectTrue(accelerateUtils::isContiguous(*input), "Input must be contiguous");
    req.expectFalse(input->isEmpty(), "Input must not be empty");
    req.expectTrue(input->rankOf() == 4, "Only 4D tensors are supported");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_ACCELERATE

}  // namespace platforms
}  // namespace ops
}  // namespace sd
