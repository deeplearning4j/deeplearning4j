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
// MIOpen Batch Normalization for ZLUDA AMD GPU Support
//

#include <system/common.h>

#if defined(HAVE_MIOPEN)

#include "miopenUtils.h"
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void batchnormMIOpen(const LaunchContext* context,
                            const NDArray* input,
                            const NDArray* mean, const NDArray* variance,
                            const NDArray* gamma, const NDArray* beta,
                            NDArray* output,
                            double epsilon, bool isNCHW) {

    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceId()));

    auto shape = input->shapeOf();
    auto rank = input->rankOf();

    int bS, iC, iH, iW;
    if (rank == 4) {
        if (isNCHW) {
            bS = static_cast<int>(shape[0]);
            iC = static_cast<int>(shape[1]);
            iH = static_cast<int>(shape[2]);
            iW = static_cast<int>(shape[3]);
        } else {
            bS = static_cast<int>(shape[0]);
            iH = static_cast<int>(shape[1]);
            iW = static_cast<int>(shape[2]);
            iC = static_cast<int>(shape[3]);
        }
    } else if (rank == 2) {
        bS = static_cast<int>(shape[0]);
        iC = static_cast<int>(shape[1]);
        iH = 1;
        iW = 1;
    } else {
        THROW_EXCEPTION("MIOpen batchnorm: only 2D and 4D inputs supported");
    }

    MIOpenTensor xDesc, bnScaleBiasMeanVarDesc;
    auto dtype = miopenDataType(input->dataType());

    xDesc.set4D(dtype, bS, iC, iH, iW);
    bnScaleBiasMeanVarDesc.set4D(dtype, 1, iC, 1, 1);

    float alpha = 1.0f;
    float betaVal = 0.0f;

    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});

    CHECK_MIOPEN(miopenBatchNormalizationForwardInference(
        handle,
        miopenBNSpatial,
        &alpha, &betaVal,
        xDesc, input->specialBuffer(),
        xDesc, output->specialBuffer(),
        bnScaleBiasMeanVarDesc,
        const_cast<void*>(gamma->specialBuffer()),
        const_cast<void*>(beta->specialBuffer()),
        const_cast<void*>(mean->specialBuffer()),
        const_cast<void*>(variance->specialBuffer()),
        epsilon));

    NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto mean = INPUT_VARIABLE(1);
    auto variance = INPUT_VARIABLE(2);
    auto gamma = INPUT_VARIABLE(3);
    auto beta = INPUT_VARIABLE(4);
    auto output = OUTPUT_VARIABLE(0);

    double epsilon = block.numT() > 0 ? T_ARG(0) : 1e-5;
    bool isNCHW = block.getIArguments()->size() > 0 ? static_cast<bool>(INT_ARG(0)) : true;

    batchnormMIOpen(block.launchContext(), input, mean, variance, gamma, beta,
                    output, epsilon, isNCHW);

    return Status::OK;
}

PLATFORM_CHECK(batchnorm, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN BATCHNORM OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), {2, 4});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Batch Normalization Backpropagation
static void batchnormBpMIOpen(const LaunchContext* context,
                              const NDArray* input, const NDArray* gradO,
                              const NDArray* mean, const NDArray* variance,
                              const NDArray* gamma,
                              NDArray* gradI, NDArray* gradGamma, NDArray* gradBeta,
                              double epsilon, bool isNCHW) {

    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceId()));

    auto shape = input->shapeOf();
    auto rank = input->rankOf();

    int bS, iC, iH, iW;
    if (rank == 4) {
        if (isNCHW) {
            bS = static_cast<int>(shape[0]);
            iC = static_cast<int>(shape[1]);
            iH = static_cast<int>(shape[2]);
            iW = static_cast<int>(shape[3]);
        } else {
            bS = static_cast<int>(shape[0]);
            iH = static_cast<int>(shape[1]);
            iW = static_cast<int>(shape[2]);
            iC = static_cast<int>(shape[3]);
        }
    } else if (rank == 2) {
        bS = static_cast<int>(shape[0]);
        iC = static_cast<int>(shape[1]);
        iH = 1;
        iW = 1;
    } else {
        THROW_EXCEPTION("MIOpen batchnorm_bp: only 2D and 4D inputs supported");
    }

    MIOpenTensor xDesc, bnScaleBiasDiffDesc;
    auto dtype = miopenDataType(input->dataType());

    xDesc.set4D(dtype, bS, iC, iH, iW);
    bnScaleBiasDiffDesc.set4D(dtype, 1, iC, 1, 1);

    float alpha = 1.0f;
    float betaVal = 0.0f;

    NDArray::prepareSpecialUse({gradI, gradGamma, gradBeta},
                               {input, gradO, mean, variance, gamma});

    CHECK_MIOPEN(miopenBatchNormalizationBackward(
        handle,
        miopenBNSpatial,
        &alpha, &betaVal, &alpha, &betaVal,
        xDesc, input->specialBuffer(),
        xDesc, gradO->specialBuffer(),
        xDesc, gradI->specialBuffer(),
        bnScaleBiasDiffDesc,
        const_cast<void*>(gamma->specialBuffer()),
        gradGamma->specialBuffer(),
        gradBeta->specialBuffer(),
        epsilon,
        const_cast<void*>(mean->specialBuffer()),
        const_cast<void*>(variance->specialBuffer())));

    NDArray::registerSpecialUse({gradI, gradGamma, gradBeta},
                                {input, gradO, mean, variance, gamma});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto mean = INPUT_VARIABLE(1);
    auto variance = INPUT_VARIABLE(2);
    auto gamma = INPUT_VARIABLE(3);
    auto gradO = INPUT_VARIABLE(4);

    auto gradI = OUTPUT_VARIABLE(0);
    auto gradGamma = OUTPUT_VARIABLE(1);
    auto gradBeta = OUTPUT_VARIABLE(2);

    double epsilon = block.numT() > 0 ? T_ARG(0) : 1e-5;
    bool isNCHW = block.getIArguments()->size() > 0 ? static_cast<bool>(INT_ARG(0)) : true;

    batchnormBpMIOpen(block.launchContext(), input, gradO, mean, variance, gamma,
                      gradI, gradGamma, gradBeta, epsilon, isNCHW);

    return Status::OK;
}

PLATFORM_CHECK(batchnorm_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(4);

    Requirements req("MIOPEN BATCHNORM_BP OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(gradO->dataType(), "gradO type"),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), {2, 4});
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN
