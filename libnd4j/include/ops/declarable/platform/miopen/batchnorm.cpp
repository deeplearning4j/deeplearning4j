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
static miopen_bridge::Tensor4D batchnormTensor(NDArray* input,
                                               bool isNCHW,
                                               int& channels) {
    const auto shape = input->shapeOf();
    const auto rank = input->rankOf();

    if (rank == 4) {
        if (!isNCHW) {
            THROW_EXCEPTION("MIOpen batchnorm requires NCHW input");
        }
        channels = static_cast<int>(shape[1]);
        return miopenTensor4D(input->dataType(),
                              static_cast<int>(shape[0]), channels,
                              static_cast<int>(shape[2]),
                              static_cast<int>(shape[3]));
    }
    if (rank == 2) {
        channels = static_cast<int>(shape[1]);
        return miopenTensor4D(input->dataType(),
                              static_cast<int>(shape[0]), channels, 1, 1);
    }
    THROW_EXCEPTION("MIOpen batchnorm supports only rank-2 and rank-4 input");
}

static void batchnormMIOpen(const LaunchContext* context,
                            NDArray* input,
                            NDArray* mean, NDArray* variance,
                            NDArray* gamma, NDArray* beta,
                            NDArray* output,
                            double epsilon, bool isNCHW) {
    int channels = 0;
    const auto tensor = batchnormTensor(input, isNCHW, channels);
    const auto parameterTensor =
        miopenTensor4D(input->dataType(), 1, channels, 1, 1);

    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});
    synchronizeZludaForMIOpen(context);
    checkMIOpenBridge(
        miopen_bridge::batchNormForwardInference(
            context->getDeviceID(), tensor, parameterTensor,
            input->specialBuffer(), output->specialBuffer(),
            gamma->specialBuffer(), beta->specialBuffer(),
            mean->specialBuffer(), variance->specialBuffer(), epsilon),
        "batchNormForwardInference");
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
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), {2, 4});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Batch Normalization Backpropagation
static void batchnormBpMIOpen(const LaunchContext* context,
                              NDArray* input, NDArray* gradO,
                              NDArray* mean, NDArray* variance,
                              NDArray* gamma,
                              NDArray* gradI, NDArray* gradGamma, NDArray* gradBeta,
                              double epsilon, bool isNCHW) {
    int channels = 0;
    const auto tensor = batchnormTensor(input, isNCHW, channels);
    const auto parameterTensor =
        miopenTensor4D(input->dataType(), 1, channels, 1, 1);

    NDArray::prepareSpecialUse({gradI, gradGamma, gradBeta},
                               {input, gradO, mean, variance, gamma});
    synchronizeZludaForMIOpen(context);
    checkMIOpenBridge(
        miopen_bridge::batchNormBackward(
            context->getDeviceID(), tensor, parameterTensor,
            input->specialBuffer(), gradO->specialBuffer(),
            gamma->specialBuffer(), mean->specialBuffer(),
            variance->specialBuffer(), gradI->specialBuffer(),
            gradGamma->specialBuffer(), gradBeta->specialBuffer(), epsilon),
        "batchNormBackward");
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
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(gradO->dataType(), "gradO type"),
                 {FLOAT32, HALF, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), {2, 4});
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN
