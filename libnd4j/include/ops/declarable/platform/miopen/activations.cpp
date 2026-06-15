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
// MIOpen Activation Functions for ZLUDA AMD GPU Support
//

#include <system/common.h>

#if defined(HAVE_MIOPEN)

#include "miopenUtils.h"
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void activationMIOpen(const LaunchContext* context,
                             const NDArray* input, NDArray* output,
                             miopenActivationMode_t mode,
                             double alpha = 0.0, double beta = 0.0, double gamma = 0.0) {

    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceId()));

    // Create tensor descriptor
    MIOpenTensor xDesc;
    auto dtype = miopenDataType(input->dataType());

    // Set up tensor descriptor based on input shape
    auto shape = input->shapeOf();
    auto rank = input->rankOf();

    if (rank == 4) {
        xDesc.set4D(dtype, shape[0], shape[1], shape[2], shape[3]);
    } else if (rank == 2) {
        xDesc.set4D(dtype, shape[0], shape[1], 1, 1);
    } else {
        // For other ranks, flatten to 4D
        LongType n = input->lengthOf();
        xDesc.set4D(dtype, 1, static_cast<int>(n), 1, 1);
    }

    // Create activation descriptor
    MIOpenActivation actDesc;
    actDesc.set(mode, alpha, beta, gamma);

    float fAlpha = 1.0f;
    float fBeta = 0.0f;

    NDArray::prepareSpecialUse({output}, {input});

    CHECK_MIOPEN(miopenActivationForward(
        handle, actDesc,
        &fAlpha, xDesc, input->specialBuffer(),
        &fBeta, xDesc, output->specialBuffer()));

    NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// ReLU Activation
PLATFORM_IMPL(relu, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    activationMIOpen(block.launchContext(), input, output, miopenActivationRELU);

    return Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN RELU OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0),
                     static_cast<LongType>(INT_MAX));
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ReLU6 Activation (clipped ReLU)
PLATFORM_IMPL(relu6, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // MIOpen uses CLIPPED_RELU with ceiling parameter
    activationMIOpen(block.launchContext(), input, output,
                     miopenActivationCLIPPEDRELU, 6.0, 0.0, 0.0);

    return Status::OK;
}

PLATFORM_CHECK(relu6, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN RELU6 OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Sigmoid Activation
PLATFORM_IMPL(sigmoid, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    activationMIOpen(block.launchContext(), input, output, miopenActivationLOGISTIC);

    return Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN SIGMOID OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Tanh Activation
PLATFORM_IMPL(tanh, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    activationMIOpen(block.launchContext(), input, output, miopenActivationTANH);

    return Status::OK;
}

PLATFORM_CHECK(tanh, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN TANH OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ELU Activation
PLATFORM_IMPL(elu, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    double alpha = block.numT() > 0 ? T_ARG(0) : 1.0;

    activationMIOpen(block.launchContext(), input, output, miopenActivationELU, alpha);

    return Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN ELU OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softplus Activation: log(1 + exp(x))
PLATFORM_IMPL(softplus, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    activationMIOpen(block.launchContext(), input, output, miopenActivationSOFTRELU);

    return Status::OK;
}

PLATFORM_CHECK(softplus, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN SOFTPLUS OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN
