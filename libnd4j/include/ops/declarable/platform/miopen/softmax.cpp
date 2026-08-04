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
// MIOpen Softmax Implementation for ZLUDA AMD GPU Support
//

#include <system/common.h>

#if defined(HAVE_MIOPEN)

#include "miopenUtils.h"
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void softmaxMIOpen(const LaunchContext* context,
                          const NDArray* input, NDArray* output,
                          int dimension, bool isLog = false) {

    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceID()));

    // Get shape info
    auto shape = input->shapeOf();
    auto rank = input->rankOf();

    // Create tensor descriptor
    MIOpenTensor xDesc;
    auto dtype = miopenDataType(input->dataType());

    // MIOpen softmax operates on 4D tensors in NCHW format
    // We need to reshape our input appropriately
    int n = 1, c = 1, h = 1, w = 1;

    if (rank == 2) {
        // [batch, features] -> softmax along features
        n = static_cast<int>(shape[0]);
        c = static_cast<int>(shape[1]);
    } else if (rank == 4) {
        n = static_cast<int>(shape[0]);
        c = static_cast<int>(shape[1]);
        h = static_cast<int>(shape[2]);
        w = static_cast<int>(shape[3]);
    } else {
        // Generic case: flatten appropriately based on dimension
        LongType outerSize = 1;
        LongType innerSize = 1;
        LongType dimSize = shape[dimension];

        for (int i = 0; i < dimension; i++) {
            outerSize *= shape[i];
        }
        for (int i = dimension + 1; i < rank; i++) {
            innerSize *= shape[i];
        }

        n = static_cast<int>(outerSize);
        c = static_cast<int>(dimSize);
        h = 1;
        w = static_cast<int>(innerSize);
    }

    xDesc.set4D(dtype, n, c, h, w);

    float alpha = 1.0f;
    float beta = 0.0f;

    NDArray::prepareSpecialUse({output}, {input});

    miopenSoftmaxAlgorithm_t algo = isLog ? miopenSoftmaxLog : miopenSoftmaxAccurate;
    miopenSoftmaxMode_t mode = miopenSoftmaxChannel;

    CHECK_MIOPEN(miopenSoftmaxForward_V2(
        handle, &alpha, xDesc, input->specialBuffer(),
        &beta, xDesc, output->specialBuffer(),
        algo, mode));

    NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(softmax, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : input->rankOf() - 1;

    softmaxMIOpen(block.launchContext(), input, output, dimension, false);

    return Status::OK;
}

PLATFORM_CHECK(softmax, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN SOFTMAX OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(log_softmax, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : input->rankOf() - 1;

    softmaxMIOpen(block.launchContext(), input, output, dimension, true);

    return Status::OK;
}

PLATFORM_CHECK(log_softmax, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("MIOPEN LOG_SOFTMAX OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softmax Backpropagation
static void softmaxBpMIOpen(const LaunchContext* context,
                            const NDArray* input, const NDArray* gradO, NDArray* gradI,
                            int dimension, bool isLog = false) {

    auto handle = getMIOpenHandle(context);
    CHECK_HIP(hipSetDevice(context->getDeviceID()));

    auto shape = input->shapeOf();
    auto rank = input->rankOf();

    MIOpenTensor xDesc;
    auto dtype = miopenDataType(input->dataType());

    int n = 1, c = 1, h = 1, w = 1;

    if (rank == 2) {
        n = static_cast<int>(shape[0]);
        c = static_cast<int>(shape[1]);
    } else if (rank == 4) {
        n = static_cast<int>(shape[0]);
        c = static_cast<int>(shape[1]);
        h = static_cast<int>(shape[2]);
        w = static_cast<int>(shape[3]);
    } else {
        LongType outerSize = 1;
        LongType innerSize = 1;
        LongType dimSize = shape[dimension];

        for (int i = 0; i < dimension; i++) {
            outerSize *= shape[i];
        }
        for (int i = dimension + 1; i < rank; i++) {
            innerSize *= shape[i];
        }

        n = static_cast<int>(outerSize);
        c = static_cast<int>(dimSize);
        h = 1;
        w = static_cast<int>(innerSize);
    }

    xDesc.set4D(dtype, n, c, h, w);

    float alpha = 1.0f;
    float beta = 0.0f;

    NDArray::prepareSpecialUse({gradI}, {input, gradO});

    miopenSoftmaxAlgorithm_t algo = isLog ? miopenSoftmaxLog : miopenSoftmaxAccurate;
    miopenSoftmaxMode_t mode = miopenSoftmaxChannel;

    CHECK_MIOPEN(miopenSoftmaxBackward_V2(
        handle, &alpha,
        xDesc, input->specialBuffer(),  // output of forward pass (softmax result)
        xDesc, gradO->specialBuffer(),  // gradient from next layer
        &beta,
        xDesc, gradI->specialBuffer(),  // gradient to propagate back
        algo, mode));

    NDArray::registerSpecialUse({gradI}, {input, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(softmax_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);
    auto gradI = OUTPUT_VARIABLE(0);

    int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : input->rankOf() - 1;

    softmaxBpMIOpen(block.launchContext(), input, gradO, gradI, dimension, false);

    return Status::OK;
}

PLATFORM_CHECK(softmax_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);

    Requirements req("MIOPEN SOFTMAX_BP OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(gradO->dataType(), "gradO type"),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(log_softmax_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);
    auto gradI = OUTPUT_VARIABLE(0);

    int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : input->rankOf() - 1;

    softmaxBpMIOpen(block.launchContext(), input, gradO, gradI, dimension, true);

    return Status::OK;
}

PLATFORM_CHECK(log_softmax_bp, ENGINE_ZLUDA_AMD) {
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);

    Requirements req("MIOPEN LOG_SOFTMAX_BP OP");
    req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                 {FLOAT32, FLOAT16, BFLOAT16}) &&
    req.expectIn(makeInfoVariable(gradO->dataType(), "gradO type"),
                 {FLOAT32, FLOAT16, BFLOAT16});
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN
