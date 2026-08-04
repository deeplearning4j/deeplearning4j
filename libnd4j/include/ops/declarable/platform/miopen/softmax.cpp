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
static miopen_bridge::Tensor4D softmaxTensor(const NDArray* input,
                                             int dimension) {
    const auto shape = input->shapeOf();
    const int rank = input->rankOf();
    if (dimension < 0) dimension += rank;
    if (dimension < 0 || dimension >= rank) {
        THROW_EXCEPTION("MIOpen softmax dimension is outside the input rank");
    }

    LongType outerSize = 1;
    LongType innerSize = 1;
    for (int index = 0; index < dimension; ++index) {
        outerSize *= shape[index];
    }
    for (int index = dimension + 1; index < rank; ++index) {
        innerSize *= shape[index];
    }

    return miopenTensor4D(input->dataType(),
                          static_cast<int>(outerSize),
                          static_cast<int>(shape[dimension]), 1,
                          static_cast<int>(innerSize));
}

static void softmaxMIOpen(const LaunchContext* context,
                          const NDArray* input, NDArray* output,
                          int dimension, bool isLog = false) {
    const auto tensor = softmaxTensor(input, dimension);

    NDArray::prepareSpecialUse({output}, {input});
    synchronizeZludaForMIOpen(context);
    checkMIOpenBridge(
        miopen_bridge::softmaxForward(
            context->getDeviceID(), tensor,
            input->specialBuffer(), output->specialBuffer(), isLog),
        "softmaxForward");
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
                            const NDArray* input, const NDArray* gradO,
                            NDArray* gradI, int dimension,
                            bool isLog = false) {
    const auto tensor = softmaxTensor(input, dimension);

    NDArray::prepareSpecialUse({gradI}, {input, gradO});
    synchronizeZludaForMIOpen(context);
    checkMIOpenBridge(
        miopen_bridge::softmaxBackward(
            context->getDeviceID(), tensor,
            input->specialBuffer(), gradO->specialBuffer(),
            gradI->specialBuffer(), isLog),
        "softmaxBackward");
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
