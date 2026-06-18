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
// Apple Accelerate framework - Gradient/backward operations
// Optimized backward passes for activation functions
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
// RELU_BP - ReLU backward: grad_input = grad_output * (input > 0)
//==============================================================================

PLATFORM_IMPL(relu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // grad_input = grad_output where input > 0, else 0
        for (vDSP_Length i = 0; i < n; i++) {
            gradInBuf[i] = (inBuf[i] > 0.0f) ? gradOutBuf[i] : 0.0f;
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        for (vDSP_Length i = 0; i < n; i++) {
            gradInBuf[i] = (inBuf[i] > 0.0) ? gradOutBuf[i] : 0.0;
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(relu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("RELU_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SIGMOID_BP - Sigmoid backward: grad_input = grad_output * sigmoid * (1 - sigmoid)
//==============================================================================

PLATFORM_IMPL(sigmoid_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute sigmoid(x)
        std::vector<float> sigmoid(n);
        float negOne = -1.0f;
        vDSP_vsmul(inBuf, 1, &negOne, sigmoid.data(), 1, n);  // -x
        vvexpf(sigmoid.data(), sigmoid.data(), &nInt);          // exp(-x)
        float one = 1.0f;
        vDSP_vsadd(sigmoid.data(), 1, &one, sigmoid.data(), 1, n);  // 1 + exp(-x)
        vvrecf(sigmoid.data(), sigmoid.data(), &nInt);              // 1 / (1 + exp(-x))

        // Compute sigmoid * (1 - sigmoid)
        std::vector<float> oneMinusSigmoid(n);
        vDSP_vsadd(sigmoid.data(), 1, &negOne, oneMinusSigmoid.data(), 1, n);  // sigmoid - 1
        vDSP_vneg(oneMinusSigmoid.data(), 1, oneMinusSigmoid.data(), 1, n);    // 1 - sigmoid

        // grad_input = grad_output * sigmoid * (1 - sigmoid)
        vDSP_vmul(sigmoid.data(), 1, oneMinusSigmoid.data(), 1, gradInBuf, 1, n);
        vDSP_vmul(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        std::vector<double> sigmoid(n);
        double negOne = -1.0;
        vDSP_vsmulD(inBuf, 1, &negOne, sigmoid.data(), 1, n);
        vvexp(sigmoid.data(), sigmoid.data(), &nInt);
        double one = 1.0;
        vDSP_vsaddD(sigmoid.data(), 1, &one, sigmoid.data(), 1, n);
        vvrec(sigmoid.data(), sigmoid.data(), &nInt);

        std::vector<double> oneMinusSigmoid(n);
        vDSP_vsaddD(sigmoid.data(), 1, &negOne, oneMinusSigmoid.data(), 1, n);
        vDSP_vnegD(oneMinusSigmoid.data(), 1, oneMinusSigmoid.data(), 1, n);

        vDSP_vmulD(sigmoid.data(), 1, oneMinusSigmoid.data(), 1, gradInBuf, 1, n);
        vDSP_vmulD(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(sigmoid_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("SIGMOID_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// TANH_BP - Tanh backward: grad_input = grad_output * (1 - tanh^2)
//==============================================================================

PLATFORM_IMPL(tanh_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute tanh(x)
        std::vector<float> tanhVal(n);
        vvtanhf(tanhVal.data(), inBuf, &nInt);

        // Compute tanh^2
        vDSP_vsq(tanhVal.data(), 1, gradInBuf, 1, n);

        // Compute 1 - tanh^2
        float one = 1.0f;
        float negOne = -1.0f;
        vDSP_vsadd(gradInBuf, 1, &negOne, gradInBuf, 1, n);
        vDSP_vneg(gradInBuf, 1, gradInBuf, 1, n);

        // grad_input = grad_output * (1 - tanh^2)
        vDSP_vmul(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        std::vector<double> tanhVal(n);
        vvtanh(tanhVal.data(), inBuf, &nInt);

        vDSP_vsqD(tanhVal.data(), 1, gradInBuf, 1, n);

        double negOne = -1.0;
        vDSP_vsaddD(gradInBuf, 1, &negOne, gradInBuf, 1, n);
        vDSP_vnegD(gradInBuf, 1, gradInBuf, 1, n);

        vDSP_vmulD(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(tanh_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("TANH_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ELU_BP - ELU backward: grad_input = grad_output * (1 if x > 0, else alpha * exp(x))
//==============================================================================

PLATFORM_IMPL(elu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    float alpha = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f;

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute exp(x) for negative values
        std::vector<float> expVal(n);
        vvexpf(expVal.data(), inBuf, &nInt);

        // grad = 1 if x > 0, else alpha * exp(x)
        for (vDSP_Length i = 0; i < n; i++) {
            float grad = (inBuf[i] > 0.0f) ? 1.0f : alpha * expVal[i];
            gradInBuf[i] = gradOutBuf[i] * grad;
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();
        double alphaD = static_cast<double>(alpha);

        std::vector<double> expVal(n);
        vvexp(expVal.data(), inBuf, &nInt);

        for (vDSP_Length i = 0; i < n; i++) {
            double grad = (inBuf[i] > 0.0) ? 1.0 : alphaD * expVal[i];
            gradInBuf[i] = gradOutBuf[i] * grad;
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(elu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("ELU_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SOFTPLUS_BP - Softplus backward: grad_input = grad_output * sigmoid(x)
//==============================================================================

PLATFORM_IMPL(softplus_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        float negOne = -1.0f;
        vDSP_vsmul(inBuf, 1, &negOne, gradInBuf, 1, n);
        vvexpf(gradInBuf, gradInBuf, &nInt);
        float one = 1.0f;
        vDSP_vsadd(gradInBuf, 1, &one, gradInBuf, 1, n);
        vvrecf(gradInBuf, gradInBuf, &nInt);

        // grad_input = grad_output * sigmoid
        vDSP_vmul(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        double negOne = -1.0;
        vDSP_vsmulD(inBuf, 1, &negOne, gradInBuf, 1, n);
        vvexp(gradInBuf, gradInBuf, &nInt);
        double one = 1.0;
        vDSP_vsaddD(gradInBuf, 1, &one, gradInBuf, 1, n);
        vvrec(gradInBuf, gradInBuf, &nInt);

        vDSP_vmulD(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(softplus_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("SOFTPLUS_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// LRELU_BP - Leaky ReLU backward
//==============================================================================

PLATFORM_IMPL(lrelu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    float alpha = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 0.01f;

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        for (vDSP_Length i = 0; i < n; i++) {
            gradInBuf[i] = gradOutBuf[i] * (inBuf[i] > 0.0f ? 1.0f : alpha);
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();
        double alphaD = static_cast<double>(alpha);

        for (vDSP_Length i = 0; i < n; i++) {
            gradInBuf[i] = gradOutBuf[i] * (inBuf[i] > 0.0 ? 1.0 : alphaD);
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(lrelu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("LRELU_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SELU_BP - SELU backward
//==============================================================================

PLATFORM_IMPL(selu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const float lambda = 1.0507009873554804934193349852946f;
    const float alpha = 1.6732632423543772848170429916717f;

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        std::vector<float> expVal(n);
        vvexpf(expVal.data(), inBuf, &nInt);

        for (vDSP_Length i = 0; i < n; i++) {
            float grad = (inBuf[i] > 0.0f) ? lambda : lambda * alpha * expVal[i];
            gradInBuf[i] = gradOutBuf[i] * grad;
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();
        double lambdaD = static_cast<double>(lambda);
        double alphaD = static_cast<double>(alpha);

        std::vector<double> expVal(n);
        vvexp(expVal.data(), inBuf, &nInt);

        for (vDSP_Length i = 0; i < n; i++) {
            double grad = (inBuf[i] > 0.0) ? lambdaD : lambdaD * alphaD * expVal[i];
            gradInBuf[i] = gradOutBuf[i] * grad;
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(selu_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("SELU_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SOFTMAX_BP - Softmax backward
//==============================================================================

PLATFORM_IMPL(softmax_bp, ENGINE_CPU) {
    auto output = INPUT_VARIABLE(0);  // softmax output
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    int axis = block.getIArguments()->size() > 0 ? static_cast<int>(INT_ARG(0)) : -1;
    if (axis < 0) axis += output->rankOf();

    const vDSP_Length n = static_cast<vDSP_Length>(output->lengthOf());

    if (output->dataType() == DataType::FLOAT32) {
        const float* outBuf = output->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute dot product of output and gradOutput along axis (simplified for 1D)
        float dotProd = 0.0f;
        vDSP_dotpr(outBuf, 1, gradOutBuf, 1, &dotProd, n);

        // grad_input = output * (grad_output - dot_product)
        for (vDSP_Length i = 0; i < n; i++) {
            gradInBuf[i] = outBuf[i] * (gradOutBuf[i] - dotProd);
        }
    } else if (output->dataType() == DataType::DOUBLE) {
        const double* outBuf = output->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        double dotProd = 0.0;
        vDSP_dotprD(outBuf, 1, gradOutBuf, 1, &dotProd, n);

        for (vDSP_Length i = 0; i < n; i++) {
            gradInBuf[i] = outBuf[i] * (gradOutBuf[i] - dotProd);
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(softmax_bp, ENGINE_CPU) {
    auto output = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("SOFTMAX_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, output, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(output->dataType()), TYPE_MSG_INPUT0);
    // Only support simple case (full reduction) for now
    reqs.expectTrue(output->rankOf() == 1, RANK_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SWISH_BP - Swish backward: grad_input = grad_output * (swish + sigmoid * (1 - swish))
//==============================================================================

PLATFORM_IMPL(swish_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute sigmoid(x)
        std::vector<float> sigmoid(n);
        float negOne = -1.0f;
        vDSP_vsmul(inBuf, 1, &negOne, sigmoid.data(), 1, n);
        vvexpf(sigmoid.data(), sigmoid.data(), &nInt);
        float one = 1.0f;
        vDSP_vsadd(sigmoid.data(), 1, &one, sigmoid.data(), 1, n);
        vvrecf(sigmoid.data(), sigmoid.data(), &nInt);

        // Compute swish = x * sigmoid
        std::vector<float> swish(n);
        vDSP_vmul(inBuf, 1, sigmoid.data(), 1, swish.data(), 1, n);

        // grad = swish + sigmoid * (1 - swish)
        // = swish + sigmoid - sigmoid * swish
        std::vector<float> sigmoidTimesSwish(n);
        vDSP_vmul(sigmoid.data(), 1, swish.data(), 1, sigmoidTimesSwish.data(), 1, n);

        vDSP_vadd(swish.data(), 1, sigmoid.data(), 1, gradInBuf, 1, n);
        vDSP_vsub(sigmoidTimesSwish.data(), 1, gradInBuf, 1, gradInBuf, 1, n);

        // Multiply by grad_output
        vDSP_vmul(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        std::vector<double> sigmoid(n);
        double negOne = -1.0;
        vDSP_vsmulD(inBuf, 1, &negOne, sigmoid.data(), 1, n);
        vvexp(sigmoid.data(), sigmoid.data(), &nInt);
        double one = 1.0;
        vDSP_vsaddD(sigmoid.data(), 1, &one, sigmoid.data(), 1, n);
        vvrec(sigmoid.data(), sigmoid.data(), &nInt);

        std::vector<double> swish(n);
        vDSP_vmulD(inBuf, 1, sigmoid.data(), 1, swish.data(), 1, n);

        std::vector<double> sigmoidTimesSwish(n);
        vDSP_vmulD(sigmoid.data(), 1, swish.data(), 1, sigmoidTimesSwish.data(), 1, n);

        vDSP_vaddD(swish.data(), 1, sigmoid.data(), 1, gradInBuf, 1, n);
        vDSP_vsubD(sigmoidTimesSwish.data(), 1, gradInBuf, 1, gradInBuf, 1, n);

        vDSP_vmulD(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(swish_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("SWISH_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// HARDSIGMOID_BP - Hard sigmoid backward
// hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
// grad = grad_output * (1/6 if -3 <= x <= 3, else 0)
//==============================================================================

PLATFORM_IMPL(hardsigmoid_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        const float scale = 1.0f / 6.0f;
        for (vDSP_Length i = 0; i < n; i++) {
            float grad = (inBuf[i] >= -3.0f && inBuf[i] <= 3.0f) ? scale : 0.0f;
            gradInBuf[i] = gradOutBuf[i] * grad;
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        const double scale = 1.0 / 6.0;
        for (vDSP_Length i = 0; i < n; i++) {
            double grad = (inBuf[i] >= -3.0 && inBuf[i] <= 3.0) ? scale : 0.0;
            gradInBuf[i] = gradOutBuf[i] * grad;
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(hardsigmoid_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("HARDSIGMOID_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// SOFTSIGN_BP - Softsign backward
// softsign(x) = x / (1 + |x|)
// grad = 1 / (1 + |x|)^2
//==============================================================================

PLATFORM_IMPL(softsign_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();

        // Compute |x|
        std::vector<float> absX(n);
        vvfabsf(absX.data(), inBuf, &nInt);

        // Compute 1 + |x|
        float one = 1.0f;
        vDSP_vsadd(absX.data(), 1, &one, gradInBuf, 1, n);

        // Compute (1 + |x|)^2
        vDSP_vsq(gradInBuf, 1, gradInBuf, 1, n);

        // Compute 1 / (1 + |x|)^2
        vvrecf(gradInBuf, gradInBuf, &nInt);

        // Multiply by grad_output
        vDSP_vmul(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();

        std::vector<double> absX(n);
        vvfabs(absX.data(), inBuf, &nInt);

        double one = 1.0;
        vDSP_vsaddD(absX.data(), 1, &one, gradInBuf, 1, n);

        vDSP_vsqD(gradInBuf, 1, gradInBuf, 1, n);

        vvrec(gradInBuf, gradInBuf, &nInt);

        vDSP_vmulD(gradInBuf, 1, gradOutBuf, 1, gradInBuf, 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(softsign_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gradOutput = INPUT_VARIABLE(1);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("SOFTSIGN_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
