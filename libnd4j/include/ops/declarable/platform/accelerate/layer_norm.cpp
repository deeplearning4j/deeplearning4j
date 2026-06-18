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
// Apple Accelerate framework - Layer normalization operations
// Forward and backward passes for layer normalization
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
// LAYER_NORM - Layer normalization forward pass
// output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
//==============================================================================

PLATFORM_IMPL(layer_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gamma = INPUT_VARIABLE(1);  // scale
    auto beta = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // shift (optional)
    auto output = OUTPUT_VARIABLE(0);

    float epsilon = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 1e-5f;

    // For simplified 1D case
    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gammaBuf = gamma->bufferAsT<float>();
        const float* betaBuf = beta ? beta->bufferAsT<float>() : nullptr;
        float* outBuf = output->bufferAsT<float>();

        // Compute mean
        float mean = 0.0f;
        vDSP_meanv(inBuf, 1, &mean, n);

        // Compute variance
        std::vector<float> deviations(n);
        float negMean = -mean;
        vDSP_vsadd(inBuf, 1, &negMean, deviations.data(), 1, n);

        std::vector<float> sqDeviations(n);
        vDSP_vsq(deviations.data(), 1, sqDeviations.data(), 1, n);

        float variance = 0.0f;
        vDSP_meanv(sqDeviations.data(), 1, &variance, n);

        // Compute 1/sqrt(variance + epsilon)
        float invStd = 1.0f / sqrtf(variance + epsilon);

        // Normalize: (x - mean) * invStd
        vDSP_vsmul(deviations.data(), 1, &invStd, outBuf, 1, n);

        // Apply scale (gamma)
        if (gamma->lengthOf() == 1) {
            // Scalar gamma
            vDSP_vsmul(outBuf, 1, gammaBuf, outBuf, 1, n);
        } else {
            // Element-wise gamma
            vDSP_vmul(outBuf, 1, gammaBuf, 1, outBuf, 1, n);
        }

        // Apply shift (beta) if provided
        if (betaBuf) {
            if (beta->lengthOf() == 1) {
                vDSP_vsadd(outBuf, 1, betaBuf, outBuf, 1, n);
            } else {
                vDSP_vadd(outBuf, 1, betaBuf, 1, outBuf, 1, n);
            }
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gammaBuf = gamma->bufferAsT<double>();
        const double* betaBuf = beta ? beta->bufferAsT<double>() : nullptr;
        double* outBuf = output->bufferAsT<double>();

        double mean = 0.0;
        vDSP_meanvD(inBuf, 1, &mean, n);

        std::vector<double> deviations(n);
        double negMean = -mean;
        vDSP_vsaddD(inBuf, 1, &negMean, deviations.data(), 1, n);

        std::vector<double> sqDeviations(n);
        vDSP_vsqD(deviations.data(), 1, sqDeviations.data(), 1, n);

        double variance = 0.0;
        vDSP_meanvD(sqDeviations.data(), 1, &variance, n);

        double invStd = 1.0 / sqrt(variance + static_cast<double>(epsilon));

        vDSP_vsmulD(deviations.data(), 1, &invStd, outBuf, 1, n);

        if (gamma->lengthOf() == 1) {
            vDSP_vsmulD(outBuf, 1, gammaBuf, outBuf, 1, n);
        } else {
            vDSP_vmulD(outBuf, 1, gammaBuf, 1, outBuf, 1, n);
        }

        if (betaBuf) {
            if (beta->lengthOf() == 1) {
                vDSP_vsaddD(outBuf, 1, betaBuf, outBuf, 1, n);
            } else {
                vDSP_vaddD(outBuf, 1, betaBuf, 1, outBuf, 1, n);
            }
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gamma = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("LAYER_NORM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gamma), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    // Only support 1D input for now (simple case)
    reqs.expectTrue(input->rankOf() == 1, RANK_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// LAYER_NORM_BP - Layer normalization backward pass
//==============================================================================

PLATFORM_IMPL(layer_norm_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gamma = INPUT_VARIABLE(1);
    auto gradOutput = INPUT_VARIABLE(2);  // gradient from next layer

    auto gradInput = OUTPUT_VARIABLE(0);
    auto gradGamma = OUTPUT_VARIABLE(1);
    auto gradBeta = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

    float epsilon = block.getTArguments()->size() > 0 ? static_cast<float>(T_ARG(0)) : 1e-5f;

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        const float* gammaBuf = gamma->bufferAsT<float>();
        const float* gradOutBuf = gradOutput->bufferAsT<float>();
        float* gradInBuf = gradInput->bufferAsT<float>();
        float* gradGammaBuf = gradGamma->bufferAsT<float>();
        float* gradBetaBuf = gradBeta ? gradBeta->bufferAsT<float>() : nullptr;

        // Forward computations needed for backward
        float mean = 0.0f;
        vDSP_meanv(inBuf, 1, &mean, n);

        std::vector<float> xCentered(n);
        float negMean = -mean;
        vDSP_vsadd(inBuf, 1, &negMean, xCentered.data(), 1, n);

        std::vector<float> sqDeviations(n);
        vDSP_vsq(xCentered.data(), 1, sqDeviations.data(), 1, n);

        float variance = 0.0f;
        vDSP_meanv(sqDeviations.data(), 1, &variance, n);

        float invStd = 1.0f / sqrtf(variance + epsilon);

        // Normalized input
        std::vector<float> xHat(n);
        vDSP_vsmul(xCentered.data(), 1, &invStd, xHat.data(), 1, n);

        // grad_beta = sum(grad_output) - if applicable
        if (gradBetaBuf) {
            if (gradBeta->lengthOf() == 1) {
                float sum = 0.0f;
                vDSP_sve(gradOutBuf, 1, &sum, n);
                *gradBetaBuf = sum;
            } else {
                memcpy(gradBetaBuf, gradOutBuf, n * sizeof(float));
            }
        }

        // grad_gamma = sum(grad_output * x_hat)
        if (gradGamma->lengthOf() == 1) {
            float dotProd = 0.0f;
            vDSP_dotpr(gradOutBuf, 1, xHat.data(), 1, &dotProd, n);
            *gradGammaBuf = dotProd;
        } else {
            vDSP_vmul(gradOutBuf, 1, xHat.data(), 1, gradGammaBuf, 1, n);
        }

        // grad_input = gamma * inv_std * (grad_out - mean(grad_out) - x_hat * mean(grad_out * x_hat))
        std::vector<float> gradOutScaled(n);
        if (gamma->lengthOf() == 1) {
            vDSP_vsmul(gradOutBuf, 1, gammaBuf, gradOutScaled.data(), 1, n);
        } else {
            vDSP_vmul(gradOutBuf, 1, gammaBuf, 1, gradOutScaled.data(), 1, n);
        }

        // mean(grad_out_scaled)
        float meanGradOut = 0.0f;
        vDSP_meanv(gradOutScaled.data(), 1, &meanGradOut, n);

        // grad_out_scaled * x_hat and its mean
        std::vector<float> gradOutXHat(n);
        vDSP_vmul(gradOutScaled.data(), 1, xHat.data(), 1, gradOutXHat.data(), 1, n);
        float meanGradOutXHat = 0.0f;
        vDSP_meanv(gradOutXHat.data(), 1, &meanGradOutXHat, n);

        // grad_input = inv_std * (grad_out_scaled - mean_grad_out - x_hat * mean_grad_out_xhat)
        float negMeanGradOut = -meanGradOut;
        vDSP_vsadd(gradOutScaled.data(), 1, &negMeanGradOut, gradInBuf, 1, n);

        std::vector<float> correction(n);
        float negMeanGradOutXHat = -meanGradOutXHat;
        vDSP_vsmul(xHat.data(), 1, &negMeanGradOutXHat, correction.data(), 1, n);

        vDSP_vadd(gradInBuf, 1, correction.data(), 1, gradInBuf, 1, n);
        vDSP_vsmul(gradInBuf, 1, &invStd, gradInBuf, 1, n);

    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        const double* gammaBuf = gamma->bufferAsT<double>();
        const double* gradOutBuf = gradOutput->bufferAsT<double>();
        double* gradInBuf = gradInput->bufferAsT<double>();
        double* gradGammaBuf = gradGamma->bufferAsT<double>();
        double* gradBetaBuf = gradBeta ? gradBeta->bufferAsT<double>() : nullptr;

        double mean = 0.0;
        vDSP_meanvD(inBuf, 1, &mean, n);

        std::vector<double> xCentered(n);
        double negMean = -mean;
        vDSP_vsaddD(inBuf, 1, &negMean, xCentered.data(), 1, n);

        std::vector<double> sqDeviations(n);
        vDSP_vsqD(xCentered.data(), 1, sqDeviations.data(), 1, n);

        double variance = 0.0;
        vDSP_meanvD(sqDeviations.data(), 1, &variance, n);

        double invStd = 1.0 / sqrt(variance + static_cast<double>(epsilon));

        std::vector<double> xHat(n);
        vDSP_vsmulD(xCentered.data(), 1, &invStd, xHat.data(), 1, n);

        if (gradBetaBuf) {
            if (gradBeta->lengthOf() == 1) {
                double sum = 0.0;
                vDSP_sveD(gradOutBuf, 1, &sum, n);
                *gradBetaBuf = sum;
            } else {
                memcpy(gradBetaBuf, gradOutBuf, n * sizeof(double));
            }
        }

        if (gradGamma->lengthOf() == 1) {
            double dotProd = 0.0;
            vDSP_dotprD(gradOutBuf, 1, xHat.data(), 1, &dotProd, n);
            *gradGammaBuf = dotProd;
        } else {
            vDSP_vmulD(gradOutBuf, 1, xHat.data(), 1, gradGammaBuf, 1, n);
        }

        std::vector<double> gradOutScaled(n);
        if (gamma->lengthOf() == 1) {
            vDSP_vsmulD(gradOutBuf, 1, gammaBuf, gradOutScaled.data(), 1, n);
        } else {
            vDSP_vmulD(gradOutBuf, 1, gammaBuf, 1, gradOutScaled.data(), 1, n);
        }

        double meanGradOut = 0.0;
        vDSP_meanvD(gradOutScaled.data(), 1, &meanGradOut, n);

        std::vector<double> gradOutXHat(n);
        vDSP_vmulD(gradOutScaled.data(), 1, xHat.data(), 1, gradOutXHat.data(), 1, n);
        double meanGradOutXHat = 0.0;
        vDSP_meanvD(gradOutXHat.data(), 1, &meanGradOutXHat, n);

        double negMeanGradOut = -meanGradOut;
        vDSP_vsaddD(gradOutScaled.data(), 1, &negMeanGradOut, gradInBuf, 1, n);

        std::vector<double> correction(n);
        double negMeanGradOutXHat = -meanGradOutXHat;
        vDSP_vsmulD(xHat.data(), 1, &negMeanGradOutXHat, correction.data(), 1, n);

        vDSP_vaddD(gradInBuf, 1, correction.data(), 1, gradInBuf, 1, n);
        vDSP_vsmulD(gradInBuf, 1, &invStd, gradInBuf, 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(layer_norm_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto gamma = INPUT_VARIABLE(1);
    auto gradOutput = INPUT_VARIABLE(2);
    auto gradInput = OUTPUT_VARIABLE(0);

    Requirements reqs("LAYER_NORM_BP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, gradInput);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*gamma), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradOutput), EWS_MSG_INPUT2);
    reqs.expectTrue(accelerateUtils::isContiguous(*gradInput), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1, RANK_MSG_INPUT0);

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
