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
// Apple Accelerate framework - Statistical reduction operations
// Variance, standard deviation, logsumexp, and moments
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
// REDUCE_VARIANCE - Compute variance using vDSP
//==============================================================================

PLATFORM_IMPL(reduce_variance, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    bool biasCorrected = block.getBArguments()->size() > 0 ? B_ARG(0) : false;
    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Compute mean
        float mean = 0.0f;
        vDSP_meanv(inBuf, 1, &mean, n);

        // Compute squared deviations and sum
        std::vector<float> deviations(n);
        float negMean = -mean;
        vDSP_vsadd(inBuf, 1, &negMean, deviations.data(), 1, n);
        vDSP_vsq(deviations.data(), 1, deviations.data(), 1, n);

        float sumSq = 0.0f;
        vDSP_sve(deviations.data(), 1, &sumSq, n);

        // Compute variance
        if (biasCorrected && n > 1) {
            *outBuf = sumSq / static_cast<float>(n - 1);
        } else {
            *outBuf = sumSq / static_cast<float>(n);
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        double mean = 0.0;
        vDSP_meanvD(inBuf, 1, &mean, n);

        std::vector<double> deviations(n);
        double negMean = -mean;
        vDSP_vsaddD(inBuf, 1, &negMean, deviations.data(), 1, n);
        vDSP_vsqD(deviations.data(), 1, deviations.data(), 1, n);

        double sumSq = 0.0;
        vDSP_sveD(deviations.data(), 1, &sumSq, n);

        if (biasCorrected && n > 1) {
            *outBuf = sumSq / static_cast<double>(n - 1);
        } else {
            *outBuf = sumSq / static_cast<double>(n);
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(reduce_variance, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("REDUCE_VARIANCE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    // Only support full reduction for now
    reqs.expectTrue(output->isScalar(), RANK_MSG_OUTPUT);

    return reqs.ok();
}

//==============================================================================
// REDUCE_STDEV - Compute standard deviation using vDSP
//==============================================================================

PLATFORM_IMPL(reduce_stdev, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    bool biasCorrected = block.getBArguments()->size() > 0 ? B_ARG(0) : false;
    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Compute mean
        float mean = 0.0f;
        vDSP_meanv(inBuf, 1, &mean, n);

        // Compute squared deviations and sum
        std::vector<float> deviations(n);
        float negMean = -mean;
        vDSP_vsadd(inBuf, 1, &negMean, deviations.data(), 1, n);
        vDSP_vsq(deviations.data(), 1, deviations.data(), 1, n);

        float sumSq = 0.0f;
        vDSP_sve(deviations.data(), 1, &sumSq, n);

        // Compute variance then sqrt for stdev
        float variance;
        if (biasCorrected && n > 1) {
            variance = sumSq / static_cast<float>(n - 1);
        } else {
            variance = sumSq / static_cast<float>(n);
        }
        int one = 1;
        vvsqrtf(outBuf, &variance, &one);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        double mean = 0.0;
        vDSP_meanvD(inBuf, 1, &mean, n);

        std::vector<double> deviations(n);
        double negMean = -mean;
        vDSP_vsaddD(inBuf, 1, &negMean, deviations.data(), 1, n);
        vDSP_vsqD(deviations.data(), 1, deviations.data(), 1, n);

        double sumSq = 0.0;
        vDSP_sveD(deviations.data(), 1, &sumSq, n);

        double variance;
        if (biasCorrected && n > 1) {
            variance = sumSq / static_cast<double>(n - 1);
        } else {
            variance = sumSq / static_cast<double>(n);
        }
        int one = 1;
        vvsqrt(outBuf, &variance, &one);
    }

    return Status::OK;
}

PLATFORM_CHECK(reduce_stdev, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("REDUCE_STDEV ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(output->isScalar(), RANK_MSG_OUTPUT);

    return reqs.ok();
}

//==============================================================================
// REDUCE_LOGSUMEXP - Compute log(sum(exp(x))) using vForce
//==============================================================================

PLATFORM_IMPL(reduce_logsumexp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());
    const int nInt = static_cast<int>(n);

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        // Find max for numerical stability
        float maxVal = 0.0f;
        vDSP_maxv(inBuf, 1, &maxVal, n);

        // Compute exp(x - max)
        std::vector<float> expVals(n);
        float negMax = -maxVal;
        vDSP_vsadd(inBuf, 1, &negMax, expVals.data(), 1, n);
        vvexpf(expVals.data(), expVals.data(), &nInt);

        // Sum the exponentials
        float sumExp = 0.0f;
        vDSP_sve(expVals.data(), 1, &sumExp, n);

        // Compute log(sum) + max
        int one = 1;
        vvlogf(outBuf, &sumExp, &one);
        *outBuf += maxVal;
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        double maxVal = 0.0;
        vDSP_maxvD(inBuf, 1, &maxVal, n);

        std::vector<double> expVals(n);
        double negMax = -maxVal;
        vDSP_vsaddD(inBuf, 1, &negMax, expVals.data(), 1, n);
        vvexp(expVals.data(), expVals.data(), &nInt);

        double sumExp = 0.0;
        vDSP_sveD(expVals.data(), 1, &sumExp, n);

        int one = 1;
        vvlog(outBuf, &sumExp, &one);
        *outBuf += maxVal;
    }

    return Status::OK;
}

PLATFORM_CHECK(reduce_logsumexp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("REDUCE_LOGSUMEXP ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(output->isScalar(), RANK_MSG_OUTPUT);

    return reqs.ok();
}

//==============================================================================
// MOMENTS - Compute mean and variance simultaneously
//==============================================================================

PLATFORM_IMPL(moments, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto meanOut = OUTPUT_VARIABLE(0);
    auto varOut = OUTPUT_VARIABLE(1);

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* meanBuf = meanOut->bufferAsT<float>();
        float* varBuf = varOut->bufferAsT<float>();

        // Compute mean
        float mean = 0.0f;
        vDSP_meanv(inBuf, 1, &mean, n);
        *meanBuf = mean;

        // Compute squared deviations and sum
        std::vector<float> deviations(n);
        float negMean = -mean;
        vDSP_vsadd(inBuf, 1, &negMean, deviations.data(), 1, n);
        vDSP_vsq(deviations.data(), 1, deviations.data(), 1, n);

        float sumSq = 0.0f;
        vDSP_sve(deviations.data(), 1, &sumSq, n);

        // Population variance (not bias corrected)
        *varBuf = sumSq / static_cast<float>(n);
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* meanBuf = meanOut->bufferAsT<double>();
        double* varBuf = varOut->bufferAsT<double>();

        double mean = 0.0;
        vDSP_meanvD(inBuf, 1, &mean, n);
        *meanBuf = mean;

        std::vector<double> deviations(n);
        double negMean = -mean;
        vDSP_vsaddD(inBuf, 1, &negMean, deviations.data(), 1, n);
        vDSP_vsqD(deviations.data(), 1, deviations.data(), 1, n);

        double sumSq = 0.0;
        vDSP_sveD(deviations.data(), 1, &sumSq, n);

        *varBuf = sumSq / static_cast<double>(n);
    }

    return Status::OK;
}

PLATFORM_CHECK(moments, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto meanOut = OUTPUT_VARIABLE(0);
    auto varOut = OUTPUT_VARIABLE(1);

    Requirements reqs("MOMENTS ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, meanOut);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(meanOut->isScalar(), RANK_MSG_OUTPUT);
    reqs.expectTrue(varOut->isScalar(), "Second output must be scalar");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
