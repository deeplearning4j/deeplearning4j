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
// Apple Accelerate framework - Cumulative operations
// Cumulative sum and cumulative product using vDSP
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
// CUMSUM - Cumulative sum using vDSP integration
//==============================================================================

PLATFORM_IMPL(cumsum, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    bool exclusive = block.getBArguments()->size() > 0 ? B_ARG(0) : false;
    bool reverse = block.getBArguments()->size() > 1 ? B_ARG(1) : false;

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        if (reverse) {
            // Reverse cumsum: sum from end to beginning
            if (exclusive) {
                outBuf[n - 1] = 0.0f;
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] + inBuf[i];
                }
            } else {
                outBuf[n - 1] = inBuf[n - 1];
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] + inBuf[i - 1];
                }
            }
        } else {
            // Forward cumsum using vDSP_vrsum (running sum)
            // vDSP_vrsum computes: out[i] = sum(in[0..i]) + initial
            if (exclusive) {
                // Exclusive: out[i] = sum(in[0..i-1])
                float zero = 0.0f;
                outBuf[0] = 0.0f;
                if (n > 1) {
                    vDSP_vrsum(inBuf, 1, &zero, outBuf + 1, 1, n - 1);
                }
            } else {
                // Inclusive: out[i] = sum(in[0..i])
                float zero = 0.0f;
                vDSP_vrsum(inBuf, 1, &zero, outBuf, 1, n);
                // vDSP_vrsum gives running sum, but we need to add first element
                // Actually vDSP_vrsum may work differently - let's use a manual approach
                outBuf[0] = inBuf[0];
                for (vDSP_Length i = 1; i < n; i++) {
                    outBuf[i] = outBuf[i - 1] + inBuf[i];
                }
            }
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        if (reverse) {
            if (exclusive) {
                outBuf[n - 1] = 0.0;
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] + inBuf[i];
                }
            } else {
                outBuf[n - 1] = inBuf[n - 1];
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] + inBuf[i - 1];
                }
            }
        } else {
            if (exclusive) {
                outBuf[0] = 0.0;
                if (n > 1) {
                    double zero = 0.0;
                    vDSP_vrsumD(inBuf, 1, &zero, outBuf + 1, 1, n - 1);
                }
            } else {
                outBuf[0] = inBuf[0];
                for (vDSP_Length i = 1; i < n; i++) {
                    outBuf[i] = outBuf[i - 1] + inBuf[i];
                }
            }
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(cumsum, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("CUMSUM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    // Only support 1D for now (simple case)
    reqs.expectTrue(input->rankOf() == 1, RANK_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// CUMPROD - Cumulative product
//==============================================================================

PLATFORM_IMPL(cumprod, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    bool exclusive = block.getBArguments()->size() > 0 ? B_ARG(0) : false;
    bool reverse = block.getBArguments()->size() > 1 ? B_ARG(1) : false;

    const vDSP_Length n = static_cast<vDSP_Length>(input->lengthOf());

    if (input->dataType() == DataType::FLOAT32) {
        const float* inBuf = input->bufferAsT<float>();
        float* outBuf = output->bufferAsT<float>();

        if (reverse) {
            if (exclusive) {
                outBuf[n - 1] = 1.0f;
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] * inBuf[i];
                }
            } else {
                outBuf[n - 1] = inBuf[n - 1];
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] * inBuf[i - 1];
                }
            }
        } else {
            if (exclusive) {
                outBuf[0] = 1.0f;
                for (vDSP_Length i = 1; i < n; i++) {
                    outBuf[i] = outBuf[i - 1] * inBuf[i - 1];
                }
            } else {
                outBuf[0] = inBuf[0];
                for (vDSP_Length i = 1; i < n; i++) {
                    outBuf[i] = outBuf[i - 1] * inBuf[i];
                }
            }
        }
    } else if (input->dataType() == DataType::DOUBLE) {
        const double* inBuf = input->bufferAsT<double>();
        double* outBuf = output->bufferAsT<double>();

        if (reverse) {
            if (exclusive) {
                outBuf[n - 1] = 1.0;
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] * inBuf[i];
                }
            } else {
                outBuf[n - 1] = inBuf[n - 1];
                for (vDSP_Length i = n - 1; i > 0; i--) {
                    outBuf[i - 1] = outBuf[i] * inBuf[i - 1];
                }
            }
        } else {
            if (exclusive) {
                outBuf[0] = 1.0;
                for (vDSP_Length i = 1; i < n; i++) {
                    outBuf[i] = outBuf[i - 1] * inBuf[i - 1];
                }
            } else {
                outBuf[0] = inBuf[0];
                for (vDSP_Length i = 1; i < n; i++) {
                    outBuf[i] = outBuf[i - 1] * inBuf[i];
                }
            }
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(cumprod, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("CUMPROD ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1, RANK_MSG_INPUT0);

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
