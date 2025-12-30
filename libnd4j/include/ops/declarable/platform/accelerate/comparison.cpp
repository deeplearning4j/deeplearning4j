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
// Apple Accelerate framework - Comparison and element-wise min/max operations
// Uses vDSP for optimized maximum, minimum, clip, sign operations
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
// MAXIMUM - Element-wise maximum of two arrays
//==============================================================================

PLATFORM_IMPL(maximum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vDSP_vmax(x->bufferAsT<float>(), 1,
                  y->bufferAsT<float>(), 1,
                  z->bufferAsT<float>(), 1, n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vDSP_vmaxD(x->bufferAsT<double>(), 1,
                   y->bufferAsT<double>(), 1,
                   z->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(maximum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("MAXIMUM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);

    return reqs.ok();
}

//==============================================================================
// MINIMUM - Element-wise minimum of two arrays
//==============================================================================

PLATFORM_IMPL(minimum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vDSP_vmin(x->bufferAsT<float>(), 1,
                  y->bufferAsT<float>(), 1,
                  z->bufferAsT<float>(), 1, n);
    } else if (x->dataType() == DataType::DOUBLE) {
        vDSP_vminD(x->bufferAsT<double>(), 1,
                   y->bufferAsT<double>(), 1,
                   z->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(minimum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("MINIMUM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(x->isSameShape(y), SHAPE_MSG_INPUT1);

    return reqs.ok();
}

//==============================================================================
// CLIP_BY_VALUE - Clip values to a range [min, max]
//==============================================================================

PLATFORM_IMPL(clip_by_value, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Get min and max from T arguments
    float minVal = T_ARG(0);
    float maxVal = T_ARG(1);

    const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        vDSP_vclip(x->bufferAsT<float>(), 1,
                   &minVal, &maxVal,
                   z->bufferAsT<float>(), 1, n);
    } else if (x->dataType() == DataType::DOUBLE) {
        double minValD = static_cast<double>(minVal);
        double maxValD = static_cast<double>(maxVal);
        vDSP_vclipD(x->bufferAsT<double>(), 1,
                    &minValD, &maxValD,
                    z->bufferAsT<double>(), 1, n);
    }

    return Status::OK;
}

PLATFORM_CHECK(clip_by_value, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("CLIP_BY_VALUE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(block.getTArguments()->size() >= 2, "Need min and max T arguments");

    return reqs.ok();
}

//==============================================================================
// SIGN - Sign function: -1 for x<0, 0 for x==0, 1 for x>0
//==============================================================================

PLATFORM_IMPL(sign, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    const int n = static_cast<int>(x->lengthOf());

    if (x->dataType() == DataType::FLOAT32) {
        const float* inBuf = x->bufferAsT<float>();
        float* outBuf = z->bufferAsT<float>();

        // vDSP doesn't have a direct sign function, implement using threshold operations
        // sign(x) = (x > 0) - (x < 0)
        for (int i = 0; i < n; i++) {
            if (inBuf[i] > 0.0f) {
                outBuf[i] = 1.0f;
            } else if (inBuf[i] < 0.0f) {
                outBuf[i] = -1.0f;
            } else {
                outBuf[i] = 0.0f;
            }
        }
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* inBuf = x->bufferAsT<double>();
        double* outBuf = z->bufferAsT<double>();

        for (int i = 0; i < n; i++) {
            if (inBuf[i] > 0.0) {
                outBuf[i] = 1.0;
            } else if (inBuf[i] < 0.0) {
                outBuf[i] = -1.0;
            } else {
                outBuf[i] = 0.0;
            }
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(sign, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("SIGN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);

    return reqs.ok();
}

//==============================================================================
// ARGMAX - Index of maximum value
//==============================================================================

PLATFORM_IMPL(argmax, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Check if this is a full reduction (no dimension specified or all dimensions)
    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));

    if (fullReduction && z->isScalar()) {
        const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

        if (x->dataType() == DataType::FLOAT32) {
            float maxVal;
            vDSP_Length maxIdx;
            vDSP_maxvi(x->bufferAsT<float>(), 1, &maxVal, &maxIdx, n);
            z->p(0, static_cast<LongType>(maxIdx));
        } else if (x->dataType() == DataType::DOUBLE) {
            double maxVal;
            vDSP_Length maxIdx;
            vDSP_maxviD(x->bufferAsT<double>(), 1, &maxVal, &maxIdx, n);
            z->p(0, static_cast<LongType>(maxIdx));
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(argmax, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));

    Requirements reqs("ARGMAX ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// ARGMIN - Index of minimum value
//==============================================================================

PLATFORM_IMPL(argmin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));

    if (fullReduction && z->isScalar()) {
        const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

        if (x->dataType() == DataType::FLOAT32) {
            float minVal;
            vDSP_Length minIdx;
            vDSP_minvi(x->bufferAsT<float>(), 1, &minVal, &minIdx, n);
            z->p(0, static_cast<LongType>(minIdx));
        } else if (x->dataType() == DataType::DOUBLE) {
            double minVal;
            vDSP_Length minIdx;
            vDSP_minviD(x->bufferAsT<double>(), 1, &minVal, &minIdx, n);
            z->p(0, static_cast<LongType>(minIdx));
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(argmin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));

    Requirements reqs("ARGMIN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// ABS_MAX - Maximum of absolute values
//==============================================================================

PLATFORM_IMPL(reduce_amax, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));

    if (fullReduction && z->isScalar()) {
        const int n = static_cast<int>(x->lengthOf());

        if (x->dataType() == DataType::FLOAT32) {
            // Use BLAS idamax which returns index of max abs value
            int idx = cblas_isamax(n, x->bufferAsT<float>(), 1);
            float maxAbsVal = std::abs(x->bufferAsT<float>()[idx]);
            z->p(0, maxAbsVal);
        } else if (x->dataType() == DataType::DOUBLE) {
            int idx = cblas_idamax(n, x->bufferAsT<double>(), 1);
            double maxAbsVal = std::abs(x->bufferAsT<double>()[idx]);
            z->p(0, maxAbsVal);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_amax, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    std::vector<LongType> dims;
    if (block.width() > 1) {
        auto axisTensor = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axisTensor->lengthOf(); i++) {
            dims.push_back(axisTensor->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (auto d : *block.getIArguments()) {
            dims.push_back(d);
        }
    }

    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));

    Requirements reqs("REDUCE_AMAX ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
