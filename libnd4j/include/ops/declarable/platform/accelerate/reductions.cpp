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
// Apple Accelerate framework - Reduction operations
// Uses vDSP for optimized reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod
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
// REDUCE_SUM - Sum reduction
//==============================================================================

PLATFORM_IMPL(reduce_sum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Check if we're reducing along all dimensions (full reduction)
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

    // Full reduction case (reduce to scalar)
    bool fullReduction = dims.empty() || (dims.size() == static_cast<size_t>(x->rankOf()));
    if (fullReduction && z->isScalar()) {
        const vDSP_Length n = static_cast<vDSP_Length>(x->lengthOf());

        if (x->dataType() == DataType::FLOAT32) {
            float sum;
            vDSP_sve(x->bufferAsT<float>(), 1, &sum, n);
            z->p(0, sum);
        } else if (x->dataType() == DataType::DOUBLE) {
            double sum;
            vDSP_sveD(x->bufferAsT<double>(), 1, &sum, n);
            z->p(0, sum);
        }
        return Status::OK;
    }

    // For partial reductions, fall back to generic implementation
    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_sum, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Check dimensions
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

    Requirements reqs("REDUCE_SUM ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    // Only support full reduction to scalar
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// REDUCE_MEAN - Mean reduction
//==============================================================================

PLATFORM_IMPL(reduce_mean, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    // Check if we're reducing along all dimensions
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
            float mean;
            vDSP_meanv(x->bufferAsT<float>(), 1, &mean, n);
            z->p(0, mean);
        } else if (x->dataType() == DataType::DOUBLE) {
            double mean;
            vDSP_meanvD(x->bufferAsT<double>(), 1, &mean, n);
            z->p(0, mean);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_mean, ENGINE_CPU) {
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

    Requirements reqs("REDUCE_MEAN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// REDUCE_MAX - Maximum reduction
//==============================================================================

PLATFORM_IMPL(reduce_max, ENGINE_CPU) {
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
            float maxVal;
            vDSP_maxv(x->bufferAsT<float>(), 1, &maxVal, n);
            z->p(0, maxVal);
        } else if (x->dataType() == DataType::DOUBLE) {
            double maxVal;
            vDSP_maxvD(x->bufferAsT<double>(), 1, &maxVal, n);
            z->p(0, maxVal);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_max, ENGINE_CPU) {
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

    Requirements reqs("REDUCE_MAX ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// REDUCE_MIN - Minimum reduction
//==============================================================================

PLATFORM_IMPL(reduce_min, ENGINE_CPU) {
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
            vDSP_minv(x->bufferAsT<float>(), 1, &minVal, n);
            z->p(0, minVal);
        } else if (x->dataType() == DataType::DOUBLE) {
            double minVal;
            vDSP_minvD(x->bufferAsT<double>(), 1, &minVal, n);
            z->p(0, minVal);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_min, ENGINE_CPU) {
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

    Requirements reqs("REDUCE_MIN ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// REDUCE_PROD - Product reduction
//==============================================================================

PLATFORM_IMPL(reduce_prod, ENGINE_CPU) {
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

        // Use log-sum-exp trick for numerical stability: prod(x) = exp(sum(log(x)))
        // But for small arrays, direct multiplication via vDSP is fine
        if (x->dataType() == DataType::FLOAT32) {
            const float* buf = x->bufferAsT<float>();
            float prod = 1.0f;
            // vDSP doesn't have a direct product function, use loop or log approach
            // For correctness, we'll use the straightforward approach
            for (int i = 0; i < n; i++) {
                prod *= buf[i];
            }
            z->p(0, prod);
        } else if (x->dataType() == DataType::DOUBLE) {
            const double* buf = x->bufferAsT<double>();
            double prod = 1.0;
            for (int i = 0; i < n; i++) {
                prod *= buf[i];
            }
            z->p(0, prod);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_prod, ENGINE_CPU) {
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

    Requirements reqs("REDUCE_PROD ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// REDUCE_NORM2 - L2 Norm (Euclidean norm)
//==============================================================================

PLATFORM_IMPL(reduce_norm2, ENGINE_CPU) {
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
            // Use BLAS nrm2
            float norm = cblas_snrm2(n, x->bufferAsT<float>(), 1);
            z->p(0, norm);
        } else if (x->dataType() == DataType::DOUBLE) {
            double norm = cblas_dnrm2(n, x->bufferAsT<double>(), 1);
            z->p(0, norm);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_norm2, ENGINE_CPU) {
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

    Requirements reqs("REDUCE_NORM2 ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, nullptr);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(fullReduction && z->isScalar(), "Only full reduction to scalar is supported");

    return reqs.ok();
}

//==============================================================================
// REDUCE_NORM1 - L1 Norm (sum of absolute values)
//==============================================================================

PLATFORM_IMPL(reduce_norm1, ENGINE_CPU) {
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
            // Use BLAS asum
            float norm = cblas_sasum(n, x->bufferAsT<float>(), 1);
            z->p(0, norm);
        } else if (x->dataType() == DataType::DOUBLE) {
            double norm = cblas_dasum(n, x->bufferAsT<double>(), 1);
            z->p(0, norm);
        }
        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(reduce_norm1, ENGINE_CPU) {
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

    Requirements reqs("REDUCE_NORM1 ACCELERATE");
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
