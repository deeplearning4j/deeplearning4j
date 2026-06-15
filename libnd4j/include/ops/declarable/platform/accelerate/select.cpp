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
// Apple Accelerate framework - Selection and indexing operations
// Uses vDSP for optimized select/where, gather, scatter operations
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
// WHERE (SELECT) - Conditional selection: output = cond ? x : y
//==============================================================================

PLATFORM_IMPL(where, ENGINE_CPU) {
    auto condition = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);
    auto z = OUTPUT_VARIABLE(0);

    const LongType n = condition->lengthOf();

    if (x->dataType() == DataType::FLOAT32) {
        const float* condBuf = condition->bufferAsT<float>();
        const float* xBuf = x->bufferAsT<float>();
        const float* yBuf = y->bufferAsT<float>();
        float* zBuf = z->bufferAsT<float>();

        // vDSP doesn't have a direct select, but we can use masking
        // z = cond * x + (1 - cond) * y for boolean conditions
        for (LongType i = 0; i < n; i++) {
            zBuf[i] = (condBuf[i] != 0.0f) ? xBuf[i] : yBuf[i];
        }
    } else if (x->dataType() == DataType::DOUBLE) {
        const double* condBuf = condition->bufferAsT<double>();
        const double* xBuf = x->bufferAsT<double>();
        const double* yBuf = y->bufferAsT<double>();
        double* zBuf = z->bufferAsT<double>();

        for (LongType i = 0; i < n; i++) {
            zBuf[i] = (condBuf[i] != 0.0) ? xBuf[i] : yBuf[i];
        }
    }

    return Status::OK;
}

PLATFORM_CHECK(where, ENGINE_CPU) {
    auto condition = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto y = INPUT_VARIABLE(2);
    auto z = OUTPUT_VARIABLE(0);

    Requirements reqs("WHERE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, x, z);
    reqs.expectTrue(accelerateUtils::isContiguous(*condition), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*x), EWS_MSG_INPUT1);
    reqs.expectTrue(accelerateUtils::isContiguous(*y), EWS_MSG_INPUT2);
    reqs.expectTrue(accelerateUtils::isContiguous(*z), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(x->dataType()), TYPE_MSG_INPUT1);
    // All must have same shape
    reqs.expectTrue(condition->isSameShape(x), "Condition must match x shape");
    reqs.expectTrue(x->isSameShape(y), "x and y must have same shape");

    return reqs.ok();
}

//==============================================================================
// GATHER - Gather elements by indices
//==============================================================================

PLATFORM_IMPL(gather, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    if (axis < 0) axis += input->rankOf();

    // For 1D case (simple gather)
    if (input->rankOf() == 1 && indices->rankOf() == 1) {
        const LongType n = indices->lengthOf();

        if (input->dataType() == DataType::FLOAT32) {
            const float* inBuf = input->bufferAsT<float>();
            float* outBuf = output->bufferAsT<float>();

            // vDSP_vgathr gathers elements using indices
            // Note: vDSP uses 1-based indices for some functions
            std::vector<vDSP_Length> vdspIndices(n);
            for (LongType i = 0; i < n; i++) {
                vdspIndices[i] = static_cast<vDSP_Length>(indices->e<LongType>(i));
            }

            // Manual gather (vDSP_vgathr uses different index semantics)
            for (LongType i = 0; i < n; i++) {
                outBuf[i] = inBuf[vdspIndices[i]];
            }
        } else if (input->dataType() == DataType::DOUBLE) {
            const double* inBuf = input->bufferAsT<double>();
            double* outBuf = output->bufferAsT<double>();

            for (LongType i = 0; i < n; i++) {
                LongType idx = indices->e<LongType>(i);
                outBuf[i] = inBuf[idx];
            }
        }

        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(gather, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("GATHER ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    // Only support 1D case
    reqs.expectTrue(input->rankOf() == 1 && indices->rankOf() == 1, "Only 1D gather supported");

    return reqs.ok();
}

//==============================================================================
// SCATTER_UPDATE - Scatter values into tensor at indices
//==============================================================================

PLATFORM_IMPL(scatter_update, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto updates = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    // Copy input to output first
    if (output != input) {
        output->assign(input);
    }

    // For 1D case
    if (input->rankOf() == 1 && indices->rankOf() == 1) {
        const LongType n = indices->lengthOf();

        if (input->dataType() == DataType::FLOAT32) {
            const float* updateBuf = updates->bufferAsT<float>();
            float* outBuf = output->bufferAsT<float>();

            for (LongType i = 0; i < n; i++) {
                LongType idx = indices->e<LongType>(i);
                outBuf[idx] = updateBuf[i];
            }
        } else if (input->dataType() == DataType::DOUBLE) {
            const double* updateBuf = updates->bufferAsT<double>();
            double* outBuf = output->bufferAsT<double>();

            for (LongType i = 0; i < n; i++) {
                LongType idx = indices->e<LongType>(i);
                outBuf[idx] = updateBuf[i];
            }
        }

        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(scatter_update, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto updates = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("SCATTER_UPDATE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*updates), EWS_MSG_INPUT2);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1 && indices->rankOf() == 1, "Only 1D scatter supported");

    return reqs.ok();
}

//==============================================================================
// SCATTER_ADD - Scatter-add values into tensor at indices
//==============================================================================

PLATFORM_IMPL(scatter_add, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto updates = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    // Copy input to output first
    if (output != input) {
        output->assign(input);
    }

    // For 1D case
    if (input->rankOf() == 1 && indices->rankOf() == 1) {
        const LongType n = indices->lengthOf();

        if (input->dataType() == DataType::FLOAT32) {
            const float* updateBuf = updates->bufferAsT<float>();
            float* outBuf = output->bufferAsT<float>();

            for (LongType i = 0; i < n; i++) {
                LongType idx = indices->e<LongType>(i);
                outBuf[idx] += updateBuf[i];
            }
        } else if (input->dataType() == DataType::DOUBLE) {
            const double* updateBuf = updates->bufferAsT<double>();
            double* outBuf = output->bufferAsT<double>();

            for (LongType i = 0; i < n; i++) {
                LongType idx = indices->e<LongType>(i);
                outBuf[idx] += updateBuf[i];
            }
        }

        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(scatter_add, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto updates = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("SCATTER_ADD ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*updates), EWS_MSG_INPUT2);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1 && indices->rankOf() == 1, "Only 1D scatter supported");

    return reqs.ok();
}

//==============================================================================
// PAD - Pad tensor with constant value
//==============================================================================

PLATFORM_IMPL(pad, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto paddings = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    double padValue = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;

    // For 1D case
    if (input->rankOf() == 1) {
        LongType padBefore = paddings->e<LongType>(0, 0);
        LongType padAfter = paddings->e<LongType>(0, 1);
        LongType inputLen = input->lengthOf();

        if (input->dataType() == DataType::FLOAT32) {
            float* outBuf = output->bufferAsT<float>();
            const float* inBuf = input->bufferAsT<float>();
            float padValF = static_cast<float>(padValue);

            // Fill with pad value
            vDSP_vfill(&padValF, outBuf, 1, static_cast<vDSP_Length>(output->lengthOf()));

            // Copy input to middle
            cblas_scopy(static_cast<int>(inputLen), inBuf, 1, outBuf + padBefore, 1);
        } else if (input->dataType() == DataType::DOUBLE) {
            double* outBuf = output->bufferAsT<double>();
            const double* inBuf = input->bufferAsT<double>();

            vDSP_vfillD(&padValue, outBuf, 1, static_cast<vDSP_Length>(output->lengthOf()));
            cblas_dcopy(static_cast<int>(inputLen), inBuf, 1, outBuf + padBefore, 1);
        }

        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(pad, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("PAD ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1, "Only 1D padding supported");

    return reqs.ok();
}

//==============================================================================
// TILE - Tile/repeat tensor
//==============================================================================

PLATFORM_IMPL(tile, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto reps = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    // For 1D case
    if (input->rankOf() == 1) {
        LongType numReps = reps->e<LongType>(0);
        LongType inputLen = input->lengthOf();

        if (input->dataType() == DataType::FLOAT32) {
            float* outBuf = output->bufferAsT<float>();
            const float* inBuf = input->bufferAsT<float>();

            // Copy input repeated numReps times
            for (LongType r = 0; r < numReps; r++) {
                cblas_scopy(static_cast<int>(inputLen), inBuf, 1, outBuf + r * inputLen, 1);
            }
        } else if (input->dataType() == DataType::DOUBLE) {
            double* outBuf = output->bufferAsT<double>();
            const double* inBuf = input->bufferAsT<double>();

            for (LongType r = 0; r < numReps; r++) {
                cblas_dcopy(static_cast<int>(inputLen), inBuf, 1, outBuf + r * inputLen, 1);
            }
        }

        return Status::OK;
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(tile, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements reqs("TILE ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, input, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*input), EWS_MSG_INPUT0);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(accelerateUtils::isAccelerateSupported(input->dataType()), TYPE_MSG_INPUT0);
    reqs.expectTrue(input->rankOf() == 1, "Only 1D tiling supported");

    return reqs.ok();
}

//==============================================================================
// CONCAT - Concatenate arrays along axis
//==============================================================================

PLATFORM_IMPL(concat, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);
    int axis = INT_ARG(0);

    // For 1D case (axis=0)
    if (axis == 0) {
        bool allFloat = true;
        bool allDouble = true;
        bool all1D = true;

        for (int i = 0; i < block.width(); i++) {
            auto arr = INPUT_VARIABLE(i);
            if (arr->dataType() != DataType::FLOAT32) allFloat = false;
            if (arr->dataType() != DataType::DOUBLE) allDouble = false;
            if (arr->rankOf() != 1) all1D = false;
        }

        if (all1D && (allFloat || allDouble)) {
            LongType offset = 0;

            for (int i = 0; i < block.width(); i++) {
                auto arr = INPUT_VARIABLE(i);
                LongType len = arr->lengthOf();

                if (allFloat) {
                    cblas_scopy(static_cast<int>(len),
                               arr->bufferAsT<float>(), 1,
                               output->bufferAsT<float>() + offset, 1);
                } else {
                    cblas_dcopy(static_cast<int>(len),
                               arr->bufferAsT<double>(), 1,
                               output->bufferAsT<double>() + offset, 1);
                }

                offset += len;
            }

            return Status::OK;
        }
    }

    return Status::KERNEL_FAILURE;
}

PLATFORM_CHECK(concat, ENGINE_CPU) {
    auto output = OUTPUT_VARIABLE(0);
    int axis = INT_ARG(0);

    bool all1D = true;
    bool allContiguous = true;
    bool allSupported = true;

    for (int i = 0; i < block.width(); i++) {
        auto arr = INPUT_VARIABLE(i);
        if (arr->rankOf() != 1) all1D = false;
        if (!accelerateUtils::isContiguous(*arr)) allContiguous = false;
        if (!accelerateUtils::isAccelerateSupported(arr->dataType())) allSupported = false;
    }

    Requirements reqs("CONCAT ACCELERATE");
    accelerateUtils::checkAccelerateRequirements(reqs, block, nullptr, output);
    reqs.expectTrue(accelerateUtils::isContiguous(*output), EWS_MSG_OUTPUT0);
    reqs.expectTrue(axis == 0, "Only axis=0 supported");
    reqs.expectTrue(all1D, "Only 1D arrays supported");
    reqs.expectTrue(allContiguous, "All inputs must be contiguous");
    reqs.expectTrue(allSupported, "All inputs must have supported dtype");

    return reqs.ok();
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_ACCELERATE
