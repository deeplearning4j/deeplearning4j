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
// @author Adam Gibson
//
// Metal Performance Shaders - Reduction operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cfloat>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Reduce Sum
//////////////////////////////////////////////////////////////////////////

static void reduceSumMPS(NDArray* input, NDArray* output, const std::vector<LongType>& axes, bool keepDims) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        if (axes.empty()) {
            // Full reduction - sum all elements
            float sum = 0.0f;
            for (LongType i = 0; i < input->lengthOf(); i++) {
                sum += inPtr[i];
            }
            outPtr[0] = sum;
        } else {
            // Partial reduction along specified axes
            // For simplicity, initialize output to 0 and accumulate
            output->nullify();

            // Get input shape info
            auto inShape = input->shapeOf();
            auto inStrides = input->stridesOf();
            int rank = input->rankOf();

            // Create a set of reduction axes for quick lookup
            std::set<int> reduceAxes(axes.begin(), axes.end());

            // Iterate through all input elements
            for (LongType i = 0; i < input->lengthOf(); i++) {
                // Convert linear index to coordinates
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                // Compute output index (skip reduced dimensions)
                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                outPtr[outIdx] += inPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(reduce_sum, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }

    reduceSumMPS(input, output, axes, keepDims);

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_sum, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_SUM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reduce Mean
//////////////////////////////////////////////////////////////////////////

static void reduceMeanMPS(NDArray* input, NDArray* output, const std::vector<LongType>& axes, bool keepDims) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        if (axes.empty()) {
            // Full reduction
            float sum = 0.0f;
            LongType count = input->lengthOf();
            for (LongType i = 0; i < count; i++) {
                sum += inPtr[i];
            }
            outPtr[0] = sum / count;
        } else {
            // Partial reduction
            output->nullify();

            auto inShape = input->shapeOf();
            int rank = input->rankOf();
            std::set<int> reduceAxes(axes.begin(), axes.end());

            // Calculate reduction size
            LongType reduceSize = 1;
            for (int ax : axes) {
                reduceSize *= inShape[ax];
            }

            // Sum elements
            for (LongType i = 0; i < input->lengthOf(); i++) {
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                outPtr[outIdx] += inPtr[i];
            }

            // Divide by reduction size
            for (LongType i = 0; i < output->lengthOf(); i++) {
                outPtr[i] /= reduceSize;
            }
        }
    }
}

PLATFORM_IMPL(reduce_mean, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }

    reduceMeanMPS(input, output, axes, keepDims);

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_mean, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_MEAN OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reduce Max
//////////////////////////////////////////////////////////////////////////

static void reduceMaxMPS(NDArray* input, NDArray* output, const std::vector<LongType>& axes, bool keepDims) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        if (axes.empty()) {
            // Full reduction
            float maxVal = -FLT_MAX;
            for (LongType i = 0; i < input->lengthOf(); i++) {
                if (inPtr[i] > maxVal) {
                    maxVal = inPtr[i];
                }
            }
            outPtr[0] = maxVal;
        } else {
            // Initialize output to -FLT_MAX
            for (LongType i = 0; i < output->lengthOf(); i++) {
                outPtr[i] = -FLT_MAX;
            }

            auto inShape = input->shapeOf();
            int rank = input->rankOf();
            std::set<int> reduceAxes(axes.begin(), axes.end());

            for (LongType i = 0; i < input->lengthOf(); i++) {
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                if (inPtr[i] > outPtr[outIdx]) {
                    outPtr[outIdx] = inPtr[i];
                }
            }
        }
    }
}

PLATFORM_IMPL(reduce_max, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }

    reduceMaxMPS(input, output, axes, keepDims);

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_max, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_MAX OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reduce Min
//////////////////////////////////////////////////////////////////////////

static void reduceMinMPS(NDArray* input, NDArray* output, const std::vector<LongType>& axes, bool keepDims) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        if (axes.empty()) {
            // Full reduction
            float minVal = FLT_MAX;
            for (LongType i = 0; i < input->lengthOf(); i++) {
                if (inPtr[i] < minVal) {
                    minVal = inPtr[i];
                }
            }
            outPtr[0] = minVal;
        } else {
            // Initialize output to FLT_MAX
            for (LongType i = 0; i < output->lengthOf(); i++) {
                outPtr[i] = FLT_MAX;
            }

            auto inShape = input->shapeOf();
            int rank = input->rankOf();
            std::set<int> reduceAxes(axes.begin(), axes.end());

            for (LongType i = 0; i < input->lengthOf(); i++) {
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                if (inPtr[i] < outPtr[outIdx]) {
                    outPtr[outIdx] = inPtr[i];
                }
            }
        }
    }
}

PLATFORM_IMPL(reduce_min, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }

    reduceMinMPS(input, output, axes, keepDims);

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_min, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_MIN OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reduce Prod
//////////////////////////////////////////////////////////////////////////

static void reduceProdMPS(NDArray* input, NDArray* output, const std::vector<LongType>& axes, bool keepDims) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        if (axes.empty()) {
            // Full reduction
            float prod = 1.0f;
            for (LongType i = 0; i < input->lengthOf(); i++) {
                prod *= inPtr[i];
            }
            outPtr[0] = prod;
        } else {
            // Initialize output to 1.0
            for (LongType i = 0; i < output->lengthOf(); i++) {
                outPtr[i] = 1.0f;
            }

            auto inShape = input->shapeOf();
            int rank = input->rankOf();
            std::set<int> reduceAxes(axes.begin(), axes.end());

            for (LongType i = 0; i < input->lengthOf(); i++) {
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                outPtr[outIdx] *= inPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(reduce_prod, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }

    reduceProdMPS(input, output, axes, keepDims);

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_prod, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_PROD OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reduce Variance
//////////////////////////////////////////////////////////////////////////

static void reduceVarianceMPS(NDArray* input, NDArray* output, const std::vector<LongType>& axes, bool keepDims, bool biasCorrected) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        if (axes.empty()) {
            // Full reduction
            LongType count = input->lengthOf();

            // Compute mean
            float mean = 0.0f;
            for (LongType i = 0; i < count; i++) {
                mean += inPtr[i];
            }
            mean /= count;

            // Compute variance
            float variance = 0.0f;
            for (LongType i = 0; i < count; i++) {
                float diff = inPtr[i] - mean;
                variance += diff * diff;
            }

            if (biasCorrected && count > 1) {
                variance /= (count - 1);
            } else {
                variance /= count;
            }

            outPtr[0] = variance;
        } else {
            // Partial reduction - compute mean first, then variance
            auto inShape = input->shapeOf();
            int rank = input->rankOf();
            std::set<int> reduceAxes(axes.begin(), axes.end());

            LongType reduceSize = 1;
            for (int ax : axes) {
                reduceSize *= inShape[ax];
            }

            // Allocate temporary for means
            std::vector<float> means(output->lengthOf(), 0.0f);

            // Compute means
            for (LongType i = 0; i < input->lengthOf(); i++) {
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                means[outIdx] += inPtr[i];
            }

            for (LongType i = 0; i < output->lengthOf(); i++) {
                means[i] /= reduceSize;
            }

            // Compute variance
            output->nullify();
            for (LongType i = 0; i < input->lengthOf(); i++) {
                std::vector<LongType> coords(rank);
                LongType remaining = i;
                for (int d = rank - 1; d >= 0; d--) {
                    coords[d] = remaining % inShape[d];
                    remaining /= inShape[d];
                }

                LongType outIdx = 0;
                LongType outMult = 1;
                for (int d = rank - 1; d >= 0; d--) {
                    if (reduceAxes.find(d) == reduceAxes.end()) {
                        outIdx += coords[d] * outMult;
                        outMult *= inShape[d];
                    }
                }

                float diff = inPtr[i] - means[outIdx];
                outPtr[outIdx] += diff * diff;
            }

            // Divide by count
            for (LongType i = 0; i < output->lengthOf(); i++) {
                if (biasCorrected && reduceSize > 1) {
                    outPtr[i] /= (reduceSize - 1);
                } else {
                    outPtr[i] /= reduceSize;
                }
            }
        }
    }
}

PLATFORM_IMPL(reduce_variance, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    bool biasCorrected = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }
    if (block.getBArguments()->size() > 1) {
        biasCorrected = B_ARG(1);
    }

    reduceVarianceMPS(input, output, axes, keepDims, biasCorrected);

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_variance, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_VARIANCE OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reduce Standard Deviation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reduce_stdev, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> axes;
    if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    bool keepDims = false;
    bool biasCorrected = false;
    if (block.getBArguments()->size() > 0) {
        keepDims = B_ARG(0);
    }
    if (block.getBArguments()->size() > 1) {
        biasCorrected = B_ARG(1);
    }

    // Compute variance first
    reduceVarianceMPS(input, output, axes, keepDims, biasCorrected);

    // Take square root
    float* outPtr = output->bufferAsT<float>();
    for (LongType i = 0; i < output->lengthOf(); i++) {
        outPtr[i] = sqrtf(outPtr[i]);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(reduce_stdev, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REDUCE_STDEV OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

#else  // !HAVE_MPS — register no-op stubs

PLATFORM_IMPL(reduce_sum, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(reduce_sum, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(reduce_mean, ENGINE_CPU)     { return sd::Status::OK; }
PLATFORM_CHECK(reduce_mean, ENGINE_CPU)    { return false; }

PLATFORM_IMPL(reduce_max, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(reduce_max, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(reduce_min, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(reduce_min, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(reduce_prod, ENGINE_CPU)     { return sd::Status::OK; }
PLATFORM_CHECK(reduce_prod, ENGINE_CPU)    { return false; }

PLATFORM_IMPL(reduce_variance, ENGINE_CPU) { return sd::Status::OK; }
PLATFORM_CHECK(reduce_variance, ENGINE_CPU){ return false; }

PLATFORM_IMPL(reduce_stdev, ENGINE_CPU)    { return sd::Status::OK; }
PLATFORM_CHECK(reduce_stdev, ENGINE_CPU)   { return false; }

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
