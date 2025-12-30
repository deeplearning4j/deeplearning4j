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
// Metal Performance Shaders - Sorting and unique operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Sort operation along a specified axis
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sort, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        bool descending = block.getIArguments()->size() > 0 ? (INT_ARG(0) != 0) : false;

        // Copy input to output
        memcpy(z->buffer(), x->buffer(), x->lengthOf() * x->sizeOfT());

        float* zPtr = z->bufferAsT<float>();
        LongType length = z->lengthOf();

        // Simple sort for 1D case
        if (z->rankOf() == 1) {
            std::vector<float> temp(zPtr, zPtr + length);
            if (descending) {
                std::sort(temp.begin(), temp.end(), std::greater<float>());
            } else {
                std::sort(temp.begin(), temp.end());
            }
            memcpy(zPtr, temp.data(), length * sizeof(float));
        } else {
            // Sort along last axis
            LongType lastDim = z->sizeAt(-1);
            LongType numRows = length / lastDim;

            for (LongType row = 0; row < numRows; row++) {
                float* rowPtr = zPtr + row * lastDim;
                std::vector<float> temp(rowPtr, rowPtr + lastDim);
                if (descending) {
                    std::sort(temp.begin(), temp.end(), std::greater<float>());
                } else {
                    std::sort(temp.begin(), temp.end());
                }
                memcpy(rowPtr, temp.data(), lastDim * sizeof(float));
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(sort, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SORT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Argsort - returns indices that would sort the array
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(argsort, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);  // Integer output

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        bool descending = block.getIArguments()->size() > 0 ? (INT_ARG(0) != 0) : false;

        const float* xPtr = x->bufferAsT<float>();
        auto zPtr = z->bufferAsT<LongType>();

        LongType length = x->lengthOf();

        if (x->rankOf() == 1) {
            // Create index array
            std::vector<LongType> indices(length);
            for (LongType i = 0; i < length; i++) {
                indices[i] = i;
            }

            // Sort indices based on values
            if (descending) {
                std::sort(indices.begin(), indices.end(),
                    [&xPtr](LongType a, LongType b) { return xPtr[a] > xPtr[b]; });
            } else {
                std::sort(indices.begin(), indices.end(),
                    [&xPtr](LongType a, LongType b) { return xPtr[a] < xPtr[b]; });
            }

            memcpy(zPtr, indices.data(), length * sizeof(LongType));
        } else {
            // Sort along last axis
            LongType lastDim = x->sizeAt(-1);
            LongType numRows = length / lastDim;

            for (LongType row = 0; row < numRows; row++) {
                const float* rowPtr = xPtr + row * lastDim;
                LongType* outRowPtr = zPtr + row * lastDim;

                std::vector<LongType> indices(lastDim);
                for (LongType i = 0; i < lastDim; i++) {
                    indices[i] = i;
                }

                if (descending) {
                    std::sort(indices.begin(), indices.end(),
                        [&rowPtr](LongType a, LongType b) { return rowPtr[a] > rowPtr[b]; });
                } else {
                    std::sort(indices.begin(), indices.end(),
                        [&rowPtr](LongType a, LongType b) { return rowPtr[a] < rowPtr[b]; });
                }

                memcpy(outRowPtr, indices.data(), lastDim * sizeof(LongType));
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(argsort, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ARGSORT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Top-K operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(top_k, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto values = OUTPUT_VARIABLE(0);   // Top K values
    auto indices = OUTPUT_VARIABLE(1);  // Indices of top K values

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int k = INT_ARG(0);
        bool sorted = block.getIArguments()->size() > 1 ? (INT_ARG(1) != 0) : true;

        const float* xPtr = x->bufferAsT<float>();
        float* valPtr = values->bufferAsT<float>();
        auto idxPtr = indices->bufferAsT<LongType>();

        LongType lastDim = x->sizeAt(-1);
        LongType numRows = x->lengthOf() / lastDim;

        for (LongType row = 0; row < numRows; row++) {
            const float* rowPtr = xPtr + row * lastDim;
            float* valRowPtr = valPtr + row * k;
            LongType* idxRowPtr = idxPtr + row * k;

            // Create (value, index) pairs
            std::vector<std::pair<float, LongType>> pairs(lastDim);
            for (LongType i = 0; i < lastDim; i++) {
                pairs[i] = {rowPtr[i], i};
            }

            // Partial sort to get top K
            std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            // If sorted, already done; otherwise could shuffle but typically sorted is true
            for (int i = 0; i < k; i++) {
                valRowPtr[i] = pairs[i].first;
                idxRowPtr[i] = pairs[i].second;
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(top_k, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS TOP_K OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// In Top-K operation - check if values are in top K
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(in_top_k, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);  // [batch, classes]
    auto targets = INPUT_VARIABLE(1);       // [batch]
    auto z = OUTPUT_VARIABLE(0);            // [batch] boolean

    if (predictions->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int k = INT_ARG(0);

        const float* predPtr = predictions->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType batch = predictions->sizeAt(0);
        LongType numClasses = predictions->sizeAt(1);

        for (LongType b = 0; b < batch; b++) {
            const float* rowPtr = predPtr + b * numClasses;
            LongType target = targets->e<LongType>(b);

            // Count how many values are greater than target's value
            float targetVal = rowPtr[target];
            int countGreater = 0;

            for (LongType c = 0; c < numClasses; c++) {
                if (rowPtr[c] > targetVal) {
                    countGreater++;
                }
            }

            // Target is in top K if fewer than K values are greater
            zPtr[b] = (countGreater < k) ? 1.0f : 0.0f;
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(in_top_k, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS IN_TOP_K OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Unique operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(unique, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto values = OUTPUT_VARIABLE(0);   // Unique values
    auto indices = OUTPUT_VARIABLE(1);  // Indices in original

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        LongType length = x->lengthOf();

        // Find unique values while preserving order
        std::vector<float> uniqueVals;
        std::vector<LongType> uniqueIdx;
        std::unordered_set<float> seen;

        for (LongType i = 0; i < length; i++) {
            float val = xPtr[i];
            if (seen.find(val) == seen.end()) {
                seen.insert(val);
                uniqueVals.push_back(val);
                uniqueIdx.push_back(i);
            }
        }

        // Copy to outputs (output shapes should be pre-allocated correctly)
        float* valPtr = values->bufferAsT<float>();
        auto idxPtr = indices->bufferAsT<LongType>();

        LongType numUnique = std::min((LongType)uniqueVals.size(), values->lengthOf());
        for (LongType i = 0; i < numUnique; i++) {
            valPtr[i] = uniqueVals[i];
            if (i < indices->lengthOf()) {
                idxPtr[i] = uniqueIdx[i];
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(unique, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS UNIQUE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Unique with counts
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(unique_with_counts, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto values = OUTPUT_VARIABLE(0);   // Unique values
    auto indices = OUTPUT_VARIABLE(1);  // Indices in original
    auto counts = OUTPUT_VARIABLE(2);   // Counts

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        LongType length = x->lengthOf();

        // Find unique values with counts
        std::vector<float> uniqueVals;
        std::vector<LongType> uniqueIdx;
        std::vector<LongType> uniqueCounts;
        std::unordered_map<float, LongType> valToPos;

        for (LongType i = 0; i < length; i++) {
            float val = xPtr[i];
            auto it = valToPos.find(val);
            if (it == valToPos.end()) {
                valToPos[val] = uniqueVals.size();
                uniqueVals.push_back(val);
                uniqueIdx.push_back(i);
                uniqueCounts.push_back(1);
            } else {
                uniqueCounts[it->second]++;
            }
        }

        // Copy to outputs
        float* valPtr = values->bufferAsT<float>();
        auto idxPtr = indices->bufferAsT<LongType>();
        auto cntPtr = counts->bufferAsT<LongType>();

        LongType numUnique = std::min((LongType)uniqueVals.size(), values->lengthOf());
        for (LongType i = 0; i < numUnique; i++) {
            valPtr[i] = uniqueVals[i];
            if (i < indices->lengthOf()) {
                idxPtr[i] = uniqueIdx[i];
            }
            if (i < counts->lengthOf()) {
                cntPtr[i] = uniqueCounts[i];
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(unique_with_counts, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS UNIQUE_WITH_COUNTS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Argmax operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(argmax, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        auto dimensions = block.getIArguments();

        const float* xPtr = x->bufferAsT<float>();
        auto zPtr = z->bufferAsT<LongType>();

        if (dimensions->empty() || x->rankOf() == 1) {
            // Global argmax
            LongType maxIdx = 0;
            float maxVal = xPtr[0];
            for (LongType i = 1; i < x->lengthOf(); i++) {
                if (xPtr[i] > maxVal) {
                    maxVal = xPtr[i];
                    maxIdx = i;
                }
            }
            zPtr[0] = maxIdx;
        } else {
            // Argmax along last axis (simplified)
            LongType lastDim = x->sizeAt(-1);
            LongType numRows = x->lengthOf() / lastDim;

            for (LongType row = 0; row < numRows; row++) {
                const float* rowPtr = xPtr + row * lastDim;
                LongType maxIdx = 0;
                float maxVal = rowPtr[0];

                for (LongType i = 1; i < lastDim; i++) {
                    if (rowPtr[i] > maxVal) {
                        maxVal = rowPtr[i];
                        maxIdx = i;
                    }
                }
                zPtr[row] = maxIdx;
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(argmax, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ARGMAX OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Argmin operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(argmin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        auto dimensions = block.getIArguments();

        const float* xPtr = x->bufferAsT<float>();
        auto zPtr = z->bufferAsT<LongType>();

        if (dimensions->empty() || x->rankOf() == 1) {
            // Global argmin
            LongType minIdx = 0;
            float minVal = xPtr[0];
            for (LongType i = 1; i < x->lengthOf(); i++) {
                if (xPtr[i] < minVal) {
                    minVal = xPtr[i];
                    minIdx = i;
                }
            }
            zPtr[0] = minIdx;
        } else {
            // Argmin along last axis (simplified)
            LongType lastDim = x->sizeAt(-1);
            LongType numRows = x->lengthOf() / lastDim;

            for (LongType row = 0; row < numRows; row++) {
                const float* rowPtr = xPtr + row * lastDim;
                LongType minIdx = 0;
                float minVal = rowPtr[0];

                for (LongType i = 1; i < lastDim; i++) {
                    if (rowPtr[i] < minVal) {
                        minVal = rowPtr[i];
                        minIdx = i;
                    }
                }
                zPtr[row] = minIdx;
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(argmin, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ARGMIN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Histogram operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(histogram, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int numBins = INT_ARG(0);

        const float* xPtr = x->bufferAsT<float>();
        auto zPtr = z->bufferAsT<LongType>();
        LongType length = x->lengthOf();

        // Find min and max
        float minVal = xPtr[0], maxVal = xPtr[0];
        for (LongType i = 1; i < length; i++) {
            if (xPtr[i] < minVal) minVal = xPtr[i];
            if (xPtr[i] > maxVal) maxVal = xPtr[i];
        }

        // Initialize bins to zero
        for (int i = 0; i < numBins; i++) {
            zPtr[i] = 0;
        }

        // Compute histogram
        float binWidth = (maxVal - minVal) / numBins;
        if (binWidth == 0.0f) binWidth = 1.0f;

        for (LongType i = 0; i < length; i++) {
            int bin = (int)((xPtr[i] - minVal) / binWidth);
            if (bin >= numBins) bin = numBins - 1;
            if (bin < 0) bin = 0;
            zPtr[bin]++;
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(histogram, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS HISTOGRAM OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Bincount operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(bincount, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);  // Integer array
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        LongType minLength = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
        LongType maxLength = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

        auto zPtr = z->bufferAsT<LongType>();
        LongType zLength = z->lengthOf();

        // Initialize to zero
        memset(zPtr, 0, zLength * sizeof(LongType));

        // Count occurrences
        for (LongType i = 0; i < x->lengthOf(); i++) {
            LongType val = x->e<LongType>(i);
            if (val >= 0 && val < zLength) {
                zPtr[val]++;
            }
        }

        // Apply weights if provided
        if (block.width() > 1) {
            auto weights = INPUT_VARIABLE(1);
            const float* wPtr = weights->bufferAsT<float>();

            // Re-compute with weights
            float* zFloatPtr = z->bufferAsT<float>();
            memset(zFloatPtr, 0, zLength * sizeof(float));

            for (LongType i = 0; i < x->lengthOf(); i++) {
                LongType val = x->e<LongType>(i);
                if (val >= 0 && val < zLength) {
                    zFloatPtr[val] += wPtr[i];
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(bincount, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS BINCOUNT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
