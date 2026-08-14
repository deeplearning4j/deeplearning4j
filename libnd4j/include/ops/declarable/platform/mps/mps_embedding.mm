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
// Metal Performance Shaders - Embedding and lookup operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cmath>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Embedding lookup operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(embedding_lookup, ENGINE_MPS) {
    auto embeddings = INPUT_VARIABLE(0);  // [numEmbeddings, embeddingDim]
    auto indices = INPUT_VARIABLE(1);      // [*] - any shape of indices
    auto z = OUTPUT_VARIABLE(0);           // [*indices.shape, embeddingDim]

    if (embeddings->isEmpty() || indices->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* embPtr = embeddings->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType embeddingDim = embeddings->sizeAt(-1);
        LongType numLookups = indices->lengthOf();

        // Handle different index types
        for (LongType i = 0; i < numLookups; i++) {
            LongType idx = indices->e<LongType>(i);

            // Bounds check
            if (idx >= 0 && idx < embeddings->sizeAt(0)) {
                const float* srcPtr = embPtr + idx * embeddingDim;
                float* dstPtr = zPtr + i * embeddingDim;

                memcpy(dstPtr, srcPtr, embeddingDim * sizeof(float));
            } else {
                // Zero out invalid indices
                float* dstPtr = zPtr + i * embeddingDim;
                memset(dstPtr, 0, embeddingDim * sizeof(float));
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(embedding_lookup, ENGINE_MPS) {
    auto embeddings = INPUT_VARIABLE(0);

    Requirements req("MPS EMBEDDING_LOOKUP OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(embeddings->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*embeddings), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// One-hot encoding operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(onehot, ENGINE_MPS) {
    auto indices = INPUT_VARIABLE(0);  // [*] - indices
    auto z = OUTPUT_VARIABLE(0);        // [*, depth] or [depth, *]

    if (indices->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int depth = INT_ARG(0);
        int axis = block.getIArguments()->size() > 1 ? INT_ARG(1) : -1;
        float onValue = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;
        float offValue = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.0f;

        float* zPtr = z->bufferAsT<float>();
        LongType numIndices = indices->lengthOf();

        // Initialize with off values
        for (LongType i = 0; i < z->lengthOf(); i++) {
            zPtr[i] = offValue;
        }

        // Set on values
        if (axis == -1 || axis == indices->rankOf()) {
            // Last axis: [*, depth]
            for (LongType i = 0; i < numIndices; i++) {
                LongType idx = indices->e<LongType>(i);
                if (idx >= 0 && idx < depth) {
                    zPtr[i * depth + idx] = onValue;
                }
            }
        } else if (axis == 0) {
            // First axis: [depth, *]
            for (LongType i = 0; i < numIndices; i++) {
                LongType idx = indices->e<LongType>(i);
                if (idx >= 0 && idx < depth) {
                    zPtr[idx * numIndices + i] = onValue;
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(onehot, ENGINE_MPS) {
    auto indices = INPUT_VARIABLE(0);

    Requirements req("MPS ONEHOT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Segment sum operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(segment_sum, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);      // [N, ...]
    auto segmentIds = INPUT_VARIABLE(1); // [N]
    auto z = OUTPUT_VARIABLE(0);         // [numSegments, ...]

    if (data->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* dataPtr = data->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numElements = data->sizeAt(0);
        LongType numSegments = z->sizeAt(0);
        LongType innerSize = data->lengthOf() / numElements;

        // Initialize output to zero
        memset(zPtr, 0, z->lengthOf() * sizeof(float));

        // Accumulate sums
        for (LongType i = 0; i < numElements; i++) {
            LongType segId = segmentIds->e<LongType>(i);
            if (segId >= 0 && segId < numSegments) {
                for (LongType j = 0; j < innerSize; j++) {
                    zPtr[segId * innerSize + j] += dataPtr[i * innerSize + j];
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(segment_sum, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);

    Requirements req("MPS SEGMENT_SUM OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(data->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*data), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Segment mean operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(segment_mean, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (data->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* dataPtr = data->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numElements = data->sizeAt(0);
        LongType numSegments = z->sizeAt(0);
        LongType innerSize = data->lengthOf() / numElements;

        // Initialize output and count
        memset(zPtr, 0, z->lengthOf() * sizeof(float));
        std::vector<LongType> counts(numSegments, 0);

        // Accumulate sums and counts
        for (LongType i = 0; i < numElements; i++) {
            LongType segId = segmentIds->e<LongType>(i);
            if (segId >= 0 && segId < numSegments) {
                counts[segId]++;
                for (LongType j = 0; j < innerSize; j++) {
                    zPtr[segId * innerSize + j] += dataPtr[i * innerSize + j];
                }
            }
        }

        // Divide by counts to get means
        for (LongType seg = 0; seg < numSegments; seg++) {
            if (counts[seg] > 0) {
                float invCount = 1.0f / counts[seg];
                for (LongType j = 0; j < innerSize; j++) {
                    zPtr[seg * innerSize + j] *= invCount;
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(segment_mean, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);

    Requirements req("MPS SEGMENT_MEAN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(data->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*data), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Segment max operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(segment_max, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (data->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* dataPtr = data->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numElements = data->sizeAt(0);
        LongType numSegments = z->sizeAt(0);
        LongType innerSize = data->lengthOf() / numElements;

        // Initialize output to negative infinity
        for (LongType i = 0; i < z->lengthOf(); i++) {
            zPtr[i] = -std::numeric_limits<float>::infinity();
        }

        // Find max values
        for (LongType i = 0; i < numElements; i++) {
            LongType segId = segmentIds->e<LongType>(i);
            if (segId >= 0 && segId < numSegments) {
                for (LongType j = 0; j < innerSize; j++) {
                    float val = dataPtr[i * innerSize + j];
                    if (val > zPtr[segId * innerSize + j]) {
                        zPtr[segId * innerSize + j] = val;
                    }
                }
            }
        }

        // Replace -inf with 0 for empty segments
        for (LongType i = 0; i < z->lengthOf(); i++) {
            if (std::isinf(zPtr[i]) && zPtr[i] < 0) {
                zPtr[i] = 0.0f;
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(segment_max, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);

    Requirements req("MPS SEGMENT_MAX OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(data->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*data), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Segment min operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(segment_min, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (data->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* dataPtr = data->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numElements = data->sizeAt(0);
        LongType numSegments = z->sizeAt(0);
        LongType innerSize = data->lengthOf() / numElements;

        // Initialize output to positive infinity
        for (LongType i = 0; i < z->lengthOf(); i++) {
            zPtr[i] = std::numeric_limits<float>::infinity();
        }

        // Find min values
        for (LongType i = 0; i < numElements; i++) {
            LongType segId = segmentIds->e<LongType>(i);
            if (segId >= 0 && segId < numSegments) {
                for (LongType j = 0; j < innerSize; j++) {
                    float val = dataPtr[i * innerSize + j];
                    if (val < zPtr[segId * innerSize + j]) {
                        zPtr[segId * innerSize + j] = val;
                    }
                }
            }
        }

        // Replace inf with 0 for empty segments
        for (LongType i = 0; i < z->lengthOf(); i++) {
            if (std::isinf(zPtr[i]) && zPtr[i] > 0) {
                zPtr[i] = 0.0f;
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(segment_min, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);

    Requirements req("MPS SEGMENT_MIN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(data->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*data), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Segment prod operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(segment_prod, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (data->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* dataPtr = data->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numElements = data->sizeAt(0);
        LongType numSegments = z->sizeAt(0);
        LongType innerSize = data->lengthOf() / numElements;

        // Initialize output to 1
        for (LongType i = 0; i < z->lengthOf(); i++) {
            zPtr[i] = 1.0f;
        }

        // Accumulate products
        for (LongType i = 0; i < numElements; i++) {
            LongType segId = segmentIds->e<LongType>(i);
            if (segId >= 0 && segId < numSegments) {
                for (LongType j = 0; j < innerSize; j++) {
                    zPtr[segId * innerSize + j] *= dataPtr[i * innerSize + j];
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(segment_prod, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);

    Requirements req("MPS SEGMENT_PROD OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(data->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*data), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Unsorted segment sum operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(unsorted_segment_sum, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);
    auto segmentIds = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (data->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* dataPtr = data->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numElements = data->sizeAt(0);
        LongType numSegments = INT_ARG(0);
        LongType innerSize = data->lengthOf() / numElements;

        // Initialize output to zero
        memset(zPtr, 0, z->lengthOf() * sizeof(float));

        // Accumulate sums (segments don't need to be sorted)
        for (LongType i = 0; i < numElements; i++) {
            LongType segId = segmentIds->e<LongType>(i);
            if (segId >= 0 && segId < numSegments) {
                for (LongType j = 0; j < innerSize; j++) {
                    zPtr[segId * innerSize + j] += dataPtr[i * innerSize + j];
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(unsorted_segment_sum, ENGINE_MPS) {
    auto data = INPUT_VARIABLE(0);

    Requirements req("MPS UNSORTED_SEGMENT_SUM OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(data->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*data), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

#else  // !HAVE_MPS

PLATFORM_IMPL(embedding_lookup, ENGINE_MPS)    { return sd::Status::OK; }
PLATFORM_CHECK(embedding_lookup, ENGINE_MPS)   { return Requirements("MPS EMBEDDING_LOOKUP STUB"); }

PLATFORM_IMPL(onehot, ENGINE_MPS)              { return sd::Status::OK; }
PLATFORM_CHECK(onehot, ENGINE_MPS)             { return Requirements("MPS ONEHOT STUB"); }

PLATFORM_IMPL(segment_sum, ENGINE_MPS)         { return sd::Status::OK; }
PLATFORM_CHECK(segment_sum, ENGINE_MPS)        { return Requirements("MPS SEGMENT_SUM STUB"); }

PLATFORM_IMPL(segment_mean, ENGINE_MPS)        { return sd::Status::OK; }
PLATFORM_CHECK(segment_mean, ENGINE_MPS)       { return Requirements("MPS SEGMENT_MEAN STUB"); }

PLATFORM_IMPL(segment_max, ENGINE_MPS)         { return sd::Status::OK; }
PLATFORM_CHECK(segment_max, ENGINE_MPS)        { return Requirements("MPS SEGMENT_MAX STUB"); }

PLATFORM_IMPL(segment_min, ENGINE_MPS)         { return sd::Status::OK; }
PLATFORM_CHECK(segment_min, ENGINE_MPS)        { return Requirements("MPS SEGMENT_MIN STUB"); }

PLATFORM_IMPL(segment_prod, ENGINE_MPS)        { return sd::Status::OK; }
PLATFORM_CHECK(segment_prod, ENGINE_MPS)       { return Requirements("MPS SEGMENT_PROD STUB"); }

PLATFORM_IMPL(unsorted_segment_sum, ENGINE_MPS)  { return sd::Status::OK; }
PLATFORM_CHECK(unsorted_segment_sum, ENGINE_MPS) { return Requirements("MPS UNSORTED_SEGMENT_SUM STUB"); }

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
