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
// Metal Performance Shaders - Transform operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cmath>
#include <algorithm>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Reshape operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reshape, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    // Reshape is a view operation - just copy data if needed
    if (x->buffer() != z->buffer()) {
        memcpy(z->buffer(), x->buffer(), x->lengthOf() * x->sizeOfT());
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(reshape, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS RESHAPE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Flatten operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(flatten, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    // Flatten is essentially a reshape to 1D
    if (x->buffer() != z->buffer()) {
        memcpy(z->buffer(), x->buffer(), x->lengthOf() * x->sizeOfT());
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(flatten, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS FLATTEN OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Squeeze operation - remove dimensions of size 1
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(squeeze, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    if (x->buffer() != z->buffer()) {
        memcpy(z->buffer(), x->buffer(), x->lengthOf() * x->sizeOfT());
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(squeeze, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SQUEEZE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Unsqueeze/expand_dims operation - add dimension of size 1
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(expand_dims, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    if (x->buffer() != z->buffer()) {
        memcpy(z->buffer(), x->buffer(), x->lengthOf() * x->sizeOfT());
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(expand_dims, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS EXPAND_DIMS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Permute/transpose operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(permute, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        auto permutationAxes = block.getIArguments();
        int rank = x->rankOf();

        if (permutationAxes->empty()) {
            // Default: reverse all dimensions
            for (int i = 0; i < rank; i++) {
                permutationAxes->push_back(rank - 1 - i);
            }
        }

        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        // Calculate strides for input
        std::vector<LongType> xStrides(rank);
        std::vector<LongType> zStrides(rank);

        LongType xStride = 1, zStride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            xStrides[i] = xStride;
            xStride *= x->sizeAt(i);
            zStrides[i] = zStride;
            zStride *= z->sizeAt(i);
        }

        // Permute using index mapping
        LongType length = z->lengthOf();
        for (LongType i = 0; i < length; i++) {
            // Convert linear index to z coordinates
            std::vector<LongType> zCoords(rank);
            LongType remaining = i;
            for (int d = 0; d < rank; d++) {
                zCoords[d] = remaining / zStrides[d];
                remaining = remaining % zStrides[d];
            }

            // Map z coordinates to x coordinates using permutation
            LongType xIdx = 0;
            for (int d = 0; d < rank; d++) {
                xIdx += zCoords[d] * xStrides[(*permutationAxes)[d]];
            }

            zPtr[i] = xPtr[xIdx];
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(permute, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS PERMUTE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Split operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(split, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    int numOutputs = block.outputWidth();
    int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        if (axis < 0) axis += x->rankOf();

        const float* xPtr = x->bufferAsT<float>();
        LongType splitSize = x->sizeAt(axis) / numOutputs;

        LongType outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= x->sizeAt(i);

        LongType innerSize = 1;
        for (int i = axis + 1; i < x->rankOf(); i++) innerSize *= x->sizeAt(i);

        for (int out = 0; out < numOutputs; out++) {
            auto z = OUTPUT_VARIABLE(out);
            float* zPtr = z->bufferAsT<float>();

            for (LongType outer = 0; outer < outerSize; outer++) {
                for (LongType split = 0; split < splitSize; split++) {
                    for (LongType inner = 0; inner < innerSize; inner++) {
                        LongType xIdx = outer * x->sizeAt(axis) * innerSize +
                                       (out * splitSize + split) * innerSize + inner;
                        LongType zIdx = outer * splitSize * innerSize + split * innerSize + inner;
                        zPtr[zIdx] = xPtr[xIdx];
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(split, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SPLIT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Space to depth operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(space_to_depth, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);  // [batch, height, width, channels] or [batch, channels, height, width]
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int blockSize = INT_ARG(0);
        bool isNHWC = block.getIArguments()->size() > 1 ? (INT_ARG(1) == 1) : true;

        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType batch, inH, inW, inC;
        LongType outH, outW, outC;

        if (isNHWC) {
            batch = x->sizeAt(0);
            inH = x->sizeAt(1);
            inW = x->sizeAt(2);
            inC = x->sizeAt(3);
            outH = inH / blockSize;
            outW = inW / blockSize;
            outC = inC * blockSize * blockSize;
        } else {
            batch = x->sizeAt(0);
            inC = x->sizeAt(1);
            inH = x->sizeAt(2);
            inW = x->sizeAt(3);
            outH = inH / blockSize;
            outW = inW / blockSize;
            outC = inC * blockSize * blockSize;
        }

        for (LongType b = 0; b < batch; b++) {
            for (LongType h = 0; h < outH; h++) {
                for (LongType w = 0; w < outW; w++) {
                    for (LongType c = 0; c < inC; c++) {
                        for (LongType bh = 0; bh < blockSize; bh++) {
                            for (LongType bw = 0; bw < blockSize; bw++) {
                                LongType inIdx, outIdx;
                                LongType outCIdx = c * blockSize * blockSize + bh * blockSize + bw;

                                if (isNHWC) {
                                    inIdx = b * inH * inW * inC +
                                           (h * blockSize + bh) * inW * inC +
                                           (w * blockSize + bw) * inC + c;
                                    outIdx = b * outH * outW * outC +
                                            h * outW * outC + w * outC + outCIdx;
                                } else {
                                    inIdx = b * inC * inH * inW +
                                           c * inH * inW +
                                           (h * blockSize + bh) * inW +
                                           (w * blockSize + bw);
                                    outIdx = b * outC * outH * outW +
                                            outCIdx * outH * outW +
                                            h * outW + w;
                                }

                                zPtr[outIdx] = xPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(space_to_depth, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SPACE_TO_DEPTH OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Depth to space operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(depth_to_space, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int blockSize = INT_ARG(0);
        bool isNHWC = block.getIArguments()->size() > 1 ? (INT_ARG(1) == 1) : true;

        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType batch, inH, inW, inC;
        LongType outH, outW, outC;

        if (isNHWC) {
            batch = x->sizeAt(0);
            inH = x->sizeAt(1);
            inW = x->sizeAt(2);
            inC = x->sizeAt(3);
            outH = inH * blockSize;
            outW = inW * blockSize;
            outC = inC / (blockSize * blockSize);
        } else {
            batch = x->sizeAt(0);
            inC = x->sizeAt(1);
            inH = x->sizeAt(2);
            inW = x->sizeAt(3);
            outH = inH * blockSize;
            outW = inW * blockSize;
            outC = inC / (blockSize * blockSize);
        }

        for (LongType b = 0; b < batch; b++) {
            for (LongType h = 0; h < inH; h++) {
                for (LongType w = 0; w < inW; w++) {
                    for (LongType c = 0; c < outC; c++) {
                        for (LongType bh = 0; bh < blockSize; bh++) {
                            for (LongType bw = 0; bw < blockSize; bw++) {
                                LongType inIdx, outIdx;
                                LongType inCIdx = c * blockSize * blockSize + bh * blockSize + bw;

                                if (isNHWC) {
                                    inIdx = b * inH * inW * inC +
                                           h * inW * inC + w * inC + inCIdx;
                                    outIdx = b * outH * outW * outC +
                                            (h * blockSize + bh) * outW * outC +
                                            (w * blockSize + bw) * outC + c;
                                } else {
                                    inIdx = b * inC * inH * inW +
                                           inCIdx * inH * inW +
                                           h * inW + w;
                                    outIdx = b * outC * outH * outW +
                                            c * outH * outW +
                                            (h * blockSize + bh) * outW +
                                            (w * blockSize + bw);
                                }

                                zPtr[outIdx] = xPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(depth_to_space, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS DEPTH_TO_SPACE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Batch to space operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(batch_to_space_nd, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto blockShape = INPUT_VARIABLE(1);
    auto crops = INPUT_VARIABLE(2);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        // Simplified implementation for 2D spatial dimensions
        LongType batch = x->sizeAt(0);
        LongType height = x->sizeAt(1);
        LongType width = x->sizeAt(2);
        LongType channels = x->sizeAt(3);

        LongType blockH = blockShape->e<LongType>(0);
        LongType blockW = blockShape->e<LongType>(1);

        LongType cropTop = crops->e<LongType>(0, 0);
        LongType cropBottom = crops->e<LongType>(0, 1);
        LongType cropLeft = crops->e<LongType>(1, 0);
        LongType cropRight = crops->e<LongType>(1, 1);

        LongType outBatch = batch / (blockH * blockW);
        LongType outH = height * blockH - cropTop - cropBottom;
        LongType outW = width * blockW - cropLeft - cropRight;

        // Initialize output to zero
        memset(zPtr, 0, z->lengthOf() * sizeof(float));

        for (LongType b = 0; b < batch; b++) {
            LongType outB = b % outBatch;
            LongType blockIdx = b / outBatch;
            LongType blockHIdx = blockIdx / blockW;
            LongType blockWIdx = blockIdx % blockW;

            for (LongType h = 0; h < height; h++) {
                for (LongType w = 0; w < width; w++) {
                    LongType outHRaw = h * blockH + blockHIdx;
                    LongType outWRaw = w * blockW + blockWIdx;

                    if (outHRaw >= cropTop && outHRaw < height * blockH - cropBottom &&
                        outWRaw >= cropLeft && outWRaw < width * blockW - cropRight) {

                        LongType finalH = outHRaw - cropTop;
                        LongType finalW = outWRaw - cropLeft;

                        for (LongType c = 0; c < channels; c++) {
                            LongType inIdx = b * height * width * channels +
                                            h * width * channels + w * channels + c;
                            LongType outIdx = outB * outH * outW * channels +
                                             finalH * outW * channels + finalW * channels + c;

                            zPtr[outIdx] = xPtr[inIdx];
                        }
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(batch_to_space_nd, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS BATCH_TO_SPACE_ND OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Space to batch operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(space_to_batch_nd, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto blockShape = INPUT_VARIABLE(1);
    auto paddings = INPUT_VARIABLE(2);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType batch = x->sizeAt(0);
        LongType height = x->sizeAt(1);
        LongType width = x->sizeAt(2);
        LongType channels = x->sizeAt(3);

        LongType blockH = blockShape->e<LongType>(0);
        LongType blockW = blockShape->e<LongType>(1);

        LongType padTop = paddings->e<LongType>(0, 0);
        LongType padBottom = paddings->e<LongType>(0, 1);
        LongType padLeft = paddings->e<LongType>(1, 0);
        LongType padRight = paddings->e<LongType>(1, 1);

        LongType paddedH = height + padTop + padBottom;
        LongType paddedW = width + padLeft + padRight;

        LongType outBatch = batch * blockH * blockW;
        LongType outH = paddedH / blockH;
        LongType outW = paddedW / blockW;

        // Initialize output to zero (for padding regions)
        memset(zPtr, 0, z->lengthOf() * sizeof(float));

        for (LongType b = 0; b < batch; b++) {
            for (LongType bh = 0; bh < blockH; bh++) {
                for (LongType bw = 0; bw < blockW; bw++) {
                    LongType outB = b * blockH * blockW + bh * blockW + bw;

                    for (LongType h = 0; h < outH; h++) {
                        for (LongType w = 0; w < outW; w++) {
                            LongType inH = h * blockH + bh - padTop;
                            LongType inW = w * blockW + bw - padLeft;

                            if (inH >= 0 && inH < height && inW >= 0 && inW < width) {
                                for (LongType c = 0; c < channels; c++) {
                                    LongType inIdx = b * height * width * channels +
                                                    inH * width * channels + inW * channels + c;
                                    LongType outIdx = outB * outH * outW * channels +
                                                     h * outW * channels + w * channels + c;

                                    zPtr[outIdx] = xPtr[inIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(space_to_batch_nd, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SPACE_TO_BATCH_ND OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
