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
// Metal Performance Shaders - Matrix and tensor manipulation operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Transpose
//////////////////////////////////////////////////////////////////////////

static void transposeMPS(NDArray* input, NDArray* output, const std::vector<LongType>& permutation) {
    @autoreleasepool {
        int rank = input->rankOf();
        auto inShape = input->shapeOf();
        auto inStrides = input->stridesOf();

        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        // Compute output strides based on permutation
        std::vector<LongType> outShape(rank);
        for (int i = 0; i < rank; i++) {
            outShape[i] = inShape[permutation[i]];
        }

        // Iterate through all output elements
        for (LongType outIdx = 0; outIdx < output->lengthOf(); outIdx++) {
            // Convert linear index to output coordinates
            std::vector<LongType> outCoords(rank);
            LongType remaining = outIdx;
            for (int d = rank - 1; d >= 0; d--) {
                outCoords[d] = remaining % outShape[d];
                remaining /= outShape[d];
            }

            // Convert output coordinates to input coordinates using inverse permutation
            std::vector<LongType> inCoords(rank);
            for (int d = 0; d < rank; d++) {
                inCoords[permutation[d]] = outCoords[d];
            }

            // Compute input linear index
            LongType inIdx = 0;
            for (int d = 0; d < rank; d++) {
                inIdx += inCoords[d] * inStrides[d];
            }

            outPtr[outIdx] = inPtr[inIdx];
        }
    }
}

PLATFORM_IMPL(transpose, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    // DSP view path: output shares input's buffer with transposed strides.
    if (input->dataBuffer() == output->dataBuffer()) return sd::Status::OK;

    std::vector<LongType> permutation;
    if (block.width() > 1) {
        auto permArr = INPUT_VARIABLE(1);
        for (LongType i = 0; i < permArr->lengthOf(); i++) {
            permutation.push_back(permArr->e<LongType>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            permutation.push_back(INT_ARG(i));
        }
    } else {
        // Default: reverse dimensions
        for (int i = input->rankOf() - 1; i >= 0; i--) {
            permutation.push_back(i);
        }
    }

    transposeMPS(input, output, permutation);

    return sd::Status::OK;
}

PLATFORM_CHECK(transpose, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS TRANSPOSE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Concat
//////////////////////////////////////////////////////////////////////////

static void concatMPS(const std::vector<NDArray*>& inputs, NDArray* output, int axis) {
    @autoreleasepool {
        if (axis < 0) {
            axis += output->rankOf();
        }

        float* outPtr = output->bufferAsT<float>();
        auto outShape = output->shapeOf();
        int rank = output->rankOf();

        LongType outerSize = 1;
        for (int i = 0; i < axis; i++) {
            outerSize *= outShape[i];
        }

        LongType innerSize = 1;
        for (int i = axis + 1; i < rank; i++) {
            innerSize *= outShape[i];
        }

        LongType outOffset = 0;
        for (size_t n = 0; n < inputs.size(); n++) {
            NDArray* input = inputs[n];
            const float* inPtr = input->bufferAsT<float>();
            LongType axisSize = input->sizeAt(axis);

            for (LongType outer = 0; outer < outerSize; outer++) {
                for (LongType a = 0; a < axisSize; a++) {
                    for (LongType inner = 0; inner < innerSize; inner++) {
                        LongType inIdx = outer * axisSize * innerSize + a * innerSize + inner;
                        LongType outIdx = outer * outShape[axis] * innerSize + (outOffset + a) * innerSize + inner;
                        outPtr[outIdx] = inPtr[inIdx];
                    }
                }
            }
            outOffset += axisSize;
        }
    }
}

PLATFORM_IMPL(concat, ENGINE_MPS) {
    auto output = OUTPUT_VARIABLE(0);

    if (block.width() == 0) return sd::Status::OK;

    int axis = 0;
    if (block.getIArguments()->size() > 0) {
        axis = INT_ARG(0);
    }

    std::vector<NDArray*> inputs;
    for (int i = 0; i < block.width(); i++) {
        auto inp = INPUT_VARIABLE(i);
        if (!inp->isEmpty()) {
            inputs.push_back(inp);
        }
    }

    if (inputs.empty()) return sd::Status::OK;

    concatMPS(inputs, output, axis);

    return sd::Status::OK;
}

PLATFORM_CHECK(concat, ENGINE_MPS) {
    Requirements req("MPS CONCAT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");

    if (block.width() > 0) {
        auto input = INPUT_VARIABLE(0);
        req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    }

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Stack
//////////////////////////////////////////////////////////////////////////

static void stackMPS(const std::vector<NDArray*>& inputs, NDArray* output, int axis) {
    @autoreleasepool {
        if (axis < 0) {
            axis += output->rankOf();
        }

        float* outPtr = output->bufferAsT<float>();
        auto outShape = output->shapeOf();
        int outRank = output->rankOf();

        LongType numInputs = inputs.size();
        LongType inputLen = inputs[0]->lengthOf();

        // Stack along new axis
        for (LongType n = 0; n < numInputs; n++) {
            const float* inPtr = inputs[n]->bufferAsT<float>();

            for (LongType i = 0; i < inputLen; i++) {
                // Compute output index
                LongType outIdx;
                if (axis == 0) {
                    outIdx = n * inputLen + i;
                } else {
                    // More complex indexing for other axes
                    int inRank = inputs[0]->rankOf();
                    auto inShape = inputs[0]->shapeOf();

                    std::vector<LongType> inCoords(inRank);
                    LongType remaining = i;
                    for (int d = inRank - 1; d >= 0; d--) {
                        inCoords[d] = remaining % inShape[d];
                        remaining /= inShape[d];
                    }

                    std::vector<LongType> outCoords(outRank);
                    int inD = 0;
                    for (int d = 0; d < outRank; d++) {
                        if (d == axis) {
                            outCoords[d] = n;
                        } else {
                            outCoords[d] = inCoords[inD++];
                        }
                    }

                    outIdx = 0;
                    LongType mult = 1;
                    for (int d = outRank - 1; d >= 0; d--) {
                        outIdx += outCoords[d] * mult;
                        mult *= outShape[d];
                    }
                }

                outPtr[outIdx] = inPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(stack, ENGINE_MPS) {
    auto output = OUTPUT_VARIABLE(0);

    if (block.width() == 0) return sd::Status::OK;

    int axis = 0;
    if (block.getIArguments()->size() > 0) {
        axis = INT_ARG(0);
    }

    std::vector<NDArray*> inputs;
    for (int i = 0; i < block.width(); i++) {
        inputs.push_back(INPUT_VARIABLE(i));
    }

    stackMPS(inputs, output, axis);

    return sd::Status::OK;
}

PLATFORM_CHECK(stack, ENGINE_MPS) {
    Requirements req("MPS STACK OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");

    if (block.width() > 0) {
        auto input = INPUT_VARIABLE(0);
        req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    }

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Gather
//////////////////////////////////////////////////////////////////////////

static void gatherMPS(NDArray* input, NDArray* indices, NDArray* output, int axis) {
    @autoreleasepool {
        if (axis < 0) {
            axis += input->rankOf();
        }

        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        auto inShape = input->shapeOf();
        int inRank = input->rankOf();

        LongType outerSize = 1;
        for (int i = 0; i < axis; i++) {
            outerSize *= inShape[i];
        }

        LongType axisSize = inShape[axis];

        LongType innerSize = 1;
        for (int i = axis + 1; i < inRank; i++) {
            innerSize *= inShape[i];
        }

        LongType numIndices = indices->lengthOf();
        const int* idxPtr = indices->bufferAsT<int>();

        for (LongType outer = 0; outer < outerSize; outer++) {
            for (LongType idx = 0; idx < numIndices; idx++) {
                int gatherIdx = idxPtr[idx];
                if (gatherIdx < 0) gatherIdx += axisSize;

                for (LongType inner = 0; inner < innerSize; inner++) {
                    LongType inOffset = outer * axisSize * innerSize + gatherIdx * innerSize + inner;
                    LongType outOffset = outer * numIndices * innerSize + idx * innerSize + inner;
                    outPtr[outOffset] = inPtr[inOffset];
                }
            }
        }
    }
}

PLATFORM_IMPL(gather, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);
    auto indices = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int axis = 0;
    if (block.getIArguments()->size() > 0) {
        axis = INT_ARG(0);
    }

    gatherMPS(input, indices, output, axis);

    return sd::Status::OK;
}

PLATFORM_CHECK(gather, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS GATHER OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Tile
//////////////////////////////////////////////////////////////////////////

static void tileMPS(NDArray* input, NDArray* output, const std::vector<LongType>& repeats) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        auto inShape = input->shapeOf();
        auto outShape = output->shapeOf();
        int rank = input->rankOf();

        for (LongType outIdx = 0; outIdx < output->lengthOf(); outIdx++) {
            // Convert output index to coordinates
            std::vector<LongType> outCoords(rank);
            LongType remaining = outIdx;
            for (int d = rank - 1; d >= 0; d--) {
                outCoords[d] = remaining % outShape[d];
                remaining /= outShape[d];
            }

            // Map to input coordinates using modulo
            std::vector<LongType> inCoords(rank);
            for (int d = 0; d < rank; d++) {
                inCoords[d] = outCoords[d] % inShape[d];
            }

            // Compute input index
            LongType inIdx = 0;
            LongType mult = 1;
            for (int d = rank - 1; d >= 0; d--) {
                inIdx += inCoords[d] * mult;
                mult *= inShape[d];
            }

            outPtr[outIdx] = inPtr[inIdx];
        }
    }
}

PLATFORM_IMPL(tile, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> repeats;
    if (block.width() > 1) {
        auto repsArr = INPUT_VARIABLE(1);
        for (LongType i = 0; i < repsArr->lengthOf(); i++) {
            repeats.push_back(repsArr->e<LongType>(i));
        }
    } else {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            repeats.push_back(INT_ARG(i));
        }
    }

    tileMPS(input, output, repeats);

    return sd::Status::OK;
}

PLATFORM_CHECK(tile, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS TILE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reverse
//////////////////////////////////////////////////////////////////////////

static void reverseMPS(NDArray* input, NDArray* output, const std::vector<int>& axes) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        auto inShape = input->shapeOf();
        int rank = input->rankOf();

        std::set<int> reverseAxes(axes.begin(), axes.end());

        for (LongType outIdx = 0; outIdx < output->lengthOf(); outIdx++) {
            // Convert output index to coordinates
            std::vector<LongType> coords(rank);
            LongType remaining = outIdx;
            for (int d = rank - 1; d >= 0; d--) {
                coords[d] = remaining % inShape[d];
                remaining /= inShape[d];
            }

            // Reverse coordinates on specified axes
            for (int ax : reverseAxes) {
                coords[ax] = inShape[ax] - 1 - coords[ax];
            }

            // Compute input index
            LongType inIdx = 0;
            LongType mult = 1;
            for (int d = rank - 1; d >= 0; d--) {
                inIdx += coords[d] * mult;
                mult *= inShape[d];
            }

            outPtr[outIdx] = inPtr[inIdx];
        }
    }
}

PLATFORM_IMPL(reverse, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<int> axes;
    if (block.width() > 1) {
        auto axesArr = INPUT_VARIABLE(1);
        for (LongType i = 0; i < axesArr->lengthOf(); i++) {
            axes.push_back(axesArr->e<int>(i));
        }
    } else if (block.getIArguments()->size() > 0) {
        for (size_t i = 0; i < block.getIArguments()->size(); i++) {
            axes.push_back(INT_ARG(i));
        }
    }

    reverseMPS(input, output, axes);

    return sd::Status::OK;
}

PLATFORM_CHECK(reverse, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS REVERSE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Pad
//////////////////////////////////////////////////////////////////////////

static void padMPS(NDArray* input, NDArray* output, const std::vector<std::pair<LongType, LongType>>& paddings, float padValue) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        auto inShape = input->shapeOf();
        auto outShape = output->shapeOf();
        int rank = input->rankOf();

        // Initialize output with pad value
        for (LongType i = 0; i < output->lengthOf(); i++) {
            outPtr[i] = padValue;
        }

        // Copy input to padded output
        for (LongType inIdx = 0; inIdx < input->lengthOf(); inIdx++) {
            // Convert input index to coordinates
            std::vector<LongType> inCoords(rank);
            LongType remaining = inIdx;
            for (int d = rank - 1; d >= 0; d--) {
                inCoords[d] = remaining % inShape[d];
                remaining /= inShape[d];
            }

            // Apply padding offset
            std::vector<LongType> outCoords(rank);
            for (int d = 0; d < rank; d++) {
                outCoords[d] = inCoords[d] + paddings[d].first;
            }

            // Compute output index
            LongType outIdx = 0;
            LongType mult = 1;
            for (int d = rank - 1; d >= 0; d--) {
                outIdx += outCoords[d] * mult;
                mult *= outShape[d];
            }

            outPtr[outIdx] = inPtr[inIdx];
        }
    }
}

PLATFORM_IMPL(pad, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);
    auto paddings = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float padValue = 0.0f;
    if (block.getTArguments()->size() > 0) {
        padValue = T_ARG(0);
    }

    // Parse paddings array
    std::vector<std::pair<LongType, LongType>> pads;
    for (LongType i = 0; i < paddings->sizeAt(0); i++) {
        pads.push_back({paddings->e<LongType>(i, 0), paddings->e<LongType>(i, 1)});
    }

    padMPS(input, output, pads, padValue);

    return sd::Status::OK;
}

PLATFORM_CHECK(pad, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS PAD OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Slice
//////////////////////////////////////////////////////////////////////////

static void sliceMPS(NDArray* input, NDArray* output, const std::vector<LongType>& begin, const std::vector<LongType>& size) {
    @autoreleasepool {
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        auto inShape = input->shapeOf();
        int rank = input->rankOf();

        for (LongType outIdx = 0; outIdx < output->lengthOf(); outIdx++) {
            // Convert output index to coordinates
            std::vector<LongType> outCoords(rank);
            LongType remaining = outIdx;
            for (int d = rank - 1; d >= 0; d--) {
                outCoords[d] = remaining % size[d];
                remaining /= size[d];
            }

            // Apply begin offset
            std::vector<LongType> inCoords(rank);
            for (int d = 0; d < rank; d++) {
                inCoords[d] = outCoords[d] + begin[d];
            }

            // Compute input index
            LongType inIdx = 0;
            LongType mult = 1;
            for (int d = rank - 1; d >= 0; d--) {
                inIdx += inCoords[d] * mult;
                mult *= inShape[d];
            }

            outPtr[outIdx] = inPtr[inIdx];
        }
    }
}

PLATFORM_IMPL(slice, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    std::vector<LongType> begin, size;

    if (block.width() >= 3) {
        auto beginArr = INPUT_VARIABLE(1);
        auto sizeArr = INPUT_VARIABLE(2);
        for (LongType i = 0; i < beginArr->lengthOf(); i++) {
            begin.push_back(beginArr->e<LongType>(i));
            size.push_back(sizeArr->e<LongType>(i));
        }
    } else {
        int rank = input->rankOf();
        for (int i = 0; i < rank; i++) {
            begin.push_back(INT_ARG(i));
            size.push_back(INT_ARG(rank + i));
        }
    }

    sliceMPS(input, output, begin, size);

    return sd::Status::OK;
}

PLATFORM_CHECK(slice, ENGINE_MPS) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS SLICE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}


//////////////////////////////////////////////////////////////////////////
// Unstack
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(unstack, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    int numOutputs = block.outputWidth();

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int rank = x->rankOf();
        if (axis < 0) axis += rank;

        const float* xPtr = x->bufferAsT<float>();
        auto xShape = x->shapeOf();

        LongType outerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= xShape[d];

        LongType axisSize = xShape[axis];

        LongType innerSize = 1;
        for (int d = axis + 1; d < rank; d++) innerSize *= xShape[d];

        for (int n = 0; n < numOutputs; n++) {
            auto z = OUTPUT_VARIABLE(n);
            float* zPtr = z->bufferAsT<float>();

            for (LongType outer = 0; outer < outerSize; outer++) {
                for (LongType inner = 0; inner < innerSize; inner++) {
                    LongType xIdx = outer * axisSize * innerSize + n * innerSize + inner;
                    LongType zIdx = outer * innerSize + inner;
                    zPtr[zIdx] = xPtr[xIdx];
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(unstack, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    Requirements req("MPS UNSTACK OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Repeat
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(repeat, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
        int rank = x->rankOf();
        if (axis < 0) axis += rank;

        LongType repeats = z->sizeAt(axis) / x->sizeAt(axis);
        auto xShape = x->shapeOf();
        auto zShape = z->shapeOf();

        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType outerSize = 1;
        for (int d = 0; d < axis; d++) outerSize *= xShape[d];

        LongType axisSize = xShape[axis];

        LongType innerSize = 1;
        for (int d = axis + 1; d < rank; d++) innerSize *= xShape[d];

        for (LongType outer = 0; outer < outerSize; outer++) {
            for (LongType a = 0; a < axisSize; a++) {
                for (LongType r = 0; r < repeats; r++) {
                    for (LongType inner = 0; inner < innerSize; inner++) {
                        LongType xIdx = outer * axisSize * innerSize + a * innerSize + inner;
                        LongType zIdx = outer * axisSize * repeats * innerSize +
                                        (a * repeats + r) * innerSize + inner;
                        zPtr[zIdx] = xPtr[xIdx];
                    }
                }
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(repeat, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    Requirements req("MPS REPEAT OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Reverse sequence
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reverse_sequence, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto seqLens = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int seqDim   = block.getIArguments()->size() > 0 ? INT_ARG(0) : 1;
        int batchDim = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;
        int rank = x->rankOf();
        if (seqDim   < 0) seqDim   += rank;
        if (batchDim < 0) batchDim += rank;

        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        auto xShape = x->shapeOf();

        LongType totalLen = x->lengthOf();

        for (LongType flatIdx = 0; flatIdx < totalLen; flatIdx++) {
            // Decompose flat index to per-dim coordinates
            std::vector<LongType> coords(rank);
            LongType rem = flatIdx;
            for (int d = rank - 1; d >= 0; d--) {
                coords[d] = rem % xShape[d];
                rem /= xShape[d];
            }

            LongType batchIdx = coords[batchDim];
            LongType seqLen = seqLens->e<LongType>(batchIdx);

            std::vector<LongType> srcCoords = coords;
            if (coords[seqDim] < seqLen) {
                srcCoords[seqDim] = seqLen - 1 - coords[seqDim];
            }

            // Compute source flat index
            LongType srcFlat = 0;
            LongType mult = 1;
            for (int d = rank - 1; d >= 0; d--) {
                srcFlat += srcCoords[d] * mult;
                mult *= xShape[d];
            }

            zPtr[flatIdx] = xPtr[srcFlat];
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(reverse_sequence, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    Requirements req("MPS REVERSE_SEQUENCE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Strided slice
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(strided_slice, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        int rank = x->rankOf();
        auto xShape = x->shapeOf();

        std::vector<LongType> begins(rank, 0);
        std::vector<LongType> ends(rank);
        std::vector<LongType> strides(rank, 1);

        for (int d = 0; d < rank; d++) ends[d] = xShape[d];

        if (block.width() >= 4) {
            auto begArr  = INPUT_VARIABLE(1);
            auto endArr  = INPUT_VARIABLE(2);
            auto strArr  = INPUT_VARIABLE(3);
            for (int d = 0; d < rank; d++) {
                begins[d]  = begArr->e<LongType>(d);
                ends[d]    = endArr->e<LongType>(d);
                strides[d] = strArr->e<LongType>(d);
            }
        } else {
            for (int d = 0; d < rank; d++) {
                begins[d]  = INT_ARG(d);
                ends[d]    = INT_ARG(rank + d);
                strides[d] = INT_ARG(2 * rank + d);
            }
        }

        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        auto zShape = z->shapeOf();
        int zRank = z->rankOf();

        for (LongType outFlat = 0; outFlat < z->lengthOf(); outFlat++) {
            std::vector<LongType> zCoords(zRank);
            LongType rem = outFlat;
            for (int d = zRank - 1; d >= 0; d--) {
                zCoords[d] = rem % zShape[d];
                rem /= zShape[d];
            }

            // Map z coords to x coords; handle rank differences (shrink/new axes ignored here)
            LongType xFlat = 0;
            LongType mult = 1;
            for (int d = rank - 1; d >= 0; d--) {
                LongType xCoord = begins[d] + zCoords[d] * strides[d];
                xFlat += xCoord * mult;
                mult *= xShape[d];
            }

            zPtr[outFlat] = xPtr[xFlat];
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(strided_slice, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    Requirements req("MPS STRIDED_SLICE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Scatter update
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(scatter_update, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);       // [N, ...]
    auto indices = INPUT_VARIABLE(1); // [K]
    auto updates = INPUT_VARIABLE(2); // [K, ...]
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        // Copy x into z first
        memcpy(z->buffer(), x->buffer(), x->lengthOf() * x->sizeOfT());

        const float* updPtr = updates->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        LongType numIndices = indices->lengthOf();
        LongType innerSize = x->lengthOf() / x->sizeAt(0);

        for (LongType i = 0; i < numIndices; i++) {
            LongType idx = indices->e<LongType>(i);
            if (idx >= 0 && idx < x->sizeAt(0)) {
                const float* src = updPtr + i * innerSize;
                float* dst = zPtr + idx * innerSize;
                memcpy(dst, src, innerSize * sizeof(float));
            }
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(scatter_update, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    Requirements req("MPS SCATTER_UPDATE OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Gather ND
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(gather_nd, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);       // Input tensor [d0, d1, ..., d_{n-1}]
    auto indices = INPUT_VARIABLE(1); // Index tensor [..., k]
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        auto xShape = x->shapeOf();
        auto idxShape = indices->shapeOf();
        int xRank = x->rankOf();
        int idxRank = indices->rankOf();
        int k = idxShape[idxRank - 1];  // depth of indexing into x

        // Output shape: indices.shape[:-1] + x.shape[k:]
        LongType outerLen = 1;
        for (int d = 0; d < idxRank - 1; d++) outerLen *= idxShape[d];

        LongType innerLen = 1;
        for (int d = k; d < xRank; d++) innerLen *= xShape[d];

        for (LongType outer = 0; outer < outerLen; outer++) {
            // Get the index vector from indices[outer, :]
            std::vector<LongType> idxVec(k);
            for (int ki = 0; ki < k; ki++) {
                LongType idxFlat = outer * k + ki;
                idxVec[ki] = indices->e<LongType>(idxFlat);
            }

            // Compute the base offset in x
            LongType xBase = 0;
            LongType mult = 1;
            // Build strides for x in reverse
            std::vector<LongType> xStrides(xRank);
            xStrides[xRank - 1] = 1;
            for (int d = xRank - 2; d >= 0; d--) {
                xStrides[d] = xStrides[d + 1] * xShape[d + 1];
            }
            for (int ki = 0; ki < k; ki++) {
                xBase += idxVec[ki] * xStrides[ki];
            }

            // Copy inner elements
            memcpy(zPtr + outer * innerLen, xPtr + xBase, innerLen * sizeof(float));
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(gather_nd, ENGINE_MPS) {
    auto x = INPUT_VARIABLE(0);
    Requirements req("MPS GATHER_ND OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");
    req.logTheSuccess();
    return req;
}

#else  // !HAVE_MPS — stub implementations so the translation unit links on non-Apple platforms

PLATFORM_IMPL(transpose, ENGINE_MPS)       { return sd::Status::OK; }
PLATFORM_CHECK(transpose, ENGINE_MPS)      { return Requirements("MPS TRANSPOSE STUB"); }

PLATFORM_IMPL(concat, ENGINE_MPS)          { return sd::Status::OK; }
PLATFORM_CHECK(concat, ENGINE_MPS)         { return Requirements("MPS CONCAT STUB"); }

PLATFORM_IMPL(stack, ENGINE_MPS)           { return sd::Status::OK; }
PLATFORM_CHECK(stack, ENGINE_MPS)          { return Requirements("MPS STACK STUB"); }

PLATFORM_IMPL(unstack, ENGINE_MPS)         { return sd::Status::OK; }
PLATFORM_CHECK(unstack, ENGINE_MPS)        { return Requirements("MPS UNSTACK STUB"); }

PLATFORM_IMPL(gather, ENGINE_MPS)          { return sd::Status::OK; }
PLATFORM_CHECK(gather, ENGINE_MPS)         { return Requirements("MPS GATHER STUB"); }

PLATFORM_IMPL(gather_nd, ENGINE_MPS)       { return sd::Status::OK; }
PLATFORM_CHECK(gather_nd, ENGINE_MPS)      { return Requirements("MPS GATHER_ND STUB"); }

PLATFORM_IMPL(tile, ENGINE_MPS)            { return sd::Status::OK; }
PLATFORM_CHECK(tile, ENGINE_MPS)           { return Requirements("MPS TILE STUB"); }

PLATFORM_IMPL(repeat, ENGINE_MPS)          { return sd::Status::OK; }
PLATFORM_CHECK(repeat, ENGINE_MPS)         { return Requirements("MPS REPEAT STUB"); }

PLATFORM_IMPL(reverse, ENGINE_MPS)         { return sd::Status::OK; }
PLATFORM_CHECK(reverse, ENGINE_MPS)        { return Requirements("MPS REVERSE STUB"); }

PLATFORM_IMPL(reverse_sequence, ENGINE_MPS) { return sd::Status::OK; }
PLATFORM_CHECK(reverse_sequence, ENGINE_MPS){ return Requirements("MPS REVERSE_SEQ STUB"); }

PLATFORM_IMPL(pad, ENGINE_MPS)             { return sd::Status::OK; }
PLATFORM_CHECK(pad, ENGINE_MPS)            { return Requirements("MPS PAD STUB"); }

PLATFORM_IMPL(slice, ENGINE_MPS)           { return sd::Status::OK; }
PLATFORM_CHECK(slice, ENGINE_MPS)          { return Requirements("MPS SLICE STUB"); }

PLATFORM_IMPL(strided_slice, ENGINE_MPS)   { return sd::Status::OK; }
PLATFORM_CHECK(strided_slice, ENGINE_MPS)  { return Requirements("MPS STRIDED_SLICE STUB"); }

PLATFORM_IMPL(scatter_update, ENGINE_MPS)  { return sd::Status::OK; }
PLATFORM_CHECK(scatter_update, ENGINE_MPS) { return Requirements("MPS SCATTER_UPDATE STUB"); }

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
