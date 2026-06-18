/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/segment_gemm.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

/**
 * Segmented GEMM CUDA kernel.
 *
 * Each block handles one (expert, token, outputChunk) triple.
 * Grid: (totalTokens, ceil(outDim / blockDim.x))
 *
 * The kernel finds which expert each token belongs to by scanning
 * segmentOffsets, then performs the matmul for that expert's weight matrix.
 */
template <typename T>
static SD_KERNEL void segmentGemmKernel(const void* vInput,
                                         const void* vWeights,
                                         const void* vSegmentOffsets,
                                         const void* vSegmentSizes,
                                         void* vOutput,
                                         const LongType numExperts,
                                         const LongType inDim,
                                         const LongType outDim,
                                         const LongType totalTokens,
                                         const LongType inputRowStride,
                                         const LongType inputElemStride,
                                         const LongType weightsExpertStride,
                                         const LongType weightsRowStride,
                                         const LongType weightsColStride,
                                         const LongType outputRowStride,
                                         const LongType outputElemStride) {
    auto input = reinterpret_cast<const T*>(vInput);
    auto weights = reinterpret_cast<const T*>(vWeights);
    auto offsets = reinterpret_cast<const LongType*>(vSegmentOffsets);
    auto sizes = reinterpret_cast<const LongType*>(vSegmentSizes);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType tokenIdx = blockIdx.x;
    if (tokenIdx >= totalTokens) return;

    // Find which expert this token belongs to
    LongType expertIdx = -1;
    for (LongType e = 0; e < numExperts; e++) {
        LongType off = offsets[e];
        LongType sz = sizes[e];
        if (tokenIdx >= off && tokenIdx < off + sz) {
            expertIdx = e;
            break;
        }
    }
    if (expertIdx < 0) return;  // token not assigned to any expert

    // Each thread computes one output element
    for (LongType j = threadIdx.x; j < outDim; j += blockDim.x) {
        float sum = 0.0f;
        LongType inputBase = tokenIdx * inputRowStride;
        LongType weightBase = expertIdx * weightsExpertStride;

        for (LongType k = 0; k < inDim; k++) {
            float inVal = static_cast<float>(input[inputBase + k * inputElemStride]);
            float wVal = static_cast<float>(weights[weightBase + k * weightsRowStride + j * weightsColStride]);
            sum += inVal * wVal;
        }
        output[tokenIdx * outputRowStride + j * outputElemStride] = static_cast<T>(sum);
    }
}

template <typename T>
static void segmentGemmLauncher(LaunchContext* context,
                                 NDArray* input, NDArray* weights,
                                 NDArray* segmentOffsets, NDArray* segmentSizes,
                                 NDArray* output) {
    auto stream = context->getCudaStream();

    auto numExperts = weights->sizeAt(0);
    auto inDim = weights->sizeAt(1);
    auto outDim = weights->sizeAt(2);
    auto totalTokens = input->sizeAt(0);

    auto inputStrides = input->stridesOf();
    auto weightsStrides = weights->stridesOf();
    auto outputStrides = output->stridesOf();

    int blockSize = 256;
    int gridSize = static_cast<int>(totalTokens);

    segmentGemmKernel<T><<<gridSize, blockSize, 0, *stream>>>(
        input->specialBuffer(),
        weights->specialBuffer(),
        segmentOffsets->specialBuffer(),
        segmentSizes->specialBuffer(),
        output->specialBuffer(),
        numExperts, inDim, outDim, totalTokens,
        inputStrides[0], inputStrides[1],
        weightsStrides[0], weightsStrides[1], weightsStrides[2],
        outputStrides[0], outputStrides[1]);
    DebugHelper::checkGlobalErrorCode("segmentGemmKernel failed");
}

void segmentGemm(LaunchContext* context,
                      NDArray* input, NDArray* weights,
                      NDArray* segmentOffsets, NDArray* segmentSizes,
                      NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input, weights, segmentOffsets, segmentSizes});
    BUILD_SINGLE_SELECTOR(input->dataType(), segmentGemmLauncher,
                          (context, input, weights, segmentOffsets, segmentSizes, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input, weights, segmentOffsets, segmentSizes});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
