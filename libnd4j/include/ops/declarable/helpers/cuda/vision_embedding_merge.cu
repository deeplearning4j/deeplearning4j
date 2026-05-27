/* ******************************************************************************
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
// CUDA implementation of vision_embedding_merge.
// Scatters vision embeddings into text embeddings at matching token positions.
// Entirely device-side — no host round-trips.
//
// Two-pass approach:
//   Pass 1: build exclusive prefix sum of target token matches per batch row
//           (gives each position its vision token index without sequential dependency)
//   Pass 2: scatter — each thread copies one element from either vision or text
//           source based on the prefix sum mapping
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/fused_llm_ops.h>
#include <system/op_boilerplate.h>
#include <system/type_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Pass 1: build exclusive prefix sum of target-token matches.
 *
 * prefixSum[b * seqLen + s] = count of target tokens before position s in batch b.
 * One thread per batch row. seqLen is small (< 8192) so serial scan per row is fine;
 * the expensive work is the element-wise scatter in pass 2.
 */
template <typename I>
SD_KERNEL static void buildPrefixSumKernel(
    const I* __restrict__ tokenIds,
    int* __restrict__ prefixSum,
    sd::LongType batch, sd::LongType seqLen,
    sd::LongType tidStride0, sd::LongType tidStride1,
    sd::LongType targetTokenId) {

    sd::LongType b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    int running = 0;
    for (sd::LongType s = 0; s < seqLen; s++) {
        prefixSum[b * seqLen + s] = running;
        sd::LongType tid = static_cast<sd::LongType>(tokenIds[b * tidStride0 + s * tidStride1]);
        if (tid == targetTokenId) {
            running++;
        }
    }
}

/**
 * Pass 2: scatter vision or text embeddings into output.
 *
 * Each thread handles one element: output[b][s][h].
 * If tokenIds[b][s] == targetTokenId, read from visionEmbeddings[b][visionIdx][h]
 * where visionIdx = prefixSum[b * seqLen + s]. Otherwise copy from textEmbeddings[b][s][h].
 */
template <typename T, typename V, typename I>
SD_KERNEL static void visionEmbeddingMergeKernel(
    const T* __restrict__ textData,
    const V* __restrict__ visionData,
    const I* __restrict__ tokenIds,
    const int* __restrict__ prefixSum,
    T* __restrict__ outputData,
    sd::LongType batch, sd::LongType seqLen, sd::LongType hiddenDim,
    sd::LongType visionTokens, sd::LongType targetTokenId,
    sd::LongType textS0, sd::LongType textS1, sd::LongType textS2,
    sd::LongType visS0, sd::LongType visS1, sd::LongType visS2,
    sd::LongType tidS0, sd::LongType tidS1,
    sd::LongType outS0, sd::LongType outS1, sd::LongType outS2) {

    sd::LongType idx = (sd::LongType)blockIdx.x * blockDim.x + threadIdx.x;
    sd::LongType total = batch * seqLen * hiddenDim;
    if (idx >= total) return;

    sd::LongType h = idx % hiddenDim;
    sd::LongType bs = idx / hiddenDim;
    sd::LongType s = bs % seqLen;
    sd::LongType b = bs / seqLen;

    sd::LongType tid = static_cast<sd::LongType>(tokenIds[b * tidS0 + s * tidS1]);
    sd::LongType outOffset = b * outS0 + s * outS1 + h * outS2;

    if (tid == targetTokenId) {
        int visionIdx = prefixSum[b * seqLen + s];
        if (visionIdx < visionTokens) {
            outputData[outOffset] = static_cast<T>(visionData[b * visS0 + visionIdx * visS1 + h * visS2]);
        } else {
            outputData[outOffset] = textData[b * textS0 + s * textS1 + h * textS2];
        }
    } else {
        outputData[outOffset] = textData[b * textS0 + s * textS1 + h * textS2];
    }
}

template <typename T, typename V, typename I>
static void visionEmbeddingMerge_(
    NDArray* textEmbeddings,
    NDArray* visionEmbeddings,
    NDArray* tokenIds,
    NDArray* output,
    sd::LongType targetTokenId,
    LaunchContext* context) {

    const sd::LongType batch = textEmbeddings->sizeAt(0);
    const sd::LongType seqLen = textEmbeddings->sizeAt(1);
    const sd::LongType hiddenDim = textEmbeddings->sizeAt(2);
    const sd::LongType visionTokens = visionEmbeddings->sizeAt(1);

    auto* stream = context->getCudaStream();

    // Allocate device-side prefix sum buffer
    int* prefixSumDev = nullptr;
    size_t allocSize = batch * seqLen * sizeof(int);
    cudaMallocAsync(&prefixSumDev, allocSize, *stream);

    // Pass 1: build prefix sum (one thread per batch row)
    {
        const dim3 dims = getLaunchDims("vision_embedding_merge");
        int blockSize = static_cast<int>(dims.y);
        int gridSize = ((int)batch + blockSize - 1) / blockSize;
        buildPrefixSumKernel<I><<<gridSize, blockSize, 0, *stream>>>(
            tokenIds->bufferAsT<I>(),
            prefixSumDev,
            batch, seqLen,
            tokenIds->stridesOf()[0], tokenIds->stridesOf()[1],
            targetTokenId);
    }

    // Pass 2: scatter (one thread per output element)
    {
        const dim3 dims = getLaunchDims("vision_embedding_merge");
        sd::LongType totalElements = batch * seqLen * hiddenDim;
        int blockSize = static_cast<int>(dims.y);
        int gridSize = ((int)totalElements + blockSize - 1) / blockSize;

        const auto textStrides = textEmbeddings->stridesOf();
        const auto visStrides = visionEmbeddings->stridesOf();
        const auto tidStrides = tokenIds->stridesOf();
        const auto outStrides = output->stridesOf();

        visionEmbeddingMergeKernel<T, V, I><<<gridSize, blockSize, 0, *stream>>>(
            textEmbeddings->bufferAsT<T>(),
            visionEmbeddings->bufferAsT<V>(),
            tokenIds->bufferAsT<I>(),
            prefixSumDev,
            output->bufferAsT<T>(),
            batch, seqLen, hiddenDim, visionTokens, targetTokenId,
            textStrides[0], textStrides[1], textStrides[2],
            visStrides[0], visStrides[1], visStrides[2],
            tidStrides[0], tidStrides[1],
            outStrides[0], outStrides[1], outStrides[2]);
    }

    cudaFreeAsync(prefixSumDev, *stream);

    DebugHelper::checkGlobalErrorCode("visionEmbeddingMergeKernel failed");
}

void visionEmbeddingMerge(
    NDArray* textEmbeddings,
    NDArray* visionEmbeddings,
    NDArray* tokenIds,
    NDArray* output,
    sd::LongType targetTokenId,
    LaunchContext* context) {

    NDArray::prepareSpecialUse({output}, {textEmbeddings, visionEmbeddings, tokenIds});

    BUILD_TRIPLE_SELECTOR(textEmbeddings->dataType(), visionEmbeddings->dataType(), tokenIds->dataType(),
        visionEmbeddingMerge_,
        (textEmbeddings, visionEmbeddings, tokenIds, output, targetTokenId, context),
        SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_INTEGER_TYPES);

    NDArray::registerSpecialUse({output}, {textEmbeddings, visionEmbeddings, tokenIds});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
