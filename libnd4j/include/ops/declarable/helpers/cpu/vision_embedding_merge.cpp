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
// CPU implementation of vision_embedding_merge.
// Scatters vision embeddings into text embeddings at target token positions.
//

#include <array/NDArray.h>
#include <ops/declarable/helpers/fused_llm_ops.h>
#include <system/op_boilerplate.h>
#include <system/type_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T, typename V, typename I>
static void visionEmbeddingMergeImpl(
    NDArray* textEmbeddings,
    NDArray* visionEmbeddings,
    NDArray* tokenIds,
    NDArray* output,
    sd::LongType targetTokenId) {

    const sd::LongType batch = textEmbeddings->sizeAt(0);
    const sd::LongType seqLen = textEmbeddings->sizeAt(1);
    const sd::LongType hiddenDim = textEmbeddings->sizeAt(2);
    const sd::LongType visionTokens = visionEmbeddings->sizeAt(1);

    const auto textBuf = textEmbeddings->bufferAsT<T>();
    const auto visBuf = visionEmbeddings->bufferAsT<V>();
    const auto tidBuf = tokenIds->bufferAsT<I>();
    auto outBuf = output->bufferAsT<T>();

    const auto textStrides = textEmbeddings->stridesOf();
    const auto visStrides = visionEmbeddings->stridesOf();
    const auto outStrides = output->stridesOf();
    const auto tidStrides = tokenIds->stridesOf();

    for (sd::LongType b = 0; b < batch; b++) {
        sd::LongType visionIdx = 0;

        for (sd::LongType s = 0; s < seqLen; s++) {
            sd::LongType tid = static_cast<sd::LongType>(tidBuf[b * tidStrides[0] + s * tidStrides[1]]);
            bool useVision = (tid == targetTokenId) && (visionIdx < visionTokens);

            const sd::LongType outBase = b * outStrides[0] + s * outStrides[1];

            if (useVision) {
                const sd::LongType visBase = b * visStrides[0] + visionIdx * visStrides[1];
                PRAGMA_OMP_SIMD
                for (sd::LongType h = 0; h < hiddenDim; h++) {
                    outBuf[outBase + h * outStrides[2]] =
                        static_cast<T>(visBuf[visBase + h * visStrides[2]]);
                }
                visionIdx++;
            } else {
                const sd::LongType txtBase = b * textStrides[0] + s * textStrides[1];
                PRAGMA_OMP_SIMD
                for (sd::LongType h = 0; h < hiddenDim; h++) {
                    outBuf[outBase + h * outStrides[2]] = textBuf[txtBase + h * textStrides[2]];
                }
            }
        }
    }
}

void visionEmbeddingMerge(
    NDArray* textEmbeddings,
    NDArray* visionEmbeddings,
    NDArray* tokenIds,
    NDArray* output,
    sd::LongType targetTokenId,
    LaunchContext* context) {

    BUILD_TRIPLE_SELECTOR(textEmbeddings->dataType(), visionEmbeddings->dataType(), tokenIds->dataType(),
        visionEmbeddingMergeImpl,
        (textEmbeddings, visionEmbeddings, tokenIds, output, targetTokenId),
        SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_INTEGER_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
