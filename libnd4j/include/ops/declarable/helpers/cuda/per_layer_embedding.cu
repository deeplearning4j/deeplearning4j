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

#include <ops/declarable/helpers/per_layer_embedding.h>
#include <array/NDArray.h>
#include <helpers/DebugHelper.h>
#include <execution/cuda/LaunchDims.h>
#include <cuda_runtime.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * CUDA kernel: per-layer embedding
 *
 * output[b, s, d] = hidden[b, s, d] + ple_weight[token_ids[b, s], d] * scale
 *
 * Grid-stride loop over total elements (B * S * D).
 * Each thread handles one output element.
 */
template <typename T>
SD_KERNEL void perLayerEmbeddingKernel(const T* __restrict__ hidden,
                                        const T* __restrict__ pleWeight,
                                        const LongType* __restrict__ tokenIds,
                                        T* __restrict__ output,
                                        const LongType batch,
                                        const LongType seqLen,
                                        const LongType hiddenDim,
                                        const LongType hiddenStride0,
                                        const LongType hiddenStride1,
                                        const LongType hiddenStride2,
                                        const LongType pleStride0,
                                        const LongType pleStride1,
                                        const LongType tokenStride0,
                                        const LongType tokenStride1,
                                        const LongType outStride0,
                                        const LongType outStride1,
                                        const LongType outStride2,
                                        const T scale) {
  auto totalElements = batch * seqLen * hiddenDim;

  for (auto idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < totalElements;
       idx += static_cast<LongType>(gridDim.x) * blockDim.x) {

    auto d = idx % hiddenDim;
    auto s = (idx / hiddenDim) % seqLen;
    auto b = idx / (hiddenDim * seqLen);

    auto tokenId = tokenIds[b * tokenStride0 + s * tokenStride1];

    output[b * outStride0 + s * outStride1 + d * outStride2] =
        hidden[b * hiddenStride0 + s * hiddenStride1 + d * hiddenStride2] +
        pleWeight[tokenId * pleStride0 + d * pleStride1] * scale;
  }
}

template <typename T>
static void perLayerEmbeddingCudaLauncher(const cudaStream_t* stream,
                                           const void* vHidden,
                                           const void* vPleWeight,
                                           const void* vTokenIds,
                                           void* vOutput,
                                           LongType batch, LongType seqLen, LongType hiddenDim,
                                           LongType hiddenStride0, LongType hiddenStride1, LongType hiddenStride2,
                                           LongType pleStride0, LongType pleStride1,
                                           LongType tokenStride0, LongType tokenStride1,
                                           LongType outStride0, LongType outStride1, LongType outStride2,
                                           double scaleDouble) {
  auto hidden = reinterpret_cast<const T*>(vHidden);
  auto pleWeight = reinterpret_cast<const T*>(vPleWeight);
  auto tokenIds = reinterpret_cast<const LongType*>(vTokenIds);
  auto output = reinterpret_cast<T*>(vOutput);

  T scale = static_cast<T>(scaleDouble);

  auto totalElements = batch * seqLen * hiddenDim;
  dim3 launchDims = getLaunchDims("per_layer_embedding");

  auto gridSize = static_cast<int>((totalElements + launchDims.y - 1) / launchDims.y);
  if (gridSize > static_cast<int>(launchDims.x)) gridSize = launchDims.x;

  perLayerEmbeddingKernel<T><<<gridSize, launchDims.y, launchDims.z, *stream>>>(
      hidden, pleWeight, tokenIds, output,
      batch, seqLen, hiddenDim,
      hiddenStride0, hiddenStride1, hiddenStride2,
      pleStride0, pleStride1,
      tokenStride0, tokenStride1,
      outStride0, outStride1, outStride2,
      scale);

  DebugHelper::checkGlobalErrorCode("per_layer_embedding kernel failed");
}

BUILD_SINGLE_TEMPLATE(void perLayerEmbeddingCudaLauncher,
                       (const cudaStream_t* stream, const void* vHidden, const void* vPleWeight,
                        const void* vTokenIds, void* vOutput,
                        LongType batch, LongType seqLen, LongType hiddenDim,
                        LongType hiddenStride0, LongType hiddenStride1, LongType hiddenStride2,
                        LongType pleStride0, LongType pleStride1,
                        LongType tokenStride0, LongType tokenStride1,
                        LongType outStride0, LongType outStride1, LongType outStride2,
                        double scaleDouble),
                       SD_FLOAT_TYPES);

void perLayerEmbedding(LaunchContext* context,
                        NDArray* hiddenStates,
                        NDArray* pleWeight,
                        NDArray* tokenIds,
                        NDArray* output,
                        double scale) {
  auto stream = context->getCudaStream();

  auto batch = hiddenStates->sizeAt(0);
  auto seqLen = hiddenStates->sizeAt(1);
  auto hiddenDim = hiddenStates->sizeAt(2);

  // Token IDs must be INT64 on device for the kernel
  NDArray tokenIdsLong = tokenIds->dataType() == INT64 ? *tokenIds : *tokenIds->cast(INT64);

  NDArray::prepareSpecialUse({output}, {hiddenStates, pleWeight, &tokenIdsLong});

  BUILD_SINGLE_SELECTOR(hiddenStates->dataType(), perLayerEmbeddingCudaLauncher,
                         (stream,
                          hiddenStates->specialBuffer(),
                          pleWeight->specialBuffer(),
                          tokenIdsLong.specialBuffer(),
                          output->specialBuffer(),
                          batch, seqLen, hiddenDim,
                          hiddenStates->strideAt(0), hiddenStates->strideAt(1), hiddenStates->strideAt(2),
                          pleWeight->strideAt(0), pleWeight->strideAt(1),
                          tokenIdsLong.strideAt(0), tokenIdsLong.strideAt(1),
                          output->strideAt(0), output->strideAt(1), output->strideAt(2),
                          scale),
                         SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({output}, {hiddenStates, pleWeight, &tokenIdsLong});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
