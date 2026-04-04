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
#include <execution/Threads.h>

#if NOT_EXCLUDED(OP_per_layer_embedding)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void perLayerEmbedding_(LaunchContext* context,
                                NDArray* hiddenStates,
                                NDArray* pleWeight,
                                NDArray* tokenIds,
                                NDArray* output,
                                double scale) {
  auto batch = hiddenStates->sizeAt(0);
  auto seqLen = hiddenStates->sizeAt(1);
  auto hiddenDim = hiddenStates->sizeAt(2);

  const T* hiddenBuf = hiddenStates->bufferAsT<T>();
  const T* pleBuf = pleWeight->bufferAsT<T>();
  T* outBuf = output->bufferAsT<T>();

  auto hiddenStride0 = hiddenStates->strideAt(0);
  auto hiddenStride1 = hiddenStates->strideAt(1);
  auto hiddenStride2 = hiddenStates->strideAt(2);

  auto pleStride0 = pleWeight->strideAt(0);
  auto pleStride1 = pleWeight->strideAt(1);

  auto outStride0 = output->strideAt(0);
  auto outStride1 = output->strideAt(1);
  auto outStride2 = output->strideAt(2);

  T scaleT = static_cast<T>(scale);

  auto tokenStride0 = tokenIds->strideAt(0);
  auto tokenStride1 = tokenIds->strideAt(1);

  auto totalPositions = batch * seqLen;

  auto func = PRAGMA_THREADS_FOR {
    for (auto pos = start; pos < stop; pos++) {
      auto b = pos / seqLen;
      auto s = pos % seqLen;

      // Use the two-index e<> accessor for correct strided element access
      auto tokenId = tokenIds->e<LongType>(b, s);

      auto hiddenBase = b * hiddenStride0 + s * hiddenStride1;
      auto outBase = b * outStride0 + s * outStride1;
      auto pleBase = tokenId * pleStride0;

      PRAGMA_OMP_SIMD
      for (LongType d = 0; d < hiddenDim; d++) {
        outBuf[outBase + d * outStride2] =
            hiddenBuf[hiddenBase + d * hiddenStride2] + pleBuf[pleBase + d * pleStride1] * scaleT;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, totalPositions);
}

void perLayerEmbedding(LaunchContext* context,
                        NDArray* hiddenStates,
                        NDArray* pleWeight,
                        NDArray* tokenIds,
                        NDArray* output,
                        double scale) {
  NDArray::preparePrimaryUse({output}, {hiddenStates, pleWeight, tokenIds});
  BUILD_SINGLE_SELECTOR(hiddenStates->dataType(), perLayerEmbedding_,
                         (context, hiddenStates, pleWeight, tokenIds, output, scale),
                         SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {hiddenStates, pleWeight, tokenIds});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
