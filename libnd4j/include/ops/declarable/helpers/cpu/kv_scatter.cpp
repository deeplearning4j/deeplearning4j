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

#include <ops/declarable/helpers/kv_scatter.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void kvScatter_(NDArray* present, NDArray* output,
                          LongType cachePos, LaunchContext* context) {
  auto lastPos = present->sizeAt(2) - 1;
  auto batch = present->sizeAt(0);
  auto heads = present->sizeAt(1);
  auto dim = present->sizeAt(3);

  auto srcSeqLen = present->sizeAt(2);
  auto dstSeqLen = output->sizeAt(2);

  const T* __restrict srcBuf = present->bufferAsT<T>();
  T* __restrict dstBuf = output->bufferAsT<T>();

  // Total number of (batch, head) slices to process — parallelize over these
  auto numSlices = batch * heads;

  auto func = PRAGMA_THREADS_FOR {
    for (auto slice = start; slice < stop; slice++) {
      auto b = slice / heads;
      auto h = slice % heads;

      auto srcOffset = b * heads * srcSeqLen * dim + h * srcSeqLen * dim + lastPos * dim;
      auto dstOffset = b * heads * dstSeqLen * dim + h * dstSeqLen * dim + cachePos * dim;

      const T* __restrict src = srcBuf + srcOffset;
      T* __restrict dst = dstBuf + dstOffset;

      PRAGMA_OMP_SIMD
      for (LongType d = 0; d < dim; d++) {
        dst[d] = src[d];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, numSlices);
}

void kvScatter(NDArray* present, NDArray* output,
               LongType cachePos, LaunchContext* context) {
  BUILD_SINGLE_SELECTOR(present->dataType(), kvScatter_, (present, output, cachePos, context), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void kvScatter_, (NDArray* present, NDArray* output,
                       LongType cachePos, LaunchContext* context), SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
