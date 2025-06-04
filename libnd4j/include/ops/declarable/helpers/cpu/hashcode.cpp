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
// @author raver119@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/hashcode.h>
#if NOT_EXCLUDED(OP_hashcode)
namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static void hashCode_(LaunchContext *context, NDArray &array, NDArray &result) {
  sd::LongType blockSize = 32;
  auto length = array.lengthOf();
  int numBlocks = length / blockSize + ((length % blockSize == 0) ? 0 : 1);
  auto tempA = NDArrayFactory::create<sd::LongType>('c', {numBlocks}, context);
  auto tempB = NDArrayFactory::create<sd::LongType>('c', {numBlocks / blockSize + 1}, context);

  auto buffer = array.bufferAsT<T>();
  auto tempBufferA = tempA.bufferAsT<sd::LongType>();
  auto tempBufferB = tempB.bufferAsT<sd::LongType>();

  // default buffer is the first one, because it might be the last one in case of small arrays (< blockSize)
  auto tempBuffer = tempBufferA;
  auto tempResult = tempBufferB;

  // we divide array into 32 element chunks, and store intermediate results once
  auto func = PRAGMA_THREADS_FOR {
    for (auto b = start; b < stop; b++) {
      auto blockBuffer = buffer + b * numBlocks;

      sd::LongType r = 1;
      for (sd::LongType e = 0; e < blockSize && e + (b * numBlocks) < length; e++) {
        auto v = longBytes<T>(blockBuffer[e]);
        r = 31 * r + v;
      }

      tempBuffer[b] = r;
    }
  };
  samediff::Threads::parallel_tad(func, 0, numBlocks);

  // we replace pointer with intermediate one, and repeat only one chunk left
  int iterationCount = 0;
  while (numBlocks > 1) {
    int lastLength = numBlocks;
    numBlocks = lastLength / blockSize + ((lastLength % blockSize == 0) ? 0 : 1);

    auto func2 = PRAGMA_THREADS_FOR {
      for (auto b = start; b < stop; b++) {
        auto blockBuffer = tempBuffer + b * numBlocks;

        sd::LongType r = 1;
        for (sd::LongType e = 0; e < blockSize && e + (b * numBlocks) < lastLength; e++) {
          auto v = longBytes<T>(static_cast<T>(blockBuffer[e]));
          r = 31 * r + v;
        }

        tempResult[b] = r;
      }
    };
    samediff::Threads::parallel_tad(func2, 0, numBlocks);

    iterationCount++;
    // swapping buffers
    if (iterationCount % 2 == 0) {
      tempBuffer = tempBufferA;
      tempResult = tempBufferB;
    } else {
      tempBuffer = tempBufferB;
      tempResult = tempBufferA;
    }
  }

  if (length <= blockSize)
    result.p(0, tempBufferA[0]);
  else
    result.p(0, tempResult[0]);
}

void hashCode(LaunchContext *context, NDArray &array, NDArray &result) {
  BUILD_SINGLE_SELECTOR(array.dataType(), hashCode_, (context, array, result), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif