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
#include <ops/declarable/helpers/hashcode.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static SD_KERNEL void splitBufferToChuncks(T* buffer, sd::LongType* tempBuffer, sd::LongType numBlocks,
                                           sd::LongType blockSize, sd::LongType length) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < numBlocks; b += gridDim.x * blockDim.x) {
    auto blockBuffer = buffer + b * numBlocks;

    sd::LongType r = 1LL;
    for (int e = 0; e < blockSize && e + (b * numBlocks) < length; e++) {
      auto v = longBytes<T>(blockBuffer[e]);
      r = 31LL * r + v;
    }

    tempBuffer[b] = r;
  }
}

template <typename T>
static SD_KERNEL void internalHash(sd::LongType* tempBuffer, sd::LongType* tempResult, sd::LongType numBlocks,
                                   sd::LongType blockSize, sd::LongType lastLength) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < numBlocks; b += gridDim.x * blockDim.x) {
    auto blockBuffer = tempBuffer + b * numBlocks;
    sd::LongType r = 1LL;

    for (sd::LongType e = 0; e < blockSize && e + (b * numBlocks) < lastLength; e++) {
      auto v = longBytes<T>(blockBuffer[e]);
      r = 31LL * r + v;
    }

    tempResult[b] = r;
  }
}

static SD_KERNEL void lastStep(sd::LongType* resultBuf, sd::LongType* tempBufferA, sd::LongType* tempResult,
                               sd::LongType length, sd::LongType blockSize) {
  if (threadIdx.x == 0) {
    if (length <= blockSize)
      *resultBuf = *tempBufferA;
    else
      *resultBuf = *tempResult;
  }
}

template <typename T>
void hashCode_(LaunchContext* context, NDArray& array, NDArray& result) {
  auto blockSize = 32;
  auto stream = context->getCudaStream();
  array.syncToDevice();

  NDArray::prepareSpecialUse({&result}, {&array});
  auto length = array.lengthOf();
  int numBlocks = length / blockSize + ((length % blockSize == 0) ? 0 : 1);
  auto tempA = NDArrayFactory::create<sd::LongType>('c', {numBlocks}, context);
  auto tempB = NDArrayFactory::create<sd::LongType>('c', {numBlocks / blockSize + 1}, context);

  auto buffer = reinterpret_cast<T*>(array.specialBuffer());                  // bufferAsT<T>();
  auto tempBufferA = reinterpret_cast<sd::LongType*>(tempA.specialBuffer());  // bufferAsT<sd::LongType>();
  auto tempBufferB = reinterpret_cast<sd::LongType*>(tempB.specialBuffer());  // bufferAsT<sd::LongType>();

  // default buffer is the first one, because it might be the last one in case of small arrays (< blockSize)
  auto tempBuffer = tempBufferA;
  auto tempResult = tempBufferB;

  // we divide array into 32 element chunks, and store intermediate results once
  splitBufferToChuncks<T><<<numBlocks, 1, 1024, *stream>>>(buffer, tempBuffer, numBlocks, blockSize, length);

  // we replace pointer with intermediate one, and repeat only one chunk left
  int iterationCount = 0;
  while (numBlocks > 1) {
    int lastLength = numBlocks;
    numBlocks = lastLength / blockSize + ((lastLength % blockSize == 0) ? 0 : 1);

    internalHash<sd::LongType>
        <<<numBlocks, 1, 1024, *stream>>>(tempBuffer, tempResult, numBlocks, blockSize, lastLength);

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

  lastStep<<<1, 1, 128, *stream>>>(reinterpret_cast<sd::LongType*>(result.specialBuffer()), tempBufferA, tempResult,
                                   length, blockSize);
  //                tempA.syncToHost();
  //                tempB.syncToHost();
  //                result.assign((length <= blockSize?tempA.e(0) : tempB.e(0)));

  NDArray::registerSpecialUse({&result}, {&array});
}

void hashCode(LaunchContext* context, NDArray& array, NDArray& result) {
  BUILD_SINGLE_SELECTOR(array.dataType(), hashCode_, (context, array, result), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void hashCode_, (LaunchContext * context, NDArray& array, NDArray& result),
                      SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
