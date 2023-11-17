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

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static SD_KERNEL void splitBufferToChuncks(T* buffer, LongType* tempBuffer, LongType numBlocks, LongType blockSize,
                                           LongType length) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < numBlocks; b += gridDim.x * blockDim.x) {
    auto blockBuffer = buffer + b * numBlocks;

    LongType r = 1LL;
    for (int e = 0; e < blockSize && e + (b * numBlocks) < length; e++) {
      auto v = longBytes<T>(blockBuffer[e]);
      r = 31LL * r + v;
    }

    tempBuffer[b] = r;
  }
}

template <typename T>
static SD_KERNEL void internalHash(LongType* tempBuffer, LongType* tempResult, LongType numBlocks, LongType blockSize,
                                   LongType lastLength) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < numBlocks; b += gridDim.x * blockDim.x) {
    auto blockBuffer = tempBuffer + b * numBlocks;
    LongType r = 1LL;

    for (LongType e = 0; e < blockSize && e + (b * numBlocks) < lastLength; e++) {
      auto v = longBytes<T>(blockBuffer[e]);
      r = 31LL * r + v;
    }

    tempResult[b] = r;
  }
}

static SD_KERNEL void lastStep(LongType* resultBuf, LongType* tempBufferA, LongType* tempResult, LongType length,
                               LongType blockSize) {
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
  auto tempA = NDArrayFactory::create<LongType>('c', {numBlocks}, context);
  auto tempB = NDArrayFactory::create<LongType>('c', {numBlocks / blockSize + 1}, context);

  auto buffer = reinterpret_cast<T*>(array.specialBuffer());                  // bufferAsT<T>();
  auto tempBufferA = reinterpret_cast<LongType*>(tempA.specialBuffer());  // bufferAsT<sd::LongType>();
  auto tempBufferB = reinterpret_cast<LongType*>(tempB.specialBuffer());  // bufferAsT<sd::LongType>();

  dim3 launchDims = getHashCodeSplit(length,numBlocks);
  // default buffer is the first one, because it might be the last one in case of small arrays (< blockSize)
  auto tempBuffer = tempBufferA;
  auto tempResult = tempBufferB;

  // we divide array into 32 element chunks, and store intermediate results once
  splitBufferToChuncks<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(buffer, tempBuffer, numBlocks, blockSize, length);
  DebugHelper::checkErrorCode(context->getCudaStream(),"splitBufferToChuncks failed");

  // we replace pointer with intermediate one, and repeat only one chunk left
  int iterationCount = 0;
  while (numBlocks > 1) {
    int lastLength = numBlocks;
    numBlocks = lastLength / blockSize + ((lastLength % blockSize == 0) ? 0 : 1);

    dim3 internalLaunchDims = getHashCodeInternal(numBlocks);
    internalHash<LongType>
        <<<internalLaunchDims.y,internalLaunchDims.x, internalLaunchDims.z, *stream>>>(tempBuffer, tempResult, numBlocks, blockSize, lastLength);
    DebugHelper::checkErrorCode(context->getCudaStream(),"internalHash failed");

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

  dim3 lastDims = getLaunchDims("hashcode_last");
  lastStep<<<lastDims.x, lastDims.y, lastDims.z, *stream>>>(reinterpret_cast<LongType*>(result.specialBuffer()), tempBufferA, tempResult,
                                   length, blockSize);
  DebugHelper::checkErrorCode(context->getCudaStream(),"lastStep failed");

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
