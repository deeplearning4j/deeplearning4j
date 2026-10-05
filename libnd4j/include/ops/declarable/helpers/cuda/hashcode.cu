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
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/hashcode.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// The hash of the elements of the array in C order (the order its logical coordinates give, not the order of its
// memory), a tree of polynomial hashes as the CPU helper builds it: blocks of 32 consecutive elements hash to one
// value each, blocks of 32 of those values to one value each, and so on until one value is left.

// level 0: a thread hashes each block of 32 consecutive elements, the elements of a dense C-order array being its
// memory in order and any other layout going through its strides
template <typename T>
static SD_KERNEL void hashBlocksKernel(const void* vx, const LongType* xShapeInfo, LongType* hashes,
                                       const LongType numBlocks, const LongType blockSize, const LongType length,
                                       const bool dense) {
  const T* x = reinterpret_cast<const T*>(vx);

  const LongType rank = shape::rank(xShapeInfo);
  const LongType* xShape = shape::shapeOf(xShapeInfo);
  const LongType* xStride = shape::stride(xShapeInfo);

  LongType coords[SD_MAX_RANK];

  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType b = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; b < numBlocks; b += step) {
    const LongType first = b * blockSize;
    const LongType last = first + blockSize < length ? first + blockSize : length;

    LongType r = 1;
    for (LongType e = first; e < last; e++) {
      LongType offset = e;
      if (!dense) {
        INDEX2COORDS(e, rank, xShape, coords);
        COORDS2INDEX(rank, xStride, coords, offset);
      }
      r = hashCodeStep(r, longBytes<T>(x[offset]));
    }

    hashes[b] = r;
  }
}

// the upper levels: a thread hashes each block of 32 consecutive hashes of the level below
static SD_KERNEL void hashLevelKernel(const LongType* hashes, LongType* merged, const LongType numMerged,
                                      const LongType blockSize, const LongType count) {
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType b = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; b < numMerged; b += step) {
    const LongType first = b * blockSize;
    const LongType last = first + blockSize < count ? first + blockSize : count;

    LongType r = 1;
    for (LongType e = first; e < last; e++) r = hashCodeStep(r, hashes[e]);

    merged[b] = r;
  }
}

// the hash left on the top level is the result
static SD_KERNEL void hashResultKernel(LongType* result, const LongType* hash) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *result = *hash;
}

// a thread for each of the work items, at most as many blocks as the named launch's grid
static void hashLaunch(const char* name, const LongType work, unsigned int& blocks, unsigned int& threads) {
  const dim3 dims = getLaunchDims(name);
  threads = dims.y > 0 ? dims.y : 1;
  const LongType needed = (work + threads - 1) / threads;
  blocks = static_cast<unsigned int>(needed < static_cast<LongType>(dims.x) ? needed : static_cast<LongType>(dims.x));
  if (blocks < 1) blocks = 1;
}

template <typename T>
void hashCode_(LaunchContext* context, NDArray& array, NDArray& result) {
  const LongType blockSize = 32;
  const LongType length = array.lengthOf();

  // the hash of no element is the seed of the polynomial
  if (length == 0) {
    result.p(0, static_cast<LongType>(1));
    return;
  }

  auto stream = context->getCudaStream();
  const LongType numBlocks = length / blockSize + ((length % blockSize == 0) ? 0 : 1);

  PointersManager manager(context, "hashCode");
  LongType* levelA = reinterpret_cast<LongType*>(manager.allocateDevMem(numBlocks * sizeof(LongType)));
  LongType* levelB = reinterpret_cast<LongType*>(manager.allocateDevMem(
      (numBlocks / blockSize + ((numBlocks % blockSize == 0) ? 0 : 1)) * sizeof(LongType)));

  NDArray::prepareSpecialUse({&result}, {&array});

  LongType* current = levelA;
  LongType* next = levelB;
  const bool dense = shape::isDenseRowMajor(array.shapeInfo());

  // we divide the array into 32 element blocks, and store each block's hash
  unsigned int blocks, threads;
  hashLaunch("hashcode_split", numBlocks, blocks, threads);
  hashBlocksKernel<T><<<blocks, threads, 0, *stream>>>(array.specialBuffer(), array.specialShapeInfo(), current,
                                                       numBlocks, blockSize, length, dense);
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("hashBlocksKernel failed");

  // then we hash the hashes of a level in blocks of 32, until one block is left
  LongType count = numBlocks;
  while (count > 1) {
    const LongType numMerged = count / blockSize + ((count % blockSize == 0) ? 0 : 1);

    hashLaunch("hashcode_internal", numMerged, blocks, threads);
    hashLevelKernel<<<blocks, threads, 0, *stream>>>(current, next, numMerged, blockSize, count);
    if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("hashLevelKernel failed");

    // the level just written is the one hashed next
    LongType* written = next;
    next = current;
    current = written;
    count = numMerged;
  }

  hashLaunch("hashcode_last", 1, blocks, threads);
  hashResultKernel<<<blocks, threads, 0, *stream>>>(reinterpret_cast<LongType*>(result.specialBuffer()), current);
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("hashResultKernel failed");

  NDArray::registerSpecialUse({&result}, {&array});
}

void hashCode(LaunchContext* context, NDArray& array, NDArray& result) {
  BUILD_SINGLE_SELECTOR(array.dataType(), hashCode_, (context, array, result), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void hashCode_, (LaunchContext * context, NDArray& array, NDArray& result),
                      SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
