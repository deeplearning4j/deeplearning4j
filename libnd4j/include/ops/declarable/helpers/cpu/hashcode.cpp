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
#include <helpers/shape.h>
#include <ops/declarable/helpers/hashcode.h>

#include <algorithm>
#include <vector>
#if NOT_EXCLUDED(OP_hashcode)
namespace sd {
namespace ops {
namespace helpers {
// The hash of the elements of the array in C order (the order its logical coordinates give, not the order of its
// memory), a tree of polynomial hashes: blocks of 32 consecutive elements hash to one value each, blocks of 32 of
// those values to one value each, and so on until one value is left. The values of the upper levels are 64-bit
// hashes and are combined as they are.
template <typename T>
static void hashCode_(LaunchContext *context, NDArray &array, NDArray &result) {
  const LongType blockSize = 32;
  const LongType length = array.lengthOf();

  // the hash of no element is the seed of the polynomial
  if (length == 0) {
    result.p(0, static_cast<LongType>(1));
    return;
  }

  const LongType numBlocks = length / blockSize + ((length % blockSize == 0) ? 0 : 1);
  std::vector<LongType> levelA(numBlocks);
  std::vector<LongType> levelB(numBlocks / blockSize + ((numBlocks % blockSize == 0) ? 0 : 1));

  const T *buffer = array.bufferAsT<T>();
  const LongType rank = array.rankOf();
  const LongType *xShape = shape::shapeOf(array.shapeInfo());
  const LongType *xStride = shape::stride(array.shapeInfo());
  // the elements of a dense C-order array are its memory in order, any other layout goes through its strides
  const bool dense = shape::isDenseRowMajor(array.shapeInfo());

  LongType *current = levelA.data();
  LongType *next = levelB.data();

  // we divide the array into 32 element blocks, and store each block's hash
  auto split = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK];
    for (auto b = start; b < stop; b++) {
      const LongType first = b * blockSize;
      const LongType last = std::min(first + blockSize, length);

      LongType r = 1;
      for (LongType e = first; e < last; e++) {
        LongType offset = e;
        if (!dense) {
          INDEX2COORDS(e, rank, xShape, coords);
          COORDS2INDEX(rank, xStride, coords, offset);
        }
        r = hashCodeStep(r, longBytes<T>(buffer[offset]));
      }

      current[b] = r;
    }
  };
  samediff::Threads::parallel_for(split, 0, numBlocks);

  // then we hash the hashes of a level in blocks of 32, until one block is left
  LongType count = numBlocks;
  while (count > 1) {
    const LongType numMerged = count / blockSize + ((count % blockSize == 0) ? 0 : 1);

    auto merge = PRAGMA_THREADS_FOR {
      for (auto b = start; b < stop; b++) {
        const LongType first = b * blockSize;
        const LongType last = std::min(first + blockSize, count);

        LongType r = 1;
        for (LongType e = first; e < last; e++) r = hashCodeStep(r, current[e]);

        next[b] = r;
      }
    };
    samediff::Threads::parallel_for(merge, 0, numMerged);

    // the level just written is the one hashed next
    std::swap(current, next);
    count = numMerged;
  }

  result.p(0, current[0]);
}

void hashCode(LaunchContext *context, NDArray &array, NDArray &result) {
  BUILD_SINGLE_SELECTOR(array.dataType(), hashCode_, (context, array, result), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
