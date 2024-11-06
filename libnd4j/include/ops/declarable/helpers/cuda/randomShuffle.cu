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
// @author Yurii Shyrma (iuriish@yahoo.com)
// implemented algorithm is GPU adaptation of algorithm described in following article:
// "MergeShuffle: A Very Fast, Parallel Random Permutation Algorithm", https://arxiv.org/abs/1508.03167
//
#include <array/ResultSet.h>
#include <execution/Threads.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static SD_KERNEL void fisherYatesCuda(graph::RandomGenerator* rng, void* vx, const LongType ews,
                                      const LongType len, const int power) {
  T* x = reinterpret_cast<T*>(vx);

  __shared__ T *shmem, temp;
  __shared__ LongType ind, blockOffset, lenPerBlock;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char sharedMemory[];
    shmem = reinterpret_cast<T*>(sharedMemory);

    blockOffset = (len * blockIdx.x) >> power;
    lenPerBlock = ((len * (blockIdx.x + 1)) >> power) - blockOffset;
    ind = blockOffset;
  }
  __syncthreads();

  // copy from global memory to shared memory
  if (threadIdx.x < lenPerBlock) shmem[threadIdx.x] = x[(blockOffset + threadIdx.x) * ews];
  __syncthreads();

  // *** apply Fisher-Yates shuffle to lenPerBlock number of elements
  if (threadIdx.x == 0) {
    for (LongType i = lenPerBlock - 1; i > 0; --i) {
      const LongType j = rng->relativeLong(ind++) % (i + 1);
      if (i != j) {
        temp = shmem[i];
        shmem[i] = shmem[j];
        shmem[j] = temp;
      }
    }
  }
  __syncthreads();

  // copy from shared memory to global memory
  if (threadIdx.x < lenPerBlock) x[(blockOffset + threadIdx.x) * ews] = shmem[threadIdx.x];
}

template <typename T>
static SD_KERNEL void mergeShuffleCuda(graph::RandomGenerator* rng, void* vx, const LongType ews,
                                       const LongType len, const int power, const LongType iterNum) {
  T* x = reinterpret_cast<T*>(vx);

  __shared__ LongType ind, blockOffset, factor, beg, mid, totLen, iterExp;

  // *** apply mergeShuffle algorithm
  if (threadIdx.x == 0) {
    factor = blockIdx.x << iterNum;
    iterExp = 1 << (iterNum - 1);
    blockOffset = (len * factor) >> power;
    mid = ((len * (factor + iterExp)) >> power) - blockOffset;  // middle
    totLen = ((len * (factor + 2 * iterExp)) >> power) - blockOffset;
    ind = iterNum * len + blockOffset;
    beg = 0;  // beginning

    while (true) {
      if (rng->relativeLong(ind++) % 2) {
        if (mid == totLen) break;
        int first = (blockOffset + beg) * ews;
        int second = blockOffset + mid * ews;
        if(first >= len || second >= len) {
          break;
        }
        math::sd_swap<T>(x[(blockOffset + beg) * ews], x[(blockOffset + mid++) * ews]);
      } else {
        if (beg == mid) break;
      }
      ++beg;
    }

    // Fisher-Yates
    while (beg < totLen) {
      const LongType e = rng->relativeLong(ind++) % (beg + 1);
      int first = (blockOffset + beg) * ews;
      int second = blockOffset + e * ews;
      if(first >= len || second >= len) {
        break;
      }
      if (beg != e) math::sd_swap<T>(x[(blockOffset + beg) * ews], x[(blockOffset + e) * ews]);
      ++beg;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Fisher-Yates shuffle
template <typename T>
static void fisherYates(graph::RandomGenerator& rng, T* buff, const LongType& len, const LongType& ews, LongType ind) {
  for (LongType i = len - 1; i > 0; --i) {
    const LongType j = rng.relativeLong(ind++) % (i + 1);
    if (i != j) math::sd_swap<T>(buff[i * ews], buff[j * ews]);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void randomShuffle_(LaunchContext* context, NDArray& input, NDArray& output, graph::RandomGenerator& rng,
                           const bool isInplace) {
  const int firstDim = input.sizeAt(0);
  LongType temp;

  if (input.lengthOf() == 1 || firstDim == 1) {
    if (!isInplace) output.assign(input);
  } else if (shape::isCommonVector(input.shapeInfo(), temp)) {
    NDArray* arr = &input;

    if (!isInplace) {
      output.assign(input);
      arr = &output;
    }

    const LongType len = arr->lengthOf();

    const int threadsPerBlock = SD_MAX_NUM_THREADS;

    int power = 0;
    while ((len >> power) > threadsPerBlock) ++power;

    dim3 fisherDims = randomShuffleFisherDims(power,input.sizeOfT());
    const int blocksPerGrid = fisherDims.y;
    const int sharedMem = fisherDims.z;

    PointersManager manager(context, "NDArray::randomShuffle cuda");

    graph::RandomGenerator* pRng = reinterpret_cast<graph::RandomGenerator*>(
        manager.replicatePointer(&rng, sizeof(graph::RandomGenerator)));

    NDArray::prepareSpecialUse({arr}, {arr});

    fisherYatesCuda<T><<<fisherDims.y, fisherDims.x, fisherDims.z, *context->getCudaStream()>>>(
        pRng, arr->specialBuffer(), arr->ews(), len, power);
    sd::DebugHelper::checkErrorCode(context->getCudaStream(), "fisherYatesCuda failed");

    for (LongType j = 1, i = 1; j < blocksPerGrid; j += j, ++i) {
      dim3 mergeShuffleDims = randomShuffleMergeDims(j, power);
      mergeShuffleCuda<T><<<mergeShuffleDims.x, mergeShuffleDims.y, mergeShuffleDims.z, *context->getCudaStream()>>>(
          pRng, arr->specialBuffer(), arr->ews(), len, power, i);
      sd::DebugHelper::checkErrorCode(context->getCudaStream(), "mergeShuffleCuda failed");

      NDArray::registerSpecialUse({arr}, {arr});

      manager.synchronize();

      rng.rewindH((len + 1) * power);
    }
  } else {
    LongType dim = 0;
    auto dimsToExclude = ShapeUtils::evalDimsToExclude(input.rankOf(),1 ,&dim);

    if (isInplace) {
      auto subArrsList = input.allTensorsAlongDimension(*dimsToExclude);

      // Fisher-Yates shuffle
      for (int i = firstDim - 1; i > 0; --i) {
        const int j = rng.relativeInt(i) % (i + 1);
        if (i != j) subArrsList.at(i)->swapUnsafe(*subArrsList.at(j));
      }
    } else {
      auto subArrsListIn = input.allTensorsAlongDimension(*dimsToExclude);
      auto subArrsListOut = output.allTensorsAlongDimension(*dimsToExclude);

      std::vector<int> indices(firstDim);
      std::iota(indices.begin(), indices.end(), 0);  // 0,1,2,3, ... firstDim-1

      // shuffle indices
      fisherYates<int>(rng, indices.data(), firstDim, 1, 0);

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; ++i) subArrsListOut.at(i)->assign(*subArrsListIn.at(indices[i]));
      };

      samediff::Threads::parallel_for(func, 0, firstDim);
    }

    rng.rewindH(firstDim - 1);

    delete dimsToExclude;
  }
}

/////////////////////////////////////////////////////////////////////////
void randomShuffle(LaunchContext* context, NDArray& input, NDArray& output, graph::RandomGenerator& rng,
                   const bool isInplace) {
  BUILD_SINGLE_SELECTOR(input.dataType(), randomShuffle_, (context, input, output, rng, isInplace), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
