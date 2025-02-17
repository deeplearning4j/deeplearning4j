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
// @author raver119@gmail.com, created on 07.10.2017.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <array/NDArray.h>
#include <helpers/Loops.h>

#include <helpers/shape.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/specials.h>
#include <types/types.h>

namespace sd {

/**
 * @brief Checks if the shape of NDArray contains 1 before(order c) or after(order f) the specified axis
 *
 * @param input
 * @param axis
 * @return int
 */
SD_INLINE int isShapeExtendedWithOnes(NDArray&input, LongType axis) {
  bool isAllOne = true;
  auto shapes = shape::shapeOf(input.shapeInfo());
  auto rank = input.rankOf();
  if (rank > axis) {
    if (input.ordering() == 'c') {
      // check before the axis
      for (sd::LongType i = 0; i < axis; i++) {
        isAllOne = isAllOne && (shapes[i] == 1);
      }
    } else {
      // check after the axis
      for (int i = axis + 1; i < rank; i++) {
        isAllOne = isAllOne && (shapes[i] == 1);
      }
    }
    return isAllOne;
  }

  return true;
}

template <typename T>
struct InputArgsCase2 {
  const T *ptr;
  int size;
};

template <typename T>
void SpecialMethods<T>::concatCpuGeneric(const std::vector<NDArray *> &inArrs, NDArray &output,
                                         const LongType axis) {
  const sd::LongType numOfInArrs = inArrs.size();
  const auto sizeofT = output.sizeOfT();

  T *zBuff = output.bufferAsT<T>();

  bool shapeExtendedWithOnes = isShapeExtendedWithOnes(output, axis);
  bool followEws1 = false;
  bool matchesOutputOrdering = true;
  for (int i = 0; i < numOfInArrs; ++i) {
    shapeExtendedWithOnes = shapeExtendedWithOnes && isShapeExtendedWithOnes(*inArrs[i], axis);
    matchesOutputOrdering = matchesOutputOrdering && inArrs[i]->ordering() == output.ordering();
  }

  bool copyCaseEws1 = followEws1 & matchesOutputOrdering;
  bool copyCase1 = numOfInArrs > 1 ? copyCaseEws1 & shapeExtendedWithOnes : copyCaseEws1;

  if (copyCase1) {
    // copyCase1:
    // When NdArrays follow the same order and unit elementwise stride and
    // the concantneation axis is 0th or has only 1 before it {1, 1, ..., axis} for "c"
    // or axis is (rank-1)th or has only 1 after it {axis, 1, 1, ..., 1} for "f"
    // we will concatenate them by sequential copying of the whole buffers

    std::vector<T *> zPtrList;
    T *z = output.bufferAsT<T>();
    for (sd::LongType i = 0; i < numOfInArrs; i++) {
      zPtrList.push_back(z);
      z += inArrs[i]->lengthOf();
    }
    auto func = [&inArrs, &zPtrList](sd::LongType thread_id, sd::LongType start, sd::LongType stop,
                                     sd::LongType increment) -> void {
      for (sd::LongType i = start; i < stop; ++i) {
        const auto memAmountToCopy = inArrs[i]->lengthOf();
        const auto inputPtr = inArrs[i]->bufferAsT<T>();

        auto zPtr = zPtrList[i];
        for (int j = 0; j < memAmountToCopy; j++) {
          zPtr[j] = inputPtr[j];
        }
      }
    };

    samediff::Threads::parallel_tad(func, 0, numOfInArrs, 1);
    return;
  }

  // for one Array
  if (numOfInArrs < 2) {
    output.assign(*inArrs[0]);
    return;
  }
  bool copyCase2 = copyCaseEws1 && output.ordering() == 'c';
  if (copyCase2) {
    sd::LongType times = 1;
    auto shapes = shape::shapeOf(output.shapeInfo());

    T *z = output.bufferAsT<T>();
    for (int i = 0; i < axis; i++) {
      times = times * shapes[i];
    }

    sd::LongType totalCopySize = output.lengthOf() / times;

    std::vector<InputArgsCase2<T>> inputArgs;
    for (sd::LongType i = 0; i < numOfInArrs; i++) {
      InputArgsCase2<T> input = {inArrs[i]->bufferAsT<T>(),
                                 static_cast<int>(inArrs[i]->lengthOf()) / static_cast<int>(times)};
      inputArgs.push_back(input);
    }

    auto func = [&inputArgs, z, totalCopySize](uint64_t thread_id, int64_t start, int64_t stop,
                                               int64_t increment) -> void {
      auto outPtr = &(z[start * totalCopySize]);
      auto numOfInArrs = inputArgs.size();
      for (int i = start; i < stop; i++) {
        for (size_t j = 0; j < numOfInArrs; j++) {
          auto inputCopySize = inputArgs[j].size;
          const T *inputBasePtr = inputArgs[j].ptr;
          auto inputPtr = &(inputBasePtr[i * inputCopySize]);
          // copy
          PRAGMA_OMP_SIMD
          for (int k = 0; k < inputCopySize; k++) {
            outPtr[k] = inputPtr[k];
          }
          outPtr += inputCopySize;
        }
      }
    };
    samediff::Threads::parallel_tad(func, 0, times, 1);
    return;
  }

  // Cache shape and stride information for output
  const sd::LongType zRank = shape::rank(output.shapeInfo());
  const sd::LongType* zShape = shape::shapeOf(output.shapeInfo());
  const sd::LongType* zStride = shape::stride(output.shapeInfo());

  // Pre-cache input arrays' shape information
  std::vector<const sd::LongType*> inShapes(numOfInArrs);
  std::vector<const sd::LongType*> inStrides(numOfInArrs);
  std::vector<sd::LongType> inRanks(numOfInArrs);

  for (sd::LongType i = 0; i < numOfInArrs; i++) {
    inRanks[i] = shape::rank(inArrs[i]->shapeInfo());
    inShapes[i] = shape::shapeOf(inArrs[i]->shapeInfo());
    inStrides[i] = shape::stride(inArrs[i]->shapeInfo());
  }

  // general case
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i += increment) {
      INDEX2COORDS(i, zRank, zShape, coords);

      sd::LongType zOffset;
      COORDS2INDEX(zRank, zStride, coords, zOffset);

      sd::LongType inArrIdx = 0;
      sd::LongType xDim = inArrs[inArrIdx]->sizeAt(axis);

      temp = coords[axis];
      while (coords[axis] >= xDim) {
        coords[axis] -= xDim;
        xDim = inArrs[++inArrIdx]->sizeAt(axis);
      }

      const T *x = inArrs[inArrIdx]->bufferAsT<T>();
      sd::LongType xOffset;
      COORDS2INDEX(inRanks[inArrIdx], inStrides[inArrIdx], coords, xOffset);

      zBuff[zOffset] = x[xOffset];

      coords[axis] = temp;
    }
  };

  samediff::Threads::parallel_for(func, 0, output.lengthOf());
}
/**
 * Concatneate multi array of the same shape together
 * along a particular dimension
 */
template <typename T>
void SpecialMethods<T>::concatCpuGeneric(LongType dimension, int numArrays,NDArray **data,
                                         NDArray *vresult) {
  auto result = reinterpret_cast<T *>(vresult);
  std::vector<NDArray *> inputs(numArrays);


  for (sd::LongType i = 0; i < numArrays; ++i)
    inputs[i] =
        new NDArray(static_cast<void *>(data[i]), data[i]->shapeInfo(), nullptr, false, 0);

  sd::SpecialMethods<T>::concatCpuGeneric(inputs, *vresult, dimension);

  for (sd::LongType i = 0; i < numArrays; ++i) {
    delete inputs[i];
  }
}

template <typename T>
void SpecialMethods<T>::splitCpuGeneric(NDArray& input, const std::vector<NDArray*>& outArrs, const LongType axis) {
  int numSplits = outArrs.size();
  const auto sizeofT = input.sizeOfT();
  auto xBuff = input.bufferAsT<T>();

  bool luckCase1 = ((axis == 0 && input.ordering() == 'c') || (axis == input.rankOf() - 1 && input.ordering() == 'f'));

  if (luckCase1) {
    T* x = const_cast<T*>(xBuff);
    for (sd::LongType i = 0; i < numSplits; ++i) {
      const auto memAmountToCopy = outArrs[i]->lengthOf();
      ops::safe_copy(x, static_cast<const T*>(outArrs[i]->buffer()), static_cast<size_t>(memAmountToCopy));
      x += memAmountToCopy;
    }
    return;
  }

  // Cache shape and stride information
  const sd::LongType xRank = shape::rank(input.shapeInfo());
  const sd::LongType* xShape = shape::shapeOf(input.shapeInfo());
  const sd::LongType* xStride = shape::stride(input.shapeInfo());

  // Pre-cache output array ranks, shapes, and strides
  std::vector<const sd::LongType*> outShapes(numSplits);
  std::vector<const sd::LongType*> outStrides(numSplits);
  std::vector<sd::LongType> outRanks(numSplits);

  for (int i = 0; i < numSplits; i++) {
    outRanks[i] = shape::rank(outArrs[i]->shapeInfo());
    outShapes[i] = shape::shapeOf(outArrs[i]->shapeInfo());
    outStrides[i] = shape::stride(outArrs[i]->shapeInfo());
  }

  sd::LongType zDim = outArrs[0]->sizeAt(axis);

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i += increment) {
      INDEX2COORDS(i, xRank, xShape, coords);
      sd::LongType xOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);

      sd::LongType outArrIdx = 0;
      temp = coords[axis];

      while (coords[axis] >= zDim) {
        coords[axis] -= zDim;
        ++outArrIdx;
      }

      T* z = outArrs[outArrIdx]->bufferAsT<T>();
      sd::LongType zOffset;
      COORDS2INDEX(outRanks[outArrIdx], outStrides[outArrIdx], coords, zOffset);
      z[zOffset] = xBuff[xOffset];

      coords[axis] = temp;
    }
  };

  samediff::Threads::parallel_for(func, 0, input.lengthOf());
}

template <typename T>
void SpecialMethods<T>::sortGeneric(NDArray *input, bool descending) {
  auto x = input->bufferAsT<T>();
  auto xShapeInfo = input->shapeInfo();
  quickSort_parallel(input, Environment::getInstance().maxMasterThreads(), descending);
}

template <typename T>
void SpecialMethods<T>::quickSort_parallel_internal(NDArray *x, int left, int right, int cutoff, bool descending) {
  if (right - left <= cutoff) {
    // Use insertion sort for small arrays
    auto xBuff = x->bufferAsT<T>();
    for (int i = left + 1; i <= right; i++) {
      T key = xBuff[i];
      int j = i - 1;
      if (descending) {
        while (j >= left && xBuff[j] < key) {
          xBuff[j + 1] = xBuff[j];
          j--;
        }
      } else {
        while (j >= left && xBuff[j] > key) {
          xBuff[j + 1] = xBuff[j];
          j--;
        }
      }
      xBuff[j + 1] = key;
    }
    return;
  }

  // Choose pivot as median of three
  auto xBuff = x->bufferAsT<T>();
  int mid = (left + right) / 2;
  if (descending) {
    if (xBuff[right] > xBuff[left]) std::swap(xBuff[right], xBuff[left]);
    if (xBuff[mid] > xBuff[left]) std::swap(xBuff[mid], xBuff[left]);
    if (xBuff[right] > xBuff[mid]) std::swap(xBuff[right], xBuff[mid]);
  } else {
    if (xBuff[right] < xBuff[left]) std::swap(xBuff[right], xBuff[left]);
    if (xBuff[mid] < xBuff[left]) std::swap(xBuff[mid], xBuff[left]);
    if (xBuff[right] < xBuff[mid]) std::swap(xBuff[right], xBuff[mid]);
  }

  // Partition
  T pivot = xBuff[mid];
  int i = left;
  int j = right;

  while (i <= j) {
    if (descending) {
      while (xBuff[i] > pivot) i++;
      while (xBuff[j] < pivot) j--;
    } else {
      while (xBuff[i] < pivot) i++;
      while (xBuff[j] > pivot) j--;
    }

    if (i <= j) {
      std::swap(xBuff[i], xBuff[j]);
      i++;
      j--;
    }
  }

  // Recursively sort sub-arrays
  if (left < j) quickSort_parallel_internal(x, left, j, cutoff, descending);
  if (i < right) quickSort_parallel_internal(x, i, right, cutoff, descending);
}

template <typename T>
void SpecialMethods<T>::quickSort_parallel(NDArray *x, int numThreads, bool descending) {
  const int CUTOFF = 32;  // Threshold for switching to insertion sort
  auto length = x->lengthOf();

  if (length <= 1) return;

  // For very small arrays, just use the internal sort
  if (length <= CUTOFF || numThreads <= 1) {
    quickSort_parallel_internal(x, 0, length - 1, CUTOFF, descending);
    return;
  }

  // For larger arrays, partition into segments and sort in parallel
  int segmentSize = length / numThreads;
  auto func = PRAGMA_THREADS_FOR {
    int threadLeft = start * segmentSize;
    int threadRight = (start == numThreads - 1) ? length - 1 : (start + 1) * segmentSize - 1;
    quickSort_parallel_internal(x, threadLeft, threadRight, CUTOFF, descending);
  };

  samediff::Threads::parallel_for(func, 0, numThreads);

  // Merge sorted segments if we used multiple threads
  if (numThreads > 1) {
    auto xBuff = x->bufferAsT<T>();
    std::vector<T> temp(length);
    for (int size = segmentSize; size < length; size *= 2) {
      for (int left = 0; left < length; left += 2 * size) {
        int mid = std::min(left + size, (int)length);
        int right = std::min(left + 2 * size, (int)length);
        int i = left, j = mid, k = left;

        // Merge two segments
        while (i < mid && j < right) {
          if (descending) {
            temp[k++] = (xBuff[i] >= xBuff[j]) ? xBuff[i++] : xBuff[j++];
          } else {
            temp[k++] = (xBuff[i] <= xBuff[j]) ? xBuff[i++] : xBuff[j++];
          }
        }
        while (i < mid) temp[k++] = xBuff[i++];
        while (j < right) temp[k++] = xBuff[j++];

        // Copy back
        for (i = left; i < right; i++) {
          xBuff[i] = temp[i];
        }
      }
    }
  }
}

template <typename T>
void SpecialMethods<T>::sortTadGeneric(NDArray *input, sd::LongType *dimension, int dimensionLength, bool descending) {
  auto x = input->bufferAsT<T>();
  sd::LongType xLength = input->lengthOf();
  sd::LongType xTadLength = shape::tadLength(input->shapeInfo(), dimension, dimensionLength);
  int numTads = xLength / xTadLength;

  const std::vector<sd::LongType> dimVector(dimension, dimension + dimensionLength);
  auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType *>(input->shapeInfo()),
                                                                    const_cast<sd::LongType *>(dimVector.data()),false);

  auto func = PRAGMA_THREADS_FOR {
    for (auto r = start; r < stop; r++) {
      NDArray *dx = pack->extractTadView(input, r);
      quickSort_parallel(dx,  xTadLength, descending);
      delete dx;
    }
  };
  samediff::Threads::parallel_tad(func, 0, numTads);
}

}  // namespace sd
