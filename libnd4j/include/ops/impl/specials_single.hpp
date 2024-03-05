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
#include <helpers/TAD.h>
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
SD_INLINE int isShapeExtendedWithOnes(const NDArray &input, LongType axis) {
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
void SpecialMethods<T>::concatCpuGeneric(const std::vector<const NDArray *> &inArrs, NDArray &output,
                                         const LongType axis) {
  const sd::LongType numOfInArrs = inArrs.size();
  const auto sizeofT = output.sizeOfT();

  T *zBuff = output.bufferAsT<T>();

  bool shapeExtendedWithOnes = isShapeExtendedWithOnes(output, axis);
  bool followEws1 = output.ews() == 1;
  bool matchesOutputOrdering = true;
  for (int i = 0; i < numOfInArrs; ++i) {
    shapeExtendedWithOnes = shapeExtendedWithOnes && isShapeExtendedWithOnes(*inArrs[i], axis);
    followEws1 = followEws1 && inArrs[i]->ews() == 1;
    matchesOutputOrdering = matchesOutputOrdering && inArrs[i]->ordering() == output.ordering();
  }

  bool copyCaseEws1 = followEws1 & matchesOutputOrdering;
  bool copyCase1 = numOfInArrs > 1 ? copyCaseEws1 & shapeExtendedWithOnes : copyCaseEws1;

  if (copyCase1) {
    printf("concat copy case 1\n");
    // copyCase1:
    // in this case:
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
    auto func = [&inArrs, &zPtrList](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) -> void {
      for (sd::LongType i = start; i < stop; ++i) {
        const auto memAmountToCopy = inArrs[i]->lengthOf();
        const auto inputPtr = inArrs[i]->bufferAsT<T>();
#if defined(__NEC__)
        auto zPtr = zPtrList[i];
        for (int j = 0; j < memAmountToCopy; j++) {
          zPtr[j] = inputPtr[j];
        }
#else
          auto zPtr = zPtrList[i];
                for (int j = 0; j < memAmountToCopy; j++) {
                  zPtr[j] = inputPtr[j];
                }
#endif
      }
    };

    samediff::Threads::parallel_tad(func, 0, numOfInArrs, 1);
    return;
  }

  // for one Array
  if (numOfInArrs < 2) {
    output.assign(inArrs[0]);
    return;
  }
  bool copyCase2 = copyCaseEws1 && output.ordering() == 'c';
  if (copyCase2) {
    printf("concat copy case 2\n");
    // copyCase2:
    // in this case:
    // when NDArrays follow the same order (here it is done for the "c" "the last index is fast" order)
    // and unit elementwise stride.
    // we will just concatenate by iterating over t=shape[0]*...*shape[axis-1] times
    // and copying all buffers with {offset: t * copy_size_i, size: copy_size_i} into output buffer with { offset: t*
    // total_copy_size, total copy size} where: copy_size_i = shape[axis] * .. * shape[rank-1] of the iTh input array
    // total copy size is sum of all {copy_size_i}

    sd::LongType times = 1;
    auto shapes = shape::shapeOf(output.shapeInfo());

    T *z = output.bufferAsT<T>();
    for (int i = 0; i < axis; i++) {
      times = times * shapes[i];
    }

    sd::LongType totalCopySize = output.lengthOf() / times;

    std::vector<InputArgsCase2<T>> inputArgs;
    for (sd::LongType i = 0; i < numOfInArrs; i++) {
      InputArgsCase2<T> input = {inArrs[i]->bufferAsT<T>(), static_cast<sd::LongType>(inArrs[i]->lengthOf()) / static_cast<sd::LongType>(times)};
      inputArgs.push_back(input);
    }

    auto func = [&inputArgs, z, totalCopySize](uint64_t thread_id, int64_t start, int64_t stop,
                                               int64_t increment) -> void {
      auto outPtr = &(z[start * totalCopySize]);
      auto numOfInArrs = inputArgs.size();
      for (int i = start; i < stop; i++) {
        for (int j = 0; j < numOfInArrs; j++) {
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

  printf("concat general case\n");
  // TODO: optimize the other cases to be NEC friendly as well
  // general case
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i += increment) {
      shape::index2coordsCPU(start, i, output.shapeInfo(), coords);

      const auto zOffset = shape::getOffset(output.shapeInfo(), coords);

      sd::LongType inArrIdx = 0;
      sd::LongType xDim = inArrs[inArrIdx]->sizeAt(axis);

      temp = coords[axis];
      while (coords[axis] >= xDim) {
        coords[axis] -= xDim;
        xDim = inArrs[++inArrIdx]->sizeAt(axis);
      }

      const T *x = inArrs[inArrIdx]->bufferAsT<T>();
      const auto xOffset = shape::getOffset(inArrs[inArrIdx]->shapeInfo(), coords);

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
void SpecialMethods<T>::concatCpuGeneric(LongType dimension, int numArrays, sd::Pointer *data, sd::Pointer *inputShapeInfo,
                                         void *vresult, sd::LongType const *resultShapeInfo) {
  auto result = reinterpret_cast<T *>(vresult);
  std::vector<const NDArray *> inputs(numArrays);

  NDArray output(static_cast<void *>(result), resultShapeInfo);

  for (sd::LongType i = 0; i < numArrays; ++i)
    inputs[i] = new NDArray(static_cast<void *>(data[i]), static_cast<sd::LongType *>(inputShapeInfo[i]));

  sd::SpecialMethods<T>::concatCpuGeneric(inputs, output, dimension);

  for (sd::LongType i = 0; i < numArrays; ++i) delete inputs[i];
}

template <typename T>
void SpecialMethods<T>::splitCpuGeneric(const NDArray &input, const std::vector<NDArray *> &outArrs,
                                        const LongType axis) {
  int numSplits = outArrs.size();

  const auto sizeofT = input.sizeOfT();

  auto xBuff = input.bufferAsT<T>();

  bool luckCase1 =
      ((axis == 0 && input.ordering() == 'c') || (axis == input.rankOf() - 1 && input.ordering() == 'f')) &&
      input.ews() == 1;

  if (luckCase1) {
    for (sd::LongType i = 0; i < numSplits; ++i) {
      luckCase1 &= outArrs[i]->ordering() == input.ordering() && outArrs[i]->ews() == 1;
      if (!luckCase1) break;
    }
  }

  if (luckCase1) {
    T *x = const_cast<T *>(xBuff);
    for (sd::LongType i = 0; i < numSplits; ++i) {
      const auto memAmountToCopy = outArrs[i]->lengthOf();
      memcpy(outArrs[i]->bufferAsT<T>(), x, memAmountToCopy * sizeofT);
      x += memAmountToCopy;
    }
    return;
  }

  sd::LongType zDim = outArrs[0]->sizeAt(axis);
  // general case

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i += increment) {
      shape::index2coordsCPU(start, i, input.shapeInfo(), coords);
      const auto xOffset = shape::getOffset(input.shapeInfo(), coords);

      sd::LongType outArrIdx = 0;
      temp = coords[axis];

      while (coords[axis] >= zDim) {
        coords[axis] -= zDim;
        ++outArrIdx;
      }

      T *z = outArrs[outArrIdx]->bufferAsT<T>();
      const auto zOffset = shape::getOffset(outArrs[outArrIdx]->shapeInfo(), coords);
      z[zOffset] = xBuff[xOffset];

      coords[axis] = temp;
    }
  };

  samediff::Threads::parallel_for(func, 0, input.lengthOf());
}

/**
 * This kernel accumulates X arrays, and stores result into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */
template <typename T>
void SpecialMethods<T>::accumulateGeneric(void **vx, void *vz, sd::LongType const *zShapeInfo, int n,
                                          const sd::LongType length) {
  auto z = reinterpret_cast<T *>(vz);
  auto x = reinterpret_cast<T **>(vx);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      for (auto ar = 0L; ar < n; ar++) {
        z[i] += x[ar][i];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

/**
 * This kernel averages X input arrays, and stores result to Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 * @param propagate
 */
template <typename T>
void SpecialMethods<T>::averageGeneric(void **vx, void *vz, sd::LongType const *zShapeInfo, int n,
                                       const sd::LongType length, bool propagate) {
  auto z = reinterpret_cast<T *>(vz);
  auto x = reinterpret_cast<T **>(vx);

  if (z == nullptr) {
    // code branch for absent Z
    z = x[0];

    PRAGMA_OMP_SIMD
    for (sd::LongType i = 0; i < length; i++) {
      z[i] /= static_cast<T>(n);
    }

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        for (sd::LongType ar = 1; ar < n; ar++) {
          z[i] += x[ar][i] / static_cast<T>(n);
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, length);

    // instead of doing element-wise propagation, we just issue memcpy to propagate data
    for (sd::LongType ar = 1; ar < n; ar++) {
      memcpy(x[ar], z, length * sizeof(T));
    }
  } else {
    // code branch for existing Z

    // memset before propagation
    memset(z, 0, length * sizeof(T));

    // aggregation step
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        for (sd::LongType ar = 0; ar < n; ar++) {
          z[i] += x[ar][i] / static_cast<T>(n);
        }
      }
    };
    samediff::Threads::parallel_for(func, 0, length);

    // instead of doing element-wise propagation, we just issue memcpy to propagate data
    for (sd::LongType ar = 0; ar < n; ar++) {
      memcpy(x[ar], z, length * sizeof(T));
    }
  }
}

template <typename T>
sd::LongType SpecialMethods<T>::getPosition(sd::LongType const *xShapeInfo, sd::LongType index) {
  LongType xEWS = shape::elementWiseStride(xShapeInfo);

  if (xEWS == 1)
    return index;
  else if (xEWS > 1)
    return index * xEWS;
  else
    return shape::getIndexOffset(index, xShapeInfo);
}

template <typename T>
void SpecialMethods<T>::quickSort_parallel_internal(T *array, sd::LongType const *xShapeInfo, int left, int right,
                                                    int cutoff, bool descending) {
  int i = left, j = right;
  T tmp;
  T pivot = array[getPosition(xShapeInfo, (left + right) / 2)];

  {
    /* PARTITION PART */
    while (i <= j) {
      if (descending) {
        while (array[getPosition(xShapeInfo, i)] > pivot) i++;
        while (array[getPosition(xShapeInfo, j)] < pivot) j--;
        if (i <= j) {
          tmp = array[getPosition(xShapeInfo, i)];
          array[getPosition(xShapeInfo, i)] = array[getPosition(xShapeInfo, j)];
          array[getPosition(xShapeInfo, j)] = tmp;
          i++;
          j--;
        }
      } else {
        while (array[getPosition(xShapeInfo, i)] < pivot) i++;
        while (array[getPosition(xShapeInfo, j)] > pivot) j--;
        if (i <= j) {
          tmp = array[getPosition(xShapeInfo, i)];
          array[getPosition(xShapeInfo, i)] = array[getPosition(xShapeInfo, j)];
          array[getPosition(xShapeInfo, j)] = tmp;
          i++;
          j--;
        }
      }
    }
  }

  //

  if (((right - left) < cutoff)) {
    if (left < j) {
      quickSort_parallel_internal(array, xShapeInfo, left, j, cutoff, descending);
    }
    if (i < right) {
      quickSort_parallel_internal(array, xShapeInfo, i, right, cutoff, descending);
    }

  } else {
    PRAGMA_OMP_TASK { quickSort_parallel_internal(array, xShapeInfo, left, j, cutoff, descending); }
    PRAGMA_OMP_TASK { quickSort_parallel_internal(array, xShapeInfo, i, right, cutoff, descending); }
  }
}

template <typename T>
void SpecialMethods<T>::quickSort_parallel(void *varray, sd::LongType const *xShapeInfo, sd::LongType lenArray,
                                           int numThreads, bool descending) {
  auto array = reinterpret_cast<T *>(varray);
  int cutoff = 1000;

  PRAGMA_OMP_PARALLEL_THREADS(numThreads) {
    PRAGMA_OMP_SINGLE_ARGS(nowait) {
      quickSort_parallel_internal(array, xShapeInfo, 0, lenArray - 1, cutoff, descending);
    }
  }
}

template <typename T>
int SpecialMethods<T>::nextPowerOf2(int number) {
  int pos = 0;

  while (number > 0) {
    pos++;
    number = number >> 1;
  }
  return (int)pow(2, pos);
}

template <typename T>
int SpecialMethods<T>::lastPowerOf2(int number) {
  int p = 1;
  while (p <= number) p <<= 1;

  p >>= 1;
  return p;
}

template <typename T>
void SpecialMethods<T>::sortGeneric(void *vx, sd::LongType const *xShapeInfo, bool descending) {
  auto x = reinterpret_cast<T *>(vx);

  quickSort_parallel(x, xShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
}

template <typename T>
void SpecialMethods<T>::sortTadGeneric(void *vx, sd::LongType const *xShapeInfo, LongType *dimension, int dimensionLength,
                                       sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets,
                                       bool descending) {
  auto x = reinterpret_cast<T *>(vx);

  // quickSort_parallel(x, xShapeInfo, shape::length(xShapeInfo), omp_get_max_threads(), descending);
  sd::LongType xLength = shape::length(xShapeInfo);
  sd::LongType xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
  int numTads = xLength / xTadLength;

  auto func = PRAGMA_THREADS_FOR {
    for (auto r = start; r < stop; r++) {
      T *dx = x + tadOffsets[r];

      quickSort_parallel(dx, tadShapeInfo, xTadLength, 1, descending);
    }
  };
  samediff::Threads::parallel_tad(func, 0, numTads);
}

template <typename T>
void SpecialMethods<T>::decodeBitmapGeneric(const void *dx, sd::LongType N, void *vz, sd::LongType const *zShapeInfo) {
  auto dz = reinterpret_cast<T *>(vz);
  auto x = reinterpret_cast<const int *>(dx);
  sd::LongType lim = N / 16 + 5;

  FloatBits2 fb;
  fb.i_ = x[2];
  float threshold = fb.f_;

  auto pPos = -1;

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      const auto v = x[e];
      for (int bitId = 0; bitId < 16; bitId++) {
        bool hasBit = (v & 1 << (bitId)) != 0;
        bool hasSign = (v & 1 << (bitId + 16)) != 0;
        auto cPos = (e - 4) * 16 + bitId;

        if (hasBit) {
          if (hasSign)
            dz[cPos] -= static_cast<T>(threshold);
          else
            dz[cPos] += static_cast<T>(threshold);
        } else if (hasSign) {
          dz[cPos] -= static_cast<T>(threshold / 2);
        }

        pPos = cPos;
      }
    }
  };

  samediff::Threads::parallel_for(func, 4, lim);
}

template <typename T>
sd::LongType SpecialMethods<T>::encodeBitmapGeneric(void *vx, sd::LongType const *xShapeInfo, sd::LongType N,
                                                    LongType *dz,
                                                    float threshold) {
  auto dx = reinterpret_cast<T *>(vx);
  const T two(2.0f);
  const T zero(0.0f);
  const T t(threshold);
  const T thalf = t / two;

  sd::LongType retVal = 0L;

  PRAGMA_OMP_PARALLEL_FOR_REDUCTION(+ : retVal)
  for (auto x = 0; x < N; x += 16) {
    int byte = 0;
    int byteId = x / 16 + 4;

    for (int f = 0; f < 16; f++) {
      auto e = x + f;

      if (e >= N) continue;

      T val = dx[e];
      T abs = sd::math::sd_abs<T>(val);

      int bitId = e % 16;

      if (abs >= t) {
        byte |= 1 << (bitId);
        retVal++;

        if (val < zero) {
          byte |= 1 << (bitId + 16);
          dx[e] += t;
        } else {
          dx[e] -= t;
        }
      } else if (abs >= thalf && val < zero) {
        byte |= 1 << (bitId + 16);
        dx[e] += thalf;

        retVal++;
      }
    }

    dz[byteId] = byte;
  }

  return retVal;
}
}  // namespace sd
