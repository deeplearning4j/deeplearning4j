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
      memcpy(outArrs[i]->bufferAsT<T>(), x, memAmountToCopy * sizeofT);
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

/**
 * This kernel accumulates X arrays, and stores result into Z
 *
 * @tparam T
 * @param x
 * @param z
 * @param n
 * @param length
 */

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
sd::LongType SpecialMethods<T>::getPosition(NDArray *input, sd::LongType index) {
  auto xShapeInfo = input->shapeInfo();
  LongType coords[SD_MAX_RANK];
  INDEX2COORDS(index, shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), coords);
  LongType offset;
  COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, offset);
  return offset;
}

template <typename T>
void SpecialMethods<T>::sortGeneric(NDArray *input, bool descending) {
  auto x = input->bufferAsT<T>();
  auto xShapeInfo = input->shapeInfo();
  quickSort_parallel(input, Environment::getInstance().maxMasterThreads(), descending);
}

template <typename T>
void SpecialMethods<T>::sortTadGeneric(NDArray *input, sd::LongType *dimension, int dimensionLength, bool descending) {
  auto x = input->bufferAsT<T>();
  sd::LongType xLength = input->lengthOf();
  sd::LongType xTadLength = shape::tadLength(input->shapeInfo(), dimension, dimensionLength);
  int numTads = xLength / xTadLength;

  const std::vector<sd::LongType> dimVector(dimension, dimension + dimensionLength);
  auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), &dimVector,false);

  auto func = PRAGMA_THREADS_FOR {
    for (auto r = start; r < stop; r++) {
      NDArray *dx = pack->extractTadView(input, r);
      quickSort_parallel(dx,  xTadLength, descending);
      delete dx;
    }
  };
  samediff::Threads::parallel_tad(func, 0, numTads);
}

template <typename T>
void SpecialMethods<T>::decodeBitmapGeneric(NDArray *input, NDArray *output,sd::LongType N) {
  auto dz = output->bufferAsT<T>();
  auto x = input->bufferAsT<int>();
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
sd::LongType SpecialMethods<T>::encodeBitmapGeneric(NDArray *input, sd::LongType N,
                                                    LongType *dz,float threshold) {
  auto dx = input->bufferAsT<T>();
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
      T abs = sd::math::sd_abs<T, T>(val);

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
