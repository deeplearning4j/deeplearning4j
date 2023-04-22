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
// Created by raver119 on 07.10.2017.
//
#include <helpers/shape.h>

namespace shape {




SD_HOST sd::LongType *computeResultShape(sd::LongType const *originalShapeBuffer, long long int *dimension,
                                         long long int dimensionLength) {
  sd::LongType *retShape;
  int retShapeLength;
  if (dimensionLength == 1 && dimension[0] == 2147483647) {
    retShape = new sd::LongType[2];
    retShape[0] = 1;
    retShape[1] = 1;
    retShapeLength = 2;
  } else {
    retShape = shape::removeIndex<sd::LongType, sd::LongType>(shape::shapeOf(originalShapeBuffer), dimension,
                                                     shape::shapeInfoLength(shape::rank(originalShapeBuffer)),
                                                     dimensionLength);
    retShapeLength = shape::rank(originalShapeBuffer) - dimensionLength;
  }
  // ensure vector is proper shape
  if (retShapeLength == 1) {
    if (dimension[0] == 0) {
      auto newRetShape = new sd::LongType[2]{1, retShape[0]};
      delete[] retShape;
      retShape = newRetShape;
      retShapeLength = 2;
    } else {
      auto newRetShape = new sd::LongType[2]{retShape[0], 1};
      delete[] retShape;
      retShape = newRetShape;
      retShapeLength = 2;
    }
  } else if (retShapeLength == 0) {
    auto newRetShape = new sd::LongType[2]{1, 1};
    delete[] retShape;
    retShape = newRetShape;
    retShapeLength = 2;
  }

  auto ret = shape::shapeBuffer(retShapeLength, sd::ArrayOptions::dataType(originalShapeBuffer), retShape);
  delete[] retShape;

  return ret;
}

SD_HOST sd::LongType *shapeInfoOnlyShapeAndStride(const sd::LongType *shapeInfo,
                                                  sd::LongType *dimension,
                                                  long long int dimensionLength,
                                                  bool reverseCopyStride, sd::LongType *buffer) {
  sd::LongType *theShape = shape::shapeOf(shapeInfo);
  sd::LongType *theStride = shape::stride(shapeInfo);
  int rank = dimensionLength == 1 ? 2 : dimensionLength;
  sd::LongType *ret = buffer;
  // set the rank
  ret[0] = rank;
  sd::LongType *retShape = shape::shapeOf(ret);
  sd::LongType *retStride = shape::stride(ret);
  int len = rank;

  if (dimensionLength == 1) {
    if (shape::isMatrix(theShape, shape::rank(shapeInfo))) {
      if (dimension[0] == 0) {
        sd::LongType newStride[2] = {theStride[dimension[0]], 1};
        sd::LongType newShape[2] = {theShape[dimension[0]], 1};
        retShape[0] = newShape[0];
        retShape[1] = newShape[1];
        retStride[0] = newStride[0];
        retStride[1] = newStride[1];
      } else {
        sd::LongType newStride[2] = {theStride[dimension[0]], 1};
        sd::LongType newShape[2] = {theShape[dimension[0]], 1};
        retShape[0] = newShape[0];
        retShape[1] = newShape[1];
        retStride[0] = newStride[0];
        retStride[1] = newStride[1];
      }
    } else {
      sd::LongType newStride[2] = {1, theStride[dimension[0]]};
      sd::LongType newShape[2] = {1, theShape[dimension[0]]};
      retShape[0] = newShape[0];
      retShape[1] = newShape[1];
      retStride[0] = newStride[0];
      retStride[1] = newStride[1];
    }

  } else {
    sd::LongType *newIndexes = dimension;
    if (reverseCopyStride)
      shape::reverseCopyTo(theStride, retStride, newIndexes, len);
    else
      shape::copyTo(len, theStride, retStride, newIndexes);
    shape::copyTo(len, theShape, retShape, newIndexes);
  }

  ret[shape::shapeInfoLength(rank) - 1] = shape::order(shapeInfo);
  return ret;
}

SD_HOST sd::LongType *shapeInfoOnlyShapeAndStride(const sd::LongType *shapeInfo,
                                                  sd::LongType *dimension,
                                                  long long int dimensionLength,
                                                  bool reverseCopyStride) {
  int rank = dimensionLength == 1 ? 2 : dimensionLength;

  traceNew(4);

  sd::LongType *ret = new sd::LongType[shape::shapeInfoLength(rank)];
  return shapeInfoOnlyShapeAndStride(shapeInfo, dimension, dimensionLength, reverseCopyStride, ret);
}

SD_HOST sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, int rank) {
  traceNew(5);

  sd::LongType *ret = new sd::LongType[shape::shapeInfoLength(rank)];

  return createShapeInfo(shape, stride, rank, ret);
}

SD_HOST sd::LongType *createShapeInfo(sd::LongType *shape, sd::LongType *stride, int rank,
                                      sd::LongType *buffer) {
  buffer[0] = rank;
  sd::LongType *retShape = shape::shapeOf(buffer);
  sd::LongType *retStride = shape::stride(buffer);
  for (int i = 0; i < rank; i++) {
    retShape[i] = shape[i];
    retStride[i] = stride[i];
  }

  return buffer;
}


#ifndef SD_CUDA

/**
 * Length of a tad given
 * the shape information
 */
SD_LIB_EXPORT SD_HOST sd::LongType tadLength(const sd::LongType *shapeInfo, sd::LongType *dimension,
                                             long long int dimensionLength) {

  if(shapeInfo == nullptr || dimension == nullptr) {
    std::string  errorMessage;
    errorMessage += "shape info null: %d";
    errorMessage += std::to_string(shapeInfo == nullptr);
    errorMessage += " dimension null: %d";
    errorMessage += std::to_string(dimension == nullptr);
    throw std::runtime_error(errorMessage.c_str());
  }

  if(dimensionLength == 0)
    return 0;

  if(shapeInfo[0] > SD_MAX_RANK || shapeInfo[0] < 0)
    throw std::runtime_error("Corrupt shape information found. Potentially dellocated?");



  if (dimensionLength == 1) {
    if(dimension[0] > SD_MAX_RANK || dimension[0] < 0)
      throw std::runtime_error("Corrupt dimension information found. Potentially dellocated?");

    return shape::shapeOf(shapeInfo)[dimension[0]];
  } else {
    sd::LongType ret = 1;
    for (int i = 0; i < shape::rank(shapeInfo); i++) {
      for (int j = 0; j < dimensionLength; j++) {
        if (i == dimension[j]) ret *= shape::shapeOf(shapeInfo)[dimension[j]];
      }
    }

    return ret;
  }
}


SD_LIB_EXPORT  SD_HOST sd::LongType tadElementWiseStride(sd::LongType *shapeInfo, sd::LongType *dimension,
                                               sd::LongType dimensionLength) {
  return reductionIndexElementWiseStride(shapeInfo, dimension, dimensionLength);
}


SD_LIB_EXPORT SD_HOST bool isEmpty(const sd::LongType *shapeInfo) {
  return ((shape::extra(shapeInfo) & ARRAY_EMPTY) == ARRAY_EMPTY);
}


SD_LIB_EXPORT SD_HOST bool strideDescendingCAscendingF(const sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  sd::LongType *strides = shape::stride(const_cast<sd::LongType *>(shapeBuffer));
  char order = shape::order(shapeBuffer);

  if (shape::isRowVector(shapeBuffer) && strides[0] == 1 && strides[1] == 1) return true;

  if (order == 'c') {
    for (int i = 1; i < rank; i++)
      if (strides[i - 1] <= strides[i]) return false;
    return true;
  } else if (order == 'f') {
    for (int i = 1; i < rank; i++)
      if (strides[i - 1] >= strides[i]) return false;
    return true;
  } else {
    printf("Unknown order for array!\n");
    return false;
  }
}

// max array is outer for min array, min array is sub-array of max array
// function calculates the coordinates of min array (and saves them into minIdxs) given coordinates of max array
// (already stored in maxIdxs)
SD_LIB_EXPORT SD_HOST void maxIndToMinInd(long long int *maxIdxs, long long int *minIdxs, const sd::LongType *maxShapeInfo,
                                          const sd::LongType *minShapeInfo,
                                          const long long int *dimsToExclude, long long int dimsLen) {
  const auto maxRank = shape::rank(maxShapeInfo);
  const auto minRank = shape::rank(minShapeInfo);



  if (dimsLen == -1) dimsLen = maxRank - minRank;  // if size is not given (= -1) then it is equal to ranks difference

  if (maxRank == minRank) {
    if (dimsToExclude == nullptr) {  // --> means dimsToExclude == {0,1,2,...,dimsLen-1}

      for (int i = 0; i < maxRank; ++i) {
        if (i < dimsLen)
          minIdxs[i] = maxIdxs[i];
        else {
          if (maxIdxs[i] > minShapeInfo[i + 1])
            minIdxs[i] = maxIdxs[i] % minShapeInfo[i + 1];
          else if (maxIdxs[i] == minShapeInfo[i + 1])
            minIdxs[i] = 0;
          else
            minIdxs[i] = maxIdxs[i];
        }
      }
    } else {
      for (int i = 0, dim = 0; i < maxRank; ++i) {
        if (dim < dimsLen && dimsToExclude[dim] == i) {
          minIdxs[i] = maxIdxs[i];
          ++dim;
          continue;
        }

        if (maxIdxs[i] > minShapeInfo[i + 1])
          minIdxs[i] = maxIdxs[i] % minShapeInfo[i + 1];
        else if (maxIdxs[i] == minShapeInfo[i + 1])
          minIdxs[i] = 0;
        else
          minIdxs[i] = maxIdxs[i];
      }
    }
  } else {
    if (dimsToExclude == nullptr) {  // --> means dimsToExclude == {0,1,2,...,dimsLen-1}

      for (int i = 0; i < minRank; ++i) {
        if (maxIdxs[i + dimsLen] > minShapeInfo[i + 1])
          minIdxs[i] = maxIdxs[i + dimsLen] % minShapeInfo[i + 1];
        else if (maxIdxs[i + dimsLen] == minShapeInfo[i + 1])
          minIdxs[i] = 0;
        else
          minIdxs[i] = maxIdxs[i + dimsLen];
      }
    } else {
      for (int minI = 0, maxI = 0, dim = 0; maxI < maxRank; ++maxI) {
        if (dim < dimsLen && dimsToExclude[dim] == maxI) {
          ++dim;
          continue;
        }

        if (maxIdxs[maxI] == minShapeInfo[minI + 1])
          minIdxs[minI] = 0;
        else if (maxIdxs[maxI] > minShapeInfo[minI + 1])
          minIdxs[minI] = maxIdxs[maxI] % minShapeInfo[minI + 1];
        else
          minIdxs[minI] = maxIdxs[maxI];
        ++minI;
      }
    }
  }
}



//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_HOST sd::LongType subArrayOffset(const sd::LongType maxIdx, const sd::LongType *maxShapeInfo,
                                                  const sd::LongType *minShapeInfo, const sd::LongType  *dimsToExclude,
                                                  const sd::LongType dimsLen) {
  sd::LongType maxIdxs[SD_MAX_RANK];
  shape::index2coords(const_cast<sd::LongType &>(maxIdx), maxShapeInfo, maxIdxs);

  sd::LongType minIdxs[SD_MAX_RANK];
  maxIndToMinInd(maxIdxs, minIdxs, maxShapeInfo, minShapeInfo, dimsToExclude, dimsLen);

  return getOffset(minShapeInfo, minIdxs);
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_HOST int outerArrayOffsets(sd::LongType *maxOffsets, const sd::LongType minIdx,
                                            const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                            sd::LongType *memBuff, const sd::LongType *dimsToExclude) {
  const auto rankMin = shape::rank(minShapeInfo);
  const auto rankMax = shape::rank(maxShapeInfo);


  const auto diff = rankMax - rankMin;  // the size of dimsToExclude is equal to diff

  sd::LongType *indices = memBuff;
  sd::LongType *increment = memBuff + rankMax;

  sd::LongType N, minI, maxI;

  // calculate min per-dim-indices which corresponds to absolute minIdx index
  shape::index2coords(minIdx, minShapeInfo, indices);

  // transform storage indices to contain per-dim max indices, purpose - memory saving
  // fill increment array as well
  if (dimsToExclude == nullptr) {  // means dimsToExclude == {0,1,2,...,diff-1}
    for (minI = rankMin - 1, maxI = rankMax - 1; maxI >= diff; --maxI, --minI) {
      increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
      indices[maxI] = indices[minI];
    }
    for (maxI = 0; maxI < diff; ++maxI) {
      increment[maxI] = 1;
      indices[maxI] = 0;
    }
  } else {
    for (N = diff - 1, minI = rankMin - 1, maxI = rankMax - 1; maxI >= 0; --maxI) {
      if (N >= 0 && dimsToExclude[N] == maxI) {
        increment[maxI] = 1;
        indices[maxI] = 0;
        --N;
      } else {
        increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
        indices[maxI] = indices[minI--];
      }
    }
  }

  maxI = rankMax - 1;
  N = 0;
  int step;
  maxOffsets[N++] = shape::getOffset(maxShapeInfo, indices);

  // nested loops - producing of absolute indices for max array
  while (maxI >= 0) {
    if (increment[maxI] != 0) {
      indices[maxI] += increment[maxI];
      if (indices[maxI] >= maxShapeInfo[maxI + 1]) {
        indices[maxI] %= increment[maxI];  // restore initial value of indices[maxI]
        step = -1;
      } else {
        maxOffsets[N++] = shape::getOffset(maxShapeInfo, indices);
        step = rankMax - 1 - maxI;
      }
    } else if (maxI == rankMax - 1)
      step = -1;

    maxI += step;
  }
  return N;
}

//////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT SD_HOST sd::LongType outerArrayIndexes(sd::LongType *maxIdxs, const sd::LongType minIdx,
                                                     const sd::LongType *maxShapeInfo, const sd::LongType *minShapeInfo,
                                                     const sd::LongType  *dimsToExclude) {
  const auto rankMin = shape::rank(minShapeInfo);
  const auto rankMax = shape::rank(maxShapeInfo);


  const sd::LongType diff = rankMax - rankMin;  // the size of dimsToExclude is equal to diff

  sd::LongType indices[SD_MAX_RANK], increment[SD_MAX_RANK];

  sd::LongType N, minI, maxI;

  // calculate min per-dim-indices which corresponds to absolute minIdx index
  shape::index2coords(minIdx, minShapeInfo, indices);

  // transform storage indices to contain per-dim max indices, purpose - memory saving
  // fill increment array as well
  if (dimsToExclude == nullptr) {  // means dimsToExclude == {0,1,2,...,diff-1}
    for (minI = rankMin - 1, maxI = rankMax - 1; maxI >= diff; --maxI, --minI) {
      increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
      indices[maxI] = indices[minI];
    }
    for (maxI = 0; maxI < diff; ++maxI) {
      increment[maxI] = 1;
      indices[maxI] = 0;
    }
  } else {
    for (N = diff - 1, minI = rankMin - 1, maxI = rankMax - 1; maxI >= 0; --maxI) {
      if (N >= 0 && dimsToExclude[N] == maxI) {
        increment[maxI] = 1;
        indices[maxI] = 0;
        --N;
      } else {
        increment[maxI] = (maxShapeInfo[maxI + 1] == minShapeInfo[minI + 1]) ? 0 : minShapeInfo[minI + 1];
        indices[maxI] = indices[minI--];
      }
    }
  }

  maxI = rankMax - 1;
  N = 0;
  int step;
  maxIdxs[N++] = shape::coords2index(maxShapeInfo, indices);

  // nested loops - producing of absolute indices for max array
  while (maxI >= 0) {
    if (increment[maxI] != 0) {
      indices[maxI] += increment[maxI];
      if (indices[maxI] >= maxShapeInfo[maxI + 1]) {
        indices[maxI] %= increment[maxI];  // restore initial value of indices[maxI]
        step = -1;
      } else {
        maxIdxs[N++] = shape::coords2index(maxShapeInfo, indices);
        step = rankMax - 1 - maxI;
      }
    } else if (maxI == rankMax - 1)
      step = -1;

    maxI += step;
  }
  return N;
}

#endif


/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_HOST sd::LongType *calcStridesFortran(sd::LongType const *shape, sd::LongType rank, sd::LongType startNum) {
  int dimensions = rank;

  traceNew(5);

  sd::LongType *stride = new sd::LongType[dimensions];
  sd::LongType st = startNum;
  for (int j = 0; j < rank; j++) {
    stride[j] = st;
    st *= shape[j];
  }

  return stride;
}

SD_HOST sd::LongType *calcStridesFortran(sd::LongType const *shape, int rank, int startNum,
                                         sd::LongType *ret) {
  // if (isVector(shape, rank)) {
  //     for (int i = 0; i < rank; i++)
  //         ret[i] = 1;
  //     return ret;

  // }

  // int dimensions = rank;

  sd::LongType st = startNum;
  for (int j = 0; j < rank; j++) {
    ret[j] = st;
    st *= shape[j];
  }

  return ret;
}

/**
 * Computes the standard packed array strides for a given shape.
 *
 * @param shape    the shape of a matrix:
 * @param startNum the start number for the strides
 * @return the strides for a matrix of n dimensions
 */
SD_HOST sd::LongType *calcStrides(sd::LongType const *shape, int rank, int startNum) {
  traceNew(7);

  sd::LongType *stride = new sd::LongType[rank];

  if (rank == 1) {
    stride[0] = 1;
    return stride;
  }

  // if (shape::isVector(shape, rank)) {
  //     for (int i = 0; i < 2; i++)
  //         stride[i] = 1;
  //     return stride;

  // }

  sd::LongType st = startNum;
  for (int j = rank - 1; j >= 0; j--) {
    stride[j] = st;
    st *= shape[j];
  }

  return stride;
}

SD_HOST sd::LongType *calcStrides(sd::LongType const *shape, int rank, int startNum,
                                  sd::LongType *ret) {
  if (rank == 1) {
    ret[0] = 1;
    return ret;
  }

  // if (shape::isVector(shape, rank)) {
  //     for (int i = 0; i < 2; i++)
  //         ret[i] = 1;
  //     return ret;

  // }

  sd::LongType st = startNum;
  for (int j = rank - 1; j >= 0; j--) {
    ret[j] = st;
    st *= shape[j];
  }

  return ret;
}

//////////////////////////////////////////////////////////////////////
SD_HOST void updateStrides(sd::LongType *shapeInfo, const char order) {
  int rank = shapeInfo[0];
  int doubleRank = 2 * rank;

  if (rank > 0) {
    if (order == 'c') {
      shapeInfo[doubleRank] = 1;  // set unity as last stride for c order
      for (int j = 1; j < rank; ++j) {
        shapeInfo[doubleRank - j] = shapeInfo[doubleRank - j + 1] * shapeInfo[rank + 1 - j];
      }
    } else {
      shapeInfo[rank + 1] = 1;  // set unity as first stride for f order
      for (int j = rank + 1; j < doubleRank; ++j) {
        shapeInfo[j + 1] = shapeInfo[j] * shapeInfo[j - rank];
      }
    }
  }
  // set last 2 elements in shapeInfo
  shapeInfo[doubleRank + 2] = 1;
  shapeInfo[doubleRank + 3] = (int)order;
}

//////////////////////////////////////////////////////////////////////
SD_HOST void updateStrides(const int rank, const sd::LongType *shapeOnly, sd::LongType *stridesOnly,
                           const char order) {
  if (rank > 0) {
    if (order == 'c') {
      stridesOnly[rank - 1] = 1;  // set unity as last stride for c order
      for (int j = 1; j < rank; ++j) stridesOnly[rank - 1 - j] = stridesOnly[rank - j] * shapeOnly[rank - j];
    } else {
      stridesOnly[0] = 1;  // set unity as first stride for f order
      for (int j = 1; j < rank; ++j) {
        stridesOnly[j] = stridesOnly[j - 1] * shapeOnly[j - 1];
      }
    }
  }
}
/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
SD_HOST ShapeInformation *shapeCopy(ShapeInformation *toCopy) {
  auto copy = new ShapeInformation;

  traceNew(8);

  copy->shape = new sd::LongType[toCopy->rank];

  memcpy(copy->shape, toCopy->shape, toCopy->rank * sizeof(sd::LongType));

  traceNew(9);

  copy->stride = new sd::LongType[toCopy->rank];
  for (int i = 0; i < toCopy->rank; i++) {
    copy->stride[i] = toCopy->stride[i];
  }
  copy->order = toCopy->order;
  copy->rank = toCopy->rank;
  copy->offset = toCopy->offset;
  copy->elementWiseStride = toCopy->elementWiseStride;
  return copy;
}

SD_HOST int computeElementWiseStride(int rank, sd::LongType const *shape, sd::LongType const *stride,
                                     int isFOrder) {
  if (rank == 0) return 1;

  if (shape::isVector(shape, rank)) {
    return stride[rank - 1];
  }

  else {
    int oldnd;
    sd::LongType *oldDims = shape::copyOf(rank, shape);
    sd::LongType *oldStrides = shape::copyOf(rank, stride);
    sd::LongType np, op, last_stride;
    sd::LongType oldStart, oldStop, ok, newStart, newStop, nk;

    traceNew(10);

    auto newStrides = new sd::LongType[rank];
    oldnd = 0;
    // set the shape to be 1 x length
    int newShapeRank = 2;
    auto newShape = new sd::LongType[newShapeRank];
    newShape[0] = 1;
    newShape[1] = shape::prodLong(shape, rank);

    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oldStart = 0; oldStart < rank; oldStart++) {
      if (shape[oldStart] != 1) {
        oldDims[oldnd] = shape[oldStart];
        oldStrides[oldnd] = stride[oldStart];
        oldnd++;
      }
    }

    np = 1;
    for (newStart = 0; newStart < newShapeRank; newStart++) {
      np *= newShape[newStart];
    }
    op = 1;
    for (oldStart = 0; oldStart < oldnd; oldStart++) {
      op *= oldDims[oldStart];
    }
    if (np != op) {
      /* different total sizes; no hope */
      delete[] newStrides;
      delete[] newShape;
      delete[] oldStrides;
      delete[] oldDims;
      return 0;
    }

    if (np == 0) {
      /* the current code does not handle 0-sized arrays, so give up */
      delete[] newStrides;
      delete[] newShape;
      delete[] oldStrides;
      delete[] oldDims;
      return 0;
    }

    /* oldStart to oldStop and newStart to newStop give the axis ranges currently worked with */
    oldStart = 0;
    oldStop = 1;
    newStart = 0;
    newStop = 1;
    while (newStart < newShapeRank && oldStart < oldnd) {
      np = newShape[newStart];
      op = oldDims[oldStart];

      while (np != op) {
        if (np < op) {
          /* Misses trailing 1s, these are handled later */
          np *= newShape[newStop++];
        } else {
          op *= oldDims[oldStop++];
        }
      }

      /* Check whether the original axes can be combined */
      for (ok = oldStart; ok < oldStop - 1; ok++) {
        if (isFOrder) {
          if (oldStrides[ok + 1] != oldDims[ok] * oldStrides[ok]) {
            /* not contiguous enough */
            delete[] newStrides;
            delete[] newShape;
            delete[] oldStrides;
            delete[] oldDims;
            return 0;
          }
        } else {
          /* C order */
          if (oldStrides[ok] != oldDims[ok + 1] * oldStrides[ok + 1]) {
            /* not contiguous enough */
            delete[] newStrides;
            delete[] newShape;
            delete[] oldStrides;
            delete[] oldDims;
            return 0;
          }
        }
      }

      /* Calculate new strides for all axes currently worked with */
      if (isFOrder) {
        newStrides[newStart] = oldStrides[oldStart];
        for (nk = newStart + 1; nk < newStop; nk++) {
          newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
        }
      } else {
        /* C order */
        newStrides[newStop - 1] = oldStrides[oldStop - 1];
        for (nk = newStop - 1; nk > newStart; nk--) {
          newStrides[nk - 1] = newStrides[nk] * newShape[nk];
        }
      }
      newStart = newStop++;
      oldStart = oldStop++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (newStart >= 1) {
      last_stride = newStrides[newStart - 1];
    } else {
      last_stride = stride[rank - 1];
    }
    if (isFOrder) {
      if (newStart >= 1) last_stride *= newShape[newStart - 1];
    }
    for (nk = newStart; nk < newShapeRank; nk++) {
      newStrides[nk] = last_stride;
    }
    // returns the last element of the new stride array
    int ret = last_stride;
    delete[] newStrides;
    delete[] newShape;
    delete[] oldStrides;
    delete[] oldDims;
    return ret;
  }
}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
SD_HOST sd::LongType *shapeBuffer(int rank, sd::DataType dtype, sd::LongType const *shape) {
  sd::LongType *stride = shape::calcStrides(shape, rank);

  traceNew(11);

  auto shapeInfo = new shape::ShapeInformation();
  shapeInfo->shape = const_cast<sd::LongType *>(shape);
  shapeInfo->stride = stride;
  shapeInfo->offset = 0;
  shapeInfo->rank = rank;
  int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);
  shapeInfo->order = 'c';
  shapeInfo->elementWiseStride = elementWiseStride;
  auto shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
  delete[] stride;
  delete shapeInfo;
  sd::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
  return shapeInfoBuffer;
}

/**
 * This is special method, it returns ONLY 2D shapebuffer.
 *
 * This method is used only for SoftMax
 */
SD_HOST sd::LongType *shapeBuffer(int rank, sd::DataType dtype, sd::LongType const *shape,
                                  sd::LongType *buffer) {
  sd::LongType stride[SD_MAX_RANK];
  shape::calcStrides(shape, rank, stride);

  shape::ShapeInformation shapeInfo;
  shapeInfo.shape = const_cast<sd::LongType *>(shape);
  shapeInfo.stride = stride;
  shapeInfo.offset = 0;
  shapeInfo.rank = rank;
  auto elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

  shapeInfo.order = 'c';
  shapeInfo.elementWiseStride = elementWiseStride;
  shape::toShapeBuffer(&shapeInfo, buffer);
  sd::ArrayOptions::setDataType(buffer, dtype);
  return buffer;
}

/**
 * Get the shape info buffer
 * for the given rank and shape.
 */
SD_HOST sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape) {
  auto stride = shape::calcStridesFortran(shape, rank);

  traceNew(12);

  auto shapeInfo = new shape::ShapeInformation();
  shapeInfo->shape = const_cast<sd::LongType *>(shape);
  shapeInfo->stride = stride;
  shapeInfo->offset = 0;
  shapeInfo->rank = rank;
  int elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

  shapeInfo->order = 'f';
  shapeInfo->elementWiseStride = elementWiseStride;
  auto shapeInfoBuffer = shape::toShapeBuffer(shapeInfo);
  delete[] stride;
  delete shapeInfo;
  sd::ArrayOptions::setDataType(shapeInfoBuffer, dtype);
  return shapeInfoBuffer;
}

SD_HOST sd::LongType *shapeBufferFortran(int rank, sd::DataType dtype, sd::LongType const *shape,
                                         sd::LongType *output) {
  sd::LongType stride[SD_MAX_RANK];
  shape::calcStridesFortran(shape, rank, stride);

  shape::ShapeInformation shapeInfo;
  shapeInfo.shape = const_cast<sd::LongType *>(shape);
  shapeInfo.stride = stride;
  shapeInfo.offset = 0;
  shapeInfo.rank = rank;
  auto elementWiseStride = shape::computeElementWiseStride(rank, shape, stride, 0);

  shapeInfo.order = 'f';
  shapeInfo.elementWiseStride = elementWiseStride;
  shape::toShapeBuffer(&shapeInfo, output);
  sd::ArrayOptions::setDataType(output, dtype);
  return output;
}

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
SD_HOST sd::LongType *doPermuteSwap(long long int length, sd::LongType *shape, long long int *rearrange) {
  traceNew(16);
  sd::LongType *ret = new sd::LongType[length];
  for (int i = 0; i < length; i++) {
    ret[i] = shape[rearrange[i]];
  }
  return ret;
}

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
SD_HOST void doPermuteSwap(int length, sd::LongType **shape, long long int *rearrange) {
  if (length == 1) {
    return;
  } else {
    sd::LongType *shapeDeref = *shape;
    if (shape::prodLong(shapeDeref, length) < 2) {
      return;
    }
  }

  bool inOrder = true;
  for (int i = 0; i < length - 1; i++) {
    inOrder = inOrder && rearrange[i] + 1 == rearrange[i + 1];
  }

  // all in order, nothing to do
  if (inOrder) return;

  sd::LongType *shapeDeref = *shape;
  // we know they are just reversed, dimension length of 2
  if (length == 2) {
    auto shapeFirst = shapeDeref[0];
    auto shapeSecond = shapeDeref[1];
    shapeDeref[0] = shapeSecond;
    shapeDeref[1] = shapeFirst;
    return;
  } else if (length == 1) {
    // no permute
    return;
  }

  auto temp = new sd::LongType[length];
  memcpy(temp, shapeDeref, sizeof(sd::LongType) * length);
  for (int i = 0; i < length; i++) {
    shapeDeref[i] = temp[rearrange[i]];
  }

  delete[] temp;
}

SD_HOST void permuteShapeBufferInPlace(sd::LongType *shapeBuffer, long long int *rearrange, sd::LongType *out) {
  if (shapeBuffer != out) memcpy(out, shapeBuffer, sizeof(sd::LongType) * shape::shapeInfoLength(shapeBuffer));

  shape::doPermuteShapeInfo(out, rearrange);
}

SD_HOST sd::LongType *permuteShapeBuffer(sd::LongType const *shapeBuffer, long long int *rearrange) {
  auto len = shape::shapeInfoLength(shape::rank(shapeBuffer));
  sd::LongType *copy = shape::copyOf(len, shapeBuffer);
  shape::doPermuteShapeInfo(copy, rearrange);
  return copy;
}

SD_HOST void doPermuteShapeInfo(sd::LongType *shapeInfo, const long long int *rearrange, sd::LongType len) {
  if (len == -1)  // calculate array length if it is not given
    len = shape::length(shapeInfo);

  // check whether shape is like {1} or {1,1} or {1,1,1,1,...} - in this case we don't need permute
  if (len == 1) return;

  const int rank = shape::rank(shapeInfo);

  // check whether rearrange is like {0,1,2,3,...}  - in this case we don't need permute as well
  bool isPermutNecessary = false;
  for (int i = 0; i < rank; ++i)
    if (rearrange[i] != i) {
      isPermutNecessary = true;
      break;
    }

  if (!isPermutNecessary) return;

  // check whether rearrange contains correct indexes
  for (int i = 0; i < rank; ++i) {
    if (rearrange[i] >= rank || rearrange[i] < 0) {
      sd_printf(
          "shape::doPermuteShapeInfo function failed: rearrange indexes are incorrect. Given permute indices must be < rank and >= 0.  Rearrange at index %d was %d\n",
          i, rearrange[i]);
      return;
    }
  }
  // if everything is ok then perform permute
  auto temp = new sd::LongType[shape::shapeInfoLength(rank) - 3];
  memcpy(temp, shapeInfo, sizeof(sd::LongType) * (shape::shapeInfoLength(rank) - 3));
  for (int i = 0; i < rank; ++i) {
    shapeInfo[i + 1] = temp[rearrange[i] + 1];
    shapeInfo[i + 1 + rank] = temp[rearrange[i] + 1 + rank];
  }

  shape::checkStridesEwsAndOrder(shapeInfo);

  delete[] temp;
}

SD_HOST sd::LongType *createPermuteIndexes(long long int originalRank, long long int *dimension,
                                           long long int dimensionLength) {
  int delta = originalRank - dimensionLength;

  traceNew(17);

  sd::LongType *ret = new sd::LongType[originalRank];
  for (int i = 0; i < delta; i++) {
    ret[i] = i + dimensionLength;
  }

  for (int i = delta; i < originalRank; i++) {
    ret[i] = i - delta;
  }

  return ret;
}
/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
SD_HOST void permute(ShapeInformation **info, long long int *rearrange, int rank) {
  ShapeInformation *infoDeref = *info;
  checkArrangeArray(rearrange, rank, rank);
  shape::doPermuteSwap(rank, &infoDeref->shape, rearrange);
  shape::doPermuteSwap(rank, &infoDeref->stride, rearrange);
  char order = getOrder(rank, infoDeref->shape, infoDeref->stride, infoDeref->elementWiseStride);
  infoDeref->order = order;
}

/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
SD_HOST void copyTo(int length, sd::LongType const *from, sd::LongType *to, sd::LongType *indexes) {
  for (int i = 0; i < length; i++) {
    to[i] = from[indexes[i]];
  }
}

SD_HOST sd::LongType *sliceOfShapeBuffer(sd::LongType sliceIdx, sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  int newRank = rank - 1;
  if (newRank < 2) newRank = 2;
  sd::LongType *newShapeBuffer = new sd::LongType[shape::shapeInfoLength(newRank)];
  newShapeBuffer[0] = newRank;
  sd::LongType *currShape = shape::shapeOf(shapeBuffer);
  sd::LongType *currStride = shape::stride(shapeBuffer);
  // initialize new shape and stride by taking the shape and stride + 1
  // and adding to the shape information
  // a slice is always just taking the existing shape and cutting the first index off
  // of the shape and stride
  sd::LongType *newShape = shape::shapeOf(newShapeBuffer);
  sd::LongType *newStride = shape::stride(newShapeBuffer);
  if (shape::isVector(shapeBuffer)) {
    sd::LongType *currShape = shape::shapeOf(shapeBuffer);
    // row vector: slice index 0 is a valid index, just copy the whole thing
    if (currShape[0] == 1) {
      if (sliceIdx == 0) {
        memcpy(newShapeBuffer, shapeBuffer, shape::shapeInfoByteLength(shape::rank(shapeBuffer)));
        return newShapeBuffer;
      }
    }
      // column vector: this will be a scalar
    else {
      delete[] newShapeBuffer;
      sd::LongType *scalar = shape::createScalarShapeInfo();
      int offset = shape::offset(shapeBuffer);
      scalar[shape::shapeInfoLength(2) - 3] = offset + sliceIdx;
      return scalar;
    }
  } else if (shape::isMatrix(shapeBuffer)) {
    newShape[0] = 1;
    newShape[1] = currShape[1];
    newStride[0] = 1;
    newStride[1] = currStride[1];
  } else {
    for (int i = 0; i < newRank; i++) {
      newShape[i] = currShape[i + 1];
      newStride[i] = currStride[i + 1];
    }
  }

  auto indices = new sd::LongType[rank];
  memset((void *)indices, 0, rank * sizeof(sd::LongType));
  indices[0] = sliceIdx;
  sd::LongType offset = shape::getOffset(newShapeBuffer, indices);
  newShapeBuffer[shape::shapeInfoLength(newRank) - 3] = offset;

  // set current order and ews
  newShapeBuffer[2 * newRank + 2] = shape::elementWiseStride(shapeBuffer);
  newShapeBuffer[2 * newRank + 3] = shape::order(shapeBuffer);

  // correct order and ews if necessary
  shape::checkStridesEwsAndOrder(newShapeBuffer);

  delete[] indices;

  return newShapeBuffer;
}

/**
 * Returns the element wise stride for this information
 * buffer relative to a dimension and reduction index
 */
SD_HOST sd::LongType reductionIndexElementWiseStride(sd::LongType *buffer, sd::LongType *dimension,
                                                     sd::LongType dimensionLength) {
  if (dimensionLength > 1) {
    if (shape::order(buffer) == 'f') {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      if (shape::shapeOf(buffer)[dimension[dimensionLength - 1]] != 1) {
        auto tadElementWiseStride = shape::stride(buffer)[dimension[0]];
        return tadElementWiseStride;
      }

      return 1;

    } else {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      if (shape::shapeOf(buffer)[dimension[dimensionLength - 1]] != 1) {
        auto tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
        return tadElementWiseStride;
      }

      return 1;
    }
  } else {
    if (shape::order(buffer) == 'f') {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      auto tadElementWiseStride = shape::stride(buffer)[dimension[0]];
      return tadElementWiseStride;
    } else {
      /**
       * The element wise stride belongs to a reduction index.
       * When used out of order, we can get rid of the data
       * dependencies and rely on using the max dimension
       * specified for stride instead.
       * Say we take the sum(0,1) along arr
       * we can use arr.stride(1) as a representation
       * along which to iterate.
       */
      auto tadElementWiseStride = shape::stride(buffer)[dimension[dimensionLength - 1]];
      return tadElementWiseStride;
    }
  }
}



SD_HOST sd::LongType *everyIndexBut(const sd::LongType *indexes, int indexesLength, int begin,
                                    int end) {
  int len = end - indexesLength;

  traceNew(20);

  auto ret = new sd::LongType[len];
  int retIdx = 0;
  // not here that we do 0 based indexing for end - this assumes things like:
  // 0 to 4 are specified
  for (int i = begin; i < end; i++) {
    bool found = false;
    for (int j = 0; j < indexesLength; j++) {
      if (indexes[j] == i) {
        found = true;
        break;
      }
    }

    if (!found) {
      ret[retIdx++] = i;
    }
  }

  return ret;
}
/**
 * Keep the given indexes in the data
 * @param data
 * @param index
 * @param indexLength
 * @param dataLength
 * @return
 */
SD_HOST sd::LongType *keep(volatile sd::LongType *data, const long long int *index, int indexLength,
                           int dataLength) {
  traceNew(23);

  sd::LongType *ret = new sd::LongType[indexLength];
  int count = 0;
  for (int i = 0; i < dataLength; i++) {
    int contains = 0;
    for (int j = 0; j < indexLength; j++) {
      if (i == index[j]) {
        contains = 1;
        break;
      }
    }

    if (contains) ret[count++] = data[i];
  }
  return ret;
}
/**
 * Get the length per slice of the
 * given shape and the dimension
 * @param rank the rank of the shape
 * @param shape the shape of to get
 * the length per slice for
 * @param dimension the dimension to
 * get the length per slice for
 * @param dimensionLength the length of the dimension array
 * @return the length per slice of the given shape
 * along the given dimension
 */
SD_HOST sd::LongType lengthPerSlice(int rank, sd::LongType const *shape, const long long int *dimension,
                                    long long int dimensionLength) {
  if (shape::isVector(shape, rank)) {
    // return total length for row vectors
    if (dimensionLength == 1 && shape[0] == 1) {
      return shape::prodLong(shape, rank);
    }
  } else if (rank == dimensionLength)
    return shape::prodLong(shape, rank);
  sd::LongType  absSelta = sd::math::sd_abs<sd::LongType >(rank - dimensionLength);
  traceNew(27);
  auto ret2 = shape::removeIndex<sd::LongType>(shape, dimension, rank, dimensionLength);
  auto ret = prodLong(ret2, absSelta);
  delete[] ret2;
  return ret;
}

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
SD_HOST sd::LongType sliceOffsetForTensor(int rank, int index, sd::LongType const *shape,
                                          sd::LongType const *tensorShape, int tensorShapeLength,
                                          const long long int *dimension, long long int dimensionLength) {
  auto tensorLength = prodLong(tensorShape, tensorShapeLength);
  auto lengthPerSlice2 = lengthPerSlice(rank, shape, dimension, dimensionLength);
  if (lengthPerSlice2 <= 0) {
    return 0;
  }

  sd::LongType offset = index * tensorLength / lengthPerSlice2;
  return offset;
}
/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
SD_HOST sd::LongType tensorsAlongDimension(volatile int rank, volatile int length,
                                           volatile sd::LongType *shape,
                                           long long int *dimension, long long int dimensionLength) {
  sd::LongType *tensorShape = shape::keep(shape, dimension, dimensionLength, rank);
  sd::LongType ret = length / shape::prodLong(tensorShape, dimensionLength);
  delete[] tensorShape;
  return ret;
}

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
SD_HOST sd::LongType tensorsAlongDimension(sd::LongType *shapeInfo, long long int *dimension,
                                           long long int dimensionLength) {
  sd::LongType *keepShape = shape::shapeOf(shapeInfo);
  sd::LongType *tensorShape = shape::keep(keepShape, dimension, dimensionLength, rank(shapeInfo));
  sd::LongType ret = shape::length(shapeInfo) / shape::prodLong(tensorShape, dimensionLength);
  delete[] tensorShape;
  return ret;
}

//////////////////////////////////////////////////////////////////////
SD_HOST void getOffsetBroadcast(const sd::LongType &startInd, const sd::LongType ind,
                                const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2,
                                const sd::LongType *shapeInfo3, const bool sameOffsets12,
                                const bool sameOffsets13, long long int *coords, sd::LongType &offset1,
                                sd::LongType &offset2, sd::LongType &offset3) {
  const sd::LongType *shape1 = shape::shapeOf(shapeInfo1);
  const sd::LongType *strides1 = shape::stride(shapeInfo1);
  const sd::LongType *shape2 = shape::shapeOf(shapeInfo2);
  const sd::LongType *strides2 = shape::stride(shapeInfo2);
  const sd::LongType *shape3 = shape::shapeOf(shapeInfo3);
  const sd::LongType *strides3 = shape::stride(shapeInfo3);

  if (startInd == ind) {
    if (shape::rank(shapeInfo1) == 0) {
      offset1 = offset2 = offset3 = 0;
      return;
    }

    shape::index2coords(ind, shapeInfo1, coords);
    offset1 = shape::getOffset(shapeInfo1, coords);

    if (sameOffsets12)
      offset2 = offset1;
    else
      offset2 = shape::getOffset(shapeInfo2, coords);

    if (sameOffsets13)
      offset3 = offset1;
    else
      offset3 = shape::getOffset(shapeInfo3, coords);

    return;
  }

  int axis = shapeInfo1[0] - 1;
  while (coords[axis] == shape1[axis] - 1) {
    if (!sameOffsets12 && shape2[axis] != 1) offset2 -= (shape2[axis] - 1) * strides2[axis];
    if (!sameOffsets13 && shape3[axis] != 1) offset3 -= (shape3[axis] - 1) * strides3[axis];
    if (shape1[axis] != 1) offset1 -= (shape1[axis] - 1) * strides1[axis];
    coords[axis--] = 0;
  }

  ++coords[axis];
  offset1 += strides1[axis];

  if (!sameOffsets12 && shape2[axis] != 1) offset2 += strides2[axis];
  if (!sameOffsets13 && shape3[axis] != 1) offset3 += strides3[axis];

  if (sameOffsets12) offset2 = offset1;
  if (sameOffsets13) offset3 = offset1;
}

/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
SD_HOST sd::LongType *toShapeBuffer(ShapeInformation *info) {
  traceNew(29);

  auto ret = new sd::LongType[shapeInfoLength(info->rank)];
  int count = 1;
  int rank = info->rank;

  ret[0] = info->rank;

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->shape[i];
  }

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->stride[i];
  }

  ret[count++] = info->offset;
  ret[count++] = info->elementWiseStride;
  ret[count] = info->order;

  return ret;
}

SD_HOST sd::LongType *toShapeBuffer(ShapeInformation *info, sd::LongType *ret) {
  int count = 1;
  int rank = info->rank;

  ret[0] = info->rank;

  if (ret[0] == 0) {
    ret[1] = 0;
    ret[2] = 1;
    ret[3] = 99;
    return ret;
  }

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->shape[i];
  }

  for (int i = 0; i < rank; i++) {
    ret[count++] = info->stride[i];
  }

  ret[count++] = info->offset;
  ret[count++] = info->elementWiseStride;
  ret[count++] = info->order;

  return ret;
}

SD_HOST void printIntArray(const sd::LongType *arr, const int length) {
  for (int i = 0; i < length; i++) {
    printf(" %lld ", (long long)arr[i]);
  }

  printf("\n");
}

SD_HOST void printIntArray(const int *arr, const int length) {
  for (int i = 0; i < length; i++) {
    printf(" %i ", arr[i]);
  }

  printf("\n");
}

SD_HOST void printShapeInfo(const sd::LongType *shapeInfo) {
  if(shapeInfo == nullptr)
    return;
  if(shapeInfo != nullptr) {
    if(shapeInfo[0] > 32 || shapeInfo[0] < 0)
      return;
  }
  int rank = shape::rank(shapeInfo);
  if(rank == 0)
    return;

  sd::LongType *shape = shape::shapeOf(shapeInfo);
  printf("Rank %d\n", rank);
  if(rank == 0) {
    return;
  }
  printf("Shape:\n");
  for (int i = 0; i < rank; i++) {
    printf(" %lld ", (long long)shape[i]);
  }

  printf("\n");

  sd::LongType *stride = shape::stride(shapeInfo);
  printf("Stride:\n");
  for (int i = 0; i < rank; i++) {
    printf(" %lld ", (long long)stride[i]);
  }

  printf("\n");

  printf("Order %c\n", shape::order(shapeInfo));
}

SD_HOST void printShapeInfoLinear(const sd::LongType *shapeInfo) {
  int rank = shape::rank(shapeInfo);
  int lim = shape::shapeInfoLength(rank);
  printf("ShapeInfo: [");
  for (int i = 0; i < lim; i++) {
    printf("%lld", (long long)shapeInfo[i]);

    if (i < lim - 1) {
      printf(", ");
    }
  }
  printf("]\n");
#ifndef __CUDA_ARCH__
  fflush(stdout);
#endif
}

SD_HOST void printShapeInfoLinear(const char *msg, int rank, const sd::LongType *shape,
                                  const sd::LongType *strides) {
  printf("%s : [", msg);
  for (int i = 0; i < rank; i++) {
    printf("%lld, ", (long long)shape[i]);
  }

  for (int i = 0; i < rank; i++) {
    printf("%lld", (long long)strides[i]);

    if (i < rank - 1) printf(", ");
  }
  printf("]\n");

#ifndef __CUDA_ARCH__
  fflush(stdout);
#endif
}

SD_HOST void printShapeInfoLinear(const char *msg, const sd::LongType *shapeInfo) {
  int rank = shape::rank(shapeInfo);
  int lim = shape::shapeInfoLength(rank);
  printf("%s : [", msg);
  for (int i = 0; i < lim; i++) {
    printf("%lld", (long long)shapeInfo[i]);

    if (i < lim - 1) {
      printf(", ");
    }
  }
  printf("]\n");
#ifndef __CUDACC__
  fflush(stdout);
#endif
}

SD_HOST void printArray(float *arr, int length) {
  printf("Array: [");
  for (int i = 0; i < length; i++) {
    printf("%f", arr[i]);
    if (i + 1 < length) printf(", ");
  }
  printf("]\n");
}
SD_HOST void transposeInplace(sd::LongType *shapeBuffer) {
  int rank = shape::rank(shapeBuffer);
  sd::LongType *shape = shape::shapeOf(shapeBuffer);
  sd::LongType *strides = shape::stride(shapeBuffer);

  // swap shape
  for (int e = 0; e < rank / 2; e++) {
    int idx1 = rank - e - 1;
    int idx2 = e;
    int tmp = shape[idx2];
    shape[idx2] = shape[idx1];
    shape[idx1] = tmp;
  }

  // swap strides
  for (int e = 0; e < rank / 2; e++) {
    int idx1 = rank - e - 1;
    int idx2 = e;
    int tmp = strides[idx2];
    strides[idx2] = strides[idx1];
    strides[idx1] = tmp;
  }

  if (shape::order(shapeBuffer) == 'c')
    shapeBuffer[shape::shapeInfoLength(shapeBuffer) - 1] = 102;
  else
    shapeBuffer[shape::shapeInfoLength(shapeBuffer) - 1] = 99;
}

SD_HOST int rearMostLeftOverItem(sd::LongType *data, sd::LongType *dimension, long long int dimensionLength) {
  sd::LongType *stride = shape::stride(data);
  // corner case: return the final item when its greater than the max, since its guaranteed to be left over
  // note here that strides are interpreted in reverse for tad
  // start from the front rather than the back

  int rank = shape::rank(data);

  if (shape::order(data) == 'f') {
    int dimIdx = dimensionLength - 1;
    for (int i = rank - 1; i >= 0; i--) {
      /**
       * Needs to find an algorithm such that:
       * looping backwards will find the highest dimension left
       * that isn't included in the dimension index list.
       *
       * This can also be thought of as the last item of the first index
       * of the difference between the full list of indices and
       * the dimension indices.
       *
       * We should avoid excessive object creation by only looping backwards.
       */
      if (dimension[dimIdx--] != i) {
        int ret = stride[i];
        return ret;
      }
    }
  }

  else {
    int dimIdx = dimensionLength - 1;

    for (int i = rank - 1; i >= 0; i--) {
      /**
       * Needs to find an algorithm such that:
       * looping backwards will find the highest dimension left
       * that isn't included in the dimension index list.
       *
       * This can also be thought of as the last item of the first index
       * of the difference between the full list of indices and
       * the dimension indices.
       *
       * We should avoid excessive object creation by only looping backwards.
       */
      if (dimension[dimIdx--] != i) {
        int ret = stride[i];
        return ret;
      }
    }
  }

  int ret = stride[0];
  return ret;
}

SD_HOST sd::LongType *shapeBufferOfNpy(cnpy::NpyArray arr) {
  return shape::shapeBufferOfNpy(arr.shape.size(), (sd::LongType *)arr.shape.data(), arr.fortranOrder);
}



SD_HOST sd::LongType *shapeBufferOfNpy(sd::LongType rank, sd::LongType *shape, bool fortranOrder) {
  if (fortranOrder) {
    sd::LongType *shapeBufferRet = shape::shapeBufferFortran(rank, sd::FLOAT32, (sd::LongType *)shape);
    return shapeBufferRet;
  } else {
    sd::LongType *newShape = new sd::LongType[rank];
    for (int i = 0; i < rank; i++) {
      newShape[i] = shape[i];
    }

    sd::LongType *shapeBufferRet = shape::shapeBuffer(rank, sd::FLOAT32, newShape);
    delete[] newShape;
    return shapeBufferRet;
  }
}



//////////////////////////////////////////////////////////////////////////
// copy-past from java hasDefaultStridesForShape function
SD_HOST bool areStridesDefault(const sd::LongType *shapeInfo) {
  const int rank = shape::rank(shapeInfo);

  if (rank == 0) return true;
  if (!strideDescendingCAscendingF(shapeInfo)) return false;

  sd::LongType defaultShapeInfo[SD_MAX_SHAPEINFOLENGTH];
  memcpy(defaultShapeInfo, shapeInfo, shape::shapeInfoByteLength(shapeInfo));
  shape::updateStrides(defaultShapeInfo, shape::order(shapeInfo));

  bool result = true;
  for (int i = rank + 1; i <= 2 * rank; ++i)
    if (defaultShapeInfo[i] != shapeInfo[i]) {
      result = false;
      break;
    }

  return result;
}

//////////////////////////////////////////////////////////////////////
SD_HOST bool reshapeC(const sd::LongType *oldShapeInfo, const char newOrder, const long long int newRank,
                      const sd::LongType *newShape, sd::LongType *newShapeInfo) {
  // copy shape from newShape into newShapeInfo
  newShapeInfo[0] = newRank;
  memcpy(newShapeInfo + 1, newShape, newRank * sizeof(sd::LongType));

  // copy order
  newShapeInfo[2 * newRank + 3] = newOrder;

  return shape::reshapeC(oldShapeInfo, newShapeInfo);
}

//////////////////////////////////////////////////////////////////////
SD_HOST bool reshapeC(const sd::LongType *oldShapeInfo, sd::LongType *newShapeInfo) {
  // newShapeInfo contains rank, shape and order; but no strides, type and ews

  const int newRank = shape::rank(newShapeInfo);

  // if oldShapeInfo is scalar or vector with length=1
  if (shape::length(oldShapeInfo) == 1) {
    for (sd::Unsigned i = 0; i < newRank; ++i) shape::stride(newShapeInfo)[i] = 1;
    newShapeInfo[2 * newRank + 1] = shape::type(oldShapeInfo);
    *shape::ews(newShapeInfo) = 1;
    return true;
  }

  const auto oldOrder = shape::order(oldShapeInfo);
  const auto newOrder = shape::order(newShapeInfo);
  const auto oldEws = shape::elementWiseStride(const_cast<sd::LongType *>(oldShapeInfo));

  if (oldEws > 0 && oldOrder != newOrder) return false;

  // *** FIRST STAGE - exclude unity dimensions from oldShapeInfo and newShapeInfo (if such are present of course),
  // since they don't affect on strides evaluation, however they complicate code

  // FIXME - indeed we don't need to allocate so large memory amount (4*SD_MAX_RANK), sufficient amount is
  // (2*oldNumOfNonUnities + 2*newNumOfNonUnities)
  sd::LongType tempBuffer[4 * SD_MAX_RANK];
  sd::LongType *oldShape = tempBuffer, *newShape = tempBuffer + 2 * SD_MAX_RANK, *oldStrides, *newStrides;

  // exclude unities from oldShapeInfo
  const int oldNumOfNonUnities = shape::excludeUnitiesFromShapeInfo(oldShapeInfo, oldShape, oldStrides);
  const int newNumOfNonUnities = shape::excludeUnitiesFromShapeInfo(newShapeInfo, newShape, newStrides);

  // *** SECOND STAGE - strides evaluation

  int oldStart(0), oldStop(1), newStart(0), newStop(1), newDim, oldDim;

  while (newStart < newNumOfNonUnities && oldStart < oldNumOfNonUnities) {
    newDim = newShape[newStart];
    oldDim = oldShape[oldStart];

    while (newDim != oldDim && newDim > 0 && oldDim > 0) {
      if (newDim < oldDim)
        newDim *= newShape[newStop++];
      else
        oldDim *= oldShape[oldStop++];
    }

    // check c-contiguous of old axes range
    for (sd::Unsigned i = oldStart; i < oldStop - 1; ++i)  // do not check value of last stride, it doesn't matter
      if (oldStrides[i] != oldShape[i + 1] * oldStrides[i + 1]) return false;  // not contiguous

    // fill newStrides in c manner
    newStrides[newStop - 1] = oldStrides[oldStop - 1];  // copy last stride
    for (int i = newStop - 2; i >= newStart; --i) newStrides[i] = newStrides[i + 1] * newShape[i + 1];

    newStart = newStop++;
    oldStart = oldStop++;
  }

  // fill new calculated strides into newShapeInfo, take into account possible unities in shape
  for (int j = 0, i = 0; i < newRank; ++i)
    shape::stride(newShapeInfo)[i] = (shape::shapeOf(newShapeInfo)[i] == 1) ? 1 : newStrides[j++];

  // set ews
  if (oldEws == 0)
    shape::checkStridesEwsAndOrder(newShapeInfo, newOrder, newNumOfNonUnities, newShape,
                                   newStrides);  // set ews and order
  else {
    newShapeInfo[2 * newRank + 3] = oldOrder;  // order
    *shape::ews(newShapeInfo) = oldEws;        // ews
  }
  newShapeInfo[2 * newShapeInfo[0] + 1] = 0;
  sd::ArrayOptions::copyDataType(newShapeInfo, oldShapeInfo);  // type

  return true;
}

SD_HOST bool canReshape(const int oldRank, sd::LongType *oldShape, const int newRank,
                        sd::LongType *newShapeOf, bool isFOrder) {
  sd::LongType oldnd;
  sd::LongType *oldDims = shape::copyOf(oldRank, shape::shapeOf(oldShape));
  sd::LongType *oldStrides = shape::copyOf(oldRank, shape::stride(oldShape));
  sd::LongType np, op, last_stride;
  sd::LongType oldStart, oldStop, ok, newStart, newStop, nk;
  auto newStrides = new sd::LongType[newRank];
  oldnd = 0;

  /*
   * Remove axes with dimension 1 from the old array. They have no effect
   * but would need special cases since their strides do not matter.
   */
  for (oldStart = 0; oldStart < oldRank; oldStart++) {
    if (shape::shapeOf(oldShape)[oldStart] != 1) {
      oldDims[oldnd] = shape::shapeOf(oldShape)[oldStart];
      oldStrides[oldnd] = shape::stride(oldShape)[oldStart];
      oldnd++;
    }
  }

  np = 1;
  for (newStart = 0; newStart < newRank; newStart++) {
    np *= newShapeOf[newStart];
  }
  op = 1;
  for (oldStart = 0; oldStart < oldnd; oldStart++) {
    op *= oldDims[oldStart];
  }
  if (np != op) {
    /* different total sizes; no hope */
    delete[] oldDims;
    delete[] oldStrides;
    delete[] newStrides;

    return false;
  }

  if (np == 0) {
    /* the current code does not handle 0-sized arrays, so give up */
    delete[] oldDims;
    delete[] oldStrides;
    delete[] newStrides;

    return false;
  }

  /* oldStart to oldStop and newStart to newStop give the axis ranges currently worked with */
  oldStart = 0;
  oldStop = 1;
  newStart = 0;
  newStop = 1;

  while (newStart < newRank && oldStart < oldnd) {
    np = newShapeOf[newStart];
    op = oldDims[oldStart];

    while (np != op) {
      if (np < op) {
        /* Misses trailing 1s, these are handled later */
        np *= newShapeOf[newStop++];
      } else {
        op *= oldDims[oldStop++];
      }
    }

    /* Check whether the original axes can be combined */
    for (ok = oldStart; ok < oldStop - 1; ok++) {
      if (isFOrder) {
        if (oldStrides[ok + 1] != oldDims[ok] * oldStrides[ok]) {
          /* not contiguous enough */
          delete[] oldDims;
          delete[] oldStrides;
          delete[] newStrides;

          return false;
        }
      } else {
        /* C order */
        if (oldStrides[ok] != oldDims[ok + 1] * oldStrides[ok + 1]) {
          /* not contiguous enough */
          delete[] oldDims;
          delete[] oldStrides;
          delete[] newStrides;

          return false;
        }
      }
    }

    /* Calculate new strides for all axes currently worked with */
    if (isFOrder) {
      newStrides[newStart] = oldStrides[oldStart];
      for (nk = newStart + 1; nk < newStop; nk++) {
        newStrides[nk] = newStrides[nk - 1] * newShapeOf[nk - 1];
      }
    } else {
      /* C order */
      newStrides[newStop - 1] = oldStrides[oldStop - 1];
      for (nk = newStop - 1; nk > newStart; nk--) {
        newStrides[nk - 1] = newStrides[nk] * newShapeOf[nk];
      }
    }
    newStart = newStop++;
    oldStart = oldStop++;
  }

  delete[] oldDims;
  delete[] oldStrides;
  delete[] newStrides;

  return true;
}




//////////////////////////////////////////////////////////////////////
void calcOffsets(const sd::LongType *shapeInfo, sd::LongType *offsets, const char order) {
  // firstly consider simple case when ews > 0
  const sd::LongType ews = shape::elementWiseStride(shapeInfo);

  if (ews > 0) {
    // set offset for first sub-array, it is equal to zero always
    offsets[0] = 0;

    sd::LongType e = 0;
    if (order != shape::order(shapeInfo))
      for (int i = 1; i <= shape::rank(shapeInfo); ++i)
        if (shapeInfo[i] != 1) ++e;  // check whether input is CommonVector

    if (order == shape::order(shapeInfo) || e == 1) {  // e==1 means common vector
      e = 1;
      sd::LongType len = shape::length(shapeInfo);
      while (e < len) {
        offsets[e] = offsets[e - 1] + ews;
        e++;
      }
      return;
    }
  }

  shape::calcOffsets(shape::rank(shapeInfo), shape::shapeOf(const_cast<sd::LongType *>(shapeInfo)),
                     shape::stride(const_cast<sd::LongType *>(shapeInfo)), offsets, order);
}

//////////////////////////////////////////////////////////////////////
void calcOffsets(const int rank, const sd::LongType *shape, const sd::LongType *strides,
                 sd::LongType *offsets, const char order) {
  const uint64_t len = shape::prodLong(shape, rank);

  // set offset for first sub-array, it is equal to zero always
  offsets[0] = 0;

  sd::Unsigned coords[SD_MAX_RANK];
  memset(coords, 0, sizeof(sd::Unsigned) * rank);

  if (order == 'c') {
    for (uint64_t i = 1; i < len; ++i) {
      int axis = rank - 1;
      offsets[i] = 0;
      while (coords[axis] == shape[axis] - 1) {
        offsets[i] -= (shape[axis] - 1) * strides[axis];
        coords[axis--] = 0;
      }
      ++coords[axis];
      offsets[i] += offsets[i - 1] + strides[axis];
    }
  } else {
    for (uint64_t i = 1; i < len; ++i) {
      int axis = 0;
      offsets[i] = 0;
      while (coords[axis] == shape[axis] - 1) {
        offsets[i] -= (shape[axis] - 1) * strides[axis];
        coords[axis++] = 0;
      }
      ++coords[axis];
      offsets[i] += offsets[i - 1] + strides[axis];
    }
  }

}

//////////////////////////////////////////////////////////////////////
void SD_HOST checkStridesEwsAndOrder(sd::LongType *shapeInfo) {
  // FIXME - indeed we don't need to allocate so large memory amount (2*SD_MAX_RANK), sufficient amount is
  // (2*oldNumOfNonUnities + 2*newNumOfNonUnities)
  sd::LongType tempBuffer[2 * SD_MAX_RANK];
  sd::LongType *shape = tempBuffer, *strides;

  // exclude unities from shapeInfo
  const int numOfNonUnities = shape::excludeUnitiesFromShapeInfo(shapeInfo, shape, strides);

  shape::checkStridesEwsAndOrder(shapeInfo, shape::order(shapeInfo), numOfNonUnities, shape, strides);
}

//////////////////////////////////////////////////////////////////////
void SD_HOST checkStridesEwsAndOrder(sd::LongType *shapeInfo, const char proposedOrder,
                                     const int numOfNonUnities, const sd::LongType *shapeNoUnities,
                                     const sd::LongType *stridesNoUnities) {
  const int rank = shape::rank(shapeInfo);

  if (shape::length(shapeInfo) == 1) {
    *shape::ews(shapeInfo) = 1;
    shapeInfo[rank * 2 + 3] = (int)proposedOrder;
    return;
  }

  if (numOfNonUnities == 1) {  // case of common vector
    *shape::ews(shapeInfo) = *stridesNoUnities;
    shapeInfo[rank * 2 + 3] = (int)proposedOrder;
    return;
  }

  bool contiguous = true;

  //*** check whether strides are in c contiguous order ***//
  for (sd::Unsigned i = 0; i < numOfNonUnities - 1; ++i) {
    if (stridesNoUnities[i] != shapeNoUnities[i + 1] * stridesNoUnities[i + 1]) {
      contiguous = false;
      break;
    }
  }

  if (contiguous) {
    *shape::ews(shapeInfo) = stridesNoUnities[numOfNonUnities - 1];
    shapeInfo[rank * 2 + 3] = 99;
    return;
  }

  contiguous = true;

  //*** check whether strides are in f contiguous order ***//
  for (sd::Unsigned i = 1; i < numOfNonUnities; ++i) {
    if (stridesNoUnities[i] != shapeNoUnities[i - 1] * stridesNoUnities[i - 1]) {
      contiguous = false;
      break;
    }
  }

  if (contiguous) {
    *shape::ews(shapeInfo) = stridesNoUnities[0];
    shapeInfo[rank * 2 + 3] = 102;
    return;
  }

  *shape::ews(shapeInfo) = 0;
  shapeInfo[rank * 2 + 3] = (int)proposedOrder;
}

//////////////////////////////////////////////////////////////////////
SD_HOST void calcSubArrsShapeInfoAndOffsets(const sd::LongType *wholeShapeInfo,
                                            const sd::LongType numOfSubArrs, const int dimsSize, const long long int *dimsToExclude, sd::LongType *subArrShapeInfo,
                                            sd::LongType *subArrOffsets, bool keepUnitiesInShape) {
  const int rank = shape::rank(wholeShapeInfo);

  if (dimsSize == rank || dimsSize == 0) {  // means there is one sub-array and it coincides with whole array, return
    // copy of wholeShapeInfo and one zero offset in this case
    memcpy(subArrShapeInfo, wholeShapeInfo, shape::shapeInfoLength(rank) * sizeof(sd::LongType));
    *subArrOffsets = 0;
    return;
  }

  const int subArrRank = keepUnitiesInShape ? rank : rank - dimsSize;

  subArrShapeInfo[0] = subArrRank;                                     // rank
  subArrShapeInfo[2 * subArrRank + 1] = 0;                             // clear (to avoid uninitialized)
  sd::ArrayOptions::copyDataType(subArrShapeInfo, wholeShapeInfo);     // type
  subArrShapeInfo[2 * subArrRank + 3] = shape::order(wholeShapeInfo);  // order

  sd::LongType *shape = new sd::LongType[dimsSize];
  sd::LongType *strides = new sd::LongType[dimsSize];

  for (int k = subArrRank - 1, j = dimsSize - 1, i = rank - 1; i >= 0; --i) {
    if (j >= 0 && i == dimsToExclude[j]) {
      strides[j] = shape::stride(wholeShapeInfo)[i];
      shape[j--] = shape::shapeOf(wholeShapeInfo)[i];

      if (keepUnitiesInShape) {
        shape::shapeOf(subArrShapeInfo)[k] = 1;
        shape::stride(subArrShapeInfo)[k--] = shape::stride(wholeShapeInfo)[i];
      }
    } else {
      shape::shapeOf(subArrShapeInfo)[k] = shape::shapeOf(wholeShapeInfo)[i];
      shape::stride(subArrShapeInfo)[k--] = shape::stride(wholeShapeInfo)[i];
    }
  }

  // calculation of sub-array offsets (subArrOffsets)
  shape::calcOffsets(dimsSize, shape, strides, subArrOffsets);

  // evaluate ews
  shape::checkStridesEwsAndOrder(subArrShapeInfo);

  delete[] strides;
  delete[] shape;
}

//////////////////////////////////////////////////////////////////////
void calcSubArrShapeInfoAndOffset(const sd::LongType *idx, const sd::LongType *maxShapeInfo,
                                  sd::LongType *minShapeInfo, sd::LongType &minOffset,
                                  const bool keepUnitiesInShape, const bool isStrided,
                                  const int numOfUntiesInMinShape) {
  const sd::Unsigned maxRank = shape::rank(maxShapeInfo);
  minOffset = 0;
  sd::Unsigned first, last, stride, n(isStrided ? 3 : 2);

  minShapeInfo[0] = keepUnitiesInShape ? maxRank : maxRank - numOfUntiesInMinShape;

  for (sd::Unsigned step = 0, j = 0, i = 0; i < maxRank; ++i, step += n) {
    if (idx[step] == idx[step + 1]) {  // means whole dimension
      shape::shapeOf(minShapeInfo)[j] = shape::shapeOf(maxShapeInfo)[i];
      shape::stride(minShapeInfo)[j++] = shape::stride(maxShapeInfo)[i];
    } else {
      first = idx[step] >= 0 ? idx[step] : idx[step] + shape::sizeAt(maxShapeInfo, i) + 1;
      last = idx[step + 1] >= 0 ? idx[step + 1] : idx[step + 1] + shape::sizeAt(maxShapeInfo, i) + 1;

      if (last < first) throw("shape::calcSubArrShapeInfoAndOffset: negative range in input indexes is found!");

      if (isStrided) {
        stride = idx[step + 2];
        last /*resulting sub-array axis*/ = (last - first + stride - 1) / stride;  // ceil (last - first) / stride;
      } else {
        stride = 1;
        last /*resulting sub-array axis*/ = last - first;
      }

      minOffset += first * shape::stride(maxShapeInfo)[i];

      if (!keepUnitiesInShape && last == 1) continue;

      shape::shapeOf(minShapeInfo)[j] = last;
      shape::stride(minShapeInfo)[j++] =
          last == 1 ? shape::stride(maxShapeInfo)[i] : shape::stride(maxShapeInfo)[i] * stride;
    }
  }

  minShapeInfo[2 * shape::rank(minShapeInfo) + 1] = 0;                           // zero
  minShapeInfo[2 * shape::rank(minShapeInfo) + 3] = shape::order(maxShapeInfo);  // order
  sd::ArrayOptions::copyDataType(minShapeInfo, maxShapeInfo);                    // type

  shape::checkStridesEwsAndOrder(minShapeInfo);
}

//////////////////////////////////////////////////////////////////////
SD_HOST int excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, sd::LongType *&shapeNoUnities,
                                        sd::LongType *&stridesNoUnities) {
  const int rank = shape::rank(inShapeInfo);
  const int numOfNonUnities = shape::numOfNonUnitDims(rank, shape::shapeOf(inShapeInfo));

  if (numOfNonUnities == rank) {  // no unities in shape, no copy procedure
    shapeNoUnities = const_cast<sd::LongType *>(inShapeInfo) + 1;
    stridesNoUnities = const_cast<sd::LongType *>(inShapeInfo) + 1 + rank;
    return numOfNonUnities;
  }

  for (sd::Unsigned j = 0, i = 0; i < rank; ++i) {
    if (shape::shapeOf(inShapeInfo)[i] != 1) {
      shapeNoUnities[j] = shape::shapeOf(inShapeInfo)[i];
      shapeNoUnities[numOfNonUnities + j++] = shape::stride(inShapeInfo)[i];
    }
  }

  stridesNoUnities = shapeNoUnities + numOfNonUnities;

  return numOfNonUnities;
}

//////////////////////////////////////////////////////////////////////
SD_HOST void excludeUnitiesFromShapeInfo(const sd::LongType *inShapeInfo, const sd::LongType *dimsToExclude,
                                         const int dimsSize, sd::LongType *outShapeInfo) {
  outShapeInfo[0] = inShapeInfo[0] - dimsSize;

  for (sd::Unsigned j = 0, k = 0, i = 0; i < inShapeInfo[0]; ++i) {
    if (j < dimsSize && i == dimsToExclude[j]) {
      ++j;
      continue;
    }

    shape::shapeOf(outShapeInfo)[k] = shape::shapeOf(inShapeInfo)[i];
    shape::stride(outShapeInfo)[k++] = shape::stride(inShapeInfo)[i];
  }
  outShapeInfo[2 * outShapeInfo[0] + 1] = 0;
  sd::ArrayOptions::copyDataType(outShapeInfo, inShapeInfo);          // type
  *shape::ews(outShapeInfo) = shape::elementWiseStride(inShapeInfo);  // ews
  outShapeInfo[2 * outShapeInfo[0] + 3] = shape::order(inShapeInfo);  // order
}

}
