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
// @author Adam Gibson
//

#ifndef LIBND4J_TAD_H
#define LIBND4J_TAD_H

#include <helpers/shape.h>

namespace shape {
/**
 * Dimension collapse is an algorithm
 * for collapsing singular dimensions.
 * This algorithm will adjust the dimensions
 * wrt the original.
 *
 * The algorithm has 3 components:
 * trailing ones
 * middle ones
 * beginning ones
 *
 * dimensions that are specified to reduce along
 * that are singular should be truncated
 *
 * dimensions that are specified that are singular
 * at the beginning should be removed with middle dimensions
 * decremented.
 *
 * For any time there is a no op, a collapse will
 * set the first dimension to be -1.
 *
 *
 */
class TAD {
 public:
  sd::LongType tadIndex = 0;
  sd::LongType dimensionLength;
  sd::LongType *dimension = nullptr;
  sd::LongType const *shapeInfo = nullptr;
  sd::LongType *tadOnlyShapeInfo = nullptr;
  sd::LongType numTads = 0;
  int tadRank = 0;
  sd::LongType *tadShape = nullptr;
  sd::LongType *tadStride = nullptr;
  sd::LongType *tadOffsets = nullptr;
  sd::LongType tadOffsetForBlock = 0;
  int rank = 0;
  int numOnes = 0;
  // pointers to original
  int originalDimensionLength;
  sd::LongType const *originalDimension = nullptr;
  sd::LongType const *originalShapeInfo = nullptr;
  bool squeezed = false;
  bool newSqueezeDimensions = false;
  int numOnesInMiddle = 0;
  bool wholeThing = false;
  // need to track whether we create a new dimension array or not, we could have just moved the pointer forward
  // due to leading ones
  bool createdNewDimension = false;

  // special case for CUDA, we're passing in __shared__ memory pointers to be used instead of new/malloc
  void *ptrManager = nullptr;
  int *ptrOutput = nullptr;

  SD_INLINE bool dimensionsDescending(int rank, const long long int *dimensions, int length);

  SD_INLINE SD_HOST_DEVICE TAD() {}

  SD_INLINE SD_HOST_DEVICE void setExternalBuffers(void *ptrManager);

  SD_INLINE SD_HOST_DEVICE void setOutputBuffer(int *ptrOutput);

  /**
   * This method is for GPU mostly, it allows to initialize TAD instance with precalculated tadOnlyShapeInfo
   */
  SD_INLINE SD_HOST_DEVICE void initWithExternalTAD(sd::LongType *existingTAD, sd::LongType *originalShape,
                                                    sd::LongType *dimension, sd::LongType dimensionLength);

  SD_INLINE SD_HOST_DEVICE void init(sd::LongType const *shapeInfo, const sd::LongType *dimension,
                                     sd::LongType dimensionLength);

  SD_INLINE SD_HOST_DEVICE void init(int tadIndex, sd::LongType const *shapeInfo, const sd::LongType *dimension,
                                     sd::LongType dimensionLength);

  template <typename T>
  SD_INLINE SD_HOST_DEVICE void printTADsND(T *x);

  SD_INLINE SD_HOST_DEVICE void permuteShapeBufferInPlace(sd::LongType const *shapeBuffer,
                                                          const long long int *rearrange,
                                                          sd::LongType *out);

  SD_INLINE SD_HOST_DEVICE sd::LongType *permuteShapeBuffer(sd::LongType const *shapeBuffer, sd::LongType *rearrange);

  SD_INLINE SD_HOST_DEVICE void createTadOnlyShapeInfo();

  SD_INLINE SD_HOST_DEVICE sd::LongType lengthPerSlice(sd::LongType const *shapeBuffer);

  SD_INLINE SD_HOST_DEVICE sd::LongType *tad2Sub(sd::LongType index);

  SD_INLINE SD_HOST_DEVICE ~TAD();

  SD_INLINE SD_HOST_DEVICE int *permuteDims();

  /**
   * Compute the tad offset given a dimension.
   *
   * The general pattern for computing a tad offset is as follows:
   * Every $STRIDE that was removed (the first dimension)
   * do a jump by the major stride of the parent array
   * (stride[0] of the parent array)
   *
   * For example given a c ordered 2,2,3,2 with stride 12,6,2,1
   * A tad of dimension 1 will jump 12 every 6 tads.
   *
   * You then end up with offsets of:
   * 0
   * 1
   * 2
   * 3
   * 4
   * 5
   * 12
   * 13
   * 14
   * 15
   * 16
   * 17
   *
   * notice there are 12 tads here. This same incremental jump will happen
   * every time.
   * Note here that by default the
   * stride of element wise stride is used for the hops.
   *
   * Sometimes a jump doesn't happen. If there are less tads
   * than the stride of the dimension you removed, the
   * element wise stride will always be used.
   *
   * For example in a dimension of 0,1, you end up with offsets of:
   * 0,1,2,3,4,5
   *
   * Given that the inner most stride of the dimensions that was removed (1)
   * had a stride of 6, we never need to do a major stride jump.
   *
   */
  SD_INLINE SD_HOST_DEVICE sd::LongType tadOffset(sd::LongType index);

  SD_INLINE SD_HOST_DEVICE sd::LongType *tensorShape();

  SD_INLINE SD_HOST_DEVICE sd::LongType *tad2Sub(sd::LongType index, void *ptrManager);

  SD_INLINE SD_HOST_DEVICE void createOffsets();

  SD_INLINE SD_HOST_DEVICE sd::LongType *shapeInfoOnlyShapeAndStride();

  /**
   * Length of a tad given
   * the shape information
   */
  SD_INLINE SD_HOST_DEVICE sd::LongType tadLength(sd::LongType const *shapeInfo, const sd::LongType *dimension,
                                                  sd::LongType dimensionLength);

  /**
   * Computes the number
   * of tensors along
   * a given dimension
   */
  SD_INLINE SD_HOST_DEVICE sd::LongType tensorsAlongDimension(sd::LongType const *shapeInfo,
                                                              const sd::LongType *dimension,
                                                              sd::LongType dimensionLength);

#ifdef __CUDACC__
  SD_INLINE SD_HOST_DEVICE void createOffsetForBlock(int blockIdx) {
    this->tadOffsetForBlock = this->tadOffset(blockIdx);
  }
#endif

  SD_INLINE SD_HOST_DEVICE void collapse();
};

SD_INLINE void TAD::setExternalBuffers(void *ptrManager) { this->ptrManager = ptrManager; }

SD_INLINE void TAD::setOutputBuffer(int *ptrOutput) { this->ptrOutput = ptrOutput; }

SD_INLINE void TAD::initWithExternalTAD(sd::LongType *existingTAD, sd::LongType *originalShape,
                                        sd::LongType *dimension, sd::LongType dimensionLength) {
  this->tadOnlyShapeInfo = existingTAD;
  this->rank = shape::rank(originalShape);

  this->originalShapeInfo = originalShape;
  this->originalDimension = dimension;
  this->originalDimensionLength = dimensionLength;

  this->shapeInfo = originalShape;
  this->dimension = dimension;
  this->dimensionLength = dimensionLength;

  this->tadShape = shapeOf(existingTAD);
  this->tadStride = stride(existingTAD);

  sd::LongType ews = elementWiseStride(originalShape);

  this->numTads = length(originalShape) / length(
          existingTAD);
  this->wholeThing =
      this->numTads == 1 ||
      ((this->dimensionLength == this->rank || this->numTads == length(this->shapeInfo)) && ews == 1);
}

SD_INLINE void TAD::init(int tadIndex, sd::LongType const *shapeInfo, const long long int *dimension,
                         long long int dimensionLength) {
  this->tadIndex = tadIndex;
  this->init(shapeInfo, dimension, dimensionLength);
}

SD_INLINE void TAD::init(sd::LongType const *shapeInfo, const long long int *dimension, long long int dimensionLength) {
  this->originalShapeInfo = shapeInfo;
  this->originalDimension = dimension;
  this->originalDimensionLength = dimensionLength;
  // start off as original references
  this->shapeInfo = shapeInfo;
  this->dimensionLength = dimensionLength;
  this->dimension = const_cast<sd::LongType *>(dimension);
  this->rank = shape::rank(shapeInfo);
  this->numTads =
      dimensionLength == 0 ? 1 : this->tensorsAlongDimension(this->shapeInfo, this->dimension, this->dimensionLength);

  sd::LongType ews = elementWiseStride(shapeInfo);

  if (dimensionLength == 0) {
    wholeThing = true;
  } else if (!isVector(shapeInfo)) {
    wholeThing = this->numTads == 1  // if number of TADs is 1, we just have input shape == TAD shape
                 ||
                 ((this->dimensionLength == this->rank  // if number of dimensions is the same as input rank, that'll be
                                                        // wholeTad too, but only if EWS==1 (aka - not a View)
                   || (this->numTads == length(shapeInfo) &&
                          order(shapeInfo) == 'c'))  // OR  number of tads equals to shapeInfo length AND input is
                                                         // in C order. if order is F - we'll have to calculate offsets
                  && ews == 1);                          // as mentioned above - last 2 rules apply only to non-views
  } else if (isScalar(shapeInfo)) {
    wholeThing = true;
    // vector case
  } else {
    // if(dimensionLength == 1 && shape::shapeOf(shapeInfo)[dimension[0]] == 1) {
    // if(dimension == 0 && ) {
    if (dimensionLength != 0 && dimension != nullptr && shapeOf(shapeInfo)[dimension[0]] == 1) {
      wholeThing = true;
    }
  }
}

template <typename T>
SD_INLINE void TAD::printTADsND(T *x) {
  if (wholeThing) {
    for (int i = 0; i < length(tadOnlyShapeInfo); i++) {
      printf(" %f ", x[i]);
    }
    printf("\n");
  } else {
    for (int i = 0; i < numTads; i++) {
      auto offset = tadOffsets[i];
      sd::LongType shapeIter[SD_MAX_RANK];
      sd::LongType coord[SD_MAX_RANK];
      int dim;
      int rankIter = shape::rank(tadOnlyShapeInfo);
      sd::LongType xStridesIter[SD_MAX_RANK];
      T *xPointer = x + offset;
      if (PrepareOneRawArrayIter<T>(rankIter, shapeOf(tadOnlyShapeInfo), xPointer, stride(tadOnlyShapeInfo), &rankIter, shapeIter, &xPointer,
                                    xStridesIter) >= 0) {
        ND4J_RAW_ITER_START(dim, shape::rank(tadOnlyShapeInfo), coord, shapeIter);
        {
          /* Process the innermost dimension */
          printf(" %f ", xPointer[0]);
        }
        ND4J_RAW_ITER_ONE_NEXT(dim, rankIter, coord, shapeIter, xPointer, xStridesIter);
        printf("\n");

      } else {
        printf("Unable to prepare array\n");
      }
    }
  }
}

SD_INLINE void TAD::permuteShapeBufferInPlace(sd::LongType const *shapeBuffer, const sd::LongType *rearrange,
                                              sd::LongType *out) {
  memcpy(out, shapeBuffer, sizeof(sd::LongType) * shapeInfoLength(this->rank));
  doPermuteShapeInfo(out, rearrange);
}

SD_INLINE sd::LongType *TAD::permuteShapeBuffer(sd::LongType const *shapeBuffer, sd::LongType *rearrange) {
  int len = shapeInfoLength(this->rank);
  sd::LongType *copy = copyOf(len, shapeBuffer);
  doPermuteShapeInfo(copy, rearrange);
  return copy;
}

SD_INLINE bool TAD::dimensionsDescending(int rank, const sd::LongType *dimensions, int length) {
  int desired = rank - 1;
  for (int e = length - 1; e >= 0; e--) {
    if (dimensions[e] != desired--) return false;
  }
  return true;
}

SD_INLINE void TAD::createTadOnlyShapeInfo() {
  this->tadOnlyShapeInfo = this->shapeInfoOnlyShapeAndStride();
  sd::ArrayOptions::setDataType(this->tadOnlyShapeInfo, sd::ArrayOptions::dataType(this->originalShapeInfo));

  // possible optimization goes here
  if (order(this->originalShapeInfo) == 'c' && strideDescendingCAscendingF(this->originalShapeInfo) &&
      dimensionsDescending(shape::rank(this->originalShapeInfo), this->originalDimension,
                           this->originalDimensionLength)) {
    // for C order, if outer dimensions are used, continuous layout is preserved
    this->tadOnlyShapeInfo[shapeInfoLength(this->tadOnlyShapeInfo) - 2] =
        this->originalShapeInfo[shapeInfoLength(this->originalShapeInfo) - 2];
  }

  // do not swap order if positive elementwise stride preserved
  if (elementWiseStride(this->tadOnlyShapeInfo) >= 1) {
    this->tadOnlyShapeInfo[shapeInfoLength(this->tadOnlyShapeInfo) - 1] = order(this->originalShapeInfo);
  }

  if (this->tadShape != nullptr) delete[] this->tadShape;

  this->tadShape = shapeOf(this->tadOnlyShapeInfo);
  this->tadStride = stride(this->tadOnlyShapeInfo);
}

SD_INLINE sd::LongType TAD::lengthPerSlice(sd::LongType const *shapeBuffer) {
  int dimension = 0;
  sd::LongType *remove = removeIndex(shapeOf(shapeBuffer), &dimension, shape::rank(shapeBuffer), 1);
  sd::LongType prod = prodLong(remove, shape::rank(shapeBuffer) - 1);
  delete[] remove;
  return prod;
}

SD_INLINE sd::LongType *TAD::tad2Sub(sd::LongType index) {
  sd::LongType *shape = shapeOf(shapeInfo);
  int rank = shape::rank(shapeInfo);
  int leftOverIndexLen = rank - originalDimensionLength;

  sd::LongType *ret = new sd::LongType[rank];
  // shape of the tad
  sd::LongType *tadShape = new sd::LongType[leftOverIndexLen];
  sd::LongType *leftOverIndexes = new sd::LongType[leftOverIndexLen];
  sd::LongType *sub = new sd::LongType[rank];

  // indexes not specified in the tad indexes

  // every coordinate starts as zero
  memset(ret, 0, shapeInfoByteLength(rank));

  // find the length of the elements we
  // are iterating over
  sd::LongType len = 1;
  // left over index cursor for initializing elements
  int leftOverIndex = 0;
  for (int i = 0; i < rank; i++) {
    // look for dimensions NOT found in dimension length (basically compute shape - dimension (set difference)
    bool found = false;
    for (int j = 0; j < originalDimensionLength; j++) {
      // skip over specified dimensions when computing left over length
      if (i == originalDimension[j]) {
        found = true;
        break;
      }
    }

    // add to the indexes that aren't specified as part of the tad dimension
    // indexes
    if (!found) {
      // accumulate the list of indexes left over used for initializing the return value
      leftOverIndexes[leftOverIndex] = i;
      // accumulate the tad shape
      tadShape[leftOverIndex] = shape[i];
      // accumulate the length (product) of the indexes that will be iterated over
      len *= shape[i];
      leftOverIndex++;
    }
  }

  // sub for indices
  index2coords(index, leftOverIndexLen, tadShape, sub);

  for (int i = 0; i < leftOverIndexLen; i++) {
    ret[leftOverIndexes[i]] = sub[i];
  }

  if (ptrManager == nullptr) {
    delete[] tadShape;
    delete[] leftOverIndexes;
    delete[] sub;
  }

  return ret;
}

SD_INLINE TAD::~TAD() {
  // we may have just moved the pointer forward, we may not need to delete the pointer here
  if (originalDimension != this->dimension && createdNewDimension) {
    delete[] this->dimension;
  }
  if (this->originalShapeInfo != this->shapeInfo) {
    delete[] this->shapeInfo;
  }
  if (this->tadOffsets != nullptr) {
    delete[] this->tadOffsets;
  }

  if (this->tadOnlyShapeInfo != nullptr && this->tadOnlyShapeInfo != shapeInfo) {
    delete[] this->tadOnlyShapeInfo;
  }
}

SD_INLINE int *TAD::permuteDims() {
  // permute dimensions for tad
  int dimIdx = 0;
  // loop backwards assuming dimension is sorted

  int *permuteDims = new int[shape::rank(shapeInfo)];

  for (int i = 0; i < shape::rank(shapeInfo); i++) {
    bool found = false;
    for (int j = 0; j < originalDimensionLength; j++) {
      if (i == originalDimension[j]) {
        found = true;
        break;
      }
    }

    // not found, append it to the end for permute
    if (!found) permuteDims[dimIdx++] = i;
  }

  for (int i = originalDimensionLength - 1; i >= 0; i--) {
    permuteDims[dimIdx++] = originalDimension[i];
  }

  /*
              for (int i = 0; i < originalDimensionLength; i++) {
                  permuteDims[i] = originalDimension[i];
              }
  */

  // permute dimensions for tad
  return permuteDims;
}

SD_INLINE sd::LongType TAD::tadOffset(sd::LongType index) {
  if (tadOnlyShapeInfo == nullptr) {
    this->createTadOnlyShapeInfo();
  }

  if (wholeThing) return index;

  if (dimensionLength > 1) {
    sd::LongType *tad2Sub = this->tad2Sub(index, ptrManager);

    sd::LongType ret = getOffset(shapeInfo, tad2Sub);

    if (ret < 0) {
     if (ptrManager == nullptr) delete[] tad2Sub;
      return -1;
    }
   if (ptrManager == nullptr) delete[] tad2Sub;

    return ret;

  } else {
    sd::LongType *tad2Sub = this->tad2Sub(index, ptrManager);

    sd::LongType ret = getOffset(shapeInfo, tad2Sub);

    if (ptrManager == nullptr) delete[] tad2Sub;

    return ret;
  }
}

SD_INLINE sd::LongType *TAD::tensorShape() {
  if (this->tadShape != nullptr) return this->tadShape;

  sd::LongType *theShape = shapeOf(shapeInfo);
  sd::LongType *tensorShape = keep(theShape, this->dimension, dimensionLength, shape::rank(shapeInfo));
  this->tadShape = tensorShape;
  this->tadRank = dimensionLength;
  return tensorShape;
}

SD_INLINE sd::LongType *TAD::tad2Sub(sd::LongType index, void *ptrManager) {
  auto shape = shapeOf(shapeInfo);
  int rank = shape::rank(shapeInfo);
  int leftOverIndexLen = rank - originalDimensionLength;
  sd::LongType *tadShape;
  sd::LongType *leftOverIndexes;
  sd::LongType *sub;
  sd::LongType *ret;

  ret = new sd::LongType[rank];
  // shape of the tad
  leftOverIndexes = new sd::LongType[leftOverIndexLen];
  sub = new sd::LongType[rank];
  tadShape = new sd::LongType[leftOverIndexLen];

  // indexes not specified in the tad indexes

  // every coordinate starts as zero
  memset(ret, 0, sizeof(sd::LongType) * rank);

  // find the length of the elements we
  // are iterating over
  sd::LongType len = 1;
  // left over index cursor for initializing elements
  int leftOverIndex = 0;
  for (sd::LongType i = 0; i < rank; i++) {
    // look for dimensions NOT found in dimension length (basically compute shape - dimension (set difference)
    bool found = false;
    for (int j = 0; j < originalDimensionLength; j++) {
      // skip over specified dimensions when computing left over length
      if (i == originalDimension[j]) {
        found = true;
        break;
      }
    }

    // add to the indexes that aren't specified as part of the tad dimension
    // indexes
    if (!found) {
      // accumulate the list of indexes left over used for initializing the return value
      leftOverIndexes[leftOverIndex] = i;
      // accumulate the tad shape
      tadShape[leftOverIndex] = shape[i];
      // accumulate the length (product) of the indexes that will be iterated over
      leftOverIndex++;
      len *= shape[i];
    }
  }

  // sub for indices
  index2coords(index, leftOverIndexLen, tadShape, sub);

  for (int i = 0; i < leftOverIndexLen; i++) {
    ret[leftOverIndexes[i]] = sub[i];
  }

  if (ptrManager == nullptr) {
    delete[] leftOverIndexes;
    delete[] tadShape;
    delete[] sub;
  }

  return ret;
}

SD_INLINE void TAD::createOffsets() {
  this->tadOffsets = new sd::LongType[this->numTads];
  sd::LongType nT = this->numTads;

  for (sd::LongType i = 0; i < nT; i++) this->tadOffsets[i] = this->tadOffset(i);
}

SD_INLINE sd::LongType *TAD::shapeInfoOnlyShapeAndStride() {

  // ensure tad shapes get setup right for vectors
  if (dimensionLength > 1 && isVector(shapeInfo))
    return copyOf(shapeInfoLength(shape::rank(shapeInfo)), shapeInfo);

  // case when tad coincides with whole array
  if (this->numTads == 1 &&
      ((shape::rank(originalShapeInfo) == originalDimensionLength) || originalDimensionLength == 0)) {
    // we might have special case here: skipped dimensions might be just full of ones
    sd::LongType *ret = copyOf(shapeInfoLength(shape::rank(shapeInfo)), shapeInfo);
    if (shape::isDimPermuted<sd::LongType >(dimension, (sd::LongType)dimensionLength))  // check whether we need permutation
      doPermuteShapeInfo(ret, dimension);

    return ret;
  }

  sd::LongType *theShape = shapeOf(shapeInfo);
  int rank = shape::rank(shapeInfo);

  if (dimensionLength == 1) {
    if (dimension[0] == 0 && isVector(shapeInfo) && theShape[1] == 1) {
      sd::LongType permuted[2] = {1, 0};
      sd::LongType *permutedRet2 = shape::permuteShapeBuffer(shapeInfo, permuted);
      return permutedRet2;
    } else if (dimension[0] == 1 && isVector(shapeInfo) && theShape[0] == 1) {
      sd::LongType *ret = copyOf(shapeInfoLength(shape::rank(shapeInfo)), shapeInfo);
      return ret;
    } else if (shapeOf(shapeInfo)[dimension[0]] == 1) {
      sd::LongType *scalarInfo = createScalarShapeInfo();
      scalarInfo[shapeInfoLength(shape::rank(scalarInfo)) - 3] = this->tadIndex;
      return scalarInfo;
    }
  }

  sd::LongType *tensorShape = this->tensorShape();
  sd::LongType *reverseDimensions = reverseCopy(dimension, dimensionLength);
  sd::LongType  *rankRange = shape::range<sd::LongType >(0, rank);
  sd::LongType  *remove = shape::removeIndex<sd::LongType >(rankRange, dimension, (sd::LongType)rank, (sd::LongType)dimensionLength);
  // concat is wrong here with the length
  sd::LongType *newPermuteDims = concat(remove, rank - dimensionLength, reverseDimensions, dimensionLength);

  sd::LongType *permuted = shape::permuteShapeBuffer(shapeInfo, newPermuteDims);

  sd::LongType sliceIndex = sliceOffsetForTensor(shape::rank(permuted), this->tadIndex, shapeOf(shapeInfo), tensorShape,
                                  dimensionLength, dimension, dimensionLength);

  sd::LongType *ret2 = sliceOfShapeBuffer(sliceIndex, permuted);
  sd::LongType tensorLength = prodLong(tensorShape, tadRank);

  sd::LongType compLength = isVector(ret2) ? length(ret2) : prodLong(tensorShape, tadRank);
  if (dimensionLength == tadRank && compLength == length(ret2)) {
    if (dimensionLength == 1 && isVector(ret2) && shapeOf(ret2)[0] == 1) {
      // go to the bottom and return ret2 after proper freeing of pointers
      // basic idea; we *don't* permute row vectors
    } else if (dimensionLength > 1) {
      // permute *then* return ret2
      sd::LongType  *finalPermuteDims = new sd::LongType [shape::rank(ret2)];
      int forward = 0;
      for (int i = shape::rank(ret2) - 1; i >= 0; i--) {
        finalPermuteDims[forward++] = i;
      }
      shape::permuteShapeBufferInPlace(ret2, finalPermuteDims, ret2);
      delete[] finalPermuteDims;
    }

  } else {
    sd::LongType length = tensorLength;
    sd::LongType lengthPerSlice = this->lengthPerSlice(ret2);
    if (lengthPerSlice < 1) {
      return ret2;
    }

    sd::LongType offset = tadIndex * tensorLength / lengthPerSlice;
    if (sliceIndex == 0 && length == lengthPerSlice) {
      sd::LongType *newRet2 = sliceOfShapeBuffer(offset, ret2);
      delete[] ret2;
      ret2 = newRet2;
      sd::LongType  *finalPermuteDims = new sd::LongType [shape::rank(ret2)];
      int forward = 0;
      for (int i = shape::rank(ret2) - 1; i >= 0; i--) {
        finalPermuteDims[forward++] = i;
      }
      // bool isRowVector2 = shape::isRowVector(ret2) && !isLikeVector;
      bool isRowVector2 = isRowVector(ret2);
      if (isRowVector2 == false) {
        shape::permuteShapeBufferInPlace(ret2, finalPermuteDims, ret2);
      }

      delete[] finalPermuteDims;

    } else if (length == lengthPerSlice) {
      offset -= slices(ret2) * (offset / slices(ret2));
      sd::LongType *newRet2 = sliceOfShapeBuffer(offset, ret2);
      delete[] ret2;
      ret2 = newRet2;
      if (dimensionLength == 1 && isVector(ret2) && shapeOf(ret2)[0] == 1) {
        // go to the bottom and return ret2 after proper freeing of pointers
        // basic idea; we *don't* permute row vectors
      } else {
        sd::LongType *finalPermuteDims = new sd::LongType [shape::rank(ret2)];
        int forward = 0;
        for (int i = shape::rank(ret2) - 1; i >= 0; i--) {
          finalPermuteDims[forward++] = i;
        }
        sd::LongType *newRet = shape::permuteShapeBuffer(ret2, finalPermuteDims);
        delete[] ret2;
        delete[] finalPermuteDims;
        ret2 = newRet;
      }

    } else {
      // execute final part, note that this is mainly so delete[] gets called
      // at the bottom of the method
      while (shape::length(ret2) > length) {
        auto lengthPerSlice2 = this->lengthPerSlice(ret2);
        sliceIndex = sliceOffsetForTensor(sliceIndex, shape::length(ret2), lengthPerSlice2);
        sliceIndex -= slices(ret2) * (sliceIndex / slices(ret2));
        auto newRet2 = sliceOfShapeBuffer(sliceIndex, ret2);
        delete[] ret2;
        ret2 = newRet2;
      }

      // don't permute on a row vector
      if (dimensionLength == 1 && isVector(ret2) && shapeOf(ret2)[0] == 1) {
        // go to the bottom and return ret2 after proper freeing of pointers
        // basic idea; we *don't* permute row vectors
      } else if (dimensionLength > 1) {
        // permute *then* return ret
        sd::LongType *finalPermuteDims = new sd::LongType[shape::rank(ret2)];
        int forward = 0;
        for (int i = shape::rank(ret2) - 1; i >= 0; i--) {
          finalPermuteDims[forward++] = i;
        }
        auto newPermute = shape::permuteShapeBuffer(ret2, finalPermuteDims);
        delete[] ret2;
        delete[] finalPermuteDims;
        ret2 = newPermute;
      }
    }
  }

  delete[] permuted;
  delete[] newPermuteDims;
  delete[] rankRange;
  delete[] remove;
  delete[] reverseDimensions;
  return ret2;
}

SD_INLINE sd::LongType TAD::tadLength(sd::LongType const *shapeInfo, const sd::LongType *dimension,
                                      sd::LongType dimensionLength) {
  if (dimensionLength == 1) {
    return shapeOf(shapeInfo)[dimension[0]];
  } else {
    sd::LongType ret = 1;
    for (int i = 0; i < shape::rank(shapeInfo); i++) {
      for (int j = 0; j < dimensionLength; j++) {
        if (i == dimension[j]) ret *= shapeOf(shapeInfo)[dimension[j]];
      }
    }
    return ret;
  }
}

SD_INLINE sd::LongType TAD::tensorsAlongDimension(sd::LongType const *shapeInfo, const sd::LongType *dimension,
                                                  sd::LongType dimensionLength) {
  return length(shapeInfo) / this->tadLength(shapeInfo, dimension, dimensionLength);
}

SD_INLINE void TAD::collapse() {
  auto shape = shapeOf(shapeInfo);
  // handle negative dimensions/backwards indexing
  for (int i = 0; i < dimensionLength; i++) {
    if ((dimension)[i] < 0) (dimension)[i] += shape::rank(this->shapeInfo);
  }

  this->dimension = new sd::LongType[dimensionLength];
  memcpy(this->dimension, this->originalDimension, sizeof(int) * dimensionLength);

  // we can drop trailing dimensions where it's all singular for example:
  // shape: 4,3,1,2
  // dimension: 0,2
  // the problem for 0,2 is equivalent to: 0
  // the rest of the algorithm handles cases suchas
  // shape: 4,1,1,2
  // dimension: 0,1
  // when this happens there are other dimensions (eg: at the end) that matter
  int trailingOneDimensions = 0;
  // trailing ones
  for (int i = dimensionLength - 1; i >= 0; i--) {
    if (shape[dimension[i]] != 1) {
      break;
    } else if (shape[dimension[i]] == 1)
      trailingOneDimensions++;
  }

  dimensionLength -= trailingOneDimensions;

  int leadingOneDimensions = 0;
  // trailing ones
  for (int i = 0; i < dimensionLength; i++) {
    if (shape[dimension[i]] != 1) {
      break;
    } else if (shape[dimension[i]] == 1)
      leadingOneDimensions++;
  }

  // bump the dimension pointer forward for however many leadingones there are
  dimension += leadingOneDimensions;
  // decrease the dimension length by the amount of leading ones
  dimensionLength -= leadingOneDimensions;

  bool preConverged = true;
  for (int i = 0; i < dimensionLength; i++) {
    if (shape[dimension[i]] == 1) {
      preConverged = false;
      break;
    }
  }

  // we took away all the singular dimensions, we can just return
  if (preConverged) return;

  // no more singular dimensions specified
  bool done = false;
  int onesDecrement = 0;
  bool changed = false;
  while (!done) {
    // terminate early: only singular dimensions specified for reduce
    if ((dimensionLength) < 1) {
      done = true;
      // signal as a no op
      dimension[0] = -1;
      break;
    }


    int intermediaryResult[SD_MAX_RANK];
    for (int i = 0; i < dimensionLength; i++) {
      intermediaryResult[i] = (dimension)[i];
    }

    bool oneEncountered = false;
    bool nonOneEncountered = false;
    bool hitBeginning = false;
    // assume intermediate collapsing of dimensions
    bool collapseMiddleDimensions = true;
    // note here that dimension length MAY end up being zero
    for (int i = (dimensionLength)-1; i >= 0; i--) {
      if (shape[(dimension)[i]] == 1) {
        oneEncountered = true;
        // trailing ones
        if (!nonOneEncountered) {
          // just drop trailing ones
          dimensionLength--;
          nonOneEncountered = false;
          collapseMiddleDimensions = false;
          // intermediary result just needs to have the results copied from dimension since we're just removing the tail
          memcpy(intermediaryResult, dimension, sizeof(int) * dimensionLength);
          changed = true;
          // break the for loop and force it to go back around starting from the new index
          break;
        } else {
          // already decremented all dimensions
          // this was a result of hitting beginning ones
          // we will only need to loop once
          if (i == 0) {
            hitBeginning = true;
          }
          // will need to shift dimensions that aren't trailing ones
          // back by onesDecrement
          // mark the intermediary result as -1 for non inclusion
          intermediaryResult[i] = -1;
          onesDecrement++;
        }
      } else {
        intermediaryResult[i] = (dimension)[i];
        nonOneEncountered = true;
      }
    }

    if (collapseMiddleDimensions && oneEncountered) {
      // collapse dimensions
      int newIntermediary[SD_MAX_RANK];
      int idx = 0;
      for (int i = 0; i < dimensionLength; i++) {
        // of note: dimension will decrease by the number of ones encountered
        if (intermediaryResult[i] >= 0) {
          // dimension 0 doesn't need to be decremented
          if (intermediaryResult[i] > 0)
            newIntermediary[idx++] = intermediaryResult[i] - onesDecrement;
          else
            newIntermediary[idx++] = intermediaryResult[i];
        }
      }

      // decrement by the number of dimensions where ones appeared
      (dimensionLength) -= onesDecrement;
      // update to current result
      memcpy(dimension, newIntermediary, sizeof(int) * (dimensionLength));
      changed = true;

    }
    // converged: no need to change result
    else {
      // update to current result
      memcpy(dimension, intermediaryResult, sizeof(int) * dimensionLength);
    }

    // converge when there are no singular dimensions specified in the reduce
    done = (!oneEncountered && nonOneEncountered) || hitBeginning;
  }

  // nothing changed but need to collapse dimension
  if (!changed && this->numOnes > 0) {
    for (int i = 0; i < dimensionLength; i++) {
      dimension[i] -= numOnes;
    }
  }
}
}  // namespace shape

#endif  // LIBND4J_TAD_H
