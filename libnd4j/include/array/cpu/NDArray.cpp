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

#ifndef NDARRAY_CPP
#define NDARRAY_CPP
#include <array/NDArray.h>

#include <helpers/ArrayUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/logger.h>

#include <indexing/NDIndex.h>
#include <loops/BroadcastPairwiseConverter.h>

#include <loops/random.h>

#include <ops/ops.h>

#include <array/NDArray.hXX>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <system/selective_rendering.h>
namespace sd {


// NDArray implementation for .cpp file
void NDArray::printBufferDebug(const char* msg, sd::LongType offset, sd::LongType limit) {
  if (msg) sd_printf("%s:\n", msg);
  if(limit < 0) limit = lengthOf();

  // Print array info
  sd_printf("NDArray: Shape=[", 0);
  for (int i = 0; i < rankOf(); i++) {
    sd_printf("%lld", (long long)sizeAt(i));
    if (i < rankOf() - 1) sd_printf(",", 0);
  }
  sd_printf("], DataType=%s, EWS=%lld, Order=%c\n",
            DataTypeUtils::asString(dataType()).c_str(), (long long)ews(), ordering());

  // Print buffer state
  if (_buffer != nullptr) {
    _buffer->printBufferDebug("Buffer contents", offset, limit);
  } else {
    sd_printf("Buffer is nullptr\n", 0);
  }
}
////////////////////////////////////////////////////////////////////////

void* NDArray::platformBuffer() { return buffer(); }

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::fillAsTriangular(const float val, int lower, int upper, NDArray& target, const char direction, const bool includeEdges) {
  if (isS()) THROW_EXCEPTION("NDArray::fillArrayAsTriangular: you can't use this method on String array!");

  if (!isSameShape(target) &&
      !(rankOf() == 1 && target.rankOf() == 2 && sizeAt(0) == target.sizeAt(0) && sizeAt(0) == target.sizeAt(1)))
    THROW_EXCEPTION("NDArray::fillArrayAsTriangular method: wrong shape of target array !");

  const T value = static_cast<T>(val);
  const auto x = reinterpret_cast<const T*>(buffer());
  auto z = reinterpret_cast<T*>(target.buffer());

  const sd::LongType xRank = rankOf();
  const sd::LongType zRank = target.rankOf();

  const auto zLen = target.lengthOf();

  const bool areSameOffsets = shape::haveSameShapeAndStrides(shapeInfo(), target.shapeInfo());

  sd::LongType *targetShape = shape::shapeOf(target.shapeInfo());
  sd::LongType *targetStride = shape::stride(target.shapeInfo());
  sd::LongType *xStride = shape::stride(shapeInfo());

  const bool dirU = direction == 'u';
  const bool dirL = direction == 'l';

  // the triangle is the one of each matrix made of the last two dimensions of the target; a vector is one row
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType xCoords[SD_MAX_RANK];

    for (sd::LongType i = start; i < stop; i++) {
      INDEX2COORDS(i, zRank, targetShape, coords);
      const sd::LongType row = zRank >= 2 ? coords[zRank - 2] : 0;
      const sd::LongType col = zRank >= 1 ? coords[zRank - 1] : 0;

      sd::LongType zOffset;
      COORDS2INDEX(zRank, targetStride, coords, zOffset);

      const bool lCompare = includeEdges ? row <= (col - lower) : row < (col - lower);
      const bool uCompare = includeEdges ? row >= (col - upper) : row > (col - upper);

      if ((dirU && lCompare) || (dirL && uCompare)) {
        z[zOffset] = value;
      } else {
        sd::LongType xOffset;
        if (areSameOffsets) {
          xOffset = zOffset;
        } else if (xRank < zRank) {
          // a vector repeated in every row of a square matrix: the element of the column
          xCoords[0] = col;
          COORDS2INDEX(xRank, xStride, xCoords, xOffset);
        } else {
          COORDS2INDEX(xRank, xStride, coords, xOffset);
        }
        z[zOffset] = x[xOffset];
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, zLen);
}
BUILD_SINGLE_TEMPLATE( void NDArray::fillAsTriangular,
                      (const float val, int lower, int upper, NDArray& target, const char direction,const bool includeEdges), SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::setIdentity() {
  if (isS()) THROW_EXCEPTION("NDArray::setIdentity: you can't use this method on String array!");

  this->nullify();

  int rank = rankOf();
  auto shape = shapeOf();
  int minDim = SD_MAX_INT;
  sd::LongType indices[SD_MAX_RANK];
  for (int j = 0; j < rank; ++j) indices[j] = 1;

  sd::LongType offset;
  COORDS2INDEX(rank, shape::stride(shapeInfo()), indices, offset);

  for (int i = 0; i < rank; ++i)
    if (minDim > shape[i]) minDim = shape[i];

  float v = 1.0f;

  for (int i = 0; i < minDim; ++i) templatedSet<float,float>(buffer(), i * offset, this->dataType(), &v);

  tickWriteHost();
  syncToDevice();
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void templatedSwap(void* xBuffer, void* yBuffer, const sd::LongType* xShapeInfo, const sd::LongType* yShapeInfo,
                          sd::LongType length) {
  auto x = reinterpret_cast<T*>(xBuffer);
  auto y = reinterpret_cast<T*>(yBuffer);

  const bool isSameOrders = shape::order(xShapeInfo) == shape::order(xShapeInfo);

  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType yRank = shape::rank(yShapeInfo);
  sd::LongType *xShape = shape::shapeOf(xShapeInfo);
  sd::LongType *yShape = shape::shapeOf(yShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);
  sd::LongType *yStride = shape::stride(yShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    if (isSameOrders) {
      for (sd::LongType i = start; i < stop; i++) {
        LongType xCoords[SD_MAX_RANK];
        LongType yCoords[SD_MAX_RANK];
        LongType xOffset;
        LongType yOffset;

        INDEX2COORDS(i, xRank, xShape, xCoords);
        COORDS2INDEX(xRank, shape::stride(xShapeInfo), xCoords, xOffset);
        INDEX2COORDS(i, yRank,yShape, yCoords);
        COORDS2INDEX(yRank, yStride, yCoords, yOffset);

        sd::math::sd_swap(x[xOffset], y[yOffset]);
      }
    } else if (shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
      for (sd::LongType i = start; i < stop; i++) {
        LongType coords[SD_MAX_RANK];
        LongType ind;

        INDEX2COORDS(i, xRank, xShape, coords);
        COORDS2INDEX(xRank, xStride, coords, ind);

        sd::math::sd_swap(x[ind], y[ind]);
      }
    } else {
      for (sd::LongType i = start; i < stop; i++) {
        LongType xCoords[SD_MAX_RANK];
        LongType yCoords[SD_MAX_RANK];
        LongType xInd;
        LongType yInd;

        INDEX2COORDS(i, xRank, xShape, xCoords);
        COORDS2INDEX(xRank, xStride, xCoords, xInd);
        INDEX2COORDS(i, yRank, yShape, yCoords);
        COORDS2INDEX(yRank, yStride, yCoords, yInd);

        sd::math::sd_swap(x[xInd], y[yInd]);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}
BUILD_SINGLE_TEMPLATE( void templatedSwap,
                      (void* xBuffer, void* yBuffer, const sd::LongType* xShapeInfo, const sd::LongType* yShapeInfo,
                          sd::LongType length),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::swapUnsafe(NDArray& other) {
  auto xType = this->dataType();

  if (xType != other.dataType())
    THROW_EXCEPTION("NDArray::swapUnsage method: both arrays must have the same data type");

  if (buffer() == nullptr || other.buffer() == nullptr)
    THROW_EXCEPTION("NDArray::swapUnsafe method: input array should not be empty!");

  if (lengthOf() != other.lengthOf())
    THROW_EXCEPTION("NDArray::swapUnsafe method: input arrays should have the same length!");

  BUILD_SINGLE_SELECTOR(xType, templatedSwap,
                        (buffer(), other.buffer(), shapeInfo(), other.shapeInfo(), this->lengthOf()), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////


void NDArray::synchronize(const char* msg)  {
  // no-op
}

void NDArray::synchronizeExecStream(const char* msg)  {
  // no-op on CPU
}

void NDArray::syncToDevice()  {}
void NDArray::syncToHost()  {}
void NDArray::forceSyncToHost()  {}
void NDArray::tickWriteHost()  {}
void NDArray::tickWriteDevice()  {}
void NDArray::tickReadHost()  {}
void NDArray::tickReadDevice()  {}
void NDArray::tickBothActual()  {}
bool NDArray::isActualOnHostSide()  { return true; }
bool NDArray::isActualOnDeviceSide()  { return true; }
void NDArray::makeBothBuffersActual()  {}

void NDArray::preparePrimaryUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList, bool synchronizeWritables) {
  // no-op
}
void NDArray::registerPrimaryUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  // no-op
}


void NDArray::prepareSpecialUse(const std::vector<NDArray*>& writeList,
                                const std::vector<NDArray*>& readList, bool synchronizeWritables) {
  // no-op
}
void NDArray::registerSpecialUse(const std::vector<NDArray*>& writeList,
                                 const std::vector<NDArray*>& readList) {
  // no-op
}

void NDArray::syncShape()  {
  // no-op
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision)  {}
template void NDArray::printCurrentBuffer<int>(const bool host, const char* msg, const int precision) ;
template void NDArray::printCurrentBuffer<float>(const bool host, const char* msg, const int precision) ;
template void NDArray::printCurrentBuffer<double>(const bool host, const char* msg, const int precision);
template void NDArray::printCurrentBuffer<sd::LongType>(const bool host, const char* msg, const int precision) ;

////////////////////////////////////////////////////////////////////////
void* NDArray::specialBuffer() {
  // On CPU, special buffer is always nullptr (no device memory).
  // Return nullptr for both null _buffer and null special — callers
  // already check for nullptr return (e.g. DSP code checks before use).
  if (_buffer == nullptr) {
    return nullptr;
  }

  void* specialBuf = _buffer->special();
  if (specialBuf == nullptr) {
    return nullptr;
  }

  return static_cast<int8_t*>(specialBuf) + (_offset * sizeOfT());
}


//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<sd::LongType>& reps)  {
  const int repsSize = reps.size();

  sd::LongType product = 1;
  for (const auto& item : reps) product *= item;
  if (product == 0) THROW_EXCEPTION("NDArray::tile method: one of the elements in reps array is zero !");

  int rankOld = rankOf();
  int diff = rankOld - repsSize;
  if (product == 1) {  // in this case 2 possibilities are present: just reshape or nothing to do
    // a copy: the copy constructor would give a view sharing this array's buffer
    std::vector<sd::LongType> shapeOld(shapeOf(), shapeOf() + rankOld);
    NDArray result(ordering(), shapeOld, dataType(), getContext());
    result.assign(this);
    if (diff < 0) {  // reshape to higher dimension
      std::vector<sd::LongType> shapeNew =
          reps;  // there is requirement to have unities at first "diff" positions of new shape
      memcpy(&shapeNew[-diff], shapeOld.data(),
             rankOld * sizeof(sd::LongType));  // put old shape numbers at rest of positions
      result.reshapei(ordering(), shapeNew);
    }
    return result;  // nothing to do, if diff >= 0 -> identity tile
  }

  // evaluate shapeInfo for resulting array
  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  // create new buffer, in any case the memory amount new buffer points to is bigger then those for old _buffer
  DataBuffer * newBuff =
      new DataBuffer(shape::length(newShapeInfo) * sizeOfT(), dataType(), getContext()->getWorkspace());
  // assign new shape and new buffer to resulting array
  NDArray result(newBuff,newShapeInfo , getContext());
  // fill newBuff, loop through all elements of newBuff
  // looping through _buffer goes automatically by means of getSubArrayIndex applying
  const auto resultLen = result.lengthOf();
  auto xType = this->dataType();
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto xOffset = result.getOffset(i);
      auto yOffset = shape::subArrayOffset(i, newShapeInfo, shapeInfo());
      BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign,
                            (result.buffer(), xOffset, this->buffer(), yOffset), SD_COMMON_TYPES);
    }
  };

  samediff::Threads::parallel_for(func, 0, resultLen);
  result.tickWriteHost();
  return result;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tile(const std::vector<LongType>& reps, NDArray& target) {
  // Validate the tile operation
  auto repProd = shape::prodLong(reps.data(), reps.size());
  if (repProd < 1)
    THROW_EXCEPTION("NDArray::tile: reps can't contain 0s");

  // Validate the target shape
  auto correctShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  if (!shape::equalsSoft(correctShapeInfo, target.shapeInfo())) {
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation!");
  }

  const auto targetLen = target.lengthOf();

  // Safely calculate source array offset
  for (LongType i = 0; i < targetLen; ++i) {
    // Relative target offset: buffer() below is already offset-shifted, while
    // getOffset() returns the absolute offset (includes the view offset).
    auto xOffset = target.getOffset(i) - target.offset();

    // Calculate source coordinates based on target coordinates
    LongType targetCoords[SD_MAX_RANK];
    INDEX2COORDS(i, shape::rank(target.shapeInfo()), shape::shapeOf(target.shapeInfo()), targetCoords);

    // Map target coordinates to source coordinates manually
    LongType sourceCoords[SD_MAX_RANK];
    for (int d = 0; d < shape::rank(shapeInfo()); d++) {
      // Apply modulo for each dimension
      sourceCoords[d] = targetCoords[d] % shape::sizeAt(shapeInfo(), d);
    }

    // Calculate source offset from source coordinates
    LongType sourceOffset;
    COORDS2INDEX(shape::rank(shapeInfo()), shape::stride(shapeInfo()), sourceCoords, sourceOffset);

    auto targetDataType = target.dataType();
    auto selfDType = dataType();
    // Copy the value
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                          (target.buffer(), xOffset, buffer(), sourceOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
  }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target)  {
  if (rankOf() > target.rankOf())
    THROW_EXCEPTION(
        "NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

  if (!ShapeUtils::areShapesBroadcastable(*this, target))
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

  // fill newBuff, loop through all elements of newBuff
  // looping through _buffer goes automatically by means of getSubArrayIndex applying
  const auto targetLen = target.lengthOf();
  for (sd::LongType i = 0; i < targetLen; ++i) {
    // Relative target offset: buffer() is offset-shifted, getOffset() is absolute.
    auto xOffset = target.getOffset(i) - target.offset();
    auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
    BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                          (target.buffer(), xOffset, buffer(), yOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void repeat_(NDArray& input, NDArray& output, const std::vector<LongType>& repeats, const LongType axis) {
  const X* x = input.bufferAsT<X>();
  Z* z = output.bufferAsT<Z>();

  const sd::LongType rank = input.rankOf();     // xRank = zRank
  const sd::LongType zLen = output.lengthOf();  // xLen <= zLen
  const sd::LongType repSize = repeats.size();

  sd::LongType outputRank = output.rankOf();
  sd::LongType* outputShape = shape::shapeOf(output.shapeInfo());
  sd::LongType* outputStride = shape::stride(output.shapeInfo());

  sd::LongType inputRank = input.rankOf();
  sd::LongType* inputShape = shape::shapeOf(input.shapeInfo());
  sd::LongType* inputStride = shape::stride(input.shapeInfo());

  // loop through input array
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i++) {
      INDEX2COORDS(i, outputRank, outputShape, coords);
      sd::LongType zOffset;
      COORDS2INDEX(outputRank, outputStride, coords, zOffset);

      temp = coords[axis];

      if (repSize > 1) {
        for (sd::LongType j = 0; j < repSize; ++j) {
          coords[axis] -= repeats[j];
          if (coords[axis] < 0) {
            coords[axis] = j;
            break;
          }
        }
      } else
        coords[axis] /= repeats[0];

      sd::LongType xOffset;
      COORDS2INDEX(inputRank,inputStride, coords, xOffset);

      z[zOffset] = x[xOffset];

      coords[axis] = temp;
    }
  };

  samediff::Threads::parallel_for(func, 0, zLen);
}

//////////////////////////////////////////////////////////////////////////
// create new array by repeating it the number of times given by repeats
NDArray NDArray::repeat(const int axis, const std::vector<LongType>& repeats)  {
  NDArray *thisArr = const_cast<NDArray*>(this);
  std::vector<sd::LongType> repeatShape = ShapeUtils::evalRepeatShape(axis, repeats, *thisArr);
  NDArray output('c',repeatShape, dataType(), getContext());

  BUILD_SINGLE_SELECTOR_TWICE(dataType(), repeat_, (*this, output, repeats, axis), SD_COMMON_TYPES);

  return output;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
void NDArray::repeat(const int axis, const std::vector<LongType>& repeats, NDArray& target)  {
  NDArray *thisArr = const_cast<NDArray*>(this);
  if (!target.isSameShape(ShapeUtils::evalRepeatShape(axis, repeats, *thisArr)))
    THROW_EXCEPTION(
        "NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) method: wrong shape of "
        "target array!");
  auto targetDataType = target.dataType();
  auto selfDType = dataType();
  BUILD_DOUBLE_SELECTOR(dataType(), target.dataType(), repeat_, (*this, target, repeats, axis), SD_COMMON_TYPES,
                        SD_COMMON_TYPES);
}

//////////////////////////////////////////////////////////////////////////
#ifndef __JAVACPP_HACK__
#include "NDArrayLambda.hpp"

#endif


}  // namespace sd

#endif
