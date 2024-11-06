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

#include <ops/gemm.h>
#include <ops/ops.h>

#include <array/NDArray.hXX>
#include <array/ArrayOptions.hXX>

#include <memory>
#include <sstream>
#include <stdexcept>

namespace sd {



////////////////////////////////////////////////////////////////////////

void* NDArray::platformBuffer() { return buffer(); }

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::fillAsTriangular(const float val, int lower, int upper, NDArray& target, const char direction,const bool includeEdges) {
  if (isS()) THROW_EXCEPTION("NDArray::fillArrayAsTriangular: you can't use this method on String array!");

  if (!isSameShape(target) &&
      !(rankOf() == 1 && target.rankOf() == 2 && sizeAt(0) == target.sizeAt(0) && sizeAt(0) == target.sizeAt(1)))
    throw std::string("NDArray::fillArrayAsTriangular method: wrong shape of target array !");




  const T value = static_cast<T>(val);
  const auto x = reinterpret_cast<const T*>(buffer());
  auto z = reinterpret_cast<T*>(target.buffer());

  const int xRank = rankOf();
  const int zRank = target.rankOf();

  const auto zLen = target.lengthOf();

  const bool areSameOffsets = shape::haveSameShapeAndStrides(shapeInfo(), target.shapeInfo());

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;
    //track input vector coordinates, only used in specific cases
    sd::LongType vectorCoord[1];
    vectorCoord[0] = 0;
    sd::LongType targetRank = target.rankOf();
    sd::LongType thisRank = this->rankOf();
    bool notVectorScalar = targetRank == 2 && thisRank == 2;
    bool thisNotVectorScalar = !shape::isScalar(this->shapeInfo()) && !shape::isVector(this->shapeInfo());
    bool targetNotVectorScalar =  !shape::isScalar(target.shapeInfo()) && !shape::isVector(target.shapeInfo());
    for (sd::LongType i = start; i < stop; i++) {
      shape::index2coordsCPU(start, i, target.shapeInfo(), coords);
      sd::LongType row = targetNotVectorScalar ?  coords[zRank - 2] : 0;
      sd::LongType col = targetNotVectorScalar ? coords[zRank - 1]: 1;
      auto zOffset = 0;
      auto xOffset = 0;
      if(target.rankOf() < 2) {
        zOffset = shape::getOffset(target.shapeInfo(),vectorCoord);
      } else {
        zOffset = shape::getOffset(target.shapeInfo(), coords);
      }

      if(!areSameOffsets && rankOf() < 2) {
        //rotate count of elements based on how many times accessed
        xOffset = shape::getOffset(shapeInfo(),vectorCoord);
      } else if(areSameOffsets) {
        xOffset = zOffset;
      } else {
        xOffset = shape::getOffset(shapeInfo(), coords);
      }


      bool rowExclusive = this->rankOf() == target.rankOf();
      bool colExclusive = this->rankOf() == target.rankOf();
      auto lCompare = includeEdges ? row <= (col - lower) : row < (col - lower);
      auto uCompare = includeEdges ? row >= (col - upper): row > (col - upper);
      if (direction == 'u' && lCompare  || direction == 'l' && uCompare ) {
        z[zOffset] = value;
      }  else  {
        z[zOffset] = x[xOffset];
      }

      if(this != &target) {
        if (xRank != zRank) {
          temp = coords[0];
          coords[0] = coords[1];
        }

        if (xRank != zRank)  // restore first coordinate
          coords[0] = temp;
      }

      if(vectorCoord[0] == this->lengthOf() - 1) {
        vectorCoord[0] = (int) 0;
      } else {
        vectorCoord[0] = vectorCoord[0] + 1;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, zLen);
}
BUILD_SINGLE_TEMPLATE(template void NDArray::fillAsTriangular,
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

  sd::LongType offset = shape::getOffset(shapeInfo(), indices);

  for (int i = 0; i < rank; ++i)
    if (minDim > shape[i]) minDim = shape[i];

  float v = 1.0f;

  for (int i = 0; i < minDim; ++i) templatedSet<float>(buffer(), i * offset, this->dataType(), &v);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void templatedSwap(void* xBuffer, void* yBuffer, const sd::LongType* xShapeInfo, const sd::LongType* yShapeInfo,
                          sd::LongType length) {
  auto x = reinterpret_cast<T*>(xBuffer);
  auto y = reinterpret_cast<T*>(yBuffer);

  const bool isSameOrders = shape::order(xShapeInfo) == shape::order(xShapeInfo);

  const auto xEws = shape::elementWiseStride(xShapeInfo);
  const auto yEws = shape::elementWiseStride(yShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    if (isSameOrders && xEws > 0 && yEws > 0) {
      for (sd::LongType i = start; i < stop; i++) sd::math::sd_swap(x[i * xEws], y[i * yEws]);
    } else if (shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
      for (sd::LongType i = start; i < stop; i++) {
        const auto ind = shape::getIndexOffset(i, xShapeInfo);
        sd::math::sd_swap(x[ind], y[ind]);
      }
    } else {
      for (sd::LongType i = start; i < stop; i++) {
        const auto xInd = shape::getIndexOffset(i, xShapeInfo);
        const auto yInd = shape::getIndexOffset(i, yShapeInfo);
        sd::math::sd_swap(x[xInd], y[yInd]);
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}
BUILD_SINGLE_TEMPLATE(template void templatedSwap,
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

#if defined(HAVE_VEDA)

void NDArray::syncToDevice() const {}

void NDArray::syncToHost() const { 
  _buffer->syncToPrimary(getContext()); 
  }

void NDArray::tickWriteHost() const { _buffer->writePrimary(); }
void NDArray::tickWriteDevice() const { _buffer->writeSpecial(); }
void NDArray::tickReadHost() const { _buffer->readPrimary(); }
void NDArray::tickReadDevice() const { _buffer->readSpecial(); }

void NDArray::tickBothActual() const {
  _buffer->writePrimary();
  _buffer->readSpecial();
}
bool NDArray::isActualOnHostSide() const { return _buffer->isPrimaryActual(); }
bool NDArray::isActualOnDeviceSide() const { return _buffer->isSpecialActual(); }
void NDArray::makeBothBuffersActual() const {
  if (!isActualOnHostSide()) syncToHost();
  if (!isActualOnDeviceSide()) syncToDevice();
}


//logic to defer buffer sync between the host and veda devices

void NDArray::synchronize(const char* msg) const {
  // no-op
}


////////////////////////////////////////////////////////////////////////
void NDArray::preparePrimaryUse(const std::vector<const NDArray*>& writeList,
                                const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
  for (const auto& a : readList)
    if (a) a->syncToHost();

  for (const auto& a : writeList) {
    if (a) {
      a->getDataBuffer()->allocatePrimary();
      if (synchronizeWritables) a->syncToHost();
      // by ticking the write counter we inform that it was taken for the writing purpose by the host operation
      // furethemore, we do it beforehand, as there could be the situation where device op is used inside
      // if such case happens then the last usage will be done by the device operation
      a->tickWriteHost();
    }
  }
}

////////////////////////////////////////////////////////////////////////
void NDArray::registerPrimaryUse(const std::vector<const NDArray*>& writeList,
                                 const std::vector<const NDArray*>& readList) {
  // as noted above for some edge cases we decided to use counters and sync inside 
  // preparePrimaryUse
  // though registerPrimaryUse will be no op, it will be called to be on par with the cuda codes
}


#else

void NDArray::synchronize(const char* msg) const {
  // no-op
}

void NDArray::syncToDevice() const {}
void NDArray::syncToHost() const {}
void NDArray::tickWriteHost() const {}
void NDArray::tickWriteDevice() const {}
void NDArray::tickReadHost() const {}
void NDArray::tickReadDevice() const {}
void NDArray::tickBothActual() const {}
bool NDArray::isActualOnHostSide() const { return true; }
bool NDArray::isActualOnDeviceSide() const { return true; }
void NDArray::makeBothBuffersActual() const {}

void NDArray::preparePrimaryUse(const std::vector<const NDArray*>& writeList,
                                const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
  // no-op
}
void NDArray::registerPrimaryUse(const std::vector<const NDArray*>& writeList,
                                 const std::vector<const NDArray*>& readList) {
  // no-op
}

#endif

void NDArray::prepareSpecialUse(const std::vector<const NDArray*>& writeList,
                                const std::vector<const NDArray*>& readList, bool synchronizeWritables) {
  // no-op
}
void NDArray::registerSpecialUse(const std::vector<const NDArray*>& writeList,
                                 const std::vector<const NDArray*>& readList) {
  // no-op
}

void NDArray::syncShape() const {
  // no-op
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::printCurrentBuffer(const bool host, const char* msg, const int precision) const {}
template void NDArray::printCurrentBuffer<int>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<float>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<double>(const bool host, const char* msg, const int precision) const;
template void NDArray::printCurrentBuffer<sd::LongType>(const bool host, const char* msg, const int precision) const;


////////////////////////////////////////////////////////////////////////
void* NDArray::specialBufferWithOffset(sd::LongType offset) { return nullptr; }

////////////////////////////////////////////////////////////////////////
const void* NDArray::specialBufferWithOffset(sd::LongType offset) const { return nullptr; }

////////////////////////////////////////////////////////////////////////
void* NDArray::specialBuffer() {
  if (!_buffer->special()) return nullptr;
  // FIXME: this should be fixed once CUDA backend added
  return static_cast<int8_t*>(_buffer->special()) + (_offset * sizeOfT());
}


const void* NDArray::specialBufferWithOffset(LongType offset) {
  return specialBuffer() != nullptr ? static_cast<int8_t*>(specialBuffer()) + (offset * sizeOfT()) : nullptr;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
NDArray NDArray::tile(const std::vector<sd::LongType>& reps) const {
  const int repsSize = reps.size();

  sd::LongType product = 1;
  for (const auto& item : reps) product *= item;
  if (product == 0) THROW_EXCEPTION("NDArray::tile method: one of the elements in reps array is zero !");

  int rankOld = rankOf();
  int diff = rankOld - repsSize;
  if (product == 1) {  // in this case 2 possibilities are present: just reshape or nothing to do
    NDArray result(*this);
    if (diff < 0) {  // reshape to higher dimension
      std::vector<sd::LongType> shapeNew =
          reps;  // there is requirement to have unities at first "diff" positions of new shape
      memcpy(&shapeNew[-diff], result.shapeInfo() + 1,
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
  auto desc = new ShapeDescriptor(newShapeInfo);
  // assign new shape and new buffer to resulting array
  NDArray result(newBuff,desc , getContext());
  // fill newBuff, loop through all elements of newBuff
  // looping through _buffer goes automatically by means of getSubArrayIndex applying
  const auto resultLen = result.lengthOf();
  auto xType = this->dataType();
  if (result.ordering() == 'c') {  //  ews == 1 always here

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto yOffset = shape::subArrayOffset(i, newShapeInfo, shapeInfo());
        BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign, (result.buffer(), i, this->buffer(), yOffset),
                              SD_COMMON_TYPES);
      }
    };

    samediff::Threads::parallel_for(func, 0, resultLen);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto xOffset = result.getOffset(i);
        auto yOffset = shape::subArrayOffset(i, newShapeInfo, shapeInfo());
        BUILD_SINGLE_SELECTOR(xType, this->template templatedAssign,
                              (result.buffer(), xOffset, this->buffer(), yOffset), SD_COMMON_TYPES);
      }
    };

    samediff::Threads::parallel_for(func, 0, resultLen);
  }
  result.tickWriteHost();
  return result;
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tile(const std::vector<sd::LongType>& reps, NDArray& target) const {
  auto repProd = shape::prodLong(reps.data(), reps.size());
  if (repProd < 1) THROW_EXCEPTION("NDArray::tile: reps can't contain 0s");

  // evaluate true tile shapeInfo for comparison with target shapeInfo
  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  if (!shape::equalsSoft(newShapeInfo, target.shapeInfo())) {
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
  }

  // fill newBuff, loop through all elements of newBuff
  // looping through _buffer goes automatically by means of getSubArrayIndex applying
  const int ews = target.ews();
  const auto targetLen = target.lengthOf();
  if (target.ordering() == 'c' && ews == 1) {  //  ews == 1 always here
    //#pragma omp parallel for simd if(targetLen > Environment::getInstance().elementwiseThreshold()) schedule(guided)
    for (sd::LongType i = 0; i < targetLen; ++i) {
      auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
      BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                            (target.buffer(), i, buffer(), yOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
    }
  } else if (target.ordering() == 'c' && ews > 1) {
    for (sd::LongType i = 0; i < targetLen; ++i) {
      auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
      BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                            (target.buffer(), i * ews, buffer(), yOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
    }
  } else {
    for (sd::LongType i = 0; i < targetLen; ++i) {
      auto xOffset = target.getOffset(i);
      auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
      BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                            (target.buffer(), xOffset, buffer(), yOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target) const {
  if (rankOf() > target.rankOf())
    THROW_EXCEPTION(
        "NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

  if (!ShapeUtils::areShapesBroadcastable(*this, target))
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

  // fill newBuff, loop through all elements of newBuff
  // looping through _buffer goes automatically by means of getSubArrayIndex applying
  const auto ews = target.ews();
  const auto targetLen = target.lengthOf();
  if (target.ordering() == 'c' && ews >= 1) {
    for (sd::LongType i = 0; i < targetLen; ++i) {
      auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
      BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                            (target.buffer(), i * ews, buffer(), yOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
    }
  } else {
    for (sd::LongType i = 0; i < targetLen; ++i) {
      auto xOffset = target.getOffset(i);
      auto yOffset = shape::subArrayOffset(i, target.shapeInfo(), shapeInfo());
      BUILD_DOUBLE_SELECTOR(target.dataType(), dataType(), templatedDoubleAssign,
                            (target.buffer(), xOffset, buffer(), yOffset), SD_COMMON_TYPES, SD_COMMON_TYPES);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void repeat_(const NDArray& input, NDArray& output, const std::vector<LongType>& repeats, const LongType axis) {
  const X* x = input.bufferAsT<X>();
  Z* z = output.bufferAsT<Z>();

  const sd::LongType rank = input.rankOf();     // xRank = zRank
  const sd::LongType zLen = output.lengthOf();  // xLen <= zLen
  const sd::LongType repSize = repeats.size();

  // loop through input array
  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK], temp;

    for (sd::LongType i = start; i < stop; i++) {
      shape::index2coordsCPU(start, i, output.shapeInfo(), coords);
      const auto zOffset = shape::getOffset(output.shapeInfo(), coords);

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

      z[zOffset] = x[shape::getOffset(input.shapeInfo(), coords)];

      coords[axis] = temp;
    }
  };

  samediff::Threads::parallel_for(func, 0, zLen);
}

//////////////////////////////////////////////////////////////////////////
// create new array by repeating it the number of times given by repeats
NDArray NDArray::repeat(const int axis, const std::vector<LongType>& repeats) const {
  NDArray *thisArr = const_cast<NDArray*>(this);
  std::vector<sd::LongType> repeatShape = ShapeUtils::evalRepeatShape(axis, repeats, *thisArr);
  NDArray output('c',repeatShape, dataType(), getContext());

  BUILD_SINGLE_SELECTOR_TWICE(dataType(), repeat_, (*this, output, repeats, axis), SD_COMMON_TYPES);

  return output;
}

//////////////////////////////////////////////////////////////////////////
// fill array by repeating it the number of times given by reps
void NDArray::repeat(const int axis, const std::vector<LongType>& repeats, NDArray& target) const {
  NDArray *thisArr = const_cast<NDArray*>(this);
  if (!target.isSameShape(ShapeUtils::evalRepeatShape(axis, repeats, *thisArr)))
    THROW_EXCEPTION(
        "NDArray::repeat(const int axis, const std::vector<int>& repeats, NDArray& target) method: wrong shape of "
        "target array!");

  BUILD_DOUBLE_SELECTOR(dataType(), target.dataType(), repeat_, (*this, target, repeats, axis), SD_COMMON_TYPES,
                        SD_COMMON_TYPES);
}

//////////////////////////////////////////////////////////////////////////
#ifndef __JAVACPP_HACK__
#include "NDArrayLambda.hpp"

#endif

/*
#ifndef __CLION_IDE__
#include "NDArray.macro"
#endif
 */
}  // namespace sd

#endif
