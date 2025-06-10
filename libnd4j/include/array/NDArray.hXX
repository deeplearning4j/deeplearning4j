/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

// $NDArray.hpp - architech-independent implementations (both cuda and cpu).
//
#ifndef __NDARRAY__HPP__
#define __NDARRAY__HPP__
#include <array/ShapeDescriptor.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <loops/BroadcastPairwiseConverter.h>
#include <array/NDArrayFactory.h>
#include <exceptions/allocation_exception.h>
#include <exceptions/datatype_exception.h>
#include <helpers/MmulHelper.h>
#include <helpers/threshold.h>
#include <indexing/IndicesList.h>
#include <legacy/NativeOpExecutioner.h>
#include <loops/broadcasting.h>
#include <loops/pairwise_transform.h>
#include <loops/transform_same.h>
#include <memory/MemoryRegistrator.h>
#include <memory/Workspace.h>
#include <system/op_boilerplate.h>
//controls precision when printing to strings on floats see:
//https://stackoverflow.com/questions/11989374/floating-point-format-for-stdostream
#include <iomanip>

#include "execution/Threads.h"
#include "helpers/ShapeUtils.h"
#include "system/buffer.h"
#include <helpers/reshapeNoCopy.h>
#include <array/NDArrayHelpers.hXX>

namespace sd {

template <>
SD_LIB_EXPORT utf8string NDArray::e( sd::LongType i);
template <>
SD_LIB_EXPORT std::string NDArray::e( sd::LongType i);
template <>
SD_LIB_EXPORT std::u16string NDArray::e( sd::LongType i);
template <>
SD_LIB_EXPORT std::u32string NDArray::e( sd::LongType i);


template <>
std::u32string NDArray::e(sd::LongType i)  {
  if (!isS()) THROW_EXCEPTION("Can't get std::u32string out of non-string array");


  if (this->dataType() == DataType::UTF8) {
    auto u = this->e<std::string>(i);
    std::u32string s;
    StringUtils::u8StringToU32String(u, s);
    return s;
  }

  if (this->dataType() == DataType::UTF16) {
    auto u16 = this->e<std::u16string>(i);
    std::u32string s;
    StringUtils::u16StringToU32String(u16, s);
    return s;
  }

  NDArray::preparePrimaryUse({}, {this});

  auto offsets = bufferAsT<sd::LongType>();
  sd::LongType offsetsLength = ShapeUtils::stringBufferHeaderRequirements(isScalar() ? 1 : lengthOf());
  sd::LongType start = offsets[i];
  sd::LongType end = offsets[i + 1];

  auto data = bufferAsT<int8_t>() + offsetsLength + start;

  std::u32string r(reinterpret_cast<const char32_t *>(data), (end - start) / sizeof(char32_t));

  registerPrimaryUse({}, {this});

  return r;
}

LongType NDArray::getOffset(const LongType i) {
  if (!isScalar() && i >= lengthOf()) {
    std::string errorMessage;
    errorMessage += "Requested index is out of range: [";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += "] vs ";
    errorMessage += StringUtils::valueToString<sd::LongType>(lengthOf());
    errorMessage += " on array with shape ";
    errorMessage += ShapeUtils::shapeAsString(shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());
  }


  return _offset + i;

}

template <>
std::string NDArray::e(const sd::LongType i)  {
  if (!isS()) THROW_EXCEPTION("Can't get std::string out of non-string array");

  if (!isScalar() && i >= lengthOf()) {
    std::string errorMessage;
    errorMessage += "Requested index is out of range: [";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += "] vs ";
    errorMessage += StringUtils::valueToString<sd::LongType>(lengthOf());
    errorMessage += " on array with shape ";
    errorMessage += ShapeUtils::shapeAsString(shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (this->dataType() == DataType::UTF16) {
    auto u16 = this->e<std::u16string>(i);
    std::string s;
    StringUtils::u16StringToU8String(u16, s);
    return s;
  }

  if (this->dataType() == DataType::UTF32) {
    auto u32 = this->e<std::u32string>(i);
    std::string s;
    StringUtils::u32StringToU8String(u32, s);
    return s;
  }

  NDArray::preparePrimaryUse({}, {this});

  auto offsets = bufferAsT<sd::LongType>();
  auto offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
  auto start = offsets[i];
  auto end = offsets[i + 1];
  auto data = bufferAsT<int8_t>() + offsetsLength + start;

  std::string r(reinterpret_cast<const char *>(data), (end - start));

  registerPrimaryUse({}, {this});

  return r;
}

template <>
std::u16string NDArray::e(sd::LongType i)  {
  if (!isS()) THROW_EXCEPTION("Can't get std::u16string out of non-string array");

  if (i == lengthOf()) THROW_EXCEPTION("Can't get std::u16string for index out of range");

  if (this->dataType() == DataType::UTF8) {
    auto u = this->e<std::string>(i);
    std::u16string s;
    StringUtils::u8StringToU16String(u, s);
    return s;
  }

  if (this->dataType() == DataType::UTF32) {
    auto u32 = this->e<std::u32string>(i);
    std::u16string s;
    StringUtils::u32StringToU16String(u32, s);
    return s;
  }

  NDArray::preparePrimaryUse({}, {this});

  auto offsets = bufferAsT<sd::LongType>();
  sd::LongType offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
  sd::LongType start = offsets[i];
  sd::LongType end = offsets[i + 1];
  auto data = bufferAsT<int8_t>() + offsetsLength + start;

  std::u16string r(reinterpret_cast<const char16_t *>(data), (end - start) / sizeof(char16_t));

  registerPrimaryUse({}, {this});

  return r;
}



SD_INLINE void prepareUse(const std::vector<NDArray *> &writeList, const std::vector<NDArray *> &readList,
                          bool synchronizeWritables = false) {
  NDArray::prepareSpecialUse(writeList, readList, synchronizeWritables);
#endif
}

SD_INLINE void registerUse(const std::vector<NDArray *> &writeList,
                           const std::vector<NDArray *> &readList) {

  NDArray::registerSpecialUse(writeList, readList);
}

////////////////////////////////////////////////////////////////////////
// copy constructor
NDArray::NDArray(const NDArray &other): NDArray(const_cast<NDArray&>(other)) {
}
NDArray::NDArray(NDArray &other) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(NDArray &other) - copy constructor \n");
#if defined(SD_GCC_FUNCTRACE)
    Printer p;
    StackTrace st;
    st.load_here();
    p.print(st);
#endif
    fflush(stdout);
  }

#ifndef __JAVACPP_HACK__
#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif
#endif
  _context = other._context;
  //we should always set an array as a view with the copy constructor
  if(!shape::isViewConst(other.shapeInfo())) {
    auto copyInfo = ShapeBuilders::copyShapeInfo(other.shapeInfo(), true, getContext() == nullptr ? nullptr : getContext()->getWorkspace());
    auto copiedInfo = ShapeBuilders::setAsView(copyInfo);
    auto shapeInfo = ConstantShapeHelper::getInstance().createConstBuffFromExisting(copiedInfo);
    if(shapeInfo == nullptr) {
      THROW_EXCEPTION("NDArray::NDArray(NDArray &other) copy constructor - shapeInfo is nullptr");
    }
    _shapeInfo = shapeInfo->primary();
    _shapeInfoD = shapeInfo->special();
  } else {
    if(other._shapeInfo == nullptr) {
#if defined(SD_GCC_FUNCTRACE)
      printf("Other shape info is nullptr\n");
      Printer p;
      p.print(other.creationTrace);
      printf("End creation trace of other\n");
#endif
      THROW_EXCEPTION("NDArray::NDArray(NDArray &other) copy constructor - other shapeInfo is nullptr");
    }


    _shapeInfo = other._shapeInfo;
    _shapeInfoD = other._shapeInfoD;
  }


  //scalar can be length 0
  if ((!isEmpty() && other.isScalar()) || other.lengthOf() > 0) {
    _buffer = other._buffer;
  } else {
    _buffer = new DataBuffer();
  }

  _offset = other._offset;
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(const char order, std::vector<sd::LongType> &shape, sd::DataType dtype,
                 sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const char order, const std::vector<sd::LongType> &shape, sd::DataType dtype, sd::LaunchContext *context) - constructor 2\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

  if ((int)shape.size() > SD_MAX_RANK) THROW_EXCEPTION("Rank of NDArray can't exceed 32");

  _context = context;
  _isAttached = _context->getWorkspace() != nullptr;

  if (shape.empty()) {
    //scalar
    auto desc = ShapeDescriptor::scalarDescriptor(dtype);
    if(desc->dataType() != dtype) {
      THROW_EXCEPTION("New data type is not reflected in the created descriptor");
    }

    setShapeInfo(desc);


  } else {
    auto desc = ShapeBuilders::createShapeInfo(dtype,order,shape);
    auto desc2 = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    setShapeInfo(desc2);
  }

  int len = isScalar() ? 1 : lengthOf();

  _buffer =
      new DataBuffer(len * DataTypeUtils::sizeOf(dtype), dtype, getContext()->getWorkspace());
  _buffer->setToZeroBuffers();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(char order, std::vector<LongType> &shape,  std::vector<double> &data, DataType dtype,
                 LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const char order, const std::vector<sd::LongType> &shape, const std::vector<double> &data, sd::DataType dtype, sd::LaunchContext *context) - constructor 3\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

  if (shape.size() > SD_MAX_RANK) THROW_EXCEPTION("Rank of NDArray can't exceed 32");

  if(dtype == DataType::UNKNOWN) {
    THROW_EXCEPTION("Unable to create array with unknown data type.");
  }

  _context = context;

  if (shape.size() == 0) {
    if (data.size() == 0) {
      auto desc = ShapeDescriptor::emptyDescriptor(dtype);
      setShapeInfo(desc);
    } else {
      auto desc = ShapeDescriptor::scalarDescriptor(dtype);
      setShapeInfo(desc);
    }
  } else {
    auto desc = new ShapeDescriptor(dtype, order, shape);
    setShapeInfo(desc);
  }

  if (static_cast<size_t>(lengthOf()) != data.size()) {
    std::string errorMessage;
    errorMessage += "NDArray constructor: data size [" + std::to_string(data.size()) +
                    "] doesn't match shape length [" + std::to_string(lengthOf()) + "]";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  int len = isScalar() ? 1 : lengthOf();
  _buffer = new DataBuffer(len * DataTypeUtils::sizeOf(dtype), dtype, getContext()->getWorkspace(),
                           true);

  for (sd::LongType i = 0; i < len; ++i) {
    BUILD_SINGLE_PARTIAL_SELECTOR(
        dtype, templatedDoubleAssign<, double>(buffer(), i, reinterpret_cast< void *>(data.data()), i),
        SD_COMMON_TYPES_ALL);
  }
  tickWriteHost();
  syncToDevice();
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(NDArray *other, const bool copyStrides, sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(NDArray *other, const bool copyStrides, sd::LaunchContext *context) - constructor 4\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;

  if (copyStrides) {
    auto desc2 = ConstantShapeHelper::getInstance().createFromExisting(other->_shapeInfo);
    setShapeInfo(desc2);
  } else {
    auto newDesc = ShapeBuilders::createShapeInfo(other->dataType(), other->ordering(), other->rankOf(),
                                                  other->shapeOf(), getContext()->getWorkspace(), false);
    auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(newDesc);
    _shapeInfo = constDesc->primary();
    _shapeInfoD = constDesc->special();

  }

  int len = isScalar() ? 1 : lengthOf();
  if (!isEmpty()) {
    _buffer = new DataBuffer(other->getDataBuffer()->primary(),
                             other->getDataBuffer()->special()
        , len * DataTypeUtils::sizeOf(other->dataType()), other->dataType(),
                             false,false,
                             getContext()->getWorkspace());
  }
}

////////////////////////////////////////////////////////////////////////
NDArray::NDArray(void *buffer, char order, std::vector<LongType> &shape, DataType dtype, LaunchContext *context, const bool isBuffAlloc) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(void *buffer, const char order, const std::vector<sd::LongType> &shape, sd::DataType dtype, sd::LaunchContext *context, const bool isBuffAlloc) - constructor 5\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

  if ((int)shape.size() > SD_MAX_RANK) THROW_EXCEPTION("Rank of NDArray can't exceed 32");

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto desc = ShapeBuilders::createShapeInfo(dtype, order, shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  int len = isScalar() ? 1 : lengthOf();
  _buffer = new DataBuffer(buffer, len * sizeOfT(), dataType(), isBuffAlloc,
                           getContext()->getWorkspace());
}

NDArray::NDArray(void *buffer, const char order, const std::vector<sd::LongType> &shape, sd::DataType dtype,
                 sd::LaunchContext *context, const bool isBuffAlloc, const bool isView, sd::LongType offset) {
  sd_print("NDArray::NDArray(void *buffer, const char order, const std::vector<sd::LongType> &shape, sd::DataType dtype, sd::LaunchContext *context, const bool isBuffAlloc, const bool isView, sd::LongType offset) - constructor 6\n");
  if ((int)shape.size() > SD_MAX_RANK) THROW_EXCEPTION("Rank of NDArray can't exceed 32");
#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif
  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto desc = ShapeBuilders::createShapeInfo(dtype, order, shape.size(), shape.data(), getContext()->getWorkspace(),
                                             false);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  int len = isScalar() ? 1 : lengthOf();
  _buffer = new DataBuffer(buffer, len * sizeOfT(), dataType(), isBuffAlloc,
                           getContext()->getWorkspace());
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros
NDArray::NDArray(sd::LongType *shapeInfo, const sd::DataType dtype, const bool copyStrides,
                 sd::LaunchContext *context, const bool nullify) {

  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const sd::LongType *shapeInfo, const sd::DataType dtype, const bool copyStrides, sd::LaunchContext *context, const bool nullify) - constructor 7\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

  if (shapeInfo == nullptr) THROW_EXCEPTION("NDArray constructor: can't be initialized without shapeinfo");

  if (shapeInfo[0] < 0 || shapeInfo[0] > SD_MAX_RANK) {
    std::string errorMessage;
    errorMessage += "NDArray constructor: rank of NDArray can't exceed 32 or be < 0 !";
    errorMessage += "Provided rank: " + std::to_string(shapeInfo[0]);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  _context = context;

  if (copyStrides) {
    auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
    _shapeInfo = constDesc->primary();
    _shapeInfoD = constDesc->special();
  } else {
    auto desc = ShapeBuilders::createShapeInfo(dtype, shape::order(shapeInfo), shape::rank(shapeInfo),
                                               shape::shapeOf(const_cast<sd::LongType *>(shapeInfo)),
                                               getContext()->getWorkspace(), false);

    if(desc[0] < 0 || desc[0] > SD_MAX_RANK)
      THROW_EXCEPTION("NDArray constructor: rank of NDArray can't exceed 32 or be < 0 !");
    auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    if(desc[0] < 0 || desc[0] > SD_MAX_RANK)
      THROW_EXCEPTION("NDArray constructor: rank of NDArray can't exceed 32 or be < 0 !");
    _shapeInfo = constDesc->primary();
    _shapeInfoD = constDesc->special();
  }

  if (!isEmpty()) {
    _length = shape::length(_shapeInfo);
    LongType len = isScalar() ? 1 : lengthOf();
    _buffer = new DataBuffer(len * DataTypeUtils::sizeOf(ArrayOptions::dataType(_shapeInfo)), dtype, getContext()->getWorkspace());

    if (nullify) _buffer->setToZeroBuffers();
  }
}

////////////////////////////////////////////////////////////////////////
// scalar constructor
NDArray::NDArray(sd::DataType dtype, sd::LaunchContext *context, const bool isScalar) {
  if (Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(sd::DataType dtype, sd::LaunchContext *context, const bool isScalar) - constructor 8\n");
    fflush(stdout);
  }

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif
  if (isScalar) {
    _length = 1;
    auto desc = ShapeBuilders::createScalarShapeInfo(dtype, getContext()->getWorkspace());
    auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    this->_shapeInfo = constDesc->primary();
    if(dataType() == DataType::UNKNOWN) {
      THROW_EXCEPTION("Unable to create array for unknown data type.");
    }

    this->_shapeInfoD = constDesc->special();
    this->_buffer = new DataBuffer(1 * sizeOfT(), dtype, getContext()->getWorkspace(),true);
    this->_buffer->setToZeroBuffers();
  } else {
    auto constDesc = ConstantShapeHelper::getInstance().emptyShapeInfo(dtype);
    this->_shapeInfo = constDesc;
    this->_shapeInfoD = nullptr;
  }
}
//////////////////////////////////////////////////////////////////////////
// move constructor
NDArray::NDArray(NDArray &&other) noexcept {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(NDArray &&other) - constructor 9\n");
    fflush(stdout);
  }


  _buffer = other._buffer;
  _shapeInfoBuffer = other._shapeInfoBuffer;
  if(other._shapeInfo == nullptr) {
    THROW_EXCEPTION("NDArray::NDArray(NDArray &&other) - other._shapeInfo is nullptr");
  }

  _shapeInfo = other._shapeInfo;
  _shapeInfoD = other._shapeInfoD;
  _context = other._context;
  _length = other._length;

  other._buffer = new DataBuffer();
  other._shapeInfo = other._shapeInfoD = nullptr;
  other._length = 0;

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif
}

////////////////////////////////////////////////////////////////////////
// constructor, create empty array at given workspace
NDArray::NDArray(sd::LaunchContext *context) {
  sd_print("NDArray::NDArray(sd::LaunchContext *context) - constructor 10\n");
  fflush(stdout);

  _buffer = new DataBuffer();
  _shapeInfoBuffer = nullptr;
  _shapeInfo = nullptr;
  _shapeInfoD = nullptr;
  _context = context;
  _length = 0;

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif
}

////////////////////////////////////////////////////////////////////////
// creates new NDArray using shape information from "shapeInfo" array, set all elements in new array to be zeros, set
// dtype as array type
NDArray::NDArray(sd::LongType *shapeInfo, const bool copyStrides, sd::LaunchContext *context, const bool nullify)
    : NDArray(shapeInfo, ArrayOptions::dataType(shapeInfo), copyStrides, context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const sd::LongType *shapeInfo, const bool copyStrides, sd::LaunchContext *context, const bool nullify) - constructor 11\n");
    fflush(stdout);
  }
#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

}

#ifndef __JAVACPP_HACK__

/**
   *  default destructor
 */
NDArray::~NDArray() {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::~NDArray() - destructor printing creation trace:\n");
    fflush(stdout);
  }
  this->_shapeInfo = nullptr;
  this->_buffer = nullptr;
  this->_context = nullptr;
  this->_shapeInfoBuffer = nullptr;
  this->_shapeInfoBuffer = nullptr;
}

NDArray::NDArray(DataBuffer *  buffer, const char order, const std::vector<sd::LongType> &shape,
                 sd::DataType dtype, sd::LaunchContext *context, const bool isBuffAlloc, const bool isView,
                 sd::LongType offset) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(DataBuffer *  buffer, const char order, const std::vector<sd::LongType> &shape, sd::DataType dtype, sd::LaunchContext *context, const bool isBuffAlloc, const bool isView, sd::LongType offset) - constructor 12\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif


  if ((int)shape.size() > SD_MAX_RANK) THROW_EXCEPTION("Rank of NDArray can't exceed 32");

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto newDesc = ShapeBuilders::createShapeInfo(dtype, order, shape.size(), shape.data(), getContext()->getWorkspace(), false);
  auto constShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(newDesc);
  this->_shapeInfo = constShapeInfo->primary();
  this->_shapeInfoD = constShapeInfo->special();
  _buffer = buffer;
}

////////////////////////////////////////////////////////////////////////

NDArray::NDArray(DataBuffer *buffer,
                 sd::LongType *shapeInfo,
                 sd::LaunchContext *context,
                 const sd::LongType offset) {
  if(!shape::isEmptyConst(shapeInfo) && buffer == nullptr) {
    THROW_EXCEPTION("NDArray::NDArray(DataBuffer *  buffer, sd::LongType *shapeInfo, sd::LaunchContext *context, const sd::LongType offset) - buffer can't be nullptr !");
  }
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
#if defined(SD_GCC_FUNCTRACE)
    creationTrace = StackTrace();
    creationTrace.load_here();
#endif
    sd_print("NDArray::NDArray(DataBuffer *  buffer, sd::LongType *shapeInfo, sd::LaunchContext *context, const sd::LongType offset) - constructor 13\n");
    fflush(stdout);

  }

  if(buffer->getNumElements() < 1 && !shape::isEmptyConst(shapeInfo)) {
    THROW_EXCEPTION("NDArray::NDArray(DataBuffer *  buffer, sd::LongType *shapeInfo, sd::LaunchContext *context, const sd::LongType offset) - buffer is empty !");
  }


  _context = context;
  _buffer = buffer;
  _length = shape::length(shapeInfo);
  _offset = offset;
  auto constShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
  if(!shape::strideEquals(shapeInfo, constShapeInfo->primary())) {
    THROW_EXCEPTION("NDArray::NDArray(DataBuffer *  buffer, sd::LongType *shapeInfo, sd::LaunchContext *context, const sd::LongType offset) - strideEquals failed !");
  }

  this->_shapeInfoBuffer = constShapeInfo;
  this->_shapeInfo = constShapeInfo->primary();
  this->_shapeInfoD = constShapeInfo->special();
  if(this->_shapeInfo == nullptr) {
    THROW_EXCEPTION("NDArray::NDArray(DataBuffer *  buffer, sd::LongType *shapeInfo, sd::LaunchContext *context, const sd::LongType offset) - shapeInfo is nullptr");
  }
  if(buffer == nullptr) {
    _length = 0;
  }
}

NDArray::NDArray(DataBuffer *buffer, ShapeDescriptor *descriptor, sd::LaunchContext *context,
                 const sd::LongType offset) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(DataBuffer *  buffer, ShapeDescriptor *descriptor, sd::LaunchContext *context, const sd::LongType offset) - constructor 14\n");
    fflush(stdout);
  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif
  _context = context;
  if(descriptor->dataType() == DataType::UNKNOWN) {
    THROW_EXCEPTION("Unable to create array with unknown data type.");
  }

  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor->toShapeInfo());
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  _buffer = buffer;
  _length = shape::length(_shapeInfo);

}


////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
NDArray::NDArray(void *buffer, sd::LongType *shapeInfo, sd::LaunchContext *context, const bool isBuffAlloc,
                 sd::LongType offset) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(void *buffer, const sd::LongType *shapeInfo, sd::LaunchContext *context, const bool isBuffAlloc) - constructor 16\n");
    fflush(stdout);

  }

#if defined(SD_GCC_FUNCTRACE)
  creationTrace = StackTrace();
  creationTrace.load_here();
#endif

  if (shapeInfo == nullptr) THROW_EXCEPTION("NDArray constructor: can't be initialized without shapeinfo !");

  if ((int)shapeInfo[0] > SD_MAX_RANK) THROW_EXCEPTION("NDArray constructor: rank of NDArray can't exceed 32 !");

  _offset = offset;
  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto constShapeBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
  this->_shapeInfo = constShapeBuffer->primary();
  this->_shapeInfoD = constShapeBuffer->special();

  if (this->isEmpty()) {
    tickReadDevice();
    tickReadHost();
  } else {
    int len = isScalar() ? 1 : lengthOf();
    _buffer = new DataBuffer(buffer, len * sizeOfT(), dataType(), isBuffAlloc,
                             getContext()->getWorkspace());
  }
}

////////////////////////////////////////////////////////////////////////
// do not allocate memory, memory for array is passed from outside
// we suppose the content of both (device and host) buffers is identical
NDArray::NDArray(void *buffer, void *bufferD, sd::LongType *shapeInfo, sd::LaunchContext *context,
                 const bool isBuffAlloc, const bool isBuffDAlloc, sd::LongType offset) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(void *buffer, void *bufferD, const sd::LongType *shapeInfo, sd::LaunchContext *context, const bool isBuffAlloc, const bool isBuffDAlloc) - constructor 17\n");
    fflush(stdout);
  }

  _offset = offset;


  if (shapeInfo == nullptr) THROW_EXCEPTION("NDArray constructor cuda: can't be initialized without shapeinfo");

  sd::LongType rank = shapeInfo[0];
  if (rank > SD_MAX_RANK || rank < 0) THROW_EXCEPTION("NDArray constructor: rank of NDArray can't exceed 32");
  _context = context;
  _length = shape::length(shapeInfo);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  _buffer = new DataBuffer(buffer,bufferD, _length * sizeOfT(), dataType(), isBuffAlloc, isBuffDAlloc,
                           _context == nullptr ? nullptr : getContext()->getWorkspace());


}
//////////////////////////////////////////////////////////////////////////
NDArray::NDArray(DataBuffer *buffer, char order, std::vector<LongType> &shape, LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(DataBuffer *  buffer, const char order, const std::vector<sd::LongType> &shape, sd::LaunchContext *context) - constructor 18\n");
    fflush(stdout);
  }


  if (shape.empty()) {
    THROW_EXCEPTION("NDArray constructor: input shape is empty !");
  }
  if ((int)shape.size() > SD_MAX_RANK) THROW_EXCEPTION("NDArray constructor: rank of NDArray can't exceed 32");

  _context = context;

  auto desc = ShapeBuilders::createShapeInfo(buffer->getDataType(), order, shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  _buffer = buffer;

}
/////////////////////////////////////////////////////////////////////////
// u16 string constructors
NDArray::NDArray(const std::u16string &u16string, sd::DataType dtype, sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::u16string &u16string, sd::DataType dtype, sd::LaunchContext *context) - constructor 19\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dtype)) {
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");
  }

  if (!unicode::isStringValidU16(u16string.data(), u16string.data() + u16string.size())) {
    THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
  }

  // one word that is why used 1
  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

  sd::LongType dataLength = [&] {
    if (dtype == DataType::UTF16) {
      return static_cast<sd::LongType>(u16string.size() * sizeof(uint16_t));
    }
    if (dtype == DataType::UTF32) {
      return unicode::offsetUtf16StringInUtf32(u16string.data(), u16string.size());
    }
    return unicode::offsetUtf16StringInUtf8(u16string.data(), u16string.size());
  }();

  sd::LongType offsets[2] = {0, dataLength};

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto desc = ShapeBuilders::createScalarShapeInfo(dtype);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  memcpy(bufferAsT<int8_t>(), &offsets[0], 2 * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);
  if (dtype == DataType::UTF8) {
    unicode::utf16to8(u16string.data(), data, u16string.size());
  } else if (dtype == DataType::UTF16) {
    memcpy(data, u16string.data(), dataLength);
  } else {
    unicode::utf16to32(u16string.data(), data, u16string.size());
  }

  tickWriteHost();
  syncToDevice();
}

/////////////////////////////////////////////////////////////////////////
// u32 string constructors
NDArray::NDArray(const std::u32string &u32string, sd::DataType dtype, sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::u32string &u32string, sd::DataType dtype, sd::LaunchContext *context) - constructor 20\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dtype)) {
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");
  }

  if (!unicode::isStringValidU32(u32string.data(), u32string.data() + u32string.size())) {
    THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
  }
  // one word that is why used 1
  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

  sd::LongType dataLength = [&] {
    if (dtype == DataType::UTF16) {
      return unicode::offsetUtf32StringInUtf16(u32string.data(), u32string.size());
    }
    if (dtype == DataType::UTF32) {
      return static_cast<sd::LongType>(sizeof(uint32_t) * u32string.size());
    }
    return unicode::offsetUtf32StringInUtf8(u32string.data(), u32string.size());
  }();

  sd::LongType offsets[2] = {0, dataLength};

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto desc = ShapeBuilders::createScalarShapeInfo(dtype);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  memcpy(bufferAsT<int8_t>(), &offsets[0], 2 * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);
  if (dtype == DataType::UTF8) {
    unicode::utf32to8(u32string.data(), data, u32string.size());
  } else if (dtype == DataType::UTF16) {
    unicode::utf32to16(u32string.data(), data, u32string.size());
  } else {
    memcpy(data, u32string.data(), u32string.size() * sizeof(uint32_t));
  }

  tickWriteHost();
  syncToDevice();
}

/////////////////////////////////////////////////////////////////////////
// u8 string constructors
NDArray::NDArray(const std::string &str, sd::DataType dtype, sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::string &str, sd::DataType dtype, sd::LaunchContext *context) - constructor 21\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dtype)) {
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");
  }

  if (!unicode::isStringValidU8(str.data(), str.data() + str.size())) {
    THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
  }

  // one word that is why used 1
  auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

  sd::LongType dataLength = [&] {
    if (dtype == DataType::UTF16) {
      return unicode::offsetUtf8StringInUtf16(str.data(), str.size());
    }
    if (dtype == DataType::UTF32) {
      return unicode::offsetUtf8StringInUtf32(str.data(), str.size());
    }
    return static_cast<sd::LongType>(str.size());
  }();

  sd::LongType offsets[2] = {0, dataLength};

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;
  _isAttached = getContext()->getWorkspace() != nullptr;
  auto desc = ShapeBuilders::createScalarShapeInfo(dtype);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  memcpy(bufferAsT<int8_t>(), &offsets[0], 2 * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);

  if (dtype == DataType::UTF8) {
    memcpy(data, str.data(), str.size());
  } else if (dtype == DataType::UTF16) {
    unicode::utf8to16(str.data(), data, str.size());
  } else {
    unicode::utf8to32(str.data(), data, str.size());
  }

  tickWriteHost();
  syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
// constructors for vector of  strings
NDArray::NDArray(std::vector<LongType> &shape, const std::vector<const char *> &strings, DataType dtype,
                 LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::vector<sd::LongType> &shape, const std::vector<const char *> &string, const sd::DataType dataType, sd::LaunchContext *context) - constructor 22\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dtype)) {
    std::string errorMessage;
    errorMessage += "NDArray::NDArray: invalid DataType, only string dataTypes have to be used";
    errorMessage += "Provided data type: " + DataTypeUtils::asString(dtype);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (static_cast<size_t>(shape::prodLong(shape.data(), shape.size())) != strings.size()) {
    std::string errorMessage;
    errorMessage += "NDArray::NDArray: Number of strings should match length of array. ";
    errorMessage += "Number of strings: " + std::to_string(strings.size()) + ", ";
    errorMessage += "length of array: " + std::to_string(shape::prodLong(shape.data(), shape.size()));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  for (const auto &str : strings) {
    if (!unicode::isStringValidU8(str, str + std::char_traits<char>::length(str))) {
      std::string errorMessage;
      errorMessage += "NDArray::NDArray: invalid character in input string: ";
      errorMessage += str;
      THROW_EXCEPTION(errorMessage.c_str());
    }
  }

  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(strings.size());

  std::vector<sd::LongType> offsets(strings.size() + 1);
  sd::LongType dataLength = 0;
  for (size_t e = 0; e < strings.size(); e++) {
    offsets[e] = dataLength;
    dataLength += [&] {
      if (dtype == DataType::UTF16)
        return unicode::offsetUtf8StringInUtf16(strings[e], std::char_traits<char>::length(strings[e]));
      if (dtype == DataType::UTF32)
        return unicode::offsetUtf8StringInUtf32(strings[e], std::char_traits<char>::length(strings[e]));
      return static_cast<sd::LongType>(std::char_traits<char>::length(strings[e]));
    }();
  }
  offsets[strings.size()] = dataLength;

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;

  auto desc = ShapeBuilders::createShapeInfo(dtype, 'c', shape);
  //note we don't delete this because this can get stored in the cache.
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();

  setAttached(context->getWorkspace() != nullptr);

  memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto cdata = data + offsets[e];
      if (dtype == DataType::UTF16) {
        unicode::utf8to16(strings[e], cdata, std::char_traits<char>::length(strings[e]));
      } else if (dtype == DataType::UTF32) {
        unicode::utf8to32(strings[e], cdata, std::char_traits<char>::length(strings[e]));
      } else {
        memcpy(cdata, strings[e], std::char_traits<char>::length(strings[e]));
      }
    }
  };

  int len = isScalar() ? 1 : lengthOf();
  samediff::Threads::parallel_for(func, 0, len, 1);

  tickWriteHost();
  syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(std::vector<sd::LongType> &shape, const std::vector<std::string> &string, const sd::DataType dataType, sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::vector<sd::LongType> &shape, const std::vector<std::string> &string, const sd::DataType dataType, sd::LaunchContext *context) - constructor 23\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dataType))
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

  if (static_cast<size_t>(shape::prodLong(shape.data(), shape.size())) != string.size())
    THROW_EXCEPTION("NDArray::NDArray: Number of strings should match length of array");

  for (const auto &str : string) {
    if (!unicode::isStringValidU8(str.data(), str.data() + str.size())) {
      THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
    }
  }

  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());
  std::vector<sd::LongType> offsets(string.size() + 1);
  sd::LongType dataLength = 0;
  for (size_t e = 0; e < string.size(); e++) {
    offsets[e] = dataLength;
    dataLength += [&] {
      if (dataType == DataType::UTF16) return unicode::offsetUtf8StringInUtf16(string[e].data(), string[e].size());
      if (dataType == DataType::UTF32) return unicode::offsetUtf8StringInUtf32(string[e].data(), string[e].size());
      return static_cast<sd::LongType>(string[e].size());
    }();
  }

  offsets[string.size()] = dataLength;
  _buffer = new DataBuffer(headerLength + dataLength, dataType, context->getWorkspace(), true);

  _context = context;

  auto buff = ShapeBuilders::createShapeInfo(dataType,'c',shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(buff);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();

  setAttached(context->getWorkspace() != nullptr);

  memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(sd::LongType));
  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);
  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto cdata = data + offsets[e];
      if (dataType == DataType::UTF16) {
        unicode::utf8to16(string[e].data(), cdata, string[e].size());
      } else if (dataType == DataType::UTF32) {
        unicode::utf8to32(string[e].data(), cdata, string[e].size());
      } else {
        memcpy(cdata, string[e].data(), string[e].size());
      }
    }
  };

  int len = isScalar() ? 1 : lengthOf();
  samediff::Threads::parallel_for(func, 0, len, 1);
  tickWriteHost();
  syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(std::vector<LongType> &shape, const std::vector<std::u16string> &string, DataType dtype,
                 LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::vector<sd::LongType> &shape, const std::vector<std::u16string> &string, sd::DataType dtype, sd::LaunchContext *context) - constructor 24\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

  if (static_cast<size_t>(shape::prodLong(shape.data(), shape.size())) != string.size())
    THROW_EXCEPTION("NDArray::NDArray: Number of strings should match length of array");

  for (const auto &str : string) {
    if (!unicode::isStringValidU16(str.data(), str.data() + str.size())) {
      THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
    }
  }

  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

  std::vector<sd::LongType> offsets(string.size() + 1);
  sd::LongType dataLength = 0;
  for (size_t e = 0; e < string.size(); e++) {
    offsets[e] = dataLength;
    dataLength += [&] {
      if (dtype == DataType::UTF16) return static_cast<sd::LongType>(sizeof(uint16_t) * string[e].size());
      if (dtype == DataType::UTF32) return unicode::offsetUtf16StringInUtf32(string[e].data(), string[e].size());
      return unicode::offsetUtf16StringInUtf8(string[e].data(), string[e].size());
    }();
  }
  offsets[string.size()] = dataLength;

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;
  int len = isScalar() ? 1 : lengthOf();
  auto desc = ShapeBuilders::createShapeInfo(dtype, 'c', shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();

  setAttached(context->getWorkspace() != nullptr);

  memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto cdata = data + offsets[e];
      if (dtype == DataType::UTF16) {
        memcpy(cdata, string[e].data(), string[e].size() * sizeof(uint16_t));
      } else if (dtype == DataType::UTF32) {
        unicode::utf16to32(string[e].data(), cdata, string[e].size());
      } else {
        unicode::utf16to8(string[e].data(), cdata, string[e].size());
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, len, 1);

  tickWriteHost();
  syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(std::vector<LongType> &shape, const std::vector<const char16_t *> &strings, DataType dtype,
                 LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::vector<sd::LongType> &shape, const std::vector<const char16_t *> &string, sd::DataType dtype, sd::LaunchContext *context) - constructor 25\n");
    fflush(stdout);

  }

  if (!DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

  if (static_cast<size_t>(shape::prodLong(shape.data(), shape.size())) != strings.size())
    THROW_EXCEPTION("NDArray::NDArray: Number of strings should match length of array");

  for (const auto &str : strings) {
    if (!unicode::isStringValidU16(str, str + std::char_traits<char16_t>::length(str))) {
      THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
    }
  }

  int len = isScalar() ? 1 : lengthOf();
  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(strings.size());

  std::vector<sd::LongType> offsets(strings.size() + 1);
  sd::LongType dataLength = 0;
  for (size_t e = 0; e < strings.size(); e++) {
    offsets[e] = dataLength;
    dataLength += [&] {
      if (dtype == DataType::UTF16)
        return static_cast<sd::LongType>(sizeof(uint16_t) * std::char_traits<char16_t>::length(strings[e]));
      if (dtype == DataType::UTF32)
        return unicode::offsetUtf16StringInUtf32(strings[e], std::char_traits<char16_t>::length(strings[e]));
      return unicode::offsetUtf16StringInUtf8(strings[e], std::char_traits<char16_t>::length(strings[e]));
    }();
  }
  offsets[strings.size()] = dataLength;

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;

  auto desc = ShapeBuilders::createShapeInfo(dtype, 'c', shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();

  setAttached(context->getWorkspace() != nullptr);

  memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto cdata = data + offsets[e];
      if (dtype == DataType::UTF16) {
        memcpy(cdata, strings[e], std::char_traits<char16_t>::length(strings[e]) * sizeof(uint16_t));
      } else if (dtype == DataType::UTF32) {
        unicode::utf16to32(strings[e], cdata, std::char_traits<char16_t>::length(strings[e]));
      } else {
        unicode::utf16to8(strings[e], cdata, std::char_traits<char16_t>::length(strings[e]));
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, len, 1);

  tickWriteHost();
  syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(std::vector<sd::LongType> &shape, const std::vector<std::u32string> &string, sd::DataType dtype,
                 sd::LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::vector<sd::LongType> &shape, const std::vector<std::u32string> &string, sd::DataType dtype, sd::LaunchContext *context) - constructor 26\n");
    fflush(stdout);
  }


  if (!DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::NDArray: invalid DataType, only string dataTypes have to be used");

  if (static_cast<size_t>(shape::prodLong(shape.data(), shape.size())) != string.size())
    THROW_EXCEPTION("NDArray::NDArray: Number of strings should match length of array");

  for (auto str : string) {
    if (!unicode::isStringValidU32(str.data(), str.data() + str.size())) {
      THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
    }
  }
  int len = isScalar() ? 1 : lengthOf();
  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

  std::vector<sd::LongType> offsets(string.size() + 1);

  sd::LongType dataLength = 0;
  for (size_t e = 0; e < string.size(); e++) {
    offsets[e] = dataLength;
    dataLength += [&] {
      if (dtype == DataType::UTF16) return unicode::offsetUtf32StringInUtf16(string[e].data(), string[e].size());
      if (dtype == DataType::UTF32) return static_cast<sd::LongType>(sizeof(uint32_t) * string[e].size());
      return unicode::offsetUtf32StringInUtf16(string[e].data(), string[e].size());
    }();
  }
  offsets[string.size()] = dataLength;

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;
  auto desc = ShapeBuilders::createShapeInfo(dtype, 'c', shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();

  setAttached(context->getWorkspace() != nullptr);

  memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto cdata = data + offsets[e];
      if (dtype == DataType::UTF16) {
        unicode::utf32to16(string[e].data(), cdata, string[e].size());
      } else if (dtype == DataType::UTF32) {
        memcpy(cdata, string[e].data(), string[e].size() * sizeof(uint32_t));
      } else {
        unicode::utf32to8(string[e].data(), cdata, string[e].size());
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, len, 1);

  tickWriteHost();
  syncToDevice();
}
/////////////////////////////////////////////////////////////////////////
NDArray::NDArray(std::vector<LongType> &shape, const std::vector<const char32_t *> &strings, DataType dtype,
                 LaunchContext *context) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::NDArray(const std::vector<sd::LongType> &shape, const std::vector<const char32_t *> &string, sd::DataType dtype, sd::LaunchContext *context) - constructor 27\n");
    fflush(stdout);
  }


  int len = isScalar() ? 1 : lengthOf();
  if (!DataTypeUtils::isS(dtype)) THROW_EXCEPTION("NDArray::NDArray: invalid DataType used");

  if (static_cast<size_t>(shape::prodLong(shape.data(), shape.size())) != strings.size())
    THROW_EXCEPTION("NDArray::NDArray: Number of strings should match length of array");

  for (const auto &str : strings) {
    if (!unicode::isStringValidU32(str, str + std::char_traits<char32_t>::length(str))) {
      THROW_EXCEPTION("NDArray::NDArray: invalid character in input string");
    }
  }

  sd::LongType headerLength = ShapeUtils::stringBufferHeaderRequirements(strings.size());

  std::vector<sd::LongType> offsets(strings.size() + 1);

  sd::LongType dataLength = 0;
  for (size_t e = 0; e < strings.size(); e++) {
    offsets[e] = dataLength;
    dataLength += [&] {
      if (dtype == DataType::UTF16)
        return unicode::offsetUtf32StringInUtf16(strings[e], std::char_traits<char32_t>::length(strings[e]));
      if (dtype == DataType::UTF32)
        return static_cast<sd::LongType>(sizeof(uint32_t) * std::char_traits<char32_t>::length(strings[e]));
      return unicode::offsetUtf32StringInUtf16(strings[e], std::char_traits<char32_t>::length(strings[e]));
    }();
  }
  offsets[strings.size()] = dataLength;

  _buffer = new DataBuffer(headerLength + dataLength, dtype, context->getWorkspace(), true);

  _context = context;

  auto desc = ShapeBuilders::createShapeInfo(dtype, 'c', shape);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();

  setAttached(context->getWorkspace() != nullptr);

  memcpy(bufferAsT<int8_t>(), offsets.data(), offsets.size() * sizeof(sd::LongType));

  auto data = reinterpret_cast<int8_t *>(bufferAsT<int8_t>() + headerLength);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      auto cdata = data + offsets[e];
      if (dtype == DataType::UTF16) {
        unicode::utf32to16(strings[e], cdata, std::char_traits<char32_t>::length(strings[e]));
      } else if (dtype == DataType::UTF32) {
        memcpy(cdata, strings[e], std::char_traits<char32_t>::length(strings[e]) * sizeof(uint32_t));
      } else {
        unicode::utf32to8(strings[e], cdata, std::char_traits<char32_t>::length(strings[e]));
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, len, 1);

  tickWriteHost();
  syncToDevice();
}


//google test print statement
static void sd_printformatted(std::ostream& os, sd::NDArray &arr, sd::LongType depth, sd::LongType limit) {
  // adapted sd_printormatted function
  if(arr.isScalar()) {
    if (arr.isR()) {
      if(arr.dataType() == sd::DataType::DOUBLE)
        os << arr.e<double>(0) << "\n";
      else
        os << arr.e<float>(0) << "\n";
    } else if (arr.isZ())
      os << arr.e<sd::LongType>(0) << "\n";
    else if (arr.isB())
      os << (arr.e<bool>(0) ? "true" : "false") << "\n";
    else if (arr.isS()) {
      os << "\"" << arr.e<std::string>(0) << "\"\n";
    }
    return;
  }

  if (arr.rankOf() == 1) {
    os << "[ ";
    for (sd::LongType i = 0; i < arr.lengthOf(); ++i) {
      if (arr.isR()) {
        if(arr.dataType() == sd::DataType::DOUBLE)
          os << arr.e<double>(i) << ", ";
        else
          os << arr.e<float>(i) << ", ";
      } else if (arr.isZ())
        os << arr.e<sd::LongType>(i) << ", ";
      else if (arr.isB())
        os << (arr.e<bool>(i) ? "true" : "false") << ", ";
      else if (arr.isS()) {
        os << "\"" << arr.e<std::string>(i) << "\", ";
      }
    }
    os << "]\n";
  } else if (arr.rankOf() == 2) {
    sd::LongType rows = arr.rows();
    sd::LongType cols = arr.columns();
    if(limit > 0) {
      cols = sd::math::sd_min<sd::LongType, sd::LongType, sd::LongType>(limit, cols);
    }
    char *padding = new char[depth + 1];
    memset(padding, ' ', depth);
    padding[depth] = 0;
    os << "[";
    for (sd::LongType row = 0; row < rows; row++) {
      if (row && depth > 0) os << padding;
      os << "[";
      for (sd::LongType col = 0; col < cols; col++) {
        if (col > 0) os << ", ";
        if (arr.isR()) {
          if(arr.dataType() == sd::DataType::DOUBLE) {
            os << std::fixed << std::setw(11) << std::setprecision(15)
               << std::setfill('0') << arr.e<double>(row, col);
          } else {
            os << std::fixed << std::setw(11) << std::setprecision(15)
               << std::setfill('0') <<  arr.e<float>(row, col);
          }
        } else if (arr.isZ()) {
          if(arr.dataType() == sd::DataType::INT64)
            os << arr.e<sd::LongType>(row, col);
          else
            os << arr.e<int>(row, col);
        } else if (arr.isB()) {
          os << (arr.e<bool>(row, col) ? "true" : "false");
        } else if (arr.isS()) {
          os << "\"" << arr.e<std::string>(row * cols + col) << "\"";
        }
      }
      if (row < rows - 1)
        os << "]\n";
      else
        os << "]";
    }
    os << "]";
    delete[] padding;
  } else {
    // assuming ShapeUtils and other required objects/methods are defined and available
    sd::LongType restCount = 2;
    os << "[";
    restCount = ShapeUtils::getNumOfSubArrs(arr.shapeInfo(), {0});
    for (sd::LongType arrIndex = 0; arrIndex < restCount; ++arrIndex) {
      NDArray subArr = arr(arrIndex, {0});
      sd_printformatted(os, subArr, depth + 1, limit);
      if (arrIndex < restCount - 1) {
        for (sd::LongType i = 1; i < arr.rankOf(); ++i) os << "\n";
        for (sd::LongType i = 0; i < depth - 2; ++i) os << " ";
      }

    }
    os << "]";
  }
}

std::ostream& operator<<(std::ostream &os,  NDArray& arr) {
  sd_printformatted(os, arr, 0, -1);
  return os;
}


std::ostream& NDArray::operator<<(std::ostream &os) {
  syncToHost();


  sd::LongType rank = rankOf();

  bool rowFlag = (rank < 2) || (rank == 2 && sizeAt(0) == 1);

  if (isEmpty()) {
    os << "Empty\n";
  } else if (rankOf() == 0) {
    if (isZ()) {
      os << e<sd::LongType>(0) << "\n";
    } else if (isR()) {
      os << e<double>(0) << "\n";
    } else if (isB()) {
      os << (e<bool>(0) ? "true" : "false") << "\n";
    } else if (isS()) {
      os << "\"" << e<std::string>(0) << "\"\n";
    }
  } else if (rowFlag && ews() == 1) {
    os << "[ ";
    for (sd::LongType i = 0; i < lengthOf(); ++i) {
      if (isR())
        os << std::fixed << std::setw(11) << std::setprecision(15)
           << std::setfill('0') <<  e<double>(i) << ", ";
      else if (isZ())
        os << e<sd::LongType>(i) << ", ";
      else if (isB())
        os << (e<bool>(i) ? "true" : "false") << ", ";
      else if (isS()) {
        os << "\"" << e<std::string>(i) << "\", ";
      }
    }
    os << "]\n";
  } else {
    if(isEmpty())
      THROW_EXCEPTION("NULL buffer found but shape is not empty.");
    sd_printformatted(os, *this, 1, lengthOf());
  }
  return os;
}



#endif
//end google test print statement
////////////////////////////////////////////////////////////////////////
// assignment operator
NDArray &NDArray::operator=(NDArray &other) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray &NDArray::operator=(NDArray &other) - move assignment operator\n");
    fflush(stdout);
  }

  if (this == &other || (_shapeInfo == other._shapeInfo && _shapeInfo == nullptr)) {
    return *this;
  }

  if (_shapeInfo != nullptr && shape::equalsTypesAndShapesSoft(_shapeInfo, other._shapeInfo)) {
    if (!other.isEmpty()) {
      this->assign(&other);
    }
  } else {
    _context = other._context;
    auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(other.shapeInfo());
    _shapeInfo = constDesc->primary();
    _shapeInfoD = constDesc->special();
    if (!other.isEmpty()) {
      int len = other.isScalar() ? 1 : other.lengthOf();
      _buffer = new DataBuffer(other.getDataBuffer()->dup());
    } else
      _buffer = new DataBuffer();
  }
  return *this;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isC()  {
  // TODO: this method must be implemented once we add support for complex numbers
  return false;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isS()  {
  return (dataType() == DataType::UTF8 || dataType() == DataType::UTF16 || dataType() == DataType::UTF32);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isR()  {
  auto xType = ArrayOptions::dataType(this->_shapeInfo);
  return xType == FLOAT32 || xType == HALF || xType == DOUBLE || xType == FLOAT8 || xType == BFLOAT16;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isZ()  {
  return !isC() && !isR() && !isB() && !isS();
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isB()  { return ArrayOptions::dataType(this->_shapeInfo) == BOOL; }

//////////////////////////////////////////////////////////////////////////
template <typename T>
std::string *  NDArray::toStringValue(T value) {
  std::ostringstream *os = new std::ostringstream();
  // throw the value into the string stream
  *os << std::fixed << std::setw(11) << std::setprecision(15)
      << std::setfill('0') <<  value;
  // convert the string stream into a string and return
  return new std::string(os->str());
}

//////////////////////////////////////////////////////////////////////////
template <>
std::string * NDArray::toStringValue(float16 value) {
  std::ostringstream *os = new std::ostringstream();
  // throw the value into the string stream
  *os << (float)value;
  // convert the string stream into a string and return
  return new std::string(os->str());
}

//////////////////////////////////////////////////////////////////////////
template <>
std::string * NDArray::toStringValue(bfloat16 value) {
  std::ostringstream *os = new std::ostringstream();
  // throw the value into the string stream
  *os <<  std::fixed << std::setw(11) << std::setprecision(15)
      << std::setfill('0') <<   (float)value;
  // convert the string stream into a string and return
  return new std::string(os->str());
}

//////////////////////////////////////////////////////////////////////////
std::string * NDArray::asIndexedString(sd::LongType limit) {
  std::ostringstream *os = new std::ostringstream();
  *os << "[";
  if (limit < 1 || limit > this->lengthOf()) limit = this->lengthOf();
  for (sd::LongType e = 0; e < limit; e++) {
    *os << toStringValue(this->e<double>(e));
    if (e < limit - 1) *os << ", ";
  }
  *os << "]";
  return new std::string(os->str());
}

//////////////////////////////////////////////////////////////////////////
std::string * NDArray::asString(sd::LongType limit) {
  if (this->dataBuffer()->primary() == nullptr) return new std::string("nullptr");
  std::ostringstream *os = new std::ostringstream();
  NDArray &array = *this;
  *os << array;
  return new std::string(os->str());
}

////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> NDArray::getBufferAsVector()  {
  int len = isScalar() ? 1 : lengthOf();
  std::vector<T> vector(len);
  for (sd::LongType e = 0; e < len; e++) vector[e] = this->e<T>(e);
  return vector;
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT std::vector, NDArray::getBufferAsVector(), SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
std::vector<int64_t> NDArray::getShapeAsFlatVector()  {
  std::vector<int64_t> vector(this->rankOf());
  for (int e = 0; e < this->rankOf(); e++) vector[e] = static_cast<int64_t>(this->sizeAt(e));
  return vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> NDArray::getShapeAsVector()  {
  std::vector<sd::LongType> *vector = new std::vector<sd::LongType>();
  for (int e = 0; e < this->rankOf(); e++) {
    vector->push_back(this->sizeAt(e));
  }

  return *vector;
}

std::vector<sd::LongType> NDArray::getStrideAsVector()  {
  std::vector<sd::LongType> *vector = new std::vector<sd::LongType>();
  for (int e = 0; e < this->rankOf(); e++) {
    vector->push_back(this->strideAt(e));
  }

  return *vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<int> NDArray::getShapeAsVectorInt()  {
  std::vector<int> vector(this->rankOf());
  for (int e = 0; e < this->rankOf(); e++) vector[e] = static_cast<int>(this->sizeAt(e));

  return vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<int64_t> NDArray::getShapeInfoAsFlatVector()  {
  int magicNumber = shape::shapeInfoLength(this->rankOf());
  std::vector<int64_t> vector(magicNumber);

  for (int e = 0; e < magicNumber; e++) vector[e] = static_cast<int64_t>(_shapeInfo[e]);

  return vector;
}

////////////////////////////////////////////////////////////////////////
std::vector<sd::LongType> NDArray::getShapeInfoAsVector()  {
  int magicNumber = shape::shapeInfoLength(this->rankOf());
  std::vector<sd::LongType> vector(magicNumber);
  for (int e = 0; e < magicNumber; e++) vector[e] = this->_shapeInfo[e];
  return vector;
}



////////////////////////////////////////////////////////////////////////
std::vector<int8_t> NDArray::asByteVector() {
  if (isS()) {
    // string data type requires special treatment
    syncToHost();
    auto numWords = isScalar() ? 1 : this->lengthOf();
    auto offsetsBuffer = this->bufferAsT<sd::LongType>();
    auto headerLength = ShapeUtils::stringBufferHeaderRequirements(numWords);
    auto dataLength = offsetsBuffer[numWords];
    std::vector<int8_t> result(headerLength + dataLength);

    memcpy(result.data(), buffer(), headerLength + dataLength);

    return result;
  } else {
    int len = isScalar() ? 1 : this->lengthOf();
    // all other types are linear
    std::vector<int8_t> result((unsigned long long)len * sizeOfT());

    if (this->isView()) {
      auto tmp = this->dup(this->ordering(), false);
      syncToHost();
      memcpy(result.data(), tmp.buffer(), (unsigned long long)len * sizeOfT());
    } else {
      syncToHost();
      memcpy(result.data(), buffer(), (unsigned long long)len * sizeOfT());
    }
    return result;
  }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::linspace(const double start) { linspace(start, 1); }

//////////////////////////////////////////////////////////////////////////
void NDArray::linspace(const double start, const double step) {
  if (isS()) THROW_EXCEPTION("NDArray::linspace: you can't use this method on String array!");
  sd::LongType numElements = isScalar() ? 1 : this->lengthOf();
  for (sd::LongType e = 0; e < numElements; e++) this->p<double>(e, start + (step * e));
}

////////////////////////////////////////////////////////////////////////
void NDArray::streamline(char o) {
  char order = o == 'a' ? this->ordering() : o;
  syncToDevice();
  int len = isScalar() ? 1 : this->lengthOf();
  DataBuffer * newBuffer =
      new DataBuffer(len * sizeOfT(), dataType(), getContext()->getWorkspace());
  auto shapeBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(dataType(), order, rankOf(), shapeOf());
  NativeOpExecutioner::execTransformSame(getContext(), transform::Copy, buffer(), shapeInfo(), specialBuffer(),
                                         specialShapeInfo(), newBuffer->primary(), shapeBuffer->primary(),
                                         newBuffer->special(), shapeBuffer->special(), nullptr, nullptr, nullptr);
  _shapeInfo = shapeBuffer->primary();
  _shapeInfoD = shapeBuffer->special();
  _buffer = newBuffer;
  tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
NDArray &NDArray::operator=(NDArray &&other) noexcept {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::operator=(NDArray &&other) - move assignment operator\n");
    fflush(stdout);

  }
  if (this == &other) return *this;

  if(other._shapeInfo == nullptr) {
    THROW_EXCEPTION("NDArray::operator= - trying to move from empty NDArray");
  }

  _buffer = other._buffer;
  _shapeInfo = other._shapeInfo;
  _shapeInfoD = other._shapeInfoD;
  _context = other._context;
  _length = other._length;

  other._buffer = nullptr;
  other._shapeInfo = other._shapeInfoD = nullptr;
  other._length = 0;

  return *this;
}

////////////////////////////////////////////////////////////////////////
template <typename T, typename>
NDArray &NDArray::operator=(T scalar) {
  if(Environment::getInstance().isLogNativeNDArrayCreation()) {
    sd_print("NDArray::operator=(NDArray &&other) - move assignment operator\n");
    fflush(stdout);
  }
  T &localValue = const_cast<T &>(scalar);
  this->assign(localValue,false);
  return *this;
}
#define OPERATOR_TEMPLATE(T) template NDArray &NDArray::operator=(GET_SECOND(T) scalar);
ITERATE_LIST((SD_COMMON_TYPES),OPERATOR_TEMPLATE)










//////////////////////////////////////////////////////////////////////////
void NDArray::copyBuffersContinuouslyFrom(NDArray &other, size_t sizeToCopyInBytes, sd::LongType offsetThis,
                                          sd::LongType offsetOther) {
  if (offsetThis == 0) offsetThis = offset();
  if (offsetOther == 0) offsetOther = other.offset();

  dataBuffer()->copyBufferFrom(*other.getDataBuffer(), sizeToCopyInBytes, offsetThis, offsetOther);
}

bool NDArray::isBroadcastableTo(NDArray &other)  {
  return ShapeUtils::areShapesBroadcastable(this->shapeInfo(), other.shapeInfo());
}

////////////////////////////////////////////////////////////////////
// This method assigns values of given NDArray to this one
void NDArray::assign(NDArray *other, bool allowParallelism) {
  NDArray::validateAssign(this, other);
  const sd::LongType* modifiedOtherShapeInfo = NDArray::modifyShapeForAssign(this, other);
  NDArray::copyDataForAssign(other, this, modifiedOtherShapeInfo, allowParallelism);
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
template <typename T, typename>
void NDArray::assign(T &value, bool allowParallelism) {
  // just fire scalar
  auto temp = new NDArray(NDArrayFactory::create(dataType(), value, this->getContext()));

  prepareUse(std::vector<NDArray *>{this}, std::vector<NDArray *>{temp});
  NativeOpExecutioner::execScalar(getContext(), sd::scalar::CopyPws, buffer(), shapeInfo(), specialBuffer(),
                                  specialShapeInfo(), buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                  temp->buffer(), temp->shapeInfo(), temp->specialBuffer(), temp->specialShapeInfo(),
                                  nullptr, allowParallelism);
  registerUse(std::vector<NDArray *>{this}, std::vector<NDArray *>{temp});
}

#define ASSIGN_SCALAR(T) template SD_LIB_EXPORT void NDArray::assign( GET_SECOND(T) &value, bool allowParallelism);
ITERATE_LIST((SD_COMMON_TYPES),ASSIGN_SCALAR)

//////////////////////////////////////////////////////////////////////////
NDArray *NDArray::detach() {
  if (!isAttached()) return this;

  DataBuffer *  newBuffer = new DataBuffer(lengthOf() * sizeOfT(), dataType());
  auto constantBuff = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo());
  auto recastShapeInfo = const_cast<sd::LongType *>(constantBuff->primary());
  auto result = new NDArray(newBuffer, recastShapeInfo, getContext());
  result->assign(this);

  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::varianceNumber(sd::variance::Ops op, bool biasCorrected) {
  NDArray res(DataTypeUtils::pickFloatingType(dataType()), getContext());

  prepareUse({&res}, {this});
  NativeOpExecutioner::execSummaryStatsScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                              specialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(),
                                              res.specialBuffer(), res.specialShapeInfo(), biasCorrected);
  registerUse({&res}, {this});

  return res;
}

//////////////////////////////////////////////////////////////////////////

NDArray NDArray::prodNumber()  {
  if (isS()) THROW_EXCEPTION("NDArray::prodNumber: you can't use this method on String array!");

  NDArray res(dataType(), getContext());

  prepareUse({&res}, {this});
  NativeOpExecutioner::execReduceSameScalar(getContext(), sd::reduce::SameOps::Prod, buffer(), shapeInfo(),
                                            specialBuffer(), specialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(),
                                            res.specialBuffer(), res.specialShapeInfo());
  registerUse({&res}, {this});

  return res;
}

// This method returns sum of all elements of this NDArray
NDArray NDArray::sumNumber()  {
  if (isS()) THROW_EXCEPTION("NDArray::sumNumber: you can't use this method on String array!");
  NDArray res(dataType(), getContext());

  prepareUse({&res}, {this});
  NativeOpExecutioner::execReduceSameScalar(getContext(), sd::reduce::SameOps::Sum, buffer(), shapeInfo(),
                                            specialBuffer(), specialShapeInfo(), nullptr, res.buffer(), res.shapeInfo(),
                                            res.specialBuffer(), res.specialShapeInfo());
  registerUse({&res}, {this});

  return res;
}

//////////////////////////////////////////////////////////////////////////
// This method returns mean number of this NDArray
NDArray NDArray::meanNumber()  {
  if (isS()) THROW_EXCEPTION("NDArray::meanNumber: you can't use this method on String array!");
  NDArray res(DataTypeUtils::pickFloatingType(dataType()), getContext());

  prepareUse({&res}, {this});
  NativeOpExecutioner::execReduceFloatScalar(getContext(), sd::reduce::FloatOps::Mean, buffer(), shapeInfo(),
                                             specialBuffer(), specialShapeInfo(), nullptr, res.buffer(),
                                             res.shapeInfo(), res.specialBuffer(), res.specialShapeInfo());
  registerUse({&res}, {this});
  return res;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::hasNaNs() {
  if (isS()) THROW_EXCEPTION("NDArray::hasNaNs: you can't use this method on String array!");
  void *extraParams = nullptr;
  return this->reduceNumber(sd::reduce::IsNan, extraParams).e<sd::LongType>(0) > 0;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::hasInfs() {
  if (isS()) THROW_EXCEPTION("NDArray::hasInfs: you can't use this method on String array!");
  void *extraParams = nullptr;
  return this->reduceNumber(sd::reduce::IsInf, extraParams).e<sd::LongType>(0) > 0;
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::isFinite() {
  if (isS()) THROW_EXCEPTION("NDArray::isFinite: you can't use this method on String array!");
  void *extraParams = nullptr;
  return this->reduceNumber(sd::reduce::IsInfOrNan, extraParams).e<sd::LongType>(0) == 0;
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
template <typename T, typename Y, typename>
void NDArray::templatedSet(void *buffer, LongType offset, void *value) {
  NDArray::preparePrimaryUse({this}, {this});
  int len = isScalar() ? 1 : lengthOf();

  // Check if offset is valid
  if (offset < 0 || offset >= len) {
    std::string errorMessage = "NDArray::templatedSet: offset out of bounds "
                               + std::to_string(offset) + " vs " + std::to_string(len);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  auto t = reinterpret_cast<T *>(buffer);
  const auto y = *(reinterpret_cast<const Y *>(value));
  if(t == nullptr) {
    THROW_EXCEPTION("NDArray::templatedSet: first buffer is nullptr");
  }

  t[offset] = y;
  tickWriteHost();
  NDArray::registerPrimaryUse({this}, {this});
}
BUILD_DOUBLE_TEMPLATE(template SD_LIB_EXPORT void NDArray::templatedSet,
                      (void *buffer,  sd::LongType offset,  void *value), SD_NUMERIC_TYPES, SD_NUMERIC_TYPES);


template <typename T, typename Y, typename>
void NDArray::templatedSetString(void *buffer, LongType offset, void *value) {
  if (!this->isS()) {
    THROW_EXCEPTION("Attempting to set string value to non-string array");
  }

  NDArray::preparePrimaryUse({this}, {this});

  const auto& str = *(reinterpret_cast<const Y*>(value));

  // Calculate header length for the offsets array
  auto numWords = isScalar() ? 1 : this->lengthOf();
  auto headerLength = ShapeUtils::stringBufferHeaderRequirements(numWords);
  auto offsetsBuffer = reinterpret_cast<LongType*>(bufferAsT<int8_t>());

  // Calculate the length of the new string in target encoding
  LongType dataLength;
  if (dataType() == DataType::UTF8) {
    if constexpr(std::is_same_v<Y, std::string>) {
      dataLength = str.size();
    } else if constexpr(std::is_same_v<Y, std::u16string>) {
      dataLength = unicode::offsetUtf16StringInUtf8(str.data(), str.size());
    } else { // u32string
      dataLength = unicode::offsetUtf32StringInUtf8(str.data(), str.size());
    }
  } else if (dataType() == DataType::UTF16) {
    if constexpr(std::is_same_v<Y, std::string>) {
      dataLength = unicode::offsetUtf8StringInUtf16(str.data(), str.size());
    } else if constexpr(std::is_same_v<Y, std::u16string>) {
      dataLength = str.size() * sizeof(char16_t);
    } else { // u32string
      dataLength = unicode::offsetUtf32StringInUtf16(str.data(), str.size());
    }
  } else { // UTF32
    if constexpr(std::is_same_v<Y, std::string>) {
      dataLength = unicode::offsetUtf8StringInUtf32(str.data(), str.size());
    } else if constexpr(std::is_same_v<Y, std::u16string>) {
      dataLength = unicode::offsetUtf16StringInUtf32(str.data(), str.size());
    } else { // u32string
      dataLength = str.size() * sizeof(char32_t);
    }
  }

  // Update offsets
  LongType prevEnd = offset == 0 ? 0 : offsetsBuffer[offset];
  for (LongType i = offset + 1; i < numWords + 1; i++) {
    auto temp = offsetsBuffer[i];
    offsetsBuffer[i] = prevEnd + dataLength;
    prevEnd = temp;
  }

  // Get pointer to data section after offsets
  auto data = reinterpret_cast<int8_t*>(bufferAsT<int8_t>() + headerLength + offsetsBuffer[offset]);

  // Convert and write the string data
  if (dataType() == DataType::UTF8) {
    if constexpr(std::is_same_v<Y, std::string>) {
      memcpy(data, str.data(), str.size());
    } else if constexpr(std::is_same_v<Y, std::u16string>) {
      unicode::utf16to8(str.data(), data, str.size());
    } else { // u32string
      unicode::utf32to8(str.data(), data, str.size());
    }
  } else if (dataType() == DataType::UTF16) {
    if constexpr(std::is_same_v<Y, std::string>) {
      unicode::utf8to16(str.data(), data, str.size());
    } else if constexpr(std::is_same_v<Y, std::u16string>) {
      memcpy(data, str.data(), str.size() * sizeof(char16_t));
    } else { // u32string
      unicode::utf32to16(str.data(), data, str.size());
    }
  } else { // UTF32
    if constexpr(std::is_same_v<Y, std::string>) {
      unicode::utf8to32(str.data(), data, str.size());
    } else if constexpr(std::is_same_v<Y, std::u16string>) {
      unicode::utf16to32(str.data(), data, str.size());
    } else { // u32string
      memcpy(data, str.data(), str.size() * sizeof(char32_t));
    }
  }

  tickWriteHost();
  NDArray::registerPrimaryUse({this}, {this});
}
template void NDArray::templatedSetString<std::string, std::string>(void*, LongType, void*);
template void NDArray::templatedSetString<std::u16string, std::u16string>(void*, LongType, void*);
template void NDArray::templatedSetString<std::u32string, std::u32string>(void*, LongType, void*);
//////////////////////////////////////////////////////////////////////////
void NDArray::setContext(sd::LaunchContext *context) {
  _context = context;
  if (getContext() == nullptr) _context = sd::LaunchContext ::defaultContext();  // empty context for default cases
}

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////

void *NDArray::specialBufferWithOffset(sd::LongType offset) {
  return const_cast<int8_t *>(specialBuffer() != nullptr ? static_cast<const int8_t *>(specialBuffer()) + (offset * sizeOfT())
                                                         : nullptr);
}

void *NDArray::bufferWithOffset(sd::LongType offset) {
  return const_cast<int8_t *>(buffer() != nullptr ? static_cast<const int8_t *>(buffer()) + (offset * sizeOfT())
                                                  : nullptr);
}

//////////////////////////////////////////////////////////////////////////
// eventually method reduces array by excluding its shapes along axes present in dimensions vector
NDArray NDArray::reduceAlongDimension(sd::reduce::FloatOps op, const std::vector<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  auto newShape = ShapeUtils::evalReduceShapeInfo(
      'c', copy, *this, isR() ? dataType() : Environment::getInstance().defaultFloatDataType(), keepDims, false,
      getContext()->getWorkspace());

  NDArray result(newShape, true, getContext());

  this->reduceAlongDimension(op, &result, copy, keepDims, false);

  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::SameOps op, const std::vector<LongType> *dimensions,
                                      const bool keepDims)  {

  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
  auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, keepDims, false, getContext()->getWorkspace());
  NDArray result(newShape, true, getContext());
  reduceAlongDimension(op, &result, copy, keepDims, false);
  delete copy;
  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::BoolOps op, const std::vector<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  auto newShape =
      ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::BOOL, keepDims, false, getContext()->getWorkspace());

  NDArray result(newShape, true, getContext());
  reduceAlongDimension(op, &result, const_cast<std::vector<sd::LongType> *>(copy), keepDims, false);
  delete copy;
  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::LongOps op, const std::vector<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  auto newShape =
      ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataType::INT64, keepDims, false, getContext()->getWorkspace());

  NDArray result(newShape, true, getContext());

  reduceAlongDimension(op, &result, copy, keepDims, false);

  return result;
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
NDArray NDArray::reduceAlongDimension(sd::reduce::FloatOps op, const std::initializer_list<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *vec = new std::vector<sd::LongType>(*dimensions);
  auto ret = reduceAlongDimension(op, vec, keepDims);
  return ret;

}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::SameOps op, const std::initializer_list<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *vec = new std::vector<sd::LongType>(*dimensions);
  auto ret = reduceAlongDimension(op, vec, keepDims);
  return ret;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::BoolOps op, const std::initializer_list<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *vec = new std::vector<sd::LongType>(*dimensions);
  auto ret = reduceAlongDimension(op, vec, keepDims);
  return ret;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceAlongDimension(sd::reduce::LongOps op, const std::initializer_list<LongType> *dimensions,
                                      const bool keepDims)  {
  std::vector<sd::LongType> *vec = new std::vector<sd::LongType>(*dimensions);
  auto ret = reduceAlongDimension(op, vec, keepDims);
  return ret;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::FloatOps op, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber FloatOps: you can't use this method on String array!");

  auto shape = ConstantShapeHelper::getInstance().scalarShapeInfo(DataTypeUtils::pickFloatingType(dataType()));
  NDArray result(shape, true, this->getContext());

  prepareUse({&result}, {this});
  NativeOpExecutioner::execReduceFloatScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                             specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(),
                                             result.specialBuffer(), result.specialShapeInfo());
  registerUse({&result}, {this});

  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::SameOps op, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber SameOps: you can't use this method on String array!");
  NDArray result(dataType(), getContext());

  prepareUse({&result}, {this});
  NativeOpExecutioner::execReduceSameScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                            specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(),
                                            result.specialBuffer(), result.specialShapeInfo());
  registerUse({&result}, {this});

  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::BoolOps op, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber BoolOps: you can't use this method on String array!");

  auto shape = ConstantShapeHelper::getInstance().scalarShapeInfo(DataType::BOOL);
  NDArray result(shape, true, this->getContext());

  prepareUse({&result}, {this});
  NativeOpExecutioner::execReduceBoolScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                            specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(),
                                            result.specialBuffer(), result.specialShapeInfo());
  registerUse({&result}, {this});

  return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::reduceNumber(sd::reduce::LongOps op, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber LongOps: you can't use this method on String array!");

  auto shape = ConstantShapeHelper::getInstance().scalarShapeInfo(DataType::INT64);
  NDArray result(shape, true, this->getContext());

  prepareUse({&result}, {this});
  NativeOpExecutioner::execReduceLongScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                            specialShapeInfo(), extraParams, result.buffer(), result.shapeInfo(),
                                            result.specialBuffer(), result.specialShapeInfo());
  registerUse({&result}, {this});

  return result;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::FloatOps op, NDArray *target, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber FloatOps: you can't use this method on String array!");
  if (target->lengthOf() > 1 || target->dataType() != DataTypeUtils::pickFloatingType(dataType()))
    THROW_EXCEPTION("NDArray::reduceNumber FloatOps: target array should be scalar and have corresponding float type!");

  prepareUse({target}, {this});
  NativeOpExecutioner::execReduceFloatScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                             specialShapeInfo(), extraParams, target->buffer(), target->shapeInfo(),
                                             target->specialBuffer(), target->specialShapeInfo());
  registerUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::SameOps op, NDArray *target, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber SameOps: you can't use this method on String array!");
  if (target->lengthOf() > 1 || target->dataType() != dataType())
    THROW_EXCEPTION("NDArray::reduceNumber SameOps: target array should be scalar and have same type as this array!");

  prepareUse({target}, {this});
  NativeOpExecutioner::execReduceSameScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                            specialShapeInfo(), extraParams, target->buffer(), target->shapeInfo(),
                                            target->specialBuffer(), target->specialShapeInfo());
  registerUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::BoolOps op, NDArray *target, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber BoolOps: you can't use this method on String array!");
  if (target->lengthOf() > 1 || target->dataType() != DataType::BOOL)
    THROW_EXCEPTION("NDArray::reduceNumber BoolOps: target array should be scalar and have bool type!");

  prepareUse({target}, {this});
  NativeOpExecutioner::execReduceBoolScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                            specialShapeInfo(), extraParams, target->buffer(), target->shapeInfo(),
                                            target->specialBuffer(), target->specialShapeInfo());
  registerUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::reduceNumber(sd::reduce::LongOps op, NDArray *target, void *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceNumber LongOps: you can't use this method on String array!");
  if (target->lengthOf() > 1 || target->dataType() != DataType::INT64)
    THROW_EXCEPTION("NDArray::reduceNumber LongOps: target array should be scalar and have long type!");

  prepareUse({target}, {this});
  NativeOpExecutioner::execReduceLongScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                            specialShapeInfo(), extraParams, target->buffer(), target->shapeInfo(),
                                            target->specialBuffer(), target->specialShapeInfo());
  registerUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::indexReduceNumber(sd::indexreduce::Ops op, ExtraArguments *extraParams) {
  if (isS()) THROW_EXCEPTION("NDArray::indexReduceNumber: you can't use this method on String array!");

  auto res = NDArrayFactory::create<sd::LongType>(0);

  prepareUse({&res}, {this});
  NativeOpExecutioner::execIndexReduceScalar(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
      extraParams == nullptr ? nullptr : extraParams->argumentsAsT(this->dataType()), res.buffer(), res.shapeInfo(),
      res.specialBuffer(), res.specialShapeInfo());
  registerUse({&res}, {this});

  return res;
}

//////////////////////////////////////////////////////////////////////////
sd::LongType NDArray::tensorsAlongDimension(std::initializer_list<LongType> dimensions)  {
  std::vector<sd::LongType> *vec = new std::vector<sd::LongType>(dimensions);
  auto ret = tensorsAlongDimension(vec);
  return ret;
}

//////////////////////////////////////////////////////////////////////////
sd::LongType NDArray::tensorsAlongDimension(const std::vector<LongType> *dimensions)  {
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
  shape::checkDimensions(rankOf(), copy);

  sd::LongType tadLength =
      shape::tadLength(this->_shapeInfo, copy->data(), (sd::LongType)copy->size());
  int len = isScalar() ? 1 : this->lengthOf();
  sd::LongType numTads = this->lengthOf() / tadLength;

  return numTads;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::printShapeInfo(const char *msg)  {
  int rank = shape::rank(_shapeInfo);
  int lim = shape::shapeInfoLength(rank);

  if (msg != nullptr) {
    sd_printf("shapeInfo %s: [", msg);
  } else {
    sd_printf("shapeInfo: [%s", "");
  }
  sd_printf("%i,  ", rank);
  for (int i = 1; i < shape::shapeInfoLength(rank) - 3; i++) {
    if (i == rank + 1) sd_print("  ");
    sd_printf("%lld,", _shapeInfo[i]);
  }
  sd_printf("  %lld,", shape::type(_shapeInfo));
  sd_printf("%lld,", shape::elementWiseStride(_shapeInfo));
  sd_printf("%lld]\n", (sd::LongType)shape::order(_shapeInfo));

}

//////////////////////////////////////////////////////////////////////////
void NDArray::printIndexedBuffer(const char *msg, sd::LongType limit)  {
  syncToHost();

  sd::LongType rank = this->rankOf();

  if (msg) sd_printf("\n%s:\n  ", msg);
  //uses the << operator instead which is used in gtest as well
  NDArray &reffed = *this;
  std::cout << reffed;

  if (msg) sd_printf("\n%s end: ", msg);

}



//////////////////////////////////////////////////////////////////////////
// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected
NDArray NDArray::transpose()  & {
  NDArray newArr(getDataBuffer(), shapeInfo(), getContext(), offset());
  newArr.transposei();
  return newArr;
}

//////////////////////////////////////////////////////////////////////////
// method makes copy of this array and applies to the copy transpose operation, this array remains unaffected
NDArray NDArray::transpose() && {
  this->transposei();
  return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
// method performs transpose operation based on this array and store result in target, this array remains unaffected
void NDArray::transpose(NDArray &target)  {
  auto correctShape = ShapeUtils::evalTransposeShapeInfo(*this, getContext()->getWorkspace());
  if (!shape::equalsStrict(correctShape, target.shapeInfo()))
    THROW_EXCEPTION("NDArray::transpose method: the shapeInfo of target array is wrong !");

  target._buffer = _buffer;
}

////////////////////////////////////////////////////////////////////////
// This method applies in-place transpose to this array, so this array becomes transposed
void NDArray::transposei() {
  std::vector<sd::LongType> perm;
  for (int e = this->rankOf() - 1; e >= 0; e--) perm.emplace_back(e);

  this->permutei(perm, false, false);
}

////////////////////////////////////////////////////////////////////////
bool NDArray::equalsTo(NDArray &other, double eps)  { return equalsTo(&other, eps); }

//////////////////////////////////////////////////////////////////////////
void NDArray::setAttached(bool reallyAttached) { _isAttached = reallyAttached; };

//////////////////////////////////////////////////////////////////////////
// calculate strides
void NDArray::updateStrides(const char order) { THROW_EXCEPTION("Forbidden method"); }

//////////////////////////////////////////////////////////////////////////

NDArray * NDArray::newShapeNoCopy(const std::vector<sd::LongType> &newShape, const char order) {
  NDArray arr = *this;
  size_t oldnd;
  std::vector<sd::LongType> *olddims = new std::vector<sd::LongType>(arr.getShapeAsVector());
  std::vector<sd::LongType> *oldstrides = new std::vector<sd::LongType>(arr.getStrideAsVector());
  sd::LongType np, op, last_stride;
  size_t oi, oj, ok, ni, nj, nk;
  std::vector<sd::LongType> newStrides(newShape.size());
  oldnd = 0;
  bool isFOrder = order == 'f';

  // Remove axes with dimension 1 from the old array
  for (oi = 0; oi < static_cast<size_t>(arr.rankOf()); oi++) {
    if (arr.sizeAt(oi) != 1) {
      olddims->at(oldnd) = arr.sizeAt(oi);
      oldstrides->at(oldnd) = arr.strideAt(oi);
      oldnd++;
    }
  }

  np = 1;
  for (ni = 0; ni < newShape.size(); ni++) {
    np *= newShape[ni];
  }
  op = 1;
  for (oi = 0; oi < oldnd; oi++) {
    op *= olddims->at(oi);
  }
  if (np != op) {
    // different total sizes; no hope
    delete olddims;
    delete oldstrides;
    return nullptr;
  }

  if (np == 0) {
    // the current code does not handle 0-sized arrays, so give up
    delete olddims;
    delete oldstrides;
    return nullptr;
  }

  // oi to oj and ni to nj give the axis ranges currently worked with
  oi = 0;
  oj = 1;
  ni = 0;
  nj = 1;
  while (ni < newShape.size() && oi < oldnd) {
    np = newShape[ni];
    op = olddims->at(oi);

    while (np != op) {
      if (np < op) {
        // Misses trailing 1s, these are handled later
        np *= newShape[nj++];
      } else {
        op *= olddims->at(oj++);
      }
    }

    // Check whether the original axes can be combined
    for (ok = oi; ok < oj - 1; ok++) {
      if (isFOrder) {
        if (oldstrides->at(ok + 1) != olddims->at(ok) * oldstrides->at(ok)) {
          // not contiguous enough
          delete olddims;
          delete oldstrides;
          return nullptr;
        }
      } else {
        // C order
        if (oldstrides->at(ok) != olddims->at(ok + 1) * oldstrides->at(ok + 1)) {
          // not contiguous enough
          delete olddims;
          delete oldstrides;
          return nullptr;
        }
      }
    }

    // Calculate new strides for all axes currently worked with
    if (isFOrder) {
      newStrides[ni] = oldstrides->at(oi);
      for (nk = ni + 1; nk < nj; nk++) {
        newStrides[nk] = newStrides[nk - 1] * newShape[nk - 1];
      }
    } else {
      // C order
      newStrides[nj - 1] = oldstrides->at(oj - 1);
      for (nk = nj - 1; nk > ni; nk--) {
        newStrides[nk - 1] = newStrides[nk] * newShape[nk];
      }
    }
    ni = nj++;
    oi = oj++;
  }

  // Set strides corresponding to trailing 1s of the new shape
  if (ni >= 1) {
    last_stride = newStrides[ni - 1];
  } else {
    last_stride = 1;
  }
  if (isFOrder && ni >= 1) {
    last_stride *= newShape[ni - 1];
  }
  for (nk = ni; nk < newShape.size(); nk++) {
    newStrides[nk] = last_stride;
  }

  // Create a new array with the same buffer but new shape and strides
  auto newShapeInfo = ShapeBuilders::createShapeInfo(arr.dataType(), order, newShape, getContext()->getWorkspace());
  shape::setStride(newShapeInfo, newStrides.data());

  delete olddims;
  delete oldstrides;

  auto retShapeInfo = ConstantShapeHelper::getInstance().createFromExisting(newShapeInfo);
  auto ret =  new NDArray(this->getDataBuffer(), retShapeInfo, getContext(), this->offset());
  return ret;
}


// set new order and shape in case of suitable array length
bool NDArray::reshapei(const char order, const std::initializer_list<sd::LongType> &shape) {
  std::vector<sd::LongType> vShape(shape);
  return reshapei(order, vShape);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::reshapei(const std::initializer_list<sd::LongType> &shape) {
  return reshapei(ordering(), shape);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::reshapei(const std::vector<sd::LongType> &shape) {
  return reshapei(ordering(), shape);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::enforce(const std::initializer_list<sd::LongType> &dimensions, char order) {
  if(order != 'c' && order != 'f') {
    std::string errorMessage;
    errorMessage += "NDArray::reshape: unknown order, must be c or f received: ";
    errorMessage += order;
    THROW_EXCEPTION(errorMessage.c_str());
  }

  std::vector<sd::LongType> dims(dimensions);
  enforce(dims, order);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::enforce(std::vector<sd::LongType> &dimensions, char o) {
  sd::LongType prod = 1;
  for (size_t e = 0; e < dimensions.size(); e++) prod *= dimensions[e];

  if (prod != this->lengthOf()) {
    std::string current = ShapeUtils::shapeAsString(this);
    std::string enforced = ShapeUtils::shapeAsString(dimensions);
    sd_printf("Can't enforce new shape, lengths mismatch. Original shape: %s; Requested shape: %s\n", current.c_str(),
              enforced.c_str());
    THROW_EXCEPTION("Incompatible shape");
  }

  char order = o == 'a' ? this->ordering() : o;
  auto desc = ShapeBuilders::createShapeInfo(dataType(), order, dimensions);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
}

//////////////////////////////////////////////////////////////////////////
sd::LongType NDArray::argMax(std::initializer_list<LongType> dimensions) {
  if (isS()) THROW_EXCEPTION("NDArray::argMax: you can't use this method on String array!");

  if (dimensions.size() == 0) {
    sd::LongType max = 0;
    auto mv = -DataTypeUtils::max<float>();
    for (sd::LongType e = 0; e < this->lengthOf(); e++) {
      auto val = this->e<float>(e);
      if (mv < val) {
        mv = val;
        max = e;
      }
    }
    return max;
  } else {
    THROW_EXCEPTION("Not implemented yet");
  }
}

// Validates whether the assignment operation can be performed
void NDArray::validateAssign(NDArray *thisArray,  NDArray *other) {
  if (&thisArray == &other) {
    sd_print("NDArray::assign: this == &other\n");
    return;
  }

  if (other->isEmpty()) {
    if (!thisArray->isEmpty()) {
      THROW_EXCEPTION("Cannot assign empty array to non-empty array");
    }
    return;
  }

  if (thisArray->isEmpty()) {
    THROW_EXCEPTION("Cannot assign to an empty array");
  }

  if (!other->isScalar() &&
      (other->lengthOf() != thisArray->lengthOf() &&
       !ShapeUtils::areShapesBroadcastable(other->shapeInfo(), thisArray->shapeInfo()))) {
    auto shapeThis = ShapeUtils::shapeAsString(thisArray);
    auto shapeThat = ShapeUtils::shapeAsString(other);
    std::string errorMessage = "Can't assign array: this shape " + shapeThis + "; other shape: " + shapeThat + "\n";
    THROW_EXCEPTION(errorMessage.c_str());
  }
}

// Performs the data copy for assignment
void NDArray::copyDataForAssign(NDArray *thisArray,  NDArray *other, const sd::LongType* otherShapeInfo, bool allowParallelism) {
  if (other->isScalar()) {
    thisArray->prepareSpecialUse({thisArray}, {other});
    NativeOpExecutioner::execScalar(thisArray->getContext(), scalar::CopyPws, thisArray->buffer(), thisArray->shapeInfo(),
                                    thisArray->specialBuffer(), thisArray->specialShapeInfo(), thisArray->buffer(),
                                    thisArray->shapeInfo(), thisArray->specialBuffer(), thisArray->specialShapeInfo(),
                                    other->buffer(), other->shapeInfo(), other->specialBuffer(),
                                    other->specialShapeInfo(), nullptr, allowParallelism);
    thisArray->registerSpecialUse({thisArray}, {other}); // Registering through the instance

  } else {
    thisArray->prepareSpecialUse({thisArray}, {other});


    NativeOpExecutioner::execTransformAny(thisArray->getContext(),
                                          transform::Assign,
                                          thisArray->buffer(),
                                          thisArray->shapeInfo(),
                                          thisArray->specialBuffer(),
                                          thisArray->specialShapeInfo(),
                                          other->buffer(),
                                          otherShapeInfo,
                                          other->specialBuffer(),
                                          other->specialShapeInfo(),
                                          nullptr
        , allowParallelism);
    thisArray->registerSpecialUse({thisArray}, {other}); // Registering through the instance
  }
}


//////////////////////////////////////////////////////////////////////////

// Adjusts shape for assignment without modifying the original array
const LongType *NDArray::modifyShapeForAssign( NDArray *thisArray,  NDArray *other) {
  if (!shape::shapeEquals(thisArray->shapeInfo(), other->shapeInfo())) {
    std::vector<sd::LongType> *thisShape = new std::vector<sd::LongType>(thisArray->getShapeAsVector());
    auto ret = reshapeShapeInfo(other, other->ordering(), *thisShape);
    delete thisShape;
    return ret;
  }
  return other->shapeInfo();
}

// Reshape logic modularized into a static helper function
// Reshape logic modularized into a static helper function
LongType *NDArray::reshapeShapeInfo(NDArray *array, char order, const std::vector<sd::LongType>& newShape) {
  //nothing to do for scalar shape info
  if(array->isScalar()) {
    //no need for strides with scalar just set shape
    sd::LongType *ret = new sd::LongType[shape::shapeInfoLength(newShape.size())];
    // Initialize memory to prevent garbage values
    memset(ret, 0, sizeof(sd::LongType) * shape::shapeInfoLength(newShape.size()));

    ret[0] = newShape.size();
    for (size_t i = 0; i < newShape.size(); i++) {
      ret[i + 1] = newShape[i];
    }


    shape::setOrder(ret, 'c');
    ArrayOptions::setExtra(ret, ArrayOptions::defaultFlag());
    ArrayOptions::setDataType(ret, array->dataType());

    auto ret2 = ConstantShapeHelper::getInstance().bufferForShapeInfo(ret);
    delete[] ret;
    return ret2->primary();
  }

  if (order != 'c' && order != 'f') {
    std::string errorMessage = "reshapeShapeInfo: unknown order, must be 'c' or 'f', received: ";
    errorMessage += order;
    THROW_EXCEPTION(errorMessage.c_str());
  }


  std::vector<sd::LongType> shape_vector;
  int numberNegativesOnes = 0;
  for (size_t i = 0; i < newShape.size(); i++) {
    if (newShape[i] < 0) {
      if (numberNegativesOnes >= 1) {
        THROW_EXCEPTION("reshapeShapeInfo: only one dimension can be negative at once!");
      }
      numberNegativesOnes++;
      sd::LongType shapeLength = 1;
      for (size_t j = 0; j < newShape.size(); j++) {
        if (i != j) shapeLength *= newShape[j];
      }
      if (shapeLength == 0) {
        THROW_EXCEPTION("reshapeShapeInfo: division by zero in negative dimension calculation");
      }
      sd::LongType realShape = array->lengthOf() / shapeLength;
      shape_vector.push_back(realShape);
    } else {
      shape_vector.push_back(newShape[i]);
    }
  }


  sd::LongType arrLength = 1;
  for (const auto& dim : shape_vector) {
    arrLength *= dim;
  }

  if (arrLength != array->lengthOf() && !array->isScalar()) {
    THROW_EXCEPTION("reshapeShapeInfo: bad length of new shape!");
  }


  size_t shapeInfoLength = shape::shapeInfoLength(shape_vector.size());


  sd::LongType* shapeInfoNew = new sd::LongType[shapeInfoLength];

  // Initialize to prevent garbage values
  memset(shapeInfoNew, 0, sizeof(sd::LongType) * shapeInfoLength);

  shapeInfoNew[0] = static_cast<sd::LongType>(shape_vector.size());

  bool canReshape = ops::helpers::reshapeNoAlloc(array->shapeInfo(), shape_vector, order, shapeInfoNew);

  if (!canReshape) {
    shape::calcStrides(shapeInfoNew, order);
    shape::setOrder(shapeInfoNew, order);
  }

  // Additional validation before passing to buffer
  if (shapeInfoNew == nullptr) {
    THROW_EXCEPTION("reshapeShapeInfo: shapeInfoNew is null after initialization!");
  }


  auto ret = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfoNew);

  delete[] shapeInfoNew;


  return ret->primary();
}

NDArray& NDArray::reshape(const char order, std::vector<sd::LongType>& shape, bool copyToNewBuff) & {
  sd::LongType* newShapeInfo = NDArray::reshapeShapeInfo(this, order, shape);

  if (copyToNewBuff) {
    NDArray* ret = new NDArray(newShapeInfo, true, getContext(), false);
    this->applyTransform(transform::Assign, ret, nullptr);
    return *ret;
  } else {
    NDArray* ret = new NDArray(getDataBuffer(), const_cast<sd::LongType*>(newShapeInfo), getContext(), offset());
    return *ret;
  }
}

//////////////////////////////////////////////////////////////////////////
NDArray &NDArray::reshape(const char order,  std::vector<sd::LongType> &shape, const bool copyToNewBuff) && {
  if(order != 'c' && order != 'f') {
    std::string errorMessage;
    errorMessage += "NDArray::reshape: unknown order, must be c or f received: ";
    errorMessage += order;
    THROW_EXCEPTION(errorMessage.c_str());
  }
  this->reshapei(order, shape);
  return (NDArray &)std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
// change an array by repeating it the number of times given by reps.
void NDArray::tilei(const std::vector<sd::LongType> &reps) { *this = this->tile(reps); }

//////////////////////////////////////////////////////////////////////////
sd::LongType NDArray::sizeAt(const int dim)  {
  if (this->rankOf() == 0 && (dim == 0 || dim == -1)) return 0;
  if (dim >= this->rankOf() || dim < -this->rankOf()) {
    std::string errorMessage;
    errorMessage += "NDArray::sizeAt: bad size index requested: ";
    errorMessage += std::to_string(dim);
    errorMessage += " for array with rank: ";
    errorMessage += std::to_string(this->rankOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (_shapeInfo == nullptr || _shapeInfo[0] < 0 || _shapeInfo[0] > SD_MAX_RANK) {
    THROW_EXCEPTION(
        "Bad shapeInfo pointer or shapeInfo[0] value is corrupt! The _shapeInfo might have been deallocated.");
  }

  if (dim >= 0) {
    return shape::shapeOf(_shapeInfo)[dim];
  } else
    return shape::shapeOf(_shapeInfo)[this->rankOf() + dim];
}

//////////////////////////////////////////////////////////////////////////
sd::LongType NDArray::strideAt(const int dim)  {
  if (dim >= this->rankOf() || dim < -this->rankOf()) THROW_EXCEPTION("NDArray::strideAt: Bad size index requested");

  if (dim >= 0)
    return shape::stride(_shapeInfo)[dim];
  else
    return shape::stride(_shapeInfo)[this->rankOf() + dim];
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(const std::initializer_list<LongType> &dimensions, const bool copyToNewBuff,
                       const bool resetStrides) {
  std::vector<sd::LongType> vec(dimensions);
  return permutei(vec, copyToNewBuff, resetStrides);
}

//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(std::vector<LongType> &dimensions, const bool copyToNewBuff, const bool resetStrides) {
  return permutei(dimensions.data(), rankOf());
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
NDArray &NDArray::permute(LongType *dimensions, const int rank, const bool copyToNewBuff, const bool resetStrides) & {
  if(copyToNewBuff) {
    auto shapeInfoPermuted = ShapeUtils::evalPermShapeInfo(dimensions, rank, this, getContext()->getWorkspace(),resetStrides);
    if(ArrayOptions::dataType(shapeInfoPermuted) == sd::DataType::UNKNOWN) {
      THROW_EXCEPTION("NDArray::permute: unknown data type of new shape !");
    }
    auto buff = ConstantShapeHelper::getInstance().createFromExisting(shapeInfoPermuted);
    DataBuffer *db = new DataBuffer(shape::length(buff) *
                                    DataTypeUtils::sizeOf(ArrayOptions::dataType(buff)),
                                    ArrayOptions::dataType(buff), getContext()->getWorkspace());
    NDArray *ret = new NDArray(db,
                               buff,
                               getContext());

    this->applyTransform(transform::Assign, ret, nullptr);
    NDArray &ret2 =  *ret;
    return ret2;

  } else {
    // evaluate shapeInfo for output (permuted) array ret
    auto shapeInfoPermuted = ShapeUtils::evalPermShapeInfo(dimensions, rank, this, getContext()->getWorkspace(),resetStrides);
    if(ArrayOptions::dataType(shapeInfoPermuted) == sd::DataType::UNKNOWN) {
      THROW_EXCEPTION("NDArray::permute: unknown data type of new shape !");
    }
    auto buff = ConstantShapeHelper::getInstance().createFromExisting(shapeInfoPermuted);

    NDArray *ret = new NDArray(getDataBuffer(), buff, getContext());
    NDArray &ret2 =  *ret;

    return ret2;
  }

}

//////////////////////////////////////////////////////////////////////////
NDArray &NDArray::permute(LongType *dimensions, const int rank, const bool copyToNewBuff, const bool resetStrides) && {
  return permute(dimensions, rankOf(), copyToNewBuff, resetStrides);
}

//////////////////////////////////////////////////////////////////////////
NDArray &NDArray::permute(std::vector<LongType> &dimensions, bool copyToNewBuff, bool resetStrides) & {
  if(dimensions.size() < 1)
    return *this;
  return permute(dimensions.data(), rankOf(), copyToNewBuff, resetStrides);
}

//////////////////////////////////////////////////////////////////////////
NDArray &NDArray::permute(std::vector<LongType> &dimensions,
                          const bool copyToNewBuff,
                          const bool resetStrides) && {
  if(dimensions.size() < 1)
    return *this;
  return permute(dimensions.data(), rankOf(), copyToNewBuff, false);
}


//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
void NDArray::permute(LongType *dimensions, const int rank, NDArray &target, const bool resetStrides) {
  if (!nonNull() || !target.nonNull() || rank != rankOf() || rank != target.rankOf())
    THROW_EXCEPTION("NDArray<T>::permute method: either arrays are nullptr or ranks are not suitable!");

  auto shapeInfoNew = ShapeUtils::evalPermShapeInfo(dimensions, rank, this, target.getContext()->getWorkspace(),resetStrides);
  target._buffer = _buffer;
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfoNew);
  target._shapeInfo = constDesc->primary();
  target._shapeInfoD = constDesc->special();
}

//////////////////////////////////////////////////////////////////////////




//////////////////////////////////////////////////////////////////////////
// check whether array is identity matrix
bool NDArray::isIdentityMatrix() {
  if (isS()) THROW_EXCEPTION("NDArray::isIdentityMatrix: you can't use this method on String array!");
  if (rankOf() != 2 || rows() != columns())
    THROW_EXCEPTION("isIdentityMatrix method: matrix must be square and have rank = 2 !");

  const double eps = 1e-5f;
  for (sd::LongType i = 0; i < rows(); ++i)
    if (sd::math::sd_abs<double,double>(e<double>(i, i) - 1.f) > eps) return false;

  for (sd::LongType i = 0; i < rows(); ++i) {
    for (sd::LongType j = 0; j < columns(); ++j) {
      if (i == j) continue;
      if (sd::math::sd_abs<double,double>(e<double>(i, j)) > eps) return false;
    }
  }
  return true;
}

//////////////////////////////////////////////////////////////////////////
// check whether array is unitary matrix
bool NDArray::isUnitary() {
  if (isS()) THROW_EXCEPTION("NDArray::isUnitary: you can't use this method on String array!");
  if (rankOf() != 2 || rows() != columns())
    THROW_EXCEPTION("isUnitary method: matrix must be square and have rank = 2 !");

  auto tr = this->transpose();
  auto trMul = MmulHelper::mmul(this, &tr, nullptr, 1.f, 0.f);

  bool result = trMul->isIdentityMatrix();
  delete trMul;

  return result;
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////


template <typename T>
T *NDArray::specialBufferasTWithOffset(LongType offset) {
  return reinterpret_cast<T *>(specialBufferWithOffset(offset));
}

BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT, *NDArray::specialBufferasTWithOffset(LongType), SD_COMMON_TYPES);


template <typename T>
T *NDArray::specialBufferasT() {
  return specialBufferasTWithOffset<T>(0);
}

BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT, *NDArray::specialBufferasT(), SD_COMMON_TYPES);


template <typename T>
T *NDArray::bufferAsT() {
  return bufferasTWithOffset<T>(0);
}

BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT, *NDArray::bufferAsT(), SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
template <typename T>
T *NDArray::bufferasTWithOffset(sd::LongType offset) {
  return reinterpret_cast<T *>(bufferWithOffset(offset));
}

BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT, *NDArray::bufferasTWithOffset(sd::LongType),
                                SD_COMMON_TYPES_ALL);

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArray::asT()  {
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = nonConst->getShapeAsVector();
  std::vector<sd::LongType> empty = {};
  std::vector<double> zeroVec = {0.};
  auto result = isScalar() ? NDArray('c', empty, zeroVec, DataTypeUtils::fromT<T>(), this->getContext())
                           : NDArray(ordering(), shape, DataTypeUtils::fromT<T>(), this->getContext());

  prepareSpecialUse({&result}, {this});
  NativeOpExecutioner::execTransformAny(getContext(), transform::AnyOps::Assign, buffer(), shapeInfo(), specialBuffer(),
                                        specialShapeInfo(), result.buffer(), result.shapeInfo(), result.specialBuffer(),
                                        result.specialShapeInfo(), nullptr, false);
  registerSpecialUse({&result}, {this});

  return result;
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT NDArray NDArray::asT, (), SD_COMMON_TYPES);
void NDArray::checkIfStringArrayAndNotEmpty()  {
  if (!isS()) {
    auto actualType = DataTypeUtils::asString(dataType());
    std::string errorMessage;
    errorMessage += "checkIfStringArrayAndNotEmpty: Expected String array but found ";
    errorMessage += actualType;
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (isEmpty()) {
    THROW_EXCEPTION("checkIfStringArrayAndNotEmpty: Array is empty. Cannot proceed");
  }
}

void NDArray::printStringType()  {
  switch (dataType()) {
    case DataType::UTF8:
      std::cout << "Data Type: UTF8" << "\n";
      break;
    case DataType::UTF16:
      std::cout << "Data Type: UTF16" << "\n";
      break;
    case DataType::UTF32:
      std::cout << "Data Type: UTF32" << "\n";
      break;
    default:
      THROW_EXCEPTION("printStringType: Unsupported data type");
  }
}

void NDArray::printStringInternalState()  {
  checkIfStringArrayAndNotEmpty();
  printStringType();

  // Length of offsets (header)
  sd::LongType offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());

  // Getting the buffer pointer
  const auto nInputoffsets = bufferAsT<sd::LongType>();
  std::cout << "Number of elements: " << lengthOf() << "\n";

  int numStrings = isScalar() ? 1 : lengthOf();
  for (sd::LongType e = 0; e < numStrings; e++) {
    sd::LongType start = nInputoffsets[e];
    sd::LongType stop = nInputoffsets[e + 1];
    sd::LongType stringLength = stop - start;

    std::cout << "String at index " << e << " Offset: " << start << " Length: " << stringLength << "\n";
  }
}

void NDArray::debugStringArray() { printStringInternalState();
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArray::asS()  {
  if (!isS()) THROW_EXCEPTION("NDArray::asS: you can use this method only for String array!");

  auto dtype = DataTypeUtils::fromT<T>();

  if (!(DataTypeUtils::isS(dtype))) THROW_EXCEPTION("NDArray::asS: invalid DataType used");

  // If the data types are the same, then simply duplicate the array
  if (dtype == dataType()) {
    return dup(ordering());
  }

  // Calculate buffer length requirements
  sd::LongType offsetsLength = ShapeUtils::stringBufferHeaderRequirements(lengthOf());
  std::vector<sd::LongType> offsets = StringUtils::calculateOffsetsForTargetDataType<T>(this);

  sd::LongType dataLength = offsets.back();

  DataBuffer *  pBuffer =
      new DataBuffer(offsetsLength + dataLength, dtype, getContext()->getWorkspace(), true);
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape2 = nonConst->getShapeAsVector();
  std::vector<sd::LongType> shape = isScalar() ? std::vector<sd::LongType>({1}) : shape2;
  auto desc = new ShapeDescriptor(dtype, ordering(), shape);
  NDArray res(pBuffer, desc, getContext());
  res.setAttached(getContext()->getWorkspace() != nullptr);

  preparePrimaryUse({&res}, {this});

  // Copy offsets
  memcpy(res.bufferAsT<int8_t>(), offsets.data(), offsetsLength * sizeof(sd::LongType));

  // Convert string data
  StringUtils::convertStringsForDifferentDataType<T>(this, &res);

  registerPrimaryUse({&res}, {this});

  return res;
}
////////////////////////////////////////////////////////////////////////
NDArray NDArray::asT(DataType dtype)  {
  if (isS() && !DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::asT: you can't use this method on String array with not string DataType!");

  if (!isS() && DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::asT: you can't use this method on not String array with string DataType!");

  if (isS()) {
    BUILD_SINGLE_SELECTOR(dtype, return asS, (), SD_STRING_TYPES);
  } else {
    BUILD_SINGLE_SELECTOR(dtype, return asT, (), SD_COMMON_TYPES);
  }

}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::cast(DataType dtype)  {
  if (isS() && !DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::cast: you can't use this method on String array with not string DataType!");

  if (!isS() && DataTypeUtils::isS(dtype))
    THROW_EXCEPTION("NDArray::cast: you can't use this method on not String array with string DataType!");

  return this->asT(dtype);
}

////////////////////////////////////////////////////////////////////////
void NDArray::cast(NDArray &target, DataType dtype) {
  if (isS()) THROW_EXCEPTION("NDArray::cast: you can't use this method on String array!");
  // TODO: to be implemented properly
  target.assign(this);
}


////////////////////////////////////////////////////////////////////////
// mathematical multiplication of two arrays
NDArray mmul(NDArray &left, NDArray &right) {
  if (left.isS() || right.isS()) THROW_EXCEPTION("mmul friend function: you can't use this function on String array!");
  auto ptr = MmulHelper::mmul(const_cast<NDArray *>(&left), const_cast<NDArray *>(&right), nullptr, 1., 0.);
  NDArray result(std::move(*ptr));
  delete ptr;
  return result;
}

////////////////////////////////////////////////////////////////////////
void NDArray::tileToShape(const std::vector<sd::LongType> &shape, NDArray &target) {
  if (&target != this) {
    this->tile(target);
    return;
  }

  std::vector<sd::LongType> thisShape(rankOf());
  for (int i = 0; i < rankOf(); ++i) thisShape[i] = sizeAt(i);

  if (!ShapeUtils::areShapesBroadcastable(shape, thisShape))
    THROW_EXCEPTION(
        "NDArray::tileToShape method: the shape of this array and input shape are not suitable for broadcast operation "
        "!");

  const int newRank = shape.size();
  std::vector<sd::LongType> repeats(newRank);

  for (int i = 1; i <= newRank; ++i) {
    if (i > rankOf())
      repeats[newRank - i] = shape[newRank - i];
    else
      repeats[newRank - i] = shape[newRank - i] / thisShape[rankOf() - i];
  }

  tilei(repeats);
}

////////////////////////////////////////////////////////////////////////
void NDArray::tileToShape(const std::initializer_list<sd::LongType> &shape, NDArray &target) {
  tileToShape(std::vector<sd::LongType>(shape), target);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::tileToShape(const sd::LongType *shapeInfo) {
  NDArray result(const_cast<sd::LongType *>(shapeInfo), false, getContext());
  tile(result);
  return result;
}

////////////////////////////////////////////////////////////////////////
double NDArray::getTrace()  {
  if (isS()) THROW_EXCEPTION("NDArray::getTrace: you can't use this method on String array!");

  int rank = rankOf();
  auto shape = shapeOf();
  int minDim = 100000000;

  sd::LongType indices[SD_MAX_RANK];
  for (int j = 0; j < rank; ++j) indices[j] = 1;

  sd::LongType offset;
  COORDS2INDEX(shape::rank(shapeInfo()), shape::stride(shapeInfo()), indices, offset);
  for (int i = 0; i < rank; ++i)
    if (minDim > shape[i]) minDim = shape[i];

  double sum = 0.;

  for (int i = 0; i < minDim; ++i) sum += e<double>(i * offset);

  return sum;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::quantize(NDArray &array) {
  if (!array.isR()) THROW_EXCEPTION("NDArray::quantize: type of array should be from real space!");

  auto ws = array.getContext()->getWorkspace();

  sd::LongType *shapeInfo = ShapeBuilders::copyShapeInfo(array.shapeInfo(), true, ws);
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_QUANTIZED);

  int len = array.isScalar() ? 1 : array.lengthOf();
  DataBuffer *  buffer = new DataBuffer(TypeCast::estimateQuantizedSize(len),
                                        ArrayOptions::dataType(shapeInfo), ws);

  auto desc = new ShapeDescriptor(shapeInfo, false);
  NDArray result(buffer, desc, array.getContext());

  return result;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyTrueBroadcast(sd::BroadcastOpsTuple op, NDArray *other, NDArray *target,
                                 const bool checkTargetShape, ExtraArguments *extraArgs)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyTrueBroadcast: you can't use this method on String array!");

  if (((op.s == scalar::Divide || op.s == scalar::FloorDiv || op.s == scalar::FloorMod) && other->isB()) ||
      (op.s == scalar::ReverseDivide && this->isB()))
    THROW_EXCEPTION("NDArray::applyTrueBroadcast method: you can't divide by bool array !");

  if (isEmpty() || other->isEmpty()) return;
  if (checkTargetShape) {
    sd::LongType *newShapeInfo = nullptr;
    if (!ShapeUtils::evalBroadcastShapeInfo(
        this->shapeInfo(), other->shapeInfo(), true, newShapeInfo,
        getContext()->getWorkspace()))  // the rank of target array must be equal to max->rankOf)()
      THROW_EXCEPTION(
          "NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast "
          "operation !");
  }
  sd::LongType const *xShapeInfoH = shapeInfo();
  sd::LongType const *yShapeInfoH = other->shapeInfo();
  sd::LongType const *xShapeInfoD = specialShapeInfo();
  sd::LongType const *yShapeInfoD = other->specialShapeInfo();

  if (!isSameShape(target)) {
    auto xPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), shapeInfo(), getContext()->getWorkspace());
    xShapeInfoH = xPack->primary();
    xShapeInfoD = xPack->special();
  }
  if (!other->isSameShape(target)) {
    auto yPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), other->shapeInfo(), other->getContext()->getWorkspace());
    yShapeInfoH = yPack->primary();
    yShapeInfoD = yPack->special();
  }

  prepareUse({target}, {this, other});
  NativeOpExecutioner::execBroadcast(getContext(), op.b, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD,
                                     other->buffer(), yShapeInfoH, other->specialBuffer(), yShapeInfoD, target->buffer(),
                                     target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo());
  registerUse({target}, {this, other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyTrueBroadcast(sd::BroadcastBoolOpsTuple op, NDArray *other, NDArray *target,
                                 const bool checkTargetShape, ExtraArguments *extraArgs)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");

  if (isEmpty() || other->isEmpty()) return;

  if (checkTargetShape) {
    sd::LongType *newShapeInfo = nullptr;
    if (!ShapeUtils::evalBroadcastShapeInfo(
        this->shapeInfo(), other->shapeInfo(), true, newShapeInfo,
        getContext()->getWorkspace())) {  // the rank of target array must be equal to max->rankOf)()
      std::string errorMessage;
      errorMessage += "NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for "
                      "broadcast operation !";
      errorMessage += " this array shape is ";
      errorMessage += ShapeUtils::shapeAsString(shapeInfo());
      errorMessage += " other array shape is ";
      errorMessage += ShapeUtils::shapeAsString(other->shapeInfo());
      errorMessage += " target array shape is ";
      errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
      errorMessage += " new shape is ";
      errorMessage += ShapeUtils::shapeAsString(newShapeInfo);
      errorMessage += " target array type is ";
      errorMessage += DataTypeUtils::asString(target->dataType());
      errorMessage += " this array type is ";
      errorMessage += DataTypeUtils::asString(dataType());
      errorMessage += " other array type is ";
      errorMessage += DataTypeUtils::asString(other->dataType());
      THROW_EXCEPTION(errorMessage.c_str());
    }
    if (!shape::equalsSoft(target->_shapeInfo, newShapeInfo) || target->dataType() != DataType::BOOL) {
      std::string errorMessage;
      errorMessage += "NDArray::applyTrueBroadcast bool method: the shape or type of target array is wrong !";
      errorMessage += " target array type is ";
      errorMessage += DataTypeUtils::asString(target->dataType());
      errorMessage += " this array type is ";
      errorMessage += DataTypeUtils::asString(dataType());
      errorMessage += " other array type is ";
      errorMessage += DataTypeUtils::asString(other->dataType());
      THROW_EXCEPTION(errorMessage.c_str());
    }

  }

  sd::LongType const *xShapeInfoH = shapeInfo();
  sd::LongType const *yShapeInfoH = other->shapeInfo();
  sd::LongType const *xShapeInfoD = specialShapeInfo();
  sd::LongType const *yShapeInfoD = other->specialShapeInfo();

  if (!isSameShape(target)) {
    auto xPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), shapeInfo(), getContext()->getWorkspace());
    xShapeInfoH = xPack->primary();
    xShapeInfoD = xPack->special();
  }
  if (!other->isSameShape(target)) {
    auto yPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), other->shapeInfo(), other->getContext()->getWorkspace());
    yShapeInfoH = yPack->primary();
    yShapeInfoD = yPack->special();
  }

  prepareSpecialUse({target}, {this, other});
  NativeOpExecutioner::execBroadcastBool(getContext(), op.b,
                                         buffer(),
                                         xShapeInfoH,
                                         specialBuffer(),
                                         xShapeInfoD,
                                         other->buffer(),
                                         yShapeInfoH,
                                         other->specialBuffer(),
                                         yShapeInfoD,
                                         target->buffer(),
                                         target->shapeInfo(),
                                         target->specialBuffer(),
                                         target->specialShapeInfo(), nullptr);
  registerSpecialUse({target}, {this, other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyTrueBroadcast(sd::BroadcastIntOpsTuple op, NDArray *other, NDArray *target,
                                 const bool checkTargetShape, ExtraArguments *extraArgs)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyTrueBroadcast bool: you can't use this method on String array!");

  if (isEmpty() || other->isEmpty()) return;


  if (checkTargetShape) {
    sd::LongType *newShapeInfo = nullptr;
    if (!ShapeUtils::evalBroadcastShapeInfo(
        this->shapeInfo(), other->shapeInfo(), false, newShapeInfo,
        getContext()->getWorkspace())) {  // the rank of target array must be equal to max->rankOf)()
      std::string errorMessage;
      errorMessage += "NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for "
                      "broadcast operation !";
      errorMessage += " this array shape is ";
      errorMessage += ShapeUtils::shapeAsString(shapeInfo());
      errorMessage += " other array shape is ";
      errorMessage += ShapeUtils::shapeAsString(other->shapeInfo());
      errorMessage += " target array shape is ";
      errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
      errorMessage += " new shape is ";
      errorMessage += ShapeUtils::shapeAsString(newShapeInfo);
      errorMessage += " target array type is ";
      errorMessage += DataTypeUtils::asString(target->dataType());
      errorMessage += " this array type is ";
      errorMessage += DataTypeUtils::asString(dataType());
      errorMessage += " other array type is ";
      errorMessage += DataTypeUtils::asString(other->dataType());
      THROW_EXCEPTION(errorMessage.c_str());
    }
    if (!shape::equalsSoft(target->_shapeInfo, newShapeInfo) || target->dataType() != this->dataType()) {
      std::string errorMessage;
      errorMessage += "NDArray::applyTrueBroadcast int method: the shape or type of target array is wrong !";
      errorMessage += " target array type is ";
      errorMessage += DataTypeUtils::asString(target->dataType());
      errorMessage += " this array type is ";
      errorMessage += DataTypeUtils::asString(dataType());
      errorMessage += " other array type is ";
      errorMessage += DataTypeUtils::asString(other->dataType());
      THROW_EXCEPTION(errorMessage.c_str());
    }

  }

  sd::LongType const *xShapeInfoH = shapeInfo();
  sd::LongType const *yShapeInfoH = other->shapeInfo();
  sd::LongType const *xShapeInfoD = specialShapeInfo();
  sd::LongType const *yShapeInfoD = other->specialShapeInfo();

  if (!isSameShape(target)) {
    auto xPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), shapeInfo(), getContext()->getWorkspace());
    xShapeInfoH = reinterpret_cast<sd::LongType const *>(xPack->primary());
    xShapeInfoD = reinterpret_cast<sd::LongType const *>(xPack->special());
  }
  if (!other->isSameShape(target)) {
    auto yPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), other->shapeInfo(), other->getContext()->getWorkspace());
    yShapeInfoH = reinterpret_cast<sd::LongType const *>(yPack->primary());
    yShapeInfoD = reinterpret_cast<sd::LongType const *>(yPack->special());
  }

  prepareUse({target}, {this, other});
  NativeOpExecutioner::execBroadcastInt(getContext(), op.b, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD,
                                        other->buffer(), yShapeInfoH, other->specialBuffer(), yShapeInfoD,
                                        target->buffer(), target->shapeInfo(), target->specialBuffer(),
                                        target->specialShapeInfo());
  registerUse({target}, {this, other});
}

//////////////////////////////////////////////////////////////////////////
NDArray *NDArray::applyTrueBroadcast(BroadcastOpsTuple op, NDArray *other, ExtraArguments *extraArgs)  {
  if (isEmpty() || other->isEmpty()) {
    if (isEmpty())
      return this;
    else
      return other;
  }

  sd::LongType *newShapeInfo = nullptr;
  if (!ShapeUtils::evalBroadcastShapeInfo(this->shapeInfo(), other->shapeInfo(), true, newShapeInfo,
                                          getContext()->getWorkspace()))  { // the rank of new array = max->rankOf)()

    std::string errorMessage;
    errorMessage += "NDArray::applyTrueBroadcast method: the shapes of this and other arrays are not suitable for broadcast operation !";
    errorMessage += " this array shape is ";
    errorMessage += ShapeUtils::shapeAsString(shapeInfo());
    errorMessage += " other array shape is ";
    errorMessage += ShapeUtils::shapeAsString(other->shapeInfo());
    errorMessage += " new array shape is ";
    errorMessage += ShapeUtils::shapeAsString(newShapeInfo);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  NDArray *result = new NDArray(newShapeInfo, true, getContext());

  this->applyTrueBroadcast(op, other, result, false, extraArgs);

  return result;
}



//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::Ops op, const std::vector<LongType> *dimensions, NDArray *tad,
                             NDArray *target, ExtraArguments *extraArgs) {
  if (dimensions->size() == 0) return;

  if (isS()) THROW_EXCEPTION("NDArray::applyBroadcast: you can't use this method on String array!");
  if (((op == broadcast::Divide || op == broadcast::FloorDiv || op == broadcast::FloorMod) && tad->isB()) ||
      (op == broadcast::ReverseDivide && this->isB())) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast method: you can't divide by bool array !";
    errorMessage += " this array type is ";
    errorMessage += DataTypeUtils::asString(dataType());
    errorMessage += " other array type is ";
    errorMessage += DataTypeUtils::asString(tad->dataType());
    errorMessage += " target array type is ";
    errorMessage += DataTypeUtils::asString(target->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (isEmpty() || tad->isEmpty()) {
    if (!target->isEmpty()) {
      std::string errorMessage;
      errorMessage += "NDArray::applyBroadcast method: when some of input arrays (or both) is empty, target array must be empty as well !";
      errorMessage += " this array shape is ";
      errorMessage += ShapeUtils::shapeAsString(shapeInfo());
      errorMessage += " other array shape is ";
      errorMessage += ShapeUtils::shapeAsString(tad->shapeInfo());
      errorMessage += " target array shape is ";
      errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
      THROW_EXCEPTION(errorMessage.c_str());
    }
    return;
  }

  if (target->dataType() != DataTypeUtils::pickPairwiseResultType(shapeInfo(), tad->shapeInfo())) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast method: wrong type of target array !";
    errorMessage += " this array type is ";
    errorMessage += DataTypeUtils::asString(dataType());
    errorMessage += " other array type is ";
    errorMessage += DataTypeUtils::asString(tad->dataType());
    errorMessage += " target array type is ";
    errorMessage += DataTypeUtils::asString(target->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (!target->isSameShape(this) && !target->isSameShape(tad)) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast method: one of of two input arrays (this or other) should has the same shape as target array!";
    errorMessage += " this array shape is ";
    errorMessage += ShapeUtils::shapeAsString(shapeInfo());
    errorMessage += " other array shape is ";
    errorMessage += ShapeUtils::shapeAsString(tad->shapeInfo());
    errorMessage += " target array shape is ";
    errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());

  }
  std::vector<LongType> copy(*dimensions);

  if (dimensions->size() > 1) std::sort(copy.begin(), copy.end());

  sd::LongType const *xShapeInfoH = shapeInfo();
  sd::LongType const *yShapeInfoH = tad->shapeInfo();
  sd::LongType const *xShapeInfoD = specialShapeInfo();
  sd::LongType const *yShapeInfoD = tad->specialShapeInfo();

  if (!isSameShape(target)) {
    auto xPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), shapeInfo(), getContext()->getWorkspace(), copy);
    xShapeInfoH = reinterpret_cast<sd::LongType const *>(xPack->primary());
    xShapeInfoD = reinterpret_cast<sd::LongType const *>(xPack->special());
  }
  if (!tad->isSameShape(target)) {
    auto yPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), tad->shapeInfo(), tad->getContext()->getWorkspace(), copy);
    yShapeInfoH = reinterpret_cast<sd::LongType const *>(yPack->primary());
    yShapeInfoD = reinterpret_cast<sd::LongType const *>(yPack->special());
  }

  prepareUse({target}, {this, tad});
  NativeOpExecutioner::execBroadcast(getContext(), op, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD,
                                     tad->buffer(), yShapeInfoH, tad->specialBuffer(), yShapeInfoD, target->buffer(),
                                     target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo());
  registerUse({target}, {this, tad});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::BoolOps op, const std::vector<LongType> *dimensions, NDArray *tad,
                             NDArray *target, ExtraArguments *extraArgs) {
  if (dimensions->size() == 0) return;

  if (isS()) THROW_EXCEPTION("NDArray::applyBroadcast BoolOps: you can't use this method on String array!");
  if (isEmpty() || tad->isEmpty()) {
    if (!target->isEmpty()) {
      std::string errorMessage;
      errorMessage += "NDArray::applyBroadcast BoolOps: when some of input arrays (or both) is empty, target array must be empty as well !";
      errorMessage += " this array shape is ";
      errorMessage += ShapeUtils::shapeAsString(shapeInfo());
      errorMessage += " other array shape is ";
      errorMessage += ShapeUtils::shapeAsString(tad->shapeInfo());
      errorMessage += " target array shape is ";
      errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
      THROW_EXCEPTION(errorMessage.c_str());

    }
    return;
  }

  if (target->dataType() != DataType::BOOL) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast BoolOps: type of target array must be BOOL!";
    errorMessage += " target array type is ";
    errorMessage += DataTypeUtils::asString(target->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (!target->isSameShape(this) && !target->isSameShape(tad)) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast BoolOps: one of of two input arrays (this or other) should has the same shape as target array!";
    errorMessage += " this array shape is ";
    errorMessage += ShapeUtils::shapeAsString(shapeInfo());
    errorMessage += " other array shape is ";
    errorMessage += ShapeUtils::shapeAsString(tad->shapeInfo());
    errorMessage += " target array shape is ";
    errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (ArrayOptions::dataType(_shapeInfo) != ArrayOptions::dataType(tad->_shapeInfo)) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast BoolOps: this and other arrays must have the same type !";
    errorMessage += " this array type is ";
    errorMessage += DataTypeUtils::asString(dataType());
    errorMessage += " other array type is ";
    errorMessage += DataTypeUtils::asString(tad->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  std::vector<LongType> copy(*dimensions);

  if (dimensions->size() > 1) std::sort(copy.begin(), copy.end());

  sd::LongType const *xShapeInfoH = shapeInfo();
  sd::LongType const *yShapeInfoH = tad->shapeInfo();
  sd::LongType const *xShapeInfoD = specialShapeInfo();
  sd::LongType const *yShapeInfoD = tad->specialShapeInfo();

  if (!isSameShape(target)) {
    auto xPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), shapeInfo(), getContext()->getWorkspace(), copy);
    xShapeInfoH = reinterpret_cast<sd::LongType const *>(xPack->primary());
    xShapeInfoD = reinterpret_cast<sd::LongType const *>(xPack->special());
  }
  if (!tad->isSameShape(target)) {
    auto yPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), tad->shapeInfo(), tad->getContext()->getWorkspace(), copy);
    yShapeInfoH = reinterpret_cast<sd::LongType const *>(yPack->primary());
    yShapeInfoD = reinterpret_cast<sd::LongType const *>(yPack->special());
  }

  prepareUse({target}, {this, tad});
  NativeOpExecutioner::execBroadcastBool(getContext(), op, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD,
                                         tad->buffer(), yShapeInfoH, tad->specialBuffer(), yShapeInfoD, target->buffer(),
                                         target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo(),
                                         nullptr);
  registerUse({target}, {this, tad});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::IntOps op, const std::vector<LongType> *dimensions, NDArray *tad,
                             NDArray *target, ExtraArguments *extraArgs) {
  if (dimensions->empty()) return;

  if (!isZ()) THROW_EXCEPTION("NDArray::applyBroadcast IntOps: you can't use this method on non-Integer array!");
  if (isEmpty() || tad->isEmpty()) {
    if (!target->isEmpty()) {
      std::string errorMessage;
      errorMessage += "NDArray::applyBroadcast IntOps: when some of input arrays (or both) is empty, target array must be empty as well !";
      errorMessage += " this array shape is ";
      errorMessage += ShapeUtils::shapeAsString(shapeInfo());
      errorMessage += " other array shape is ";
      errorMessage += ShapeUtils::shapeAsString(tad->shapeInfo());
      errorMessage += " target array shape is ";
      errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
      THROW_EXCEPTION(errorMessage.c_str());
    }
    return;
  }

  if (target->dataType() != dataType()) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast IntOps: type of target array must be the same as this array type!";
    errorMessage += " this array type is ";
    errorMessage += DataTypeUtils::asString(dataType());
    errorMessage += " target array type is ";
    errorMessage += DataTypeUtils::asString(target->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (!target->isSameShape(this) && !target->isSameShape(tad)) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast IntOps: one of of two input arrays (this or other) should has the same shape as target array!";
    errorMessage += " this array shape is ";
    errorMessage += ShapeUtils::shapeAsString(shapeInfo());
    errorMessage += " other array shape is ";
    errorMessage += ShapeUtils::shapeAsString(tad->shapeInfo());
    errorMessage += " target array shape is ";
    errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (ArrayOptions::dataType(_shapeInfo) != ArrayOptions::dataType(tad->_shapeInfo)) {
    std::string errorMessage;
    errorMessage += "NDArray::applyBroadcast IntOps: this and other arrays must have the same type !";
    errorMessage += " this array type is ";
    errorMessage += DataTypeUtils::asString(dataType());
    errorMessage += " other array type is ";
    errorMessage += DataTypeUtils::asString(tad->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  std::vector<LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  if (dimensions->size() > 1) std::sort(copy->begin(), copy->end());

  sd::LongType const *xShapeInfoH = shapeInfo();
  sd::LongType const *yShapeInfoH = tad->shapeInfo();
  sd::LongType const *xShapeInfoD = specialShapeInfo();
  sd::LongType const *yShapeInfoD = tad->specialShapeInfo();

  if (!isSameShape(target)) {
    auto xPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), shapeInfo(), getContext()->getWorkspace(), *copy);
    xShapeInfoH = reinterpret_cast<sd::LongType const *>(xPack->primary());
    xShapeInfoD = reinterpret_cast<sd::LongType const *>(xPack->special());
  }
  if (!tad->isSameShape(target)) {
    auto yPack = ConstantShapeHelper::getInstance().createShapeInfoWithUnitiesForBroadcast(
        target->shapeInfo(), tad->shapeInfo(), tad->getContext()->getWorkspace(), *copy);
    yShapeInfoH = reinterpret_cast<sd::LongType const *>(yPack->primary());
    yShapeInfoD = reinterpret_cast<sd::LongType const *>(yPack->special());
  }

  prepareUse({target}, {this, tad});
  NativeOpExecutioner::execBroadcastInt(getContext(), op, buffer(), xShapeInfoH, specialBuffer(), xShapeInfoD,
                                        tad->buffer(), yShapeInfoH, tad->specialBuffer(), yShapeInfoD, target->buffer(),
                                        target->shapeInfo(), target->specialBuffer(), target->specialShapeInfo());

  // delete copy;
  registerUse({target}, {this, tad});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyBroadcast(sd::broadcast::Ops op, const std::initializer_list<LongType> *dimensions,
                             NDArray *tad, NDArray *target, ExtraArguments *extraArgs) {
  std::vector<LongType> vec(*dimensions);
  applyBroadcast(op, &vec, tad, target, extraArgs);
}

////////////////////////////////////////////////////////////////////////
void *NDArray::operator new(size_t i) {
  if (sd::memory::MemoryRegistrator::getInstance().hasWorkspaceAttached()) {
    sd::memory::Workspace *ws = sd::memory::MemoryRegistrator::getInstance().getWorkspace();
    return ws->allocateBytes((sd::LongType)i);
  } else {
    auto p = malloc(i);
    CHECK_ALLOC(p, "Failed to allocate new NDArray", i);
    return p;
  }
}

////////////////////////////////////////////////////////////////////////
void NDArray::operator delete(void *p) {
  if (!sd::memory::MemoryRegistrator::getInstance().hasWorkspaceAttached()) {
    free(p);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> NDArray::asVectorT() {
  if(isScalar()) {
    std::vector<T> result(1);
    result[0] = this->e<T>(0);
    return result;
  }

  if(isEmpty()) {
    sd_debug("asVectorT before return empty vector\n",0);
    return std::vector<T>();
  }


  int len = isScalar() ? 1 : lengthOf();

  std::vector<T> result(len);
  for (int e = 0; e < len; e++) {
    result[e] = this->e<T>(e);
  }

  return result;
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT std::vector, NDArray::asVectorT(), SD_COMMON_TYPES_ALL);

//////////////////////////////////////////////////////////////////////////
// set new order and shape in case of suitable array length
bool NDArray::reshapei(const char order, const std::vector<sd::LongType> &cshape) {
  // check firstly whether cshape is identical to shape of array, if yes then reshape is unnecessary
  if (order == ordering() && shape::shapeEquals(rankOf(), shapeOf(), cshape.size(), cshape.data())) return true;

  const bool isOutShapeEmpty = std::find(cshape.begin(), cshape.end(), 0) != cshape.end();

  if (isEmpty() && !isOutShapeEmpty) {
    std::string errorMessage;
    errorMessage += "NDArray::reshapei: can't reshape empty array to non-empty !\n";
    errorMessage += "Empty array shape: ";
    errorMessage += std::string(shape::shapeInfoString(shapeInfo()));
    errorMessage += "\n";
    errorMessage += "New shape: ";
    errorMessage += std::string(shape::shapeInfoString(this->shapeInfo()));
    errorMessage += "\n";
    errorMessage += "Order: ";
    errorMessage += this->ordering();
    errorMessage += "\n";
    THROW_EXCEPTION(errorMessage.c_str());

  }

  if (isEmpty() && isOutShapeEmpty) {
    sd::LongType *shapeInfoNew = ShapeBuilders::emptyShapeInfo(dataType(), order, cshape, getContext()->getWorkspace());
    auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfoNew);
    _shapeInfo = constDesc->primary();
    _shapeInfoD = constDesc->special();
    return true;
  }

  std::vector<sd::LongType> shape(cshape);
  int rank = shape.size();

  // looking for negative in shape

  int numberNegativesOnes = 0;

  sd::LongType *shape_ = shape.data();
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] < 0) {
      if (numberNegativesOnes >= 1) {
        std::string errorMessage;
        errorMessage += "NDArray::reshapei: only one dimension can be negative at once !\n";
        errorMessage += "Shape: ";
        errorMessage += ShapeUtils::shapeAsString(this);
        errorMessage += "\n";
        errorMessage += "New shape: ";
        errorMessage += ShapeUtils::shapeAsString(shape);
        errorMessage += "\n";
        errorMessage += "Order: ";
        errorMessage += this->ordering();
        errorMessage += "\n";
        THROW_EXCEPTION(errorMessage.c_str());
      }

      numberNegativesOnes++;

      sd::LongType shapeLength = 1;
      for (size_t j = 0; j < shape.size(); j++)
        if (i != j) shapeLength *= shape_[j];

      sd::LongType realShape = sd::math::sd_abs<sd::LongType,sd::LongType>(lengthOf() / shapeLength);
      auto thisNewShape = new sd::LongType[shape.size()];

      for (size_t j = 0; j < shape.size(); j++)
        if (i != j)
          thisNewShape[j] = shape_[j];
        else
          thisNewShape[j] = realShape;

      shape_ = thisNewShape;
    }
  }

  for (size_t e = 0; e < shape.size(); e++) shape[e] = shape_[e];


  sd::LongType arrLength = 1;
  for (const auto &item : shape) arrLength *= item;

  //don't validate scalar case reshape 0 -> 1,1 should be valid
  if (arrLength != this->lengthOf() && !isScalar()) {
    std::string errorMessage;
    errorMessage += "NDArray::reshapei: bad length of new shape !\n";
    errorMessage += "Shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += "\n";
    errorMessage += "New shape: ";
    errorMessage += ShapeUtils::shapeAsString(shape);
    errorMessage += "\n";
    errorMessage += "Order: ";
    errorMessage += this->ordering();
    errorMessage += "\n";
    errorMessage += "Length of new shape: ";
    errorMessage += std::to_string(arrLength);
    errorMessage += "\n";
    errorMessage += "Length of array: ";
    errorMessage += std::to_string(this->lengthOf());
    errorMessage += "\n";
    errorMessage += "Number of elements in array: ";
    errorMessage += std::to_string(this->lengthOf());
    errorMessage += "\n";
    errorMessage += "Number of elements in new shape: ";
    errorMessage += std::to_string(arrLength);
    errorMessage += "\n";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  sd::LongType *shapeInfoNew;
  ALLOCATE(shapeInfoNew, getContext()->getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);

  bool canReshape = shape::reshapeC(shapeInfo(), order, shape.size(), shape.data(), shapeInfoNew);

  if(!ArrayOptions::hasPropertyBitSet(shapeInfoNew,sd::ArrayOptions::flagForDataType(ArrayOptions::dataType(_shapeInfo)))) {
    std::string errorMessage;
    errorMessage += "NDArray::reshapei: bad data type of new shape !\n";
    errorMessage += "Shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += "\n";
    errorMessage += "New shape: ";
    errorMessage += ShapeUtils::shapeAsString(shape);
    errorMessage += "\n";
    errorMessage += "Order: ";
    errorMessage += this->ordering();
    errorMessage += "\n";
    errorMessage += "Length of new shape: ";
    errorMessage += std::to_string(arrLength);
    errorMessage += "\n";
    errorMessage += "Length of array: ";
    errorMessage += std::to_string(this->lengthOf());
    errorMessage += "\n";
    errorMessage += "Original data type: ";
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(_shapeInfo));
    //add what the expected flag is and what the extra property flag is
    errorMessage += "\n";
    errorMessage += "Expected data type: ";
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(shapeInfoNew));
    errorMessage += "\n";
    errorMessage += "Extra property flag: ";
    errorMessage += std::to_string(ArrayOptions::extra(shapeInfoNew));
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (canReshape) {
    auto newShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfoNew);
    _shapeInfo = newShape->primary();
    _shapeInfoD = newShape->special();
  }

  RELEASE(shapeInfoNew, getContext()->getWorkspace());

  return canReshape;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::nullify() {
  if (isEmpty()) return;
  int baseAssign = 0;
  if (isView())
    assign(baseAssign);
  else
    _buffer->setToZeroBuffers();
}

////////////////////////////////////////////////////////////////////////

template <typename T, typename Y, typename>
void NDArray::templatedSet(void *buffer, LongType xOffset, DataType dtype, void *value) {
  BUILD_SINGLE_PARTIAL_SELECTOR(dtype, templatedSet<, T>(buffer, xOffset, value), SD_COMMON_TYPES);
}
BUILD_DOUBLE_TEMPLATE(template void NDArray::templatedSet,
                      (void *buffer, LongType xOffset, DataType dtype, void *value),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(pairwise::Ops op, NDArray *other, NDArray *target,
                                     ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyPairwiseTransform: you can't use this method on String array!");
  NativeOpExecutioner::execPairwiseTransform(
      getContext(), op,
      buffer(),
      shapeInfo(), specialBuffer(),
      specialShapeInfo(), other->buffer(),
      other->shapeInfo(),
      other->specialBuffer(), other->specialShapeInfo(),
      target->buffer(), target->shapeInfo(),
      target->specialBuffer(),
      target->specialShapeInfo(),
      extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::prepareSpecialUse({target}, {this, other});
  NDArray::registerSpecialUse({target}, {this, other});
  if (extraParams != nullptr) synchronize("NDArray::applyPairwiseTransform");
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(pairwise::BoolOps op, NDArray *other, NDArray *target,
                                     ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyPairwiseTransform BoolOps: you can't use this method on String array!");
  if (other->lengthOf() != target->lengthOf())
    THROW_EXCEPTION("NDArray::applyPairwiseTransform BoolOps method - lengths of arrays are mismatched");
  if (!target->isB()) THROW_EXCEPTION("NDArray::applyPairwiseTransform BoolOps method - result must have bool type");

  prepareUse({target}, {this, other});
  NativeOpExecutioner::execPairwiseBoolTransform(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other->buffer(), other->shapeInfo(),
      other->specialBuffer(), other->specialShapeInfo(), target->buffer(), target->shapeInfo(), target->specialBuffer(),
      target->specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  registerUse({target}, {this, other});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(pairwise::IntOps op, NDArray *other, NDArray *target,
                                     ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyPairwiseTransform IntOps: you can't use this method on String array!");
  if (other->lengthOf() != target->lengthOf())
    THROW_EXCEPTION("NDArray::applyPairwiseTransform IntOps method - lengths of arrays are mismatched");
  if (!target->isZ()) THROW_EXCEPTION("NDArray::applyPairwiseTransform IntOps method - result must have bool type");
  if (dataType() != other->dataType())
    THROW_EXCEPTION("NDArray::applyPairwiseTransform IntOps method - this and other arrays must have the same type !");

  prepareUse({target}, {this, other});
  NativeOpExecutioner::execPairwiseIntTransform(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), other->buffer(), other->shapeInfo(),
      other->specialBuffer(), other->specialShapeInfo(), target->buffer(), target->shapeInfo(), target->specialBuffer(),
      target->specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  registerUse({target}, {this, other});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyPairwiseTransform(pairwise::Ops op, NDArray *other, ExtraArguments *extraParams) {
  applyPairwiseTransform(op, other, this, extraParams);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void NDArray::templatedDoubleAssign(void *xBuffer,  sd::LongType xOffset,  void *yBuffer,
                                    sd::LongType yOffset)  {
  auto x = reinterpret_cast<X *>(xBuffer);
  const auto y = reinterpret_cast<const Y *>(yBuffer);
  if(x == nullptr)
    THROW_EXCEPTION("NDArray::templatedDoubleAssign: x buffer is nullptr !");
  if(y == nullptr)
    THROW_EXCEPTION("NDArray::templatedDoubleAssign: y buffer is nullptr !");

  x[xOffset] = y[yOffset];
}
BUILD_DOUBLE_TEMPLATE(template SD_LIB_EXPORT void NDArray::templatedDoubleAssign,
                      (void *xBuffer,  sd::LongType xOffset,  void *yBuffer,  sd::LongType yOffset),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////
void NDArray::varianceAlongDimension(sd::variance::Ops op, NDArray &target, const bool biasCorrected,
                                     const std::vector<LongType> *dimensions)  {
  if (isS()) THROW_EXCEPTION("NDArray::varianceAlongDimension: you can't use this method on String array!");

  if (!target.isR()) THROW_EXCEPTION("NDArray::varianceAlongDimension: target array must have FLOAT type");

  prepareUse({&target}, {this});

  if (static_cast<size_t>(rankOf()) == dimensions->size() || dimensions->empty())
    NativeOpExecutioner::execSummaryStatsScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                                specialShapeInfo(), nullptr, target.buffer(), target.shapeInfo(),
                                                target.specialBuffer(), target.specialShapeInfo(), biasCorrected);
  else {
    std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
    auto pDims = sd::Environment::getInstance().isCPU() ? copy->data() : nullptr;
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), copy);
    NativeOpExecutioner::execSummaryStats(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                          nullptr, target.buffer(), target.shapeInfo(), target.specialBuffer(),
                                          target.specialShapeInfo(), pDims, dimensions->size(),
                                          packX->platformShapeInfo(), packX->platformOffsets(), biasCorrected);
    delete copy;
    synchronize("NDArray::varianceAlongDimension");
  }

  registerUse({&target}, {this});
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::varianceAlongDimension(sd::variance::Ops op, const bool biasCorrected,
                                        const std::vector<LongType> *dimensions)  {
  if (isS()) THROW_EXCEPTION("NDArray::varianceAlongDimension: you can't use this method on String array!");

  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
  if (copy->size() > 1) std::sort(copy->begin(), copy->end());

  auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(dataType()), false,
                                                  false, getContext()->getWorkspace());
  NDArray result(newShape, true, getContext());

  this->varianceAlongDimension(op, result, biasCorrected, copy);
  delete copy;
  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::varianceAlongDimension(sd::variance::Ops op, const bool biasCorrected,
                                        const std::initializer_list<LongType> *dimensions)  {
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
  auto ret = varianceAlongDimension(op, biasCorrected, copy);
  delete copy;
  return ret;
}

////////////////////////////////////////////////////////////////////////
void NDArray::varianceAlongDimension(sd::variance::Ops op, NDArray &target, const bool biasCorrected,
                                     const std::initializer_list<LongType> *dimensions)  {
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
  varianceAlongDimension(op, target, biasCorrected, copy);
  delete copy;
}

////////////////////////////////////////////////////////////////////////
// This method returns new copy of this NDArray, optionally in different order
NDArray NDArray::dup(const char newOrder, bool forceOriginalBuffer)  {
  if (isEmpty()) return NDArrayFactory::empty(dataType(), getContext());
  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = nonConst->getShapeAsVector();
  if(forceOriginalBuffer) {
    NDArray *result = new NDArray(ordering(), shape, dataType(), getContext());
    DataBuffer *thisBuff = this->getDataBuffer();
    DataBuffer *otherBuff = result->getDataBuffer();
    DataBuffer::memcpy(otherBuff, thisBuff, 0, 0);
    return *result;
  }

  char order = newOrder == 'a' ? ordering() : newOrder;

  int len = isScalar() ? 1 : lengthOf();
  // for now string arrays require special treatment
  if (isS()) {
    if (dataType() == DataType::UTF8) {
      std::vector<std::string> strings(len);

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          strings[i] = this->e<std::string>(i);
        }
      };

      samediff::Threads::parallel_for(func, 0, len, 1);
      return NDArray(shape, strings, dataType(), getContext());
    }
    if (dataType() == DataType::UTF16) {
      std::vector<std::u16string> strings(len);

      auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
          strings[i] = this->e<std::u16string>(i);
        }
      };

      samediff::Threads::parallel_for(func, 0, len, 1);
      return NDArray(shape, strings, dataType(), getContext());
    }

    std::vector<std::u32string> strings(len);
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        strings[i] = this->e<std::u32string>(i);
      }
    };

    samediff::Threads::parallel_for(func, 0,len, 1);
    return NDArray(shape, strings, dataType(), getContext());
  }

  std::vector<sd::LongType> shape3 = isScalar() ? std::vector<sd::LongType>({0}) : shape;
  NDArray result(order, shape3, dataType(), getContext());
  result.assign(this);

  return result;
}

////////////////////////////////////////////////////////////////////////
// This method returns true if two arrays are equal, with custom or default Eps value of 1e-5, false otherwise
bool NDArray::equalsTo(NDArray *other, double eps)  {
  if(isEmpty() && other->isEmpty())
    return true;

  if (dataType() != other->dataType() || (lengthOf() != other->lengthOf() && !isScalar())) {
    return false;
  }

  if(isScalar()) {
    auto thisVal = e<double>(0);
    auto otherVal = other->e<double>(0);
    return sd::math::sd_abs<double,double>(thisVal - otherVal) <= eps;
  }


    // we need to be able to compare [1, len] to [len]
  else if (!shape::equalsSoft(shapeInfo(), other->shapeInfo())) {
    return false;
  }
  if (isS()) {
    // string is special case, we'll compare them one by one, considering both arrays are guaranteed to have the same
    // length

    if (dataType() == DataType::UTF8) {
      for (sd::LongType e = 0; e < this->lengthOf(); e++) {
        auto s1 = this->e<std::string>(e);
        auto s2 = other->e<std::string>(e);

        if (s1 != s2) return false;
      }
    } else if (dataType() == DataType::UTF16) {
      for (sd::LongType e = 0; e < this->lengthOf(); e++) {
        auto s1 = this->e<std::u16string>(e);
        auto s2 = other->e<std::u16string>(e);

        if (s1 != s2) return false;
      }
    } else {
      for (sd::LongType e = 0; e < this->lengthOf(); e++) {
        auto s1 = this->e<std::u32string>(e);
        auto s2 = other->e<std::u32string>(e);

        if (s1 != s2) return false;
      }
    }

    return true;
  } else {
    //NOTE leave max precision here. Crashes can occur otherwise for arrays where data type is of higher
    // regular numeric types
    NDArray tmp(sd::DataType::DOUBLE, getContext());  // scalar = 0

    ExtraArguments extras({0.0, 0.0, eps});
#if defined(SD_CUDA)
    prepareUse({&tmp}, {this, other});
#else
    NDArray::preparePrimaryUse({&tmp}, {this, other});
#endif
    NativeOpExecutioner::execReduce3Scalar(getContext(), reduce3::EqualsWithEps, buffer(), shapeInfo(), specialBuffer(),
                                           specialShapeInfo(), extras.argumentsAsT(DataType::DOUBLE), other->buffer(),
                                           other->shapeInfo(), other->specialBuffer(), other->specialShapeInfo(),
                                           tmp.buffer(), tmp.shapeInfo(), tmp.specialBuffer(), tmp.specialShapeInfo());
#if defined(SD_CUDA)
    NDArray::registerSpecialUse({&tmp}, {this, other});
#else
    NDArray::registerPrimaryUse({&tmp}, {this, other});
#endif
    synchronize("NDArray::equalsTo");

    if (tmp.e<sd::LongType>(0) != 0) {
      return false;
    }

    return true;
  }
}

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
template <>
utf8string NDArray::e(const sd::LongType i)  {
  if (!isS()) THROW_EXCEPTION("This method is available for String arrays only");

  auto rp = getOffset(i);

  syncToHost();
  tickReadHost();

  return *(reinterpret_cast<utf8string *const *>(buffer())[rp]);
}

/////////////////////////////////////////////////////////////////////////
template <typename T>
T NDArray::e( sd::LongType i)  {
  //note: we'd validate this but depending on how a buffer is created
  //(basically if it's passed in as a void buffer) the number of elements
  //can be wrong. This at least happens in calculateOutputShapes2(..) and may
  //or may not happen in other places. Ideally, in the future we'd fix that.
  //sometimes we don't know the number of elements.
  //Due to this we have to omit validation here.
  const auto rp = getOffset(i);
  std::vector<NDArray *> *emptyVec = new std::vector<NDArray *>();
  std::vector<NDArray *> *thisVec =  new std::vector<NDArray *>();
  thisVec->push_back(this);
  NDArray::preparePrimaryUse(*emptyVec, *thisVec);
  NDArray::registerPrimaryUse(*emptyVec, *thisVec);
  if(getDataBuffer() != nullptr)
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), rp), SD_COMMON_TYPES_ALL);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT, NDArray::e(sd::LongType), SD_COMMON_TYPES_ALL);

//////////////////////////////////////////////////////////////////////////
// Returns value from 2D matrix by coordinates/indexes
template <typename T>
T NDArray::e(sd::LongType i, sd::LongType j) {
  if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1]) {
    std::string errorMessage;
    errorMessage += "NDArray::e(i,j): one of input indexes is out of array length or rank!=2 !";
    errorMessage += " Requested indexes: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += ",";
    errorMessage += StringUtils::valueToString<sd::LongType>(j);
    errorMessage += ", array shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += ", array rank: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(rankOf());
    errorMessage += ", array order: ";
    errorMessage += ordering();
    THROW_EXCEPTION(errorMessage.c_str());
  }
  sd::LongType indices[2] = {i, j};
  sd::LongType xOffset;
  COORDS2INDEX(rankOf(), shape::stride(shapeInfo()), indices, xOffset);
  if (static_cast<size_t>(xOffset) >= getDataBuffer()->getNumElements()) {
    std::string errorMessage;
    errorMessage += "NDArray::e: index is out of array length !";
    errorMessage += " Requested index: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += ",";
    errorMessage += StringUtils::valueToString<sd::LongType>(j);
    errorMessage += ", array length: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(lengthOf());
    errorMessage += ", array shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += ", array rank: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(rankOf());
    errorMessage += ", array order: ";
    errorMessage += ordering();
    errorMessage += ", offset: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(xOffset);
    auto shapeInfoString = shape::shapeInfoString(shapeInfo());
    errorMessage += ", shapeInfo: ";
    errorMessage += shapeInfoString;
    errorMessage += "data buffer num elements: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(getDataBuffer()->getNumElements());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  NDArray::preparePrimaryUse({}, {this});
  NDArray::registerPrimaryUse({}, {this});

  if (getDataBuffer() != nullptr)
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), xOffset), SD_COMMON_TYPES_ALL);

  return static_cast<T>(119);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT, NDArray::e( sd::LongType,  sd::LongType),
                                SD_COMMON_TYPES_ALL);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
template <typename T>
T NDArray::e(const sd::LongType i, const sd::LongType j, const sd::LongType k)  {
  if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2]) {
    std::string errorMessage;
    errorMessage += "NDArray::e(i,j,k): one of input indexes is out of array length or rank!=3 !";
    errorMessage += " Requested indexes: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += ", ";
    errorMessage += StringUtils::valueToString<sd::LongType>(j);
    errorMessage += ", ";
    errorMessage += StringUtils::valueToString<sd::LongType>(k);
    errorMessage += ", array shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += ", array rank: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(rankOf());
    errorMessage += ", array order: ";
    errorMessage += ordering();
    errorMessage += ", array length: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  sd::LongType indices[3] = {i,j,k};
  sd::LongType xOffset;
  COORDS2INDEX(this->rankOf(), shape::stride(this->shapeInfo()), indices, xOffset);
  if (static_cast<size_t>(xOffset) >= this->getDataBuffer()->getNumElements()) {
    std::string errorMessage;
    errorMessage += "NDArray::e: index is out of array length !";
    errorMessage += " Requested index: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += ", array length: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(lengthOf());
    errorMessage += ", array shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += ", array rank: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(rankOf());
    errorMessage += ", array order: ";
    errorMessage += ordering();

    THROW_EXCEPTION(errorMessage.c_str());
  }
  NDArray::preparePrimaryUse({}, {this});
  NDArray::registerPrimaryUse({}, {this});

  if(getDataBuffer() != nullptr)
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), xOffset), SD_COMMON_TYPES_ALL);

  return static_cast<T>(119);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT,
                                NDArray::e(const sd::LongType, const sd::LongType, const sd::LongType) ,
                                SD_COMMON_TYPES_ALL);

//////////////////////////////////////////////////////////////////////////
// returns value from 3D tensor by coordinates
template <typename T>
T NDArray::e(const sd::LongType i, const sd::LongType j, const sd::LongType k, const sd::LongType l)  {
  if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
    THROW_EXCEPTION("NDArray::e(i,j,k,l): one of input indexes is out of array length or rank!=4 !");

  sd::LongType indices[4] = {i,j,k,l};
  sd::LongType xOffset;
  COORDS2INDEX(this->rankOf(), shape::stride(this->shapeInfo()), indices, xOffset);
  if (static_cast<size_t>(xOffset) >= this->getDataBuffer()->getNumElements()) {
    std::string errorMessage;
    errorMessage += "NDArray::e: index is out of array length !";
    errorMessage += " Requested index: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(i);
    errorMessage += ", array length: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(lengthOf());
    errorMessage += ", array shape: ";
    errorMessage += ShapeUtils::shapeAsString(this);
    errorMessage += ", array rank: ";
    errorMessage += StringUtils::valueToString<sd::LongType>(rankOf());
    errorMessage += ", array order: ";
    errorMessage += ordering();
    THROW_EXCEPTION(errorMessage.c_str());
  }
  NDArray::preparePrimaryUse({}, {this});
  NDArray::registerPrimaryUse({}, {this});
  if(getDataBuffer() != nullptr)
    BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), return templatedGet<, T>(buffer(), xOffset), SD_COMMON_TYPES_ALL);

  return static_cast<T>(119);
}
BUILD_SINGLE_UNCHAINED_TEMPLATE(template SD_LIB_EXPORT,
                                NDArray::e( sd::LongType,  sd::LongType,  sd::LongType,
                                            sd::LongType),
                                SD_COMMON_TYPES_ALL);

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::e(const sd::LongType i)  {
  const auto offset = getOffset(i);
  NDArray scalar(dataType(), getContext());
  NDArray &dereffed = *this;
  scalar.copyBuffersContinuouslyFrom(dereffed, sizeOfT(), 0, this->offset() + offset);

  return scalar;
}

//////////////////////////////////////////////////////////////////////////
// perform array transformation
void NDArray::applyTransform(sd::transform::FloatOps op, NDArray *target, ExtraArguments *extraParams) {
  if (isS()) THROW_EXCEPTION("NDArray::applyTransform FloatOps: you can't use this method on String array!");

  if (!target->isR()) THROW_EXCEPTION("NDArray::applyTransform FloatOps: target array must have one of FLOAT types");

  NDArray::prepareSpecialUse({target}, {this});
  NativeOpExecutioner::execTransformFloat(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(),
      extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::registerSpecialUse({target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::AnyOps op, NDArray *target, ExtraArguments *extraParams) {
  if (isS()) THROW_EXCEPTION("NDArray::applyTransform AnyOps: you can't use this method on String array!");

  NDArray::prepareSpecialUse({target}, {this});
  NativeOpExecutioner::execTransformAny(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(),
      extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr, false);
  NDArray::registerSpecialUse({target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::SameOps op, NDArray *target, ExtraArguments *extraParams) {
  if (isS()) THROW_EXCEPTION("NDArray::applyTransform SameOps: you can't use this method on String array!");

  if (target->dataType() != dataType())
    THROW_EXCEPTION("NDArray::applyTransform SameOps: target array must have the same data type as original array");

  NDArray::prepareSpecialUse({target}, {this});
  NativeOpExecutioner::execTransformSame(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(),
      extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr, nullptr, nullptr);
  NDArray::registerSpecialUse({target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(transform::StrictOps op, NDArray *target, ExtraArguments *extraParams) {
  if (isS()) THROW_EXCEPTION("NDArray::applyTransform StrictOps: you can't use this method on String array!");

  if (!this->isR() || !target->isR() || (this->dataType() != target->dataType()))
    THROW_EXCEPTION("NDArray::applyTransform StrictOps: both Source and Target array must have same FLOAT type !");

  NDArray::prepareSpecialUse({target}, {this});
  NativeOpExecutioner::execTransformStrict(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(),
      extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::registerSpecialUse({target}, {this});
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyTransform(sd::transform::BoolOps op, NDArray *target, ExtraArguments *extraParams) {
  if (isS()) THROW_EXCEPTION("NDArray::applyTransform BoolOps: you can't use this method on String array!");

  if (!target->isB()) THROW_EXCEPTION("NDArray::applyTransform BoolOps: target array must have one of BOOL types");

  NDArray::prepareSpecialUse({target}, {this});
  NativeOpExecutioner::execTransformBool(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(),
      extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::registerSpecialUse({target}, {this});
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::FloatOps op, void *extraParams)  & {
  if (isS()) THROW_EXCEPTION("NDArray::transform FloatOps: you can't use this method on String array!");

  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = nonConst->getShapeAsVector();
  NDArray result(ordering(), shape, DataTypeUtils::pickFloatingType(dataType()), getContext());

  NDArray::prepareSpecialUse({&result}, {this});
  NativeOpExecutioner::execTransformFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                          result.buffer(), result.shapeInfo(), result.specialBuffer(),
                                          result.specialShapeInfo(), extraParams);
  NDArray::registerSpecialUse({&result}, {this});

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::FloatOps op, void *extraParams) && {
  if (isS()) THROW_EXCEPTION("NDArray::transform SameOps: you can't use this method on String array!");

  NDArray::prepareSpecialUse({this}, {this});
  NativeOpExecutioner::execTransformFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                          buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams);
  NDArray::registerSpecialUse({this}, {this});

  return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::SameOps op, void *extraParams)  & {
  if (isS()) THROW_EXCEPTION("NDArray::transform SameOps: you can't use this method on String array!");

  NDArray result(shapeInfo(), false, getContext());

  NDArray::prepareSpecialUse({&result}, {this});
  NativeOpExecutioner::execTransformSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         result.buffer(), result.shapeInfo(), result.specialBuffer(),
                                         result.specialShapeInfo(), extraParams, nullptr, nullptr);
  NDArray::registerSpecialUse({&result}, {this});

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::SameOps op, void *extraParams) && {
  if (isS()) THROW_EXCEPTION("NDArray::transform SameOps: you can't use this method on String array!");

  NDArray::prepareSpecialUse({this}, {this});
  NativeOpExecutioner::execTransformSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams,
                                         nullptr, nullptr);
  NDArray::registerSpecialUse({this}, {this});

  return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::StrictOps op, void *extraParams)  & {
  if (!this->isR()) THROW_EXCEPTION("Source array must have one of FLOAT types");

  NDArray result(shapeInfo(), false, getContext());

  NDArray::prepareSpecialUse({&result}, {this});
  NativeOpExecutioner::execTransformStrict(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                           result.buffer(), result.shapeInfo(), result.specialBuffer(),
                                           result.specialShapeInfo(), extraParams);
  NDArray::registerSpecialUse({&result}, {this});

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::StrictOps op, void *extraParams) && {
  if (!this->isR()) THROW_EXCEPTION("Source array must have one of FLOAT types");

  NDArray::prepareSpecialUse({this}, {this});
  NativeOpExecutioner::execTransformStrict(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                           buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams);
  NDArray::registerSpecialUse({this}, {this});

  return std::move(*this);
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::BoolOps op, void *extraParams)  & {
  if (isS()) THROW_EXCEPTION("NDArray::transform BoolOps: you can't use this method on String array!");

  auto nonConst = const_cast<NDArray *>(this);
  std::vector<sd::LongType> shape = nonConst->getShapeAsVector();
  NDArray result(ordering(), shape, sd::DataType::BOOL, getContext());

  NDArray::prepareSpecialUse({&result}, {this});
  NativeOpExecutioner::execTransformBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         result.buffer(), result.shapeInfo(), result.specialBuffer(),
                                         result.specialShapeInfo(), extraParams);
  NDArray::registerSpecialUse({&result}, {this});

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::transform(sd::transform::BoolOps op, void *extraParams) && {
  if (isS()) THROW_EXCEPTION("NDArray::transform BoolOps: you can't use this method on String array!");

  NDArray::prepareSpecialUse({this}, {this});
  NativeOpExecutioner::execTransformBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), extraParams);
  NDArray::registerSpecialUse({this}, {this});

  return std::move(*this);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(sd::scalar::Ops op, NDArray *scalar, NDArray *target, ExtraArguments *extraParams) {
  if (scalar->lengthOf() > 1) THROW_EXCEPTION("NDArray::applyScalarArr method: operand is not a scalar!");

  if (target->dataType() != DataTypeUtils::pickPairwiseResultType(shapeInfo(), scalar->shapeInfo()) &&
      !(target->dataType() == dataType() || target->dataType() == scalar->dataType())) {
    std::string errorMessage;
    errorMessage += "NDArray::applyScalarArr method: wrong type of target array !\n";
    errorMessage += "Expected array with type: ";
    errorMessage += DataTypeUtils::asString(DataTypeUtils::pickPairwiseResultType(shapeInfo(), scalar->shapeInfo()));
    errorMessage += " or ";
    errorMessage += DataTypeUtils::asString(dataType());
    errorMessage += " or ";
    errorMessage += DataTypeUtils::asString(scalar->dataType());
    errorMessage += ", but got ";
    errorMessage += DataTypeUtils::asString(target->dataType());

    THROW_EXCEPTION(errorMessage.c_str());

  }



  NDArray::prepareSpecialUse({target}, {this, scalar});
  NativeOpExecutioner::execScalar(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(),
      scalar->specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::registerSpecialUse({target}, {this, scalar});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(sd::scalar::BoolOps op, NDArray *scalar, NDArray *target,
                             ExtraArguments *extraParams)  {
  if (!target->isB()) THROW_EXCEPTION("NDArray::applyScalarArr bool method: target has not bool type!");
  if (dataType() != scalar->dataType()) {
    sd_printf("NDArray::applyScalarArr BoolOps: this dtype: [%i]; scalar dtype: [%i]\n", this->dataType(),
              scalar->dataType());
    THROW_EXCEPTION("NDArray::applyScalarArr bool method: this and scalar arrays must have the same type!");
  }

  NDArray::prepareSpecialUse({target}, {this, scalar});
  NativeOpExecutioner::execScalarBool(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(),
      scalar->specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::registerSpecialUse({target}, {this, scalar});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::applyScalarArr(sd::scalar::IntOps op, NDArray *scalar, NDArray *target,
                             ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyScalarArr IntOps: you can't use this method on String array!");

  if (target->dataType() != this->dataType())
    THROW_EXCEPTION("NDArray::applyScalarArr int method: target has not bool type!");
  if (dataType() != scalar->dataType()) {
    sd_printf("NDArray::applyScalarArr IntOps: this dtype: [%i]; scalar dtype: [%i]\n", this->dataType(),
              scalar->dataType());
    THROW_EXCEPTION("NDArray::applyScalarArr int method: this and scalar arrays must have the same type!");
  }

  NDArray::prepareSpecialUse({target}, {this, scalar});
  NativeOpExecutioner::execScalarInt(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), target->buffer(), target->shapeInfo(),
      target->specialBuffer(), target->specialShapeInfo(), scalar->buffer(), scalar->shapeInfo(), scalar->specialBuffer(),
      scalar->specialShapeInfo(), extraParams != nullptr ? extraParams->argumentsAsT(target->dataType()) : nullptr);
  NDArray::registerSpecialUse({target}, {this, scalar});
}

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::applyScalar(sd::scalar::IntOps op, const T scalar, NDArray *target, ExtraArguments *extraParams)  {
  NDArray scalarArr = NDArrayFactory::create(this->dataType(), scalar, getContext());
  applyScalarArr(op, &scalarArr, target, extraParams);
}

template <>
SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::IntOps op, NDArray *scalar, NDArray *target,
                                        ExtraArguments *extraParams)  {
  THROW_EXCEPTION("NDArray::applyScalar<NDArray*> method: do not use me!");
}
template SD_LIB_EXPORT void NDArray::applyScalar<double>(sd::scalar::IntOps op,  double scalar, NDArray *target,
                                                         ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<float>(sd::scalar::IntOps op,  float scalar, NDArray *target,
                                                        ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<float16>(sd::scalar::IntOps op,  float16 scalar, NDArray *target,
                                                          ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<bfloat16>(sd::scalar::IntOps op,  bfloat16 scalar,
                                                           NDArray *target, ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<sd::LongType>(sd::scalar::IntOps op,  sd::LongType scalar,
                                                               NDArray *target, ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<int>(sd::scalar::IntOps op,  int scalar, NDArray *target,
                                                      ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<int16_t>(sd::scalar::IntOps op,  int16_t scalar, NDArray *target,
                                                          ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<int8_t>(sd::scalar::IntOps op,  int8_t scalar, NDArray *target,
                                                         ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<uint8_t>(sd::scalar::IntOps op,  uint8_t scalar, NDArray *target,
                                                          ExtraArguments *extraParams) ;
template SD_LIB_EXPORT void NDArray::applyScalar<bool>(sd::scalar::IntOps op,  bool scalar, NDArray *target,
                                                       ExtraArguments *extraParams);

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::applyScalar(sd::scalar::Ops op, const T scalar, NDArray *target, ExtraArguments *extraParams) {
  auto scalarArr = NDArrayFactory::create<T>(dataType(), scalar, this->getContext());
  applyScalarArr(op, &scalarArr, target, extraParams);
}
template <>
SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, NDArray *scalar, NDArray *target,
                                        ExtraArguments *extraParams) {
  THROW_EXCEPTION("NDArray::applyScalar<NDArray*> method: do not use me!");
}
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const double scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const float scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const float16 scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const bfloat16 scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const sd::LongType scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const int scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const int16_t scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const int8_t scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const uint8_t scalar, NDArray *target,
                                                 ExtraArguments *extraParams);
template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::Ops op, const bool scalar, NDArray *target,
                                                 ExtraArguments *extraParams);

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::applyScalar(sd::scalar::BoolOps op, const T scalar, NDArray *target, ExtraArguments *extraParams)  {
  NDArray scalarArr = NDArrayFactory::create<T>(dataType(), scalar, getContext());
  applyScalarArr(op, &scalarArr, target, extraParams);
}

#define INSTANTIATE__APPLY_SCALAR_BOOL(T) template SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::BoolOps op, const GET_SECOND(T) scalar, NDArray *target, ExtraArguments *extraParams);
ITERATE_LIST((SD_NUMERIC_TYPES), INSTANTIATE__APPLY_SCALAR_BOOL)

template <>
SD_LIB_EXPORT void NDArray::applyScalar(sd::scalar::BoolOps op, NDArray *scalar, NDArray *target,
                                        ExtraArguments *extraParams)  {
  THROW_EXCEPTION("NDArray::applyScalar<NDArray*> method: do not use me!");
}

////////////////////////////////////////////////////////////////////////
void NDArray::applyIndexReduce(sd::indexreduce::Ops op, NDArray *target, const std::vector<LongType> *dimensions,
                               const ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyIndexReduce: you can't use this method on String array!");

  if (target->dataType() != sd::DataType::INT64 && target->dataType() != sd::DataType::INT32)
    THROW_EXCEPTION("NDArray::applyIndexReduce operations return INT32/INT64");

  void *params =
      extraParams != nullptr ? const_cast<ExtraArguments *>(extraParams)->argumentsAsT(this->dataType()) : nullptr;

  NDArray::prepareSpecialUse({target}, {this});

  if (target->isScalar()) {
    NativeOpExecutioner::execIndexReduceScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                               specialShapeInfo(), params, target->buffer(), target->shapeInfo(),
                                               target->specialBuffer(), target->specialShapeInfo());
  } else {
    std::vector<sd::LongType> *copy = const_cast<std::vector<sd::LongType> *>(dimensions);
    shape::checkDimensions(rankOf(), copy);
    auto pDims = sd::Environment::getInstance().isCPU() ? copy->data() : nullptr;
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(shapeInfo(), copy);
    NativeOpExecutioner::execIndexReduce(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         params, target->buffer(), target->shapeInfo(), target->specialBuffer(),
                                         target->specialShapeInfo(), pDims, copy->size(), packX->platformShapeInfo(),
                                         packX->platformOffsets());
    synchronize("NDArray::applyIndexReduce");
  }

  registerSpecialUse({target}, {this});
}

////////////////////////////////////////////////////////////////////////
// reduce dimensions in this array relying on index operations
NDArray NDArray::applyIndexReduce(sd::indexreduce::Ops op, const std::vector<LongType> *dimensions,
                                  const ExtraArguments *extraParams)  {
  const std::vector<sd::LongType> *copy = dimensions;
  auto newShape = ShapeUtils::evalReduceShapeInfo('c', const_cast<std::vector<LongType> *>(copy), *this,
                                                  DataType::INT64, false, false, getContext()->getWorkspace());
  NDArray result(newShape, true, getContext());

  applyIndexReduce(op, &result, const_cast<std::vector<LongType> *>(copy), extraParams);

  return result;
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 operations to this and other array, return result in new output array
NDArray NDArray::applyReduce3(sd::reduce3::Ops op, NDArray *other, const ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyReduce3 method: you can't use this method on String array!");
  if (dataType() != other->dataType())
    THROW_EXCEPTION("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");
  // check shapes consistency
  if (!isSameShape(other))
    THROW_EXCEPTION("NDArray::applyReduce3 method: the shapes of this and other arrays must be the same !");
  // create shapeInfo for scalar
  auto newShape =
      ShapeBuilders::createScalarShapeInfo(DataTypeUtils::pickFloatingType(dataType()), getContext()->getWorkspace());
  // create output array (scalar)
  NDArray result(newShape, true, getContext());
  RELEASE(newShape, getContext()->getWorkspace());
  // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
  void *params = extraParams != nullptr ? const_cast<ExtraArguments *>(extraParams)->argumentsAsT(dataType()) : nullptr;

  NDArray::prepareSpecialUse({&result}, {this, other});
  NativeOpExecutioner::execReduce3Scalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         params, other->buffer(), other->shapeInfo(), other->specialBuffer(),
                                         other->specialShapeInfo(), result.buffer(), result.shapeInfo(),
                                         result.specialBuffer(), result.specialShapeInfo());
  NDArray::registerSpecialUse({&result}, {this, other});

  return result;
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 (exec) operations to this and other array, return result in new output array
NDArray NDArray::applyReduce3(sd::reduce3::Ops op, NDArray *other, const std::vector<LongType> &dimensions,
                              const ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyReduce3: you can't use this method on String array!");
  if (dataType() != other->dataType())
    THROW_EXCEPTION("NDArray::applyReduce3 method: the types of this and other arrays must be the same !");

  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(dimensions);
  shape::checkDimensions(rankOf(), copy);
  shape::checkDimensions(other->rankOf(), copy);

  auto newShape = ShapeUtils::evalReduceShapeInfo('c', copy, *this, DataTypeUtils::pickFloatingType(dataType()), false,
                                                  false, getContext()->getWorkspace());
  NDArray result(newShape, true, getContext());
  // create temporary dynamic array of extra parameters if array extraParams is empty (==nullptr)
  void *params = extraParams != nullptr ? const_cast<ExtraArguments *>(extraParams)->argumentsAsT(dataType()) : nullptr;

  NDArray::prepareSpecialUse({&result}, {this, other});

  // perform calculations
  if (static_cast<size_t>(rankOf()) == copy->size() && static_cast<size_t>(other->rankOf()) == copy->size()) {
    NativeOpExecutioner::execReduce3Scalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                           params, other->buffer(), other->shapeInfo(), other->specialBuffer(),
                                           other->specialShapeInfo(), result.buffer(), result.shapeInfo(),
                                           result.specialBuffer(), result.specialShapeInfo());
  } else {
    auto pDims = sd::Environment::getInstance().isCPU() ? copy->data() : nullptr;

    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(shapeInfo(), copy);
    auto packY = sd::ConstantTadHelper::getInstance().tadForDimensions(other->shapeInfo(), copy);

    if (!shape::equalsSoft(packX->primaryShapeInfo(), packY->primaryShapeInfo()) ||
        (packX->numberOfTads() != packY->numberOfTads() && packY->numberOfTads() != 1 && packY->numberOfTads() != 1))
      THROW_EXCEPTION("NDArray::applyReduce3 cuda method: arrays tads are inconsistent !");

    NativeOpExecutioner::execReduce3(
        getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, other->buffer(),
        other->shapeInfo(), other->specialBuffer(), other->specialShapeInfo(), result.buffer(), result.shapeInfo(),
        result.specialBuffer(), result.specialShapeInfo(), pDims, copy->size(), packX->platformShapeInfo(),
        packX->platformOffsets(), packY->platformShapeInfo(), packY->platformOffsets());
  }

  registerSpecialUse({&result}, {this, other});

  return result;
}

////////////////////////////////////////////////////////////////////////
// apply reduce3 (execAll) operations to this and other array, return result in new output array
NDArray NDArray::applyAllReduce3(sd::reduce3::Ops op, NDArray *other, const std::vector<LongType> *dimensions,
                                 const ExtraArguments *extraParams)  {
  if (isS()) THROW_EXCEPTION("NDArray::applyAllReduce3: you can't use this method on String array!");
  if (dataType() != other->dataType())
    THROW_EXCEPTION("NDArray::applyAllReduce3 method: the types of this and other arrays must be the same !");

  // be careful, copy array may undergo changes (sort, transformation of negative dimensions to positive, duplicates
  // removing )
  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);
  shape::checkDimensions(rankOf(), copy);
  shape::checkDimensions(other->rankOf(), copy);

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(shapeInfo(), copy);
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(other->shapeInfo(), copy);

  // check tads shapes
  if (!shape::equalsSoft(packX->primaryShapeInfo(), packY->primaryShapeInfo()))
    THROW_EXCEPTION("NDArray::applyAllReduce3 method: the shapes of array tads are different !");

  // set newShape for output array
  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(DataTypeUtils::pickFloatingType(dataType()), 'c',
                                                                     {packX->numberOfTads(), packY->numberOfTads()});

  // create output array
  NDArray result(newShape, true, getContext());

  // create dynamic array of extra parameters if array extraParams is empty (==nullptr)
  void *params = extraParams != nullptr ? const_cast<ExtraArguments *>(extraParams)->argumentsAsT(dataType()) : nullptr;

  auto pDims = sd::Environment::getInstance().isCPU() ? copy->data() : nullptr;

  NDArray::prepareSpecialUse({&result}, {this, other});
  NativeOpExecutioner::execReduce3All(
      getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(), params, other->buffer(),
      other->shapeInfo(), other->specialBuffer(), other->specialShapeInfo(), result.buffer(), result.shapeInfo(),
      result.specialBuffer(), result.specialShapeInfo(), pDims, copy->size(), packX->platformShapeInfo(),
      packX->platformOffsets(), packY->platformShapeInfo(), packY->platformOffsets());
  NDArray::registerSpecialUse({&result}, {this, other});

  return result;
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::FloatOps op, NDArray *target, const std::vector<LongType> *dimensions,
                                   const bool keepDims, const bool checkTargetShape)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceAlongDimension FloatOps: you can't use this method on String array!");
  if (!target->isR())
    THROW_EXCEPTION(
        "NDArray::reduceAlongDimension FloatOps: requires target array to be present and have type form real space!");

  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  if (checkTargetShape) {
    auto newShape =
        ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, false, getContext()->getWorkspace());
    if (!shape::shapeEquals(newShape, target->shapeInfo()))
      THROW_EXCEPTION("NDArray::reduceAlongDimension FloatOps: wrong target shape!");
  }

  NDArray::prepareSpecialUse({target}, {this});

  if (static_cast<size_t>(rankOf()) == copy->size() || copy->empty()) {
    NativeOpExecutioner::execReduceFloatScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                               specialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(),
                                               target->specialBuffer(), target->specialShapeInfo());
  } else {
    const sd::LongType *zShapeInfoH = target->shapeInfo();
    const sd::LongType *zShapeInfoD = target->specialShapeInfo();

    if (rankOf() - dimensions->size() != static_cast<size_t>(target->rankOf())) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
          target->shapeInfo(), copy, target->getContext()->getWorkspace());
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = ShapeUtils::evalDimsForReduceOp(rankOf(), copy);
    NativeOpExecutioner::execReduceFloat(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                         nullptr, target->buffer(), zShapeInfoH, target->specialBuffer(), zShapeInfoD,
                                         dims->data(), dims->size());
  }
  synchronize("NDArray::reduceAlongDimension FloatOps");

  NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::SameOps op, NDArray *target, const std::vector<LongType> *dimensions,
                                   const bool keepDims, const bool checkTargetShape)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceAlongDimension SameOps: you can't use this method on String array!");
  if (target->dataType() != dataType())
    THROW_EXCEPTION(
        "NDArray::reduceAlongDimension SameOps: requires target array to be present and have same dtype as input");
  std::vector<sd::LongType> *copy =  new std::vector<sd::LongType>(*dimensions);
  if (checkTargetShape) {
    auto newShape =
        ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, false, getContext()->getWorkspace());
    if (!shape::shapeEquals(newShape, target->shapeInfo())) {
      std::string errorMessage;
      errorMessage += "NDArray::reduceAlongDimension SameOps: wrong target shape!\n";
      errorMessage += "Expected: "; errorMessage += ShapeUtils::shapeAsString(target->shapeInfo());
      errorMessage += " vs "; errorMessage += ShapeUtils::shapeAsString(newShape);
      THROW_EXCEPTION(errorMessage.c_str());
    }
  }

  NDArray::prepareSpecialUse({target}, {this});

  if (static_cast<size_t>(rankOf()) == copy->size() || copy->empty()) {
    NativeOpExecutioner::execReduceSameScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                              specialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(),
                                              target->specialBuffer(), target->specialShapeInfo());
  } else {
    const sd::LongType *zShapeInfoH = target->shapeInfo();
    const sd::LongType *zShapeInfoD = target->specialShapeInfo();

    if (rankOf() - dimensions->size() != static_cast<size_t>(target->rankOf())) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
          target->shapeInfo(), copy, target->getContext()->getWorkspace());
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = ShapeUtils::evalDimsForReduceOp(rankOf(), copy);
    NativeOpExecutioner::execReduceSame(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                        nullptr, target->buffer(), zShapeInfoH, target->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());
  }
  synchronize("NDArray::reduceAlongDimension SameOps");

  NDArray::registerSpecialUse({target}, {this});

  delete copy;

}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::LongOps op, NDArray *target, const std::vector<LongType> *dimensions,
                                   const bool keepDims, const bool checkTargetShape)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceAlongDimension LongOps: you can't use this method on String array!");
  if (target->dataType() != DataType::INT64)
    THROW_EXCEPTION(
        "NDArray::reduceAlongDimension LongOps: requires target array to be present and have type of INT64");

  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  if (checkTargetShape) {
    auto newShape =
        ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, false, getContext()->getWorkspace());
    if (!shape::shapeEquals(newShape, target->shapeInfo()))
      THROW_EXCEPTION("NDArray::reduceAlongDimension LongOps: wrong target shape!");
  }

  NDArray::prepareSpecialUse({target}, {this});

  if (static_cast<size_t>(rankOf()) == copy->size() || copy->empty()) {
    NativeOpExecutioner::execReduceLongScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                              specialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(),
                                              target->specialBuffer(), target->specialShapeInfo());
  } else {
    const sd::LongType *zShapeInfoH = target->shapeInfo();
    const sd::LongType *zShapeInfoD = target->specialShapeInfo();

    if (rankOf() - dimensions->size() != static_cast<size_t>(target->rankOf())) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
          target->shapeInfo(), copy, target->getContext()->getWorkspace());
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = ShapeUtils::evalDimsForReduceOp(rankOf(), copy);
    NativeOpExecutioner::execReduceLong(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                        nullptr, target->buffer(), zShapeInfoH, target->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());
  }
  synchronize("NDArray::reduceAlongDimension LongOps");

  NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////
// method reduces array by excluding its shapes along axes present in dimensions vector
void NDArray::reduceAlongDimension(sd::reduce::BoolOps op, NDArray *target, const std::vector<LongType> *dimensions,
                                   const bool keepDims, const bool checkTargetShape)  {
  if (isS()) THROW_EXCEPTION("NDArray::reduceAlongDimension BoolOps cuda: you can't use this method on String array!");
  if (!target->isB())
    THROW_EXCEPTION(
        "NDArray::reduceAlongDimension BoolOps cuda: requires target array to be present and have BOOL type!");

  std::vector<sd::LongType> *copy = new std::vector<sd::LongType>(*dimensions);

  if (checkTargetShape) {
    auto newShape =
        ShapeUtils::evalReduceShapeInfo(target->ordering(), copy, *this, keepDims, false, getContext()->getWorkspace());
    if (!shape::shapeEquals(newShape, target->shapeInfo()))
      THROW_EXCEPTION("NDArray::reduceAlongDimension BoolOps cuda: wrong target shape!");
  }

  NDArray::prepareSpecialUse({target}, {this});

  if (static_cast<size_t>(rankOf()) == copy->size() || copy->empty()) {
    NativeOpExecutioner::execReduceBoolScalar(getContext(), op, buffer(), shapeInfo(), specialBuffer(),
                                              specialShapeInfo(), nullptr, target->buffer(), target->shapeInfo(),
                                              target->specialBuffer(), target->specialShapeInfo());
  } else {
    const sd::LongType *zShapeInfoH = target->shapeInfo();
    const sd::LongType *zShapeInfoD = target->specialShapeInfo();

    if (rankOf() - dimensions->size() != static_cast<size_t>(target->rankOf())) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(
          target->shapeInfo(), copy, target->getContext()->getWorkspace());
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> *dims = ShapeUtils::evalDimsForReduceOp(rankOf(), copy);
    NativeOpExecutioner::execReduceBool(getContext(), op, buffer(), shapeInfo(), specialBuffer(), specialShapeInfo(),
                                        nullptr, target->buffer(), zShapeInfoH, target->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());
    delete dims;
  }
  synchronize("NDArray::reduceAlongDimension LongOps");

  NDArray::registerSpecialUse({target}, {this});
}

//////////////////////////////////////////////////////////////////////////


// Definition in NDArray.hXX

//////////////////////////////////////////////////////////////////////////
// This method sets value in 2D matrix to position i, j
template <typename T>
void NDArray::p(const sd::LongType i, const sd::LongType j, const T value) {
  if (rankOf() != 2 || i >= shapeOf()[0] || j >= shapeOf()[1])
    THROW_EXCEPTION("NDArray:pe(i,j, value): one of input indexes is out of array length or rank!=2 !");

  void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
  auto xOffset = i * strideAt(0) + j * strideAt(1);

  NDArray::preparePrimaryUse({this}, {}, true);
  BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), SD_COMMON_TYPES);
  NDArray::registerPrimaryUse({this}, {});
}

#define INSTANTIATE_P_I_J(T) template SD_LIB_EXPORT void NDArray::p(const sd::LongType i, const sd::LongType j, const GET_SECOND(T) value);
ITERATE_LIST((SD_NUMERIC_TYPES),INSTANTIATE_P_I_J);
template SD_LIB_EXPORT void NDArray::p(const sd::LongType i, const sd::LongType j, const bool value);

//////////////////////////////////////////////////////////////////////////
// This method sets value in 3D matrix to position i,j,k
template <typename T>
void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k, const T value) {
  //(*this)(i,j,k) = value;
  if (rankOf() != 3 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2])
    THROW_EXCEPTION("NDArray:pe(i,j,k, value): one of input indexes is out of array length or rank!=3 !");

  void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
  auto xOffset = i * strideAt(0) + j * strideAt(1) + k * strideAt(2);

  NDArray::preparePrimaryUse({this}, {}, true);
  BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), SD_COMMON_TYPES);
  NDArray::registerPrimaryUse({this}, {});
}

template SD_LIB_EXPORT void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k,
                                       const bool value);

#define INSTANTIATE_P_I_J_K(T) template SD_LIB_EXPORT void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k, const GET_SECOND(T) value);
ITERATE_LIST((SD_NUMERIC_TYPES),INSTANTIATE_P_I_J_K);

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k, const sd::LongType l, const T value) {
  //(*this)(i,j,k) = value;
  if (rankOf() != 4 || i >= shapeOf()[0] || j >= shapeOf()[1] || k >= shapeOf()[2] || l >= shapeOf()[3])
    THROW_EXCEPTION("NDArray::p(i,j,k,l, value): one of input indexes is out of array length or rank!=4 !");

  void *p = reinterpret_cast<void *>(const_cast<T *>(&value));
  auto xOffset = i * strideAt(0) + j * strideAt(1) + k * strideAt(2) + l * strideAt(3);

  NDArray::preparePrimaryUse({this}, {}, true);
  BUILD_SINGLE_PARTIAL_SELECTOR(dataType(), templatedSet<, T>(this->buffer(), xOffset, p), SD_COMMON_TYPES);
  NDArray::registerPrimaryUse({this}, {});
}

#define INSTANTIATE_P_I_J_K_L(T) template SD_LIB_EXPORT void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k, const sd::LongType l, const GET_SECOND(T) value);
ITERATE_LIST((SD_NUMERIC_TYPES),INSTANTIATE_P_I_J_K_L);

template SD_LIB_EXPORT void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k,
                                       const sd::LongType l, const bool value);

////////////////////////////////////////////////////////////////////////
void NDArray::p(const sd::LongType i, NDArray *scalar) {
  if (scalar->lengthOf() > 1) {
    THROW_EXCEPTION("NDArray::p method: input array must be scalar!");
  }
  if (i >= _length) {
    std::string errorMessage;
    errorMessage += "NDArray::p(i, NDArray_scalar): input index is out of array length !";
    errorMessage += " Array length: " + std::to_string(_length);
    errorMessage += ", input index: " + std::to_string(i);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  NDArray::preparePrimaryUse({this}, {scalar}, true);
  auto rp = getOffset(i);

  if (scalar->isS()) {
    switch(scalar->dataType()) {
      case DataType::UTF8:
        templatedSetString<std::string, std::string>(buffer(), rp, scalar->buffer());
        break;
      case DataType::UTF16:
        templatedSetString<std::u16string, std::u16string>(buffer(), rp, scalar->buffer());
        break;
      case DataType::UTF32:
        templatedSetString<std::u32string, std::u32string>(buffer(), rp, scalar->buffer());
        break;
      default:
        THROW_EXCEPTION("Unknown string type");
    }
  } else {
    // Modified to use both template parameters and match the three-argument version
    switch(scalar->dataType()) {
      case FLOAT32:
        templatedSet<float, float>(buffer(), rp, scalar->buffer());
        break;
      case DOUBLE:
        templatedSet<double, double>(buffer(), rp, scalar->buffer());
        break;
      case INT32:
        templatedSet<int, int>(buffer(), rp, scalar->buffer());
        break;
      case INT64:
        templatedSet<sd::LongType, sd::LongType>(buffer(), rp, scalar->buffer());
        break;
      case BOOL:
        templatedSet<bool, bool>(buffer(), rp, scalar->buffer());
        break;
      case UINT8:
        templatedSet<uint8_t, uint8_t>(buffer(), rp, scalar->buffer());
        break;
      case UINT16:
        templatedSet<uint16_t, uint16_t>(buffer(), rp, scalar->buffer());
        break;
      case UINT32:
        templatedSet<uint32_t, uint32_t>(buffer(), rp, scalar->buffer());
        break;
      case UINT64:
        templatedSet<uint64_t, uint64_t>(buffer(), rp, scalar->buffer());
        break;

        // Add other numeric types as needed
      default:
        THROW_EXCEPTION("Unsupported numeric type");
    }
  }

  NDArray::registerPrimaryUse({this}, {scalar});
}

template<typename T>
typename std::enable_if<DataTypeUtils::scalarTypesForNDarray<T>::value, void>::type
NDArray::p(const sd::LongType i, const T value) {  // Note: value not NDArray reference
  if (i >= _length) {
    std::string errorMessage = "NDArray::p(i, value): input index is out of array length ! i = "
                               + std::to_string(i) + " length = " + std::to_string(_length);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if(isScalar()) {
    // Direct value assignment, no need for scalar.isS() check
    templatedSet<T, T>(buffer(), 0, dataType(), (void*)&value);
    return;
  }

  sd::LongType coords[1] = {i};
  sd::LongType xOffset;
  COORDS2INDEX(1, shape::stride(shapeInfo()), coords, xOffset);
  // Direct value assignment, no need for scalar.isS() check
  templatedSet<T, T>(buffer(), xOffset, dataType(), (void*)&value);
}

#define INSTANTIATE_P_SINGLE(T) template SD_LIB_EXPORT void NDArray::p(const sd::LongType i,const GET_SECOND(T) value);
ITERATE_LIST((SD_NUMERIC_TYPES),INSTANTIATE_P_SINGLE);
template SD_LIB_EXPORT void NDArray::p(const sd::LongType i,const bool value);

template<typename T>
typename std::enable_if<DataTypeUtils::stringTypesForNDarray<T>::value, void>::type
NDArray::p(const sd::LongType i, const T value) {  // Note: value not NDArray reference
  if (i >= _length) {
    std::string errorMessage = "NDArray::p(i, value): input index is out of array length ! i = "
                               + std::to_string(i) + " length = " + std::to_string(_length);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  sd::LongType coords[1] = {i};
  sd::LongType xOffset;
  COORDS2INDEX(shape::rank(shapeInfo()), shape::stride(shapeInfo()), coords, xOffset);

  // Direct value assignment, no need for scalar.isS() check
  templatedSetString<T,T>(buffer(), xOffset, (void*)&value);
}

#define INSTANTIATE_P_SINGLE_STRING(T) template SD_LIB_EXPORT void NDArray::p(const sd::LongType i,const GET_SECOND(T) value);
ITERATE_LIST((SD_STRING_TYPES),INSTANTIATE_P_SINGLE_STRING);


////////////////////////////////////////////////////////////////////////
void NDArray::p(const sd::LongType i, const sd::LongType j, const sd::LongType k, const sd::LongType l, NDArray *scalar) {
  if (!scalar->isScalar()) THROW_EXCEPTION("NDArray::p method: input array must be scalar!");
  if (i >= _length) {
    std::string errorMessage;
    errorMessage += "NDArray::p(i, NDArray_scalar): input index is out of array length !";
    errorMessage += " i = " + std::to_string(i);
    errorMessage += " j = " + std::to_string(j);
    errorMessage += " k = " + std::to_string(k);
    errorMessage += " l = " + std::to_string(l);
    errorMessage += " length = " + std::to_string(_length);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  sd::LongType coords[4] = {i, j, k, l};
  sd::LongType xOffset;
  COORDS2INDEX(shape::rank(shapeInfo()), shape::stride(shapeInfo()), coords, xOffset);
  NDArray::preparePrimaryUse({this}, {scalar}, true);
  auto rp = xOffset;

  if (scalar->isS()) {
    switch(scalar->dataType()) {
      case DataType::UTF8:
        templatedSetString<std::string, std::string>(buffer(), rp, scalar->buffer());
        break;
      case DataType::UTF16:
        templatedSetString<std::u16string, std::u16string>(buffer(), rp, scalar->buffer());
        break;
      case DataType::UTF32:
        templatedSetString<std::u32string, std::u32string>(buffer(), rp, scalar->buffer());
        break;
      default:
        THROW_EXCEPTION("Unknown string type");
    }
  } else {
    // Modified to use both template parameters and match the three-argument version
    switch(scalar->dataType()) {
      case FLOAT32:
        templatedSet<float, float>(buffer(), rp, scalar->buffer());
        break;
      case DOUBLE:
        templatedSet<double, double>(buffer(), rp, scalar->buffer());
        break;
      case INT32:
        templatedSet<int, int>(buffer(), rp, scalar->buffer());
        break;
      case INT64:
        templatedSet<sd::LongType, sd::LongType>(buffer(), rp, scalar->buffer());
        break;
        // Add other numeric types as needed
      default:
        THROW_EXCEPTION("Unsupported numeric type");
    }
  }
  NDArray::registerPrimaryUse({this}, {scalar});
}
//////////////////////////////////////////////////////////////////////////
void NDArray::addRowVector(NDArray *row, NDArray *target)  {
  if (isS()) THROW_EXCEPTION("NDArray::addRowVector: you can't use this method on String array!");
  if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() ||
      !row->isRowVector() || columns() != row->lengthOf()) {
    std::string errorMessage;
    errorMessage += "NDArray::addRowVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Row is row vector " + std::to_string(row->isRowVector());
    errorMessage += ", Number of columns: " + std::to_string(columns());
    errorMessage += ", Row length: " + std::to_string(row->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (target->dataType() != DataTypeUtils::pickPairwiseResultType(dataType(), row->dataType()) &&
      !(isR() && row->isR() && target->isR()))
    THROW_EXCEPTION("NDArray::addRowVector: wrong type of target array !");

  int dimension = 1;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), dimension);

  NDArray::prepareSpecialUse({target}, {this, row});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), row->buffer(), row->shapeInfo(), row->specialBuffer(),
                                     row->specialShapeInfo(), target->buffer(), target->shapeInfo(),
                                     target->specialBuffer(), target->specialShapeInfo(), nullptr, 1,
                                     packX->platformShapeInfo(), packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({target}, {this, row});
}



//////////////////////////////////////////////////////////////////////////
void NDArray::mulRowVector(NDArray *row, NDArray *target)  {
  if (isS()) THROW_EXCEPTION("NDArray::mulRowVector: you can't use this method on String array!");
  if (rankOf() != 2 || target->rankOf() != 2  || columns() != target->columns() ||
      !row->isRowVector() || columns() != row->columns()) {
    std::string errorMessage;
    errorMessage += "NDArray::mulRowVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Row is row vector " + std::to_string(row->isRowVector());
    errorMessage += ", Number of columns: " + std::to_string(columns());
    errorMessage += ", Row length: " + std::to_string(row->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (target->dataType() != DataTypeUtils::pickPairwiseResultType(dataType(), row->dataType()))
    THROW_EXCEPTION("NDArray::mulRowVector: wrong type of target array !");

  int dimension = 1;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(),dimension);

  NDArray::prepareSpecialUse({target}, {this, row});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Multiply, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), row->buffer(), row->shapeInfo(), row->specialBuffer(),
                                     row->specialShapeInfo(), target->buffer(), target->shapeInfo(),
                                     target->specialBuffer(), target->specialShapeInfo(), nullptr, 1,
                                     packX->platformShapeInfo(), packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({target}, {this, row});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::divRowVector(NDArray *row, NDArray *target)  {
  if (isS()) THROW_EXCEPTION("NDArray::divRowVector: you can't use this method on String array!");
  if (row->isB()) THROW_EXCEPTION("NDArray::divRowVector: you can't divide by bool row!");
  if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() ||
      !row->isRowVector() || columns() != row->columns()) {
    std::string errorMessage;
    errorMessage += "NDArray::divRowVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Row is row vector " + std::to_string(row->isRowVector());
    errorMessage += ", Number of columns: " + std::to_string(columns());
    errorMessage += ", Row length: " + std::to_string(row->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (target->dataType() != DataTypeUtils::pickPairwiseResultType(dataType(), row->dataType()))
    THROW_EXCEPTION("NDArray::divRowVector: wrong type of target array !");

  int dimension = 1;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), dimension);

  NDArray::prepareSpecialUse({target}, {this, row});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Divide, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), row->buffer(), row->shapeInfo(), row->specialBuffer(),
                                     row->specialShapeInfo(), target->buffer(), target->shapeInfo(),
                                     target->specialBuffer(), target->specialShapeInfo(), nullptr, 1,
                                     packX->platformShapeInfo(), packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({target}, {this, row});
}

//////////////////////////////////////////////////////////////////////////
// This method adds given row to all rows in this NDArray, this array becomes affected
void NDArray::addiRowVector(NDArray *row) {
  if (isS()) THROW_EXCEPTION("NDArray::addiRowVector: you can't use this method on String array!");
  if (rankOf() != 2 || !row->isRowVector() || columns() != row->lengthOf()) {
    std::string errorMessage;
    errorMessage += "NDArray::addiRowVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Row is row vector " + std::to_string(row->isRowVector());
    errorMessage += ", Number of columns: " + std::to_string(columns());
    errorMessage += ", Row length: " + std::to_string(row->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  int dimension = 1;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), dimension);

  NDArray::prepareSpecialUse({this}, {row});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), row->buffer(), row->shapeInfo(), row->specialBuffer(),
                                     row->specialShapeInfo(), this->buffer(), this->shapeInfo(), this->specialBuffer(),
                                     this->specialShapeInfo(), nullptr, 1, packX->platformShapeInfo(),
                                     packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({this}, {row});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::addColumnVector(NDArray *column, NDArray *target)  {
  if (isS()) THROW_EXCEPTION("NDArray::addColumnVector: you can't use this method on String array!");
  if (rankOf() != 2 || target->rankOf() != 2 || rows() != target->rows() || columns() != target->columns() ||
      !column->isColumnVector() || rows() != column->lengthOf()) {
    std::string errorMessage;
    errorMessage += "NDArray::addColumnVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Vector is column vector " + std::to_string(column->isColumnVector());
    errorMessage += ", Number of rows: " + std::to_string(rows());
    errorMessage += ", Column length: " + std::to_string(column->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (target->dataType() != DataTypeUtils::pickPairwiseResultType(dataType(), column->dataType()))
    THROW_EXCEPTION("NDArray::addColumnVector: wrong type of target array !");

  int dimension = 0;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), dimension);

  NDArray::prepareSpecialUse({target}, {this, column});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), column->buffer(), column->shapeInfo(), column->specialBuffer(),
                                     column->specialShapeInfo(), target->buffer(), target->shapeInfo(),
                                     target->specialBuffer(), target->specialShapeInfo(), nullptr, 1,
                                     packX->platformShapeInfo(), packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({target}, {this, column});
}

//////////////////////////////////////////////////////////////////////////
// This method adds given column to all columns in this NDArray, this array becomes affected
void NDArray::addiColumnVector(NDArray *column) {
  if (isS()) THROW_EXCEPTION("NDArray::addiColumnVector: you can't use this method on String array!");
  if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf()) {
    std::string errorMessage;
    errorMessage += "NDArray::addiColumnVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Vector is column vector " + std::to_string(column->isColumnVector());
    errorMessage += ", Number of rows: " + std::to_string(rows());
    errorMessage += ", Column length: " + std::to_string(column->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  int dimension = 0;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), dimension);

  NDArray::prepareSpecialUse({this}, {column});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Add, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), column->buffer(), column->shapeInfo(), column->specialBuffer(),
                                     column->specialShapeInfo(), this->buffer(), this->shapeInfo(),
                                     this->specialBuffer(), this->specialShapeInfo(), nullptr, 1,
                                     packX->platformShapeInfo(), packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({this}, {column});
}

//////////////////////////////////////////////////////////////////////////
// This method multiplies each column of this array by given argument-column, this array becomes affected
void NDArray::muliColumnVector(NDArray *column) {
  if (isS()) THROW_EXCEPTION("NDArray::muliColumnVector: you can't use this method on String array!");
  if (rankOf() != 2 || !column->isColumnVector() || rows() != column->lengthOf()) {
    std::string errorMessage;
    errorMessage += "NDArray::muliColumnVector Input rank " + std::to_string(rankOf());
    errorMessage += ", Vector is column vector " + std::to_string(column->isColumnVector());
    errorMessage += ", Number of rows: " + std::to_string(rows());
    errorMessage += ", Column length: " + std::to_string(column->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  int dimension = 0;

  auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(this->shapeInfo(), dimension);

  NDArray::prepareSpecialUse({this}, {column});
  NativeOpExecutioner::execBroadcast(getContext(), sd::broadcast::Ops::Multiply, buffer(), shapeInfo(), specialBuffer(),
                                     specialShapeInfo(), column->buffer(), column->shapeInfo(), column->specialBuffer(),
                                     column->specialShapeInfo(), this->buffer(), this->shapeInfo(),
                                     this->specialBuffer(), this->specialShapeInfo(), nullptr, 1,
                                     packX->platformShapeInfo(), packX->platformOffsets(), nullptr, nullptr);
  NDArray::registerSpecialUse({this}, {column});
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArray::templatedAssign(void *xBuffer, sd::LongType xOffset,  void *yBuffer,
                              sd::LongType yOffset)  {
  if (xBuffer != nullptr && yBuffer != nullptr)
    *(reinterpret_cast<T *>(xBuffer) + xOffset) = *(reinterpret_cast<const T *>(yBuffer) + yOffset);
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT void NDArray::templatedAssign,
                      (void *xBuffer,  sd::LongType xOffset,  void *yBuffer,  sd::LongType yOffset),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
bool NDArray::permutei(sd::LongType *dimensions, const int rank) {
  auto shapeInfo = ShapeUtils::evalPermShapeInfo(dimensions, rank, this, getContext()->getWorkspace(),true);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
  _shapeInfo = constDesc->primary();
  _shapeInfoD = constDesc->special();
  return true;
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::multipleTensorsAlongDimension(const std::vector<LongType> &indices,
                                                 const std::vector<LongType> &dimensions)  {
  ResultSet result;

  if (indices.size() == 0) return result;

  auto pack = ConstantTadHelper::getInstance().tadForDimensions(
      shapeInfo(), const_cast<sd::LongType *>(dimensions.data()), dimensions.size());

  auto tadLength = shape::length(pack->primaryShapeInfo());
  auto numTads = lengthOf() / tadLength;

  for (auto idx : indices) {
    if (idx >= numTads) {
      sd_printf("NDArray::multipleTensorsAlongDimension: index %i is higher then number of TADs: %i\n", idx, numTads);
      THROW_EXCEPTION("Bad index");
    }

    auto newShapeInfoCast = pack->primaryShapeInfo();
    auto array =
        new NDArray(getDataBuffer(), newShapeInfoCast, getContext(), pack->primaryOffsets()[idx] + offset());
    result.push_back(array);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::allTensorsAlongDimension(const std::initializer_list<LongType> &dimensions)  {
  return allTensorsAlongDimension(std::vector<sd::LongType>(dimensions));
}

////////////////////////////////////////////////////////////////////////
ResultSet NDArray::allExamples()  {
  std::vector<sd::LongType> dimensions(rankOf() - 1);
  for (int e = 1; e < rankOf(); e++) dimensions[e - 1] = e;

  return allTensorsAlongDimension(dimensions);
}

////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
NDArray *NDArray::like() {
  NDArray *ret = new  NDArray(shapeInfo(), this->dataType(), false, getContext());
  return ret;
}

////////////////////////////////////////////////////////////////////////
NDArray *NDArray::ulike()  {
  NDArray * ret = new  NDArray(this, false, getContext());
  return ret;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArray::diagonal(const char type)  {
  if (isS()) THROW_EXCEPTION("NDArray::diagonal: you can't use this method on String array!");

  const char order = ordering();
  const int rank = rankOf();
  sd::LongType *outShapeInfo;
  ALLOCATE(outShapeInfo, getContext()->getWorkspace(), 8, sd::LongType);
  outShapeInfo[0] = 2;
  outShapeInfo[5] = 0;

  if (isVector() || isScalar()) {
    outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
    outShapeInfo[6] = 1;
    outShapeInfo[7] = (int)order;
  } else {
    int diagSize = 100000000;
    sd::LongType indices[SD_MAX_RANK];

    for (int i = 0; i < rank; ++i) {
      if (diagSize > shapeOf()[i]) diagSize = shapeOf()[i];
      indices[i] = 1;
    }

    sd::LongType step;
    COORDS2INDEX(shape::rank(shapeInfo()), shape::stride(shapeInfo()), indices, step);
    if (type == 'c') {
      outShapeInfo[1] = diagSize;
      outShapeInfo[2] = 1;
    } else {
      outShapeInfo[1] = 1;
      outShapeInfo[2] = diagSize;
    }
    shape::updateStrides(outShapeInfo, order, false);

    outShapeInfo[3] *= step;
    outShapeInfo[4] *= step;
    outShapeInfo[6] = 0;
  }

  ArrayOptions::setDataType(outShapeInfo, this->dataType());
  auto buff = ConstantShapeHelper::getInstance().bufferForShapeInfo(outShapeInfo);
  NDArray result(_buffer, buff->primary(), getContext(), offset());

  RELEASE(outShapeInfo, getContext()->getWorkspace());

  return result;
}


void NDArray::printAllTensorsAlongDimension(const std::vector<LongType> &dimensions)  {
  auto allTads = allTensorsAlongDimension(dimensions);
  for(int i = 0; i < allTads.size(); i++) {
    sd_printf("TAD: %d\n",i);
    allTads.at(i)->printIndexedBuffer("");
  }

}

//used in gtest printing
void PrintTo(const sd::NDArray *arr, std::ostream *os) {
  *os << &arr;
}

void NDArray::printAllTensorsAlongDimension(const std::initializer_list<LongType> &dimensions)  {
  printAllTensorsAlongDimension(std::vector<sd::LongType>(dimensions));
}
void NDArray::printTensorAlongDimension(sd::LongType index, const std::initializer_list<LongType> &dimensions)  {
  printTensorAlongDimension(index, std::vector<sd::LongType>(dimensions));
}
void NDArray::printTensorAlongDimension(sd::LongType index, const std::vector<LongType> &dimensions)  {
  auto tad = this->multipleTensorsAlongDimension(dimensions, {index});
  tad.at(0)->printIndexedBuffer("");
}
////////////////////////////////////////////////////////////////////////
ResultSet NDArray::allTensorsAlongDimension(const std::vector<LongType> &dimensions)  {
  ResultSet result;
  if (dimensions.size() == 0) {
    return result;
  }
  if (isScalar() && dimensions.size() == 1 && dimensions[0] == 0) {
    auto array = new NDArray(_buffer, this->shapeInfo(), getContext(), offset());
    result.push_back(array);
    sd_debug("NDArray::allTensorsAlongDimension: Dimensions were equal %d with this rank of %d\n", dimensions.back(),
             rankOf());
    return result;
  }

  if (dimensions.back() >= rankOf()) {
    sd_debug("Dimensions failure %d and rank %d\n", dimensions.back(), rankOf());
    THROW_EXCEPTION(
        "NDArray::allTensorsAlongDimension static function: all input dimensions must be smaller than rank of input "
        "array !");
  }

  auto pack = ConstantTadHelper::getInstance().tadForDimensions(
      _shapeInfo, const_cast<sd::LongType *>(dimensions.data()), dimensions.size());
  auto numTads = pack->numberOfTads();
  auto newShapeInfoCast = const_cast<sd::LongType *>(pack->primaryShapeInfo());
//print shape info and dimensions being created
  if(Environment::getInstance().isDebug() && Environment::getInstance().isVerbose())
    pack->print("allTensorsAlongDimension");


  for (sd::LongType idx = 0; idx < numTads; idx++) {
    auto array = new NDArray(_buffer, newShapeInfoCast, getContext(), pack->primaryOffsets()[idx] + offset());
    if(Environment::getInstance().isDebug() && Environment::getInstance().isVerbose())
      sd_printf("TAD %lld has primary offsets at %lld\n",idx, pack->primaryOffsets()[idx]);
    result.push_back(array);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////
// operator returns sub-array with buffer pointing at this->_buffer + certain offset
NDArray& NDArray::operator()(const std::vector<sd::LongType> &idx, const bool keepUnitiesInShape,
                             const bool isStrided)  {
  if (isEmpty()) THROW_EXCEPTION("NDArray::operator(sub-arrays): array is empty !");

  sd::LongType numOfUntiesInSubArrShape = 0;

  sd::LongType *subArrShapeInfo = nullptr;

  if (!keepUnitiesInShape) {
    int n(isStrided ? 3 : 2), first = 0, last = 0;

    // calculate the number of unities in shape
    for (sd::LongType d = 0; d < rankOf(); ++d) {
      if (idx[n * d] != idx[n * d + 1]) {
        first = idx[n * d] >= 0 ? idx[n * d] : idx[n * d] + sizeAt(d) + 1;
        last = idx[n * d + 1] >= 0 ? idx[n * d + 1] : idx[n * d + 1] + sizeAt(d) + 1;
        if (last - first == 1) ++numOfUntiesInSubArrShape;
      }
    }
  }

  ALLOCATE(subArrShapeInfo, getContext()->getWorkspace(), shape::shapeInfoLength(rankOf() - numOfUntiesInSubArrShape),
           sd::LongType);

  sd::LongType offset = -1;
  auto inOrder = shape::order(shapeInfo());
  if(inOrder != 'c' && inOrder != 'f')
    THROW_EXCEPTION("Invalid in order for deriving order for view!");
  shape::calcSubArrShapeInfoAndOffset(idx.data(), shapeInfo(), subArrShapeInfo, offset, keepUnitiesInShape, isStrided,
                                      numOfUntiesInSubArrShape);

  auto newShapeInfo = ConstantShapeHelper::getInstance().createFromExisting(subArrShapeInfo);
  NDArray *result = new NDArray(_buffer, const_cast<sd::LongType *>(newShapeInfo), getContext(), offset + this->offset());
  ShapeDescriptor descriptor(newShapeInfo, false);
  descriptor.validate();

  return *result;
}

////////////////////////////////////////////////////////////////////////
NDArray& NDArray::operator()(const sd::LongType subArrIdx,
                             const std::vector<sd::LongType> &dimsToExclude,
                             bool keepUnitiesInShape)  {
  std::vector<sd::LongType> idxRanges(2 * rankOf());

  const sd::LongType rank = rankOf();
  const sd::LongType subArrRank = static_cast<sd::LongType>(dimsToExclude.size());

  if (subArrRank > rank)
    THROW_EXCEPTION(
        "NDArray::operator(const sd::LongType subArrIdx, const std::vector<sd::LongType>& dimsToExclude, bool "
        "keepUnitiesInShape): static method: dimsToExclude is empty or has size > rank of array !");

  memset(idxRanges.data(), 0, 2 * rank * sizeof(sd::LongType));

  // subArrRank == 0 means whole array, idxRanges should contain zeros only
  if (subArrRank != 0) {
    std::vector<sd::LongType> shapeOfSubArr(subArrRank), indexes(subArrRank);
    for (sd::LongType i = 0; i < subArrRank; ++i) shapeOfSubArr[i] = sizeAt(dimsToExclude[i]);

    INDEX2COORDS(subArrIdx, subArrRank, shape::shapeOf(shapeOfSubArr.data()), indexes.data());
    for (sd::LongType i = 0; i < subArrRank; ++i) {
      sd::LongType currIdx = 2 * dimsToExclude[i];
      idxRanges[currIdx] = indexes[i];
      idxRanges[currIdx + 1] = indexes[i] + 1;
    }
  }

  NDArray *ret =  new NDArray((*this)(idxRanges, keepUnitiesInShape));
  return *ret;
}

////////////////////////////////////////////////////////////////////////
void NDArray::getSubArrShapeAndOffsets(const std::vector<LongType> &dimsToExclude, sd::LongType *subArrShapeInfo,
                                       sd::LongType *subArrOffsets, bool keepUnitiesInShape)  {
  if (isEmpty()) THROW_EXCEPTION("NDArray::getSubArrShapeAndOffsets: array is empty !");

  const sd::LongType rank = rankOf();
  const sd::LongType subArrRank =
      (static_cast<size_t>(rank) == dimsToExclude.size() || keepUnitiesInShape) ? rank : rank - dimsToExclude.size();
  const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(_shapeInfo, dimsToExclude);

  // allocate memory
  ALLOCATE(subArrShapeInfo, getContext()->getWorkspace(), shape::shapeInfoLength(subArrRank), sd::LongType);
  ALLOCATE(subArrOffsets, getContext()->getWorkspace(), numOfSubArrs, sd::LongType);

  shape::calcSubArrsShapeInfoAndOffsets(_shapeInfo, numOfSubArrs, dimsToExclude.size(), dimsToExclude.data(),
                                        subArrShapeInfo, subArrOffsets, keepUnitiesInShape);
}

//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(sd::LongType *shapeInfo) {
  if (shapeInfo != nullptr) {
    auto shapeBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
    _shapeInfoBuffer = shapeBuffer;
    _shapeInfo = shapeBuffer->primary();
    if(!shape::shapeEquals(_shapeInfo, shapeBuffer->primary())) {
      THROW_EXCEPTION("New shape is not reflected in the created descriptor");
    }

    if(!shape::strideEquals(_shapeInfo, shapeBuffer->primary())) {
      THROW_EXCEPTION("New strides are not reflected in the created descriptor");
    }

#ifdef __CUDABLAS__
    _shapeInfoD = shapeBuffer->special();
#endif

    if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
      _length = 0;
    else
      _length = shape::length(_shapeInfo);
  } else {
    _shapeInfoD = _shapeInfo = nullptr;
  }
}

//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(ShapeDescriptor *descriptor) {
  if (descriptor == nullptr) {
    THROW_EXCEPTION("NDArray:setShapeInfo Passed in descriptor can't be null!");
  }

  auto shapeBuffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(descriptor->toShapeInfo());
  _shapeInfoBuffer = shapeBuffer;
  _shapeInfo = shapeBuffer->primary();
  if(!shape::shapeEquals(_shapeInfo, descriptor->toShapeInfo())) {
    THROW_EXCEPTION("New shape is not reflected in the created descriptor");
  }
  if(ArrayOptions::dataType(_shapeInfo) != descriptor->dataType()) {
    THROW_EXCEPTION("New data type is not reflected in the created descriptor");
  }
#ifdef __CUDABLAS__
  _shapeInfoD = shapeBuffer->special();
#endif

  if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
    _length = 0;
  else
    _length = shape::length(_shapeInfo);

}

//////////////////////////////////////////////////////////////////////////
void NDArray::setShapeInfo(const ConstantShapeBuffer *shapeBuffer) {
  _shapeInfoBuffer = const_cast<ConstantShapeBuffer *>(shapeBuffer);
  _shapeInfo = _shapeInfoBuffer->primary();
#ifdef __CUDABLAS__
  _shapeInfoD = _shapeInfoBuffer->special();
#endif

  if (ArrayOptions::arrayType(_shapeInfo) == ArrayType::EMPTY)
    _length = 0;
  else
    _length = shape::length(_shapeInfo);

}

}

