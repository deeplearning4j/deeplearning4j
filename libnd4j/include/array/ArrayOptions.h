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

#ifndef ND4J_ARRAY_OPTIONS_H
#define ND4J_ARRAY_OPTIONS_H

#include <system/common.h>
#include <system/op_boilerplate.h>
#include <array/ArrayType.h>
#include <array/DataType.h>
#include <array/SpaceType.h>
#include <array/SparseType.h>


#include <initializer_list>

#define ARRAY_SPARSE 2
#define ARRAY_COMPRESSED 4
#define ARRAY_EMPTY 8
#define ARRAY_RAGGED 16

#define ARRAY_CSR 32
#define ARRAY_CSC 64
#define ARRAY_COO 128

// complex values
#define ARRAY_COMPLEX 512

// quantized values
#define ARRAY_QUANTIZED 1024

//  16 bit float FP16
#define ARRAY_HALF 4096

//  16 bit bfloat16
#define ARRAY_BHALF 2048

// regular 32 bit float
#define ARRAY_FLOAT 8192

// regular 64 bit float
#define ARRAY_DOUBLE 16384

// 8 bit integer
#define ARRAY_CHAR 32768

// 16 bit integer
#define ARRAY_SHORT 65536

// 32 bit integer
#define ARRAY_INT 131072

// 64 bit integer
#define ARRAY_LONG 262144

// boolean values
#define ARRAY_BOOL 524288

// UTF values
#define ARRAY_UTF8 1048576
#define ARRAY_UTF16 4194304
#define ARRAY_UTF32 16777216

// flag for extras
#define ARRAY_EXTRAS 2097152

// flag for signed/unsigned integers
#define ARRAY_UNSIGNED 8388608

// flag for arrays with padded buffer
#define ARRAY_HAS_PADDED_BUFFER (1 << 25)

namespace sd {
class SD_LIB_EXPORT ArrayOptions {
 private:
  static SD_INLINE SD_HOST_DEVICE sd::LongType &extra(sd::LongType *shape);

 public:
  static SD_INLINE SD_HOST_DEVICE bool isNewFormat(const sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE bool hasPropertyBitSet(const sd::LongType *shapeInfo, int property);
  static SD_INLINE SD_HOST_DEVICE bool togglePropertyBit(sd::LongType *shapeInfo, int property);
  static SD_INLINE SD_HOST_DEVICE void unsetPropertyBit(sd::LongType *shapeInfo, int property);

  static SD_INLINE SD_HOST_DEVICE void setPropertyBit(sd::LongType *shapeInfo, int property);
  static SD_INLINE SD_HOST_DEVICE void setPropertyBits(sd::LongType *shapeInfo, std::initializer_list<int> properties);

  static SD_INLINE SD_HOST_DEVICE bool isSparseArray(sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE bool isUnsigned(sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE sd::DataType dataType(const sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE SpaceType spaceType(sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE SpaceType spaceType(const sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE ArrayType arrayType(sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE ArrayType arrayType(const sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE SparseType sparseType(sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE SparseType sparseType(const sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE bool hasExtraProperties(sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE bool hasPaddedBuffer(const sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE void flagAsPaddedBuffer(sd::LongType *shapeInfo);

  static SD_INLINE SD_HOST_DEVICE void resetDataType(sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE sd::LongType propertyWithoutDataType(const sd::LongType *shapeInfo);
  static SD_INLINE SD_HOST_DEVICE void setDataType(sd::LongType *shapeInfo, const sd::DataType dataType);

  static SD_INLINE SD_HOST_DEVICE void copyDataType(sd::LongType *to, const sd::LongType *from);
};

SD_INLINE SD_HOST_DEVICE sd::LongType &ArrayOptions::extra(sd::LongType *shape) {
  return shape[shape[0] + shape[0] + 1];
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::isNewFormat(const sd::LongType *shapeInfo) {
  return (extra(const_cast<sd::LongType *>(shapeInfo)) != 0);
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::isSparseArray(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_SPARSE);
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::hasExtraProperties(sd::LongType *shapeInfo) {
  return hasPropertyBitSet(shapeInfo, ARRAY_EXTRAS);
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::hasPropertyBitSet(const sd::LongType *shapeInfo, int property) {
  if (!isNewFormat(shapeInfo)) return false;

  return ((extra(const_cast<sd::LongType *>(shapeInfo)) & property) == property);
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::isUnsigned(sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return false;

  return hasPropertyBitSet(shapeInfo, ARRAY_UNSIGNED);
}

SD_INLINE SD_HOST_DEVICE sd::DataType ArrayOptions::dataType(const sd::LongType *shapeInfo) {
  if (hasPropertyBitSet(shapeInfo, ARRAY_FLOAT))
    return sd::DataType::FLOAT32;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_DOUBLE))
    return sd::DataType::DOUBLE;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_HALF))
    return sd::DataType::HALF;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_BHALF))
    return sd::DataType::BFLOAT16;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_BOOL))
    return sd::DataType ::BOOL;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UNSIGNED)) {
    if (hasPropertyBitSet(shapeInfo, ARRAY_CHAR))
      return sd::DataType ::UINT8;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_SHORT))
      return sd::DataType ::UINT16;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_INT))
      return sd::DataType ::UINT32;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_LONG))
      return sd::DataType ::UINT64;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF8))
      return sd::DataType ::UTF8;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF16))
      return sd::DataType ::UTF16;
    else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF32))
      return sd::DataType ::UTF32;
    else {
      // shape::printShapeInfoLinear("Bad unsigned datatype (not)stored in shape",
      // const_cast<sd::LongType*>(shapeInfo));
#ifndef __CUDA_ARCH__
      THROW_EXCEPTION("Bad datatype A");
#endif
    }
  } else if (hasPropertyBitSet(shapeInfo, ARRAY_CHAR))
    return sd::DataType::INT8;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_SHORT))
    return sd::DataType::INT16;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_INT))
    return sd::DataType::INT32;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_LONG))
    return sd::DataType::INT64;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF8))
    return sd::DataType::UTF8;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF16))
    return sd::DataType::UTF16;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_UTF32))
    return sd::DataType::UTF32;
  else {
#ifndef __CUDA_ARCH__
    THROW_EXCEPTION("Bad datatype B");
#endif
  }
}

SD_INLINE SD_HOST_DEVICE SpaceType ArrayOptions::spaceType(const sd::LongType *shapeInfo) {
  return spaceType(const_cast<sd::LongType *>(shapeInfo));
}

SD_INLINE SD_HOST_DEVICE SpaceType ArrayOptions::spaceType(sd::LongType *shapeInfo) {
  if (hasPropertyBitSet(shapeInfo, ARRAY_QUANTIZED)) return SpaceType::QUANTIZED;
  if (hasPropertyBitSet(shapeInfo, ARRAY_COMPLEX))
    return SpaceType::COMPLEX;
  else  // by default we return continuous type here
    return SpaceType::CONTINUOUS;
}

SD_INLINE SD_HOST_DEVICE ArrayType ArrayOptions::arrayType(const sd::LongType *shapeInfo) {
  return arrayType(const_cast<sd::LongType *>(shapeInfo));
}

SD_INLINE SD_HOST_DEVICE ArrayType ArrayOptions::arrayType(sd::LongType *shapeInfo) {
  if (hasPropertyBitSet(shapeInfo, ARRAY_SPARSE))
    return ArrayType::SPARSE;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_COMPRESSED))
    return ArrayType::COMPRESSED;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_EMPTY))
    return ArrayType::EMPTY;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_RAGGED))
    return ArrayType::RAGGED;
  else  // by default we return DENSE type here
    return ArrayType::DENSE;
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::togglePropertyBit(sd::LongType *shapeInfo, int property) {
  extra(shapeInfo) ^= property;

  return hasPropertyBitSet(shapeInfo, property);
}

SD_INLINE SD_HOST_DEVICE void ArrayOptions::setPropertyBit(sd::LongType *shapeInfo, int property) {
  extra(shapeInfo) |= property;
}

SD_INLINE SD_HOST_DEVICE void ArrayOptions::unsetPropertyBit(sd::LongType *shapeInfo, int property) {
  extra(shapeInfo) &= ~property;
}

SD_INLINE SD_HOST_DEVICE SparseType ArrayOptions::sparseType(const sd::LongType *shapeInfo) {
  return sparseType(const_cast<sd::LongType *>(shapeInfo));
}

SD_INLINE SD_HOST_DEVICE SparseType ArrayOptions::sparseType(sd::LongType *shapeInfo) {
#ifndef __CUDA_ARCH__
  if (!isSparseArray(shapeInfo)) THROW_EXCEPTION("Not a sparse array");
#endif

  if (hasPropertyBitSet(shapeInfo, ARRAY_CSC))
    return SparseType::CSC;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_CSR))
    return SparseType::CSR;
  else if (hasPropertyBitSet(shapeInfo, ARRAY_COO))
    return SparseType::COO;
  else
    return SparseType::LIL;
}

SD_INLINE SD_HOST_DEVICE void ArrayOptions::setPropertyBits(sd::LongType *shapeInfo,
                                                            std::initializer_list<int> properties) {
  for (auto v : properties) {
    if (!hasPropertyBitSet(shapeInfo, v)) setPropertyBit(shapeInfo, v);
  }
}

SD_INLINE SD_HOST_DEVICE void ArrayOptions::flagAsPaddedBuffer(sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return;

  return setPropertyBit(shapeInfo, ARRAY_HAS_PADDED_BUFFER);
}

SD_INLINE SD_HOST_DEVICE bool ArrayOptions::hasPaddedBuffer(const sd::LongType *shapeInfo) {
  if (!isNewFormat(shapeInfo)) return false;

  return hasPropertyBitSet(shapeInfo, ARRAY_HAS_PADDED_BUFFER);
}

SD_INLINE SD_HOST_DEVICE sd::LongType ArrayOptions::propertyWithoutDataType(const sd::LongType *shapeInfo) {
  sd::LongType property = shapeInfo[shapeInfo[0] + shapeInfo[0] + 1];
  property = property & (~ARRAY_BOOL);
  property = property & (~ARRAY_HALF);
  property = property & (~ARRAY_BHALF);
  property = property & (~ARRAY_FLOAT);
  property = property & (~ARRAY_DOUBLE);
  property = property & (~ARRAY_INT);
  property = property & (~ARRAY_LONG);
  property = property & (~ARRAY_CHAR);
  property = property & (~ARRAY_SHORT);
  property = property & (~ARRAY_UNSIGNED);
  return property;
}

SD_INLINE SD_HOST_DEVICE void ArrayOptions::resetDataType(sd::LongType *shapeInfo) {
  extra(shapeInfo) = propertyWithoutDataType(shapeInfo);
}

SD_INLINE SD_HOST_DEVICE void ArrayOptions::setDataType(sd::LongType *shapeInfo, const sd::DataType dataType) {
  resetDataType(shapeInfo);
  if (dataType == sd::DataType::UINT8 || dataType == sd::DataType::UINT16 || dataType == sd::DataType::UINT32 ||
      dataType == sd::DataType::UINT64) {
    setPropertyBit(shapeInfo, ARRAY_UNSIGNED);
  }

  switch (dataType) {
    case sd::DataType::BOOL:
      setPropertyBit(shapeInfo, ARRAY_BOOL);
      break;
    case sd::DataType::HALF:
      setPropertyBit(shapeInfo, ARRAY_HALF);
      break;
    case sd::DataType::BFLOAT16:
      setPropertyBit(shapeInfo, ARRAY_BHALF);
      break;
    case sd::DataType::FLOAT32:
      setPropertyBit(shapeInfo, ARRAY_FLOAT);
      break;
    case sd::DataType::DOUBLE:
      setPropertyBit(shapeInfo, ARRAY_DOUBLE);
      break;
    case sd::DataType::INT8:
      setPropertyBit(shapeInfo, ARRAY_CHAR);
      break;
    case sd::DataType::INT16:
      setPropertyBit(shapeInfo, ARRAY_SHORT);
      break;
    case sd::DataType::INT32:
      setPropertyBit(shapeInfo, ARRAY_INT);
      break;
    case sd::DataType::INT64:
      setPropertyBit(shapeInfo, ARRAY_LONG);
      break;
    case sd::DataType::UINT8:
      setPropertyBit(shapeInfo, ARRAY_CHAR);
      break;
    case sd::DataType::UINT16:
      setPropertyBit(shapeInfo, ARRAY_SHORT);
      break;
    case sd::DataType::UINT32:
      setPropertyBit(shapeInfo, ARRAY_INT);
      break;
    case sd::DataType::UINT64:
      setPropertyBit(shapeInfo, ARRAY_LONG);
      break;
    case sd::DataType::UTF8:
      setPropertyBit(shapeInfo, ARRAY_UTF8);
      break;
    case sd::DataType::UTF16:
      setPropertyBit(shapeInfo, ARRAY_UTF16);
      break;
    case sd::DataType::UTF32:
      setPropertyBit(shapeInfo, ARRAY_UTF32);
      break;
    default:
#ifndef __CUDA_ARCH__
      THROW_EXCEPTION("Can't set unknown data type");
#else
      printf("Can't set unknown data type");
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////
SD_INLINE SD_HOST_DEVICE void ArrayOptions::copyDataType(sd::LongType *to, const sd::LongType *from) {
  setDataType(to, dataType(from));
}
}  // namespace sd

#endif  // ND4J_ARRAY_OPTIONS_H :)
