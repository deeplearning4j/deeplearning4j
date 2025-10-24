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

//
// @author raver119@gmail.com
//

#ifndef DATATYPEUTILS_H
#define DATATYPEUTILS_H

#include <array/ArrayOptions.h>
#include <array/DataType.h>
#include <graph/generated/array_generated.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

#include <helpers/logger.h>

namespace sd {
class SD_LIB_EXPORT DataTypeUtils {
 public:
  static int asInt(DataType type);
  static DataType fromInt(int dtype);
  static DataType fromFlatDataType(::graph::DType dtype);
  SD_INLINE static std::string asString(DataType dataType);

  template <typename T>
  static SD_INLINE SD_HOST_DEVICE DataType fromT();
  static SD_INLINE SD_HOST_DEVICE size_t sizeOfElement(DataType type);

  // returns the smallest finite value of the given type
  template <typename T>
  SD_INLINE static SD_HOST_DEVICE T min();
  // returns 0 or higher for certain numerical types, used in certain applications where min can't return negative
  template <typename T>
  SD_INLINE static SD_HOST_DEVICE T min_positive();

  // returns the largest finite value of the given type
  template <typename T>
  SD_INLINE static SD_HOST_DEVICE T max();

  /**
   * returns inf for float/double and max for everything else
   */
  template <typename T>
  SD_INLINE static SD_HOST_DEVICE T infOrMax();

  template <typename T>
  SD_INLINE static SD_HOST_DEVICE T nanOrZero();

  // returns the difference between 1.0 and the next representable value of the given floating-point type
  template <typename T>
  SD_INLINE static T eps();

  SD_INLINE static SD_HOST_DEVICE size_t sizeOf(DataType type);
  SD_INLINE static SD_HOST_DEVICE size_t sizeOf(const LongType *shapeInfo);

  SD_INLINE static SD_HOST_DEVICE bool isR(DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isZ(DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isB(DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isU(DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isS(DataType dataType);

  SD_INLINE static DataType pickPairwiseResultType(DataType typeX, DataType typeY);

  SD_INLINE static DataType pickPairwiseResultType(const LongType *shapeInfo1, const LongType *shapeInfo2);

  SD_INLINE static DataType pickFloatingType(DataType typeX);

  template <typename T1, typename T2>
  SD_INLINE static std::vector<T2> convertVector(const std::vector<T1> &vector);

  template <typename T>
  SD_INLINE static bool castShapeInfo(const LongType *originalShapeInfo, T *newShapeInfo);

  template <typename T>
  struct scalarTypesForNDarray {
    static bool const value =
        false
#ifdef HAS_DOUBLE
        || std::is_same<double, T>::value
#endif
#ifdef HAS_FLOAT32
        || std::is_same<float, T>::value
#endif
#ifdef HAS_INT32
        || std::is_same<int, T>::value
#endif
#ifdef HAS_UINT32
        || std::is_same<unsigned int, T>::value
#endif
#ifdef HAS_LONG
        || std::is_same<long long, T>::value
#endif
#ifdef HAS_UNSIGNEDLONG
        || std::is_same<unsigned long long, T>::value
        || std::is_same<long unsigned int, T>::value
#endif
#ifdef HAS_LONG
        || std::is_same<long int, T>::value
#endif
#ifdef HAS_INT8
        || std::is_same<int8_t, T>::value
        || std::is_same<signed char, T>::value
        || (std::is_signed<char>::value && std::is_same<char, T>::value)
#endif
#ifdef HAS_UINT8
        || std::is_same<uint8_t, T>::value
        || std::is_same<unsigned char, T>::value
        || (!std::is_signed<char>::value && std::is_same<char, T>::value)
#endif
#ifdef HAS_INT16
        || std::is_same<int16_t, T>::value
        || std::is_same<short int, T>::value
#endif
#ifdef HAS_UINT16
        || std::is_same<uint16_t, T>::value
#endif
#ifdef HAS_BOOL
        || std::is_same<bool, T>::value
#endif
#ifdef HAS_BFLOAT16
        || std::is_same<bfloat16, T>::value
#endif
#ifdef HAS_FLOAT16
        || std::is_same<float16, T>::value
#endif
        ;
  };

  template <typename T>
  struct stringTypesForNDarray {
    static bool const value =
        false
#ifdef HAS_UTF8
        || std::is_same<std::string, T>::value
#endif
#ifdef HAS_UTF16
        || std::is_same<std::u16string, T>::value
#endif
#ifdef HAS_UTF32
        || std::is_same<std::u32string, T>::value
#endif
        ;
  };

  template <typename T>
  struct scalarTypesForExecution {
    static bool const value = 
        false
#ifdef HAS_DOUBLE
        || std::is_same<double, T>::value
#endif
#ifdef HAS_FLOAT32
        || std::is_same<float, T>::value
#endif
#ifdef HAS_LONG
        || std::is_same<LongType, T>::value
#endif
#ifdef HAS_INT32
        || std::is_same<int, T>::value
#endif
#ifdef HAS_BOOL
        || std::is_same<bool, T>::value
#endif
        ;
  };
  static bool validDataType(DataType dataType);
};

//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////

SD_INLINE DataType DataTypeUtils::pickFloatingType(DataType typeX) {
  // if proposed dataType is already floating point - return it
  if (isR(typeX)) return typeX;
  return Environment::getInstance().defaultFloatDataType();
}

SD_INLINE bool DataTypeUtils::isR(DataType dataType) {
  return false
#ifdef HAS_FLOAT32
         || dataType == FLOAT32
#endif
#ifdef HAS_BFLOAT16
         || dataType == BFLOAT16
#endif
#ifdef HAS_FLOAT16
         || dataType == HALF
#endif
#ifdef HAS_DOUBLE
         || dataType == DOUBLE
#endif
         ;
}

SD_INLINE bool DataTypeUtils::isB(DataType dataType) { 
#ifdef HAS_BOOL
  return dataType == BOOL;
#else
  return false;
#endif
}

SD_INLINE bool DataTypeUtils::isS(DataType dataType) {
  return false
#ifdef HAS_UTF8
         || dataType == UTF8
#endif
#ifdef HAS_UTF16
         || dataType == UTF16
#endif
#ifdef HAS_UTF32
         || dataType == UTF32
#endif
         ;
}

SD_INLINE bool DataTypeUtils::isZ(DataType dataType) { 
  return !isR(dataType) && !isB(dataType) && !isS(dataType); 
}

SD_INLINE bool DataTypeUtils::isU(DataType dataType) {
  return false
#ifdef HAS_UINT8
         || dataType == UINT8
#endif
#ifdef HAS_UINT16
         || dataType == UINT16
#endif
#ifdef HAS_UINT32
         || dataType == UINT32
#endif
#ifdef HAS_UNSIGNEDLONG
         || dataType == UINT64
#endif
         ;
}

SD_INLINE DataType DataTypeUtils::pickPairwiseResultType(DataType typeX, DataType typeY) {
  // if both dtypes are the same - just return it
  if (typeX == typeY) return typeX;
  auto sd_max = [](DataType typeX, DataType typeY) { return typeX > typeY ? typeX : typeY; };
  auto rX = isR(typeX);
  auto rY = isR(typeY);

  // if X is float - use it
  if (rX && !rY) return typeX;

  // if Y is float - use it
  if (!rX && rY) return typeY;

  // if both data types are float - return biggest one
  if (rX && rY) {
    // if we allow precision boost, then we pick bigger data type
    if (Environment::getInstance().precisionBoostAllowed()) {
      return sd_max(typeX, typeY);
    } else {
      // and we return first operand otherwise
      return typeX;
    }
  }

  // if that's not real type, we apply same rules
  if (!rX && !rY) {
    if (Environment::getInstance().precisionBoostAllowed()) {
      return sd_max(typeX, typeY);
    } else {
      // and we return first operand otherwise
      return typeX;
    }
  }

  return typeX;
}

///////////////////////////////////////////////////////////////////
SD_INLINE DataType DataTypeUtils::pickPairwiseResultType(const LongType *shapeInfo1,
                                                             const LongType *shapeInfo2) {
  return pickPairwiseResultType(ArrayOptions::dataType(shapeInfo1), ArrayOptions::dataType(shapeInfo2));
}

///////////////////////////////////////////////////////////////////
SD_INLINE size_t DataTypeUtils::sizeOf(DataType type) { return sizeOfElement(type); }

///////////////////////////////////////////////////////////////////
SD_INLINE size_t DataTypeUtils::sizeOf(const LongType *shapeInfo) {
  return sizeOfElement(ArrayOptions::dataType(shapeInfo));
}

// returns the smallest finite value of the given type
#ifdef HAS_INT32
template <>
SD_INLINE SD_HOST_DEVICE int DataTypeUtils::min<int>() {
  return std::numeric_limits<int>::min();
}

template <>
SD_INLINE SD_HOST_DEVICE int DataTypeUtils::min_positive<int>() {
  return 0;
}
#endif

#ifdef HAS_UINT8
template <>
SD_INLINE SD_HOST_DEVICE uint8_t DataTypeUtils::min<uint8_t>() {
  return (uint8_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t DataTypeUtils::min_positive<uint8_t>() {
  return (uint8_t)0;
}
#endif

template <>
SD_INLINE SD_HOST_DEVICE char DataTypeUtils::min<char>() {
  return std::numeric_limits<char>::min();
}

template <>
SD_INLINE SD_HOST_DEVICE char DataTypeUtils::min_positive<char>() {
  return 0;
}

#ifdef HAS_UINT16
template <>
SD_INLINE SD_HOST_DEVICE uint16_t DataTypeUtils::min<uint16_t>() {
  return (uint16_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t DataTypeUtils::min_positive<uint16_t>() {
  return (uint16_t)0;
}
#endif

#ifdef HAS_BOOL
template <>
SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::min<bool>() {
  return false;
}

template <>
SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::min_positive<bool>() {
  return false;
}
#endif

#ifdef HAS_LONG
template <>
SD_INLINE SD_HOST_DEVICE LongType DataTypeUtils::min<LongType>() {
  return std::numeric_limits<LongType>::min();
}

template <>
SD_INLINE SD_HOST_DEVICE LongType DataTypeUtils::min_positive<LongType>() {
  return (LongType)0;
}

template <>
SD_INLINE SD_HOST_DEVICE long DataTypeUtils::min<long>() {
  return std::numeric_limits<long>::min();
}

template <>
SD_INLINE SD_HOST_DEVICE long DataTypeUtils::min_positive<long>() {
  return 0L;
}
#endif

#ifdef HAS_INT8
template <>
SD_INLINE SD_HOST_DEVICE int8_t DataTypeUtils::min<int8_t>() {
  return (int8_t)-128;
}

template <>
SD_INLINE SD_HOST_DEVICE int8_t DataTypeUtils::min_positive<int8_t>() {
  return (int8_t)0;
}
#endif

#ifdef HAS_UNSIGNEDLONG
template <>
SD_INLINE SD_HOST_DEVICE uint64_t DataTypeUtils::min<uint64_t>() {
  return (uint64_t)0L;
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t DataTypeUtils::min_positive<uint64_t>() {
  return (uint64_t)0L;
}
#endif

#ifdef HAS_UINT32
template <>
SD_INLINE SD_HOST_DEVICE uint32_t DataTypeUtils::min<uint32_t>() {
  return 0;
}
#endif

#ifdef HAS_INT16
template <>
SD_INLINE SD_HOST_DEVICE int16_t DataTypeUtils::min<int16_t>() {
  return (int16_t)-32768;
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t DataTypeUtils::min_positive<int16_t>() {
  return (int16_t)0;
}
#endif

#ifdef HAS_FLOAT32
template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::min<float>() {
  return (float)1.175494e-38;
}

template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::min_positive<float>() {
  return (float)0;
}
#endif

#ifdef HAS_FLOAT16
template <>
SD_INLINE SD_HOST_DEVICE float16 DataTypeUtils::min<float16>() {
  return (float16)6.1035e-05;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 DataTypeUtils::min_positive<float16>() {
  return (float16)6.1035e-05;
}
#endif

#ifdef HAS_BFLOAT16
template <>
SD_INLINE SD_HOST_DEVICE bfloat16 DataTypeUtils::min<bfloat16>() {
  return bfloat16::min();
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 DataTypeUtils::min_positive<bfloat16>() {
  return bfloat16::min_positive();
}
#endif

#ifdef HAS_DOUBLE
template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::min<double>() {
  return (double)2.2250738585072014e-308;
}

template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::min_positive<double>() {
  return (double)2.2250738585072014e-308;
}
#endif

///////////////////////////////////////////////////////////////////
// returns the largest finite value of the given type
template <typename T>
SD_INLINE SD_HOST_DEVICE T DataTypeUtils::max() {
  return std::numeric_limits<T>::max();
}

#ifdef HAS_INT32
template <>
SD_INLINE SD_HOST_DEVICE int DataTypeUtils::max<int>() {
  return (int)2147483647;
}
#endif

#ifdef HAS_BOOL
template <>
SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::max<bool>() {
  return true;
}
#endif

#ifdef HAS_INT8
template <>
SD_INLINE SD_HOST_DEVICE int8_t DataTypeUtils::max<int8_t>() {
  return 127;
}
#endif

#ifdef HAS_UINT8
template <>
SD_INLINE SD_HOST_DEVICE uint8_t DataTypeUtils::max<uint8_t>() {
  return (uint8_t)255;
}
#endif

#ifdef HAS_INT16
template <>
SD_INLINE SD_HOST_DEVICE int16_t DataTypeUtils::max<int16_t>() {
  return 32767;
}
#endif

#ifdef HAS_UINT16
template <>
SD_INLINE SD_HOST_DEVICE uint16_t DataTypeUtils::max<uint16_t>() {
  return 65535;
}
#endif

#ifdef HAS_LONG
template <>
SD_INLINE SD_HOST_DEVICE LongType DataTypeUtils::max<LongType>() {
  return 9223372036854775807LL;
}
#endif

#ifdef HAS_UINT32
template <>
SD_INLINE SD_HOST_DEVICE uint32_t DataTypeUtils::max<uint32_t>() {
  return 4294967295;
}
#endif

#ifdef HAS_UNSIGNEDLONG
template <>
SD_INLINE SD_HOST_DEVICE UnsignedLong DataTypeUtils::max<UnsignedLong>() {
  return 18446744073709551615LLU;
}
#endif

#ifdef HAS_FLOAT32
template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::max<float>() {
  return 3.402823e+38;
}
#endif

#ifdef HAS_DOUBLE
template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::max<double>() {
  return 1.7976931348623157E308;
}
#endif

#ifdef HAS_FLOAT16
template <>
SD_INLINE SD_HOST_DEVICE float16 DataTypeUtils::max<float16>() {
  return static_cast<float16>(65504.f);
}
#endif

#ifdef HAS_BFLOAT16
template <>
SD_INLINE SD_HOST_DEVICE bfloat16 DataTypeUtils::max<bfloat16>() {
  return bfloat16::max();
}
#endif

#ifdef HAS_FLOAT32
template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::infOrMax<float>() {
  return std::numeric_limits<float>::infinity();
}
#endif

#ifdef HAS_DOUBLE
template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::infOrMax<double>() {
  return std::numeric_limits<double>::infinity();
}
#endif

template <typename T>
SD_INLINE SD_HOST_DEVICE T DataTypeUtils::infOrMax() {
  return DataTypeUtils::max<T>();
}

#ifdef HAS_FLOAT32
template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::nanOrZero<float>() {
  return std::numeric_limits<float>::quiet_NaN();
}
#endif

#ifdef HAS_DOUBLE
template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::nanOrZero<double>() {
  return std::numeric_limits<double>::quiet_NaN();
}
#endif

template <typename T>
SD_INLINE SD_HOST_DEVICE T DataTypeUtils::nanOrZero() {
  return static_cast<T>(0);
}


SD_INLINE  bool DataTypeUtils::validDataType(DataType dataType) {
  switch (dataType) {
#ifdef HAS_INT8
    case INT8:
#endif
#ifdef HAS_INT16
    case INT16:
#endif
#ifdef HAS_INT32
    case INT32:
#endif
#ifdef HAS_LONG
    case INT64:
#endif
#ifdef HAS_BFLOAT16
    case BFLOAT16:
#endif
#ifdef HAS_FLOAT32
    case FLOAT32:
#endif
#ifdef HAS_DOUBLE
    case DOUBLE:
#endif
#ifdef HAS_FLOAT16
    case HALF:
#endif
#ifdef HAS_BOOL
    case BOOL:
#endif
#ifdef HAS_UINT8
    case UINT8:
#endif
#ifdef HAS_UINT16
    case UINT16:
#endif
#ifdef HAS_UINT32
    case UINT32:
#endif
#ifdef HAS_UNSIGNEDLONG
    case UINT64:
#endif
#ifdef HAS_UTF8
    case UTF8:
#endif
#ifdef HAS_UTF16
    case UTF16:
#endif
#ifdef HAS_UTF32
    case UTF32:
#endif
      return true;
    case UNKNOWN:
    default:
      return false;
  }
}

SD_INLINE std::string DataTypeUtils::asString(DataType dataType) {
  switch (dataType) {
#ifdef HAS_INT8
    case INT8:
      return std::string("INT8");
#endif
#ifdef HAS_INT16
    case INT16:
      return std::string("INT16");
#endif
#ifdef HAS_INT32
    case INT32:
      return std::string("INT32");
#endif
#ifdef HAS_LONG
    case INT64:
      return std::string("INT64");
#endif
#ifdef HAS_BFLOAT16
    case BFLOAT16:
      return std::string("BFLOAT16");
#endif
#ifdef HAS_FLOAT32
    case FLOAT32:
      return std::string("FLOAT");
#endif
#ifdef HAS_DOUBLE
    case DOUBLE:
      return std::string("DOUBLE");
#endif
#ifdef HAS_FLOAT16
    case HALF:
      return std::string("HALF");
#endif
#ifdef HAS_BOOL
    case BOOL:
      return std::string("BOOL");
#endif
#ifdef HAS_UINT8
    case UINT8:
      return std::string("UINT8");
#endif
#ifdef HAS_UINT16
    case UINT16:
      return std::string("UINT16");
#endif
#ifdef HAS_UINT32
    case UINT32:
      return std::string("UINT32");
#endif
#ifdef HAS_UNSIGNEDLONG
    case UINT64:
      return std::string("UINT64");
#endif
#ifdef HAS_UTF8
    case UTF8:
      return std::string("UTF8");
#endif
#ifdef HAS_UTF16
    case UTF16:
      return std::string("UTF16");
#endif
#ifdef HAS_UTF32
    case UTF32:
      return std::string("UTF32");
#endif
    case UNKNOWN:
    default:
      return std::string("UNKNOWN");
  }
}

template <typename T>
SD_INLINE bool DataTypeUtils::castShapeInfo(const LongType *originalShapeInfo, T *newShapeInfo) {
  auto shapeInfoLength = *originalShapeInfo * 2 + 4;
  for (auto e = 0; e < shapeInfoLength; e++) {
    if (originalShapeInfo[e] < static_cast<LongType>(DataTypeUtils::max<T>())) {
      newShapeInfo[e] = static_cast<T>(originalShapeInfo[e]);
    } else
      return false;
  }

  return true;
}



///////////////////////////////////////////////////////////////////
// returns the difference between 1.0 and the next representable value of the given floating-point type
template <typename T>
SD_INLINE SD_HOST_DEVICE T DataTypeUtils::eps() {
#ifdef HAS_DOUBLE
  if constexpr (std::is_same_v<T, double>) {
    return std::numeric_limits<double>::epsilon();
  } else
#endif
#ifdef HAS_FLOAT32
  if constexpr (std::is_same_v<T, float>) {
    return std::numeric_limits<float>::epsilon();
  } else
#endif
#ifdef HAS_FLOAT16
  if constexpr (std::is_same_v<T, float16>) {
    return float16(0.00097656);
  } else
#endif
#ifdef HAS_BFLOAT16
  if constexpr (std::is_same_v<T, bfloat16>) {
    return bfloat16(0.0078125);  // Approximate bfloat16 epsilon
  } else
#endif
  {
    return T(0);
  }
}

template <typename T1, typename T2>
SD_INLINE std::vector<T2> DataTypeUtils::convertVector(const std::vector<T1> &vector) {
  std::vector<T2> result(vector.size());
  LongType vecSize = vector.size();
  for (LongType e = 0; e < vecSize; e++) result[e] = static_cast<T2>(vector[e]);

  return result;
}

SD_INLINE SD_HOST_DEVICE size_t DataTypeUtils::sizeOfElement(DataType type) {
  switch (type) {
#ifdef HAS_UINT8
    case UINT8:
#endif
#ifdef HAS_INT8
    case INT8:
#endif
#ifdef HAS_FLOAT8
    case FLOAT8:
#endif
#ifdef HAS_QINT8
    case QINT8:
#endif
#ifdef HAS_BOOL
    case BOOL:
#endif
#if defined(HAS_UINT8) || defined(HAS_INT8) || defined(HAS_FLOAT8) || defined(HAS_QINT8) || defined(HAS_BOOL)
      return (size_t)1;
#endif

#ifdef HAS_BFLOAT16
    case BFLOAT16:
#endif
#ifdef HAS_FLOAT16
    case HALF:
#endif
#ifdef HAS_INT16
    case INT16:
#endif
#ifdef HAS_QINT16
    case QINT16:
#endif
#ifdef HAS_UINT16
    case UINT16:
#endif
#if defined(HAS_BFLOAT16) || defined(HAS_FLOAT16) || defined(HAS_INT16) || defined(HAS_QINT16) || defined(HAS_UINT16)
      return (size_t)2;
#endif

#ifdef HAS_UTF8
    case UTF8:
#endif
#ifdef HAS_UTF16
    case UTF16:
#endif
#ifdef HAS_UTF32
    case UTF32:
#endif
#ifdef HAS_INT32
    case INT32:
#endif
#ifdef HAS_UINT32
    case UINT32:
#endif
#ifdef HAS_HALF2
    case HALF2:
#endif
#ifdef HAS_FLOAT32
    case FLOAT32:
#endif
#if defined(HAS_UTF8) || defined(HAS_UTF16) || defined(HAS_UTF32) || defined(HAS_INT32) || defined(HAS_UINT32) || defined(HAS_HALF2) || defined(HAS_FLOAT32)
      return (size_t)4;
#endif

#ifdef HAS_UNSIGNEDLONG
    case UINT64:
#endif
#ifdef HAS_LONG
    case INT64:
#endif
#ifdef HAS_DOUBLE
    case DOUBLE:
#endif
#if defined(HAS_UNSIGNEDLONG) || defined(HAS_LONG) || defined(HAS_DOUBLE)
      return (size_t)8;
#endif

    default: {
      sd_printf("Unknown DataType used: [%i]\n", asInt(type));
#ifndef __CUDA_ARCH__
      std::string errorMessage;
      errorMessage += "Unknown data type requested DataTypeUtils:";
      errorMessage += asString(type);
      THROW_EXCEPTION(errorMessage.c_str());
      return -1;
#endif
    }
  }
}

template <typename T>
SD_INLINE SD_HOST_DEVICE DataType DataTypeUtils::fromT() {
#ifdef HAS_BOOL
  if (std::is_same<T, bool>::value) {
    return BOOL;
  } else
#endif
#ifdef HAS_UTF8
  if (std::is_same<T, std::string>::value) {
    return UTF8;
  } else
#endif
#ifdef HAS_UTF16
  if (std::is_same<T, std::u16string>::value) {
    return UTF16;
  } else
#endif
#ifdef HAS_UTF32
  if (std::is_same<T, std::u32string>::value) {
    return UTF32;
  } else
#endif
#ifdef HAS_FLOAT32
  if (std::is_same<T, float>::value) {
    return FLOAT32;
  } else
#endif
#ifdef HAS_FLOAT16
  if (std::is_same<T, float16>::value) {
    return HALF;
  } else
#endif
#ifdef HAS_BFLOAT16
  if (std::is_same<T, bfloat16>::value) {
    return BFLOAT16;
  } else
#endif
#ifdef HAS_DOUBLE
  if (std::is_same<T, double>::value) {
    return DOUBLE;
  } else
#endif
#ifdef HAS_INT8
  if (std::is_same<T, int8_t>::value) {
    return INT8;
  } else
#endif
#ifdef HAS_INT16
  if (std::is_same<T, int16_t>::value) {
    return INT16;
  } else
  if (std::is_same<T, short int>::value) {
    return INT16;
  } else
#endif
#ifdef HAS_INT32
  if (std::is_same<T, int>::value) {
    return INT32;
  } else
#endif
#ifdef HAS_LONG
  if (std::is_same<T, LongType>::value) {
    return INT64;
  } else
#endif
#ifdef HAS_UINT8
  if (std::is_same<T, uint8_t>::value) {
    return UINT8;
  } else
#endif
#ifdef HAS_UINT16
  if (std::is_same<T, uint16_t>::value) {
    return UINT16;
  } else
#endif
#ifdef HAS_UINT32
  if (std::is_same<T, uint32_t>::value) {
    return UINT32;
  } else
#endif
#ifdef HAS_UNSIGNEDLONG
  if (std::is_same<T, UnsignedLong>::value) {
    return UINT64;
  } else
#endif
  {
    return INHERIT;
  }
}
}  // namespace sd

#endif  // DATATYPEUTILS_H