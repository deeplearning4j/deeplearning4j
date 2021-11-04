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
//#include <templatemath.h>
//#include <helpers/shape.h>
#include <helpers/logger.h>

namespace sd {
class SD_LIB_EXPORT DataTypeUtils {
 public:
  static int asInt(DataType type);
  static DataType fromInt(int dtype);
  static DataType fromFlatDataType(sd::graph::DType dtype);
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
  SD_INLINE static SD_HOST_DEVICE size_t sizeOf(const sd::LongType *shapeInfo);

  SD_INLINE static SD_HOST_DEVICE bool isR(sd::DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isZ(sd::DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isB(sd::DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isU(sd::DataType dataType);

  SD_INLINE static SD_HOST_DEVICE bool isS(sd::DataType dataType);

  SD_INLINE static sd::DataType pickPairwiseResultType(sd::DataType typeX, sd::DataType typeY);

  SD_INLINE static sd::DataType pickPairwiseResultType(const sd::LongType *shapeInfo1, const sd::LongType *shapeInfo2);

  SD_INLINE static sd::DataType pickFloatingType(sd::DataType typeX);

  template <typename T1, typename T2>
  SD_INLINE static std::vector<T2> convertVector(const std::vector<T1> &vector);

  template <typename T>
  SD_INLINE static bool castShapeInfo(const sd::LongType *originalShapeInfo, T *newShapeInfo);

  template <typename T>
  // struct scalarTypesForNDarray { static bool const value = std::is_same<double, T>::value || std::is_same<float,
  // T>::value || std::is_same<int, T>::value || std::is_same<bfloat16, T>::value || std::is_same<float16, T>::value ||
  // std::is_same<long long, T>::value; };
  struct scalarTypesForNDarray {
    static bool const value =
        std::is_same<double, T>::value || std::is_same<float, T>::value || std::is_same<int, T>::value ||
        std::is_same<unsigned int, T>::value || std::is_same<long long, T>::value ||
        std::is_same<unsigned long long, T>::value || std::is_same<long int, T>::value ||
        std::is_same<long unsigned int, T>::value || std::is_same<int8_t, T>::value ||
        std::is_same<uint8_t, T>::value || std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value ||
        std::is_same<bool, T>::value || std::is_same<bfloat16, T>::value || std::is_same<float16, T>::value;
  };

  template <typename T>
  struct scalarTypesForExecution {
    static bool const value = std::is_same<double, T>::value || std::is_same<float, T>::value ||
                              std::is_same<sd::LongType, T>::value || std::is_same<int, T>::value ||
                              std::is_same<bool, T>::value;
  };
};

//////////////////////////////////////////////////////////////////////////
///// IMLEMENTATION OF INLINE METHODS /////
//////////////////////////////////////////////////////////////////////////

SD_INLINE sd::DataType DataTypeUtils::pickFloatingType(sd::DataType typeX) {
  // if proposed dataType is already floating point - return it
  if (isR(typeX)) return typeX;
  return Environment::getInstance().defaultFloatDataType();
}

SD_INLINE bool DataTypeUtils::isR(sd::DataType dataType) {
  return dataType == sd::DataType::FLOAT32 || dataType == sd::DataType::BFLOAT16 || dataType == sd::DataType::HALF ||
         dataType == sd::DataType::DOUBLE;
}

SD_INLINE bool DataTypeUtils::isB(sd::DataType dataType) { return dataType == sd::DataType::BOOL; }

SD_INLINE bool DataTypeUtils::isS(sd::DataType dataType) {
  return dataType == sd::DataType::UTF8 || dataType == sd::DataType::UTF16 || dataType == sd::DataType::UTF32;
}

SD_INLINE bool DataTypeUtils::isZ(sd::DataType dataType) { return !isR(dataType) && !isB(dataType) && !isS(dataType); }

SD_INLINE bool DataTypeUtils::isU(sd::DataType dataType) {
  return dataType == sd::DataType::UINT8 || dataType == sd::DataType::UINT16 || dataType == sd::DataType::UINT32 ||
         dataType == sd::DataType::UINT64;
}

SD_INLINE sd::DataType DataTypeUtils::pickPairwiseResultType(sd::DataType typeX, sd::DataType typeY) {
  // if both dtypes are the same - just return it
  if (typeX == typeY) return typeX;
  auto sd_max = [](sd::DataType typeX, sd::DataType typeY) { return typeX > typeY ? typeX : typeY; };
  auto rX = isR(typeX);
  auto rY = isR(typeY);

  // if X is float - use it
  if (rX && !rY) return typeX;

  // if Y is float - use it
  if (!rX && rY) return typeY;

  // if both data types are float - return biggest one
  if (rX && rY) {
    // if we allow precision boost, then we pick bigger data type
    if (sd::Environment::getInstance().precisionBoostAllowed()) {
      return sd_max(typeX, typeY);
    } else {
      // and we return first operand otherwise
      return typeX;
    }
  }

  // if that's not real type, we apply same rules
  if (!rX && !rY) {
    if (sd::Environment::getInstance().precisionBoostAllowed()) {
      return sd_max(typeX, typeY);
    } else {
      // and we return first operand otherwise
      return typeX;
    }
  }

  return typeX;
}

///////////////////////////////////////////////////////////////////
SD_INLINE sd::DataType DataTypeUtils::pickPairwiseResultType(const sd::LongType *shapeInfo1,
                                                             const sd::LongType *shapeInfo2) {
  return pickPairwiseResultType(ArrayOptions::dataType(shapeInfo1), ArrayOptions::dataType(shapeInfo2));
}

///////////////////////////////////////////////////////////////////
SD_INLINE size_t DataTypeUtils::sizeOf(DataType type) { return sizeOfElement(type); }

///////////////////////////////////////////////////////////////////
SD_INLINE size_t DataTypeUtils::sizeOf(const sd::LongType *shapeInfo) {
  return sizeOfElement(ArrayOptions::dataType(shapeInfo));
}

// returns the smallest finite value of the given type
template <>
SD_INLINE SD_HOST_DEVICE int DataTypeUtils::min<int>() {
  return std::numeric_limits<int>::min();
}

template <>
SD_INLINE SD_HOST_DEVICE int DataTypeUtils::min_positive<int>() {
  return 0;
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t DataTypeUtils::min<uint8_t>() {
  return (uint8_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t DataTypeUtils::min_positive<uint8_t>() {
  return (uint8_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE char DataTypeUtils::min<char>() {
  return std::numeric_limits<char>::min();
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t DataTypeUtils::min<uint16_t>() {
  return (uint16_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t DataTypeUtils::min_positive<uint16_t>() {
  return (uint16_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::min<bool>() {
  return false;
}

template <>
SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::min_positive<bool>() {
  return false;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType DataTypeUtils::min<sd::LongType>() {
  return (sd::LongType)1L;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType DataTypeUtils::min_positive<sd::LongType>() {
  return (sd::LongType)0;
}

template <>
SD_INLINE SD_HOST_DEVICE int8_t DataTypeUtils::min<int8_t>() {
  return (int8_t)-128;
}

template <>
SD_INLINE SD_HOST_DEVICE int8_t DataTypeUtils::min_positive<int8_t>() {
  return (int8_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t DataTypeUtils::min<uint64_t>() {
  return (uint64_t)0L;
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t DataTypeUtils::min_positive<uint64_t>() {
  return (uint64_t)0L;
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t DataTypeUtils::min<uint32_t>() {
  return 0;
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t DataTypeUtils::min<int16_t>() {
  return (int16_t)-32768;
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t DataTypeUtils::min_positive<int16_t>() {
  return (int16_t)0;
}

template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::min<float>() {
  return (float)1.175494e-38;
}

template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::min_positive<float>() {
  return (float)0;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 DataTypeUtils::min<float16>() {
  return (float16)6.1035e-05;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 DataTypeUtils::min_positive<float16>() {
  return (float16)6.1035e-05;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 DataTypeUtils::min<bfloat16>() {
  return bfloat16::min();
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 DataTypeUtils::min_positive<bfloat16>() {
  return 0;
}

template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::min<double>() {
  return (double)2.2250738585072014e-308;
}

template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::min_positive<double>() {
  return (double)2.2250738585072014e-308;
}

///////////////////////////////////////////////////////////////////
// returns the largest finite value of the given type
template <>
SD_INLINE SD_HOST_DEVICE int DataTypeUtils::max<int>() {
  return (int)2147483647;
}

template <>
SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::max<bool>() {
  return true;
}

template <>
SD_INLINE SD_HOST_DEVICE int8_t DataTypeUtils::max<int8_t>() {
  return 127;
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t DataTypeUtils::max<uint8_t>() {
  return (uint8_t)255;
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t DataTypeUtils::max<int16_t>() {
  return 32767;
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t DataTypeUtils::max<uint16_t>() {
  return 65535;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType DataTypeUtils::max<sd::LongType>() {
  return 9223372036854775807LL;
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t DataTypeUtils::max<uint32_t>() {
  return 4294967295;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::UnsignedLong DataTypeUtils::max<sd::UnsignedLong>() {
  return 18446744073709551615LLU;
}

template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::max<float>() {
  return 3.402823e+38;
}

template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::max<double>() {
  return 1.7976931348623157E308;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 DataTypeUtils::max<float16>() {
  return static_cast<float16>(65504.f);
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 DataTypeUtils::max<bfloat16>() {
  return bfloat16::max();
}

template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::infOrMax<float>() {
  return std::numeric_limits<float>::infinity();
}

template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::infOrMax<double>() {
  return std::numeric_limits<double>::infinity();
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T DataTypeUtils::infOrMax() {
  return DataTypeUtils::max<T>();
}

template <>
SD_INLINE SD_HOST_DEVICE float DataTypeUtils::nanOrZero<float>() {
  return std::numeric_limits<float>::quiet_NaN();
}

template <>
SD_INLINE SD_HOST_DEVICE double DataTypeUtils::nanOrZero<double>() {
  return std::numeric_limits<double>::quiet_NaN();
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T DataTypeUtils::nanOrZero() {
  return static_cast<T>(0);
}

SD_INLINE std::string DataTypeUtils::asString(DataType dataType) {
  switch (dataType) {
    case INT8:
      return std::string("INT8");
    case INT16:
      return std::string("INT16");
    case INT32:
      return std::string("INT32");
    case INT64:
      return std::string("INT64");
    case BFLOAT16:
      return std::string("BFLOAT16");
    case FLOAT32:
      return std::string("FLOAT");
    case DOUBLE:
      return std::string("DOUBLE");
    case HALF:
      return std::string("HALF");
    case BOOL:
      return std::string("BOOL");
    case UINT8:
      return std::string("UINT8");
    case UINT16:
      return std::string("UINT16");
    case UINT32:
      return std::string("UINT32");
    case UINT64:
      return std::string("UINT64");
    case UTF8:
      return std::string("UTF8");
    case UTF16:
      return std::string("UTF16");
    case UTF32:
      return std::string("UTF32");
    default:
      throw std::runtime_error("Unknown data type used");
  }
}

template <typename T>
SD_INLINE bool DataTypeUtils::castShapeInfo(const sd::LongType *originalShapeInfo, T *newShapeInfo) {
  auto shapeInfoLength = *originalShapeInfo * 2 + 4;
  for (auto e = 0; e < shapeInfoLength; e++) {
    if (originalShapeInfo[e] < static_cast<sd::LongType>(DataTypeUtils::max<T>())) {
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
  if (std::is_same<T, double>::value)
    return std::numeric_limits<double>::epsilon();
  else if (std::is_same<T, float>::value)
    return std::numeric_limits<float>::epsilon();
  else if (std::is_same<T, float16>::value)
    return 0.00097656;
  else if (std::is_same<T, bfloat16>::value)
    return bfloat16::eps();
  else
    return 0;
}

template <typename T1, typename T2>
SD_INLINE std::vector<T2> DataTypeUtils::convertVector(const std::vector<T1> &vector) {
  std::vector<T2> result(vector.size());
  sd::LongType vecSize = vector.size();
  for (sd::LongType e = 0; e < vecSize; e++) result[e] = static_cast<T2>(vector[e]);

  return result;
}

SD_INLINE SD_HOST_DEVICE size_t DataTypeUtils::sizeOfElement(sd::DataType type) {
  switch (type) {
    case sd::DataType::UINT8:
    case sd::DataType::INT8:
    case sd::DataType::FLOAT8:
    case sd::DataType::QINT8:
    case sd::DataType::BOOL:
      return (size_t)1;

    case sd::DataType::BFLOAT16:
    case sd::DataType::HALF:
    case sd::DataType::INT16:
    case sd::DataType::QINT16:
    case sd::DataType::UINT16:
      return (size_t)2;

    case sd::DataType::UTF8:
    case sd::DataType::UTF16:
    case sd::DataType::UTF32:
    case sd::DataType::INT32:
    case sd::DataType::UINT32:
    case sd::DataType::HALF2:
    case sd::DataType::FLOAT32:
      return (size_t)4;

    case sd::DataType::UINT64:
    case sd::DataType::INT64:
    case sd::DataType::DOUBLE:
      return (size_t)8;

    default: {
      sd_printf("Unknown DataType used: [%i]\n", asInt(type));
#ifndef __CUDA_ARCH__
      throw std::runtime_error("Unknown DataType requested");
#endif
    }
  }
}

template <typename T>
SD_INLINE SD_HOST_DEVICE sd::DataType sd::DataTypeUtils::fromT() {
  if (std::is_same<T, bool>::value) {
    return sd::DataType::BOOL;
  } else if (std::is_same<T, std::string>::value) {
    return sd::DataType::UTF8;
  } else if (std::is_same<T, std::u16string>::value) {
    return sd::DataType::UTF16;
  } else if (std::is_same<T, std::u32string>::value) {
    return sd::DataType::UTF32;
  } else if (std::is_same<T, float>::value) {
    return sd::DataType::FLOAT32;
  } else if (std::is_same<T, float16>::value) {
    return sd::DataType::HALF;
  } else if (std::is_same<T, bfloat16>::value) {
    return sd::DataType::BFLOAT16;
  } else if (std::is_same<T, double>::value) {
    return sd::DataType::DOUBLE;
  } else if (std::is_same<T, int8_t>::value) {
    return sd::DataType::INT8;
  } else if (std::is_same<T, int16_t>::value) {
    return sd::DataType::INT16;
  } else if (std::is_same<T, int>::value) {
    return sd::DataType::INT32;
  } else if (std::is_same<T, sd::LongType>::value) {
    return sd::DataType::INT64;
  } else if (std::is_same<T, uint8_t>::value) {
    return sd::DataType::UINT8;
  } else if (std::is_same<T, uint16_t>::value) {
    return sd::DataType::UINT16;
  } else if (std::is_same<T, uint32_t>::value) {
    return sd::DataType::UINT32;
  } else if (std::is_same<T, sd::UnsignedLong>::value) {
    return sd::DataType::UINT64;
  } else {
    return sd::DataType::INHERIT;
  }
}
}  // namespace sd

#endif  // DATATYPEUTILS_H
