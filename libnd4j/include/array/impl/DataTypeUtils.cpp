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
#include <array/DataType.h>
#include <array/DataTypeUtils.h>
#include <system/Environment.h>
#include <types/float16.h>
#include <system/selective_rendering.h>

#include <cmath>
#include <limits>
#include <string>

namespace sd {
namespace {
// Integer maxima wider than double's significand round up on conversion; step back to the double
// just below so a clamp to the returned bound never yields a value the type cannot hold.
template <typename T>
double finiteMaxAsDouble() {
  const double bound = static_cast<double>(DataTypeUtils::max<T>());
  if (std::numeric_limits<T>::is_integer && bound >= std::ldexp(1.0, std::numeric_limits<T>::digits))
    return std::nextafter(bound, 0.0);
  return bound;
}
}  // namespace

double DataTypeUtils::max(DataType dataType) {
  switch (dataType) {
#ifdef HAS_BOOL
    case BOOL:
      return finiteMaxAsDouble<bool>();
#endif
#ifdef HAS_INT8
    case INT8:
      return finiteMaxAsDouble<int8_t>();
#endif
#ifdef HAS_UINT8
    case UINT8:
      return finiteMaxAsDouble<uint8_t>();
#endif
#ifdef HAS_INT16
    case INT16:
      return finiteMaxAsDouble<int16_t>();
#endif
#ifdef HAS_UINT16
    case UINT16:
      return finiteMaxAsDouble<uint16_t>();
#endif
#ifdef HAS_INT32
    case INT32:
      return finiteMaxAsDouble<int>();
#endif
#ifdef HAS_UINT32
    case UINT32:
      return finiteMaxAsDouble<uint32_t>();
#endif
#ifdef HAS_LONG
    case INT64:
      return finiteMaxAsDouble<LongType>();
#endif
#ifdef HAS_UNSIGNEDLONG
    case UINT64:
      return finiteMaxAsDouble<UnsignedLong>();
#endif
#ifdef HAS_FLOAT16
    case HALF:
      return finiteMaxAsDouble<float16>();
#endif
#ifdef HAS_BFLOAT16
    case BFLOAT16:
      return finiteMaxAsDouble<bfloat16>();
#endif
#ifdef HAS_FLOAT32
    case FLOAT32:
      return finiteMaxAsDouble<float>();
#endif
#ifdef HAS_DOUBLE
    case DOUBLE:
      return finiteMaxAsDouble<double>();
#endif
#ifdef HAS_FLOAT8
    case FLOAT8:
      return finiteMaxAsDouble<float8>();
    case FLOAT8_E5M2:
      return finiteMaxAsDouble<float8_e5m2>();
#endif
    default: {
      const std::string message = "DataTypeUtils::max: data type " + asString(dataType) + " has no numeric range";
      THROW_EXCEPTION(message.c_str());
      return 0.0;
    }
  }
}

DataType DataTypeUtils::fromInt(int val) { return (DataType)val; }

DataType DataTypeUtils::fromFlatDataType(graph::DType dtype) { return (DataType)dtype; }

int DataTypeUtils::asInt(DataType type) { return static_cast<int>(type); }

DataType DataTypeUtils::pickFloatingType(DataType typeX) {
  if (isR(typeX)) return typeX;
  return Environment::getInstance().defaultFloatDataType();
}

DataType DataTypeUtils::pickPairwiseResultType(DataType typeX, DataType typeY) {
  if (typeX == typeY) return typeX;
  auto sd_max = [](DataType typeX, DataType typeY) { return typeX > typeY ? typeX : typeY; };
  auto rX = isR(typeX);
  auto rY = isR(typeY);

  if (rX && !rY) return typeX;
  if (!rX && rY) return typeY;

  // For floating-point pairs, always use the higher-precision type.
  // precisionBoostAllowed() only gates integer type promotion — using a lower-precision
  // float type for a mixed-precision operation produces garbage bit patterns in the
  // output buffer (HALF data written into a FLOAT32 buffer) which are not NaN but yield
  // wrong argmax results.  Java's Shape.pickPairwiseResultType always takes max for floats.
  if (rX && rY) {
    return sd_max(typeX, typeY);
  }

  if (!rX && !rY) {
    if (Environment::getInstance().precisionBoostAllowed()) {
      return sd_max(typeX, typeY);
    } else {
      return typeX;
    }
  }

  return typeX;
}

DataType DataTypeUtils::pickPairwiseResultType(const LongType *shapeInfo1, const LongType *shapeInfo2) {
  return pickPairwiseResultType(ArrayOptions::dataType(shapeInfo1), ArrayOptions::dataType(shapeInfo2));
}

/**
 * Check if a triple of data types is enabled for compilation in selective rendering
 */
static bool isCompiledTypeTriple(DataType type1, DataType type2, DataType type3);

}  // namespace sd
