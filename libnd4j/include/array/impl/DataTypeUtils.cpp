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

namespace sd {
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
