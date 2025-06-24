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
#include <types/float16.h>
#include <system/selective_rendering.h>

namespace sd {
DataType DataTypeUtils::fromInt(int val) { return (DataType)val; }

DataType DataTypeUtils::fromFlatDataType(graph::DType dtype) { return (DataType)dtype; }

int DataTypeUtils::asInt(DataType type) { return static_cast<int>(type); }

/**
 * Check if a triple of data types is enabled for compilation in selective rendering
 */
static bool isCompiledTypeTriple(DataType type1, DataType type2, DataType type3);


SD_INLINE SD_HOST_DEVICE bool DataTypeUtils::isCompiledDataType(DataType dataType) {
  // First check basic validity
  if (!validDataType(dataType)) return false;
  // Use the generated macro from selective_rendering.h
  return SD_IS_SINGLE_TYPE_COMPILED(dataType);

}

SD_INLINE SD_HOST_DEVICE  bool DataTypeUtils::isCompiledTypePair(DataType type1, DataType type2) {
  // First check basic validity
  if (!validDataType(type1) || !validDataType(type2)) return false;
  // Use the generated macro from selective_rendering.h
  return SD_IS_PAIR_TYPE_COMPILED(type1, type2);

}

SD_INLINE bool DataTypeUtils::isCompiledTypeTriple(DataType type1, DataType type2, DataType type3) {
  // First check basic validity
  if (!validDataType(type1) || !validDataType(type2) || !validDataType(type3)) return false;
  // Use the generated macro from selective_rendering.h
  return SD_IS_TRIPLE_TYPE_COMPILED(type1, type2, type3);

}
}  // namespace sd
