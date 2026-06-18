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
// Created by raver119 on 22.11.2017.
//
#include <array/ByteOrder.h>
#include <array/ByteOrderUtils.h>
#include <array/DataTypeConversions.h>
#include <array/DataTypeUtils.h>
#include <array/NDArrayFactory.h>
#include <graph/FlatUtils.h>
#include <helpers/BitwiseUtils.h>

namespace sd {
namespace graph {
std::pair<int, int> FlatUtils::fromIntPair(::graph::IntPair *pair) { return std::pair<int, int>(pair->first(), pair->second()); }

std::pair<LongType, LongType> FlatUtils::fromLongPair(::graph::LongPair *pair) {
  return std::pair<LongType, LongType>(pair->first(), pair->second());
}

NDArray *FlatUtils::fromFlatArray(const ::graph::FlatArray *flatArray) {
  auto rank = static_cast<int>(flatArray->shape()->Get(0));
  auto newShape = new LongType[shape::shapeInfoLength(rank) + SD_SHAPE_ALLOC_PADDING];
  memcpy(newShape, flatArray->shape()->data(), shape::shapeInfoByteLength(rank));

  auto length = shape::length(newShape);
  auto dtype = DataTypeUtils::fromFlatDataType(flatArray->dtype());

  // empty arrays is special case, nothing to restore here
  if (shape::isEmptyConst(newShape)) {
    delete[] newShape;
    return NDArrayFactory::empty_(dtype, nullptr);
  }
  // Only UTF8 string arrays are supported. UTF16 and UTF32 lack a vector-based
  // NDArrayFactory::string_ overload, so arrays with those dtypes fall through to
  // the generic buffer copy path below (BUILD_SINGLE_SELECTOR), which handles the
  // raw bytes correctly for non-string numeric types but will not reconstruct
  // string offsets for UTF16/UTF32. If UTF16/UTF32 string array support is needed,
  // add NDArrayFactory::string_(vector<LongType>, vector<u16string>/vector<u32string>)
  // and mirror the UTF8 branch below for those dtypes.
  if (dtype == UTF8) {

    std::vector<std::string> substrings(length);
    std::vector<LongType> shapeVector(rank);
    for (int e = 0; e < rank; e++) shapeVector[e] = newShape[e + 1];

    auto rawPtr = (void *)flatArray->buffer()->data();
    auto longPtr = reinterpret_cast<LongType *>(rawPtr);
    auto charPtr = reinterpret_cast<char *>(longPtr + length + 1);
    auto offsets = new LongType[length + 1 + SD_SHAPE_ALLOC_PADDING]();

    // Determine whether the serialized byte order matches the host byte order.
    // If they differ, the LongType offset values must be byte-swapped.
    bool hostIsBE = BitwiseUtils::isBE();
    ByteOrder serializedOrder = ByteOrderUtils::fromFlatByteOrder(flatArray->byteOrder());
    bool canKeep = (hostIsBE && serializedOrder == BE) || (!hostIsBE && serializedOrder == LE);
    for (LongType e = 0; e <= length; e++) {
      auto o = longPtr[e];
      offsets[e] = canKeep ? o : BitwiseUtils::swap_bytes<LongType>(o);
    }

    for (LongType e = 0; e < length; e++) {
      auto start = offsets[e];
      auto end = offsets[e + 1];
      auto len = end - start;

      auto c = (char *)malloc(len + 1);
      CHECK_ALLOC(c, "Failed temp allocation", len + 1);
      memset(c, '\0', len + 1);
      memcpy(c, charPtr + start, len);

      std::string val(c);
      substrings[e] = val;
      free(c);
    }

    delete[] offsets;
    delete[] newShape;
    // string order always 'c'
    return NDArrayFactory::string_(shapeVector, substrings);
  }

  auto newBuffer = new int8_t[length * DataTypeUtils::sizeOf(dtype)];

  BUILD_SINGLE_SELECTOR(dtype, DataTypeConversions,
                        ::convertType(newBuffer, (void *)flatArray->buffer()->data(), dtype,
                                      ByteOrderUtils::fromFlatByteOrder(flatArray->byteOrder()), length),
                        SD_COMMON_TYPES);

  auto array = new NDArray(newBuffer, newShape, LaunchContext::defaultContext(), true, 0);

  delete[] newShape;
  return array;
}

flatbuffers::Offset<::graph::FlatArray> FlatUtils::toFlatArray(flatbuffers::FlatBufferBuilder &builder, NDArray &array) {
  auto byteVector = array.asByteVector();

  auto fBuffer = builder.CreateVector(byteVector);
  auto vec = array.getShapeInfoAsFlatVector();
  auto fShape = builder.CreateVector(*vec);
  delete vec;
  auto bo = static_cast<::graph::ByteOrder>(BitwiseUtils::asByteOrder());

  return CreateFlatArray(builder, fShape, fBuffer, static_cast<::graph::DType>(array.dataType()), bo);
}
}  // namespace graph
}  // namespace sd
