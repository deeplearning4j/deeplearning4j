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

namespace sd {
namespace graph {
std::pair<int, int> FlatUtils::fromIntPair(IntPair *pair) { return std::pair<int, int>(pair->first(), pair->second()); }

std::pair<sd::LongType, sd::LongType> FlatUtils::fromLongPair(LongPair *pair) {
  return std::pair<sd::LongType, sd::LongType>(pair->first(), pair->second());
}

NDArray *FlatUtils::fromFlatArray(const sd::graph::FlatArray *flatArray) {
  auto rank = static_cast<int>(flatArray->shape()->Get(0));
  auto newShape = new sd::LongType[shape::shapeInfoLength(rank)];
  memcpy(newShape, flatArray->shape()->data(), shape::shapeInfoByteLength(rank));

  auto length = shape::length(newShape);
  auto dtype = DataTypeUtils::fromFlatDataType(flatArray->dtype());

  // empty arrays is special case, nothing to restore here
  if (shape::isEmpty(newShape)) {
    delete[] newShape;
    return NDArrayFactory::empty_(dtype, nullptr);
  }
  // TODO fix UTF16 and UTF32
  if (dtype == UTF8) {
    bool isBe = BitwiseUtils::isBE();
    bool canKeep = (isBe && flatArray->byteOrder() == sd::graph::ByteOrder_BE) ||
                   (!isBe && flatArray->byteOrder() == sd::graph::ByteOrder_LE);

    std::vector<std::string> substrings(length);
    std::vector<sd::LongType> shapeVector(rank);
    for (int e = 0; e < rank; e++) shapeVector[e] = newShape[e + 1];

    auto rawPtr = (void *)flatArray->buffer()->data();
    auto longPtr = reinterpret_cast<sd::LongType *>(rawPtr);
    auto charPtr = reinterpret_cast<char *>(longPtr + length + 1);
    auto offsets = new sd::LongType[length + 1];
#if defined(__NEC__)
    #pragma _NEC novector
#endif
    for (sd::LongType e = 0; e <= length; e++) {
      auto o = longPtr[e];
      // FIXME: BE vs LE on partials
      // auto v = canKeep ?  o : BitwiseUtils::swap_bytes<sd::LongType>(o);
      offsets[e] = o;
    }

    for (sd::LongType e = 0; e < length; e++) {
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

  auto array = new NDArray(newBuffer, newShape, sd::LaunchContext::defaultContext(), true);

  delete[] newShape;
  return array;
}

flatbuffers::Offset<FlatArray> FlatUtils::toFlatArray(flatbuffers::FlatBufferBuilder &builder, NDArray &array) {
  auto byteVector = array.asByteVector();

  auto fBuffer = builder.CreateVector(byteVector);
  auto fShape = builder.CreateVector(array.getShapeInfoAsFlatVector());

  auto bo = static_cast<sd::graph::ByteOrder>(BitwiseUtils::asByteOrder());

  return CreateFlatArray(builder, fShape, fBuffer, static_cast<sd::graph::DType>(array.dataType()), bo);
}
}  // namespace graph
}  // namespace sd
