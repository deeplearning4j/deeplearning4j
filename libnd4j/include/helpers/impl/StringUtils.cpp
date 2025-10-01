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
// Created by raver119 on 20/04/18.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//
#include <exceptions/datatype_exception.h>
#include <helpers/BitwiseUtils.h>
#include <helpers/StringUtils.h>

#include <bitset>

#include "execution/Threads.h"
#include "helpers/ShapeUtils.h"

namespace sd {

void StringUtils::setValueForDifferentDataType(NDArray* arr, LongType idx, NDArray* input, DataType zType) {
  switch(zType) {
#if HAS_UTF8
    case UTF8: {
      switch(input->dataType()) {
        case UTF8:
          arr->p<std::string>(idx, input->e<std::string>(idx));
          break;
        case UTF16:
          arr->p<std::string>(idx, std::string(input->e<std::u16string>(idx).begin(), input->e<std::u16string>(idx).end()));
          break;
        case UTF32:
          arr->p<std::string>(idx, std::string(input->e<std::u32string>(idx).begin(), input->e<std::u32string>(idx).end()));
          break;
        default:
         THROW_EXCEPTION("Unsupported DataType for source string.");
      }
      break;
    }
#endif
#if HAS_UTF16
    case UTF16: {
      switch(input->dataType()) {
        case UTF8:
          arr->p<std::u16string>(idx, std::u16string(input->e<std::string>(idx).begin(), input->e<std::string>(idx).end()));
          break;
        case UTF16:
          arr->p<std::u16string>(idx, input->e<std::u16string>(idx));
          break;
        case UTF32:
          arr->p<std::u16string>(idx, std::u16string(input->e<std::u32string>(idx).begin(), input->e<std::u32string>(idx).end()));
          break;
        default:
          THROW_EXCEPTION("Unsupported DataType for source string.");
      }
      break;
    }
#endif
#if HAS_UTF32
    case UTF32: {
      switch(input->dataType()) {
        case UTF8:
          arr->p<std::u32string>(idx, std::u32string(input->e<std::string>(idx).begin(), input->e<std::string>(idx).end()));
          break;
        case UTF16:
          arr->p<std::u32string>(idx, std::u32string(input->e<std::u16string>(idx).begin(), input->e<std::u16string>(idx).end()));
          break;
        case UTF32:
          arr->p<std::u32string>(idx, input->e<std::u32string>(idx));
          break;
        default:
          THROW_EXCEPTION("Unsupported DataType for source string.");
      }
      break;
    }
#endif
    default:
      THROW_EXCEPTION("Unsupported DataType for destination string.");
  }
}

void StringUtils::broadcastStringAssign(NDArray* x, NDArray* z) {
  if (!x->isBroadcastableTo(*z)) {
    THROW_EXCEPTION("Shapes of x and z are not broadcastable.");
  }

  auto zType = z->dataType();
  auto xCasted = x->cast(zType);

  std::vector<LongType> zeroVec = {0};
  std::vector<LongType> *restDims = ShapeUtils::evalDimsToExclude(x->rankOf(), 1, zeroVec.data());

  auto xTensors = xCasted->allTensorsAlongDimension(*restDims);
  auto zTensors = z->allTensorsAlongDimension(*restDims);

  delete restDims;

  if (xCasted->isScalar()) {
    for (int e = 0; e < zTensors.size(); e++) {
      for (int f = 0; f < zTensors.at(e)->lengthOf(); f++) {
        setValueForDifferentDataType(zTensors.at(e), f, xCasted, zType);
      }
    }
  } else {
    for (int e = 0; e < xTensors.size(); e++) {
      auto tensor = xTensors.at(e);
      for (int f = 0; f < tensor->lengthOf(); f++) {
        setValueForDifferentDataType(zTensors.at(e), f, tensor, zType);
      }
    }
  }
}

template <typename T>
void StringUtils::convertStringsForDifferentDataType(NDArray* sourceArray, NDArray* targetArray) {
  if (!sourceArray->isS() || !targetArray->isS()) THROW_EXCEPTION("Source or target array is not a string array!");

  int numStrings = sourceArray->isScalar() ? 1 : sourceArray->lengthOf();

  auto inData = sourceArray->bufferAsT<int8_t>() + ShapeUtils::stringBufferHeaderRequirements(sourceArray->lengthOf());
  auto outData = targetArray->bufferAsT<int8_t>() + ShapeUtils::stringBufferHeaderRequirements(targetArray->lengthOf());

  const auto nInputoffsets = sourceArray->bufferAsT<LongType>();
  const auto nOutputoffsets = targetArray->bufferAsT<LongType>();

  for (int e = 0; e < numStrings; e++) {
    auto idata = inData + nInputoffsets[e];
    auto cdata = outData + nOutputoffsets[e];

    auto start = nInputoffsets[e];
    auto end = nInputoffsets[e + 1];

    // Convert based on target type (using UTF conversions)
    if (DataTypeUtils::fromT<T>() == UTF16) {
      if (sourceArray->dataType() == UTF8) {
        unicode::utf8to16(idata, cdata, end);
      } else if(sourceArray->dataType() == UTF32) {
        unicode::utf32to16(idata, cdata, (end / sizeof(char32_t)));
      }
    } else if (DataTypeUtils::fromT<T>() == UTF32) {
      if (sourceArray->dataType() == UTF8) {
        unicode::utf8to32(idata, cdata, end);
      } else if(sourceArray->dataType() == UTF16) {
        unicode::utf16to32(idata, cdata, (end / sizeof(char16_t)));
      }
    } else {
      if (sourceArray->dataType() == UTF16) {
        unicode::utf16to8(idata, cdata, (end / sizeof(char16_t)));
      } else if(sourceArray->dataType() == UTF32) {
        unicode::utf32to8(idata, cdata, (end / sizeof(char32_t)));
      }
    }
  }
}

#define DEFINE_CONVERT(T) template void StringUtils::convertStringsForDifferentDataType<GET_SECOND(T)>(NDArray* sourceArray, NDArray* targetArray);
ITERATE_LIST((SD_STRING_TYPES),DEFINE_CONVERT)


template <typename T>
std::vector<LongType> StringUtils::calculateOffsetsForTargetDataType(NDArray* sourceArray) {
  if (!sourceArray->isS()) THROW_EXCEPTION("Source array is not a string array!");

  LongType offsetsLength = ShapeUtils::stringBufferHeaderRequirements(sourceArray->lengthOf());

  std::vector<LongType> offsets(sourceArray->lengthOf() + 1);

  const auto nInputoffsets = sourceArray->bufferAsT<LongType>();

  LongType start = 0, stop = 0;
  LongType dataLength = 0;

  int numStrings = sourceArray->isScalar() ? 1 : sourceArray->lengthOf();
  auto data = sourceArray->bufferAsT<int8_t>() + offsetsLength;
  for (LongType e = 0; e < numStrings; e++) {
    offsets[e] = dataLength;
    start = nInputoffsets[e];
    stop = nInputoffsets[e + 1];

    // Determine size difference based on the target type (using UTF conversions)
    if (sourceArray->dataType() == UTF8) {
      dataLength += (DataTypeUtils::fromT<T>() == UTF16)
                        ? unicode::offsetUtf8StringInUtf16(data + start, stop)
                        : unicode::offsetUtf8StringInUtf32(data + start, stop);
    } else if (sourceArray->dataType() == UTF16) {
      dataLength += (DataTypeUtils::fromT<T>() == UTF32)
                        ? unicode::offsetUtf16StringInUtf32(data + start, (stop / sizeof(char16_t)))
                        : unicode::offsetUtf16StringInUtf8(data + start, (stop / sizeof(char16_t)));
    } else if (sourceArray->dataType() == UTF32) {
      dataLength += (DataTypeUtils::fromT<T>() == UTF16)
                        ? unicode::offsetUtf32StringInUtf16(data + start, (stop / sizeof(char32_t)))
                        : unicode::offsetUtf32StringInUtf8(data + start, (stop / sizeof(char32_t)));
    }
  }

  offsets[numStrings] = dataLength;

  return offsets;
}
#define DEFINE_OFFSET(T) template std::vector<LongType> StringUtils::calculateOffsetsForTargetDataType<GET_SECOND(T)>(NDArray* sourceArray);
ITERATE_LIST((SD_STRING_TYPES),DEFINE_OFFSET)

static SD_INLINE bool match(const LongType* haystack, const LongType* needle, LongType length) {
  for (int e = 0; e < length; e++)
    if (haystack[e] != needle[e]) return false;

  return true;
}

template <typename T>
std::string StringUtils::bitsToString(T value) {
  return std::bitset<sizeof(T) * 8>(value).to_string();
}

template std::string StringUtils::bitsToString(int value);
template std::string StringUtils::bitsToString(uint32_t value);
template std::string StringUtils::bitsToString(LongType value);
template std::string StringUtils::bitsToString(uint64_t value);

LongType StringUtils::countSubarrays(const void* haystack, LongType haystackLength, const void* needle,
                                     LongType needleLength) {
  auto haystack2 = reinterpret_cast<const LongType*>(haystack);
  auto needle2 = reinterpret_cast<const LongType*>(needle);

  LongType number = 0;

  for (LongType e = 0; e < haystackLength - needleLength; e++) {
    if (match(&haystack2[e], needle2, needleLength)) number++;
  }

  return number;
}

LongType StringUtils::byteLength(NDArray& array) {
  if (!array.isS())
    throw datatype_exception::build("StringUtils::byteLength expects one of String types;", array.dataType());

  auto buffer = array.bufferAsT<LongType>();
  return buffer[array.lengthOf()];
}

std::vector<std::string> StringUtils::split(const std::string& haystack, const std::string& delimiter) {
  std::vector<std::string> output;

  std::string::size_type prev_pos = 0, pos = 0;

  // iterating through the haystack till the end
  while ((pos = haystack.find(delimiter, pos)) != std::string::npos) {
    output.emplace_back(haystack.substr(prev_pos, pos - prev_pos));
    prev_pos = ++pos;
  }

  output.emplace_back(haystack.substr(prev_pos, pos - prev_pos));  // Last word

  return output;
}

bool StringUtils::u8StringToU16String(const std::string& u8, std::u16string& u16) {
  if (u8.empty()) return false;

  u16.resize(unicode::offsetUtf8StringInUtf16(u8.data(), u8.size()) / sizeof(char16_t));
  if (u8.size() == u16.size())
    u16.assign(u8.begin(), u8.end());
  else
    return unicode::utf8to16(u8.data(), &u16[0], u8.size());

  return true;
}

bool StringUtils::u8StringToU32String(const std::string& u8, std::u32string& u32) {
  if (u8.empty()) return false;

  u32.resize(unicode::offsetUtf8StringInUtf32(u8.data(), u8.size()) / sizeof(char32_t));
  if (u8.size() == u32.size())
    u32.assign(u8.begin(), u8.end());
  else
    return unicode::utf8to32(u8.data(), &u32[0], u8.size());

  return true;
}

bool StringUtils::u16StringToU32String(const std::u16string& u16, std::u32string& u32) {
  if (u16.empty()) return false;

  u32.resize(unicode::offsetUtf16StringInUtf32(u16.data(), u16.size()) / sizeof(char32_t));
  if (u16.size() == u32.size())
    u32.assign(u16.begin(), u16.end());
  else
    return unicode::utf16to32(u16.data(), &u32[0], u16.size());

  return true;
}

bool StringUtils::u16StringToU8String(const std::u16string& u16, std::string& u8) {
  if (u16.empty()) return false;

  u8.resize(unicode::offsetUtf16StringInUtf8(u16.data(), u16.size()));
  if (u16.size() == u8.size())
    u8.assign(u16.begin(), u16.end());
  else
    return unicode::utf16to8(u16.data(), &u8[0], u16.size());

  return true;
}

bool StringUtils::u32StringToU16String(const std::u32string& u32, std::u16string& u16) {
  if (u32.empty()) return false;

  u16.resize(unicode::offsetUtf32StringInUtf16(u32.data(), u32.size()) / sizeof(char16_t));
  if (u32.size() == u16.size())
    u16.assign(u32.begin(), u32.end());
  else
    return unicode::utf32to16(u32.data(), &u16[0], u32.size());

  return true;
}

bool StringUtils::u32StringToU8String(const std::u32string& u32, std::string& u8) {
  if (u32.empty()) return false;

  u8.resize(unicode::offsetUtf32StringInUtf8(u32.data(), u32.size()));
  if (u32.size() == u8.size())
    u8.assign(u32.begin(), u32.end());
  else
    return unicode::utf32to8(u32.data(), &u8[0], u32.size());

  return true;
}

template <typename T>
std::string StringUtils::vectorToString(const std::vector<T>& vec) {
  std::string result;
  for (auto v : vec) result += valueToString<T>(v);

  return result;
}

template std::string StringUtils::vectorToString(const std::vector<int>& vec);
template std::string StringUtils::vectorToString(const std::vector<LongType>& vec);
template std::string StringUtils::vectorToString(const std::vector<int16_t>& vec);
template std::string StringUtils::vectorToString(const std::vector<uint32_t>& vec);
}  // namespace sd
