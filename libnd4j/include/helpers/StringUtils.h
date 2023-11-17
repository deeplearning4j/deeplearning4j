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

#ifndef LIBND4J_STRINGUTILS_H
#define LIBND4J_STRINGUTILS_H

#include <array/NDArray.h>
#include <helpers/unicode.h>
#include <system/op_boilerplate.h>

#include <sstream>
#include <string>
#include <vector>

namespace sd {
class SD_LIB_EXPORT StringUtils {
 public:
  template <typename T>
  static SD_INLINE std::string valueToString(T value) {
    std::ostringstream os;

    os << value;

    // convert the string stream into a string and return
    return os.str();

  }

  static NDArray* createDataBufferFromVector(const std::vector<LongType>& vec, DataType dataType);

  static void broadcastStringAssign(NDArray* x, NDArray* z);

  static std::vector<LongType>* determineOffsetsAndLengths(const NDArray& array, DataType dtype);

  static void convertDataForDifferentDataType(int8_t* outData, const int8_t* inData, const std::vector<LongType>& offsets, DataType inType, DataType outType);

  static std::shared_ptr<DataBuffer> createBufferForStringData(const std::vector<LongType>& offsets, DataType dtype, const LaunchContext* context);

  static NDArray createStringNDArray(const NDArray& array, const std::vector<LongType>& offsets, DataType dtype);

  template <typename T>
  static void convertStringsForDifferentDataType(const NDArray* sourceArray, NDArray* targetArray);

  template <typename T>
  static std::vector<LongType> calculateOffsetsForTargetDataType(const NDArray* sourceArray);

  std::vector<LongType> determineOffsets(const std::string& input, const std::vector<LongType>& lengths);

  std::vector<LongType> determineLengths(const std::string& input);

  static void setValueForDifferentDataType(NDArray* arr, LongType idx, NDArray* input, DataType zType);

  static void assignStringData(NDArray& dest, const NDArray& src, const std::vector<LongType>& offsets, DataType dtype);


  /**
   * These methods convert integer values to string with 0s and 1s
   * @param value
   * @return
   */
  template <typename T>
  static std::string bitsToString(T value);

  /**
   * This method just concatenates error message with a given graphId
   * @param message
   * @param graphId
   * @return
   */
  static SD_INLINE std::string buildGraphErrorMessage(const char* message, LongType graphId) {
    std::string result(message);
    result += " [";
    result += valueToString<LongType>(graphId);
    result += "]";

    return result;
  }

  /**
   * This method returns number of needle matches within haystack
   * PLEASE NOTE: this method operates on 8-bit arrays interpreted as uint8
   *
   * @param haystack
   * @param haystackLength
   * @param needle
   * @param needleLength
   * @return
   */
  static LongType countSubarrays(const void* haystack, LongType haystackLength, const void* needle,
                                 LongType needleLength);

  /**
   * This method returns number of bytes used for string NDArrays content
   * PLEASE NOTE: this doesn't include header
   *
   * @param array
   * @return
   */
  static LongType byteLength(const NDArray& array);

  /**
   * This method splits a string into substring by delimiter
   *
   * @param haystack
   * @param delimiter
   * @return
   */
  static std::vector<std::string> split(const std::string& haystack, const std::string& delimiter);

  /**
   * This method convert u8 string to u16
   * @param const reference to input string
   * @param reference to output u16string
   * @return boolean status
   */
  static bool u8StringToU16String(const std::string& u8, std::u16string& u16);

  /**
   * This method convert u8 string to u32
   * @param const reference to input string
   * @param reference to output u32string
   * @return boolean status
   */
  static bool u8StringToU32String(const std::string& u8, std::u32string& u32);

  /**
   * This method convert u16 string to u32
   * @param const reference to input u16string
   * @param reference to output u32string
   * @return boolean status
   */
  static bool u16StringToU32String(const std::u16string& u16, std::u32string& u32);

  /**
   * This method convert u16 string to u8 string
   * @param const reference to input u16string
   * @param reference to output string
   * @return boolean status
   */
  static bool u16StringToU8String(const std::u16string& u16, std::string& u8);

  /**
   * This method convert u32 string to u16 string
   * @param const reference to input u32string
   * @param reference to output u16string
   * @return boolean status
   */
  static bool u32StringToU16String(const std::u32string& u32, std::u16string& u16);

  /**
   * This method convert u32 string to u8 string
   * @param const reference to input u32string
   * @param reference to output string
   * @return boolean status
   */
  static bool u32StringToU8String(const std::u32string& u32, std::string& u8);

  template <typename T>
  static std::string vectorToString(const std::vector<T>& vec);
};
}  // namespace sd

#endif  // LIBND4J_STRINGUTILS_H
