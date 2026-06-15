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
// Created by raver119 on 2018-09-16.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
// @author Abdelrauf

#ifndef DEV_TESTS_NDARRAYFACTORY_H
#define DEV_TESTS_NDARRAYFACTORY_H
#include <array/NDArray.h>

#include <initializer_list>
#include <vector>
#include <execution/LaunchContext.h>

#include <string>

namespace sd {

class SD_LIB_EXPORT NDArrayFactory {
 private:
  template <typename T>
  static void memcpyFromVector(void *ptr, const std::vector<T> &vector);

 public:
  template <typename T>
  static NDArray *empty_(LaunchContext *context = LaunchContext ::defaultContext());

  static NDArray *empty_(DataType dataType, LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *empty(LaunchContext *context = LaunchContext ::defaultContext());

  static NDArray *empty(DataType dataType, LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *valueOf(const std::initializer_list<LongType> &shape, T value, char order = 'c',
                          LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *valueOf(std::vector<LongType> &shape, T value, const char order = 'c',
                          LaunchContext *context = LaunchContext ::defaultContext());

  static NDArray *valueOf(std::vector<LongType> &shape, NDArray&value, const char order = 'c',
                          LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *linspace(T from, T to, LongType numElements);

  static NDArray *create(ShapeDescriptor *shapeDescriptor, LaunchContext *context = LaunchContext ::defaultContext());

  static NDArray *create(DataType dtype, LaunchContext *context = LaunchContext ::defaultContext());


  template <typename T>
  static NDArray *create_(const T value, LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *create_(DataType dtype, LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *create(const T scalar, LaunchContext *context = LaunchContext ::defaultContext());
  template <typename T>
  static NDArray *create(DataType type, const T scalar, LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *vector(LongType length, T startingValue = (T)0,
                         LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *create_(const char order, std::vector<LongType> &shape,
                          LaunchContext *context = LaunchContext ::defaultContext());

  static NDArray *create_(const char order, std::vector<LongType> &shape, DataType dataType,
                          LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *create_(char order, const std::vector<LongType> &shape, const std::vector<T> &data,
                          LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *create(const char order, const std::vector<LongType> &shape, const std::vector<T> &data,
                        LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *create(char order, const std::vector<LongType> &shape,
                        LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *create(char order, const std::vector<LongType> &shape, DataType dtype,
                        LaunchContext *context = LaunchContext ::defaultContext());

  template <typename T>
  static NDArray *create(const std::vector<T> &values, LaunchContext *context = LaunchContext ::defaultContext());

#ifndef __JAVACPP_HACK__
  // this method only available out of javacpp

  template <typename T>
  static NDArray *create(T *buffer, char order, const std::initializer_list<LongType> &shape,
                        LaunchContext *context = LaunchContext ::defaultContext());

  /**
   * This method creates NDArray from .npy file
   * @param fileName
   * @return
   */
  static NDArray fromNpyFile(const char *fileName);

  #if defined(HAS_UTF8)
  /**
   * This factory create array from utf8 string
   * @return NDArray default dataType UTF8
   */
  static NDArray *string(const char *string, DataType dtype = UTF8,
                        LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *string_(const char *string, DataType dtype = UTF8,
                          LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *string_(const std::string &string, DataType dtype = UTF8,
                          LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *string(const std::string &string, DataType dtype = UTF8,
                        LaunchContext *context = LaunchContext::defaultContext());

  static NDArray *string(std::vector<LongType> &shape, const std::vector<const char *> &strings,
                        DataType dataType = UTF8, LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *string(std::vector<LongType> &shape, const std::vector<std::string> &string,
                        DataType dataType = UTF8, LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::vector<const char *> &strings,
                          DataType dataType = UTF8, LaunchContext *context = LaunchContext ::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::vector<std::string> &string,
                          DataType dataType = UTF8, LaunchContext *context = LaunchContext ::defaultContext());
#endif

#if defined(HAS_UTF16)
  /**
   * This factory create array from utf16 string
   * @return NDArray default dataType UTF16
   */
  static NDArray *string(const char16_t *u16string, DataType dtype = UTF16,
                        LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(const char16_t *u16string, DataType dtype = UTF16,
                          LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(const std::u16string &u16string, DataType dtype = UTF16,
                          LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(const std::u16string &u16string, DataType dtype = UTF16,
                        LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This factory create array from vector of utf16 strings
   * @return NDArray default dataType UTF16
   */
  static NDArray *string(std::vector<LongType> &shape, const std::initializer_list<const char16_t *> &strings,
                        DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(std::vector<LongType> &shape, const std::initializer_list<std::u16string> &string,
                        DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(std::vector<LongType> &shape, const std::vector<const char16_t *> &strings,
                        DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(std::vector<LongType> &shape, const std::vector<std::u16string> &string,
                        DataType dtype = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape,
                          const std::initializer_list<const char16_t *> &strings,
                          DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::initializer_list<std::u16string> &string,
                          DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::vector<const char16_t *> &strings,
                          DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::vector<std::u16string> &string,
                          DataType dataType = UTF16, LaunchContext *context = LaunchContext::defaultContext());
#endif

#if defined(HAS_UTF32)
  /**
   * This factory create array from utf32 string
   * @return NDArray default dataType UTF32
   */
  static NDArray *string(const char32_t *u32string, DataType dtype = UTF32,
                        LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(const char32_t *u32string, DataType dtype = UTF32,
                          LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(const std::u32string &u32string, DataType dtype = UTF32,
                          LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(const std::u32string &u32string, DataType dtype = UTF32,
                        LaunchContext *context = LaunchContext::defaultContext());

  /**
   * This factory create array from vector of utf32 strings
   * @return NDArray default dataType UTF32
   */
  static NDArray *string(std::vector<LongType> &shape, const std::initializer_list<const char32_t *> &strings,
                        DataType dataType = UTF32, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(std::vector<LongType> &shape, const std::initializer_list<std::u32string> &string,
                        DataType dataType = UTF32, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string(std::vector<LongType> &shape, const std::vector<const char32_t *> &strings, DataType dtype,
                        LaunchContext *context);
  static NDArray *string(std::vector<LongType> &shape, const std::vector<std::u32string> &string,
                        DataType dtype = UTF32, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape,
                          const std::initializer_list<const char32_t *> &strings,
                          DataType dataType = UTF32, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::initializer_list<std::u32string> &string,
                          DataType dataType = UTF32, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::vector<const char32_t *> &strings,
                          DataType dataType = UTF32, LaunchContext *context = LaunchContext::defaultContext());
  static NDArray *string_(std::vector<LongType> &shape, const std::vector<std::u32string> &string,
                          DataType dataType = UTF32, LaunchContext *context = LaunchContext::defaultContext());
#endif

#endif
};
}  // namespace sd

#endif  // DEV_TESTS_NDARRAYFACTORY_H
