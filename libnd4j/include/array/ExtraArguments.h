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

#ifndef DEV_TESTS_EXTRAARGUMENTS_H
#define DEV_TESTS_EXTRAARGUMENTS_H

#include <array/DataType.h>
#include <stdlib.h>
#include <system/common.h>

#include <initializer_list>
#include <vector>

namespace sd {
class SD_LIB_EXPORT ExtraArguments {
 private:
  std::vector<double> _fpArgs;
  std::vector<sd::LongType> _intArgs;

  std::vector<sd::Pointer> _pointers;

  template <typename T>
  void convertAndCopy(sd::Pointer pointer, sd::LongType offset);

  void *allocate(size_t length, size_t elementSize);

 public:
  explicit ExtraArguments(std::initializer_list<double> arguments);
  explicit ExtraArguments(std::initializer_list<sd::LongType> arguments);

  explicit ExtraArguments(const std::vector<double> &arguments);
  explicit ExtraArguments(const std::vector<int> &arguments);
  explicit ExtraArguments(const std::vector<sd::LongType> &arguments);

  explicit ExtraArguments();
  ~ExtraArguments();

  template <typename T>
  void *argumentsAsT(sd::LongType offset = 0);

  void *argumentsAsT(sd::DataType dataType, sd::LongType offset = 0);

  size_t length();
};
}  // namespace sd

#endif  // DEV_TESTS_EXTRAARGUMENTS_H
