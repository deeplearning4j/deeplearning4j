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

#ifndef DEV_TESTS_CONSTANTDESCRIPTOR_H
#define DEV_TESTS_CONSTANTDESCRIPTOR_H

#include <array/ConstantDataBuffer.h>
#include <array/DataType.h>
#include <system/common.h>

#include <unordered_map>
#include <vector>

namespace sd {
class SD_LIB_EXPORT ConstantDescriptor {
 private:
  std::vector<LongType> _integerValues;
  std::vector<double> _floatValues;

 public:
  ConstantDescriptor(double *values, int length);
  ConstantDescriptor(LongType const *values, int length);
  ConstantDescriptor(std::initializer_list<double> values);

  explicit ConstantDescriptor(std::vector<LongType> &values);
  explicit ConstantDescriptor(std::vector<double> &values);

  ~ConstantDescriptor() = default;

  // equal to operator
  bool operator==(const ConstantDescriptor &other) const;

  // less than operator
  bool operator<(const ConstantDescriptor &other) const;

  bool isInteger() const;
  bool isFloat() const;

  LongType length() const;

  const std::vector<LongType> &integerValues() const;
  const std::vector<double> &floatValues() const;
};
}  // namespace sd

#ifndef __JAVACPP_HACK__

namespace std {
template <>
class SD_LIB_EXPORT hash<sd::ConstantDescriptor> {
 public:
  size_t operator()(const sd::ConstantDescriptor &k) const;
};
}  // namespace std

#endif

#endif  // DEV_TESTS_CONSTANTDESCRIPTOR_H
