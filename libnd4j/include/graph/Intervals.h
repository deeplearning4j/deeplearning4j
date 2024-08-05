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
// Created by yurii@skymind.io on 24.10.2017.
//

#ifndef LIBND4J_INTERVALS_H
#define LIBND4J_INTERVALS_H

#include <system/common.h>

#include <initializer_list>
#include <vector>

namespace sd {

class SD_LIB_EXPORT Intervals {
 private:
  std::vector<std::vector<LongType>> _content;

 public:
  // default constructor
  Intervals();

  // constructor
  Intervals(const std::initializer_list<std::vector<LongType>>& content);
  Intervals(const std::vector<std::vector<LongType>>& content);

  // accessing operator
  std::vector<LongType> operator[](const LongType i) const;

  // returns size of _content
  int size() const;
};

}  // namespace sd

#endif  // LIBND4J_INTERVALS_H
