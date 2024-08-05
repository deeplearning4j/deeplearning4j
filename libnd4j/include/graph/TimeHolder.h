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
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_TIMEHOLDER_H
#define LIBND4J_TIMEHOLDER_H
#include <system/common.h>

#include <map>

namespace sd {
namespace graph {
class SD_LIB_EXPORT TimeHolder {
 private:
  std::map<int, LongType> _outer;
  std::map<int, LongType> _inner;

 public:
  TimeHolder() = default;
  ~TimeHolder() = default;

  void setOuterTime(int nodeId, LongType time);
  void setInnerTime(int nodeId, LongType time);

  LongType outerTime(int nodeId);
  LongType innerTime(int nodeId);
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_TIMEHOLDER_H
