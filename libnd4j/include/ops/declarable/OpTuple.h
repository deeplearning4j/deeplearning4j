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
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_OPTUPLE_H
#define LIBND4J_OPTUPLE_H
#include <array/NDArray.h>

#include <initializer_list>
#include <vector>

namespace sd {
namespace ops {
class SD_LIB_EXPORT OpTuple {
 public:
  std::string _opName;
  std::vector<NDArray*> _inputs;
  std::vector<NDArray*> _outputs;
  std::vector<double> _tArgs;
  std::vector<LongType> _iArgs;

  OpTuple(const char* opName);
  OpTuple(const char* opName, std::initializer_list<NDArray*>&& inputs, std::initializer_list<double>&& tArgs,
          std::initializer_list<LongType>&& iArgs);
  ~OpTuple();

  OpTuple* addInput(NDArray* array);
  OpTuple* addOutput(NDArray* array);
  OpTuple* setTArgs(std::initializer_list<double> tArgs);
  OpTuple* setIArgs(std::initializer_list<LongType> iArgs);
};
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_OPTUPLE_H
