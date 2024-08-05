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
// Created by raver119 on 15/11/17.
//

#ifndef LIBND4J_VARIABLESSET_H
#define LIBND4J_VARIABLESSET_H
#include <graph/Variable.h>

#include <iterator>
#include <vector>

namespace sd {
namespace graph {
class SD_LIB_EXPORT VariablesSet {
 protected:
  std::vector<Variable*> _holder;
  Status _status;

 public:
  VariablesSet(Status status = Status::OK);
  ~VariablesSet();

  Status status();

  int size();

  void push_back(Variable* variable);

  Variable* at(int index);
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_VARIABLESSET_H
