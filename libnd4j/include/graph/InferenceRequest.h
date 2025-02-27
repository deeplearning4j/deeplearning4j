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
#ifndef DEV_TESTS_INFERENCEREQUEST_H
#define DEV_TESTS_INFERENCEREQUEST_H
#include <graph/Variable.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include "ExecutorConfiguration.h"
#include "graph/generated/request_generated.h"

namespace sd {
namespace graph {
class SD_LIB_EXPORT InferenceRequest {
 private:
  LongType _id;
  std::vector<Variable *> _variables;
  std::vector<Variable *> _deletables;

  ExecutorConfiguration *_configuration = nullptr;

  void insertVariable(Variable *variable);

 public:
  InferenceRequest(LongType graphId, ExecutorConfiguration *configuration = nullptr);
  ~InferenceRequest();

  void appendVariable(int id, NDArray *array);
  void appendVariable(int id, int index, NDArray *array);
  void appendVariable(std::string &name, NDArray *array);
  void appendVariable(std::string &name, int id, int index, NDArray *array);
  void appendVariable(Variable *variable);

#ifndef __JAVACPP_HACK__
  flatbuffers::Offset<::graph::FlatInferenceRequest> asFlatInferenceRequest(flatbuffers::FlatBufferBuilder &builder);
#endif
};
}  // namespace graph
}  // namespace sd

#endif  // DEV_TESTS_INFERENCEREQUEST_H
