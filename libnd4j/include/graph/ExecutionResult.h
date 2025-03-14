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

#ifndef LIBND4J_EXECUTION_RESULT
#define LIBND4J_EXECUTION_RESULT
#include <flatbuffers/flatbuffers.h>
#include <graph/Variable.h>

#include <initializer_list>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {
class ExecutionResult {
 private:
  std::vector<Variable *> _variables;
  SD_MAP_IMPL<std::string, Variable *> _stringIdMap;
  SD_MAP_IMPL<std::pair<int, int>, Variable *> _pairIdMap;

  // this flag is used to optionally release variables
  bool _releasable = false;

 public:
  ExecutionResult(const ::graph::FlatResult *flatResult);
  ExecutionResult(std::initializer_list<Variable *> variables);
  ExecutionResult() = default;
  ~ExecutionResult();

  /**
   * This method adds variable pointer to result
   */
  void emplace_back(Variable *variable);

  /**
   * This method returns Variable by its position in output
   */
  Variable *at(int position);

  /**
   * This method returns Variable by its string id
   */
  Variable *byId(std::string &id);

  /**
   * This method returns Variable by its string id
   */
  Variable *byId(const char *str);

  /**
   * This method returns Variable by its numeric id:index pair
   */
  Variable *byId(std::pair<int, int> &id);

  /**
   * This method returns Variable by its numeric id with index 0
   */
  Variable *byId(int id);

  /**
   * This method returns number of elements stored in this entity
   * @return
   */
  LongType size();

#ifndef __JAVACPP_HACK__
  /**
   * This method converts ExecutionResult entity to FlatResult
   */
  flatbuffers::Offset<::graph::FlatResult> asFlatResult(flatbuffers::FlatBufferBuilder &builder);
#endif
};
}  // namespace graph
}  // namespace sd

#endif
