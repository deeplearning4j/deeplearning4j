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
// Created by raver119 on 23.01.18.
//

#ifndef LIBND4J_GRAPHSTATE_H
#define LIBND4J_GRAPHSTATE_H

#include <graph/ArgumentsList.h>
#include <graph/Graph.h>
#include <graph/Scope.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/DeclarableOp.h>
#include <system/op_boilerplate.h>
#include <types/pair.h>

#include <map>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

class SD_LIB_EXPORT GraphState {
 protected:
  // id of this GraphState instance
  LongType _id = 0;

  // map of scopes. Scope id is used as key, since it's referred in calls later anyway
  SD_MAP_IMPL<int, Scope*> _scopes;

  // this variable space holds temp references
  VariableSpace _variableSpace;

  Graph* _graph;

 public:
  explicit GraphState(LongType id);
  ~GraphState();

  /**
   *
   * @return
   */
  LongType id();

  /**
   * This method adds scope to this state tracker
   *
   * @param scopeId
   * @return
   */
  Status registerScope(int scopeId);

  /**
   * This method cheks if scope with given ID exists
   *
   * @param scopeId - ID of the scope
   * @return - TRUE if scope exists, FALSE otherwise
   */
  bool hasScope(int scopeId);

  /**
   * This method removes specified scope from this state tracker
   *
   * @param scopeId
   * @return
   */
  Status forgetScope(int scopeId);

#ifndef __JAVACPP_HACK__
  /**
   * This method adds given op to the end of specified scope
   * PLEASE NOTE: This method is used for tests mostly
   *
   * @param scopeId
   * @param op
   * @return
   */
  Status attachOpToScope(int scopeId, int nodeId, ops::DeclarableOp* op, ArgumentsList inputs);

  /**
   * This method returns pointer to the scope with given id
   *
   * @param scopeId - id of the scope
   */
  Scope* getScope(int scopeId);

  Graph* graph();
#endif
  /**
   * This method adds given op to the end of specified scope
   *
   * @param scopeId
   * @param opNum
   * @param type
   * @return
   */
  Status attachOpToScope(int scopeId, LongType opNum, int type, ArgumentsList inputs);

  /**
   * This method adds return statement to specified scope
   *
   * PLEASE NOTE: should be used only in body scopes
   *
   * @param scopeId
   * @param nodeId
   * @param args
   * @return
   */
  Status defineReturn(int scopeId, int nodeId, ArgumentsList args);

  /**
   * This method returns current variable space of this state holder
   *
   * @return
   */
  VariableSpace* variableSpace();
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPHSTATE_H
