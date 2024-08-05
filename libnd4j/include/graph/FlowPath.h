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

#ifndef LIBND4J_FLOWPATH_H
#define LIBND4J_FLOWPATH_H
#include <graph/FrameState.h>
#include <graph/NodeState.h>
#include <graph/profiling/GraphProfile.h>
#include <system/op_boilerplate.h>

#include <map>
#include <unordered_map>

namespace sd {
namespace graph {
class SD_LIB_EXPORT FlowPath {
 private:
  SD_MAP_IMPL<int, NodeState> _states;
  SD_MAP_IMPL<LongType, FrameState> _frames;

  void ensureNode(int nodeId);
  void ensureFrame(int nodeId);

  GraphProfile _profile;

 public:
  FlowPath() = default;
  ~FlowPath() = default;

  void setInnerTime(int nodeId, LongType time);
  void setOuterTime(int nodeId, LongType time);

  LongType innerTime(int nodeId);
  LongType outerTime(int nodeId);

  bool isNodeActive(int nodeId);
  void markNodeActive(int nodeId, bool isActive);

  bool wasExecuted(int nodeId);
  void markExecuted(int nodeId, bool wasExecuted);

  int branch(int nodeId);
  void markBranch(int nodeId, int index);

  // Frame-related methods

  void registerFrame(LongType frameId);
  void forgetFrame(LongType frameId);

  bool isFrameActive(LongType frameId);
  void markFrameActive(LongType frameId, bool isActive);

  bool isRewindPlanned(LongType frameId);
  void planRewind(LongType frameId, bool reallyRewind);

  int getRewindPosition(LongType frameId);
  void setRewindPosition(LongType frameId, int position);
  void setRewindPositionOnce(LongType frameId, int position);

  void incrementNumberOfCycles(LongType frameId);
  LongType getNumberOfCycles(LongType frameId);

  GraphProfile* profile();
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_FLOWPATH_H
