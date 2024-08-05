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

#ifndef LIBND4J_NODE_PROFILE_H
#define LIBND4J_NODE_PROFILE_H

#include <system/common.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {
class SD_LIB_EXPORT NodeProfile {
 private:
  int _id;
  std::string _name;

  LongType _merges = 1L;

  // time spent during deserialization
  LongType _buildTime = 0L;

  // time spent before op execution
  LongType _preparationTime = 0L;

  // time spent for op execution
  LongType _executionTime = 0L;

  // total time spent during node execution
  LongType _totalTime = 0L;

  // time spent for output shape creation
  LongType _shapeTime = 0L;

  // time spent for output arrays creation
  LongType _arrayTime = 0L;

  LongType _inputTime = 0L;

  // amount of memory used for outputs
  LongType _memoryActivations = 0L;

  // amount of memory used internally for temporary arrays
  LongType _memoryTemporary = 0L;

  // amount of memory used internally for objects
  LongType _memoryObjects = 0L;

  // total amount of memory used during execution
  LongType _memoryTotal = 0L;

  std::vector<std::string> _inputShapes;
  std::vector<std::string> _outputShapes;

 public:
  NodeProfile() = default;
  ~NodeProfile() = default;

  explicit NodeProfile(int id, const char* name);

  void setBuildTime(LongType time);
  void setPreparationTime(LongType time);
  void setExecutionTime(LongType time);
  void setTotalTime(LongType time);
  void setShapeFunctionTime(LongType time);
  void setArrayTime(LongType time);
  void setInputTime(LongType time);

  void setActivationsSize(LongType bytes);
  void setTemporarySize(LongType bytes);
  void setObjectsSize(LongType bytes);
  void setTotalSize(LongType bytes);

  void addInputShape(LongType const* shapeInfo);
  void addOutputShape(LongType const* shapeInfo);

  LongType getActivationsSize() const;
  LongType getTemporarySize() const;
  LongType getObjectsSize() const;
  LongType getTotalSize() const;

  LongType getExecutionTime() const;

  std::string& name();

  void merge(NodeProfile* other);
  void assign(NodeProfile* other);

  void printOut();
};
}  // namespace graph
}  // namespace sd

#endif
