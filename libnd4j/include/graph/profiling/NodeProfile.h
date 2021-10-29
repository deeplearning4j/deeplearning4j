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

  sd::LongType _merges = 1L;

  // time spent during deserialization
  sd::LongType _buildTime = 0L;

  // time spent before op execution
  sd::LongType _preparationTime = 0L;

  // time spent for op execution
  sd::LongType _executionTime = 0L;

  // total time spent during node execution
  sd::LongType _totalTime = 0L;

  // time spent for output shape creation
  sd::LongType _shapeTime = 0L;

  // time spent for output arrays creation
  sd::LongType _arrayTime = 0L;

  sd::LongType _inputTime = 0L;

  // amount of memory used for outputs
  sd::LongType _memoryActivations = 0L;

  // amount of memory used internally for temporary arrays
  sd::LongType _memoryTemporary = 0L;

  // amount of memory used internally for objects
  sd::LongType _memoryObjects = 0L;

  // total amount of memory used during execution
  sd::LongType _memoryTotal = 0L;

  std::vector<std::string> _inputShapes;
  std::vector<std::string> _outputShapes;

 public:
  NodeProfile() = default;
  ~NodeProfile() = default;

  explicit NodeProfile(int id, const char* name);

  void setBuildTime(sd::LongType time);
  void setPreparationTime(sd::LongType time);
  void setExecutionTime(sd::LongType time);
  void setTotalTime(sd::LongType time);
  void setShapeFunctionTime(sd::LongType time);
  void setArrayTime(sd::LongType time);
  void setInputTime(sd::LongType time);

  void setActivationsSize(sd::LongType bytes);
  void setTemporarySize(sd::LongType bytes);
  void setObjectsSize(sd::LongType bytes);
  void setTotalSize(sd::LongType bytes);

  void addInputShape(sd::LongType const* shapeInfo);
  void addOutputShape(sd::LongType const* shapeInfo);

  sd::LongType getActivationsSize() const;
  sd::LongType getTemporarySize() const;
  sd::LongType getObjectsSize() const;
  sd::LongType getTotalSize() const;

  sd::LongType getExecutionTime() const;

  std::string& name();

  void merge(NodeProfile* other);
  void assign(NodeProfile* other);

  void printOut();
};
}  // namespace graph
}  // namespace sd

#endif
