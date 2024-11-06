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

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H
#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <graph/FlowPath.h>
#include <graph/Stash.h>
#include <graph/Variable.h>
#include <helpers/helper_random.h>
#include <helpers/logger.h>
#include <memory/Workspace.h>

#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {
class SD_LIB_EXPORT VariableSpace {
 protected:
  memory::Workspace* _workspace;

  // stash is NOT cloned
  Stash _stash;

  SD_MAP_IMPL<std::pair<int, int>, sd::graph::Variable*> _paired;
  SD_MAP_IMPL<std::string, sd::graph::Variable*> _symbolic;
  SD_MAP_IMPL<int, sd::graph::Variable*> _variables;
  std::vector<sd::graph::Variable*> _external;
  std::vector<sd::graph::Variable*> _internal;

  std::vector<NDArrayList*> _lists;

  std::vector<sd::graph::Variable*> _placeholders;

  void silentPutVariable(std::pair<int, int>& pair, sd::graph::Variable* variable);

  int _auto_counter = -1;

  std::mutex _varmap;

  SD_MAP_IMPL<int, sd::graph::Variable*> _temporary;

  std::vector<sd::graph::Variable*>* _handles;

  FlowPath* _flow = nullptr;

 public:
  VariableSpace();
  virtual ~VariableSpace();

  virtual VariableSpace& operator=(const VariableSpace& other);

  virtual int numberOfPlaceholders();
  virtual std::vector<sd::graph::Variable*>* getPlaceholders();
  virtual void setWorkspace(memory::Workspace* workspace);

  virtual LaunchContext* launchContext();

  virtual bool hasExternalVariable(int it);
  virtual bool hasExternalVariable(std::pair<int, int>& pair);
  virtual bool hasExternalVariable(std::string* symbol);

  virtual bool hasVariable(int id);
  virtual bool hasVariable(int id, int idx);
  virtual bool hasVariable(std::pair<int, int>& pair);
  virtual bool hasVariable(std::string* symbol);

  virtual Variable* getVariable(int id);
  virtual Variable* getVariable(int id, int idx);
  virtual Variable* getVariable(std::pair<int, int>& pair);
  virtual Variable* getVariable(std::string* symbol);

  virtual std::vector<sd::graph::Variable*> getVariables();

  virtual sd::graph::Variable* putVariable(std::pair<int, int>& pair, NDArray* array);
  virtual void putVariable(std::pair<int, int>& pair, Variable* variable);
  virtual void putVariable(int id, sd::graph::Variable* variable);
  virtual void putVariable(int id, NDArray* array);
  virtual Variable* putVariable(int id, int idx, NDArray* array);
  virtual void putVariable(int id, int idx, NDArray& array);
  virtual void putVariable(int id, int idx, sd::graph::Variable* array);

  virtual void dropVariable(std::pair<int, int>& pair);
  virtual void dropVariable(int id, int idx);

  virtual void trackList(NDArrayList* list);

  virtual void putOutputVariable(sd::graph::Variable* variable);

  virtual void replaceVariable(sd::graph::Variable* variable);

  // memory-related statistics
  virtual LongType externalMemory();
  virtual LongType internalMemory();
  virtual LongType totalMemory();

  virtual int externalEntries();
  virtual int internalEntries();
  virtual int totalEntries();

  virtual VariableSpace* clone();

  std::vector<sd::graph::Variable*>* handles();

  VariableSpace* asT();
  void injectVariable(std::pair<int, int>& pair, sd::graph::Variable* variable);

  virtual Stash* getStash();

  virtual std::vector<sd::graph::Variable*>* getExternalVariables();

  virtual void setFlowPath(FlowPath* timers);
  virtual FlowPath* flowPath();
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_VARIABLESPACE_H
