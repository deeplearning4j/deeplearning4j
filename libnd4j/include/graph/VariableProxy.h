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
//  @author raver119@gmail.com
//
#include <graph/VariableSpace.h>

namespace sd {
namespace graph {
class SD_LIB_EXPORT VariableProxy : public VariableSpace {
 protected:
  VariableSpace *_backed = nullptr;
  VariableSpace *_current = nullptr;

 public:
  explicit VariableProxy(VariableSpace *reference);
  ~VariableProxy();

  virtual VariableSpace &operator=(const VariableSpace &other);

  virtual int numberOfPlaceholders();
  virtual std::vector<Variable *> *getPlaceholders();

  virtual memory::Workspace *workspace();

  virtual bool hasExternalVariable(int it);
  virtual bool hasExternalVariable(std::pair<int, int> &pair);
  virtual bool hasExternalVariable(std::string *symbol);

  virtual bool hasVariable(int id);
  virtual bool hasVariable(int id, int idx);
  virtual bool hasVariable(std::pair<int, int> &pair);
  virtual bool hasVariable(std::string *symbol);

  virtual Variable *getVariable(int id);
  virtual Variable *getVariable(int id, int idx);
  virtual Variable *getVariable(std::pair<int, int> &pair);
  virtual Variable *getVariable(std::string *symbol);

  virtual std::vector<Variable *> getVariables();

  virtual Variable *putVariable(std::pair<int, int> &pair, NDArray *array);
  virtual void putVariable(std::pair<int, int> &pair, Variable *variable);
  virtual void putVariable(int id, Variable *variable);
  virtual void putVariable(int id, NDArray *array);
  virtual Variable *putVariable(int id, int idx, NDArray *array);
  virtual void putVariable(int id, int idx, NDArray &array);
  virtual void putVariable(int id, int idx, Variable *array);

  virtual void replaceVariable(Variable *variable);

  virtual void dropVariable(std::pair<int, int> &pair);
  virtual void dropVariable(int id, int idx);

  virtual void putOutputVariable(Variable *variable);

  virtual void trackList(NDArrayList *list);

  // memory-related statistics
  virtual LongType externalMemory();
  virtual LongType internalMemory();
  virtual LongType totalMemory();

  virtual int externalEntries();
  virtual int internalEntries();
  virtual int totalEntries();

  virtual VariableSpace *clone();

  virtual Stash *getStash();
  virtual void setFlowPath(FlowPath *timers);
  virtual FlowPath *flowPath();
};
}  // namespace graph
}  // namespace sd
