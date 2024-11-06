/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author raver119@gmail.com
//

#ifndef LIBND4J_CONTEXT_H
#define LIBND4J_CONTEXT_H
#include <system/common.h>
#include <array/NDArray.h>
#include <execution/Engine.h>
#include <graph/ContextPrototype.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <memory/Workspace.h>


#include <vector>

namespace sd {
namespace graph {
/**
 * This class defines input desired for any given node/operation within graph
 */
class SD_LIB_EXPORT Context : public ContextPrototype {
 protected:
  memory::Workspace* _workspace = nullptr;
  VariableSpace* _variableSpace = nullptr;
  std::pair<LongType, LongType> _executionTime;
  random::RandomBuffer* _rng = nullptr;

  DataType _dataType = FLOAT32;
  // branch for divergent_op
  int _branch = 0;

  // temporary context for standalone ops execution
  LaunchContext* _context = nullptr;

  std::vector<DataType> _dataTypes;

  // fields for fast execution (out-of-graph ops use)
  std::vector<NDArray*> _fastpath_in;
  std::vector<NDArray*> _fastpath_out;
  std::vector<NDArray*> _intermediateResults;
  std::vector<NDArray*> _handles;

  bool _helpersAllowed = true;

  // in some cases we might be able to skip shape function for validation purposes
  bool _shapeFunctionOverride = false;

  // special flag used during conversion from Graph exec to FastPath exec
  bool _forbidFastPath = false;

 public:
  Context(ContextPrototype* prototype, VariableSpace* variableSpace);
  explicit Context(int nodeId, VariableSpace* variableSpace = nullptr);
  Context(int nodeId, VariableSpace* variableSpace, bool isInplace);

  // default destructor
  ~Context();

  // these methods are for execution timing
  void setOuterTime(LongType time);
  void setInnerTime(LongType time);
  LongType getOuterTime();
  LongType getInnerTime();

  DataType dataType() override;

  DataType dataType(int index) override;
  void setDataType(int index, DataType type) override;
  // these methods are related to Workspace abstraction
  bool hasWorkspaceProvided();
  void attachWorkspace(memory::Workspace* workspace);
  void forgetWorkspace();

  // these methods return full-time workspace
  memory::Workspace* getWorkspace();
  memory::Workspace* workspace();
  memory::Workspace* fWorkspace();

  // this method returns workspace for temporary allocations
  memory::Workspace* tWorkspace();

  // this method returns workspace for object allocations
  memory::Workspace* oWorkspace();

  void setVariableSpace(VariableSpace* variableSpace);

  random::RandomBuffer* getRNG();
  void setRNG(random::RandomBuffer* rng);

  void setTargetEngine(samediff::Engine engine);

  VariableSpace* getVariableSpace();

  LaunchContext* launchContext();

  // these fields define, if we can execute specific node in-place, without generating new array

  // these variables are only for Divergent Nodes
  int getBranch();
  void setBranch(int branch);

  /**
   *
   * @return
   */
  Stash* getStash();

  /**
   *
   */
  void trackList(NDArrayList* list);

  /**
   * This method returns variable for a given input index for this block
   * @param idx
   * @return
   */
  Variable* getVariable(int idx);
  Variable* variable(int idx);

  /**
   * This method is shortcut to getVariable(int idx);
   *
   * + it check fastpath for array availability (preferred)
   * @return
   */
  NDArray* getNDArray(int idx);
  NDArray* array(int idx);

  /**
   * An intermediate results
   * is a performance optimization
   * meant for use with backpropagation.
   * There are many ops where a part of the forward
   * pass is used as a component of the backward pass.
   * By storing this in the context
   * it can be passed down to a backward op.
   * @param idx the index of the intermediate result
   * @return
   */
  NDArray *intermediateResult(int idx) {
    return _intermediateResults.at(idx);
  }

  /**
   * Add an intermediate result as described
   * in {@link #intermediateResult(int)}
   * @param array the intermediate result to add
   */
  void addIntermediateResult(NDArray *array) {
        _intermediateResults.push_back(array);
  }



  /**
   * This method returns the number of intermediate results
   * in this context.
   * @return
   */
  int numIntermediates() {
    return _intermediateResults.size();
  }

  bool hasIntermediateResults() {
    return numIntermediates() > 0;
  }

  /**
   * This method fetches variable from VariableSpace DIRECTLY
   * @param p
   * @return
   */
  Variable* variable(int node, int index);
  Variable* variable(std::pair<int, int>& p);
  Variable* variable(std::initializer_list<int> p);

  void pushNDArrayToVariableSpace(int nodeId, int index, NDArray* array, bool removable = true);
  void pushNDArrayToVariableSpace(std::pair<int, int>& pair, NDArray* array, bool removable = true);

  void pushNDArrayListToVariableSpace(int nodeId, int index, NDArrayList* list, bool track = true);
  void pushNDArrayListToVariableSpace(std::pair<int, int>& pair, NDArrayList* list, bool track = true);

  bool isValueAvailable(int idx = 0);

  Variable* ensureVariable(int idx = 0);

  unsigned long width() override;

  unsigned long outputWidth();

  // methods used in java interop
  /**
   * This method checks if Context uses fastpath variable access
   * @return
   */
  bool isFastPath();

  /**
   * Method allows to forbid FastPath execution
   * @param reallyForbid
   */
  void forbidFastPath(bool reallyForbid);


  std::vector<NDArray*>& fastpath_in();
  std::vector<NDArray*>& fastpath_out();



  std::vector<NDArray*>& intermediateResults() {
    return _intermediateResults;
  }

  void pushIntermediateResult(NDArray* array) {
    _intermediateResults.push_back(array);
  }

  void setIntermediateResult(int idx, NDArray* array) {
    if(intermediateResults().size() < idx) {
        intermediateResults().resize(idx + 1);
    }

    _intermediateResults[idx] = array;
  }

  void setInputArrays(int numArrays,NDArray** array, bool removable = false);

  void setInputArray(int index, NDArray* array, bool removable = false);


  void setOutputArray(int index, NDArray* array, bool removable = false);


  void setOutputArrays(int numArrays,NDArray** array, bool removable = false);

  void setTArguments(double* arguments, int numberOfArguments);
  void setIArguments(LongType* arguments, int numberOfArguments);
  void setBArguments(bool* arguments, int numberOfArguments);
  void setDArguments(DataType* arguments, int numberOfArguments);

  void setTArguments(const std::vector<double>& tArgs);
  void setIArguments(const std::vector<LongType>& tArgs);
  void setBArguments(const std::vector<bool>& tArgs);
  void setDArguments(const std::vector<DataType>& dArgs);

  /**
   * This method purges fastpath in/out contents and releases all the handles.
   *
   * PLEASE NOTE: I/T/B/D args will stay intact
   */
  void clearFastPath();

  void setCudaContext(Pointer cudaStream, Pointer reductionPointer, Pointer allocationPointer);

  void allowHelpers(bool reallyAllow);
  bool helpersAllowed();

  void setShapeFunctionOverride(bool reallyOverride);
  bool shapeFunctionOverride();

  samediff::ExecutionMode executionMode();
  void setExecutionMode(samediff::ExecutionMode executionMode);

  bool isTraining();
  bool isInference();

  NDArray* outputArray(int idx);
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_BLOCK_H
