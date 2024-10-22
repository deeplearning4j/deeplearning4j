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
#include <array/InteropDataBuffer.h>
#include <graph/Context.h>
#include <helpers/ShapeUtils.h>

namespace sd {
namespace graph {
Context::Context(ContextPrototype *prototype, VariableSpace *variableSpace) {
  _variableSpace = variableSpace;
  _dataType = prototype->dataType();

  if (prototype != nullptr) {
    for (const auto &v : *(prototype->inputs())) {
      this->_inputs.push_back(v);
    }

    for (const auto &v : *(prototype->getTArguments())) {
      this->_tArgs.push_back(v);
    }

    for (const auto &v : *(prototype->getIArguments())) {
      this->_iArgs.push_back(v);
    }

    for (const auto &v : *(prototype->getBArguments())) {
      this->_bArgs.push_back(v);
    }

    for (const auto &v : *(prototype->getAxis())) {
      this->_axis.push_back(v);
    }

    for(auto v : *(prototype->getDArguments())) {
      this->_dataTypes.push_back(v);
    }

    this->_opNum = prototype->opNum();
    this->_isInplace = prototype->isInplace();
    this->_nodeId = prototype->nodeId();
    this->_useONEDNN = prototype->isUseONEDNN();
  }

  if (variableSpace != nullptr && variableSpace->launchContext()->getWorkspace() != nullptr)
    this->_workspace = variableSpace->launchContext()->getWorkspace();
}
DataType Context::dataType(int index) {
  if(numD() < 1) {
    if(width() > 0) {
      return this->array(index)->dataType();
    } else {
      std::string errorMessage;
      errorMessage += std::string("Context::dataType: Unable to determine data type. Both d args and inputs are empty.");
      errorMessage += std::string(" Index: ");
      errorMessage += std::to_string(index);
      errorMessage += std::string(" Width: ");
      errorMessage += std::to_string(width());
      THROW_EXCEPTION(errorMessage.c_str());
    };
  }


  return getDArguments()->at(index);
}

DataType Context::dataType() { return dataType(0); }

void Context::setDataType(int index, DataType type) {
  if (this->_dataTypes.size() > (size_t)index) _dataTypes[index] = type;
  _dataType = type;
}

Context::Context(int nodeId, VariableSpace *variableSpace) {
  this->_nodeId = nodeId;
  this->_variableSpace = variableSpace;
  this->_isInplace = false;
  this->_workspace = nullptr;

  this->_executionTime.first = 0;
  this->_executionTime.second = 0;

  if (variableSpace != nullptr && variableSpace->launchContext()->getWorkspace() != nullptr)
    this->_workspace = variableSpace->launchContext()->getWorkspace();
}

Context::Context(int nodeId, VariableSpace *variableSpace, bool isInplace) : Context(nodeId, variableSpace) {
  this->_isInplace = isInplace;
}

Context::~Context() {
  this->_iArgs.clear();
  this->_tArgs.clear();
  this->_inputs.clear();
  this->_fastpath_in.clear();
  this->_fastpath_out.clear();

  for (auto v : _handles) delete v;

  if (_context != nullptr) delete _context;
}

void Context::setTargetEngine(samediff::Engine engine) { _engine = engine; }

bool Context::hasWorkspaceProvided() { return this->_workspace != nullptr; }

void Context::attachWorkspace(sd::memory::Workspace *workspace) { this->_workspace = workspace; }

void Context::setVariableSpace(VariableSpace *variableSpace) { this->_variableSpace = variableSpace; }

void Context::forgetWorkspace() { _workspace = nullptr; }

std::vector<NDArray *> &Context::fastpath_in() { return _fastpath_in; }

std::vector<NDArray *> &Context::fastpath_out() { return _fastpath_out; }

bool Context::isFastPath() {
  auto ie = _fastpath_in.empty();
  auto io = _fastpath_out.empty();
  // two options here.
  // either both IN/OUT are filled
  auto b1 = (!ie && !io) || (!ie && _isInplace);

  // or at least something is filled, and FastPath is NOT forbidden
  auto b2 = (!ie || !io) && !_forbidFastPath;
  return b1 || b2;
}

void Context::forbidFastPath(bool reallyForbid) { _forbidFastPath = reallyForbid; }

VariableSpace *Context::getVariableSpace() { return _variableSpace; }

memory::Workspace *Context::getWorkspace() { return _workspace; }

memory::Workspace *Context::workspace() { return _workspace; }

random::RandomBuffer *Context::getRNG() { return _rng; }

void Context::setRNG(random::RandomBuffer *rng) { _rng = rng; }


Stash *Context::getStash() { return _variableSpace->getStash(); }

void Context::trackList(NDArrayList *list) { _variableSpace->trackList(list); }

int Context::getBranch() { return _variableSpace->flowPath()->branch(this->nodeId()); }

void Context::setBranch(int branch) {
  //_branch = branch;
  if (_variableSpace->flowPath() != nullptr) _variableSpace->flowPath()->markBranch(this->nodeId(), branch);
}

LongType Context::getOuterTime() { return this->_executionTime.first; }

LongType Context::getInnerTime() { return this->_executionTime.second; }

void Context::setOuterTime(LongType time) { this->_executionTime.first = time; }

void Context::setInnerTime(LongType time) { this->_executionTime.second = time; }

Variable *Context::getVariable(int idx) {
  if (idx >= this->_inputs.size()) {
    sd_printf("Node %i; Variable [%i] requested, but only %i inputs available\n", this->_nodeId, idx,
              this->_inputs.size());
    THROW_EXCEPTION("Context: bad Variable index");
  }

  auto p = this->_inputs[idx];

  auto v = variable(p);
  // preconditioned with v->variableType()==VariableType::NDARRAY as for other cases getNDArray() can throw exception
  if (Environment::getInstance().isDebugAndVerbose() && v != nullptr && v->variableType() == NDARRAY &&
      v->getNDArray() != nullptr) {
    auto array = v->getNDArray();
    std::string shape_ = ShapeUtils::shapeAsString(array);
    auto type = DataTypeUtils::asString(array->dataType());
    float m = std::numeric_limits<float>::quiet_NaN();
    if (!array->isEmpty()) {
      LongType maxLen = sd::math::sd_min<LongType>(16, array->lengthOf() - 1);

      sd_printf("Debug info for node_%i input[%i]; shape: %s; ews: [%i]; order: [%c]; dtype: [%s];\n",
                this->_nodeId, idx, shape_.c_str(),array->ews(), array->ordering(), type.c_str());
      std::vector<sd::LongType> shapeLen = {array->lengthOf()};
      NDArray &raveled = array->reshape(array->ordering(), shapeLen);
      sd_printf("Values: [ ",0);
      for (LongType i = 0; i < maxLen; i++) {
        auto v = raveled.e<float>(i);
        sd_printf("%f, ", v);
      }
      sd_printf("]\n",0);

    } else {
      sd_printf("Debug info for node_%i input[%i]; shape: %s; ews: [%i]; order: [%c]; dtype: [%s]; mean value: [%f]\n",
                this->_nodeId, idx, shape_.c_str(), (int)array->ews(), array->ordering(), type.c_str(), m);
    }
  }

  return v;
}

Variable *Context::variable(int idx) { return getVariable(idx); }

Variable *Context::variable(std::initializer_list<int> p) {
  if (p.size() != 2) THROW_EXCEPTION("Variable address should have size of 2");

  std::vector<int> vec(p);
  std::pair<int, int> pair(vec[0], vec[1]);
  return variable(pair);
}

Variable *Context::variable(int node, int idx) {
  std::pair<int, int> pair(node, idx);
  return variable(pair);
}

Variable *Context::variable(std::pair<int, int> &p) {
  try {
    return _variableSpace->getVariable(p);
  } catch (std::exception &e) {
    sd_printf("Node %i; Non-existent variable requested: [%i:%i]\n", this->_nodeId, p.first, p.second);
    THROW_EXCEPTION("Bad variable");
  }
}

void Context::pushNDArrayToVariableSpace(int nodeId, int index, NDArray *array, bool removable) {
  std::pair<int, int> pair(nodeId, index);
  pushNDArrayToVariableSpace(pair, array, removable);
}

void Context::pushNDArrayToVariableSpace(std::pair<int, int> &pair, NDArray *array, bool removable) {
  if (_variableSpace != nullptr) {
    if (!_variableSpace->hasVariable(pair)) {
      auto var = new Variable(array, nullptr, pair.first, pair.second);
      _variableSpace->putVariable(pair, var);
      var->markRemovable(removable);
    } else {
      sd_debug("Context: Getting variable in push ndarray\n",0);
      auto var = _variableSpace->getVariable(pair);
      sd_debug("Context: After getting variable in push ndarray to variable space\n",0);
      if (var->hasNDArray()) {
        if (var->getNDArray() != array) {
          if (var->isRemovable() && var->hasNDArray() && !var->getNDArray()->isView()) {
            delete var->getNDArray();
          }
          var->setNDArray(array);
          var->markRemovable(removable);
        }
      } else {
        var->setNDArray(array);
        var->markRemovable(removable);
      }
    }
  }
}

void Context::pushNDArrayListToVariableSpace(int nodeId, int index, NDArrayList *list, bool track) {
  std::pair<int, int> pair(nodeId, index);
  pushNDArrayListToVariableSpace(pair, list, track);
}

void Context::pushNDArrayListToVariableSpace(std::pair<int, int> &pair, NDArrayList *list, bool track) {
  sd_debug("Pre push variable list\n",0);
  if (!_variableSpace->hasVariable(pair)) {
    sd_debug("Context::pushNDArrayListToVariableSpace: Pre create variable when none exists\n",0);
    auto var = new Variable(nullptr, nullptr, pair.first, pair.second);
    sd_debug("Context::pushNDArrayListToVariableSpace: Created when none exists\n",0);
    var->setNDArrayList(list);
    _variableSpace->putVariable(pair, var);
    sd_debug("Context::pushNDArrayListToVariableSpace: Put variable\n",0);
  } else {
    sd_debug("Context::pushNDArrayListToVariableSpace: In else: Getting variable\n",0);
    auto var = _variableSpace->getVariable(pair);
    sd_debug("Context::pushNDArrayListToVariableSpace: Got variable setting list\n",0);
    var->setNDArrayList(list);
  }

  sd_debug("Context::pushNDArrayListToVariableSpace: pre tracking\n",0);

  if (track) _variableSpace->trackList(list);
}

Variable *Context::ensureVariable(int idx) {
  std::pair<int, int> pair(this->nodeId(), idx);

  if (_variableSpace == nullptr) THROW_EXCEPTION("Context::ensureVariable VariableSpace is NULL!");

  if (!_variableSpace->hasVariable(pair)) {
    auto var = new Variable(nullptr, nullptr, this->nodeId(), idx);
    _variableSpace->putVariable(pair, var);
    return var;
  } else {
    sd_debug("Before ensure variable",0);
    return _variableSpace->getVariable(pair);
  }
}

bool Context::isValueAvailable(int idx) {
  auto var = ensureVariable(idx);

  if (var->variableType() == NDARRAY) {
    return var->hasNDArray();
  } else if (var->variableType() == ARRAY_LIST) {
    return var->hasNDArrayList();
  }

  return false;
}

NDArray *Context::getNDArray(int idx) { return array(idx); }


NDArray *Context::outputArray(int idx) {
  // we check for fastpath first
  if (!_fastpath_out.empty() && _fastpath_out.size() > idx) {
    return _fastpath_out[idx];
  }

  std::string errorMessage;
  errorMessage += std::string("Context::outputArray: Fastpath is empty");
  errorMessage += std::string(" Index: ");
  errorMessage += std::to_string(idx);
  errorMessage += std::string(" Fastpath size: ");
  errorMessage += std::to_string(_fastpath_out.size());

  THROW_EXCEPTION(errorMessage.c_str());
}

NDArray *Context::array(int idx) {
  // we check for fastpath first
  if (!_fastpath_in.empty() && _fastpath_in.size() > idx) {
    return _fastpath_in[idx];
  }
  // if no luck for fastpath - return whatever is available
  return getVariable(idx)->getNDArray();
}

memory::Workspace *Context::fWorkspace() { return workspace(); }

memory::Workspace *Context::tWorkspace() { return nullptr; }

memory::Workspace *Context::oWorkspace() { return nullptr; }

LaunchContext *Context::launchContext() {
  // FIXME: we need proper context to be shared here
  if (_context == nullptr) {
    return LaunchContext::defaultContext();
  } else {
    return _context;
  }
}


unsigned long Context::outputWidth() {
  return _fastpath_out.size();
}

unsigned long Context::width() {
  if (!_fastpath_in.empty())
    return _fastpath_in.size();
  else
    return _inputs.size();
}

void Context::setInputArray(int index, NDArray *array, bool removable) {
  if(array->shapeInfo() == nullptr) {
    std::string errorMessage;
    errorMessage += std::string("Array at index ");
    errorMessage += std::to_string(index);
    errorMessage += std::string(" has a null shape buffer!");
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if(array->dataType() != ArrayOptions::dataType(array->shapeInfo())) {
    std::string errorMessage;
    errorMessage += std::string("Array at index ");
    errorMessage += std::to_string(index);
    errorMessage += std::string(" has a different data type than the shape buffer!");
    //add the shape info as a string to the error message
    errorMessage += std::string(" Shape info: ");
    errorMessage += ShapeUtils::shapeAsString(array->shapeInfo());
    errorMessage += std::string(" Data type: ");
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(array->shapeInfo()));
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (_fastpath_in.size() < index + 1) _fastpath_in.resize(index + 1);

  _fastpath_in[index] = array;
  if (removable) _handles.emplace_back(array);
}




void Context::setOutputArray(int index, NDArray *array, bool removable) {
  if (_fastpath_out.size() < index + 1) _fastpath_out.resize(index + 1);
  if(array->dataType() != ArrayOptions::dataType(array->shapeInfo())) {
    std::string errorMessage;
    errorMessage += std::string("Array at index ");
    errorMessage += std::to_string(index);
    errorMessage += std::string(" has a different data type than the shape buffer!");
    //add the shape info as a string to the error message
    errorMessage += std::string(" Shape info: ");
    errorMessage += ShapeUtils::shapeAsString(array->shapeInfo());
    errorMessage += std::string(" Data type: ");
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(array->shapeInfo()));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  _fastpath_out[index] = array;

  if (removable) _handles.emplace_back(array);
}





void validateBufferAndShape(InteropDataBuffer* dataBuffer, LongType * newShapeInfoCast, int index) {
  bool errorFound = false;
  std::string errorMessage;
  //opaque/interop data buffers are created with int8 on purpose and therefore will be excluded from validation here.
  //see more here: https://github.com/deeplearning4j/deeplearning4j/blob/8aa0ef12794ca40a2d00c5c80206a24a3bd6529c/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common/src/main/java/org/nd4j/linalg/cpu/nativecpu/buffer/BaseCpuDataBuffer.java#L386

  bool isString = ArrayOptions::dataType(newShapeInfoCast) == UTF8 || ArrayOptions::dataType(newShapeInfoCast) == UTF16 ||
                  ArrayOptions::dataType(newShapeInfoCast) == UTF32;
  if(isString || shape::isEmptyConst(newShapeInfoCast) || dataBuffer->getDataBuffer()->getDataType() == INT8) return;
  if (dataBuffer != nullptr) {
    if (!shape::isEmptyConst(newShapeInfoCast)) {
      if (dataBuffer->dataBuffer() != nullptr) {

        //opaque/interop data buffers are created with int8 on purpose and therefore will be excluded from validation here.
        //see more here: https://github.com/deeplearning4j/deeplearning4j/blob/8aa0ef12794ca40a2d00c5c80206a24a3bd6529c/nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common/src/main/java/org/nd4j/linalg/cpu/nativecpu/buffer/BaseCpuDataBuffer.java#L386
        if (!isString && dataBuffer->getDataBuffer()->getDataType() != ArrayOptions::dataType(newShapeInfoCast)) {
          errorMessage += "Data type mismatch between data buffer and shape buffer. ";
          errorMessage += "Data buffer data type: " + DataTypeUtils::asString(dataBuffer->dataBuffer()->getDataType()) + ". ";
          errorMessage += "Shape buffer data type: " + DataTypeUtils::asString(ArrayOptions::dataType(newShapeInfoCast)) + ". ";
          errorFound = true;
        }
        if (!DataTypeUtils::validDataType(dataBuffer->dataBuffer()->getDataType())) {
          errorMessage += "Invalid data type in data buffer. ";
          errorFound = true;
        }
      } else {
        errorMessage += "Data buffer is null. ";
        errorFound = true;
      }

      if (!DataTypeUtils::validDataType(ArrayOptions::dataType(newShapeInfoCast))) {
        errorMessage += "Invalid data type in shape buffer. ";
        errorFound = true;
      }
    } else if (dataBuffer->dataBuffer() != nullptr && (dataBuffer->dataBuffer()->primary() != nullptr || dataBuffer->dataBuffer()->special() != nullptr)) {
      errorMessage += "Shape Buffer at index " + std::to_string(index) + " is marked as empty but data buffer is not null! ";
      errorFound = true;
    }
  }

  if (errorFound) {
    errorMessage += "Shape info: " + ShapeUtils::shapeAsString(newShapeInfoCast) + ". ";
    errorMessage += "Data type: " + DataTypeUtils::asString(ArrayOptions::dataType(newShapeInfoCast)) + ". ";
    if (dataBuffer->dataBuffer() != nullptr) {
      errorMessage += "Data buffer: " + std::string(dataBuffer->dataBuffer()->primary() != nullptr ? "not null" : "null") + ". ";
      errorMessage += "Special buffer: " + std::string(dataBuffer->dataBuffer()->special() != nullptr ? "not null" : "null") + ". ";
    }
    errorMessage += "Elements: ";
    for(int i = 0; i < shape::shapeInfoLength(newShapeInfoCast); i++) {
      errorMessage += std::to_string(newShapeInfoCast[i]) + ", ";
    }
    errorMessage += "\n";

    THROW_EXCEPTION(errorMessage.c_str());
  }
}



void Context::setTArguments(double *arguments, int numberOfArguments) {
  _tArgs.clear();
  _tArgs.reserve(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) _tArgs.push_back(arguments[e]);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("float values set in context: ");
    for (auto d : _bArgs) {
      printf("%s\n, ", std::to_string(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setIArguments(LongType *arguments, int numberOfArguments) {
  _iArgs.clear();
  _iArgs.reserve(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) _iArgs.push_back(arguments[e]);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("int arguments set in context: ");
    for (auto d : _bArgs) {
      printf("%s\n, ", std::to_string(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setBArguments(bool *arguments, int numberOfArguments) {
  _bArgs.clear();
  _bArgs.reserve(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) _bArgs.push_back(arguments[e]);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("boolean types set in context: ");
    for (auto d : _bArgs) {
      printf("%s\n, ", std::to_string(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setCudaContext(Pointer cudaStream, Pointer reductionPointer, Pointer allocationPointer) {
#ifdef __CUDABLAS__
  _context = new LaunchContext(cudaStream, reductionPointer, allocationPointer);

  // FIXME: either pass handle from outside, or make sure outside we use the same handle
  _context->setCublasHandle(LaunchContext::defaultContext()->getCublasHandle());

  for (auto v : _fastpath_out) v->setContext(_context);

  for (auto v : _fastpath_in) v->setContext(_context);
#endif
}

void Context::allowHelpers(bool reallyAllow) { _helpersAllowed = reallyAllow; }

bool Context::helpersAllowed() { return _helpersAllowed; }

void Context::setTArguments(const std::vector<double> &tArgs) {
  for (auto t : tArgs) _tArgs.emplace_back(t);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("t argument types set in context: ");
    for (auto d : _bArgs) {
      printf("%s\n, ", std::to_string(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setIArguments(const std::vector<LongType> &iArgs) {
  for (auto i : iArgs) _iArgs.emplace_back(i);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("int argument types set in context: ");
    for (auto d : iArgs) {
      printf("%s\n, ", std::to_string(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setBArguments(const std::vector<bool> &bArgs) {
  for (auto b : bArgs) _bArgs.push_back(b);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("boolean types set in context: ");
    for (auto d : _bArgs) {
      printf("%s\n, ", std::to_string(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setShapeFunctionOverride(bool reallyOverride) { _shapeFunctionOverride = reallyOverride; }

bool Context::shapeFunctionOverride() { return _shapeFunctionOverride; }

samediff::ExecutionMode Context::executionMode() { return _execMode; }

void Context::setExecutionMode(samediff::ExecutionMode executionMode) { _execMode = executionMode; }

bool Context::isTraining() { return _execMode == samediff::ExecutionMode::MODE_TRAINING; }

bool Context::isInference() { return _execMode == samediff::ExecutionMode::MODE_INFERENCE; }

void Context::setDArguments(DataType *arguments, int numberOfArguments) {
  _dArgs.clear();
  for (int e = 0; e < numberOfArguments; e++) _dArgs.emplace_back(arguments[e]);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("data types set in context: ");
    for (auto d : _dArgs) {
      printf("%s\n, ", DataTypeUtils::asString(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::setDArguments(const std::vector<DataType> &dArgs) {
  _dArgs.clear();
  for (auto d : dArgs) _dArgs.emplace_back(d);
  if(Environment::getInstance().isDebug() || Environment::getInstance().isVerbose()) {
    printf("data types set in context: ");
    for (auto d : dArgs) {
      printf("%s\n, ", DataTypeUtils::asString(d).c_str());
    }
    fflush(stdout);
  }
}

void Context::clearFastPath() {
  _fastpath_in.clear();
  _fastpath_out.clear();


  _handles.clear();
}

void Context::setInputArrays(int numArrays,NDArray** array, bool removable) {
  for(int i = 0; i < numArrays; i++) {
    setInputArray(i,array[i],removable);
  }
}

void Context::setOutputArrays(int numArrays,NDArray** array, bool removable) {
  for(int i = 0; i < numArrays; i++) {
    setOutputArray(i,array[i],removable);
  }
}

}  // namespace graph
}  // namespace sd
