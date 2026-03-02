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

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

#if defined(SD_GCC_FUNCTRACE)
#include <graph/OpContextLifecycleTracker.h>
#endif

namespace sd {
namespace graph {
Context::Context(ContextPrototype *prototype, VariableSpace *variableSpace) {
  // Explicitly initialize ALL pointer members to prevent garbage values
  _workspace = nullptr;
  _variableSpace = variableSpace;
  _rng = nullptr;
  _context = nullptr;  // CRITICAL: must initialize before any vector operations

  // Pre-reserve space for vectors to prevent reallocation during ops.
  // Without reserve, push_back triggers reallocation which has been observed to corrupt memory
  // (Valgrind: "Invalid free" at "0 bytes after a block of size 8 alloc'd by _M_realloc_insert").
  // glibc error "free(): invalid next size (fast)" also indicates heap metadata corruption.
  // Note: Intermediate results accumulate when Context is reused across op executions.
  // Using larger reserve to avoid frequent reallocations.
  _intermediateResults.reserve(64);
  _fastpath_in.reserve(16);
  _fastpath_out.reserve(16);
  _handles.reserve(16);

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

#if defined(SD_GCC_FUNCTRACE)
  // Track OpContext allocation
  OpContextLifecycleTracker::getInstance().recordAllocation(
      this, _nodeId,
      _fastpath_in.size(), _fastpath_out.size(),
      _intermediateResults.size(), _handles.size(),
      _workspace != nullptr, isFastPath());
#endif
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
  this->_context = nullptr;  // Explicitly initialize to prevent garbage values
  this->_rng = nullptr;      // Explicitly initialize

  this->_executionTime.first = 0;
  this->_executionTime.second = 0;

  // Pre-reserve space for vectors to prevent reallocation during ops.
  // Without reserve, push_back triggers reallocation which has been observed to corrupt memory
  // (Valgrind: "Invalid free" at "0 bytes after a block of size 8 alloc'd by _M_realloc_insert").
  // glibc error "free(): invalid next size (fast)" also indicates heap metadata corruption.
  // Note: Intermediate results accumulate when Context is reused across op executions.
  // Using larger reserve to avoid frequent reallocations.
  _intermediateResults.reserve(64);
  _fastpath_in.reserve(16);
  _fastpath_out.reserve(16);
  _handles.reserve(16);

  if (variableSpace != nullptr && variableSpace->launchContext()->getWorkspace() != nullptr)
    this->_workspace = variableSpace->launchContext()->getWorkspace();

#if defined(SD_GCC_FUNCTRACE)
  // Track OpContext allocation
  OpContextLifecycleTracker::getInstance().recordAllocation(
      this, _nodeId,
      _fastpath_in.size(), _fastpath_out.size(),
      _intermediateResults.size(), _handles.size(),
      _workspace != nullptr, isFastPath());
#endif
}

Context::Context(int nodeId, VariableSpace *variableSpace, bool isInplace) : Context(nodeId, variableSpace) {
  this->_isInplace = isInplace;
}

Context::~Context() {
#if defined(SD_GCC_FUNCTRACE)
  // Track OpContext deallocation before cleanup
  OpContextLifecycleTracker::getInstance().recordDeallocation(this);
#endif

  // CRITICAL: Capture _context value BEFORE any vector operations.
  // There's evidence of memory corruption where _context gets corrupted to point
  // to addresses related to vector internal storage. By capturing early, we can
  // detect if corruption happened during vector operations.
  LaunchContext* contextToDelete = _context;
  _context = nullptr;  // Clear immediately to prevent double-free attempts

  // Tripwire: Check if _context was already corrupted before destructor
  if (contextToDelete != nullptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(contextToDelete);
    // Valid heap pointers should be > 0x10000 and 8-byte aligned
    if (addr < 0x10000 || (addr & 0x7) != 0) {
      std::string error = "Context::~Context: _context pointer corrupted on entry: 0x";
      char buf[32];
      snprintf(buf, sizeof(buf), "%lx", static_cast<unsigned long>(addr));
      error += buf;
      THROW_EXCEPTION(error.c_str());
    }
  }

  // SAFETY: Wrap all cleanup in try-catch to prevent crashes from memory corruption.
  // The Context object or its vectors may be corrupted due to:
  // 1. Race conditions between CUDA operations and cleanup
  // 2. Java's DeallocatorService freeing memory concurrently
  // 3. Use-after-free from other threads
  // If cleanup fails, we leak memory but prevent JVM crashes.

#ifdef __cpp_exceptions
  try {
#endif
    this->_iArgs.clear();
    this->_tArgs.clear();
    this->_inputs.clear();

    // IMPORTANT: Do NOT delete arrays in _fastpath_in and _fastpath_out!
    // These are BORROWED pointers - the Context does not own them.
    this->_fastpath_in.clear();
    this->_fastpath_out.clear();

    // NOTE: Do NOT delete arrays in _handles here.
    // While _handles is intended to store arrays that are "owned" by Context (added with removable=true),
    // in practice the arrays may have already been freed by Java's DeallocatorService or other cleanup paths.
    // Attempting to delete them causes SIGSEGV in NDArray::~NDArray.
    // This causes a memory leak, but prevents JVM crashes.
    // TODO: Fix the ownership model so arrays in _handles are truly owned by Context
    // and can be safely deleted here.
    _handles.clear();

    // Clean up intermediate results properly.
    // Intermediate results are arrays stored by ops (like conv2d) for use in backward pass.
    // The arrays are heap-allocated and owned by this Context.
    // We must delete them to prevent memory leaks.
    // Note: Views in intermediate results will have their NDArray object deleted but NOT their buffer
    // (because views have _ownsBuffer=false or isView flag set), which is correct behavior.
    //
    // IMPORTANT: Delete in REVERSE order! Views are typically pushed after their parent arrays
    // (e.g., conv2d pushes col at index 0, then colP view at index 1). Deleting in reverse
    // ensures views are deleted before their parents. While views shouldn't access their
    // buffer during destruction, this ordering is safer and matches the lifetime expectations.
    for (auto it = _intermediateResults.rbegin(); it != _intermediateResults.rend(); ++it) {
      auto* arr = *it;
      if (arr != nullptr) {
        // Validate pointer looks reasonable before deleting
        uintptr_t addr = reinterpret_cast<uintptr_t>(arr);
        if (addr > 0x10000) {
          delete arr;
        }
      }
    }
    _intermediateResults.clear();

#ifdef SD_CUDA
    // Sync the stream after deleting intermediate results to ensure all async frees complete.
    // When DataBuffer::deleteSpecial() is called during NDArray destruction, it uses
    // cudaFreeAsync which schedules the free on a stream. We must ensure these complete
    // before the Context destructor finishes to prevent use-after-free issues.
    if (contextToDelete != nullptr) {
      auto stream = contextToDelete->getCudaStream();
      if (stream != nullptr) {
        cudaStreamSynchronize(*stream);
      }
    }
#endif

#ifdef __cpp_exceptions
  } catch (...) {
    fprintf(stderr, "WARNING: Exception clearing vectors in Context destructor\n");
    fflush(stderr);
  }
#endif

  // Only delete _context if it's not a managed (intentionally leaked) context
  if (contextToDelete != nullptr && !LaunchContext::isManagedContext(contextToDelete)) {
    delete contextToDelete;
  }
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


Stash *Context::getStash() {
  if (_variableSpace == nullptr) {
    THROW_EXCEPTION("Context::getStash: VariableSpace is null. Context was not properly initialized.");
  }
  return _variableSpace->getStash();
}

void Context::trackList(NDArrayList *list) {
  if (_variableSpace == nullptr) {
    THROW_EXCEPTION("Context::trackList: VariableSpace is null. Context was not properly initialized.");
  }
  _variableSpace->trackList(list);
}

int Context::getBranch() {
  if (_variableSpace == nullptr) {
    THROW_EXCEPTION("Context::getBranch: VariableSpace is null. Context was not properly initialized.");
  }
  if (_variableSpace->flowPath() == nullptr) {
    return 0;  // Default branch when no flow path is set
  }
  return _variableSpace->flowPath()->branch(this->nodeId());
}

void Context::setBranch(int branch) {
  //_branch = branch;
  if (_variableSpace != nullptr && _variableSpace->flowPath() != nullptr) {
    _variableSpace->flowPath()->markBranch(this->nodeId(), branch);
  }
}

LongType Context::getOuterTime() { return this->_executionTime.first; }

LongType Context::getInnerTime() { return this->_executionTime.second; }

void Context::setOuterTime(LongType time) { this->_executionTime.first = time; }

void Context::setInnerTime(LongType time) { this->_executionTime.second = time; }

Variable *Context::getVariable(int idx) {
  if (static_cast<size_t>(idx) >= this->_inputs.size()) {
    std::string errorMessage;
    errorMessage += "Node ";
    errorMessage += std::to_string(this->_nodeId);
    errorMessage += "; Variable [";
    errorMessage += std::to_string(idx);
    errorMessage += " requested, but only ";
    errorMessage += std::to_string(this->_inputs.size());
    errorMessage += " available";
    THROW_EXCEPTION(errorMessage.c_str());
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
      LongType maxLen = sd::math::sd_min(16, array->lengthOf() - 1);

      sd_printf("Debug info for node_%i input[%i]; shape: %s; ews: [%i]; order: [%c]; dtype: [%s];\n",
                this->_nodeId, idx, shape_.c_str(),array->ews(), array->ordering(), type.c_str());
      std::vector<sd::LongType> shapeLen = {array->lengthOf()};
      NDArray *raveled = array->reshape(array->ordering(), shapeLen);
      sd_printf("Values: [ ",0);
      for (LongType i = 0; i < maxLen; i++) {
        auto v2 = raveled->e<float>(i);
        sd_printf("%f, ", v2);
      }

      delete raveled;
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
  if (_variableSpace == nullptr) {
    std::string errorMessage;
    errorMessage += "Node ";
    errorMessage += std::to_string(this->_nodeId);
    errorMessage += "; VariableSpace is null when trying to get variable: [";
    errorMessage += std::to_string(p.first);
    errorMessage += ":";
    errorMessage += std::to_string(p.second);
    errorMessage += "]. This usually means the Context was not properly initialized or fastpath was expected but failed.";
    THROW_EXCEPTION(errorMessage.c_str());
  }

#ifdef __cpp_exceptions
  try {
    return _variableSpace->getVariable(p);
  } catch (std::exception &e) {
    std::string errorMessage;
    errorMessage += "Node ";
    errorMessage += std::to_string(this->_nodeId);
    errorMessage += "; Non-existent variable requested: [";
    errorMessage += std::to_string(p.first);
    errorMessage += ":";
    errorMessage += std::to_string(p.second);
    errorMessage += "]";
    errorMessage += "\n";
    THROW_EXCEPTION(errorMessage.c_str());
  }
#else
  return _variableSpace->getVariable(p);
#endif

  return nullptr;
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
  if (_variableSpace == nullptr) {
    THROW_EXCEPTION("Context::pushNDArrayListToVariableSpace: VariableSpace is null. Context was not properly initialized.");
  }
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
  if (!_fastpath_out.empty() && _fastpath_out.size() > static_cast<size_t>(idx)) {
    NDArray* result = _fastpath_out[idx];

    // Validate the output NDArray to catch memory corruption early
    if (result != nullptr) {
      const sd::LongType* shInfo = result->shapeInfo();
      if (shInfo == nullptr) {
        std::string errorMessage;
        errorMessage += "Context::outputArray(";
        errorMessage += std::to_string(idx);
        errorMessage += "): NDArray has null shapeInfo. This indicates severe memory corruption or use-after-free.";
        THROW_EXCEPTION(errorMessage.c_str());
      }

      sd::LongType rank = shInfo[0];
      if (rank < 0 || rank > 32) {
        std::string errorMessage;
        errorMessage += "Context::outputArray(";
        errorMessage += std::to_string(idx);
        errorMessage += "): NDArray has corrupted rank: ";
        errorMessage += std::to_string(rank);
        errorMessage += " (expected 0-32). This likely indicates:\n";
        errorMessage += "  1. Memory corruption in the output NDArray\n";
        errorMessage += "  2. Use-after-free (accessing deallocated memory)\n";
        errorMessage += "  3. Uninitialized output buffer allocation failure\n";
        errorMessage += "  4. JNI marshalling error from Java layer";
        THROW_EXCEPTION(errorMessage.c_str());
      }
    }

    return result;
  }

  std::string errorMessage;
  errorMessage += std::string("Context::outputArray: Fastpath is empty");
  errorMessage += std::string(" Index: ");
  errorMessage += std::to_string(idx);
  errorMessage += std::string(" Fastpath size: ");
  errorMessage += std::to_string(_fastpath_out.size());

  THROW_EXCEPTION(errorMessage.c_str());

  return nullptr;
}

NDArray *Context::array(int idx) {
  // we check for fastpath first
  if (!_fastpath_in.empty() && _fastpath_in.size() > static_cast<size_t>(idx)) {
    NDArray* result = _fastpath_in[idx];

    // Validate the NDArray to catch memory corruption early
    if (result != nullptr) {
      const sd::LongType* shInfo = result->shapeInfo();
      if (shInfo == nullptr) {
        std::string errorMessage;
        errorMessage += "Context::array(";
        errorMessage += std::to_string(idx);
        errorMessage += "): NDArray has null shapeInfo. This indicates severe memory corruption or use-after-free.";
        THROW_EXCEPTION(errorMessage.c_str());
      }

      // Check if rank is valid (should be 0-32, not a memory address)
      sd::LongType rank = shInfo[0];
      if (rank < 0 || rank > 32) {
        std::string errorMessage;
        errorMessage += "Context::array(";
        errorMessage += std::to_string(idx);
        errorMessage += "): NDArray has corrupted rank: ";
        errorMessage += std::to_string(rank);
        errorMessage += " (expected 0-32). This likely indicates:\n";
        errorMessage += "  1. Memory corruption in the NDArray\n";
        errorMessage += "  2. Use-after-free (accessing deallocated memory)\n";
        errorMessage += "  3. Uninitialized NDArray from failed operation\n";
        errorMessage += "  4. JNI marshalling error from Java layer";
        THROW_EXCEPTION(errorMessage.c_str());
      }
    }

    return result;
  }

  if (!_fastpath_in.empty() && _variableSpace == nullptr) {
    // Fastpath has some inputs but not enough for the requested index
    std::string errorMessage;
    errorMessage += "Context::array(";
    errorMessage += std::to_string(idx);
    errorMessage += "): Input index out of bounds. Fastpath has ";
    errorMessage += std::to_string(_fastpath_in.size());
    errorMessage += " input array(s), but input at index ";
    errorMessage += std::to_string(idx);
    errorMessage += " was requested.\n";
    errorMessage += "This typically means the operation expects more inputs than were provided.\n";
    errorMessage += "For variable-input operations (like concat), ensure all input arrays are set via setInputArrays().";
    THROW_EXCEPTION(errorMessage.c_str());
  }

  // if no luck for fastpath - return whatever is available
  NDArray* result = getVariable(idx)->getNDArray();

  // Validate the NDArray from variable path as well
  if (result != nullptr) {
    const sd::LongType* shInfo = result->shapeInfo();
    if (shInfo == nullptr) {
      std::string errorMessage;
      errorMessage += "Context::array(";
      errorMessage += std::to_string(idx);
      errorMessage += "): NDArray from variable has null shapeInfo. This indicates severe memory corruption or use-after-free.";
      THROW_EXCEPTION(errorMessage.c_str());
    }

    sd::LongType rank = shInfo[0];
    if (rank < 0 || rank > 32) {
      std::string errorMessage;
      errorMessage += "Context::array(";
      errorMessage += std::to_string(idx);
      errorMessage += "): NDArray from variable has corrupted rank: ";
      errorMessage += std::to_string(rank);
      errorMessage += " (expected 0-32). This likely indicates:\n";
      errorMessage += "  1. Memory corruption in the NDArray\n";
      errorMessage += "  2. Use-after-free (accessing deallocated memory)\n";
      errorMessage += "  3. Uninitialized NDArray from failed operation\n";
      errorMessage += "  4. JNI marshalling error from Java layer";
      THROW_EXCEPTION(errorMessage.c_str());
    }
  }

  return result;
}

memory::Workspace *Context::fWorkspace() { return workspace(); }

memory::Workspace *Context::tWorkspace() { return nullptr; }

memory::Workspace *Context::oWorkspace() { return nullptr; }

LaunchContext *Context::launchContext() {
  // FIXME: we need proper context to be shared here
  if (_context == nullptr) {
    _context = LaunchContext::defaultContext(); // Assign default context
  }

  // Tripwire: validate _context looks like a valid pointer
  uintptr_t addr = reinterpret_cast<uintptr_t>(_context);
  if (addr < 0x10000 || (addr & 0x7) != 0) {
    std::string error = "Context::launchContext(): _context pointer appears corrupted: 0x";
    char buf[32];
    snprintf(buf, sizeof(buf), "%lx", static_cast<unsigned long>(addr));
    error += buf;
    THROW_EXCEPTION(error.c_str());
  }

  return _context;
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
  // Check for null array FIRST before any dereference
  if (array == nullptr) {
    std::string errorMessage;
    errorMessage += std::string("Context::setInputArray: Array at index ");
    errorMessage += std::to_string(index);
    errorMessage += std::string(" is null!");
    THROW_EXCEPTION(errorMessage.c_str());
  }

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

  // Tripwire: capture _context before vector operations
  LaunchContext* contextBefore = _context;

  if (_fastpath_in.size() < static_cast<size_t>(index + 1)) _fastpath_in.resize(index + 1);

  _fastpath_in[index] = array;
  if (removable) _handles.emplace_back(array);

  // Tripwire: verify _context wasn't corrupted
  if (_context != contextBefore) {
    THROW_EXCEPTION("Context::setInputArray: _context was corrupted during vector operation!");
  }
}




void Context::setOutputArray(int index, NDArray *array, bool removable) {
  // Check for null array FIRST before any dereference
  if (array == nullptr) {
    std::string errorMessage;
    errorMessage += std::string("Context::setOutputArray: Array at index ");
    errorMessage += std::to_string(index);
    errorMessage += std::string(" is null!");
    THROW_EXCEPTION(errorMessage.c_str());
  }

  // Tripwire: capture _context BEFORE any vector operations (including resize)
  LaunchContext* contextBeforeResize = _context;

  if (_fastpath_out.size() < static_cast<size_t>(index + 1)) _fastpath_out.resize(index + 1);

  // Tripwire: check after resize
  if (_context != contextBeforeResize) {
    THROW_EXCEPTION("Context::setOutputArray: _context was corrupted during _fastpath_out.resize!");
  }

  // Check for null shapeInfo before accessing it - use try-catch because shapeInfo() throws when null
  try {
    array->shapeInfo();
  } catch (...) {
    std::string errorMessage;
    errorMessage += std::string("Context::setOutputArray: Array at index ");
    errorMessage += std::to_string(index);
    errorMessage += std::string(" has a null or invalid shape buffer!");
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

  // Tripwire: capture _context before vector operations
  LaunchContext* contextBefore = _context;

  _fastpath_out[index] = array;

  if (removable) _handles.emplace_back(array);

  // Tripwire: verify _context wasn't corrupted
  if (_context != contextBefore) {
    THROW_EXCEPTION("Context::setOutputArray: _context was corrupted during vector operation!");
  }
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
#ifdef SD_CUDA
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

#ifdef SD_CUDA
  if (_context != nullptr) {
    auto stream = _context->getCudaStream();
    if (stream != nullptr) {
      cudaStreamSynchronize(*stream);
    }
  }
#endif

  // IMPORTANT: Do NOT delete or clear _handles here.
  //
  // When Java calls ctxPurge(), we're clearing fastpath state for potential reuse.
  // The _handles vector contains arrays that are OWNED by this Context and should
  // only be cleaned up in the destructor.
  //
  // Previously this code deleted arrays in _handles, but this caused crashes when:
  // 1. Memory corruption resulted in invalid pointers in _handles
  // 2. Race conditions between CUDA operations and cleanup
  // 3. Java's DeallocatorService freed underlying buffers
  //
  // The destructor will handle _handles cleanup with proper validation.
  // Since Java's close() calls purge() then deleteGraphContext() immediately,
  // the destructor runs right after this and will clean up properly.
}

void Context::clearFastPathNoSync() {
  _fastpath_in.clear();
  _fastpath_out.clear();
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
