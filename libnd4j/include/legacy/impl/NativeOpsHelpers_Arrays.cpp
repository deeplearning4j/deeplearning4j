/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <helpers/ConstantTadHelper.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include "execution/Threads.h"
#include "helpers/OpTracker.h"

#include <exceptions/allocation_exception.h>
#include <fcntl.h>
#include <graph/GraphExecutioner.h>

#include <helpers/BlasHelper.h>
#include <helpers/helper_ptrmap.h>
#include <helpers/logger.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <loops/type_conversions.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/transforms.h>
#include <stdio.h>
#include <stdlib.h>
#include <types/float8.h>
#include <types/types.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>

#else
#include <helpers/mman.h>
#include <io.h>
#endif
#include <errno.h>
#include <ops/declarable/CustomOperations.h>
#include <sys/types.h>
#include <unordered_map>
#include <memory>


bool experimentalSupport = false;

// External reference to TadPack registry (defined in NativeOpsHelpers_DataBuffers.cpp)
extern std::unordered_map<sd::TadPack*, std::shared_ptr<sd::TadPack>> g_tadPackRegistry;
extern std::mutex g_tadPackMutex;

// OpaqueNDArray allocation tracking
static std::atomic<size_t> g_opaqueArrayCount{0};
static std::atomic<size_t> g_opaqueArrayBytes{0};
static std::mutex g_opaqueArrayMutex;

// InteropDataBuffer/OpaqueDataBuffer allocation tracking
static std::atomic<size_t> g_dataBufferCount{0};
static std::atomic<size_t> g_dataBufferBytes{0};
static std::mutex g_dataBufferMutex;

#include <execution/Threads.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>

#include <ops/declarable/OpRegistrator.h>
#include <ops/specials.h>
#include <system/Environment.h>
#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif
#include <array/DataType.h>
#include <array/DataTypeUtils.h>




/*
 * TypeDef:
 *     void convertTypes(Pointer *extras, DataType srcType, Pointer hX, long N, DataType dstType, Pointer hZ);
 */
void deleteNDArray(OpaqueNDArray array) {
  if (array == nullptr) {
    return;
  }

  // Track deallocation
  size_t bytes = array->lengthOf() * array->sizeOfT();
  g_opaqueArrayCount.fetch_sub(1, std::memory_order_relaxed);
  g_opaqueArrayBytes.fetch_sub(bytes, std::memory_order_relaxed);

  if(sd::Environment::getInstance().isVerbose()) {
    sd_printf("deleteNDArray: deallocating array at %p, count=%zu, total_bytes=%zu, freed_bytes=%zu\n",
              array, g_opaqueArrayCount.load(), g_opaqueArrayBytes.load(), bytes);
  }

  delete array;
}

sd::LongType getOpaqueNDArrayOffset(OpaqueNDArray array) {
  return array->offset();
}


const sd::LongType* getOpaqueNDArrayShapeInfo(OpaqueNDArray array) {
  return array->shapeInfo();
}



void* getOpaqueNDArrayBuffer(OpaqueNDArray array) {
  if(array == nullptr || array->dataBuffer() == nullptr) {
    THROW_EXCEPTION("getOpaqueNDArrayBuffer: Array or data buffer was null!");
  }
  return array->dataBuffer()->primary();
}

void* getOpaqueNDArraySpecialBuffer(OpaqueNDArray array) {
  if(array == nullptr || array->dataBuffer() == nullptr) {
    THROW_EXCEPTION("getOpaqueNDArraySpecialBuffer: Array or data buffer was null!");
  }
  return array->dataBuffer()->special();
}

sd::LongType getShapeInfoLength(OpaqueNDArray array) {
  return shape::shapeInfoLength(array->rankOf());
}

sd::LongType getOpaqueNDArrayLength(OpaqueNDArray array) {
  return array->dataBuffer()->getNumElements();
}


OpaqueNDArray createOpaqueNDArray(OpaqueDataBuffer *shapeInfo,
                                  OpaqueDataBuffer *buffer,
                                  OpaqueDataBuffer *specialBuffer,
                                  sd::LongType offset) {
  if(shapeInfo == nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: Shape info was null!");
  }

  sd::LongType* shapeInfoCast = reinterpret_cast<sd::LongType*>(shapeInfo->primary());

  // If primary() returns nullptr, the NDArray constructor will fail with undefined behavior
  // when it tries to call shape::length(nullptr) and other shape functions.
  // This check provides clear error message at the source rather than cryptic failures downstream.
  if (shapeInfoCast == nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: shapeInfo->primary() returned nullptr - shape buffer is invalid! "
                    "This indicates the Java-side DataBuffer for shape information is corrupted or deallocated.");
  }

  if(shape::isEmpty(shapeInfoCast) && buffer != nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: Shape info was empty but buffer was not null!");
  } else if(!shape::isEmpty(shapeInfoCast) && buffer == nullptr) {
    THROW_EXCEPTION("createOpaqueNDArray: Shape info was not empty but buffer was null!");
  }

  sd::NDArray* ret = new sd::NDArray(
    buffer != nullptr ? buffer->getDataBuffer() : nullptr,
    shapeInfoCast,
    sd::LaunchContext::defaultContext(),
    offset
  );

  // Track allocation
  if (ret != nullptr) {
    size_t bytes = ret->lengthOf() * ret->sizeOfT();
    g_opaqueArrayCount.fetch_add(1, std::memory_order_relaxed);
    g_opaqueArrayBytes.fetch_add(bytes, std::memory_order_relaxed);

    if(sd::Environment::getInstance().isVerbose()) {
      sd_printf("createOpaqueNDArray: allocated array at %p, count=%zu, total_bytes=%zu, this_bytes=%zu\n",
                ret, g_opaqueArrayCount.load(), g_opaqueArrayBytes.load(), bytes);
    }
  }

  return ret;
}


void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) {
  sd::DataBuffer::memcpy(target->dataBuffer(), from->dataBuffer(), targetOffset, fromOffset);
}



int contextNumInputs(void *contextPointer) {
  Context *context = (Context *) contextPointer;
  return context->width();
}

int contextNumOutputs(void *contextPointer) {
  Context *context = (Context *) contextPointer;
  return context->outputWidth();
}



int numInputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers->size();
}

int numOutputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers->size();
}

std::vector<bool> * bArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &trace->bArgs;
}

std::vector<std::string> * sArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return (&trace->sArguments);
}
std::vector<double> * tArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return (&trace->tArgs);

}

std::vector<int> * dArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  std::vector<int> *dArgs = new std::vector<int>();
  for (size_t e = 0; e < trace->dArgs.size(); e++) {
    dArgs->push_back(trace->dArgs[e]);
  }
  return dArgs;
}

std::vector<sd::LongType> * iArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &(trace->iArgs);
}

std::vector<const sd::LongType *> *inputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers;
}

std::vector<const sd::LongType *> *outputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers;
}

char *opName(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return const_cast<char *>(trace->opName->c_str());
}

void setElementThreshold(int num) {
  if (num > 0) sd::Environment::getInstance().setElementwiseThreshold(num);
}

void setTADThreshold(int num) {
  if (num > 0) sd::Environment::getInstance().setTadThreshold(num);
}


sd::Status registerGraph(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer flatBufferPointer) {
#ifdef __cpp_exceptions
  try {
    auto graph = sd::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    GraphHolder::getInstance().registerGraph(graphId, graph);

    return sd::Status::OK;
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return sd::Status::BAD_INPUT;
  }
#else
  auto graph = sd::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

  GraphHolder::getInstance().registerGraph(graphId, graph);

  return sd::Status::OK;
#endif
}

static VariablesSet *executeStoredGraphT(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer *inputBuffers,
                                         sd::Pointer *inputShapes, int *inputIndices, int numInputs) {
  auto graph = sd::graph::GraphHolder::getInstance().cloneGraph(graphId);
  auto varSpace = graph->getVariableSpace();

  std::vector<sd::NDArray *> handles;

  for (int e = 0; e < numInputs; e++) {
    auto idx = inputIndices[e];

    // we'll delete this array later, together with cloned VariableSpace
    auto array = new sd::NDArray(inputBuffers[e], reinterpret_cast<sd::LongType  *>(inputShapes[e]), nullptr, 0, 0);
    handles.emplace_back(array);

    if (varSpace->hasVariable(idx)) {
      auto var = varSpace->getVariable(idx);
      if (var->hasNDArray()) delete var->getNDArray();

      var->setNDArray(array);
    } else
      varSpace->putVariable(idx, array);
  }

  auto hZ = sd::graph::GraphExecutioner::execute(graph, varSpace);
  auto varSet = new sd::graph::VariablesSet(hZ);

  if (hZ == sd::Status::OK) {
    // pull back results, and provide them
    auto outputs = graph->fetchOutputs();
    int size = static_cast<int>(outputs->size());
    for (int e = 0; e < size; e++) {
      // we're only getting variable ID/Index from original grap. values will be taken from cloned workspace
      std::pair<int, int> varId(outputs->at(e)->id(), outputs->at(e)->index());

      auto var = varSpace->getVariable(varId);

      varSet->push_back(var->clone());
    }

    delete outputs;
  }

  delete graph;

  return varSet;
}


VariablesSet *executeStoredGraph(sd::Pointer *extraPointers, sd::LongType  graphId, sd::Pointer *inputBuffers, sd::Pointer *inputShapes,
                                 int *inputIndices, int numInputs) {
#ifdef __cpp_exceptions
  try {
    return executeStoredGraphT(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return nullptr;
  }
#else
  return executeStoredGraphT(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
#endif
}

sd::LongType  getVariablesSetSize(OpaqueVariablesSet *set) { return set->size(); }

sd::Status getVariablesSetStatus(OpaqueVariablesSet *set) { return set->status(); }

OpaqueVariable *getVariable(OpaqueVariablesSet *set, sd::LongType  i) { return set->at(i); }

int getVariableId(Variable *variable) { return variable->id(); }

int getVariableIndex(Variable *variable) { return variable->index(); }

const char *getVariableName(Variable *variable) { return variable->getName()->c_str(); }

sd::LongType  const *getVariableShape(Variable *variable) { return variable->getNDArray()->shapeInfo(); }

void *getVariableBuffer(Variable *variable) { return variable->getNDArray()->buffer(); }

sd::Status unregisterGraph(sd::Pointer *extraPointers, sd::LongType  graphId) {
#ifdef __cpp_exceptions
  try {
    GraphHolder::getInstance().dropGraphAny(graphId);

    return sd::Status::OK;
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return sd::Status::BAD_INPUT;
  }
#else
  GraphHolder::getInstance().dropGraphAny(graphId);

  return sd::Status::OK;
#endif
}

void deletePointerArray(sd::Pointer pointer) {
  sd::Pointer *ptr = reinterpret_cast<sd::Pointer *>(pointer);
  delete[] ptr;
}

void deleteCharArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<char *>(pointer);
  delete[] ptr;
}

void deleteIntArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<int *>(pointer);
  delete[] ptr;
}

void deleteLongArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<sd::LongType  *>(pointer);
  delete[] ptr;
}

void deleteVariablesSet(VariablesSet *pointer) {
  delete pointer;
}

void deleteShapeList(sd::Pointer shapeList) {
  sd::ShapeList *list = reinterpret_cast<sd::ShapeList *>(shapeList);
  delete list;
}

const char *getAllOperations() { return sd::OpTracker::getInstance().exportOperations(); }

sd::Pointer getGraphState(sd::LongType  id) { return (sd::Pointer) new GraphState(id); }

void deleteGraphState(sd::Pointer state) {
  auto stateP = reinterpret_cast<GraphState *>(state);
  delete stateP;
}

sd::Status execCustomOpWithScope_(sd::Pointer *extraPointers, sd::graph::GraphState *state, sd::LongType  opHash,
                                  sd::LongType  *scopes, int numScopes, sd::Pointer *inputBuffers,
                                  sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers,
                                  sd::Pointer *outputShapes, int numOutputs) {
  /**
   * That's basically exec, with VariableSpace provided in GraphState:
   * depending on operation (i.e. while of if), different logic executors could be used
   */

  auto graph = state->graph();
  auto varSpace = state->variableSpace();

  // Node is dynamically created, and has nothing beyond it: only inputs and outputs
  // this node has id of 0, and inputs are
  Node node(::graph::OpType_LOGIC, opHash, 0);

  // mapping inputs
  for (int e = 0; e < numInputs; e++) {
    auto buffer = inputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType  *>(inputShapes[e]);

    auto array = new sd::NDArray(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace
    varSpace->putVariable(0, e, *array);
    node.pickInput(0, e);
  }

  // mapping scopes
  for (int e = 0; e < numScopes; e++) {
    // we should check scope existence in GraphState/Graph
    int scopeId = (int)scopes[e];
    if (!state->hasScope(scopeId)) {
      return sd::Logger::logKernelFailureMsg();
    }
    node.pickInput(scopeId, 0);
  }

  auto hZ = LogicExecutor::processNode(graph, &node);
  if (hZ != sd::Status::OK) return hZ;

  // mapping outputs

  for (int e = 0; e < numOutputs; e++) {
    auto buffer = outputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType  *>(outputShapes[e]);

    sd::NDArray array(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace to the same ID
    // varSpace->putVariable(0, e, array);

    auto t = varSpace->getVariable(0, e)->getNDArray();
    array.assign(t);
  }

  // removing input variables
  for (int e = 0; e < numInputs; e++) {
    varSpace->dropVariable(0, e);
  }

  return sd::Status::OK;
}

void deleteResultWrapper(sd::Pointer ptr) {
  auto p = reinterpret_cast<ResultWrapper *>(ptr);
  delete p;
}


template <typename T>
SD_INLINE int estimateThresholdGeneric(sd::Pointer *extraPointers, sd::Pointer hX, int N, float threshold) {
  auto buffer = reinterpret_cast<T *>(hX);
  int span = (N / 6) + 8;
  // Cast the threshold to the appropriate type T
  T typedThreshold = static_cast<T>(threshold);

  auto func = PRAGMA_REDUCE_LONG {
    int64_t cnt = 0;
    PRAGMA_OMP_SIMD
    for (auto e = start; e < stop; e++) {
      auto v = sd::math::sd_abs<T,T>(buffer[e]);
      if (v >= typedThreshold) cnt++;
    }

    return cnt;
  };

  return samediff::Threads::parallel_long(
      func, LAMBDA_AL { return _old + _new; }, 0, N);
}

int estimateThreshold(sd::Pointer *extraPointers, sd::Pointer hX, sd::LongType const *hXShapeInfo, int N,
                      float threshold) {
#ifdef __cpp_exceptions
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, return estimateThresholdGeneric, (extraPointers, hX, N, threshold), SD_FLOAT_TYPES);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return 0;
  }
#else
  auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

  BUILD_SINGLE_SELECTOR(xType, return estimateThresholdGeneric, (extraPointers, hX, N, threshold), SD_FLOAT_TYPES);
#endif

  return 0;
}



void deleteTadPack(sd::TadPack *ptr) {
  if (!ptr) return;

  // The registry holds a shared_ptr<TadPack> to keep TadPacks alive while Java uses them
  // When Java is done and calls deleteTadPack, we remove it from the registry
  // This decrements the shared_ptr refcount, and if it reaches 0, the TadPack is deleted
  {
    std::lock_guard<std::mutex> lock(g_tadPackMutex);
    auto it = g_tadPackRegistry.find(ptr);
    if (it != g_tadPackRegistry.end()) {
      // Found in registry - erase it (this decrements refcount)
      g_tadPackRegistry.erase(it);
      // DON'T delete ptr manually - shared_ptr destructor will handle it when refcount reaches 0
    } else {
      // Not in registry - this might be a TadPack created without going through tadOnlyShapeInfo
      // Or it's already been removed from registry. Safe to delete directly.
      delete ptr;
    }
  }
}




OpaqueConstantDataBuffer constantBufferLong(sd::DataType dtype, sd::LongType  *data, int length) {
  return sd::ConstantHelper::getInstance().constantBuffer(sd::ConstantDescriptor(data, length), dtype);
}

OpaqueConstantDataBuffer constantBufferDouble(sd::DataType dtype, double *data, int length) {
  return sd::ConstantHelper::getInstance().constantBuffer(sd::ConstantDescriptor(data, length), dtype);
}

OpaqueConstantDataBuffer constantBuffer(sd::DataType dtype, sd::ConstantDescriptor *descriptor) {
  return sd::ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
}

sd::Pointer getConstantDataBufferPrimary(OpaqueConstantDataBuffer dbf) { return dbf->primary(); }
sd::Pointer getConstantDataBufferSpecial(OpaqueConstantDataBuffer dbf) { return dbf->special(); }
sd::LongType getConstantDataBufferLength(OpaqueConstantDataBuffer dbf) { return dbf->length(); }
sd::LongType getConstantDataBufferSizeOf(OpaqueConstantDataBuffer dbf) { return dbf->sizeOf(); }

sd::Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer dbf) { return const_cast<sd::LongType *>(dbf->primary()); }

sd::Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer dbf) { return const_cast<sd::LongType *>(dbf->special()); }

const char* getConstantShapeBufferStackTrace(OpaqueConstantShapeBuffer buffer) {
  if (buffer == nullptr) {
    return "ConstantShapeBuffer is null";
  }

  //
  // ROOT CAUSE: thread_local uses R_X86_64_GOTPC32_TLSDESC relocations which have ±2GB limit
  // When SD_GCC_FUNCTRACE is enabled, binary size exceeds 2GB → TLS relocations fail
  //
  // SOLUTION: Use regular static instead of thread_local
  // - Eliminates all TLS relocations from this function
  // - Trade-off: Not thread-safe (acceptable for debugging function)
  // - If called concurrently by multiple threads, traces may interleave (rare edge case)
  //
  // This is fundamentally different from Sessions #159-164 which tried linker workarounds
  // Those approaches CAN'T work - TLS relocations are architectural limitation
  static std::string cachedTrace;
  cachedTrace = buffer->getStackTraceAsString();

  return cachedTrace.c_str();
}

Context *createGraphContext(int nodeId) { return new Context(nodeId); }

OpaqueRandomGenerator getGraphContextRandomGenerator(Context *ptr) { return &ptr->randomGenerator(); }

void markGraphContextInplace(Context *ptr, bool reallyInplace) { ptr->markInplace(reallyInplace); }


//note here for javacpp mapping we have to use this odd type alias as a pointer
//to make the typedef work properly.
void setGraphContextInputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextInputArraysArr: Input arrays were null!");
  for (int i = 0; i < numArrays; i++) {
    if(arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextInputArraysArr: Input array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }

    OpaqueNDArray &ref = *arr[i];
    ptr->setInputArray(i, ref, false);
  }
}



void setGraphContextTArguments(Context *ptr, double *arguments, int numberOfArguments) {
  ptr->setTArguments(arguments, numberOfArguments);
}

void setGraphContextIArguments(Context *ptr, sd::LongType *arguments, int numberOfArguments) {
  ptr->setIArguments(arguments, numberOfArguments);
}

void setGraphContextBArguments(Context *ptr, bool *arguments, int numberOfArguments) {
  ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) {
  std::vector<sd::DataType> dtypes(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) dtypes[e] = sd::DataTypeUtils::fromInt(arguments[e]);

  ptr->setDArguments(dtypes);
}

void deleteGraphContext(Context *ptr) {
  delete ptr;
}

OpaqueRandomGenerator createRandomGenerator(sd::LongType rootSeed, sd::LongType nodeSeed) {
#ifdef __cpp_exceptions
  try {
    return new RandomGenerator(rootSeed, nodeSeed);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return nullptr;
  }
#else
  return new RandomGenerator(rootSeed, nodeSeed);
#endif
}

sd::LongType getRandomGeneratorRootState(OpaqueRandomGenerator ptr) { return ptr->rootState(); }

sd::LongType getRandomGeneratorNodeState(OpaqueRandomGenerator ptr) { return ptr->nodeState(); }

void setRandomGeneratorStates(OpaqueRandomGenerator ptr, sd::LongType rootSeed, sd::LongType nodeSeed) {
  ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(OpaqueRandomGenerator ptr, sd::LongType index) { return ptr->relativeInt(index); }

sd::LongType getRandomGeneratorRelativeLong(OpaqueRandomGenerator ptr, sd::LongType index) {
  return ptr->relativeLong(index);
}

int getRandomGeneratorNextInt(OpaqueRandomGenerator ptr) {
  // to nullify  _nodeState._long ^= (steps ^ 0xdeadbeef);
  // we will use step = 0xdeadbeef
  auto result = ptr->relativeInt(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

sd::LongType getRandomGeneratorNextLong(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeLong(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

float getRandomGeneratorNextFloat(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeT<float>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

double getRandomGeneratorNextDouble(OpaqueRandomGenerator ptr) {
  auto result = ptr->relativeT<double>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

void deleteRandomGenerator(OpaqueRandomGenerator ptr) { delete ptr; }


/**
 * Get the shape buffer from a
 * numpy array.
 * **Warning** this allocates memory
 * @param npyArray
 * @return
 */
sd::Pointer shapeBufferForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  auto shape = new sd::LongType[arr.shape.size()];
  for (unsigned int i = 0; i < arr.shape.size(); i++) {
    shape[i] = arr.shape[i];
  }

  auto shapeBuffer = shape::shapeBufferOfNpy(arr.shape.size(), shape, arr.fortranOrder);
  delete[] shape;
  return reinterpret_cast<sd::Pointer>(shapeBuffer);
}

/**
 *
 * @param npyArray
 * @return
 */
sd::Pointer dataPointForNumpyHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arr.data);
  return dataToPrint;
}

/**
 *
 * @param npyArray
 * @return
 */
sd::Pointer dataPointForNumpyStruct(sd::Pointer npyArrayStruct) {
  cnpy::NpyArray* arrPointer = reinterpret_cast<cnpy::NpyArray*>(npyArrayStruct);
  unsigned char* dataToPrint = reinterpret_cast<unsigned char*>(arrPointer->data);
  return reinterpret_cast<sd::Pointer>(dataToPrint);
}

/**
 *
 * @param npyArray
 * @param fromFile
 * @return
 */
sd::Pointer dataPointForNumpy(sd::Pointer npyArray) {
  char* npyArrayBuffer = reinterpret_cast<char*>(npyArray);
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(npyArrayBuffer);
  return dataPointForNumpyStruct(reinterpret_cast<sd::Pointer>(&arr));
}

/**
 * Load a numpy array from a file
 * and return it as an sd::Pointer
 * @param path
 * @return
 */
sd::Pointer numpyFromFile(std::string path) {
  char* numpyBuffer = cnpy::loadFile(path.data());
  return reinterpret_cast<sd::Pointer>(numpyBuffer);
}

////// NPZ //////

