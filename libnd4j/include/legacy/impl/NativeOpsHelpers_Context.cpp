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
#include <ops/declarable/OpExecutionLogger.h>

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


extern bool experimentalSupport; // Defined in NativeOpsHelpers_Arrays.cpp

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
OpaqueNDArray getOutputArrayNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return nullptr;
  return ptr->outputArray(idx);
}


OpaqueNDArray getInputArrayNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return nullptr;
  return ptr->array(idx);
}


sd::LongType dataTypeNativeAt(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0;
  return static_cast<sd::LongType>(ptr->dataType(idx));

}


bool bArgAtNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return false;
  return ptr->getBArguments()->at(idx);

}

sd::LongType iArgumentAtNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0;
  return ptr->getIArguments()->at(idx);

}

sd::LongType numDNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numD();
}

sd::LongType numBNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numB();
}

sd::LongType numOutputsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->outputWidth();
}
sd::LongType numInputsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->width();
}

double tArgumentNative(OpaqueContext* ptr, int idx) {
  if(ptr == nullptr)
    return 0.0;
  return ptr->getTArguments()->at(idx);
}

sd::LongType numTArgumentsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numT();
}

sd::LongType numIArgumentsNative(OpaqueContext* ptr) {
  if(ptr == nullptr)
    return 0;
  return ptr->numI();
}




void setGraphContextOutputArray(OpaqueContext* ptr, int index,OpaqueNDArray arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextOutputArray: Input arrays were null!");

  ptr->setOutputArray(index,arr,false);


}

void setGraphContextInputArray(OpaqueContext* ptr,int index,OpaqueNDArray arr) {
  if(arr == nullptr)
    THROW_EXCEPTION("setGraphContextInputArray: Input arrays were null!");

  ptr->setInputArray(index, arr, false);

}

// NOTE ABOUT SIGNATURE AND JAVACPP MAPPING
// ----------------------------------------
// OpaqueNDArrayArr represents `NDArray**` (a pointer to an array of NDArray*).
//
// Earlier versions of this function used the signature:
//   void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays, OpaqueNDArrayArr* arr)
// which treated the argument as `NDArray***`. That required double‑dereferencing
// (e.g. `(*arr)[i]`) and did not match how JavaCPP passes the native pointer.
//
// In the JavaCPP mapping, the Java side already passes an `NDArray**` directly for
// this parameter. Using `OpaqueNDArrayArr*` added an extra level of indirection,
// so the native code tried to dereference one level too many, leading to invalid
// pointers and hard‑to‑debug crashes.
//
// The corrected signature below:
//   void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays, OpaqueNDArrayArr arr)
// matches the JavaCPP mapping exactly: `arr` is already an `NDArray**`, so
// `arr[i]` yields the i‑th `NDArray*` without any extra dereference.
void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays, OpaqueNDArrayArr arr) {
  if (arr == nullptr) THROW_EXCEPTION("setGraphContextOutputArraysArr: Output arrays were null!");
  if (ptr == nullptr) THROW_EXCEPTION("setGraphContextOutputArraysArr: Context was null!");

  for (int i = 0; i < numArrays; i++) {
    if (arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextOutputArraysArr: Output array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }

    ptr->setOutputArray(i, arr[i], false);
  }
}


sd::LongType getOpaqueNDArrayLeakCount() {
  return static_cast<sd::LongType>(g_opaqueArrayCount.load(std::memory_order_relaxed));
}

sd::LongType getOpaqueNDArrayLeakBytes() {
  return static_cast<sd::LongType>(g_opaqueArrayBytes.load(std::memory_order_relaxed));
}



sd::Pointer createUtf8String(sd::Pointer *extraPointers, const char *string, int length) {
  auto u = new sd::utf8string(string, length);
  return reinterpret_cast<sd::Pointer>(u);
}

sd::LongType getUtf8StringLength(sd::Pointer *extraPointers, sd::Pointer ptr) {
  return reinterpret_cast<sd::utf8string *>(ptr)->_length;
}
char *getUtf8StringBuffer(sd::Pointer *extraPointers, sd::Pointer ptr) {
  return reinterpret_cast<sd::utf8string *>(ptr)->_buffer;
}

void deleteUtf8String(sd::Pointer *extraPointers, sd::Pointer ptr) { delete (reinterpret_cast<sd::utf8string *>(ptr)); }

int dataTypeFromNpyHeader(void *header) { return (int)cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header)); }



OpaqueConstantShapeBuffer shapeBufferEx(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                        char order,
                                        sd::LongType ews, sd::LongType extras) {
#ifdef __cpp_exceptions
    auto desc = sd::ShapeBuilders::createShapeInfo(dtype, order,rank, shape, strides,nullptr, extras);
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    delete[] desc;
    return buffer;
#else
    auto desc = sd::ShapeBuilders::createShapeInfo(dtype, order,rank, shape, strides,nullptr, extras);
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    delete[] desc;
    return buffer;
#endif
}

void inspectArray(sd::Pointer *extraPointers, sd::Pointer buffer, sd::LongType *shapeInfo, sd::Pointer specialBuffer,
                  sd::LongType *specialShapeInfo, sd::Pointer debugInfo) {
#ifdef __cpp_exceptions
  try {
    auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
    sd::NDArray array(buffer, shapeInfo, nullptr, 0, 0);
    sd::DebugHelper::retrieveDebugStatistics(p, &array);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    THROW_EXCEPTION(e.what());
  }
#else
  auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
  sd::NDArray array(buffer, shapeInfo, nullptr, 0, 0);
  sd::DebugHelper::retrieveDebugStatistics(p, &array);
#endif

}



void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) {
  // Cache owns all ConstantShapeBuffer objects - JNI should not delete them
  // This function is a no-op now
}

void deleteConstantDataBuffer(OpaqueConstantDataBuffer *ptr) {
  if (ptr != nullptr && *ptr != nullptr) {
    delete *ptr;
  }
}

OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(sd::LongType *shapeInfo) {
#ifdef __cpp_exceptions
  try {
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
    return buffer;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
#else
  auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
  return buffer;
#endif

  return nullptr;
}

sd::LongType *mmapFile(sd::Pointer *extraPointers, const char *fileName, sd::LongType length) {
  auto hZ = new sd::LongType[2];
  sd::LongType ptr = 0;
  errno = 0;
#ifdef __cpp_exceptions
  try {
#if defined(_WIN32) || defined(_WIN64)
    _mmap(hZ, static_cast<size_t>(length), fileName);
    _mmap(hZ, static_cast<size_t>(length), fileName);
#else
    int fd = open(fileName, O_RDWR, 0);  // checking for failed fopen
    if (fd < 0) {
      sd_printf("Errno: %i\n", errno);
      THROW_EXCEPTION("Failed to open file for MMAP");
    }

    void *ptr2 = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0);
    if (ptr2 == MAP_FAILED) {
      sd_printf("Errno: %i\n", errno);
      THROW_EXCEPTION("Failed to mmap file");
    }
    hZ[0] = (sd::LongType)ptr2;
    hZ[1] = fd;

#endif

    return hZ;
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    THROW_EXCEPTION(e.what());
  }
#else
#if defined(_WIN32) || defined(_WIN64)
  _mmap(hZ, static_cast<size_t>(length), fileName);
  _mmap(hZ, static_cast<size_t>(length), fileName);
#else
  int fd = open(fileName, O_RDWR, 0);  // checking for failed fopen
  if (fd < 0) {
    sd_printf("Errno: %i\n", errno);
    safeSetErrorContext(1, "Failed to open file for MMAP");
    return nullptr;
  }

  void *ptr2 = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0);
  if (ptr2 == MAP_FAILED) {
    sd_printf("Errno: %i\n", errno);
    safeSetErrorContext(1, "Failed to mmap file");
    return nullptr;
  }
  hZ[0] = (sd::LongType)ptr2;
  hZ[1] = fd;

#endif
  return hZ;
#endif

  return nullptr;
}
void munmapFile(sd::Pointer *extraPointers, sd::LongType  *ptrMap, sd::LongType  length) {}

ResultWrapper *executeFlatGraph(sd::Pointer *extraPointers, sd::Pointer flatBufferPointer) {
#ifdef __cpp_exceptions
  try {
    return sd::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
    return nullptr;
  }
#else
  return sd::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
#endif
}

sd::LongType  getResultWrapperSize(ResultWrapper *ptr) { return ptr->size(); }
sd::Pointer getResultWrapperPointer(ResultWrapper *ptr) { return ptr->pointer(); }

const char *getAllCustomOps() { return sd::ops::OpRegistrator::getInstance().getAllCustomOperations(); }

OpaqueShapeList *calculateOutputShapes2(sd::Pointer *extraPointers, sd::LongType hash, OpaqueContext *context) {
#ifdef __cpp_exceptions
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

#if defined(SD_GCC_FUNCTRACE)
    // Set op name BEFORE calculateOutputShape so shape allocations are tagged
    if (op->getOpName() != nullptr) {
        sd::ops::OpExecutionLogger::setCurrentOpName(*op->getOpName());
    }
#endif

    sd::ShapeList inShapes;

    for (size_t e = 0; e < context->width(); e++) {
      if (context->array(e) == nullptr) {
        std::string errorMessage = "Input array at index " + std::to_string(e) + " was null!";
#if defined(SD_GCC_FUNCTRACE)
        sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif
        THROW_EXCEPTION(errorMessage.c_str());
      }
      inShapes.push_back(context->array(e)->shapeInfo());
    }

    auto shapeList = op->calculateOutputShape(&inShapes, *context);

#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif

    return shapeList;
  } catch (std::exception &e) {
#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif
    safeSetErrorContext(1, e.what());
    return nullptr;
  }
#else
  auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

#if defined(SD_GCC_FUNCTRACE)
  // Set op name BEFORE calculateOutputShape so shape allocations are tagged
  if (op->getOpName() != nullptr) {
      sd::ops::OpExecutionLogger::setCurrentOpName(*op->getOpName());
  }
#endif

  sd::ShapeList inShapes;

  for (size_t e = 0; e < context->width(); e++) {
    if (context->array(e) == nullptr) {
      std::string errorMessage = "Input array at index " + std::to_string(e) + " was null!";
#if defined(SD_GCC_FUNCTRACE)
      sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif
      safeSetErrorContext(1, errorMessage.c_str());
      return nullptr;
    }
    inShapes.push_back(context->array(e)->shapeInfo());
  }

  auto shapeList = op->calculateOutputShape(&inShapes, *context);

#if defined(SD_GCC_FUNCTRACE)
  sd::ops::OpExecutionLogger::clearCurrentOpName();
#endif

  return shapeList;
#endif
}

bool checkOpaqueNDArrayElementsNull(OpaqueNDArrayArr elements,int numElements) {
  for (int i = 0; i < numElements; i++) {
    if (elements[i] == nullptr) return true;
  }
  return false;
}


sd::LongType  getShapeListSize(sd::ShapeList *list) { return list->size(); }

sd::LongType  const *getShape(sd::ShapeList *list, sd::LongType  i) { return list->at(i); }




// Function to execute a custom operation
sd::Status execCustomOp(sd::Pointer *extraPointers, sd::LongType  hash, OpaqueNDArrayArr inputs, int numInputs,
                        OpaqueNDArrayArr outputs, int numOutputs, double *tArgs, int numTArgs,
                        sd::LongType  *iArgs, int numIArgs, bool *bArgs, int numBArgs, bool isInplace) {
#ifdef __cpp_exceptions
  try {
    // Convert NDArray** inputs and outputs to std::vector<NDArray*>
    const std::vector<sd::NDArray*> inputVec(inputs, inputs + numInputs);
    const std::vector<sd::NDArray*> outputVec(outputs, outputs + numOutputs);
    const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
    const std::vector<sd::LongType > iArgsVec(iArgs, iArgs + numIArgs);
    const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

    // Retrieve the operation based on the hash
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    if (op == nullptr) {
      THROW_EXCEPTION("Operation not found for the given hash.");
    }

    // Execute the custom operation
    return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec, {}, isInplace);
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages
    safeSetErrorContext(1, e.what());
    return sd::Status::KERNEL_FAILURE;
  }
#else
  // Convert NDArray** inputs and outputs to std::vector<NDArray*>
  const std::vector<sd::NDArray*> inputVec(inputs, inputs + numInputs);
  const std::vector<sd::NDArray*> outputVec(outputs, outputs + numOutputs);
  const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
  const std::vector<sd::LongType > iArgsVec(iArgs, iArgs + numIArgs);
  const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

  // Retrieve the operation based on the hash
  auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
  if (op == nullptr) {
    safeSetErrorContext(1, "Operation not found for the given hash.");
    return sd::Status::KERNEL_FAILURE;
  }

  // Execute the custom operation
  return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec, {}, isInplace);
#endif
}

void toggleOpTrace(bool opTrace) { sd::ops::OpRegistrator::getInstance().toggleTraceOps(opTrace);
}

void purgeOpTrace() { sd::ops::OpRegistrator::getInstance().purgeOpExecs();
}




void printOpTrace() {
  auto execTrace = *sd::ops::OpRegistrator::getInstance().execTrace();
  for(size_t i = 0; i < execTrace.size(); i++) {
    auto curr = execTrace[i];
    if(curr->opName != nullptr) {
      sd_printf("Op name: %s\n", curr->opName->c_str());
    }
    sd_printf(" Input buffers:\n",0);
    if(curr->inputShapeBuffers == nullptr || curr->inputShapeBuffers->size() == 0) {
      sd_printf("No input buffers\n",0);
      continue;
    } else {
      auto currInputShapeBuffers = *(curr->inputShapeBuffers);
      for(size_t j = 0; j < currInputShapeBuffers.size(); j++) {
        auto buff = currInputShapeBuffers[j];
        shape::printShapeInfo(buff);
        sd_printf("\n",0);
      }
    }

    if(curr->outputShapeBuffers == nullptr || curr->outputShapeBuffers->size() == 0) {
      sd_printf("No output buffers\n",0);
      continue;
    } else {
      auto currOutputShapeBuffers = *(curr->outputShapeBuffers);
      for(size_t j = 0; j < curr->outputShapeBuffers->size(); j++) {
        shape::printShapeInfo(currOutputShapeBuffers[j]);
        sd_printf("\n",0);
      }

    }


  }

}


std::vector<ExecTrace*> * listOpTraces() {
  return sd::ops::OpRegistrator::getInstance().execTrace();
}

