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
// Created by agibsonccc on 2/21/16.
//

#define __STDC_CONSTANT_MACROS

#include <array/NDArray.h>
#include <exceptions/allocation_exception.h>
#include <fcntl.h>
#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
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
#include <system/pairwise_util.h>
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
char *name;
bool nameSet = false;

bool experimentalSupport = false;
#include <execution/Threads.h>
#include <graph/Context.h>
#include <graph/ResultWrapper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/TAD.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/specials.h>
#include <system/Environment.h>
#ifdef CPU_FEATURES
#include <cpuinfo_x86.h>
#endif

#include <ops/declarable/OpRegistrator.h>

using namespace sd;

//note we only include this if we're running gcc linux
//and should not be enabled in default builds.
#if defined(SD_GCC_FUNCTRACE)
#include <cxxabi.h>  // needed  __cxa_demangle
#include <dlfcn.h>   // needed for dladdr

#include "exceptions/backward.hpp"




//note this is outside extern C. This is fine.


#endif




int contextNumInputs(void *contextPointer) {
  graph::Context *context = (graph::Context *) contextPointer;
  return context->width();
}

int contextNumOutputs(void *contextPointer) {
  graph::Context *context = (graph::Context *) contextPointer;
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
std::vector<LongType> * iArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return &(trace->iArgs);
}
std::vector<const LongType *> *inputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers;
}
std::vector<const LongType *> *outputShapeBuffers(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->outputShapeBuffers;
}

std::vector<int> * dArgs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  std::vector<int> *dArgs = new std::vector<int>();
        for (int e = 0; e < trace->dArgs.size(); e++) {
        dArgs->push_back(trace->dArgs[e]);
        }
  return dArgs;
}
char *opName(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return const_cast<char *>(trace->opName->c_str());
}

void setElementThreshold(int num) {
  if (num > 0) Environment::getInstance().setElementwiseThreshold(num);
}

void setTADThreshold(int num) {
  if (num > 0) Environment::getInstance().setTadThreshold(num);
}


void printOpTrace() {
  auto execTrace = *ops::OpRegistrator::getInstance().execTrace();
  for(int i = 0; i < execTrace.size(); i++) {
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
      for(int j = 0; j < currInputShapeBuffers.size(); j++) {
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
      for(int j = 0; j < curr->outputShapeBuffers->size(); j++) {
        shape::printShapeInfo(currOutputShapeBuffers[j]);
        sd_printf("\n",0);
      }

    }


  }

}




std::vector<ExecTrace*> * listOpTraces() {
  return ops::OpRegistrator::getInstance().execTrace();
}

void toggleOpTrace(bool opTrace) {
  ops::OpRegistrator::getInstance().toggleTraceOps(opTrace);
}

void purgeOpTrace() {
  ops::OpRegistrator::getInstance().purgeOpExecs();
}

void dbPrintAllocationTrace(OpaqueDataBuffer *db) {
  db->printDbAllocationTrace();
}


void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) {
  OpaqueDataBuffer *copyFrom = dbCreateView(from, n);
  OpaqueDataBuffer *targetView = dbCreateView(target, n);
  DataBuffer *targetBuf = copyFrom->dataBuffer();
  DataBuffer *srcBuf = targetView->dataBuffer();
  DataBuffer::memcpy(targetBuf, srcBuf, 0, 0);
}

OpaqueNDArray createOpaqueNDArray(OpaqueDataBuffer shapeInfo,
                                  OpaqueDataBuffer buffer,
                                  OpaqueDataBuffer specialBuffer,
                                  sd::LongType offset) {
  sd::LongType* shapeInfoCast = reinterpret_cast<sd::LongType*>(shapeInfo.primary());
  sd::NDArray* ret = new sd::NDArray(buffer.getDataBuffer(),
                                     shapeInfoCast,
                                     LaunchContext::defaultContext(),
                                     offset);
  return ret;
}

void deleteNDArray(OpaqueNDArray array) {
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

OpaqueConstantShapeBuffer shapeBuffer(int rank, LongType *shape, LongType *strides, DataType dtype,
                                       char order, LongType ews, bool empty) {
  return shapeBufferEx(rank, shape, strides, dtype, order, ews, empty ? ARRAY_EMPTY : 0);
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 */
void execIndexReduceScalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                           const LongType *hXShapeInfo, const LongType *dXShapeInfo, void *extraParams,
                           OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                               extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execIndexReduce(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                     const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const LongType *hDimensionShape, const LongType *dDimensionShape) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    auto hz = reinterpret_cast<LongType *>(dbZ != nullptr ? dbZ->primary() : nullptr);

    NativeOpExecutioner::execIndexReduce(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                         extraParams, hz, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, dimension,
                                         dimensionLength, hTADShapeInfo, hTADOffsets);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execBroadcast(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                   const LongType *dXShapeInfo, OpaqueDataBuffer *dbY, const LongType *hYShapeInfo,
                   const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                   const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension, const LongType *hDimensionShape,
                   const LongType *dDimensionShape) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    auto dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    auto tadPackZ = ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, dimension, dimensionLength);
    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NativeOpExecutioner::execBroadcast(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                       dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                       hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, dimension, dimensionLength,
                                       hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execBroadcastBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                       const LongType *dXShapeInfo, OpaqueDataBuffer *dbY, const LongType *hYShapeInfo,
                       const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                       const LongType *dZShapeInfo, void *extraParams, OpaqueDataBuffer *dbDimension,
                       const LongType *hDimensionShape, const LongType *dDimensionShape) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    auto tadPackZ = ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, dimension, dimensionLength);

    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NativeOpExecutioner::execBroadcastBool(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                           dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                           hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams, dimension,
                                           dimensionLength, hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execBroadcastBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                       NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->shapeOf(),
                                                                      dimension->lengthOf());
    auto tadPackZ = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
                                                                      dimension->shapeOf(),
                                                                      dimension->lengthOf());

    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execBroadcastBool(nullptr, opNum,
                                           x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                           z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams,
                                           dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                           hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

    NDArray::registerSpecialUse({z}, {x, y});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


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

//note here for javacpp mapping we have to use this odd type alias as a pointer
//to make the typedef work properly.
void setGraphContextOutputArraysArr(OpaqueContext* ptr, int numArrays,OpaqueNDArrayArr *arr) {
  if (arr == nullptr) THROW_EXCEPTION("setGraphContextOutputArraysArr: Input arrays were null!");
  for (int i = 0; i < numArrays; i++) {
    if (arr[i] == nullptr) {
      std::string errorMessage;
      errorMessage += "setGraphContextOutputArraysArr: Input array at index ";
      errorMessage += std::to_string(i);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }
    for (int i = 0; i < numArrays; i++) {
      ptr->setOutputArray(i, *arr[i], false);
    }
  }
}

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



/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param extraParams
 * @param n
 */
void execPairwiseTransform(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                           const LongType *hXShapeInfo, const LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                           const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                           const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execPairwiseTransform(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                               dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                               hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execPairwiseTransformBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                               const LongType *hXShapeInfo, const LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                               const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                               const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execPairwiseBoolTransform(
        nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo, dbY->primary(), hYShapeInfo,
        dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void execReduceFloat(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                     const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                               extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                    const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                    const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceSameScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                              extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                    const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                    const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceBoolScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                              extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                    const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                    const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceLongScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                              extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void execReduceFloat2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                      const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                      const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                      const LongType *hDimensionShape, const LongType *dDimensionShape) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    auto dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    const auto zLen = shape::length(hZShapeInfo);

    std::vector<LongType> *dimensions = new std::vector<LongType>();
    for(LongType i = 0; i < dimensionLength; i++) {
      dimensions->push_back(dimension[i]);
    }

    const LongType *zShapeInfoH = hZShapeInfo;
    const LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : new std::vector<LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceFloat(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                         extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, zShapeInfoH, dbZ != nullptr ? dbZ->special() : nullptr, zShapeInfoD,
                                         dims->data(), dims->size());
    delete dims;
    delete dimensions;
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                     const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const LongType *hDimensionShape, const LongType *dDimensionShape) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    auto dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    std::vector<LongType> *dimensions = new std::vector<LongType>();
    for(LongType i = 0; i < dimensionLength; i++) {
      dimensions->push_back(dimension[i]);
    }

    const auto zLen = shape::length(hZShapeInfo);

    const LongType *zShapeInfoH = hZShapeInfo;
    const LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo)) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : new std::vector<LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceBool(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                        extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, zShapeInfoH, dbZ != nullptr ? dbZ->special() : nullptr, zShapeInfoD,
                                        dims->data(), dims->size());
    delete dims;
    delete dimensions;
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                     const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const LongType *hDimensionShape, const LongType *dDimensionShape) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));
    std::vector<LongType> *dimensions = new std::vector<LongType>();
    for(LongType i = 0; i < dimensionLength; i++) {
      dimensions->push_back(static_cast<LongType>(dimension[i]));
    }


    const auto zLen = shape::length(hZShapeInfo);

    const LongType *zShapeInfoH = hZShapeInfo;
    const LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : new std::vector<LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceSame(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                        extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, zShapeInfoH, dbZ != nullptr ? dbZ->special() : nullptr, zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;
    delete dimensions;
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                     const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const LongType *hDimensionShape, const LongType *dDimensionShape) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    std::vector<LongType> *dimensions = new std::vector<LongType>();
    for(LongType i = 0; i < dimensionLength; i++) {
      dimensions->push_back(dimension[i]);
    }

    const auto zLen = shape::length(hZShapeInfo);

    const LongType *zShapeInfoH = hZShapeInfo;
    const LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : new std::vector<LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceLong(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                        extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, zShapeInfoH, dbZ != nullptr ? dbZ->special() : nullptr, zShapeInfoD,
                                        dims->data(), dims->size());
    delete dims;
    delete dimensions;
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 */
void execReduce3(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                 const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                 const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execReduce3(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                     extraParams, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo,
                                     dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 */
void execReduce3Scalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                       const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                       const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                       const LongType *hZShapeInfo, const LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execReduce3Scalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                           extraParams, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo,
                                           dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParamsVals
 * @param hY
 * @param hYShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execReduce3Tad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                    const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                    const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                    const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                    const LongType *hDimensionShape, const LongType *dDimensionShape,
                    const LongType *tadOnlyShapeInfo, const LongType *tadOffsets,
                    const LongType *yTadOnlyShapeInfo, const LongType *yTadOffsets) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    if (extraPointers == nullptr || extraPointers[2] == 0) {
      OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
      NativeOpExecutioner::execReduce3(
          LaunchContext::defaultContext(), opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo, extraParams,
          dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr,
          dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
      OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
    } else {
      // going tad-way
      auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);

      auto hTADShapeInfo = tadPack->primaryShapeInfo();
      auto hTADOffsets = tadPack->primaryOffsets();

      OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
      NativeOpExecutioner::execReduce3TAD(
          LaunchContext::defaultContext(), opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo, extraParams,
          dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr,
          dZShapeInfo, dimension, dimensionLength, hTADShapeInfo, hTADOffsets, nullptr, nullptr);
      OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

bool isBlasVersionMatches(int major, int minor, int build) { return true; }

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param hScalar
 * @param extraParams
 * @param n
 */
void execScalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                const LongType *dZShapeInfo, OpaqueDataBuffer *dbScalar, const LongType *hScalarShapeInfo,
                const LongType *dScalarShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbScalar});
    NativeOpExecutioner::execScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, dbScalar->primary(),
                                    hScalarShapeInfo, dbScalar->special(), dScalarShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbScalar});

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                    const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                    const LongType *dZShapeInfo, OpaqueDataBuffer *dbScalar, const LongType *hScalarShapeInfo,
                    const LongType *dScalarShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execScalarBool(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                        dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, dbScalar->primary(),
                                        hScalarShapeInfo, dbScalar->special(), dScalarShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 */
void execSummaryStatsScalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                            const LongType *hXShapeInfo, const LongType *dXShapeInfo, void *extraParams,
                            OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo, const LongType *dZShapeInfo,
                            bool biasCorrected) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
                                                dXShapeInfo, extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr,
                                                dZShapeInfo, biasCorrected);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 */
void execSummaryStats(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                      const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                      const LongType *hZShapeInfo, const LongType *dZShapeInfo, bool biasCorrected) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execSummaryStats(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                          extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                          biasCorrected);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 * @param hZ
 * @param hZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execSummaryStatsTad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                         const LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                         const LongType *hZShapeInfo, const LongType *dZShapeInfo,
                         OpaqueDataBuffer *dbDimension, const LongType *hDimensionShape,
                         const LongType *dDimensionShape, bool biasCorrected, const LongType *tadShapeInfo,
                         const LongType *tadOffsets) {

  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = shape::length(hDimensionShape);

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execSummaryStats(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                          extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                          dimension, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param hZ
 * @param hZShapeInfo
 * @param extraParams
 * @param n
 */
void execTransformFloat(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                        const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                        const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformFloat(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                            dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams,
                                            nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformSame(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                       const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                       const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformSame(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                           dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams,
                                           nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                       const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                       const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformBool(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                           dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams,
                                           nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformAny(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                      const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                      const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformAny(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                          dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams,
                                          nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformStrict(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                         const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                         const LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformStrict(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                             dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraParams,
                                             nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduce3All(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                    const LongType *dXShapeInfo, void *extraParamsVals, OpaqueDataBuffer *dbY,
                    const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                    const LongType *hZShapeInfo, const LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                    const LongType *hDimensionShape, const LongType *dDimensionShape,
                    const LongType *xTadShapeInfo, const LongType *xOffsets, const LongType *yTadShapeInfo,
                    const LongType *yOffsets) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execReduce3All(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                        extraParamsVals, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo,
                                        dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, dimension,
                                        dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 * Concatneate multi array of the same shape together
 * along a particular dimension
 */
void specialConcat(Pointer *extraPointers, int dimension, int numArrays, OpaqueNDArray *data, OpaqueNDArray dZ) {
  try {
    BUILD_SINGLE_SELECTOR(dZ->dataType(), sd::SpecialMethods,
                          ::concatCpuGeneric(dimension, numArrays, data, dZ),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void initializeDevicesAndFunctions() {}

void initializeFunctions(Pointer *functions) { BlasHelper::getInstance().initializeFunctions(functions); }

/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
Pointer mallocHost(LongType memorySize, int flags) {
#if defined(SD_ALIGNED_ALLOC)
  return static_cast<Pointer *>(
      aligned_alloc(SD_DESIRED_ALIGNMENT, (memorySize + SD_DESIRED_ALIGNMENT - 1) & (-SD_DESIRED_ALIGNMENT)));
#else
  return reinterpret_cast<Pointer>(new int8_t[memorySize]);
#endif
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Pointer mallocDevice(LongType memorySize, int deviceId, int flags) {
  // not supported
  return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int freeHost(Pointer pointer) {
#if defined(SD_ALIGNED_ALLOC)
  free(pointer);
#else
  delete[] reinterpret_cast<int8_t *>(pointer);
#endif
  return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * PLEASE NOTE: This method is NOT supported and has NO effect in CPU-based backend.
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
int freeDevice(Pointer pointer, int deviceId) {
  // not supported
  return 0L;
}

/**
 * Returns the maximum number open mp threads
 */
int ompGetMaxThreads() { return omp_get_max_threads(); }

/**
 * Returns the number open mp threads
 */
int ompGetNumThreads() { return omp_get_num_threads(); }

/**
 * Sets the number of openmp threads
 */
void setOmpNumThreads(int threads) { omp_set_num_threads(threads); }

Pointer createContext() { return 0L; }

Pointer createStream() { return 0L; }

Pointer createEvent() { return 0L; }

int getDeviceMajor(int deviceId) { return 0; }

int getDeviceMinor(int deviceId) { return 0; }

int registerEvent(Pointer event, Pointer stream) { return 0L; }

int setDevice(int deviceId) { return 0L; }

LongType getDeviceFreeMemory(int deviceId) { return 0L; }

LongType getDeviceFreeMemoryDefault() { return 0L; }

LongType getDeviceTotalMemory(int deviceId) { return 0L; }

int memcpySync(Pointer dst, Pointer src, LongType size, int flags, Pointer reserved) { return 0L; }

int memcpyAsync(Pointer dst, Pointer src, LongType size, int flags, Pointer reserved) { return 0L; }

int memsetSync(Pointer dst, int value, LongType size, int flags, Pointer reserved) { return 0L; }

int memsetAsync(Pointer dst, int value, LongType size, int flags, Pointer reserved) { return 0L; }

int destroyEvent(Pointer event) { return 0L; }

int streamSynchronize(Pointer stream) { return 0L; }

int eventSynchronize(Pointer event) { return 0L; }

int getAvailableDevices() { return 0L; }

void enableDebugMode(bool reallyEnable) { Environment::getInstance().setDebug(reallyEnable); }

void enableVerboseMode(bool reallyEnable) { Environment::getInstance().setVerbose(reallyEnable); }

void setGridLimit(int gridSize) {
  // no-op
}

TadPack *tadOnlyShapeInfo(OpaqueDataBuffer *hXShapeInfo, LongType *dimension, LongType dimensionLength) {
  try {
    auto buffPrim = reinterpret_cast<sd::LongType *>(hXShapeInfo->primary());
    auto rankVal = buffPrim[0];
    if(rankVal == 0) {
      //detect when the shape buffer values are unset.
      auto len = shape::shapeInfoLength(rankVal);
      //min number of values in a shape info buffer
      bool allZero = true;
      for(int i = 0; i < len; i++) {
        if(buffPrim[i] != 0) {
          allZero = false;
          break;
        }
      }

      if(allZero) {
        THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
      }
    }

    auto pack = ConstantTadHelper::getInstance().tadForDimensions(reinterpret_cast<sd::LongType *>(hXShapeInfo->primary()), dimension, dimensionLength);
    return pack;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }


}

LongType const *getPrimaryShapeInfo(TadPack *pack) {
  return const_cast<LongType *>(pack->primaryShapeInfo());
}

LongType const *getPrimaryOffsets(TadPack *pack) {
  if(pack->primaryOffsets() == nullptr)
    THROW_EXCEPTION("getPrimaryOffsets: primaryOffsets is nullptr!");
  return const_cast<LongType *>(pack->primaryOffsets());
}

LongType const *getSpecialShapeInfo(TadPack *pack) {
  return const_cast<LongType *>(pack->specialShapeInfo());
}

LongType const *getSpecialOffsets(TadPack *pack) { return const_cast<LongType *>(pack->specialOffsets()); }

LongType getNumberOfTads(TadPack *pack) { return pack->numberOfTads(); }

int getShapeInfoLength(TadPack *pack) { return pack->shapeInfoLength(); }

int memcpyConstantAsync(LongType dst, Pointer src, LongType size, int flags, Pointer reserved) {
  // no-op
  return 0L;
}

Pointer getConstantSpace() {
  // no-op
  return 0L;
}

template <typename T>
void pullRowsGeneric(void *vx, LongType const *hXShapeInfo, void *vz, LongType const *hZShapeInfo, const int n,
                     LongType const *indexes, LongType const *tadShapeInfo, LongType const *tadOffsets,
                     LongType const *zTadShapeInfo, LongType const *zTadOffsets) {
  auto hX = static_cast<T *>(vx);
  auto hZ = static_cast<T *>(vz);

  const auto xEWS = shape::elementWiseStride(tadShapeInfo);
  const auto zEWS = shape::elementWiseStride(zTadShapeInfo);
  const auto tadLength = shape::length(tadShapeInfo);

  int elementsPerThread = n / TAD_THRESHOLD;
  int _threads = math::sd_max<int>(1, elementsPerThread);
  _threads = math::sd_min<int>(_threads, Environment::getInstance().maxThreads());

  auto func = PRAGMA_THREADS_FOR {
    for (auto idx = start; idx < stop; idx++) {
      auto xTadOffsetForBlock = tadOffsets[indexes[idx]];
      auto zTadOffsetForBlock = zTadOffsets[idx];

      auto rX = hX + xTadOffsetForBlock;
      auto rZ = hZ + zTadOffsetForBlock;

      if (xEWS == 1 && zEWS == 1) {
        PRAGMA_OMP_SIMD
        for (LongType i = 0; i < tadLength; i++) {
          rZ[i] = rX[i];
        }
      } else if (xEWS >= 1 && zEWS >= 1) {
        PRAGMA_OMP_SIMD
        for (LongType i = 0; i < tadLength; i++) {
          rZ[i * zEWS] = rX[i * xEWS];
        }
      } else {
        for (LongType i = 0; i < tadLength; i++) {
          auto xOffset = xTadOffsetForBlock + shape::getIndexOffset(i, tadShapeInfo);
          auto zOffset = zTadOffsetForBlock + shape::getIndexOffset(i, zTadShapeInfo);
          hZ[zOffset] = hX[xOffset];
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, n, 1, _threads);
}

void pullRows(Pointer *extraPointers, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
              LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
              LongType const *dZShapeInfo, LongType n, LongType *indexes, LongType const *tadShapeInfo,
              LongType const *tadOffsets, LongType const *zTadShapeInfo, LongType const *zTadOffsets) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric,
                          (dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, n, indexes, tadShapeInfo,
                              tadOffsets, zTadShapeInfo, zTadOffsets),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

template <typename T>
void tearGeneric(void *vx, LongType const *hXShapeInfo, Pointer *targets, LongType const *hZShapeInfo,
                 LongType const *tadShapeInfo, LongType const *tadOffsets) {
  auto hX = reinterpret_cast<T *>(vx);

  const auto tadLength = shape::length(tadShapeInfo);
  auto tadEWS = shape::elementWiseStride(tadShapeInfo);
  auto zEWS = shape::elementWiseStride(hZShapeInfo);
  auto numTads = shape::length(hXShapeInfo) / tadLength;

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto hZ = reinterpret_cast<T *>(targets[i]);
      auto s = hX + tadOffsets[i];

      if (zEWS == 1 && tadEWS == 1) {
        PRAGMA_OMP_SIMD
        for (LongType j = 0; j < tadLength; j++) {
          hZ[j] = s[j];
        }
      } else if (zEWS > 0 && tadEWS > 0) {
        PRAGMA_OMP_SIMD
        for (LongType j = 0; j < tadLength; j++) {
          hZ[j * zEWS] = s[j * tadEWS];
        }
      } else {
        for (LongType j = 0; j < tadLength; j++)
          hZ[shape::getIndexOffset(j, hZShapeInfo)] = s[shape::getIndexOffset(j, tadShapeInfo)];
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numTads);
}

void tear(Pointer *extraPointers, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
          LongType const *dXShapeInfo, Pointer *targets, LongType const *hZShapeInfo,
          LongType const *tadShapeInfo, LongType const *tadOffsets) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, tearGeneric,
                          (dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, targets, hZShapeInfo, tadShapeInfo, tadOffsets),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void average(Pointer *extras, Pointer *hX, const LongType *hXShapeInfo, Pointer *dX,
             const LongType *dXShapeInfo, void *z, const LongType *hZShapeInfo, void *dz,
             const LongType *dZShapeInfo, int n, LongType length, bool propagate) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, SpecialMethods, ::averageGeneric(hX, z, hZShapeInfo, n, length, propagate),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void accumulate(Pointer *extras, Pointer *hX, LongType const *hXShapeInfo, Pointer *dX,
                LongType const *dXShapeInfo, void *hz, LongType const *hZShapeInfo, void *dz,
                LongType const *dZShapeInfo, int n, LongType length) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, SpecialMethods, ::accumulateGeneric(hX, hz, hZShapeInfo, n, length),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void enableP2P(bool enable) {
  // no-op
}



bool isP2PAvailable() {
  // always TRUE for cpu backend
  return true;
}

void checkP2P() {
  // no-op
}

template <typename T>
void shuffleGeneric(void **hX, LongType *const *hXShapeInfo, void **dz, LongType *const *hZShapeInfo, int N,
                    int *shuffleMap, LongType *const *tadOnlyShapeInfo, LongType *const *tadOffsets) {
  auto dX = reinterpret_cast<T **>(hX);
  auto dZ = reinterpret_cast<T **>(dz);

  auto func = PRAGMA_THREADS_FOR {
    for (auto f = start; f < stop; f++) {
      auto hX = reinterpret_cast<T *>(dX[f]);

      auto xShapeInfo = hXShapeInfo[f];
      auto tadOffset = reinterpret_cast<LongType *>(tadOffsets[f]);

      const auto tadLength = shape::length(tadOnlyShapeInfo[f]);
      auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
      auto tadRank = shape::rank(tadOnlyShapeInfo[f]);
      auto numTads = shape::length(hXShapeInfo[f]) / tadLength;


      if (shape::rank(xShapeInfo) == 1) {
        auto xLength = shape::length(xShapeInfo);
        auto ews = shape::elementWiseStride(xShapeInfo);
        for (LongType r = 0; r < xLength; r++) {
          auto swapIdx = shuffleMap[r];
          if (swapIdx < 0) continue;

          math::sd_swap<T>(hX[r * ews], hX[swapIdx * ews]);
        }
      } else {
        for (LongType r = 0; r < numTads; r++) {
          if (shuffleMap[r] < 0) continue;

          auto oldOffset = tadOffset[r];
          auto newOffset = tadOffset[shuffleMap[r]];

          auto rX = hX + oldOffset;
          auto rY = hX + newOffset;

          if (tadEWS == 1) {
            for (LongType i = 0; i < tadLength; i++) {
              math::sd_swap<T>(rX[i], rY[i]);
            }
          } else {
            for (LongType i = 0; i < tadLength; i++) {
              auto offset = shape::getIndexOffset(i, tadOnlyShapeInfo[f]);
              math::sd_swap<T>(hX[offset + oldOffset], hX[offset + newOffset]);
            }
          }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, N);
}

void shuffle(Pointer *extras, Pointer *hX, Pointer *hXShapeInfo, Pointer *dX, Pointer *dXShapeInfo,
             Pointer *hz, Pointer *hZShapeInfo, Pointer *dz, Pointer *dZShapeInfo, int N,
             int *shuffleMap, Pointer *tadShapeInfo, Pointer *tadOffsets) {
  try {
    auto xShape = reinterpret_cast<LongType *const *>(hXShapeInfo);
    auto zShape = reinterpret_cast<LongType *const *>(hZShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<LongType *const *>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<LongType *const *>(tadOffsets);

    auto xType = ArrayOptions::dataType(xShape[0]);

    BUILD_SINGLE_SELECTOR(xType, shuffleGeneric, (hX, xShape, hz, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

bool isExperimentalEnabled() { return Environment::getInstance().isExperimentalBuild(); }

void setOmpMinThreads(int threads) {
  // TODO: to be implemented
}

int getDevice() { return 0; }

void execScalarTad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                   LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                   LongType const *dZShapeInfo, OpaqueDataBuffer *dbScalars, LongType const *hScalarShapeInfo,
                   LongType const *dScalarShapeInfo, void *extraParams, OpaqueDataBuffer *dbDimension,
                   LongType const *hDimensionShape, LongType const *dDimensionShape,
                   LongType const *tadShapeInfo, LongType const *tadOffsets, LongType const *tadShapeInfoZ,
                   LongType const *tadOffsetsZ) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execScalar(nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    extraParams, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                    dbScalars->primary(), hScalarShapeInfo, dbScalars->special(), dScalarShapeInfo,
                                    dimension, shape::length(hDimensionShape), tadShapeInfo, tadOffsets, tadShapeInfoZ,
                                    tadOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBoolTad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const LongType *hXShapeInfo,
                       const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const LongType *hZShapeInfo,
                       const LongType *dZShapeInfo, OpaqueDataBuffer *dbScalars,
                       const LongType *hScalarShapeInfo, const LongType *dScalarShapeInfo, void *extraParams,
                       OpaqueDataBuffer *dbDimension, const LongType *hDimensionShape,
                       const LongType *dDimensionShape, const LongType *tadShapeInfo,
                       const LongType *tadOffsets, const LongType *tadShapeInfoZ,
                       const LongType *tadOffsetsZ) {
  try {
    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execScalarBool(
        nullptr, opNum, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo, extraParams, dbZ != nullptr ? dbZ->primary() : nullptr,
        hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, dbScalars->primary(), hScalarShapeInfo, dbScalars->special(),
        dScalarShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

const char *getDeviceName(int deviceId) {
  try {
    if (!nameSet) {
      name = reinterpret_cast<char *>(malloc(256 * sizeof(char)));

      CHECK_ALLOC(name, "Failed to allocate new string buffer", 256);

      std::memset(name, 0, 256 * sizeof(char));
      nameSet = true;

      // TODO: provide proper CPU model name here
      sprintf(name, "x86-compatible CPU");
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }

  return name;
}

void execAggregate(Pointer *extraPointers, int opNum, void **arguments, int numArguments,
                   LongType **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments,
                   int **intArrays, int numIntArrays, void *realArguments, int numRealArguments, DataType dtype) {}

void batchExecutor(Pointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                   int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments,
                   DataType dtype) {}

void execAggregateBatch(Pointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                        int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments,
                        DataType dtype) {}

void execRandom(Pointer *extraPointers, int opNum, Pointer state, OpaqueDataBuffer *dbZ,
                const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo,
                                    extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom3(Pointer *extraPointers, int opNum, Pointer state, OpaqueDataBuffer *dbX,
                 const LongType *hXShapeInfo, const LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                 const LongType *hYShapeInfo, const LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ != nullptr ? dbZ->primary() : nullptr,
                                    hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom2(Pointer *extraPointers, int opNum, Pointer state, OpaqueDataBuffer *dbX,
                 const LongType *hXShapeInfo, const LongType *dXShapeInfo, OpaqueDataBuffer *dbZ,
                 const LongType *hZShapeInfo, const LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX != nullptr ? dbX->primary() : nullptr, hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr, dXShapeInfo,
                                    dbZ != nullptr ? dbZ->primary() : nullptr, hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

Pointer initRandom(Pointer *extraPointers, long seed, long bufferSize, Pointer ptrToBuffer) {
  try {
    auto generator = new graph::RandomGenerator(seed, seed);

    return (Pointer)generator;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}

void refreshBuffer(Pointer *extraPointers, long seed, Pointer ptrRandom) {
  auto generator = reinterpret_cast<graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void reSeedBuffer(Pointer *extraPointers, long seed, Pointer ptrRandom) {
  auto generator = reinterpret_cast<graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void destroyRandom(Pointer ptrBuffer) {
  auto buffer = reinterpret_cast<graph::RandomGenerator *>(ptrBuffer);
  delete buffer;
}

/**
 * Return the length of a shape buffer
 * based on the pointer
 * @param buffer  the buffer pointer to check
 * @return
 */
int lengthForShapeBufferPointer(Pointer buffer) {
  auto shapeBuffer = reinterpret_cast<LongType *>(buffer);
  return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
 * The pointer to get the address for
 *
 * @param address the address to get the pointer
 * @return the pointer for the given address
 */

Pointer pointerForAddress(LongType address) { return reinterpret_cast<Pointer>(address); }

void sort(Pointer *extraPointers, void *hX, const LongType *hXShapeInfo, void *dX,
          const LongType *dXShapeInfo, bool descending) {
  try {
    NativeOpExecutioner::execSort(hX, hXShapeInfo, descending);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(Pointer *extraPointers, void *hX, const LongType *hXShapeInfo, void *dX,
             const LongType *dXShapeInfo, LongType *dimension, LongType dimensionLength, const LongType *tadShapeInfo,
             const LongType *tadOffsets, bool descending) {
  try {
    NativeOpExecutioner::execSort(hX, hXShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortCooIndices(Pointer *extraPointers, LongType *indices, void *x, LongType length,
                    const LongType *xShapeInfo) {
  try {
    NativeOpExecutioner::execSortCooIndices(indices, x, length, xShapeInfo);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}



void ravelMultiIndex(Pointer *extraPointers, LongType *indices, LongType *flatIndices, LongType length,
                     LongType *shapeInfo, int mode) {
  NativeOpExecutioner::execRavelMultiIndex(indices, flatIndices, length, shapeInfo, mode);
}

void unravelIndex(Pointer *extraPointers, LongType *indices, LongType *flatIndices, LongType length,
                  LongType *shapeInfo) {
  NativeOpExecutioner::execUnravelIndex(indices, flatIndices, length, shapeInfo);
}


LongType *mmapFile(Pointer *extraPointers, const char *fileName, LongType length) {
  auto hZ = new LongType[2];
  errno = 0;
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
    void *ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // check for failed allocation
    if (ptr == MAP_FAILED) return nullptr;

    hZ[0] = (LongType)ptr;
    hZ[1] = fd;

#endif

    return hZ;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}

void munmapFile(Pointer *extraPointers, LongType *ptrMap, LongType length) {
  munmap((Pointer)ptrMap[0], length);
#if defined(_WIN32) || defined(_WIN64)
  CloseHandle(reinterpret_cast<HANDLE>(ptrMap[1]));
#else
  close((int)ptrMap[1]);
#endif

  delete[] ptrMap;
}

graph::ResultWrapper *executeFlatGraph(Pointer *extraPointers, Pointer flatBufferPointer) {
  try {
    return graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType getResultWrapperSize(graph::ResultWrapper *ptr) { return ptr->size(); }
Pointer getResultWrapperPointer(graph::ResultWrapper *ptr) { return ptr->pointer(); }

const char *getAllCustomOps() { return ops::OpRegistrator::getInstance().getAllCustomOperations(); }

template <typename T>
SD_INLINE int estimateThresholdGeneric(Pointer *extraPointers, Pointer hX, int N, T threshold) {
  auto buffer = reinterpret_cast<T *>(hX);
  int span = (N / 6) + 8;

  auto func = PRAGMA_REDUCE_LONG {
    int64_t cnt = 0;
    PRAGMA_OMP_SIMD
    for (auto e = start; e < stop; e++) {
      auto v = math::sd_abs<T,T>(buffer[e]);
      if (v >= threshold) cnt++;
    }

    return cnt;
  };

  return samediff::Threads::parallel_long(
      func, LAMBDA_AL { return _old + _new; }, 0, N);
}

int estimateThreshold(Pointer *extraPointers, Pointer hX, LongType const *hXShapeInfo, int N,
                      float threshold) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, return estimateThresholdGeneric, (extraPointers, hX, N, threshold), SD_FLOAT_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return 0;
  }
}

LongType getShapeListSize(ShapeList *list) { return list->size(); }

LongType const *getShape(ShapeList *list, LongType i) {
  return const_cast<LongType const *>(list->at(i));
}

void deleteShapeList(Pointer shapeList) {
   auto list = reinterpret_cast<ShapeList *>(shapeList);

   list->destroy();
   delete list;
}

ShapeList *_calculateOutputShapes(Pointer *extraPointers, ops::DeclarableOp *op, Pointer *inputBuffers,
                                  Pointer *inputShapes, int numInputShapes, double *tArgs, int numTArgs,
                                  LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs, int numDArgs,
                                  sd::LongType *offsets) {

  graph::VariableSpace varSpace;
  Context block(2, &varSpace);
  ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numBArgs; e++) block.getBArguments()->push_back(bArgs[e]);

  for (int e = 0; e < numDArgs; e++) block.getDArguments()->push_back(DataTypeUtils::fromInt(dArgs[e]));

  for (int e = 0; e < numInputShapes; e++) {
    auto shape_ = reinterpret_cast<LongType *>(inputShapes[e]);
    if(shape_ == nullptr) {
      THROW_EXCEPTION("Input shape was null!");
    }

    if((shape_ != nullptr && shape_[0] > SD_MAX_RANK) || shape_[0] < 0) {
      THROW_EXCEPTION("Input shape rank is invalid. Either > 32 or < 0. Likely corrupt. Please check your input shapes.");
    }



    // we shouldn't copy buffer if that's empty array
    void *buffer_ = ArrayOptions::arrayType(shape_) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

    auto array = new NDArray(buffer_, shape_, block.launchContext(), 0, offsets[e]);


    // block should contain references to proper variable
    varSpace.putVariable(1, e, array);
    block.pickInput(1, e);

    inShapes.push_back(shape_);
  }

  auto status = op->validateDataTypes(block);
  if (status != Status::OK) THROW_EXCEPTION("Data types validation failed");

  auto shapeList = op->calculateOutputShape(&inShapes, block);

  if (varSpace.launchContext() != nullptr) shapeList->detach();

  return shapeList;
}

ShapeList *calculateOutputShapes2(Pointer *extraPointers, LongType hash, Pointer *inputBuffers, Pointer *inputShapes,
                                  int numInputShapes, double *tArgs, int numTArgs, LongType *iArgs, int numIArgs,
                                  bool *bArgs, int numBArgs, int *dArgs, int numDArgs, sd::LongType *offsets) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs,
                                  numIArgs, bArgs, numBArgs, dArgs, numDArgs, offsets);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}





ShapeList *_calculateOutputShapes(Pointer *extraPointers, ops::DeclarableOp *op, Pointer *inputShapes,
                                  int numInputShapes, double *tArgs, int numTArgs, LongType *iArgs,
                                  int numIArgs) {
  Context block(1);
  ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numInputShapes; e++) {
    if(inputShapes[e] == nullptr) {
      std::string errorMessage;
      errorMessage += "Input shape at index ";
      errorMessage += std::to_string(e);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }
    inShapes.push_back(reinterpret_cast<LongType *>(inputShapes[e]));
  }

  auto shapeList = op->calculateOutputShape(&inShapes, block);
  shapeList->detach();

  return shapeList;
}

ShapeList *calculateOutputShapes(Pointer *extraPointers, LongType hash, Pointer *inputShapes,
                                 int numInputShapes, double *tArgs, int numTArgs, LongType *iArgs,
                                 int numIArgs) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}

Status execCustomOp2(Pointer *extraPointers, LongType hash, Pointer opContext) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);
    auto context = reinterpret_cast<Context *>(opContext);

    return op->execute(context);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
    return Status::VALIDATION;
  }
}

Status realExec(ops::DeclarableOp *op, Pointer *extraPointers, LongType hash, Pointer *inputBuffers,
                Pointer *inputShapes, int numInputs, Pointer *outputBuffers, Pointer *outputShapes,
                int numOutputs, double *tArgs, int numTArgs, LongType *iArgs, int numIArgs, bool *bArgs,
                int numBArgs, bool isInplace) {
  if (op == nullptr) sd_printf("Can't find requested operation: [%lld]\n", hash);

  // we're using the same fake nodeId everywhere here

  std::vector<NDArray *> inputs(numInputs);
  std::vector<NDArray *> outputs(numOutputs);
  std::vector<double> ttArgs(numTArgs);
  std::vector<LongType> iiArgs(numIArgs);
  std::vector<bool> biArgs(numBArgs);

  // filling block now with inputs
  for (int e = 0; e < numInputs; e++) {
    auto shape = reinterpret_cast<LongType *>(inputShapes[e]);
    void *buffer = ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

    inputs[e] = new NDArray(buffer, shape, nullptr, 0, 0);
  }

  // if not inplace - transferring output arrays

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      // we want to keep original output shape intact
      auto shape = shape::copyShape(reinterpret_cast<LongType *>(outputShapes[e]));
      void *buffer = ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : outputBuffers[e];

      // FIXME: revisit this.
      bool canNullify = true;
      for (int i = 0; i < numInputs; i++) {
        void *ibuffer = ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[i];
        if (ibuffer == buffer) {
          canNullify = false;
          break;
        }
      }

      if (canNullify)
        memset((uint8_t *)buffer, '\0',
               shape::length(shape) * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape)));

      auto array = new NDArray(buffer, shape, nullptr, 0, 0);
      outputs[e] = array;

      // and we want to release shape copy once we're done
      delete[] shape;
    }

  for (int e = 0; e < numIArgs; e++) iiArgs[e] = iArgs[e];

  for (int e = 0; e < numTArgs; e++) ttArgs[e] = tArgs[e];

  for (int e = 0; e < numBArgs; e++) biArgs[e] = bArgs[e];

  // hypothetically at this point we have everything filled
  auto hZ = op->execute(inputs, outputs, ttArgs, iiArgs, biArgs, std::vector<DataType>(), isInplace);

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      if (outputs[e]->ordering() != shape::order(reinterpret_cast<LongType *>(outputShapes[e])))
        outputs[e]->streamline(shape::order(reinterpret_cast<LongType *>(outputShapes[e])));
    }

  for (auto v : inputs) delete v;

  for (auto v : outputs) delete v;

  return hZ;
}

Status execCustomOp(Pointer *extraPointers, LongType hash, Pointer *inputBuffers,
                    Pointer *inputShapes, int numInputs, Pointer *outputBuffers, Pointer *outputShapes,
                    int numOutputs, double *tArgs, int numTArgs, LongType *iArgs, int numIArgs, bool *bArgs,
                    int numBArgs, bool isInplace) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);
    return realExec(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes,
                    numOutputs, tArgs, numTArgs, iArgs, numIArgs, bArgs, numBArgs, isInplace);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::BAD_INPUT;
  }
}

Status registerGraph(Pointer *extraPointers, LongType graphId, Pointer flatBufferPointer) {
  try {
    auto graph = graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    graph::GraphHolder::getInstance().registerGraph(graphId, graph);

    return Status::OK;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::BAD_INPUT;
  }
}

static VariablesSet *executeStoredGraphT(Pointer *extraPointers, LongType graphId, Pointer *inputBuffers,
                                         Pointer *inputShapes, int *inputIndices, int numInputs) {
  auto graph = graph::GraphHolder::getInstance().cloneGraph(graphId);
  auto varSpace = graph->getVariableSpace();

  std::vector<NDArray *> handles;

  for (int e = 0; e < numInputs; e++) {
    auto idx = inputIndices[e];

    // we'll delete this array later, together with cloned VariableSpace
    auto array = new NDArray(inputBuffers[e], reinterpret_cast<LongType *>(inputShapes[e]), nullptr, 0, 0);
    handles.emplace_back(array);

    if (varSpace->hasVariable(idx)) {
      auto var = varSpace->getVariable(idx);
      if (var->hasNDArray()) delete var->getNDArray();

      var->setNDArray(array);
    } else
      varSpace->putVariable(idx, array);
  }

  auto hZ = graph::GraphExecutioner::execute(graph, varSpace);
  auto varSet = new graph::VariablesSet(hZ);

  if (hZ == Status::OK) {
    // pull back results, and provide them
    auto outputs = graph->fetchOutputs();
    for (int e = 0; e < outputs->size(); e++) {
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

graph::VariablesSet *executeStoredGraph(Pointer *extraPointers, LongType graphId, Pointer *inputBuffers,
                                        Pointer *inputShapes, int *inputIndices, int numInputs) {
  return nullptr;
}

LongType getVariablesSetSize(graph::VariablesSet *set) { return set->size(); }

Status getVariablesSetStatus(graph::VariablesSet *set) { return set->status(); }

graph::Variable *getVariable(graph::VariablesSet *set, LongType i) { return set->at(i); }

int getVariableId(graph::Variable *variable) { return variable->id(); }

int getVariableIndex(graph::Variable *variable) { return variable->index(); }

const char *getVariableName(graph::Variable *variable) { return variable->getName()->c_str(); }

LongType const *getVariableShape(graph::Variable *variable) {
  return const_cast<LongType const *>(variable->getNDArray()->shapeInfo());
}

void *getVariableBuffer(graph::Variable *variable) { return variable->getNDArray()->buffer(); }

Status unregisterGraph(Pointer *extraPointers, LongType graphId) {
  graph::GraphHolder::getInstance().dropGraphAny(graphId);

  return Status::OK;
}

void deletePointerArray(Pointer pointer) {
  auto ptr = reinterpret_cast<Pointer *>(pointer);
  delete[] ptr;
}

void deleteCharArray(Pointer pointer) {
  auto ptr = reinterpret_cast<char *>(pointer);
  delete[] ptr;
}

void deleteIntArray(Pointer pointer) {
  auto ptr = reinterpret_cast<int *>(pointer);
  delete[] ptr;
}

void deleteLongArray(Pointer pointer) {
  auto ptr = reinterpret_cast<LongType *>(pointer);
  delete[] ptr;
}

void deleteVariablesSet(graph::VariablesSet *pointer) {
  delete pointer;
}

const char *getAllOperations() { return OpTracker::getInstance().exportOperations(); }

Pointer getGraphState(LongType id) { return (Pointer) new graph::GraphState(id); }

void deleteGraphState(Pointer state) {
  auto stateP = reinterpret_cast<graph::GraphState *>(state);
  delete stateP;
}

Status execCustomOpWithScope_(Pointer *extraPointers, graph::GraphState *state, LongType opHash,
                              LongType *scopes, int numScopes, Pointer *inputBuffers,
                              Pointer *inputShapes, int numInputs, Pointer *outputBuffers,
                              Pointer *outputShapes, int numOutputs) {
  /**
   * That's basically exec, with VariableSpace provided in GraphState:
   * depending on operation (i.e. while of if), different logic executors could be used
   */

  auto graph = state->graph();
  auto varSpace = state->variableSpace();

  // Node is dynamically created, and has nothing beyond it: only inputs and outputs
  // this node has id of 0, and inputs are
  Node node(OpType_LOGIC, opHash, 0);

  // mapping inputs
  for (int e = 0; e < numInputs; e++) {
    auto buffer = inputBuffers[e];
    auto shapeInfo = reinterpret_cast<LongType *>(inputShapes[e]);

    auto array = new NDArray(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace
    varSpace->putVariable(0, e, array);
    node.pickInput(0, e);
  }

  // mapping scopes
  for (int e = 0; e < numScopes; e++) {
    // we should check scope existence in GraphState/Graph
    int scopeId = (int)scopes[e];
    if (!state->hasScope(scopeId)) {
      return Logger::logKernelFailureMsg();
    }
    node.pickInput(scopeId, 0);
  }

  auto hZ = LogicExecutor::processNode(graph, &node);
  if (hZ != Status::OK) return hZ;

  // mapping outputs

  for (int e = 0; e < numOutputs; e++) {
    auto buffer = outputBuffers[e];
    auto shapeInfo = reinterpret_cast<LongType *>(outputShapes[e]);

    NDArray array(buffer, shapeInfo, varSpace->launchContext(), 0, 0);

    // now we just put array to VarSpace to the same ID
    // varSpace->putVariable(0, e, array);

    auto t = varSpace->getVariable(0, e)->getNDArray();
    array.assign(*t);
  }

  // removing input variables
  for (int e = 0; e < numInputs; e++) {
    varSpace->dropVariable(0, e);
  }

  // after some bla-bla-bla we should have Graph and Node for current op
  return Status::OK;
}

Status execCustomOpWithScope(Pointer *extraPointers, Pointer state, LongType opHash,
                             LongType *scopes, int numScopes, Pointer *inputBuffers,
                             Pointer *inputShapes, int numInputs, Pointer *outputBuffers,
                             Pointer *outputShapes, int numOutputs) {
  try {
    return execCustomOpWithScope_(extraPointers, reinterpret_cast<graph::GraphState *>(state), opHash, scopes,
                                  numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes,
                                  numOutputs);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::BAD_INPUT;
  }
}

void deleteResultWrapper(Pointer ptr) {
  // just 0 room for compiler s@!t
  auto p = reinterpret_cast<graph::ResultWrapper *>(ptr);
  delete p;
}

/*
 * TypeDef:
 *     void convertTypes(Pointer *extras, int srcType, Pointer hX, long N, int dstType, Pointer hZ);
 */
void convertTypes(Pointer *extras, int srcType, Pointer hX, LongType N, int dstType, Pointer hZ) {
  auto hx = reinterpret_cast<void *>(hX);
  auto hz = reinterpret_cast<void *>(hZ);

  if (srcType == ND4J_FLOAT8) {
    if (dstType == ND4J_FLOAT8) {
      // convertGeneric<double, float8>(hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      // TypeCast::convertGeneric<float8, int8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      // TypeCast::convertGeneric<float8, uint8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      // TypeCast::convertGeneric<float8, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      // TypeCast::convertGeneric<float8, int16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      // TypeCast::convertGeneric<float8, uint16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
    } else if (dstType == ND4J_FLOAT32) {
      // TypeCast::convertGeneric<float8, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      // TypeCast::convertGeneric<float8, double>(nullptr, hx, N, hz);
    } else {
      sd_debug("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_INT8) {
    if (dstType == ND4J_FLOAT8) {
      // TypeCast::convertGeneric<int8, float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      // convertGeneric<int8, int8>(hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      TypeCast::convertGeneric<int8_t, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      TypeCast::convertGeneric<int8_t, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      TypeCast::convertGeneric<int8_t, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      // TypeCast::convertGeneric<int8_t, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO: eventually we might want to add it
    } else if (dstType == ND4J_FLOAT32) {
      TypeCast::convertGeneric<int8_t, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      TypeCast::convertGeneric<int8_t, double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_UINT8) {
    if (dstType == ND4J_FLOAT8) {
      //    TypeCast::convertGeneric<uint8_t, float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      TypeCast::convertGeneric<uint8_t, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      TypeCast::convertGeneric<uint8_t, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      TypeCast::convertGeneric<uint8_t, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      TypeCast::convertGeneric<uint8_t, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //       TypeCast::convertGeneric<uint8_t, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO: still might want to add
    } else if (dstType == ND4J_FLOAT32) {
      TypeCast::convertGeneric<uint8_t, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      TypeCast::convertGeneric<uint8_t, double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_FLOAT16) {
    if (dstType == ND4J_FLOAT8) {
      //    TypeCast::convertGeneric<float16, float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      TypeCast::convertGeneric<float16, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      TypeCast::convertGeneric<float16, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      TypeCast::convertGeneric<float16, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      TypeCast::convertGeneric<float16, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            TypeCast::convertGeneric<float16, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO: .... ^^^
    } else if (dstType == ND4J_FLOAT32) {
      TypeCast::convertGeneric<float16, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      TypeCast::convertGeneric<float16, double>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_THRESHOLD) {
      TypeCast::convertToThreshold<float16>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_INT16) {
    if (dstType == ND4J_FLOAT8) {
      //   TypeCast::convertGeneric<int16_t, float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      TypeCast::convertGeneric<int16_t, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      TypeCast::convertGeneric<int16_t, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      TypeCast::convertGeneric<int16_t, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      // TypeCast::convertGeneric<int16_t, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            TypeCast::convertGeneric<int16_t, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO...
    } else if (dstType == ND4J_FLOAT32) {
      TypeCast::convertGeneric<int16_t, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      TypeCast::convertGeneric<int16_t, double>(nullptr, hx, N, hz);
    } else {
      printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_FLOAT24) {
  } else if (srcType == ND4J_FLOAT32) {
    if (dstType == ND4J_FLOAT8) {
      //    TypeCast::convertGeneric<float, float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      TypeCast::convertGeneric<float, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      TypeCast::convertGeneric<float, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      TypeCast::convertGeneric<float, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      TypeCast::convertGeneric<float, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
    } else if (dstType == ND4J_FLOAT24) {
    } else if (dstType == ND4J_DOUBLE) {
      TypeCast::convertGeneric<float, double>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_THRESHOLD) {
      TypeCast::convertToThreshold<float>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_DOUBLE) {
    if (dstType == ND4J_FLOAT8) {
    } else if (dstType == ND4J_INT8) {
      TypeCast::convertGeneric<double, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      TypeCast::convertGeneric<double, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      TypeCast::convertGeneric<double, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      TypeCast::convertGeneric<double, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            TypeCast::convertGeneric<double, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
    } else if (dstType == ND4J_FLOAT32) {
      TypeCast::convertGeneric<double, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      //
    } else if (dstType == ND4J_THRESHOLD) {
      TypeCast::convertToThreshold<double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_THRESHOLD) {
    if (dstType == ND4J_FLOAT16) {
      TypeCast::convertFromThreshold<float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT32) {
      TypeCast::convertFromThreshold<float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      TypeCast::convertFromThreshold<double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else {
    sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
  }
}



void setShapeBuffer(LongType *inputShapeData,DataType dt,LongType *bufferToSet,char order,int elementWiseStride,bool isEmpty,bool isView) {
  if(inputShapeData == nullptr)
    THROW_EXCEPTION("setShapeBuffer: inputShapeData is null");

  if(bufferToSet == nullptr)
    THROW_EXCEPTION("setShapeBuffer: bufferToSet is null");
  LongType  rank = inputShapeData[0];
  if(rank > SD_MAX_RANK || rank < 0)
    THROW_EXCEPTION("Invalid rank for shape buffer.");
  std::vector<LongType> shape;
  std::vector<LongType> strides;
  //shape, stride, data type
  for(LongType i = 1; i < rank * 2 + 1; i++) {
    if(i <= rank) {
      shape.push_back(inputShapeData[i]);
    } else if(shape.size() == rank) {
      strides.push_back(inputShapeData[i]);
    }
  }

  bufferToSet[0] = rank;

  shape::setOrder(bufferToSet,order);

  auto len = shape::shapeInfoLength(rank);

  auto origShape = shape::shapeOf(inputShapeData);
  auto origStride = shape::stride(inputShapeData);
  shape::setShape(bufferToSet,origShape);
  shape::setStride(bufferToSet,origStride);

  ArrayOptions::setDataType(bufferToSet,dt);
  if(isView) {
    ArrayOptions::toggleIsView(bufferToSet);
  }
  if(!ArrayOptions::isEmpty(inputShapeData) && isEmpty) {
    ArrayOptions::toggleIsEmpty(bufferToSet);
  }


  if(rank == 0) {
    //detect when the shape buffer values are unset.
    auto len = shape::shapeInfoLength(rank);
    //min number of values in a shape info buffer
    bool allZero = true;
    for(int i = 0; i < len; i++) {
      if(bufferToSet[i] != 0) {
        allZero = false;
        break;
      }
    }

    if(allZero) {
      THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
    }

  }


}




Pointer createUtf8String(Pointer *extraPointers, const char *string, int length) {
  auto u = new utf8string(string, length);
  return reinterpret_cast<Pointer>(u);
}

LongType getUtf8StringLength(Pointer *extraPointers, Pointer ptr) {
  return reinterpret_cast<utf8string *>(ptr)->_length;
}
char *getUtf8StringBuffer(Pointer *extraPointers, Pointer ptr) {
  return reinterpret_cast<utf8string *>(ptr)->_buffer;
}

void deleteUtf8String(Pointer *extraPointers, Pointer ptr) {
  delete (reinterpret_cast<utf8string *>(ptr));
}

template <typename I>
static void _scatterUpdate(Pointer *extraPointers, int opCode, int numOfSubArrs, void *hX,
                           const LongType *hXShapeInfo, const LongType *hXOffsets, void *dX,
                           const LongType *dXShapeInfo, const LongType *dXOffsets, void *hY,
                           const LongType *hYShapeInfo, const LongType *hYOffsets, void *dY,
                           const LongType *dYShapeInfo, const LongType *dYOffsets, void *vIindexes,
                           const LongType *hIndicesShapeInfo, void *dIindexes,
                           const LongType *dIndicesShapeInfo) {
  auto hIindexes = reinterpret_cast<I *>(vIindexes);
  auto func = PRAGMA_THREADS_DO {
    for (int i = 0; i < numOfSubArrs; ++i) {
      int threadIndex = thread_id;
      const auto xIndex = hIindexes[i];
      const bool isOwner = xIndex < numThreads ? threadIndex == xIndex : threadIndex == xIndex % numThreads;

      if (!isOwner) continue;

      NDArray inSubArr(reinterpret_cast<int8_t *>(hX) + (hXOffsets[hIindexes[i]] * DataTypeUtils::sizeOf(hXShapeInfo)),
                       hXShapeInfo, nullptr, 0, 0);
      NDArray updSubArr(reinterpret_cast<int8_t *>(hY) + (hYOffsets[i] * DataTypeUtils::sizeOf(hXShapeInfo)),
                        hYShapeInfo, nullptr, 0, 0);

      if (inSubArr.lengthOf() != updSubArr.lengthOf()) {
        continue;
      }

      switch (opCode) {
        case 0:
          inSubArr.applyPairwiseTransform(pairwise::Add, updSubArr, inSubArr);
          break;
        case 1:
          inSubArr.applyPairwiseTransform(pairwise::Subtract, updSubArr, inSubArr);
          break;
        case 2:
          inSubArr.applyPairwiseTransform(pairwise::Multiply, updSubArr, inSubArr);
          break;
        case 3:
          inSubArr.applyPairwiseTransform(pairwise::Divide, updSubArr, inSubArr);
          break;
        case 4:
          inSubArr.applyPairwiseTransform(pairwise::ReverseSubtract, updSubArr, inSubArr);
          break;
        case 5:
          inSubArr.applyPairwiseTransform(pairwise::ReverseDivide, updSubArr, inSubArr);
          break;
        case 6:
          inSubArr.applyPairwiseTransform(pairwise::CopyPws, updSubArr, inSubArr);
          break;
        default:
          continue;
      }
    }
  };

  samediff::Threads::parallel_do(func);
}

////////////////////////////////////////////////////////////////////////
void scatterUpdate(Pointer *extraPointers, int opCode, int numOfSubArrs, void *hX, const LongType *hXShapeInfo,
                   const LongType *hXOffsets, void *dX, const LongType *dXShapeInfo,
                   const LongType *dXOffsets, void *hY, const LongType *hYShapeInfo,
                   const LongType *hYOffsets, void *dY, const LongType *dYShapeInfo,
                   const LongType *dYOffsets, void *hIindexes, const LongType *hIndicesShapeInfo,
                   void *dIindexes, const LongType *dIndicesShapeInfo) {
  auto iType = ArrayOptions::dataType(hIndicesShapeInfo);

  try {
    BUILD_SINGLE_SELECTOR(
        iType, _scatterUpdate,
        (extraPointers, opCode, numOfSubArrs, hX, hXShapeInfo, hXOffsets, dX, dXShapeInfo, dXOffsets, hY, hYShapeInfo,
            hYOffsets, dY, dYShapeInfo, dYOffsets, hIindexes, hIndicesShapeInfo, dIindexes, dIndicesShapeInfo),
        SD_INDEXING_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void inspectArray(Pointer *extraPointers, Pointer buffer, LongType *shapeInfo, Pointer specialBuffer,
                  LongType *specialShapeInfo, Pointer debugInfo) {
  try {
    auto p = reinterpret_cast<DebugInfo *>(debugInfo);
    NDArray array(buffer, shapeInfo, nullptr, 0, 0);
    DebugHelper::retrieveDebugStatistics(p, &array);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void tryPointer(Pointer extra, Pointer p, int len) {
  try {
    auto buf = reinterpret_cast<int8_t *>(p);
    int cnt = 0;
    for (int i = 0; i < len; i++) cnt += buf[cnt];
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(sd::LongType *shapeInfo) {
  try {
    auto buffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(shapeInfo);
    return buffer;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

OpaqueConstantShapeBuffer shapeBufferEx(int rank, LongType *shape, LongType *strides, DataType dtype,
                                         char order, LongType ews, LongType extras) {
  try {

    auto desc = new ShapeDescriptor(dtype, order, shape, strides, rank, extras);
    auto buffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    auto buffPrim = buffer->primary();
    auto rankVal = buffPrim[0];
    if(rankVal == 0) {
      //detect when the shape buffer values are unset.
      auto len = shape::shapeInfoLength(rankVal);
      //min number of values in a shape info buffer
      bool allZero = true;
      for(int i = 0; i < len; i++) {
        if(buffPrim[i] != 0) {
          allZero = false;
          break;
        }
      }

      if(allZero) {
        THROW_EXCEPTION("Found shape buffer with all zero values. Values likely unset.");
      }
    }

    return buffer;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) {
  //implemented in cuda backend: used there only
  //constant buffers otherwise should stick around
}

void deleteConstantDataBuffer(ConstantDataBuffer *ptr) {
  //implemented in cuda backend: used there only
  //constant buffers otherwise should stick around
}

void deleteTadPack(TadPack *ptr) {
  delete ptr;
}

ConstantDataBuffer *constantBufferLong(DataType dtype, const LongType *data, int length) { return nullptr; }

ConstantDataBuffer *constantBufferDouble(DataType dtype, double *data, int length) { return nullptr; }

ConstantDataBuffer *constantBuffer(DataType dtype, ConstantDescriptor *descriptor) {
  try {
    return ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}

Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer dbf) {
  return const_cast<LongType *>(dbf->primary());
}

Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer dbf) {
  return const_cast<LongType *>(dbf->special());
}

Pointer getConstantDataBufferPrimary(ConstantDataBuffer *dbf) { return dbf->primary(); }
Pointer getConstantDataBufferSpecial(ConstantDataBuffer *dbf) { return dbf->special(); }
LongType getConstantDataBufferLength(ConstantDataBuffer *dbf) { return dbf->length(); }
LongType getConstantDataBufferSizeOf(ConstantDataBuffer *dbf) { return dbf->sizeOf(); }

graph::Context *createGraphContext(int nodeId) {
  try {
    return new graph::Context(nodeId);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}
graph::RandomGenerator *getGraphContextRandomGenerator(graph::Context *ptr) { return &ptr->randomGenerator(); }
void markGraphContextInplace(graph::Context *ptr, bool reallyInplace) { ptr->markInplace(reallyInplace); }
void setGraphContextCudaContext(graph::Context *ptr, void *stream, void *reductionPointer,
                                void *allocationPointer) {}





void setGraphContextTArguments(graph::Context *ptr, double *arguments, int numberOfArguments) {
  ptr->setTArguments(arguments, numberOfArguments);
}
void setGraphContextIArguments(graph::Context *ptr, LongType *arguments, int numberOfArguments) {
  ptr->setIArguments(arguments, numberOfArguments);
}
void setGraphContextBArguments(graph::Context *ptr, bool *arguments, int numberOfArguments) {
  ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) {
  std::vector<DataType> dtypes;
  for (int e = 0; e < numberOfArguments; e++) {
    dtypes.push_back(DataTypeUtils::fromInt(arguments[e]));
  }

  ptr->setDArguments(dtypes);
}

void deleteGraphContext(graph::Context *ptr) {
  delete ptr;
}

void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) { ptr->allowHelpers(reallyAllow); }

void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) {
  if (execMode < 0 || execMode > 2) execMode = 0;

  ptr->setExecutionMode((samediff::ExecutionMode)execMode);
}

void ctxPurge(OpaqueContext *ptr) { ptr->clearFastPath(); }

graph::RandomGenerator *createRandomGenerator(LongType rootSeed, LongType nodeSeed) {
  return new graph::RandomGenerator(rootSeed, nodeSeed);
}

LongType getRandomGeneratorRootState(graph::RandomGenerator *ptr) {
  if(ptr == nullptr)
    THROW_EXCEPTION("Unable to get the root state from a null pointer. Please ensure this is created.");
  return ptr->rootState();
}

LongType getRandomGeneratorNodeState(graph::RandomGenerator *ptr) { return ptr->nodeState(); }

void setRandomGeneratorStates(graph::RandomGenerator *ptr, LongType rootSeed, LongType nodeSeed) {
  if(ptr == nullptr)
    THROW_EXCEPTION("Unable to get the root state from a null pointer. Please ensure this is created.");

  ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(graph::RandomGenerator *ptr, LongType index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(graph::RandomGenerator *ptr, LongType index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(graph::RandomGenerator *ptr, LongType index) {
  return ptr->relativeInt(index);
}

LongType getRandomGeneratorRelativeLong(graph::RandomGenerator *ptr, LongType index) {
  return ptr->relativeLong(index);
}

int getRandomGeneratorNextInt(graph::RandomGenerator *ptr) {
  // to nullify  _nodeState._long ^= (steps ^ 0xdeadbeef);
  // we will use step = 0xdeadbeef
  auto result = ptr->relativeInt(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

LongType getRandomGeneratorNextLong(graph::RandomGenerator *ptr) {
  auto result = ptr->relativeLong(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

float getRandomGeneratorNextFloat(graph::RandomGenerator *ptr) {
  auto result = ptr->relativeT<float>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

double getRandomGeneratorNextDouble(graph::RandomGenerator *ptr) {
  auto result = ptr->relativeT<double>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

void deleteRandomGenerator(graph::RandomGenerator *ptr) {
  delete ptr;
}


void saveNpy(std::string fname, const OpaqueDataBuffer *data, const unsigned int *shape, const unsigned int ndims,
             std::string mode) {
  auto dtype = data->getDataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(dtype,cnpy::npy_save,(fname,data->getDataBuffer()->primary(),shape,ndims,mode),SD_COMMON_TYPES);
}

int dataTypeFromNpyHeader(void *header) { return (int)cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header)); }

Pointer shapeBufferForNumpy(Pointer npyArray) {
  try {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int shapeSize = arr.shape.size();
    std::vector<LongType> shape(shapeSize);
    bool _empty = false;
    for (unsigned int i = 0; i < shapeSize; i++) {
      shape[i] = arr.shape[i];

      if (arr.shape[i] == 0) _empty = true;
    }

    auto dtype = cnpy::dataTypeFromHeader(reinterpret_cast<char *>(npyArray));

    LongType *shapeBuffer;
    if (shape.size() == 1 && shape[0] == 0) {
      // scalar case
      shapeBuffer = ShapeBuilders::createScalarShapeInfo(dtype);
    } else if (_empty) {
      if (shapeSize > 0)
        shapeBuffer = ShapeBuilders::emptyShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
      else
        shapeBuffer = ShapeBuilders::emptyShapeInfo(dtype);
    } else {
      shapeBuffer = ShapeBuilders::createShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
    }
    return const_cast<LongType *>(ConstantShapeHelper::getInstance().createFromExisting(shapeBuffer, true));
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void sortByKey(Pointer *extraPointers, void *x, const LongType *xShapeInfo, void *dx,
               const LongType *dxShapeInfo, void *y, const LongType *yShapeInfo, void *dy,
               const LongType *dyShapeInfo, bool descending) {
  try {
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods, ::sortByKey(x, xShapeInfo, y, yShapeInfo, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByValue(Pointer *extraPointers, void *x, const LongType *xShapeInfo, void *dx,
                 const LongType *dxShapeInfo, void *y, const LongType *yShapeInfo, void *dy,
                 const LongType *dyShapeInfo, bool descending) {
  try {
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods, ::sortByValue(x, xShapeInfo, y, yShapeInfo, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByKey(Pointer *extraPointers,NDArray *x,NDArray *y,
                  LongType *dimension, LongType dimensionLength, bool descending) {
  try {
    auto xType = x->dataType();
    auto yType = y->dataType();

    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods,
                          ::sortTadByKey(x->buffer(), x->shapeInfo(), y->buffer(),
                                         y->shapeInfo(), dimension, dimensionLength, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByValue(Pointer *extraPointers, NDArray *x,
                    NDArray *y, LongType *dimension,
                    LongType dimensionLength, bool descending)
    try {
    auto xType = x->dataType();
    auto yType = y->dataType();

    BUILD_DOUBLE_SELECTOR(xType, yType, DoubleMethods,
                          ::sortTadByValue(x, x->shapeInfo(), y, y->shapeInfo(), dimension, dimensionLength, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }



void execIndexReduceScalar(Pointer *extraPointers, int opNum, NDArray *x,
                           NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum,
                                               x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execIndexReduce(Pointer *extraPointers, int opNum, NDArray *x,
                     NDArray *z, NDArray *dimension,
                     void *extraParams) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execIndexReduce(nullptr, opNum,
                                         x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                         dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                         hTADShapeInfo, hTADOffsets);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execBroadcast(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                   NDArray *z, NDArray *dimension) {
  try {
    auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                      dimension->shapeOf(),
                                                                      dimension->lengthOf());
    auto tadPackZ = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(),
                                                                      dimension->shapeOf(),
                                                                      dimension->lengthOf());

    auto hTADShapeInfo = tadPackX->primaryShapeInfo();
    auto hTADOffsets = tadPackX->primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ->primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ->primaryOffsets();

    NativeOpExecutioner::execBroadcast(nullptr, opNum,
                                       x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                       y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                       z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                       dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                       hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execPairwiseTransform(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                           NDArray *z, void *extraParams) {
  try {
    /**
     * TODO: look in to offsets here as left over change from ndarrays being available?
     */
    NativeOpExecutioner::execPairwiseTransform(nullptr, opNum,
                                               x->bufferWithOffset(x->offset()),
                                               x->shapeInfo(),
                                               x->specialBufferWithOffset(x->offset()),
                                               x->specialShapeInfo(),
                                               y->bufferWithOffset(y->offset()),
                                               y->shapeInfo(),
                                               y->specialBufferWithOffset(y->offset()),
                                               y->specialShapeInfo(),
                                               z->bufferWithOffset(z->offset()),
                                               z->shapeInfo(),
                                               const_cast<void *>(z->specialBufferWithOffset(z->offset())),
                                               z->specialShapeInfo(),
                                               extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceFloat(Pointer *extraPointers, int opNum, NDArray *x,
                     NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum,
                                               x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                               extraParams,
                                               z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame(Pointer *extraPointers, int opNum, NDArray *x,
                    NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execReduceSameScalar(nullptr, opNum,
                                              x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                              extraParams,
                                              z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool(Pointer *extraPointers, int opNum, NDArray *x,
                    NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execReduceBoolScalar(nullptr, opNum,
                                              x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                              extraParams,
                                              z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong(Pointer *extraPointers, int opNum, NDArray *x,
                    NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execReduceLongScalar(nullptr, opNum,
                                              x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                              extraParams,
                                              z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceFloat2(Pointer *extraPointers, int opNum, NDArray *x,
                      NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceFloat(nullptr, opNum,
                                         x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                         extraParams,
                                         z->dataBuffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                         dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool2(Pointer *extraPointers, int opNum, NDArray *x,
                     NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo())) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceBool(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->dataBuffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame2(Pointer *extraPointers, int opNum, NDArray *x,
                     NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceSame(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->dataBuffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong2(Pointer *extraPointers, int opNum, NDArray *x,
                     NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    std::vector<LongType> dimensions(dimension->lengthOf());
    for(LongType i = 0; i < dimension->lengthOf(); i++) {
      dimensions[i] = dimension->e<LongType>(i);
    }

    const LongType *zShapeInfoH = z->shapeInfo();
    const LongType *zShapeInfoD = z->specialShapeInfo();

    if (shape::rank(x->shapeInfo()) - dimension->lengthOf() != shape::rank(z->shapeInfo()) && z->lengthOf() != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<LongType const *>(zPack->special());
    }

    std::vector<LongType> *dims = (z->lengthOf() != 1) ?
                                  ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) :
                                  new std::vector<LongType>();

    NativeOpExecutioner::execReduceLong(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->dataBuffer(), zShapeInfoH, z->specialBuffer(), zShapeInfoD,
                                        dims->data(), dims->size());

    delete dims;

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduce3(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                 NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execReduce3(nullptr, opNum,
                                     x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                     extraParams,
                                     y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                     z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void execReduce3Scalar(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                       NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execReduce3Scalar(nullptr, opNum,
                                           x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           extraParams,
                                           y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                           z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduce3Tad(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                    NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    auto hTADShapeInfo = tadPack->primaryShapeInfo();
    auto hTADOffsets = tadPack->primaryOffsets();

    NativeOpExecutioner::execReduce3TAD(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                        hTADShapeInfo, hTADOffsets, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalar(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                NDArray *scalar, void *extraParams) {
  try {
    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->dataBuffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                    NDArray *scalar, void *extraParams) {
  try {
    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->dataBuffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execSummaryStatsScalar(Pointer *extraPointers, int opNum, NDArray *x,
                            NDArray *z, void *extraParams, bool biasCorrected) {
  try {
    NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum,
                                                x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                extraParams,
                                                z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                biasCorrected);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execSummaryStats(Pointer *extraPointers, int opNum, NDArray *x,
                      NDArray *z, void *extraParams, bool biasCorrected) {
  try {
    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          biasCorrected);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execSummaryStatsTad(Pointer *extraPointers, int opNum, NDArray *x,
                         NDArray *z, NDArray *dimension, void *extraParams,
                         bool biasCorrected) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execSummaryStats(nullptr, opNum,
                                          x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          extraParams,
                                          z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                          tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                          biasCorrected);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformFloat(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                        void *extraParams) {
  try {
    NativeOpExecutioner::execTransformFloat(nullptr, opNum,
                                            x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                            z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                            extraParams, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformSame(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                       void *extraParams) {
  try {
    NativeOpExecutioner::execTransformSame(nullptr, opNum,
                                           x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                       void *extraParams) {
  try {
    NativeOpExecutioner::execTransformBool(nullptr, opNum,
                                           x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                           z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extraParams, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformAny(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                      void *extraParams) {
  try {
    NativeOpExecutioner::execTransformAny(nullptr, opNum,
                                          x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                          z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                          extraParams, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformStrict(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                         void *extraParams) {
  try {
    NativeOpExecutioner::execTransformStrict(nullptr, opNum,
                                             x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                             z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                             extraParams, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduce3All(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                    NDArray *z, NDArray *dimension, void *extraParams) {
  try {
    NativeOpExecutioner::execReduce3All(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                        z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                        nullptr, nullptr, nullptr, nullptr);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom(Pointer *extraPointers, int opNum, Pointer state, NDArray *z,
                void *extraArguments) {
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom3(Pointer *extraPointers, int opNum, Pointer state, NDArray *x, NDArray *y, NDArray *z,
                 void *extraArguments) {
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                    z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarTad(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                   NDArray *scalar, NDArray *dimension, void *extraParams) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalar(nullptr, opNum,
                                    x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    extraParams,
                                    z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    scalar->dataBuffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                    dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                    tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBoolTad(Pointer *extraPointers, int opNum, NDArray *x, NDArray *z,
                       NDArray *scalar, NDArray *dimension, void *extraParams) {
  try {
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(),
                                                                     dimension->shapeOf(),
                                                                     dimension->lengthOf());

    NativeOpExecutioner::execScalarBool(nullptr, opNum,
                                        x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                        extraParams,
                                        z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                        scalar->dataBuffer(), scalar->shapeInfo(), scalar->specialBuffer(), scalar->specialShapeInfo(),
                                        dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets(),
                                        tadPack->primaryShapeInfo(), tadPack->primaryOffsets());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void execPairwiseTransformBool(Pointer *extraPointers, int opNum, NDArray *x, NDArray *y,
                               NDArray *z, void *extraParams) {
  try {
    NativeOpExecutioner::execPairwiseBoolTransform(nullptr, opNum,
                                                   x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                                   y->dataBuffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
                                                   z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                                   extraParams);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

Status execCustomOp(Pointer *extraPointers, LongType hash, NDArray **inputs, int numInputs,
                    NDArray **outputs, int numOutputs, double *tArgs, int numTArgs,
                    LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, bool isInplace) {
  try {
    const std::vector<NDArray*> inputVec(inputs, inputs + numInputs);
    const std::vector<NDArray*> outputVec(outputs, outputs + numOutputs);
    const std::vector<double> tArgsVec(tArgs, tArgs + numTArgs);
    const std::vector<LongType> iArgsVec(iArgs, iArgs + numIArgs);
    const std::vector<bool> bArgsVec(bArgs, bArgs + numBArgs);

    auto op = ops::OpRegistrator::getInstance().getOperation(hash);
    return op->execute(inputVec, outputVec, tArgsVec, iArgsVec, bArgsVec,{},false);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::KERNEL_FAILURE;
  }
}


void sort(Pointer *extraPointers, NDArray *x, bool descending) {
  try {
    NativeOpExecutioner::execSort(x->buffer(), x->shapeInfo(), descending);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(Pointer *extraPointers, NDArray *x, NDArray *dimension,
             const LongType *tadShapeInfo, const LongType *tadOffsets, bool descending) {
  try {
    NativeOpExecutioner::execSort(x->buffer(), x->shapeInfo(),
                                  dimension->bufferAsT<LongType>(), dimension->lengthOf(),
                                  tadShapeInfo, tadOffsets, descending);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortCooIndices(Pointer *extraPointers, NDArray *indices, NDArray *values) {
  try {
    NativeOpExecutioner::execSortCooIndices(indices->bufferAsT<LongType>(), values->buffer(),
                                            values->lengthOf(), values->shapeInfo());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}



void execRandom2(Pointer *extraPointers, int opNum, Pointer state,
                 NDArray *x, NDArray *z, void *extraArguments) {
  try {
    NativeOpExecutioner::execRandom(nullptr, opNum, state,
                                    x->dataBuffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                                    z->dataBuffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                    extraArguments);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}




void ravelMultiIndex(Pointer *extraPointers, NDArray *indices, NDArray *flatIndices,
                     NDArray *shapeInfo, int mode) {
  try {
    NativeOpExecutioner::execRavelMultiIndex(indices->bufferAsT<LongType>(),
                                             flatIndices->bufferAsT<LongType>(),
                                             flatIndices->lengthOf(),
                                             shapeInfo->bufferAsT<LongType>(), mode);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void unravelIndex(Pointer *extraPointers, NDArray *indices, NDArray *flatIndices,
                  NDArray *shapeInfo) {
  try {
    NativeOpExecutioner::execUnravelIndex(indices->bufferAsT<LongType>(),
                                          flatIndices->bufferAsT<LongType>(),
                                          flatIndices->lengthOf(),
                                          shapeInfo->bufferAsT<LongType>());
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

LongType getCachedMemory(int deviceId) { return ConstantHelper::getInstance().getCachedAmount(deviceId); }



int lastErrorCode() {
  if( LaunchContext::defaultContext()->errorReference() != nullptr)
    return LaunchContext::defaultContext()->errorReference()->errorCode();
  return 0;
}

const char *lastErrorMessage() {
  if( LaunchContext::defaultContext()->errorReference() != nullptr)
    return LaunchContext::defaultContext()->errorReference()->errorMessage();
  return "";
}

void ctxShapeFunctionOverride(OpaqueContext *ptr, bool reallyOverride) {
  ptr->setShapeFunctionOverride(reallyOverride);
}

int binaryLevel() {
#ifdef CPU_FEATURES

  #if defined(F_X64)
  return 1;
#elif defined(F_AVX2)
  return 2;
#elif defined(F_AVX512)
  return 3;
#else
  return 0;
#endif

#else
  return 0;
#endif
}

int optimalLevel() {
#ifdef CPU_FEATURES
  auto features = cpu_features::GetX86Info().features;

  if (features.avx && features.avx2 && features.avx512f && features.avx512vl && features.avx512bw &&
      features.avx512dq && features.avx512cd)
    return 3;
  else if (features.avx && features.avx2)
    return 2;
  else
    return 1;

#else
  return 0;
#endif
}

bool isMinimalRequirementsMet() {
#ifdef CPU_FEATURES
  auto features = cpu_features::GetX86Info().features;

#if defined(F_X64)
  return true;
#elif defined(F_AVX2)
  return features.avx && features.avx2;
#elif defined(F_AVX512)
  // we're optimizing for skylake-avx512 features, so we'll check those out
  return features.avx && features.avx2 && features.avx512f && features.avx512vl && features.avx512bw &&
         features.avx512dq && features.avx512cd;
#else
  return true;
#endif

#else
  return true;
#endif
}

bool isOptimalRequirementsMet() {
#ifdef CPU_FEATURES
  auto b = ::binaryLevel();
  auto o = ::optimalLevel();

  if (b == o)
    return true;
  else
    return false;
#else
  return true;
#endif
}


template <typename T>
void _printHostBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) {
  auto xType = buffer->dataBuffer()->getDataType();
  LongType len = buffer->dataBuffer()->getNumElements();
  auto buff = buffer->dataBuffer()->template primaryAsT<T>();
  sd_printf("Data type %s: ", DataTypeUtils::asString(xType).c_str());
  sd_printf("Host buffer: ",0);
  for(int i = offset; i < len; i++) {
    sd_printf("%f ",(double) buff[i]);
  }

  sd_printf("\n",0);
}

void printDeviceBuffer(OpaqueDataBuffer *buffer, sd::LongType offset) {
  if(buffer->special() != nullptr) {
    sd_printf("Device pointer address: %d\n", buffer->special());
  } else {
    sd_printf("Device pointer address: none\n",0);
  }

  if(buffer->primary() != nullptr) {
    sd_printf("Host pointer address: %d\n", buffer->primary());
  } else  {
    sd_printf("Host pointer address: none\n",0);
  }

  auto xType = buffer->dataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(buffer,offset),SD_COMMON_TYPES_ALL);


}

void setIntermediateResult(OpaqueContext *contextPointer,
                           int index,
                           OpaqueDataBuffer *buffer,
                           OpaqueDataBuffer *shapeInfo,
                           sd::LongType dataOffset) {
  if(shapeInfo == nullptr) {
    THROW_EXCEPTION("Set Intermediate Result: shapeInfo is null");
  }
  auto casted = reinterpret_cast<LongType *>(shapeInfo->primary());
  auto desc = new ShapeDescriptor(casted, false);
  auto arr = new NDArray(buffer->dataBuffer(),
                         desc,
                         LaunchContext::defaultContext(),
                         dataOffset);
  contextPointer->setIntermediateResult(index, arr);
}


std::vector<const LongType *> intermediateResultsShapeInfo(OpaqueContext *contextPointer) {
  std::vector<const LongType *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    const LongType *buff = v->shapeInfo();
    intermediates.push_back(buff);
  }

  return intermediates;
}

std::vector<OpaqueDataBuffer *> intermediateResults(OpaqueContext *contextPointer) {
  std::vector<OpaqueDataBuffer *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    OpaqueDataBuffer *buff = new OpaqueDataBuffer (v->dataBuffer());
    intermediates.push_back(buff);
  }

  return intermediates;
}

int numIntermediateResults(OpaqueContext *contextPointer) {
  return contextPointer->numIntermediates();
}

void pushIntermediateResult(OpaqueContext *contextPointer,
                            OpaqueDataBuffer *buffer,
                            OpaqueDataBuffer *shapeInfo,
                            LongType offset) {
  auto shapeInfoCast = reinterpret_cast<LongType *>(shapeInfo->primary());
  auto desc = new ShapeDescriptor(shapeInfoCast, false);
  auto arr = new NDArray(buffer->dataBuffer(), desc, LaunchContext::defaultContext(), offset);
  contextPointer->pushIntermediateResult(arr);
}

OpaqueDataBuffer  * intermediateResultDataAt(int index, OpaqueContext *contextPointer) {
  auto arr = contextPointer->intermediateResult(index);
  return new OpaqueDataBuffer(arr->dataBuffer());
}

const sd::LongType * intermediateResultShapeInfoAt(int index, OpaqueContext *contextPointer) {
  auto context = reinterpret_cast<graph::Context *>(contextPointer);
  auto arr = context->intermediateResult(index);
  return arr->shapeInfo();
}


OpaqueDataBuffer *dbAllocateDataBuffer(LongType elements, int dataType, bool allocateBoth) {
  return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer *allocateDataBuffer(LongType elements, int dataType, bool allocateBoth) {
  try {
    auto dtype = DataTypeUtils::fromInt(dataType);
    LongType totalElementSize = elements == 0 ?  DataTypeUtils::sizeOf(dtype) : elements * DataTypeUtils::sizeOf(dtype);
    return new OpaqueDataBuffer(totalElementSize, dtype, allocateBoth);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }
}

LongType dbBufferLength(OpaqueDataBuffer *dataBuffer) {
  return dataBuffer->dataBuffer()->getNumElements();
}


Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is nullptr");
  return dataBuffer->primary();
}

Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) { return dataBuffer->special(); }

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
  delete dataBuffer;
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(LongType elements, int dataType, Pointer primary,
                                             Pointer special) {
  auto buffer = dbAllocateDataBuffer(0, dataType, false);
  buffer->markOwner(false);

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) buffer->setSpecial(special, elements);

  return buffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, Pointer primaryBuffer, LongType numBytes) {
  dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, Pointer specialBuffer, LongType numBytes) {
  dataBuffer->setSpecial(specialBuffer, numBytes);

}

void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  dataBuffer->dataBuffer()->allocatePrimary();
}

void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->allocateSpecial(); }

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, LongType elements) {
  try {
    dataBuffer->dataBuffer()->expand(elements * DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueDataBuffer *dbCreateView(OpaqueDataBuffer *dataBuffer, LongType length) {
  return new OpaqueDataBuffer(dataBuffer, length);
}


int dbUseCount(OpaqueDataBuffer* dataBuffer){
  if(dataBuffer) return dataBuffer->useCount();
  return 0;
}

void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->syncToSpecial(); }

void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->syncToPrimary(nullptr); }

void dbTickHostRead(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->readPrimary(); }

void dbTickHostWrite(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->writePrimary(); }

void dbTickDeviceRead(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->readSpecial(); }

void dbTickDeviceWrite(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->writeSpecial(); }

void dbExpand(OpaqueDataBuffer *dataBuffer, LongType elements) { dataBuffer->expand(elements); }

int dbLocality(OpaqueDataBuffer *dataBuffer) { return 0; }

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) { dataBuffer->setDeviceId(deviceId); }

int dbDeviceId(OpaqueDataBuffer *dataBuffer) { return dataBuffer->deviceId(); }

void dbClose(OpaqueDataBuffer *dataBuffer) {
  dataBuffer->getDataBuffer()->close();
}


BUILD_SINGLE_TEMPLATE(template void pullRowsGeneric,
                      (void *, LongType const *, void *, LongType const *, const int, LongType const *,
                          LongType const *, LongType const *, LongType const *, LongType const *),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void tearGeneric,
                      (void *, LongType const *, Pointer *, LongType const *, LongType const *,
                          LongType const *),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void shuffleGeneric,
                      (void **, LongType *const *, void **, LongType *const *, int, int *,
                          LongType *const *, LongType *const *),
                      SD_COMMON_TYPES);
