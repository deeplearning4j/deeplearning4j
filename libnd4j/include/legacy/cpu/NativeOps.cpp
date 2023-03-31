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
#if defined(HAVE_VEDA)
#include <ops/declarable/platform/vednn/veda_helper.h>
#endif
char *name;
bool nameSet = false;

#ifdef SD_EXPERIMENTAL_ENABLED
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif
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

#if defined(HAVE_VEDA)
#include <ops/declarable/PlatformHelperLegacy.h>
#endif
#include <ops/declarable/OpRegistrator.h>

using namespace sd;

SD_LIB_EXPORT int contextNumInputs(void *contextPointer) {
  sd::graph::Context *context = (sd::graph::Context *) contextPointer;
  return context->width();
}

SD_LIB_EXPORT int contextNumOutputs(void *contextPointer) {
  sd::graph::Context *context = (sd::graph::Context *) contextPointer;
  return context->outputWidth();
}

SD_LIB_EXPORT int numInputs(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return trace->inputShapeBuffers->size();
}

SD_LIB_EXPORT int numOutputs(void *execTrace) {
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

#if defined(HAVE_VEDA)
static bool execHelper(const char *entryPrefix, int opNum, void *extraParams, const sd::LongType *hZShapeInfo,
                       OpaqueDataBuffer *dbZ, const sd::LongType *hXShapeInfo, OpaqueDataBuffer *dbX,
                       const sd::LongType *hYShapeInfo, OpaqueDataBuffer *dbY, bool syncDbY = true) {
  if (sd::Environment::getInstance().helpersAllowed()) {
    sd::ops::platforms::PlatformHelperLegacyEntry entry{entryPrefix, opNum, samediff::ENGINE_CPU};
    auto helper = sd::ops::OpRegistrator::getInstance().getPlatformHelperLegacy(entry);
    if (helper && helper->isUsable(extraParams, hZShapeInfo, hXShapeInfo, hYShapeInfo)) {
      // make sure its synced before calling
      VEDA_HANDLE &handle = VEDA::getInstance().getVEDA_HANDLE(0);
      SCOPED_VEDA_CONTEXT scopedContext(handle.getDevice());

      dbX->getDataBuffer()->allocVeda();
      dbX->getDataBuffer()->asyncToVeda();
      if (dbY && syncDbY) {
        dbY->getDataBuffer()->allocVeda();
        dbY->getDataBuffer()->asyncToVeda();
      }
      dbZ->getDataBuffer()->allocVeda();
      dbZ->getDataBuffer()->writeSpecial();

      helper->invokeHelper(extraParams, hZShapeInfo, dbZ, hXShapeInfo, dbX, hYShapeInfo, dbY);
      return true;
    }
  }
  return false;
}

static bool execHelperTransformStrict(int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                                      OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo, void *extraParams) {
  // Note: output comes first with order (shapeInfo, buffer )
  return execHelper(UNIQUE_TRANSFORM_STRICT_PREFIX, opNum, extraParams, hZShapeInfo, dbZ, hXShapeInfo, dbX, nullptr,
                    nullptr);
}

static bool execHelperScalar(int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo, OpaqueDataBuffer *dbY,
                             const sd::LongType *hYShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                             void *extraParams) {
  // Note: output comes first with order (shapeInfo, buffer )
  //we will not sync dbY as its scalar and can be passed as argument
  return execHelper(UNIQUE_SCALAROP_PREFIX, opNum, extraParams, hZShapeInfo, dbZ, hXShapeInfo, dbX, hYShapeInfo, dbY, false);
}

#endif

void printOpTrace() {
  auto execTrace = *sd::ops::OpRegistrator::getInstance().execTrace();
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

    sd_printf(" Output  buffers:\n",0);
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
  return sd::ops::OpRegistrator::getInstance().execTrace();
}

void toggleOpTrace(bool opTrace) {
  sd::ops::OpRegistrator::getInstance().toggleTraceOps(opTrace);
}

void purgeOpTrace() {
  sd::ops::OpRegistrator::getInstance().purgeOpExecs();
}

void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) {
  OpaqueDataBuffer *copyFrom = dbCreateView(from,n,fromOffset);
  OpaqueDataBuffer *targetView = dbCreateView(target,n,targetOffset);
  const DataBuffer targetBuf = *copyFrom->dataBuffer().get();
  const DataBuffer srcBuf = *targetView->dataBuffer().get();
  DataBuffer::memcpy(targetBuf,srcBuf);
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 */
void execIndexReduceScalar(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                           const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, void *extraParams,
                           OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execIndexReduceScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                               extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execIndexReduce(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                     const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);

    auto hTADShapeInfo = tadPack.primaryShapeInfo();
    auto hTADOffsets = tadPack.primaryOffsets();

    auto hz = reinterpret_cast<sd::LongType *>(dbZ->primary());

    NativeOpExecutioner::execIndexReduce(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                         extraParams, hz, hZShapeInfo, dbZ->special(), dZShapeInfo, dimension,
                                         dimensionLength, hTADShapeInfo, hTADOffsets);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execBroadcast(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                   const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbY, const sd::LongType *hYShapeInfo,
                   const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                   const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension, const sd::LongType *hDimensionShape,
                   const sd::LongType *dDimensionShape) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, dimension, dimensionLength);

    auto hTADShapeInfo = tadPackX.primaryShapeInfo();
    auto hTADOffsets = tadPackX.primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ.primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ.primaryOffsets();

    NativeOpExecutioner::execBroadcast(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                       dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(),
                                       hZShapeInfo, dbZ->special(), dZShapeInfo, dimension, dimensionLength,
                                       hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execBroadcastBool(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                       const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbY, const sd::LongType *hYShapeInfo,
                       const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                       const sd::LongType *dZShapeInfo, void *extraParams, OpaqueDataBuffer *dbDimension,
                       const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(hZShapeInfo, dimension, dimensionLength);

    auto hTADShapeInfo = tadPackX.primaryShapeInfo();
    auto hTADOffsets = tadPackX.primaryOffsets();
    auto hTADShapeInfoZ = tadPackZ.primaryShapeInfo();
    auto hTADOffsetsZ = tadPackZ.primaryOffsets();

    NativeOpExecutioner::execBroadcastBool(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                           dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(),
                                           hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams, dimension,
                                           dimensionLength, hTADShapeInfo, hTADOffsets, hTADShapeInfoZ, hTADOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void setGraphContextInputArrays(OpaqueContext* ptr, int numArrays, sd::Pointer * buffer, sd::Pointer * shapeInfo,
                                sd::Pointer * specialBuffer, sd::Pointer * specialShapeInfo) {

  auto inputBuffers = (void **) buffer;
  auto inputShapeBuffers = (void **) shapeInfo;
  for(int i = 0; i < numArrays; i++) {
    ptr->setInputArray(i,inputBuffers != nullptr && inputBuffers[i] != nullptr ? inputBuffers[i] : nullptr,inputShapeBuffers[i],specialBuffer != nullptr ? specialBuffer[i] : nullptr,specialShapeInfo != nullptr ? specialShapeInfo[i] : nullptr);
  }

}
void setGraphContextOutputArrays(OpaqueContext* ptr, int numArrays, void** buffer, sd::Pointer * shapeInfo,
                                 sd::Pointer * specialBuffer, sd::Pointer * specialShapeInfo) {
  auto inputBuffers = (void **) buffer;
  auto inputShapeBuffers = (void **) shapeInfo;
  for(int i = 0; i < numArrays; i++) {
    ptr->setOutputArray(i,inputBuffers != nullptr && inputBuffers[i] != nullptr  ? inputBuffers[i] : nullptr,inputShapeBuffers[i],specialBuffer != nullptr ? specialBuffer[i] : nullptr,specialShapeInfo != nullptr ? specialShapeInfo[i] : nullptr);
  }

}
void  setGraphContextInputBuffers(OpaqueContext* ptr, int numArrays, OpaqueDataBuffer** buffer, sd::Pointer * shapeInfo,
                                  sd::Pointer * specialShapeInfo) {
  auto inputShapeBuffers = (void **) shapeInfo;
  if(shapeInfo == nullptr)
    throw std::runtime_error("Input shape info was null!");
  for(int i = 0; i < numArrays; i++) {
    if(inputShapeBuffers[i] == nullptr)
      throw std::runtime_error("Input shape at index was null!");
    if(buffer != nullptr && buffer[i] != nullptr) {
      setGraphContextInputBuffer(ptr,i,buffer[i],inputShapeBuffers[i],specialShapeInfo != nullptr ? specialShapeInfo[i] : nullptr);
    }
    else {
      setGraphContextInputBuffer(ptr,i, nullptr,inputShapeBuffers[i],specialShapeInfo);
    }
  }

}
void setGraphContextOutputBuffers(OpaqueContext* ptr, int numArrays, OpaqueDataBuffer** buffer, sd::Pointer* shapeInfo,
                                  sd::Pointer * specialShapeInfo) {
  auto inputShapeBuffers = (void **) shapeInfo;

  for(int i = 0; i < numArrays; i++) {
    if(buffer != nullptr && buffer[i] != nullptr)
      setGraphContextOutputBuffer(ptr,i,buffer[i],inputShapeBuffers[i],specialShapeInfo != nullptr ? specialShapeInfo[i] : nullptr);
    else {
      setGraphContextOutputBuffer(ptr,i, nullptr,inputShapeBuffers[i],specialShapeInfo);
    }

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
void execPairwiseTransform(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                           const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                           const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                           const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execPairwiseTransform(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                               dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(),
                                               hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execPairwiseTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                               const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                               const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                               const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execPairwiseBoolTransform(
        nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, dbY->primary(), hYShapeInfo,
        dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execReduceFloat(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                     const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceFloatScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                               extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                    const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                    const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceSameScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                              extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                    const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                    const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceBoolScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                              extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                    const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                    const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceLongScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                              extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execReduceFloat2(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                      const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                      const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                      const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape) {
  try {
    auto dimension = reinterpret_cast<int *>(dbDimension->primary());
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    const auto zLen = shape::length(hZShapeInfo);

    std::vector<sd::LongType> dimensions(dimension, dimension + dimensionLength);

    const sd::LongType *zShapeInfoH = hZShapeInfo;
    const sd::LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<sd::LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceFloat(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                         extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD,
                                         dims.data(), dims.size());
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceBool2(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                     const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape) {
  try {
    auto dimension = reinterpret_cast<int *>(dbDimension->primary());
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    std::vector<sd::LongType> dimensions(dimension, dimension + dimensionLength);

    const auto zLen = shape::length(hZShapeInfo);

    const sd::LongType *zShapeInfoH = hZShapeInfo;
    const sd::LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo)) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<sd::LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceBool(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                        extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD,
                                        dims.data(), dims.size());
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceSame2(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                     const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape) {
  try {
    auto dimension = reinterpret_cast<int *>(dbDimension->primary());
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    std::vector<sd::LongType> dimensions(dimension, dimension + dimensionLength);

    const auto zLen = shape::length(hZShapeInfo);

    const sd::LongType *zShapeInfoH = hZShapeInfo;
    const sd::LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<sd::LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceSame(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                        extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD,
                                        dims.data(), dims.size());
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduceLong2(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                     const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                     const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                     const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape) {
  try {
    auto dimension = reinterpret_cast<int *>(dbDimension->primary());
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    std::vector<sd::LongType> dimensions(dimension, dimension + dimensionLength);

    const auto zLen = shape::length(hZShapeInfo);

    const sd::LongType *zShapeInfoH = hZShapeInfo;
    const sd::LongType *zShapeInfoD = dZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, dimensions);
      zShapeInfoH = reinterpret_cast<sd::LongType const *>(zPack->primary());
      zShapeInfoD = reinterpret_cast<sd::LongType const *>(zPack->special());
    }

    std::vector<sd::LongType> dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), dimensions) : std::vector<sd::LongType>();
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execReduceLong(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                        extraParams, dbZ->primary(), zShapeInfoH, dbZ->special(), zShapeInfoD,
                                        dims.data(), dims.size());
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});

  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execReduce3(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                 const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                 const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execReduce3(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                     extraParams, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo,
                                     dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execReduce3Scalar(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                       const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                       const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                       const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execReduce3Scalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                           extraParams, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo,
                                           dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execReduce3Tad(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                    const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                    const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                    const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                    const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape,
                    const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets,
                    const sd::LongType *yTadOnlyShapeInfo, const sd::LongType *yTadOffsets) {
  try {
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    if (extraPointers == nullptr || extraPointers[2] == 0) {
      OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
      NativeOpExecutioner::execReduce3(
          LaunchContext::defaultContext(), opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams,
          dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(),
          dZShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
      OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
    } else {
      // going tad-way
      auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);

      auto hTADShapeInfo = tadPack.primaryShapeInfo();
      auto hTADOffsets = tadPack.primaryOffsets();

      OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
      NativeOpExecutioner::execReduce3TAD(
          LaunchContext::defaultContext(), opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams,
          dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(), hZShapeInfo, dbZ->special(),
          dZShapeInfo, dimension, dimensionLength, hTADShapeInfo, hTADOffsets, nullptr, nullptr);
      OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
    }
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execScalar(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbScalar, const sd::LongType *hScalarShapeInfo,
                const sd::LongType *dScalarShapeInfo, void *extraParams) {
  try {
#if defined(HAVE_VEDA)
    auto helperIsUsed =
        execHelperScalar(opNum, dbX, hXShapeInfo, dbScalar, hScalarShapeInfo, dbZ, hZShapeInfo, extraParams);
    if (!helperIsUsed) {
#endif
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbScalar});
    NativeOpExecutioner::execScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                    dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, dbScalar->primary(),
                                    hScalarShapeInfo, dbScalar->special(), dScalarShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbScalar});
#if defined(HAVE_VEDA)
    }
#endif
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBool(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                    const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                    const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbScalar, const sd::LongType *hScalarShapeInfo,
                    const sd::LongType *dScalarShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execScalarBool(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                        dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, dbScalar->primary(),
                                        hScalarShapeInfo, dbScalar->special(), dScalarShapeInfo, extraParams);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param hX
 * @param hXShapeInfo
 * @param extraParams
 */
void execSummaryStatsScalar(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX,
                            const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, void *extraParams,
                            OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo,
                            bool biasCorrected) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execSummaryStatsScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(),
                                                dXShapeInfo, extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(),
                                                dZShapeInfo, biasCorrected);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execSummaryStats(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                      const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                      const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, bool biasCorrected) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execSummaryStats(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                          extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo,
                                          biasCorrected);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execSummaryStatsTad(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                         const sd::LongType *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                         const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo,
                         OpaqueDataBuffer *dbDimension, const sd::LongType *hDimensionShape,
                         const sd::LongType *dDimensionShape, bool biasCorrected, const sd::LongType *tadShapeInfo,
                         const sd::LongType *tadOffsets) {
  try {
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execSummaryStats(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                          extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo,
                                          dimension, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
void execTransformFloat(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                        const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                        const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformFloat(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                            dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams,
                                            nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformSame(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                       const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                       const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformSame(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                           dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams,
                                           nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformBool(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                       const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                       const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformBool(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                           dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams,
                                           nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformAny(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                      const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                      const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformAny(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                          dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams,
                                          nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execTransformStrict(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                         const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                         const sd::LongType *dZShapeInfo, void *extraParams) {
  try {
#if defined(HAVE_VEDA)
    auto helperIsUsed = execHelperTransformStrict(opNum, dbX, hXShapeInfo, dbZ, hZShapeInfo, extraParams);
    if (!helperIsUsed) {
#endif
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execTransformStrict(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                             dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraParams,
                                             nullptr, nullptr);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
#if defined(HAVE_VEDA)
    }
#endif
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execReduce3All(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                    const sd::LongType *dXShapeInfo, void *extraParamsVals, OpaqueDataBuffer *dbY,
                    const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                    const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                    const sd::LongType *hDimensionShape, const sd::LongType *dDimensionShape,
                    const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets, const sd::LongType *yTadShapeInfo,
                    const sd::LongType *yOffsets) {
  try {
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    auto dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execReduce3All(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                        extraParamsVals, dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo,
                                        dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, dimension,
                                        dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 * Concatneate multi array of the same shape together
 * along a particular dimension
 */
void specialConcat(sd::Pointer *extraPointers, int dimension, int numArrays, sd::Pointer *data,
                   sd::Pointer *inputShapeInfo, void *hZ, sd::LongType const *hZShapeInfo, sd::Pointer *tadPointers,
                   sd::Pointer *offsetPointers) {
  try {
    auto zType = sd::ArrayOptions::dataType(hZShapeInfo);

    BUILD_SINGLE_SELECTOR(zType, sd::SpecialMethods,
                          ::concatCpuGeneric(dimension, numArrays, data, inputShapeInfo, hZ, hZShapeInfo),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 * This is dummy method for JNI compatibility
 * Since we'll use this from java, jni compiler would like to have method no matter what.
 */
void initializeDevicesAndFunctions() {}

void initializeFunctions(sd::Pointer *functions) { sd::BlasHelper::getInstance().initializeFunctions(functions); }

/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
sd::Pointer mallocHost(sd::LongType memorySize, int flags) {
#if defined(SD_ALIGNED_ALLOC)
  return static_cast<sd::Pointer *>(
      aligned_alloc(SD_DESIRED_ALIGNMENT, (memorySize + SD_DESIRED_ALIGNMENT - 1) & (-SD_DESIRED_ALIGNMENT)));
#else
  return reinterpret_cast<sd::Pointer>(new int8_t[memorySize]);
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
sd::Pointer mallocDevice(sd::LongType memorySize, int deviceId, int flags) {
  // not supported
  return 0L;
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int freeHost(sd::Pointer pointer) {
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
int freeDevice(sd::Pointer pointer, int deviceId) {
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

sd::Pointer createContext() { return 0L; }

sd::Pointer createStream() { return 0L; }

sd::Pointer createEvent() { return 0L; }

int getDeviceMajor(int deviceId) { return 0; }

int getDeviceMinor(int deviceId) { return 0; }

int registerEvent(sd::Pointer event, sd::Pointer stream) { return 0L; }

int setDevice(int deviceId) { return 0L; }

sd::LongType getDeviceFreeMemory(int deviceId) { return 0L; }

sd::LongType getDeviceFreeMemoryDefault() { return 0L; }

sd::LongType getDeviceTotalMemory(int deviceId) { return 0L; }

int memcpySync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int memcpyAsync(sd::Pointer dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int memsetSync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int memsetAsync(sd::Pointer dst, int value, sd::LongType size, int flags, sd::Pointer reserved) { return 0L; }

int destroyEvent(sd::Pointer event) { return 0L; }

int streamSynchronize(sd::Pointer stream) { return 0L; }

int eventSynchronize(sd::Pointer event) { return 0L; }

int getAvailableDevices() { return 0L; }

void enableDebugMode(bool reallyEnable) { sd::Environment::getInstance().setDebug(reallyEnable); }

void enableVerboseMode(bool reallyEnable) { sd::Environment::getInstance().setVerbose(reallyEnable); }

void setGridLimit(int gridSize) {
  // no-op
}

sd::TadPack *tadOnlyShapeInfo(sd::LongType const *hXShapeInfo, LongType *dimension, int dimensionLength) {
  auto pack = new TadPack();
  try {
    *pack = sd::ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }

  return pack;
}

sd::LongType const *getPrimaryShapeInfo(sd::TadPack *pack) {
  return const_cast<sd::LongType *>(pack->primaryShapeInfo());
}

sd::LongType const *getPrimaryOffsets(sd::TadPack *pack) { return const_cast<sd::LongType *>(pack->primaryOffsets()); }

sd::LongType const *getSpecialShapeInfo(sd::TadPack *pack) {
  return const_cast<sd::LongType *>(pack->specialShapeInfo());
}

sd::LongType const *getSpecialOffsets(sd::TadPack *pack) { return const_cast<sd::LongType *>(pack->specialOffsets()); }

sd::LongType getNumberOfTads(sd::TadPack *pack) { return pack->numberOfTads(); }

int getShapeInfoLength(sd::TadPack *pack) { return pack->shapeInfoLength(); }

int memcpyConstantAsync(sd::LongType dst, sd::Pointer src, sd::LongType size, int flags, sd::Pointer reserved) {
  // no-op
  return 0L;
}

sd::Pointer getConstantSpace() {
  // no-op
  return 0L;
}

template <typename T>
void pullRowsGeneric(void *vx, sd::LongType const *hXShapeInfo, void *vz, sd::LongType const *hZShapeInfo, const int n,
                     sd::LongType const *indexes, sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets,
                     sd::LongType const *zTadShapeInfo, sd::LongType const *zTadOffsets) {
  auto hX = reinterpret_cast<T *>(vx);
  auto hZ = reinterpret_cast<T *>(vz);

  const auto xEWS = shape::elementWiseStride(tadShapeInfo);
  const auto zEWS = shape::elementWiseStride(zTadShapeInfo);
  const auto tadLength = shape::length(tadShapeInfo);

  int elementsPerThread = n / TAD_THRESHOLD;
  int _threads = sd::math::sd_max<int>(1, elementsPerThread);
  _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

  auto func = PRAGMA_THREADS_FOR {
    for (auto idx = start; idx < stop; idx++) {
      auto xTadOffsetForBlock = tadOffsets[indexes[idx]];
      auto zTadOffsetForBlock = zTadOffsets[idx];

      auto rX = hX + xTadOffsetForBlock;
      auto rZ = hZ + zTadOffsetForBlock;

      if (xEWS == 1 && zEWS == 1) {
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < tadLength; i++) {
          rZ[i] = rX[i];
        }
      } else if (xEWS >= 1 && zEWS >= 1) {
        PRAGMA_OMP_SIMD
        for (sd::LongType i = 0; i < tadLength; i++) {
          rZ[i * zEWS] = rX[i * xEWS];
        }
      } else {
        for (sd::LongType i = 0; i < tadLength; i++) {
          auto xOffset = xTadOffsetForBlock + shape::getIndexOffset(i, tadShapeInfo);
          auto zOffset = zTadOffsetForBlock + shape::getIndexOffset(i, zTadShapeInfo);
          hZ[zOffset] = hX[xOffset];
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, n, 1, _threads);
}

void pullRows(sd::Pointer *extraPointers, OpaqueDataBuffer *dbX, sd::LongType const *hXShapeInfo,
              sd::LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, sd::LongType const *hZShapeInfo,
              sd::LongType const *dZShapeInfo, sd::LongType n, sd::LongType *indexes, sd::LongType const *tadShapeInfo,
              sd::LongType const *tadOffsets, sd::LongType const *zTadShapeInfo, sd::LongType const *zTadOffsets) {
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, pullRowsGeneric,
                          (dbX->primary(), hXShapeInfo, dbZ->primary(), hZShapeInfo, n, indexes, tadShapeInfo,
                              tadOffsets, zTadShapeInfo, zTadOffsets),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

template <typename T>
void tearGeneric(void *vx, sd::LongType const *hXShapeInfo, sd::Pointer *targets, sd::LongType const *hZShapeInfo,
                 sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets) {
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
        for (sd::LongType j = 0; j < tadLength; j++) {
          hZ[j] = s[j];
        }
      } else if (zEWS > 0 && tadEWS > 0) {
        PRAGMA_OMP_SIMD
        for (sd::LongType j = 0; j < tadLength; j++) {
          hZ[j * zEWS] = s[j * tadEWS];
        }
      } else {
        for (sd::LongType j = 0; j < tadLength; j++)
          hZ[shape::getIndexOffset(j, hZShapeInfo)] = s[shape::getIndexOffset(j, tadShapeInfo)];
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numTads);
}

void tear(sd::Pointer *extraPointers, OpaqueDataBuffer *dbX, sd::LongType const *hXShapeInfo,
          sd::LongType const *dXShapeInfo, sd::Pointer *targets, sd::LongType const *hZShapeInfo,
          sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets) {
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, tearGeneric,
                          (dbX->primary(), hXShapeInfo, targets, hZShapeInfo, tadShapeInfo, tadOffsets),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void average(sd::Pointer *extras, sd::Pointer *hX, const sd::LongType *hXShapeInfo, sd::Pointer *dX,
             const sd::LongType *dXShapeInfo, void *z, const sd::LongType *hZShapeInfo, void *dz,
             const sd::LongType *dZShapeInfo, int n, sd::LongType length, bool propagate) {
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::averageGeneric(hX, z, hZShapeInfo, n, length, propagate),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void accumulate(sd::Pointer *extras, sd::Pointer *hX, sd::LongType const *hXShapeInfo, sd::Pointer *dX,
                sd::LongType const *dXShapeInfo, void *hz, sd::LongType const *hZShapeInfo, void *dz,
                sd::LongType const *dZShapeInfo, int n, sd::LongType length) {
  try {
    auto xType = sd::ArrayOptions::dataType(hXShapeInfo);

    BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::accumulateGeneric(hX, hz, hZShapeInfo, n, length),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void enableP2P(bool enable) {
  // no-op
}

void encodeThresholdP1(sd::Pointer *extraPointers, void *hX, sd::LongType const *hXShapeInfo, sd::LongType N, int *dz,
                       float threshold) {
  // TODO: to be implemented
}

void encodeThresholdP2Int(sd::Pointer *extraPointers, int *hX, sd::LongType N, int *dz) {
  // TODO: to be implemented
}

void encodeThresholdP3(sd::Pointer *extraPointers, void *hX, sd::LongType const *hXShapeInfo, int *offsets,
                       sd::LongType N, int *dz) {
  // offsets won't be used here

  // TODO: to be implemented
}

void decodeThreshold(sd::Pointer *extraPointers, void *hX, sd::LongType N, void *dz, const sd::LongType *hZShapeInfo) {
  // TODO: to be implemented
}

bool isP2PAvailable() {
  // always TRUE for cpu backend
  return true;
}

void checkP2P() {
  // no-op
}

void decodeBitmap(sd::Pointer *extraPointers, void *hX, sd::LongType N, void *dz, sd::LongType const *hZShapeInfo) {
  NativeOpExecutioner::decodeBitmap(hX, N, dz, hZShapeInfo);
}

template <typename T>
void shuffleGeneric(void **hX, sd::LongType *const *hXShapeInfo, void **dz, sd::LongType *const *hZShapeInfo, int N,
                    int *shuffleMap, sd::LongType *const *tadOnlyShapeInfo, sd::LongType *const *tadOffsets) {
  auto dX = reinterpret_cast<T **>(hX);
  auto dZ = reinterpret_cast<T **>(dz);

  auto func = PRAGMA_THREADS_FOR {
    for (auto f = start; f < stop; f++) {
      auto hX = reinterpret_cast<T *>(dX[f]);
      // auto hZ = reinterpret_cast<T *>(dZ[f]);

      auto xShapeInfo = hXShapeInfo[f];
      auto tadOffset = reinterpret_cast<sd::LongType *>(tadOffsets[f]);

      const auto tadLength = shape::length(tadOnlyShapeInfo[f]);
      auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo[f]);
      auto tadRank = shape::rank(tadOnlyShapeInfo[f]);
      auto numTads = shape::length(hXShapeInfo[f]) / tadLength;

      auto tadShape = shape::shapeOf(tadOnlyShapeInfo[f]);
      auto tadStride = shape::stride(tadOnlyShapeInfo[f]);

      if (shape::rank(xShapeInfo) == 1) {
        auto xLength = shape::length(xShapeInfo);
        auto ews = shape::elementWiseStride(xShapeInfo);
        for (sd::LongType r = 0; r < xLength; r++) {
          auto swapIdx = shuffleMap[r];
          if (swapIdx < 0) continue;

          sd::math::sd_swap<T>(hX[r * ews], hX[swapIdx * ews]);
        }
      } else {
        for (sd::LongType r = 0; r < numTads; r++) {
          if (shuffleMap[r] < 0) continue;

          auto oldOffset = tadOffset[r];
          auto newOffset = tadOffset[shuffleMap[r]];

          auto rX = hX + oldOffset;
          auto rY = hX + newOffset;

          if (tadEWS == 1) {
            for (sd::LongType i = 0; i < tadLength; i++) {
              sd::math::sd_swap<T>(rX[i], rY[i]);
            }
          } else {
            for (sd::LongType i = 0; i < tadLength; i++) {
              auto offset = shape::getIndexOffset(i, tadOnlyShapeInfo[f]);
              sd::math::sd_swap<T>(hX[offset + oldOffset], hX[offset + newOffset]);
            }
          }
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, N);
}

void shuffle(sd::Pointer *extras, sd::Pointer *hX, sd::Pointer *hXShapeInfo, sd::Pointer *dX, sd::Pointer *dXShapeInfo,
             sd::Pointer *hz, sd::Pointer *hZShapeInfo, sd::Pointer *dz, sd::Pointer *dZShapeInfo, int N,
             int *shuffleMap, sd::Pointer *tadShapeInfo, sd::Pointer *tadOffsets) {
  try {
    auto xShape = reinterpret_cast<sd::LongType *const *>(hXShapeInfo);
    auto zShape = reinterpret_cast<sd::LongType *const *>(hZShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<sd::LongType *const *>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<sd::LongType *const *>(tadOffsets);

    auto xType = sd::ArrayOptions::dataType(xShape[0]);

    BUILD_SINGLE_SELECTOR(xType, shuffleGeneric, (hX, xShape, hz, zShape, N, shuffleMap, tadOnlyShapeInfo, tadOffset),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

bool isExperimentalEnabled() { return sd::Environment::getInstance().isExperimentalBuild(); }

void setOmpMinThreads(int threads) {
  // TODO: to be implemented
}

int getDevice() { return 0; }

void execScalarTad(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, sd::LongType const *hXShapeInfo,
                   sd::LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, sd::LongType const *hZShapeInfo,
                   sd::LongType const *dZShapeInfo, OpaqueDataBuffer *dbScalars, sd::LongType const *hScalarShapeInfo,
                   sd::LongType const *dScalarShapeInfo, void *extraParams, OpaqueDataBuffer *dbDimension,
                   sd::LongType const *hDimensionShape, sd::LongType const *dDimensionShape,
                   sd::LongType const *tadShapeInfo, sd::LongType const *tadOffsets, sd::LongType const *tadShapeInfoZ,
                   sd::LongType const *tadOffsetsZ) {
  try {
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execScalar(nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                    extraParams, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo,
                                    dbScalars->primary(), hScalarShapeInfo, dbScalars->special(), dScalarShapeInfo,
                                    dimension, shape::length(hDimensionShape), tadShapeInfo, tadOffsets, tadShapeInfoZ,
                                    tadOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execScalarBoolTad(sd::Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, const sd::LongType *hXShapeInfo,
                       const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ, const sd::LongType *hZShapeInfo,
                       const sd::LongType *dZShapeInfo, OpaqueDataBuffer *dbScalars,
                       const sd::LongType *hScalarShapeInfo, const sd::LongType *dScalarShapeInfo, void *extraParams,
                       OpaqueDataBuffer *dbDimension, const sd::LongType *hDimensionShape,
                       const sd::LongType *dDimensionShape, const sd::LongType *tadShapeInfo,
                       const sd::LongType *tadOffsets, const sd::LongType *tadShapeInfoZ,
                       const sd::LongType *tadOffsetsZ) {
  try {
    auto dimension = reinterpret_cast<sd::LongType *>(dbDimension->primary());
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execScalarBool(
        nullptr, opNum, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo, extraParams, dbZ->primary(),
        hZShapeInfo, dbZ->special(), dZShapeInfo, dbScalars->primary(), hScalarShapeInfo, dbScalars->special(),
        dScalarShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }

  return name;
}

void execAggregate(sd::Pointer *extraPointers, int opNum, void **arguments, int numArguments,
                   sd::LongType **shapeArguments, int numShapeArguments, int *indexArguments, int numIndexArguments,
                   int **intArrays, int numIntArrays, void *realArguments, int numRealArguments, sd::DataType dtype) {}

void batchExecutor(sd::Pointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                   int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments,
                   sd::DataType dtype) {}

void execAggregateBatch(sd::Pointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                        int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments,
                        sd::DataType dtype) {}

void execRandom(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer *dbZ,
                const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo,
                                    extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom3(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer *dbX,
                 const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbY,
                 const sd::LongType *hYShapeInfo, const sd::LongType *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX, dbY});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                    dbY->primary(), hYShapeInfo, dbY->special(), dYShapeInfo, dbZ->primary(),
                                    hZShapeInfo, dbZ->special(), dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execRandom2(sd::Pointer *extraPointers, int opNum, sd::Pointer state, OpaqueDataBuffer *dbX,
                 const sd::LongType *hXShapeInfo, const sd::LongType *dXShapeInfo, OpaqueDataBuffer *dbZ,
                 const sd::LongType *hZShapeInfo, const sd::LongType *dZShapeInfo, void *extraArguments) {
  try {
    OpaqueDataBuffer::preparePrimaryUse({dbZ}, {dbX});
    NativeOpExecutioner::execRandom(nullptr, opNum, state, dbX->primary(), hXShapeInfo, dbX->special(), dXShapeInfo,
                                    dbZ->primary(), hZShapeInfo, dbZ->special(), dZShapeInfo, extraArguments);
    OpaqueDataBuffer::registerPrimaryUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

sd::Pointer initRandom(sd::Pointer *extraPointers, long seed, long bufferSize, sd::Pointer ptrToBuffer) {
  try {
    auto generator = new graph::RandomGenerator(seed, seed);

    return (sd::Pointer)generator;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());

    return nullptr;
  }
}

void refreshBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) {
  auto generator = reinterpret_cast<sd::graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void reSeedBuffer(sd::Pointer *extraPointers, long seed, sd::Pointer ptrRandom) {
  auto generator = reinterpret_cast<sd::graph::RandomGenerator *>(ptrRandom);

  generator->setStates(seed);
}

void destroyRandom(sd::Pointer ptrBuffer) {
  auto buffer = reinterpret_cast<sd::graph::RandomGenerator *>(ptrBuffer);
  delete buffer;
}

/**
 * Return the length of a shape buffer
 * based on the pointer
 * @param buffer  the buffer pointer to check
 * @return
 */
int lengthForShapeBufferPointer(sd::Pointer buffer) {
  auto shapeBuffer = reinterpret_cast<sd::LongType *>(buffer);
  return shape::shapeInfoLength(shape::rank(shapeBuffer));
}

/**
 * The pointer to get the address for
 *
 * @param address the address to get the pointer
 * @return the pointer for the given address
 */

sd::Pointer pointerForAddress(sd::LongType address) { return reinterpret_cast<sd::Pointer>(address); }

void sort(sd::Pointer *extraPointers, void *hX, const sd::LongType *hXShapeInfo, void *dX,
          const sd::LongType *dXShapeInfo, bool descending) {
  try {
    NativeOpExecutioner::execSort(hX, hXShapeInfo, descending);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(sd::Pointer *extraPointers, void *hX, const sd::LongType *hXShapeInfo, void *dX,
             const sd::LongType *dXShapeInfo, LongType *dimension, int dimensionLength, const sd::LongType *tadShapeInfo,
             const sd::LongType *tadOffsets, bool descending) {
  try {
    NativeOpExecutioner::execSort(hX, hXShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortCooIndices(sd::Pointer *extraPointers, sd::LongType *indices, void *x, sd::LongType length,
                    const sd::LongType *xShapeInfo) {
  try {
    NativeOpExecutioner::execSortCooIndices(indices, x, length, xShapeInfo);

  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}



void ravelMultiIndex(sd::Pointer *extraPointers, sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                     sd::LongType *shapeInfo, int mode) {
  NativeOpExecutioner::execRavelMultiIndex(indices, flatIndices, length, shapeInfo, mode);
}

void unravelIndex(sd::Pointer *extraPointers, sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                  sd::LongType *shapeInfo) {
  NativeOpExecutioner::execUnravelIndex(indices, flatIndices, length, shapeInfo);
}

sd::LongType encodeBitmap(sd::Pointer *extraPointers, void *hX, sd::LongType const *hXShapeInfo, sd::LongType N,
                          LongType *dz, float threshold) {
  return NativeOpExecutioner::encodeBitmap(hX, hXShapeInfo, N, dz, threshold);
}

sd::LongType *mmapFile(sd::Pointer *extraPointers, const char *fileName, sd::LongType length) {
  auto hZ = new sd::LongType[2];
  errno = 0;
  try {
#if defined(_WIN32) || defined(_WIN64)
    _mmap(hZ, static_cast<size_t>(length), fileName);
    _mmap(hZ, static_cast<size_t>(length), fileName);
#else
    int fd = open(fileName, O_RDWR, 0);  // checking for failed fopen
    if (fd < 0) {
      sd_printf("Errno: %i\n", errno);
      throw std::runtime_error("Failed to open file for MMAP");
    }
    void *ptr = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // check for failed allocation
    if (ptr == MAP_FAILED) return nullptr;

    hZ[0] = (sd::LongType)ptr;
    hZ[1] = fd;

#endif

    return hZ;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void munmapFile(sd::Pointer *extraPointers, sd::LongType *ptrMap, sd::LongType length) {
  munmap((sd::Pointer)ptrMap[0], length);
#if defined(_WIN32) || defined(_WIN64)
  CloseHandle(reinterpret_cast<HANDLE>(ptrMap[1]));
#else
  close((int)ptrMap[1]);
#endif

  delete[] ptrMap;
}

sd::graph::ResultWrapper *executeFlatGraph(sd::Pointer *extraPointers, sd::Pointer flatBufferPointer) {
  try {
    return sd::graph::GraphExecutioner::executeFlatBuffer(flatBufferPointer);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::LongType getResultWrapperSize(sd::graph::ResultWrapper *ptr) { return ptr->size(); }
sd::Pointer getResultWrapperPointer(sd::graph::ResultWrapper *ptr) { return ptr->pointer(); }

const char *getAllCustomOps() { return sd::ops::OpRegistrator::getInstance().getAllCustomOperations(); }

template <typename T>
SD_INLINE int estimateThresholdGeneric(sd::Pointer *extraPointers, sd::Pointer hX, int N, T threshold) {
  auto buffer = reinterpret_cast<T *>(hX);
  int span = (N / 6) + 8;

  auto func = PRAGMA_REDUCE_LONG {
    int64_t cnt = 0;
    PRAGMA_OMP_SIMD
    for (auto e = start; e < stop; e++) {
      auto v = sd::math::sd_abs<T>(buffer[e]);
      if (v >= threshold) cnt++;
    }

    return cnt;
  };

  return samediff::Threads::parallel_long(
      func, LAMBDA_AL { return _old + _new; }, 0, N);
}

int estimateThreshold(sd::Pointer *extraPointers, sd::Pointer hX, sd::LongType const *hXShapeInfo, int N,
                      float threshold) {
  try {
    auto xType = ArrayOptions::dataType(hXShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, return estimateThresholdGeneric, (extraPointers, hX, N, threshold), SD_FLOAT_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return 0;
  }
}

sd::LongType getShapeListSize(sd::ShapeList *list) { return list->size(); }

sd::LongType const *getShape(sd::ShapeList *list, sd::LongType i) {
  return const_cast<sd::LongType const *>(list->at(i));
}

void deleteShapeList(sd::Pointer shapeList) {
  auto list = reinterpret_cast<sd::ShapeList *>(shapeList);

  // list->destroy();
  delete list;
}

sd::ShapeList *_calculateOutputShapes(sd::Pointer *extraPointers, sd::ops::DeclarableOp *op, sd::Pointer *inputBuffers,
                                      sd::Pointer *inputShapes, int numInputShapes, double *tArgs, int numTArgs,
                                      sd::LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs,
                                      int numDArgs) {

  sd::graph::VariableSpace varSpace;
  Context block(2, &varSpace);
  sd::ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numBArgs; e++) block.getBArguments()->push_back(bArgs[e]);

  for (int e = 0; e < numDArgs; e++) block.getDArguments()->push_back((sd::DataType)dArgs[e]);

  for (int e = 0; e < numInputShapes; e++) {
    auto shape_ = reinterpret_cast<sd::LongType *>(inputShapes[e]);
    if(shape_ == nullptr) {
      throw std::runtime_error("Input shape was null!");
    }

    if((shape_ != nullptr && shape_[0] > SD_MAX_RANK) || shape_[0] < 0) {
      throw std::runtime_error("Input shape rank is invalid. Either > 32 or < 0. Likely corrupt. Please check your input shapes.");
    }

    // we shouldn't copy buffer if that's empty array
    void *buffer_ = sd::ArrayOptions::arrayType(shape_) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

    auto array = new sd::NDArray(buffer_, shape_, varSpace.launchContext(), false);

    // block should contain references to proper variable
    varSpace.putVariable(1, e, array);
    block.pickInput(1, e);

    inShapes.push_back(shape_);
  }

  auto status = op->validateDataTypes(block);
  if (status != sd::Status::OK) throw std::runtime_error("Data types validation failed");

  auto shapeList = op->calculateOutputShape(&inShapes, block);

  if (varSpace.launchContext() != nullptr) shapeList->detach();

  return shapeList;
}

sd::ShapeList *calculateOutputShapes2(sd::Pointer *extraPointers, sd::LongType hash, sd::Pointer *inputBuffers,
                                      sd::Pointer *inputShapes, int numInputShapes, double *tArgs, int numTArgs,
                                      sd::LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs,
                                      int numDArgs) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs,
                                  numIArgs, bArgs, numBArgs, dArgs, numDArgs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

#if defined(__NEC__)
void setGraphContextArgs(OpaqueContext *ctx, int numArr, sd::Pointer *inputArrDataShapePairs, int numIArgs,
                         sd::LongType *iArgsPtr, int numDArgs, int *dArgsPtr, int numTArgs, double *tArgsPtr,
                         int numBArgs, bool *bArgsPtr) {
  if (numIArgs > 0) {
    auto vecPtr = ctx->getIArguments();
    vecPtr->resize(numIArgs);
    auto vecData = vecPtr->data();
    for (int e = 0; e < numIArgs; e++) vecData[e] = iArgsPtr[e];
  }

  if (numDArgs > 0) {
    auto vecPtr = ctx->getDArguments();
    vecPtr->resize(numDArgs);
    auto vecData = vecPtr->data();
    for (int e = 0; e < numDArgs; e++) vecData[e] = (sd::DataType)dArgsPtr[e];
  }

  if (numTArgs > 0) {
    auto vecPtr = ctx->getTArguments();
    vecPtr->resize(numTArgs);
    auto vecData = vecPtr->data();
    for (int e = 0; e < numTArgs; e++) vecData[e] = tArgsPtr[e];
  }

  if (numBArgs > 0) {
    auto vecPtr = ctx->getBArguments();
    vecPtr->clear();
    for (int e = 0; e < numBArgs; e++) vecPtr->push_back(bArgsPtr[e]);
  }

  int i = 0;
  for (int e = 0; e < numArr; e += 2) {
    ctx->setInputArray(i, inputArrDataShapePairs[e], inputArrDataShapePairs[e + 1], nullptr);
    ++i;
  }
}

sd::ShapeList *calculateOutputShapesFromContext(sd::graph::Context *ctx, sd::LongType hash) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    auto status = op->validateDataTypes(*ctx);
    if (status != sd::Status::OK) throw std::runtime_error("Data types validation failed");
    sd::ShapeList inShapes;

    for (int e = 0; e < ctx->width(); e++) {
      auto arr = ctx->array(e);
      auto shape_ = arr->shapeInfo();
      inShapes.push_back(shape_);
    }

    auto shapeList = op->calculateOutputShape(&inShapes, *ctx);

    return shapeList;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

/**
 * @brief Calculate output shapes for the given operation and context and fills the buffer with shape information
 * @note The caller is responsible for setting the handle state nullptr/zeroes before calling and
 * calling the function until it gets nullptr/zeroes in handleState
 *
 * @param ctx  Graph operation context
 * @param hash , it it the hash of the operation
 * @param handleState  the state value to be checked
   @note It should be nullptr for the first time, if the returned handle state
 * is nullptr as well, it means all shapes were filled, if not the caller should call the function to consume all
 shapes until the handle state is nullptr
 * @param outBufferSizeInBytes size of the Buffer for shapes in bytes. @Note It should be enough to fill shape of the
 biggest possible NDArray
 * @param outConcatenatedShapesBuffer pointer to the buffer
 * @return int  returns number of full shapes that was copied into buffer, negative value means there was an error and
 the error can be obtained using  lastErrorCode/lastErrorMessage
 */
int calculateOutputShapesAndFill(sd::graph::Context *ctx, sd::LongType hash, void **handleState,
                                 int outBufferSizeInBytes, sd::LongType *outConcatenatedShapesBuffer) {
  struct ShapeFillerHandle {
    sd::ShapeList *shapeList = nullptr;
    size_t last_index = 0;
  };

  ShapeFillerHandle *sHandle = nullptr;
  sd::ShapeList *shapeList = nullptr;
  if (!handleState) {
    sd_printf("%s\n", "handleState can not be null");
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(2);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("handleState can not be null");
    return -1;
  }
  int requiredMem = shape::shapeInfoLength(SD_MAX_RANK) * sizeof(sd::LongType);
  if (outBufferSizeInBytes < requiredMem) {
    sd_printf(
        "Buffersize (%d bytes ) should be enough (%d bytes ) to fill shape of the biggest possible NDArray "
        "(max-rank: "
        "%d )\n",
        outBufferSizeInBytes, requiredMem, SD_MAX_RANK);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(4);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
        "Buffersize should enough to fill shape of the biggest possible NDArray");
    return -1;
  }

  if (*handleState != nullptr) {
    sHandle = reinterpret_cast<ShapeFillerHandle *>(*handleState);
    shapeList = sHandle->shapeList;
  } else {
    sHandle = new ShapeFillerHandle();
    shapeList = calculateOutputShapesFromContext(ctx, hash);
    sHandle->shapeList = shapeList;
    if (!shapeList) return -1;
  }

  size_t total = shapeList->size();
  size_t old_index = sHandle->last_index;
  size_t i = sHandle->last_index;
  sd::LongType *p = outConcatenatedShapesBuffer;
  sd::LongType *endp = outConcatenatedShapesBuffer + outBufferSizeInBytes / sizeof(sd::LongType);
  while (i < total) {
    const sd::LongType *shape = shapeList->at(i);
    // copy shape buffer
    int len = shape::shapeInfoLength(shape);
    if (p + len > endp) break;
    for (int j = 0; j < len; j++) {
      p[j] = shape[j];
    }
    p += len;
    sHandle->last_index = ++i;
  }

  int count = (sHandle->last_index - old_index);
  // destroy everything in case filling is completed
  if (sHandle->last_index >= shapeList->size()) {
    delete shapeList;
    delete sHandle;
    // reset handle
    sHandle = nullptr;
  }

  // pass handle back to be called again as the buffer was not enough to store all shapes
  *handleState = sHandle;
  return count;
}

#endif

sd::ShapeList *_calculateOutputShapes(sd::Pointer *extraPointers, sd::ops::DeclarableOp *op, sd::Pointer *inputShapes,
                                      int numInputShapes, double *tArgs, int numTArgs, sd::LongType *iArgs,
                                      int numIArgs) {
  Context block(1);
  sd::ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numInputShapes; e++) inShapes.push_back(reinterpret_cast<sd::LongType *>(inputShapes[e]));

  auto shapeList = op->calculateOutputShape(&inShapes, block);
  shapeList->detach();

  return shapeList;
}

sd::ShapeList *calculateOutputShapes(sd::Pointer *extraPointers, sd::LongType hash, sd::Pointer *inputShapes,
                                     int numInputShapes, double *tArgs, int numTArgs, sd::LongType *iArgs,
                                     int numIArgs) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::Status execCustomOp2(sd::Pointer *extraPointers, sd::LongType hash, sd::Pointer opContext) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    auto context = reinterpret_cast<Context *>(opContext);

    return op->execute(context);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::VALIDATION;
  }
}

sd::Status realExec(sd::ops::DeclarableOp *op, sd::Pointer *extraPointers, sd::LongType hash, sd::Pointer *inputBuffers,
                    sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers, sd::Pointer *outputShapes,
                    int numOutputs, double *tArgs, int numTArgs, sd::LongType *iArgs, int numIArgs, bool *bArgs,
                    int numBArgs, bool isInplace) {
  if (op == nullptr) sd_printf("Can't find requested operation: [%lld]\n", hash);

  // we're using the same fake nodeId everywhere here

  std::vector<sd::NDArray *> inputs(numInputs);
  std::vector<sd::NDArray *> outputs(numOutputs);
  std::vector<double> ttArgs(numTArgs);
  std::vector<sd::LongType> iiArgs(numIArgs);
  std::vector<bool> biArgs(numBArgs);

  // filling block now with inputs
  for (int e = 0; e < numInputs; e++) {
    auto shape = reinterpret_cast<sd::LongType *>(inputShapes[e]);
    void *buffer = sd::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[e];

    inputs[e] = new sd::NDArray(buffer, shape);
  }

  // if not inplace - transferring output arrays

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      // we want to keep original output shape intact
      auto shape = shape::copyShape(reinterpret_cast<sd::LongType *>(outputShapes[e]));
      void *buffer = sd::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : outputBuffers[e];

      // FIXME: revisit this.
      bool canNullify = true;
      for (int i = 0; i < numInputs; i++) {
        void *ibuffer = sd::ArrayOptions::arrayType(shape) == ArrayType::EMPTY ? nullptr : inputBuffers[i];
        if (ibuffer == buffer) {
          canNullify = false;
          break;
        }
      }

      if (canNullify)
        memset((uint8_t *)buffer, '\0',
               shape::length(shape) * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape)));

      auto array = new sd::NDArray(buffer, shape);
      outputs[e] = array;

      // and we want to release shape copy once we're done
      delete[] shape;
    }

  for (int e = 0; e < numIArgs; e++) iiArgs[e] = iArgs[e];

  for (int e = 0; e < numTArgs; e++) ttArgs[e] = tArgs[e];

  for (int e = 0; e < numBArgs; e++) biArgs[e] = bArgs[e];

  // hypothetically at this point we have everything filled
  auto hZ = op->execute(inputs, outputs, ttArgs, iiArgs, biArgs, std::vector<sd::DataType>(), isInplace);

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      if (outputs[e]->ordering() != shape::order(reinterpret_cast<sd::LongType *>(outputShapes[e])))
        outputs[e]->streamline(shape::order(reinterpret_cast<sd::LongType *>(outputShapes[e])));
    }

  for (auto v : inputs) delete v;

  for (auto v : outputs) delete v;

  return hZ;
}

sd::Status execCustomOp(sd::Pointer *extraPointers, sd::LongType hash, sd::Pointer *inputBuffers,
                        sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers, sd::Pointer *outputShapes,
                        int numOutputs, double *tArgs, int numTArgs, sd::LongType *iArgs, int numIArgs, bool *bArgs,
                        int numBArgs, bool isInplace) {
  try {
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    return realExec(op, extraPointers, hash, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes,
                    numOutputs, tArgs, numTArgs, iArgs, numIArgs, bArgs, numBArgs, isInplace);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

sd::Status registerGraph(sd::Pointer *extraPointers, sd::LongType graphId, sd::Pointer flatBufferPointer) {
  try {
    auto graph = sd::graph::GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    sd::graph::GraphHolder::getInstance().registerGraph(graphId, graph);

    return sd::Status::OK;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

static VariablesSet *executeStoredGraphT(sd::Pointer *extraPointers, sd::LongType graphId, sd::Pointer *inputBuffers,
                                         sd::Pointer *inputShapes, int *inputIndices, int numInputs) {
  auto graph = sd::graph::GraphHolder::getInstance().cloneGraph(graphId);
  auto varSpace = graph->getVariableSpace();

  std::vector<sd::NDArray *> handles;

  for (int e = 0; e < numInputs; e++) {
    auto idx = inputIndices[e];

    // we'll delete this array later, together with cloned VariableSpace
    auto array = new sd::NDArray(inputBuffers[e], reinterpret_cast<sd::LongType *>(inputShapes[e]));
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

sd::graph::VariablesSet *executeStoredGraph(sd::Pointer *extraPointers, sd::LongType graphId, sd::Pointer *inputBuffers,
                                            sd::Pointer *inputShapes, int *inputIndices, int numInputs) {
  return nullptr;
}

sd::LongType getVariablesSetSize(sd::graph::VariablesSet *set) { return set->size(); }

sd::Status getVariablesSetStatus(sd::graph::VariablesSet *set) { return set->status(); }

sd::graph::Variable *getVariable(sd::graph::VariablesSet *set, sd::LongType i) { return set->at(i); }

int getVariableId(sd::graph::Variable *variable) { return variable->id(); }

int getVariableIndex(sd::graph::Variable *variable) { return variable->index(); }

const char *getVariableName(sd::graph::Variable *variable) { return variable->getName()->c_str(); }

sd::LongType const *getVariableShape(sd::graph::Variable *variable) {
  return const_cast<sd::LongType const *>(variable->getNDArray()->shapeInfo());
}

void *getVariableBuffer(sd::graph::Variable *variable) { return variable->getNDArray()->buffer(); }

sd::Status unregisterGraph(sd::Pointer *extraPointers, sd::LongType graphId) {
  sd::graph::GraphHolder::getInstance().dropGraphAny(graphId);

  return sd::Status::OK;
}

void deletePointerArray(sd::Pointer pointer) {
  auto ptr = reinterpret_cast<sd::Pointer *>(pointer);
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
  auto ptr = reinterpret_cast<sd::LongType *>(pointer);
  delete[] ptr;
}

void deleteVariablesSet(sd::graph::VariablesSet *pointer) { delete pointer; }

const char *getAllOperations() { return sd::OpTracker::getInstance().exportOperations(); }

sd::Pointer getGraphState(sd::LongType id) { return (sd::Pointer) new sd::graph::GraphState(id); }

void deleteGraphState(sd::Pointer state) {
  auto stateP = reinterpret_cast<sd::graph::GraphState *>(state);
  delete stateP;
}

sd::Status execCustomOpWithScope_(sd::Pointer *extraPointers, sd::graph::GraphState *state, sd::LongType opHash,
                                  sd::LongType *scopes, int numScopes, sd::Pointer *inputBuffers,
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
  Node node(OpType_LOGIC, opHash, 0);

  // mapping inputs
  for (int e = 0; e < numInputs; e++) {
    auto buffer = inputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType *>(inputShapes[e]);

    auto array = new sd::NDArray(buffer, shapeInfo, varSpace->launchContext());

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
  if (hZ != sd::Status::OK) return hZ;

  // mapping outputs

  for (int e = 0; e < numOutputs; e++) {
    auto buffer = outputBuffers[e];
    auto shapeInfo = reinterpret_cast<sd::LongType *>(outputShapes[e]);

    NDArray array(buffer, shapeInfo, varSpace->launchContext());

    // now we just put array to VarSpace to the same ID
    // varSpace->putVariable(0, e, array);

    auto t = varSpace->getVariable(0, e)->getNDArray();
    array.assign(t);
  }

  // removing input variables
  for (int e = 0; e < numInputs; e++) {
    varSpace->dropVariable(0, e);
  }

  // after some bla-bla-bla we should have Graph and Node for current op
  return sd::Status::OK;
}

sd::Status execCustomOpWithScope(sd::Pointer *extraPointers, sd::Pointer state, sd::LongType opHash,
                                 sd::LongType *scopes, int numScopes, sd::Pointer *inputBuffers,
                                 sd::Pointer *inputShapes, int numInputs, sd::Pointer *outputBuffers,
                                 sd::Pointer *outputShapes, int numOutputs) {
  try {
    return execCustomOpWithScope_(extraPointers, reinterpret_cast<sd::graph::GraphState *>(state), opHash, scopes,
                                  numScopes, inputBuffers, inputShapes, numInputs, outputBuffers, outputShapes,
                                  numOutputs);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::BAD_INPUT;
  }
}

void deleteResultWrapper(sd::Pointer ptr) {
  // just 0 room for compiler s@!t
  auto p = reinterpret_cast<sd::graph::ResultWrapper *>(ptr);
  delete p;
}

/*
 * TypeDef:
 *     void convertTypes(sd::Pointer *extras, int srcType, sd::Pointer hX, long N, int dstType, sd::Pointer hZ);
 */
void convertTypes(sd::Pointer *extras, int srcType, sd::Pointer hX, sd::LongType N, int dstType, sd::Pointer hZ) {
  auto hx = reinterpret_cast<void *>(hX);
  auto hz = reinterpret_cast<void *>(hZ);

  if (srcType == ND4J_FLOAT8) {
    if (dstType == ND4J_FLOAT8) {
      // convertGeneric<double, sd::float8>(hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      // sd::TypeCast::convertGeneric<sd::float8, sd::int8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      // sd::TypeCast::convertGeneric<sd::float8, sd::uint8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      // sd::TypeCast::convertGeneric<sd::float8, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      // sd::TypeCast::convertGeneric<sd::float8, sd::int16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      // sd::TypeCast::convertGeneric<sd::float8, sd::uint16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
    } else if (dstType == ND4J_FLOAT32) {
      // sd::TypeCast::convertGeneric<sd::float8, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      // sd::TypeCast::convertGeneric<sd::float8, double>(nullptr, hx, N, hz);
    } else {
      sd_debug("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_INT8) {
    if (dstType == ND4J_FLOAT8) {
      // sd::TypeCast::convertGeneric<sd::int8, sd::float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      // convertGeneric<sd::int8, sd::int8>(hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      sd::TypeCast::convertGeneric<int8_t, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertGeneric<int8_t, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      sd::TypeCast::convertGeneric<int8_t, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      // sd::TypeCast::convertGeneric<int8_t, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO: eventually we might want to add it
    } else if (dstType == ND4J_FLOAT32) {
      sd::TypeCast::convertGeneric<int8_t, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      sd::TypeCast::convertGeneric<int8_t, double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_UINT8) {
    if (dstType == ND4J_FLOAT8) {
      //    sd::TypeCast::convertGeneric<uint8_t, sd::float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      sd::TypeCast::convertGeneric<uint8_t, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      sd::TypeCast::convertGeneric<uint8_t, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertGeneric<uint8_t, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      sd::TypeCast::convertGeneric<uint8_t, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //       sd::TypeCast::convertGeneric<uint8_t, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO: still might want to add
    } else if (dstType == ND4J_FLOAT32) {
      sd::TypeCast::convertGeneric<uint8_t, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      sd::TypeCast::convertGeneric<uint8_t, double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_FLOAT16) {
    if (dstType == ND4J_FLOAT8) {
      //    sd::TypeCast::convertGeneric<float16, sd::float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      sd::TypeCast::convertGeneric<float16, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      sd::TypeCast::convertGeneric<float16, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertGeneric<float16, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      sd::TypeCast::convertGeneric<float16, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            sd::TypeCast::convertGeneric<float16, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO: .... ^^^
    } else if (dstType == ND4J_FLOAT32) {
      sd::TypeCast::convertGeneric<float16, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      sd::TypeCast::convertGeneric<float16, double>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_THRESHOLD) {
      sd::TypeCast::convertToThreshold<float16>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_INT16) {
    if (dstType == ND4J_FLOAT8) {
      //   sd::TypeCast::convertGeneric<int16_t, sd::float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      sd::TypeCast::convertGeneric<int16_t, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      sd::TypeCast::convertGeneric<int16_t, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertGeneric<int16_t, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      // sd::TypeCast::convertGeneric<int16_t, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            sd::TypeCast::convertGeneric<int16_t, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
      // TODO...
    } else if (dstType == ND4J_FLOAT32) {
      sd::TypeCast::convertGeneric<int16_t, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      sd::TypeCast::convertGeneric<int16_t, double>(nullptr, hx, N, hz);
    } else {
      printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_FLOAT24) {
  } else if (srcType == ND4J_FLOAT32) {
    if (dstType == ND4J_FLOAT8) {
      //    sd::TypeCast::convertGeneric<float, sd::float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      sd::TypeCast::convertGeneric<float, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      sd::TypeCast::convertGeneric<float, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertGeneric<float, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      sd::TypeCast::convertGeneric<float, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            sd::TypeCast::convertGeneric<float, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
    } else if (dstType == ND4J_DOUBLE) {
      sd::TypeCast::convertGeneric<float, double>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_THRESHOLD) {
      sd::TypeCast::convertToThreshold<float>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_DOUBLE) {
    if (dstType == ND4J_FLOAT8) {
      //   sd::TypeCast::convertGeneric<double, sd::float8>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT8) {
      sd::TypeCast::convertGeneric<double, int8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT8) {
      sd::TypeCast::convertGeneric<double, uint8_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertGeneric<double, float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_INT16) {
      sd::TypeCast::convertGeneric<double, int16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_UINT16) {
      //            sd::TypeCast::convertGeneric<double, uint16_t>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT24) {
    } else if (dstType == ND4J_FLOAT32) {
      sd::TypeCast::convertGeneric<double, float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      //
    } else if (dstType == ND4J_THRESHOLD) {
      sd::TypeCast::convertToThreshold<double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else if (srcType == ND4J_THRESHOLD) {
    if (dstType == ND4J_FLOAT16) {
      sd::TypeCast::convertFromThreshold<float16>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_FLOAT32) {
      sd::TypeCast::convertFromThreshold<float>(nullptr, hx, N, hz);
    } else if (dstType == ND4J_DOUBLE) {
      sd::TypeCast::convertFromThreshold<double>(nullptr, hx, N, hz);
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } else {
    sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
  }
}



void setShapeBuffer(sd::LongType *inputShapeData,sd::DataType dt,sd::LongType *bufferToSet,char order,int elementWiseStride,bool isEmpty) {
  sd::LongType  rank = inputShapeData[0];
  if(rank > SD_MAX_RANK || rank < 0)
    throw std::runtime_error("Invalid rank for shape buffer.");
  std::vector<sd::LongType> shape;
  std::vector<sd::LongType> strides;
  //shape, stride, data type
  for(sd::LongType i = 1; i < rank * 2 + 1; i++) {
    if(i <= rank) {
      shape.push_back(inputShapeData[i]);
    } else if(shape.size() == rank) {
      strides.push_back(inputShapeData[i]);
    }
  }


  auto len = shape::shapeInfoLength(rank);
  auto descriptor = ShapeDescriptor(dt ,order,shape,strides,elementWiseStride);
  if(isEmpty) {
    descriptor._extraProperties = ARRAY_EMPTY;
  }

  auto buffer = descriptor.toShapeInfo();
  for(sd::LongType i = 0; i < len; i++) {
    bufferToSet[i] = buffer[i];
  }




  delete[] buffer;
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

template <typename I>
static void _scatterUpdate(sd::Pointer *extraPointers, int opCode, int numOfSubArrs, void *hX,
                           const sd::LongType *hXShapeInfo, const sd::LongType *hXOffsets, void *dX,
                           const sd::LongType *dXShapeInfo, const sd::LongType *dXOffsets, void *hY,
                           const sd::LongType *hYShapeInfo, const sd::LongType *hYOffsets, void *dY,
                           const sd::LongType *dYShapeInfo, const sd::LongType *dYOffsets, void *vIindexes,
                           const sd::LongType *hIndicesShapeInfo, void *dIindexes,
                           const sd::LongType *dIndicesShapeInfo) {
  auto hIindexes = reinterpret_cast<I *>(vIindexes);
  auto func = PRAGMA_THREADS_DO {
    for (int i = 0; i < numOfSubArrs; ++i) {
      int threadIndex = thread_id;
      const auto xIndex = hIindexes[i];
      const bool isOwner = xIndex < numThreads ? threadIndex == xIndex : threadIndex == xIndex % numThreads;

      if (!isOwner) continue;

      NDArray inSubArr(reinterpret_cast<int8_t *>(hX) + (hXOffsets[hIindexes[i]] * DataTypeUtils::sizeOf(hXShapeInfo)),
                       hXShapeInfo);
      NDArray updSubArr(reinterpret_cast<int8_t *>(hY) + (hYOffsets[i] * DataTypeUtils::sizeOf(hXShapeInfo)),
                        hYShapeInfo);

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
void scatterUpdate(sd::Pointer *extraPointers, int opCode, int numOfSubArrs, void *hX, const sd::LongType *hXShapeInfo,
                   const sd::LongType *hXOffsets, void *dX, const sd::LongType *dXShapeInfo,
                   const sd::LongType *dXOffsets, void *hY, const sd::LongType *hYShapeInfo,
                   const sd::LongType *hYOffsets, void *dY, const sd::LongType *dYShapeInfo,
                   const sd::LongType *dYOffsets, void *hIindexes, const sd::LongType *hIndicesShapeInfo,
                   void *dIindexes, const sd::LongType *dIndicesShapeInfo) {
  auto iType = ArrayOptions::dataType(hIndicesShapeInfo);

  try {
    BUILD_SINGLE_SELECTOR(
        iType, _scatterUpdate,
        (extraPointers, opCode, numOfSubArrs, hX, hXShapeInfo, hXOffsets, dX, dXShapeInfo, dXOffsets, hY, hYShapeInfo,
            hYOffsets, dY, dYShapeInfo, dYOffsets, hIindexes, hIndicesShapeInfo, dIindexes, dIndicesShapeInfo),
        SD_INDEXING_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void inspectArray(sd::Pointer *extraPointers, sd::Pointer buffer, sd::LongType *shapeInfo, sd::Pointer specialBuffer,
                  sd::LongType *specialShapeInfo, sd::Pointer debugInfo) {
  try {
    auto p = reinterpret_cast<sd::DebugInfo *>(debugInfo);
    NDArray array(buffer, shapeInfo);
    sd::DebugHelper::retrieveDebugStatistics(p, &array);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void tryPointer(sd::Pointer extra, sd::Pointer p, int len) {
  try {
    auto buf = reinterpret_cast<int8_t *>(p);
    int cnt = 0;
    for (int i = 0; i < len; i++) cnt += buf[cnt];
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueConstantShapeBuffer *shapeBuffer(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                       char order, sd::LongType ews, bool empty) {
  return shapeBufferEx(rank, shape, strides, dtype, order, ews, empty ? ARRAY_EMPTY : 0);
}

OpaqueConstantShapeBuffer *shapeBufferEx(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                         char order, sd::LongType ews, sd::LongType extras) {
  try {
    auto desc = new  ShapeDescriptor(dtype, order, shape, strides, rank, ews, extras);
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(
        desc);
    delete desc;
    return buffer;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) {
  //implemented in cuda backend: used there only
  //constant buffers otherwise should stick around
}

void deleteConstantDataBuffer(sd::ConstantDataBuffer *ptr) {
  //implemented in cuda backend: used there only
  //constant buffers otherwise should stick around
}

void deleteTadPack(sd::TadPack *ptr) { delete ptr; }

sd::ConstantDataBuffer *constantBufferLong(sd::DataType dtype, const sd::LongType *data, int length) { return nullptr; }

sd::ConstantDataBuffer *constantBufferDouble(sd::DataType dtype, double *data, int length) { return nullptr; }

sd::ConstantDataBuffer *constantBuffer(sd::DataType dtype, sd::ConstantDescriptor *descriptor) {
  try {
    return sd::ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer *dbf) {
  return const_cast<sd::LongType *>(dbf->primary());
}

sd::Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer *dbf) {
  return const_cast<sd::LongType *>(dbf->special());
}

sd::Pointer getConstantDataBufferPrimary(sd::ConstantDataBuffer *dbf) { return dbf->primary(); }
sd::Pointer getConstantDataBufferSpecial(sd::ConstantDataBuffer *dbf) { return dbf->special(); }
sd::LongType getConstantDataBufferLength(sd::ConstantDataBuffer *dbf) { return dbf->length(); }
sd::LongType getConstantDataBufferSizeOf(sd::ConstantDataBuffer *dbf) { return dbf->sizeOf(); }

sd::graph::Context *createGraphContext(int nodeId) {
  try {
    return new sd::graph::Context(nodeId);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}
sd::graph::RandomGenerator *getGraphContextRandomGenerator(sd::graph::Context *ptr) { return &ptr->randomGenerator(); }
void markGraphContextInplace(sd::graph::Context *ptr, bool reallyInplace) { ptr->markInplace(reallyInplace); }
void setGraphContextCudaContext(sd::graph::Context *ptr, void *stream, void *reductionPointer,
                                void *allocationPointer) {}
void setGraphContextInputArray(sd::graph::Context *ptr, int index, void *buffer, void *shapeInfo, void *specialBuffer,
                               void *specialShapeInfo) {
  ptr->setInputArray(index, buffer, shapeInfo, specialBuffer, specialShapeInfo);
}



void setGraphContextOutputArray(sd::graph::Context *ptr, int index, void *buffer, void *shapeInfo, void *specialBuffer,
                                void *specialShapeInfo) {
  ptr->setOutputArray(index, buffer, shapeInfo, specialBuffer, specialShapeInfo);
}

void setGraphContextInputBuffer(OpaqueContext *ptr, int index, OpaqueDataBuffer *buffer, void *shapeInfo,
                                void *specialShapeInfo) {
  if(ptr == nullptr)
    throw std::runtime_error("Context pointer is null!");
  sd::LongType *shapeInfoCast = reinterpret_cast<sd::LongType *>(shapeInfo);
  if(shape::rank(shapeInfoCast) > SD_MAX_RANK || shape::rank(shapeInfoCast) < 0) {
    std::string error;
    error += std::string("Shape Buffer at index ");
    error += std::string(" ");
    error += std::to_string(index);
    error += std::string(" ");
    error += std::string(" was corrupt! This is likely due to deallocation. Please double check the passed in shape  buffer.");
    throw std::runtime_error(error.c_str());
  }

  ptr->setInputArray(index, buffer, shapeInfo, specialShapeInfo);
}

void setGraphContextOutputBuffer(OpaqueContext *ptr, int index, OpaqueDataBuffer *buffer, void *shapeInfo,
                                 void *specialShapeInfo) {
  ptr->setOutputArray(index, buffer, shapeInfo, specialShapeInfo);
}

void setGraphContextTArguments(sd::graph::Context *ptr, double *arguments, int numberOfArguments) {
  ptr->setTArguments(arguments, numberOfArguments);
}
void setGraphContextIArguments(sd::graph::Context *ptr, sd::LongType *arguments, int numberOfArguments) {
  ptr->setIArguments(arguments, numberOfArguments);
}
void setGraphContextBArguments(sd::graph::Context *ptr, bool *arguments, int numberOfArguments) {
  ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) {
  std::vector<sd::DataType> dtypes(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) dtypes[e] = (sd::DataType)arguments[e];

  ptr->setDArguments(dtypes);
}

void deleteGraphContext(sd::graph::Context *ptr) { delete ptr; }

void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) { ptr->allowHelpers(reallyAllow); }

void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) {
  if (execMode < 0 || execMode > 2) execMode = 0;

  ptr->setExecutionMode((samediff::ExecutionMode)execMode);
}

void ctxPurge(OpaqueContext *ptr) { ptr->clearFastPath(); }

sd::graph::RandomGenerator *createRandomGenerator(sd::LongType rootSeed, sd::LongType nodeSeed) {
  return new sd::graph::RandomGenerator(rootSeed, nodeSeed);
}

sd::LongType getRandomGeneratorRootState(sd::graph::RandomGenerator *ptr) { return ptr->rootState(); }

sd::LongType getRandomGeneratorNodeState(sd::graph::RandomGenerator *ptr) { return ptr->nodeState(); }

void setRandomGeneratorStates(sd::graph::RandomGenerator *ptr, sd::LongType rootSeed, sd::LongType nodeSeed) {
  ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(sd::graph::RandomGenerator *ptr, sd::LongType index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(sd::graph::RandomGenerator *ptr, sd::LongType index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(sd::graph::RandomGenerator *ptr, sd::LongType index) {
  return ptr->relativeInt(index);
}

sd::LongType getRandomGeneratorRelativeLong(sd::graph::RandomGenerator *ptr, sd::LongType index) {
  return ptr->relativeLong(index);
}

int getRandomGeneratorNextInt(sd::graph::RandomGenerator *ptr) {
  // to nullify  _nodeState._long ^= (steps ^ 0xdeadbeef);
  // we will use step = 0xdeadbeef
  auto result = ptr->relativeInt(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

sd::LongType getRandomGeneratorNextLong(sd::graph::RandomGenerator *ptr) {
  auto result = ptr->relativeLong(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

float getRandomGeneratorNextFloat(sd::graph::RandomGenerator *ptr) {
  auto result = ptr->relativeT<float>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

double getRandomGeneratorNextDouble(sd::graph::RandomGenerator *ptr) {
  auto result = ptr->relativeT<double>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

void deleteRandomGenerator(sd::graph::RandomGenerator *ptr) { delete ptr; }


void saveNpy(std::string fname, const InteropDataBuffer *data, const unsigned int *shape, const unsigned int ndims,
             std::string mode) {
  auto dtype = data->getDataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(dtype,cnpy::npy_save,(fname,data->getDataBuffer()->primary(),shape,ndims,mode),SD_COMMON_TYPES);
}

int dataTypeFromNpyHeader(void *header) { return (int)cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header)); }

sd::Pointer shapeBufferForNumpy(sd::Pointer npyArray) {
  try {
    cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char *>(npyArray));
    unsigned int shapeSize = arr.shape.size();
    std::vector<sd::LongType> shape(shapeSize);
    bool _empty = false;
    for (unsigned int i = 0; i < shapeSize; i++) {
      shape[i] = arr.shape[i];

      if (arr.shape[i] == 0) _empty = true;
    }

    auto dtype = cnpy::dataTypeFromHeader(reinterpret_cast<char *>(npyArray));

    sd::LongType *shapeBuffer;
    if (shape.size() == 1 && shape[0] == 0) {
      // scalar case
      shapeBuffer = sd::ShapeBuilders::createScalarShapeInfo(dtype);
    } else if (_empty) {
      if (shapeSize > 0)
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
      else
        shapeBuffer = sd::ShapeBuilders::emptyShapeInfo(dtype);
    } else {
      shapeBuffer = sd::ShapeBuilders::createShapeInfo(dtype, arr.fortranOrder ? 'f' : 'c', shape);
    }
    return const_cast<sd::LongType *>(sd::ConstantShapeHelper::getInstance().createFromExisting(shapeBuffer, true));
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void sortByKey(sd::Pointer *extraPointers, void *x, const sd::LongType *xShapeInfo, void *dx,
               const sd::LongType *dxShapeInfo, void *y, const sd::LongType *yShapeInfo, void *dy,
               const sd::LongType *dyShapeInfo, bool descending) {
  try {
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByKey(x, xShapeInfo, y, yShapeInfo, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByValue(sd::Pointer *extraPointers, void *x, const sd::LongType *xShapeInfo, void *dx,
                 const sd::LongType *dxShapeInfo, void *y, const sd::LongType *yShapeInfo, void *dy,
                 const sd::LongType *dyShapeInfo, bool descending) {
  try {
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods, ::sortByValue(x, xShapeInfo, y, yShapeInfo, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByKey(sd::Pointer *extraPointers, void *x, const sd::LongType *xShapeInfo, void *dx,
                  const sd::LongType *dxShapeInfo, void *y, const sd::LongType *yShapeInfo, void *dy,
                  const sd::LongType *dyShapeInfo, LongType *dimension, int dimensionLength, bool descending) {
  try {
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods,
                          ::sortTadByKey(x, xShapeInfo, y, yShapeInfo, dimension, dimensionLength, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByValue(sd::Pointer *extraPointers, void *x, const sd::LongType *xShapeInfo, void *dx,
                    const sd::LongType *dxShapeInfo, void *y, const sd::LongType *yShapeInfo, void *dy,
                    const sd::LongType *dyShapeInfo, LongType *dimension, int dimensionLength, bool descending) {
  try {
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, sd::DoubleMethods,
                          ::sortTadByValue(x, xShapeInfo, y, yShapeInfo, dimension, dimensionLength, descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

sd::LongType getCachedMemory(int deviceId) { return sd::ConstantHelper::getInstance().getCachedAmount(deviceId); }

sd::LaunchContext *defaultLaunchContext() { return LaunchContext::defaultContext(); }

sd::Pointer lcScalarPointer(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcReductionPointer(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcAllocationPointer(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcExecutionStream(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcCopyStream(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcBlasHandle(OpaqueLaunchContext *lc) { return nullptr; }

sd::Pointer lcSolverHandle(OpaqueLaunchContext *lc) { return nullptr; }

int lastErrorCode() { return sd::LaunchContext::defaultContext()->errorReference()->errorCode(); }

const char *lastErrorMessage() { return sd::LaunchContext::defaultContext()->errorReference()->errorMessage(); }

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
void _printHostBuffer(InteropDataBuffer *buffer) {
  auto xType = buffer->dataBuffer()->getDataType();
  sd::LongType len = buffer->dataBuffer()->getNumElements();
  auto buff = buffer->dataBuffer()->template primaryAsT<T>();
  sd_printf("Host buffer: ",0);
  for(int i = 0; i < len; i++) {
    sd_printf("%f ",buff[i]);
  }

  sd_printf("\n",0);
}

void printDeviceBuffer(OpaqueDataBuffer *buffer) {
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
  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(buffer),SD_COMMON_TYPES_ALL);


}

OpaqueDataBuffer *dbAllocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer *allocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  try {
    auto dtype = DataTypeUtils::fromInt(dataType);
    return new sd::InteropDataBuffer(elements * DataTypeUtils::sizeOf(dtype), dtype, allocateBoth);
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

sd::Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) { return dataBuffer->primary(); }

sd::Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) { return dataBuffer->special(); }

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
  delete dataBuffer;
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary,
                                             sd::Pointer special) {
  auto buffer = dbAllocateDataBuffer(0, dataType, false);

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) buffer->setSpecial(special, elements);

  return buffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer primaryBuffer, sd::LongType numBytes) {
  dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer specialBuffer, sd::LongType numBytes) {
  dataBuffer->setSpecial(specialBuffer, numBytes);
}

void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->allocatePrimary(); }

void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) { dataBuffer->dataBuffer()->allocateSpecial(); }

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  try {
    dataBuffer->dataBuffer()->expand(elements * DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueDataBuffer *dbCreateView(OpaqueDataBuffer *dataBuffer, sd::LongType length, sd::LongType offset) {
  return new InteropDataBuffer(*dataBuffer, length, offset);
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

void dbExpand(OpaqueDataBuffer *dataBuffer, sd::LongType elements) { dataBuffer->expand(elements); }

int dbLocality(OpaqueDataBuffer *dataBuffer) { return 0; }

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) { dataBuffer->setDeviceId(deviceId); }

int dbDeviceId(OpaqueDataBuffer *dataBuffer) { return dataBuffer->deviceId(); }

void dbClose(OpaqueDataBuffer *dataBuffer) { dataBuffer->getDataBuffer()->close(); }

void setVedaDeviceLibFolder(std::string path) {
  sd::Environment::getInstance().setVedaDeviceDir(path);
#if defined(HAVE_VEDA)
  VEDA::getInstance();
#endif
}

BUILD_SINGLE_TEMPLATE(template void pullRowsGeneric,
                      (void *, sd::LongType const *, void *, sd::LongType const *, const int, sd::LongType const *,
                          sd::LongType const *, sd::LongType const *, sd::LongType const *, sd::LongType const *),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void tearGeneric,
                      (void *, sd::LongType const *, sd::Pointer *, sd::LongType const *, sd::LongType const *,
                          sd::LongType const *),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void shuffleGeneric,
                      (void **, sd::LongType *const *, void **, sd::LongType *const *, int, int *,
                          sd::LongType *const *, sd::LongType *const *),
                      SD_COMMON_TYPES);
