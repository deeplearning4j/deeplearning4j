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

#include <cuda.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <execution/AffinityManager.h>
#include <graph/GraphExecutioner.h>
#include <graph/GraphHolder.h>
#include <helpers/BlasHelper.h>
#include <helpers/CudaLaunchHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/threshold.h>
#include <legacy/NativeOpExecutioner.h>
#include <legacy/NativeOps.h>
#include <loops/reduce_bool.h>
#include <loops/reduce_long.h>
#include <loops/scalar.h>
#include <loops/transform_any.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/specials_cuda.h>
#include <system/buffer.h>


#include <curand.h>
#include <helpers/DebugHelper.h>

using namespace sd;
#include <loops/special_kernels.h>
#include <ops/declarable/OpRegistrator.h>
#include <execution/cuda/LaunchDims.h>
cudaDeviceProp *deviceProperties;
cudaFuncAttributes *funcAttributes = new cudaFuncAttributes[64];
int blockLimit = 128;
int maxThreads = 512;
bool allowedP2P = false;
bool supportedP2P = false;
#ifdef SD_EXPERIMENTAL_ENABLED
bool experimentalSupport = true;
#else
bool experimentalSupport = false;
#endif


//note we only include this if we're running gcc linux
//and should not be enabled in default builds.
#if defined(SD_GCC_FUNCTRACE)
#include <cxxabi.h>  // needed  __cxa_demangle
#include <dlfcn.h>   // needed for dladdr

#include "exceptions/backward.hpp"
#include "execution/cuda/LaunchDims.h"

// note this is a c++ 17 feature
#ifndef INSTRUMENT_FILE_DEF
#pragma once
#define INSTRUMENT_FILE_DEF 1
FILE* instrumentFile = nullptr;
#endif


// this is mainly a c based function.
extern "C" {



//we need to tell -finstrument-functions not to include the logger otherwise it will recursively
// stack overflow and segfault.
__attribute__((no_instrument_function)) SD_LIB_EXPORT  void writeLog(bool enter,void *this_fn,void *call_site) {
  if(instrumentFile == nullptr) {
    return;
  }
  Dl_info info;
  if (dladdr(this_fn, &info)) {
    int status;
    const char *funcName;
    char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);
    if (status == 0) {
      funcName = demangled  != nullptr ? demangled : "null_demangled";
    } else {
      funcName = info.dli_sname ? info.dli_sname : "null_dli_sname";
    }

    printf(" %s %s (%s)\n",enter ? "enter" : "exit", funcName, info.dli_fname);
    fprintf( instrumentFile," %s %s (%s)\n",enter ? "enter" : "exit", funcName, info.dli_fname);
    if (demangled != nullptr) {
      delete demangled;
      demangled = nullptr;
    }
  } else {
    printf("%s %s\n", enter ? "enter" : "exit","unknown");
    fprintf(instrumentFile, "%s %s\n", enter ? "enter" : "exit","unknown");
    fflush(instrumentFile);
  }
}
//we need to tell -finstrument-functions not to include the logger otherwise it will recursively
// stack overflow and segfault.
__attribute__((no_instrument_function)) SD_LIB_EXPORT void __cyg_profile_func_enter(void *this_fn,
                                                                                    void *call_site) {
  writeLog(true,this_fn, call_site);
}


//we need to tell -finstrument-functions not to include the logger otherwise it will recursively
// stack overflow and segfault.
__attribute__((no_instrument_function)) SD_LIB_EXPORT void __cyg_profile_func_exit  (void *this_fn,
                                                                                     void *call_site) {
  writeLog(false,this_fn, call_site);

}


}

//note this is outside extern C. This is fine.


#endif






int minThreads = 32;

__constant__ char deviceConstantMemory[49152];

void toggleOpTrace(bool opTrace) { ops::OpRegistrator::getInstance().toggleTraceOps(opTrace);
}

void purgeOpTrace() { ops::OpRegistrator::getInstance().purgeOpExecs();
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

void copyBuffer(OpaqueDataBuffer *target, long n,  OpaqueDataBuffer *from, long fromOffset, long targetOffset) {
  OpaqueDataBuffer *copyFrom = dbCreateView(from,n,fromOffset);
  OpaqueDataBuffer *targetView = dbCreateView(target,n,targetOffset);
  const DataBuffer targetBuf = *copyFrom->dataBuffer().get();
  const DataBuffer srcBuf = *targetView->dataBuffer().get();
  DataBuffer::memcpy(targetBuf,srcBuf);
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

char *opName(void *execTrace) {
  ExecTrace *trace = (ExecTrace *) execTrace;
  return const_cast<char *>(trace->opName->c_str());
}

// this method just does type conversion in fancy way
int getDeviceId(Pointer ptrToDeviceId) { return (int)(LongType)ptrToDeviceId; }

/*
 * Basic CUDA constants here: number of blocks per MP
 */
int getDeviceBlockThreshold(int deviceId) {
  int ccMinor = deviceProperties[deviceId].minor;
  int ccMajor = deviceProperties[deviceId].major;

  int blockThreshold = 8;

  if (ccMajor >= 5)
    blockThreshold = 32;
  else if (ccMajor == 3)
    blockThreshold = 16;
  else if (ccMajor < 3)
    blockThreshold = 8;

  return blockThreshold;
}

/*
 * This message returns shared memory threshold value. default overflow ratio is 0.3
 */
int getDeviceSharedThreshold(int deviceId) {
  int ccMinor = deviceProperties[deviceId].minor;
  int ccMajor = deviceProperties[deviceId].major;

  // please note threshold isn't multiple of 32, and that's NOT a mistake

  int shmemThreshold;
  if (ccMajor == 6 && ccMinor == 0)
    shmemThreshold = 65536;
  else if (ccMajor == 6 && ccMinor == 1)
    shmemThreshold = 49152;
  else if (ccMajor == 5 && ccMinor == 2)
    shmemThreshold = 98304;
  else if (ccMajor == 5)
    shmemThreshold = 65536;
  else if (ccMajor == 3 && ccMinor == 7)
    shmemThreshold = 114688;
  else
    shmemThreshold = 49152;

  return shmemThreshold / 0.3;
}

buffer::Buffer<LongType> *createScalarBuffer(cudaStream_t stream) {
  auto scalarShapeInfo = shape::createScalarShapeInfo();
  auto buff = buffer::createBuffer(scalarShapeInfo, shape::shapeInfoLength(2), stream);
  copyDataToGpu(&buff, stream);
  return buff;
}

class ScalarShapeInformation {
 private:
  buffer::Buffer<LongType> *scalarDimension;
  buffer::Buffer<LongType> *scalarShapeInfo;

 public:
  ScalarShapeInformation(cudaStream_t stream) {
    auto scalarDimensionBuff = reinterpret_cast<LongType *>(malloc(sizeof(LongType)));

    CHECK_ALLOC(scalarDimensionBuff, "Failed to allocate ShapeInfoBuffer", sizeof(sd::LongType));

    scalarDimensionBuff[0] = SD_MAX_DIMENSION;
    scalarDimension = buffer::createBuffer(scalarDimensionBuff, 1, stream);
    scalarShapeInfo = createScalarBuffer(stream);
  }
  ~ScalarShapeInformation() {
    freeBuffer(&scalarShapeInfo);
    freeBuffer(&scalarDimension);
  }

  LongType *getShapeInfoHostPointer() { return scalarShapeInfo->data; }

  LongType *getShapeInfoGpuPointer() { return scalarShapeInfo->gData; }

  LongType *getDimensionHostPointer() { return scalarDimension->data; }

  LongType *getDimensionGpuPointer() { return scalarDimension->gData; }
};

template <typename T>
SD_KERNEL  void _printBuffers(void* buffer, LongType bufferLength) {
  T * inputBuffer = reinterpret_cast<T *>(buffer);
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid == 0) {
    printf("DEVICE buffer: ");
  }
  const auto step = gridDim.x * blockDim.x;
  for (int t = tid; t < bufferLength; t += step) {
    if(t == 0) {
      printf("DEVICE buffer: ");
    }
    printf(" %f ",(double) inputBuffer[t]);
    if(t == bufferLength - 1) {
      printf("\n");
    }
  }



}

template <typename T>
void _printHostBuffer(InteropDataBuffer *buffer) {
  auto xType = buffer->dataBuffer()->getDataType();
  LongType len = buffer->dataBuffer()->getNumElements();
  auto buff = buffer->dataBuffer()->template primaryAsT<T>();
  sd_printf("Host buffer: ",0);
  for(int i = 0; i < len; i++) {
    sd_printf("%f ",(double) buff[i]);
  }

  sd_printf("\n",0);
}

template <typename T>
void _printDeviceBuffer(InteropDataBuffer *buffer) {
  auto xType = buffer->dataBuffer()->getDataType();
  LongType len = buffer->dataBuffer()->getNumElements();
  _printBuffers<T><<<256, 512, 1024>>>(buffer->special(),len);
  cudaDeviceSynchronize();
  DebugHelper::checkGlobalErrorCode("print device buffer(...) failed");


}

void printDeviceBuffer(InteropDataBuffer *buffer) {
  auto xType = buffer->dataBuffer()->getDataType();
  sd_printf("Data type %s: ", DataTypeUtils::asString(xType).c_str());

  if(buffer->special() != nullptr) {
    sd_printf("Device pointer address: %d\n", reinterpret_cast<sd::LongType>(buffer->special()));
  } else {
    sd_printf("Device pointer address: none\n",0);
  }
  BUILD_SINGLE_SELECTOR(xType, _printDeviceBuffer,(buffer),SD_COMMON_TYPES_ALL);


  if(buffer->primary() != nullptr) {
    sd_printf("Host pointer address: %d\n",  reinterpret_cast<sd::LongType>(buffer->primary()));
  } else  {
    sd_printf("Host pointer address: none\n",0);
  }

  BUILD_SINGLE_SELECTOR(xType, _printHostBuffer,(buffer),SD_COMMON_TYPES_ALL);

}



void execPairwiseTransform(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                           LongType const *dXShapeInfo, OpaqueDataBuffer *dbY, LongType const *hYShapeInfo,
                           LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                           LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execPairwiseTransform(
        &lc, opNum, dbX->primary(), hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), dbY->primary(), hYShapeInfo,
        dbY->special(), ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(), dbZ->primary(),
        hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        extraParams);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execPairwiseTransformBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                               LongType const *dXShapeInfo, OpaqueDataBuffer *dbY, LongType const *hYShapeInfo,
                               LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                               LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execPairwiseBoolTransform(
        &lc, opNum, dbX->primary(), hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), dbY->primary(), hYShapeInfo,
        dbY->special(), ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(), dbZ->primary(),
        hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        extraParams);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execSummaryStatsScalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                            LongType const *dXShapeInfo, void *extraParams,
                            OpaqueDataBuffer *dbZ,
                            LongType const *hZShapeInfo, LongType const *dZShapeInfo,
                            bool biasCorrected) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execSummaryStatsScalar(
        &lc, opNum, dbX->primary(), hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), extraParams, dbZ->primary(),
        hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        biasCorrected);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execBroadcastBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                       LongType const *dXShapeInfo, OpaqueDataBuffer *dbY, LongType const *hYShapeInfo,
                       LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                       LongType const *dZShapeInfo, void *extraParams, OpaqueDataBuffer *dbDimension,
                       LongType const *hDimensionShape, LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    auto hTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[9]);
    auto tadOnlyShapeInfo = reinterpret_cast<LongType *>(extraPointers[10]);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers[11]);
    auto tadOnlyShapeInfoZ = reinterpret_cast<LongType *>(extraPointers[12]);
    auto tadOffsetsZ = reinterpret_cast<LongType *>(extraPointers[13]);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execBroadcastBool(
        &lc, opNum, dbX->primary(), hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), dbY->primary(), hYShapeInfo,
        dbY->special(), ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(), dbZ->primary(),
        hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        extraParams, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param dY
 * @param dYShapeInfo
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
void execBroadcast(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                   LongType const *dXShapeInfo, OpaqueDataBuffer *dbY, LongType const *hYShapeInfo,
                   LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                   LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension, LongType const *hDimensionShape,
                   LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto hTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[9]);
    auto tadOnlyShapeInfo = reinterpret_cast<LongType *>(extraPointers[10]);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers[11]);
    auto tadOnlyShapeInfoZ = reinterpret_cast<LongType *>(extraPointers[12]);
    auto tadOffsetsZ = reinterpret_cast<LongType *>(extraPointers[13]);

    auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto yType = ArrayOptions::dataType(hYShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execBroadcast(
        &lc, opNum, dbX->primary(), hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), dbY->primary(), hYShapeInfo,
        dbY->special(), ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(), dbZ->primary(),
        hZShapeInfo, dbZ != nullptr ? dbZ->special() : nullptr, ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 */
////////////////////////////////////////////////////////////////////////
void execReduceFloat(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                     LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                     LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceFloatScalar(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special() ,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), extraParams, dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceSame(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                    LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                    LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceSameScalar(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr: dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), extraParams, dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo)  ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceSame2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                     LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                     LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension, LongType const *hDimensionShape,
                     LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    const auto zLen = shape::length(hZShapeInfo);

    std::vector<LongType> dimensions(dimension, dimension + dimensionLength);

    const LongType *zShapeInfoH = hZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), &dimensions) : new std::vector<LongType>();
    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceSame(&lc,
                                        opNum,
                                        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                        hXShapeInfo,
                                        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                        extraParams,
                                        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(zShapeInfoH)->special(),
                                        dims->data(), dims->size());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                     LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                     LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension, LongType const *hDimensionShape,
                     LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    const auto zLen = shape::length(hZShapeInfo);

    std::vector<LongType> dimensions(dimension, dimension + dimensionLength);

    const LongType *zShapeInfoH = hZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceLong(&lc, opNum,
                                        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                        hXShapeInfo,
                                        shape::isEmptyConst(hXShapeInfo) ?  nullptr : dbX->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                        extraParams, dbZ->primary(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(zShapeInfoH) ? nullptr : dbZ->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(zShapeInfoH)->special(),
                                        dims->data(), dims->size());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                    LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                    LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto hTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[9]);
    auto dTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[10]);

    auto reductionPointer = reinterpret_cast<void *>(extraPointers[4]);

    auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (zType != INT64)
      throw datatype_exception::build("execReduceLong wrong Z data type", INT64, zType);

    //TODO hello
    auto xLength = shape::length(hXShapeInfo);
    dim3 launchDims = getReduceDims(xLength);

    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::reduce::ReduceLongFunction,
        ::execReduceScalar(launchDims,
                           stream, opNum,
                           shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), hXShapeInfo,
                           extraParams,
                           shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(), hXShapeInfo,
                           nullptr,
                           0,
                           reductionPointer,
                           dTADShapeInfo),
        SD_COMMON_TYPES, SD_LONG_TYPES);

    DebugHelper::checkErrorCode(stream, "execReduceLong(...) failed");

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceBool2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                     LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                     LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension, LongType const *hDimensionShape,
                     LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    const auto zLen = shape::length(hZShapeInfo);

    const std::vector<LongType> dimensions(dimension, dimension + dimensionLength);

    const LongType *zShapeInfoH = hZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo, &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceBool(&lc,
                                        opNum,
                                        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                        hXShapeInfo, dbX != nullptr ? dbX->special() : nullptr,
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                        extraParams,
                                        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(zShapeInfoH) ? nullptr :  dbZ->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(zShapeInfoH)->special(),
                                        dims->data(), dims->size());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                    LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                    LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto hTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[9]);
    auto dTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[10]);

    auto reductionPointer = reinterpret_cast<void *>(extraPointers[4]);

    auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (zType != BOOL) THROW_EXCEPTION("execReduceBool requires Z operand to have BOOL type");

    auto xLength = shape::length(hXShapeInfo);
    dim3 launchDims = getReduceDims(xLength);


    BUILD_DOUBLE_SELECTOR(
        xType, zType, functions::reduce::ReduceBoolFunction,
        ::execReduceScalar(launchDims,
                           stream,
                           opNum,
                           shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), hXShapeInfo,
                           extraParams,
                           shape::isEmptyConst(hZShapeInfo) ? nullptr :dbZ->special(),
                           dZShapeInfo,
                           hZShapeInfo,
                           nullptr,
                           0,
                           reductionPointer,
                           dTADShapeInfo),
        SD_COMMON_TYPES, SD_BOOL_TYPES);

    DebugHelper::checkErrorCode(stream, "execReduceBool(...) failed");

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
////////////////////////////////////////////////////////////////////////
void execIndexReduce(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                     LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                     LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension, LongType const *hDimensionShape,
                     LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    auto tadPack =
        ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, shape::length(hDimensionShape));

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execIndexReduce(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        extraParams,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        (LongType *)dbDimension->special(), dimensionLength, tadPack->specialShapeInfo(), tadPack->specialOffsets());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 * @param dZ
 * @param dZShapeInfo
 */
////////////////////////////////////////////////////////////////////////
void execReduceFloat2(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                      LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                      LongType const *hZShapeInfo, LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                      LongType const *hDimensionShape, LongType const *dDimensionShape) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    const auto zLen = shape::length(hZShapeInfo);

    std::vector<LongType> dimensions(dimension, dimension + dimensionLength);

    const LongType *zShapeInfoH = hZShapeInfo;

    if (shape::rank(hXShapeInfo) - dimensionLength != shape::rank(hZShapeInfo) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(hZShapeInfo,
                                                                                            &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(hXShapeInfo), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceFloat(&lc,
                                         opNum,
                                         shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                         hXShapeInfo,
                                         shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
                                         ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                         extraParams,
                                         dbZ->primary(),
                                         zShapeInfoH,
                                         shape::isEmptyConst(zShapeInfoH) ? nullptr :  dbZ->special(),
                                         ConstantShapeHelper::getInstance().bufferForShapeInfo(zShapeInfoH)->special(),
                                         dims->data(), dims->size());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

/**
 *
 * @param opNum
 * @param dX
 * @param dXShapeInfo
 * @param extraParams
 */
////////////////////////////////////////////////////////////////////////
void execIndexReduceScalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                           LongType const *dXShapeInfo, void *extraParams,
                           OpaqueDataBuffer *dbZ,
                           LongType const *hZShapeInfo, LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execIndexReduceScalar(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        extraParams,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformSame(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                       LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                       LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformSame(&lc, opNum,
                                           shape::isEmptyConst(hXShapeInfo) ? nullptr :dbX->primary(),
                                           hXShapeInfo,
                                           shape::isEmptyConst(hXShapeInfo) ?  nullptr : dbX->special(),
                                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                           shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                           hZShapeInfo,
                                           shape::isEmptyConst(hZShapeInfo)  ? nullptr : dbZ->special() ,
                                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                           extraParams, tadShapeInfo, tadOffsets);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                       LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                       LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformBool(&lc,
                                           opNum,
                                           shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                           hXShapeInfo,
                                           shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
                                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                           shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                           hZShapeInfo,
                                           shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
                                           ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                           extraParams,
                                           tadShapeInfo,
                                           tadOffsets);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformAny(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                      LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                      LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto streamSpecial = reinterpret_cast<cudaStream_t &>(extraPointers[4]);
    LaunchContext lc(stream, streamSpecial, extraPointers[5], extraPointers[3],
                     reinterpret_cast<int *>(extraPointers[6]));
    NativeOpExecutioner::execTransformAny(&lc,
                                          opNum,
                                          shape::isEmptyConst(hXShapeInfo) ? nullptr :dbX->primary(),
                                          hXShapeInfo,
                                          shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
                                          ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                          shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                          hZShapeInfo,
                                          shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special() ,
                                          ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                          extraParams, nullptr, nullptr);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformStrict(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                         LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                         LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[10] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[11] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformStrict(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(), extraParams,
        tadShapeInfo, tadOffsets);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformFloat(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                        LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                        LongType const *dZShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[10] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[11] : nullptr);

    printf("launching execTransformFloat nativeops\n");
    LaunchContext lc(extraPointers[1],
                     extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformFloat(
        &lc,
        opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special() ,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(), extraParams,
        tadShapeInfo,
        tadOffsets);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void checkP2P() {
  int curDevice = 0;

  cudaGetDevice(&curDevice);

  int devCnt = 0;
  cudaGetDeviceCount(&devCnt);

  if (curDevice < 0 && curDevice > devCnt) curDevice = 0;

  bool tempSupport = true;

  if (devCnt > 1) {
    for (int dX = 0; dX < devCnt; dX++) {
      for (int dY = 0; dY < devCnt; dY++) {
        if (dX == dY) continue;

        int canAccess = 0;
        cudaSetDevice(dX);

        cudaDeviceCanAccessPeer(&canAccess, dX, dY);

        if (!canAccess) {
          tempSupport = false;
          break;
        }
      }
    }

    supportedP2P = tempSupport;

    cudaSetDevice(curDevice);
  } else {
    // if we have only 1 device - we say that we support P2P, since all data will be on 1 device
    supportedP2P = true;
  }
}

void enableP2P(bool enable) {
  if (enable == allowedP2P) return;

  int curDevice = 0;

  cudaGetDevice(&curDevice);

  int devCnt = 0;
  cudaGetDeviceCount(&devCnt);

  if (curDevice < 0 && curDevice > devCnt) curDevice = 0;

  if (devCnt > 1) {
    for (int dX = 0; dX < devCnt; dX++) {
      for (int dY = 0; dY < devCnt; dY++) {
        if (dX == dY) continue;

        int canAccess = 0;
        cudaSetDevice(dX);

        cudaDeviceCanAccessPeer(&canAccess, dX, dY);

        if (canAccess) {
          if (enable) {
            cudaDeviceEnablePeerAccess(dY, 0);
          } else {
            cudaDeviceDisablePeerAccess(dY);
          }
        } else {
          if (Environment::getInstance().isVerbose()) printf("Peer access [%i] -> [%i] isn't possible\n", dX, dY);
        }
      }
    }

    cudaSetDevice(curDevice);
  }

  allowedP2P = enable;

  cudaSetDevice(curDevice);
}

bool isP2PAvailable() { return supportedP2P; }

void initializeDevicesAndFunctions() {
  try {
    int devCnt = 0;
    cudaGetDeviceCount(&devCnt);
    deviceProperties = new cudaDeviceProp[devCnt];
    for (int i = 0; i < devCnt; i++) {
      cudaSetDevice(i);
      cudaGetDeviceProperties(&deviceProperties[i], i);

      cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    }

    cudaSetDevice(0);

    checkP2P();

    // enabling p2p gpu access if it's supported
    if (supportedP2P && devCnt > 1) enableP2P(allowedP2P);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void initializeFunctions(Pointer *functions) { BlasHelper::getInstance().initializeDeviceFunctions(functions);
}


/**
 * This method acquires memory chunk of requested size on host side
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param flags optional parameter
 */
Pointer mallocHost(LongType memorySize, int flags) {
  Pointer pointer;
  // cudaHostAllocMapped |cudaHostAllocPortable
  auto res = cudaHostAlloc(reinterpret_cast<void **>(&pointer), memorySize + 8, cudaHostAllocDefault);
  if (res != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaHostAlloc failed");
  }

  return reinterpret_cast<int8_t *>(pointer);
}

/**
 * This method acquires memory chunk of requested size on specified device
 *
 * @param pointer pointer that'll be used for allocation
 * @param memorySize memory size, in bytes
 * @param ptrToDeviceId pointer to deviceId. For cuda that's just and int, for OpenCL that's pointer to device_id, etc
 * @param flags optional parameter
 */
Pointer mallocDevice(LongType memorySize, int deviceId, int flags) {
  Pointer pointer;
  auto res = cudaMalloc(reinterpret_cast<void **>(&pointer), memorySize + 8);
  if (res != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMalloc failed");
  }

  return reinterpret_cast<int8_t *>(pointer);
}

/**
 * This method releases previously allocated host memory space
 *
 * @param pointer pointer that'll be freed
 */
int freeHost(Pointer pointer) {
  auto res = cudaFreeHost(reinterpret_cast<void *>(pointer));
  if (res != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaFreeHost failed");
  }

  return 1L;
}

/**
 * This method releases previously allocated memory space on device
 *
 * @param pointer pointer that'll be freed
 * @param ptrToDeviceId pointer to deviceId.
 */
int freeDevice(Pointer pointer, int deviceId) {
  auto res = cudaFree(reinterpret_cast<void *>(pointer));

  // we're intentionally skipping
  if (res != 0 && res != 1) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(res);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaFree failed");
  }

  return res == 0 ? 1L : 0L;
}

Pointer createContext() { return 0L; }

Pointer createStream() {
  auto stream = new cudaStream_t();
  auto dZ = cudaStreamCreate(stream);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaStreamCreate failed");
  }

  return stream;
}

Pointer createEvent() {
  Pointer nativeEvent = (Pointer)malloc(sizeof(cudaEvent_t));

  CHECK_ALLOC(nativeEvent, "Failed to allocate new CUDA event buffer", sizeof(cudaEvent_t));

  auto dZ = cudaEventCreateWithFlags(reinterpret_cast<cudaEvent_t *>(&nativeEvent), cudaEventDisableTiming);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventCreateWithFlags failed");
  }

  return nativeEvent;
}

int registerEvent(Pointer event, Pointer stream) {
  auto pEvent = reinterpret_cast<cudaEvent_t *>(&event);
  auto pStream = reinterpret_cast<cudaStream_t *>(stream);

  auto dZ = cudaEventRecord(*pEvent, *pStream);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventRecord failed");
  }

  return 1;
}

int setDevice(int deviceId) {
  AffinityManager::setCurrentDevice(deviceId);
  return 1;
}

LongType getDeviceFreeMemoryDefault() {
  size_t memFree = 0;
  size_t memTotal = 0;

  cudaMemGetInfo(&memFree, &memTotal);

  return (LongType)memFree;
}

LongType getDeviceFreeMemory(int device) {
  int orig = -1;

  cudaGetDevice(&orig);

  if (device >= 0 && device != orig) {
    cudaSetDevice(device);
  }

  size_t memFree = 0;
  size_t memTotal = 0;

  cudaMemGetInfo(&memFree, &memTotal);

  if (device >= 0 && device != orig) {
    cudaSetDevice(orig);
  }

  return (LongType)memFree;
}

LongType getDeviceTotalMemory(int device) {
  int orig = -1;

  cudaGetDevice(&orig);

  if (device >= 0 && device != orig) {
    cudaSetDevice(device);
  }
  size_t memFree = 0;
  size_t memTotal = 0;

  cudaMemGetInfo(&memFree, &memTotal);

  if (device >= 0 && device != orig) {
    cudaSetDevice(orig);
  }

  return (LongType)memTotal;
}

int memcpySync(Pointer dst, Pointer src, LongType size, int flags, Pointer reserved) {
  cudaMemcpyKind kind;

  switch (flags) {
    case 0: {
      kind = cudaMemcpyHostToHost;
    } break;
    case 1: {
      kind = cudaMemcpyHostToDevice;
    } break;
    case 2: {
      kind = cudaMemcpyDeviceToHost;
    } break;
    case 3: {
      kind = cudaMemcpyDeviceToDevice;
    } break;
    default: {
      LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      LaunchContext::defaultContext()->errorReference()->setErrorMessage("UNDEFNED MEMCPY");
      return 0;
    }
  }

  auto dZ = cudaMemcpy(reinterpret_cast<void *>(dst), const_cast<const void *>(reinterpret_cast<void *>(src)),
                       static_cast<size_t>(size), kind);
  if (dZ != 0) {
    printf("Failed on [%p] -> [%p], size: [%i], direction: [%i], dZ: [%i]\n", src, dst, size, flags,
           static_cast<int>(dZ));
    fflush(stdout);
    fflush(stderr);
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemcpy failed");
    return 0;
  }

  return 1;
}

int memcpyAsync(Pointer dst, Pointer src, LongType size, int flags, Pointer reserved) {
  auto pStream = reinterpret_cast<cudaStream_t *>(reserved);

  cudaMemcpyKind kind;


  switch (flags) {
    case 0: {
      kind = cudaMemcpyHostToHost;
    } break;
    case 1: {
      kind = cudaMemcpyHostToDevice;
    } break;
    case 2: {
      kind = cudaMemcpyDeviceToHost;
    } break;
    case 3: {
      kind = cudaMemcpyDeviceToDevice;
    } break;
    default: {
      LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      LaunchContext::defaultContext()->errorReference()->setErrorMessage("UNDEFINED MEMCPY");
      return 0;
    }
  }

  auto dZ = cudaMemcpyAsync(reinterpret_cast<void *>(dst), const_cast<const void *>(reinterpret_cast<void *>(src)),
                            static_cast<size_t>(size), kind, *pStream);

  if (dZ != 0) {
    printf("Failed on [%p] -> [%p], size: [%i], direction: [%i], dZ: [%i]\n", src, dst, size, flags,
           static_cast<int>(dZ));

    fflush(stdout);
    fflush(stderr);
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemcpyAsync failed");
    return 0;
  }


  return 1;
}

int memsetSync(Pointer dst, int value, LongType size, int flags, Pointer reserved) {
  auto dZ = cudaMemset(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size));
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemset failed");
  }

  return 1;
}

int memsetAsync(Pointer dst, int value, LongType size, int flags, Pointer reserved) {
  auto pStream = reinterpret_cast<cudaStream_t *>(reserved);

  auto dZ = cudaMemsetAsync(reinterpret_cast<void *>(dst), value, static_cast<size_t>(size), *pStream);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemsetAsync failed");
  }

  return 1;
}

int destroyEvent(Pointer event) {
  auto pEvent = reinterpret_cast<cudaEvent_t *>(&event);
  auto dZ = cudaEventDestroy(*pEvent);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventDestroy failed");
  }

  return 1;
}

int streamSynchronize(Pointer stream) {
  auto pStream = reinterpret_cast<cudaStream_t *>(stream);

  auto dZ = cudaStreamSynchronize(*pStream);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaStreamSynchronize failed");
  }

  return 1L;
}

int eventSynchronize(Pointer event) {
  auto pEvent = reinterpret_cast<cudaEvent_t *>(&event);

  auto dZ = cudaEventSynchronize(*pEvent);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaEventSynchronize failed");
  }

  return 1L;
}

int getAvailableDevices() {
  int devCnt = 0;
  cudaGetDeviceCount(&devCnt);
  return devCnt;
}

void enableDebugMode(bool reallyEnable) { Environment::getInstance().setDebug(reallyEnable); }

void setGridLimit(int gridSize) {
  if (gridSize > 8192) gridSize = 8192;
  if (gridSize < 1) gridSize = 1;
  blockLimit = gridSize;
}

int ompGetMaxThreads() { return maxThreads; }

int ompGetNumThreads() { return maxThreads; }

void setOmpNumThreads(int threads) {
  if (threads > 1024) threads = 1024;
  if (threads < 32) threads = 32;
  maxThreads = threads;
}

void enableVerboseMode(bool reallyEnable) { Environment::getInstance().setVerbose(reallyEnable); }

int getDeviceMajor(int device) { return deviceProperties[device].major; }

int getDeviceMinor(int device) { return deviceProperties[device].minor; }

const char *getDeviceName(int device) { return deviceProperties[device].name; }

void specialConcat(Pointer *extraPointers, int dimension, int numArrays, Pointer *data, Pointer *inputShapeInfo, void *dZ, LongType const *dZShapeInfo, Pointer *tadPointers, Pointer *offsetPointers) {
  try {
    BUILD_SINGLE_SELECTOR(ArrayOptions::dataType(dZShapeInfo), sd::SpecialMethods,
                          ::concatCpuGeneric(dimension, numArrays, data, inputShapeInfo, dZ, dZShapeInfo),
                          SD_COMMON_TYPES);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void saveNpy(std::string fname, const InteropDataBuffer *data, const unsigned int *shape, const unsigned int ndims,
             std::string mode) {
  auto dtype = data->getDataBuffer()->getDataType();
  BUILD_SINGLE_SELECTOR(dtype,cnpy::npy_save,(fname,data->getDataBuffer()->primary(),shape,ndims,mode),SD_COMMON_TYPES);
}


/**
 * This method saves
 */
TadPack *tadOnlyShapeInfo(const LongType *hXShapeInfo, LongType *dimension, LongType dimensionLength) {
  try {
    auto pack = ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, dimensionLength);
    return pack;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType const *getPrimaryShapeInfo(TadPack *pack) { return pack->primaryShapeInfo(); }
LongType const *getPrimaryOffsets(TadPack *pack) { return pack->primaryOffsets(); }
LongType const *getSpecialShapeInfo(TadPack *pack) { return pack->specialShapeInfo(); }
LongType const *getSpecialOffsets(TadPack *pack) { return pack->specialOffsets(); }
LongType getNumberOfTads(TadPack *pack) { return pack->numberOfTads(); }
int getShapeInfoLength(TadPack *pack) { return pack->shapeInfoLength(); }

int memcpyConstantAsync(LongType dst, Pointer src, LongType size, int flags, Pointer reserved) {
  cudaStream_t *pStream = reinterpret_cast<cudaStream_t *>(reserved);

  cudaMemcpyKind kind;

  DEBUG_KERNEL(pStream, -1);

  switch (flags) {
    case 0: {
      kind = cudaMemcpyHostToHost;
    } break;
    case 1: {
      kind = cudaMemcpyHostToDevice;
    } break;
    case 2: {
      kind = cudaMemcpyDeviceToHost;
    }
    case 3: {
      kind = cudaMemcpyDeviceToDevice;
    } break;
  }
  auto dZ = cudaMemcpyToSymbolAsync(deviceConstantMemory, const_cast<const void *>(src), size, dst, kind, *pStream);
  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaMemcpyToSymbolAsync failed");
  }

  return 1;
}

Pointer getConstantSpace() {
  Pointer dConstAddr;
  cudaError_t dZ = cudaGetSymbolAddress(reinterpret_cast<void **>(&dConstAddr), deviceConstantMemory);

  if (dZ != 0) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(dZ);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("cudaGetSymbolAddress failed");
  }

  return dConstAddr;
}

void pullRows(Pointer *extraPointers, OpaqueDataBuffer *dbX, LongType const *xShapeInfo, LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *zShapeInfo, LongType const *dZShapeInfo, LongType n,
              LongType *indexes, LongType const *tadShapeInfo, LongType const *tadOffsets,
              LongType const *zTadShapeInfo, LongType const *zTadOffsets) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    dim3 launchDims = getLaunchDims("pullRows");
    auto xType = ArrayOptions::dataType(xShapeInfo);
    BUILD_SINGLE_SELECTOR(xType, pullRowsKernelGeneric,
                          (launchDims,
                              stream,
                              shape::isEmptyConst(xShapeInfo) ? nullptr : dbX->special(),
                              shape::isEmptyConst(zShapeInfo) ? nullptr :  dbZ->special() ,
                              n, indexes, tadShapeInfo, tadOffsets,
                              zTadShapeInfo, zTadOffsets),
                          SD_COMMON_TYPES);

    DEBUG_KERNEL(stream, -1);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void average(Pointer *extras, Pointer *x, LongType const *xShapeInfo, Pointer *dx, LongType const *dXShapeInfo, void *z,
             LongType const *zShapeInfo, void *dz, LongType const *dzShapeInfo, int n, LongType length, bool propagate) {
  try {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extras[1]);
    int mode = getDeviceId(extras[3]);

    auto dX = reinterpret_cast<void **>(dx);

    if (Environment::getInstance().isDebugAndVerbose()) printf("averageFloat called\n");

    auto xType = ArrayOptions::dataType(xShapeInfo);
    // launching on gpu
    if (mode == 0) {
      dim3 launchDims = getLaunchDims("average");
      BUILD_SINGLE_SELECTOR(xType, averagingKernelGeneric, (launchDims, stream, dX, dz, n, length, propagate),
                            SD_COMMON_TYPES);
      DebugHelper::checkErrorCode(stream, "AverageFloat(...) failed");
    } else {
      // launching on host memory
      BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::averageGeneric(x, z, zShapeInfo, n, length, propagate),
                            SD_COMMON_TYPES);
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void accumulate(Pointer *extras, Pointer *x, LongType const *xShapeInfo, Pointer *dx, LongType const *dXShapeInfo, void *z, LongType const *zShapeInfo, void *dz, LongType const *dzShapeInfo, int n, LongType length) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extras[1]);
    int mode = getDeviceId(extras[3]);

    auto dX = reinterpret_cast<void **>(dx);

    if (Environment::getInstance().isDebugAndVerbose()) printf("accumulateFloat called\n");
    auto xType = ArrayOptions::dataType(xShapeInfo);

    // launching on gpu
    if (mode == 0) {
      dim3 launchDims = getAccumDims(n);
      BUILD_SINGLE_SELECTOR(xType, accumulateKernelGeneric, (launchDims, stream, dX, dz, n, length), SD_COMMON_TYPES);
      DebugHelper::checkErrorCode(stream, "AccumulateFloat(...) failed");
    } else {
      // launching on host memory
      BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::accumulateGeneric(x, z, zShapeInfo, n, length),
                            SD_COMMON_TYPES);
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void shuffle(Pointer *extras, Pointer *x, Pointer *xShapeInfo, Pointer *dx, Pointer *dXShapeInfo, Pointer *z,
             Pointer *zShapeInfo, Pointer *dz, Pointer *dZShapeInfo, int N, int *shuffleMap, Pointer *tadShapeInfo,
             Pointer *tadOffsets) {
  try {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extras[1]);

    auto dX = reinterpret_cast<void **>(dx);
    auto dZ = reinterpret_cast<void **>(dz);
    auto xShape = reinterpret_cast<LongType **>(xShapeInfo);
    auto dxShape = reinterpret_cast<LongType **>(dXShapeInfo);
    auto tadOnlyShapeInfo = reinterpret_cast<LongType **>(tadShapeInfo);
    auto tadOffset = reinterpret_cast<LongType **>(tadOffsets);

    auto xType = ArrayOptions::dataType(xShape[0]);
    dim3 launchDims = getLaunchDims("shuffle");
    BUILD_SINGLE_SELECTOR(xType, shuffleKernelGeneric,
                          (launchDims, stream, dX, dxShape, dZ, N, shuffleMap, tadOnlyShapeInfo, tadOffset),
                          SD_COMMON_TYPES);

    DebugHelper::checkErrorCode(stream, "shuffle(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

bool isExperimentalEnabled() { return Environment::getInstance().isExperimentalBuild(); }

void setOmpMinThreads(int threads) {
  minThreads = sd::math::sd_max<int>(32, threads);
  minThreads = sd::math::sd_min<int>(maxThreads, minThreads);
}

int getDevice() { return AffinityManager::currentDeviceId(); }

void setElementThreshold(int num) {
  // this is no-op for CUDA
}

void setTADThreshold(int num) {
  // this is no-op for CUDA
}

////////////////////////////////////////////////////////////////////////
void execSummaryStats(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                      LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                      LongType const *hZShapeInfo, LongType const *dZShapeInfo, bool biasCorrected) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execSummaryStats(&lc,
                                          opNum,
                                          shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                          hXShapeInfo,
                                          shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
                                          ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                          extraParams,
                                          shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                          hZShapeInfo,
                                          shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
                                          ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                          biasCorrected);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execSummaryStatsTad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                         LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbZ,
                         LongType const *hZShapeInfo, LongType const *dZShapeInfo,
                         OpaqueDataBuffer *dbDimension,
                         LongType const *hDimensionShape, LongType const *dDimensionShape, bool biasCorrected,
                         LongType const *tadShapeInfo, LongType const *tadOffsets) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbDimension});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    int dimensionLength = static_cast<int>(shape::length(hDimensionShape));

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execSummaryStats(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        extraParams,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        reinterpret_cast<LongType *>(dbDimension->special()), dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbDimension});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                 LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY, LongType const *hYShapeInfo,
                 LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                 LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduce3(&lc,
                                     opNum,
                                     shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                     hXShapeInfo,
                                     shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
                                     ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                     extraParams,
                                     shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->primary(),
                                     hYShapeInfo,
                                     shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->special(),
                                     ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(),
                                     shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                     hZShapeInfo,
                                     shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
                                     ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3Tad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                    LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY, LongType const *hYShapeInfo,
                    LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                    LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension, LongType const *hDimensionShape,
                    LongType const *dDimensionShape, LongType const *tadOnlyShapeInfo, LongType const *tadOffsets,
                    LongType const *yTadOnlyShapeInfo, LongType const *yTadOffsets) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    auto tadPack =
        ConstantTadHelper::getInstance().tadForDimensions(hXShapeInfo, dimension, shape::length(hDimensionShape));
    auto tadLength = shape::length(tadPack->primaryShapeInfo());
    auto yLength = shape::length(hYShapeInfo);
    auto xLength = shape::length(hXShapeInfo);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);

    if (tadLength == yLength || tadLength == xLength) {
      NativeOpExecutioner::execReduce3(
          &lc, opNum,
          shape::isEmptyConst(hXShapeInfo) ? nullptr: dbX->primary(),
          hXShapeInfo,
          shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
          ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), extraParams, dbY->primary(),
          hYShapeInfo,
          shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->special(),
          ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(),
          shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
          hZShapeInfo,
          shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
          ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
          dimension, dimensionLength,
          tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);
    } else
      NativeOpExecutioner::execReduce3TAD(
          &lc, opNum,
          shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
          hXShapeInfo,
          shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
          ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
          extraParams,
          shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->primary(),
          hYShapeInfo,
          shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->special(),
          ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(),
          shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
          hZShapeInfo,
          shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
          ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(), dimension, dimensionLength,
          tadOnlyShapeInfo, yTadOffsets, yTadOnlyShapeInfo, yTadOffsets);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduce3Scalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                       LongType const *dXShapeInfo, void *extraParams, OpaqueDataBuffer *dbY,
                       LongType const *hYShapeInfo, LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ,
                       LongType const *hZShapeInfo, LongType const *dZShapeInfo) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduce3Scalar(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special() ,
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(), extraParams, dbY->primary(),
        hYShapeInfo,
        shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(),
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special());

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarBool(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                    LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                    LongType const *dZShapeInfo, OpaqueDataBuffer *dbScalar, LongType const *hScalarShapeInfo,
                    LongType const *dScalarShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbScalar});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execScalarBool(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalar->primary(),
        hScalarShapeInfo,
        shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalar->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hScalarShapeInfo)->special(), extraParams);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbScalar});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarBoolTad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                       LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                       LongType const *dZShapeInfo, OpaqueDataBuffer *dbScalars, LongType const *hScalarShapeInfo,
                       LongType const *dScalarShapeInfo, void *extraParams,
                       OpaqueDataBuffer *dbDimension,
                       LongType const *hDimensionShape, LongType const *dDimensionShape, LongType const *tadShapeInfo,
                       LongType const *tadOffsets, LongType const *tadShapeInfoZ, LongType const *tadOffsetsZ) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbScalars});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execScalarBool(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        extraParams,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalars->primary(),
        hScalarShapeInfo,
        shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalars->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hScalarShapeInfo)->special(),
        dimension, dimensionLength,
        tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbScalars});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalar(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                LongType const *dZShapeInfo, OpaqueDataBuffer *dbScalar, LongType const *hScalarShapeInfo,
                LongType const *dScalarShapeInfo, void *extraParams) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbScalar});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execScalar(
        &lc, opNum,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
        shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalar->primary(),
        hScalarShapeInfo,
        shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalar->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hScalarShapeInfo)->special(), extraParams);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbScalar});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarTad(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                   LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ, LongType const *hZShapeInfo,
                   LongType const *dZShapeInfo, OpaqueDataBuffer *dbScalars, LongType const *hScalarShapeInfo,
                   LongType const *dScalarShapeInfo, void *extraParams, OpaqueDataBuffer *dbDimension,
                   LongType const *hDimensionShape, LongType const *dDimensionShape, LongType const *tadShapeInfo,
                   LongType const *tadOffsets, LongType const *tadShapeInfoZ, LongType const *tadOffsetsZ) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbScalars});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto xType = ArrayOptions::dataType(hXShapeInfo);
    auto yType = ArrayOptions::dataType(hScalarShapeInfo);
    auto zType = ArrayOptions::dataType(hZShapeInfo);

    if (yType != xType && yType != BOOL && !isExperimentalEnabled())
      throw datatype_exception::build("execScalar both operands must have same data type", xType, yType);

    dim3 launchDims = getLaunchDims("scalarTad");

#ifdef SD_EXPERIMENTAL_ENABLED
    BUILD_PAIRWISE_SELECTOR(
        xType, yType, zType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(launchDims, stream, opType, dX, dXShapeInfo, dZ, dZShapeInfo, dScalars, extraParams,
                                    dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
        SD_COMMON_TYPES, SD_COMMON_TYPES);
#else
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(
            launchDims, stream, opNum,
            shape::isEmptyConst(hXShapeInfo) ? nullptr :  dbX->special(),
            ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
            shape::isEmptyConst(hZShapeInfo) ? nullptr :  dbZ->special(),
            ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
            shape::isEmptyConst(hScalarShapeInfo) ? nullptr : dbScalars->special(),
            extraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
        SD_COMMON_TYPES);
#endif

    DEBUG_KERNEL(stream, opNum);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbScalars});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void execAggregate(Pointer *extraPointers, int opNum, void **arguments, int numArguments, LongType **shapes,
                   int numShapes, int *indexArguments, int numIndexArguments, int **intArrays, int numIntArrays,
                   void *realArguments, int numRealArguments, DataType dtype) {}

void batchExecutor(Pointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                   int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments, DataType dtype) {}

void execAggregateBatch(Pointer *extraPointers, int numAggregates, int opNum, int maxArgs, int maxShapes,
                        int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals, void *ptrToArguments,
                        DataType dtype) {}

////////////////////////////////////////////////////////////////////////
void execRandom(Pointer *extraPointers, int opNum, Pointer stateHost, OpaqueDataBuffer *dbZ,
                LongType const *hZShapeInfo, LongType const *dZShapeInfo, void *extraArguments) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execRandom(&lc, opNum, stateHost,
                                    shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(), hZShapeInfo,
                                    shape::isEmptyConst(hZShapeInfo) ? nullptr :dbZ->special(),
                                    ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                    extraArguments);

    InteropDataBuffer::registerSpecialUse({dbZ}, {});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execRandom2(Pointer *extraPointers, int opNum, Pointer stateHost, OpaqueDataBuffer *dbX,
                 LongType const *hXShapeInfo, LongType const *dXShapeInfo, OpaqueDataBuffer *dbZ,
                 LongType const *hZShapeInfo, LongType const *dZShapeInfo, void *extraArguments) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execRandom(
        &lc, opNum, stateHost,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
        hXShapeInfo,
        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
        hZShapeInfo,
        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(), extraArguments);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execRandom3(Pointer *extraPointers, int opNum, Pointer stateHost, OpaqueDataBuffer *dbX,
                 LongType const *hXShapeInfo, LongType const *dXShapeInfo, OpaqueDataBuffer *dbY,
                 LongType const *hYShapeInfo, LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ,
                 LongType const *hZShapeInfo, LongType const *dZShapeInfo, void *extraArguments) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execRandom(&lc, opNum, stateHost, shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                    hXShapeInfo, shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
                                    ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                    shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->primary(), hYShapeInfo,
                                    shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->special(),
                                    ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(),
                                    shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(), hZShapeInfo,
                                    shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
                                    ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                    extraArguments);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

Pointer initRandom(Pointer *extraPointers, long seed, long bufferSize, Pointer ptrToBuffer) {
  unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
  cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

  // we don't synchronize at random initialization, it's safe to go async here

  auto ptrDev = reinterpret_cast<unsigned long long *>(ptrToBuffer);
  auto buffer = new random::RandomBuffer(seed, bufferSize, reinterpret_cast<uint64_t *>(ptrHost),
                                             reinterpret_cast<uint64_t *>(ptrDev));
  buffer->propagateToDevice(buffer, *stream);

  DebugHelper::checkErrorCode(stream, "initRandom(...) failed A");

  // we generate sequence in the host memory
  random::Xoroshiro128 generator(buffer);
  generator.refreshBuffer();

  // and copy it to gpu
  cudaMemcpyAsync(ptrDev, ptrHost, bufferSize * 8, cudaMemcpyHostToDevice, *stream);
  DebugHelper::checkErrorCode(stream, "initRandom(...) failed B");

  return buffer;
}

void destroyRandom(Pointer ptrBuffer) {
  random::RandomBuffer *buffer = reinterpret_cast<random::RandomBuffer *>(ptrBuffer);

  // FIXME: it's bad thing, but we can't know in advance, which stream(s) where using this generator in practice
  cudaDeviceSynchronize();

  delete buffer;
}

void refreshBuffer(Pointer *extraPointers, long seed, Pointer ptrRandom) {
  random::RandomBuffer *buffer = reinterpret_cast<random::RandomBuffer *>(ptrRandom);

  unsigned long long *ptrHost = reinterpret_cast<unsigned long long *>(extraPointers[0]);
  cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
  cudaStreamSynchronize(*stream);

  uint64_t *ptrDev = buffer->getDeviceBuffer();

  // update rng state
  buffer->setSeed(seed);
  buffer->setOffset(0);
  buffer->propagateToDevice(buffer, *stream);

  // refresh buffer on host size
  random::Xoroshiro128 generator(buffer);
  generator.refreshBuffer();

  // copy back to gpu
  cudaMemcpyAsync(ptrDev, ptrHost, buffer->getSize() * 8, cudaMemcpyHostToDevice, *stream);
}

void reSeedBuffer(Pointer *extraPointers, long seed, Pointer ptrRandom) {
  random::RandomBuffer *buffer = reinterpret_cast<random::RandomBuffer *>(ptrRandom);

  cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
  cudaStreamSynchronize(*stream);

  // update rng state
  buffer->reSeed(seed);
  buffer->setOffset(0);
  buffer->propagateToDevice(buffer, *stream);
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

void tear(Pointer *extras, OpaqueDataBuffer *dbX, LongType const *xShapeInfo, LongType const *dXShapeInfo,
          Pointer *targets, LongType const *zShapeInfo, LongType const *tadShapeInfo, LongType const *tadOffsets) {
  try {
    InteropDataBuffer::prepareSpecialUse({}, {dbX});

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extras[1]);
    dim3 launchDims = getLaunchDims("tear");
    auto xType = ArrayOptions::dataType(xShapeInfo);
    BUILD_SINGLE_SELECTOR(
        xType, tearKernelGeneric,
        (launchDims, stream,
            shape::isEmptyConst(xShapeInfo) ? nullptr :  dbX->special(),
            dXShapeInfo,
            targets,
            zShapeInfo,
            tadShapeInfo,
            tadOffsets),
        SD_COMMON_TYPES);

    DebugHelper::checkErrorCode(stream, "tearFloat(...) failed");

    InteropDataBuffer::registerSpecialUse({}, {dbX});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void prescanArrayRecursive(Pointer *extras, int *dZ, int *dX, int numElements, int level) {
  auto stream = reinterpret_cast<cudaStream_t *>(extras[1]);
  auto g_scanBlockSums = reinterpret_cast<int **>(extras[2]);

  int blockSize = 512;  // max size of the thread blocks
  int numBlocks = sd::math::sd_max<int>(1, static_cast<int>(ceil(static_cast<float>(numElements) / (2.f * blockSize))));
  int numThreads;

  if (numBlocks > 1)
    numThreads = blockSize;
  else if (isPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = floorPow2(numElements);

  int numEltsPerBlock = numThreads * 2;

  // if this is a non-power-of-2 array, the last block will be non-full
  // compute the smallest power of 2 able to compute its scan.
  int numEltsLastBlock = numElements - (numBlocks - 1) * numEltsPerBlock;
  int numThreadsLastBlock = sd::math::sd_max<int>(1, numEltsLastBlock / 2);
  int np2LastBlock = 0;
  int sharedMemLastBlock = 0;

  if (numEltsLastBlock != numEltsPerBlock) {
    np2LastBlock = 1;

    if (!isPowerOfTwo(numEltsLastBlock)) numThreadsLastBlock = floorPow2(numEltsLastBlock);

    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
  }

  // padding space is used to avoid shared memory bank conflicts
  int extraSpace = numEltsPerBlock / NUM_BANKS;
  int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

  // setup execution parameters
  // if NP2, we process the last block separately
  dim3 grid(sd::math::sd_max<int>(1, numBlocks - np2LastBlock), 1, 1);
  dim3 threads(numThreads, 1, 1);
  dim3 gridOnes(1, 1, 1);
  dim3 threadsOnes(numThreadsLastBlock, 1, 1);

  if (sharedMemSize < 2048) sharedMemSize = 2048;

  if (sharedMemLastBlock < 2048) sharedMemLastBlock = 2048;

  // execute the scan
  if (numBlocks > 1) {
    sd::prescanLauncher<true, false>(grid, threads, sharedMemSize, stream, dZ, dX, g_scanBlockSums[level],
                                     numThreads * 2, 0, 0);
    if (np2LastBlock) {
      sd::prescanLauncher<true, true>(gridOnes, threadsOnes, sharedMemLastBlock, stream, dZ, dX, g_scanBlockSums[level],
                                      numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
    }

    // After scanning all the sub-blocks, we are mostly done.  But now we
    // need to take all of the last values of the sub-blocks and scan those.
    // This will give us a new value that must be sdded to each block to
    // get the final results.
    // recursive (CPU) call
    prescanArrayRecursive(extras, g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level + 1);

    uniformAdd<<<grid, threads, 1024, *stream>>>(dZ, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
    DebugHelper::checkGlobalErrorCode("uniform addfailed(...) failed");

    if (np2LastBlock) {
      uniformAdd<<<1, numThreadsLastBlock, 1024, *stream>>>(dZ, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1,
                                                            numElements - numEltsLastBlock);
      DebugHelper::checkGlobalErrorCode("concat general case failed(...) failed");

    }
  } else if (isPowerOfTwo(numElements)) {
    sd::prescanLauncher<false, false>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numThreads * 2, 0, 0);

  } else {
    sd::prescanLauncher<false, true>(grid, threads, sharedMemSize, stream, dZ, dX, 0, numElements, 0, 0);
  }

  DebugHelper::checkErrorCode(stream, "prescanArray(...) failed");
}

////////////////////////////////////////////////////////////////////////
void execReduce3All(Pointer *extraPointers, int opNum, OpaqueDataBuffer *dbX, LongType const *hXShapeInfo,
                    LongType const *dXShapeInfo, void *extraParamsVals, OpaqueDataBuffer *dbY,
                    LongType const *hYShapeInfo, LongType const *dYShapeInfo, OpaqueDataBuffer *dbZ,
                    LongType const *hZShapeInfo, LongType const *dZShapeInfo, OpaqueDataBuffer *dbDimension,
                    LongType const *hDimensionShape, LongType const *dDimensionShape, LongType const *xTadShapeInfo,
                    LongType const *xOffsets, LongType const *yTadShapeInfo, LongType const *yOffsets) {
  try {
    InteropDataBuffer::prepareSpecialUse({dbZ}, {dbX, dbY, dbDimension});
    InteropDataBuffer::preparePrimaryUse({}, {dbDimension});

    auto dimension = dbDimension != nullptr ? reinterpret_cast<LongType *>(dbDimension->primary()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(hDimensionShape));

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduce3All(&lc, opNum,
                                        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->primary(),
                                        hXShapeInfo,
                                        shape::isEmptyConst(hXShapeInfo) ? nullptr : dbX->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(hXShapeInfo)->special(),
                                        extraParamsVals,
                                        shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->primary(),
                                        hYShapeInfo,
                                        shape::isEmptyConst(hYShapeInfo) ? nullptr : dbY->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(hYShapeInfo)->special(),
                                        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->primary(),
                                        hZShapeInfo,
                                        shape::isEmptyConst(hZShapeInfo) ? nullptr : dbZ->special(),
                                        ConstantShapeHelper::getInstance().bufferForShapeInfo(hZShapeInfo)->special(),
                                        reinterpret_cast<LongType *>(dbDimension->special()),
                                        dimensionLength, xTadShapeInfo,
                                        xOffsets, yTadShapeInfo, yOffsets);

    InteropDataBuffer::registerSpecialUse({dbZ}, {dbX, dbY});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sort(Pointer *extraPointers, void *x, LongType const *xShapeInfo, void *dX, LongType const *dXShapeInfo, bool descending) {
  try {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto xLength = shape::length(xShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = ArrayOptions::dataType(xShapeInfo);

    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      dim3 launchDims = getSortFullDims(xLength);

      for (int k = 2; k <= xLength; k = 2 * k) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
          BUILD_SINGLE_SELECTOR(xType, bitonicSortStepGeneric,
                                (launchDims, stream, dX, dXShapeInfo, j, k, xLength, descending), SD_COMMON_TYPES);
        }
      }
    } else {
      dim3 launchDims = getSortFullDims(xLength);

      int max = 2, dg = 0;
      while (max < xLength) {
        max <<= 1;
        dg++;
      }
      max <<= 1;

      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          int half = n >> 1;
          BUILD_SINGLE_SELECTOR(xType, bitonicArbitraryStepGeneric,
                                (launchDims, stream, dX, dXShapeInfo, n, xLength, rev, descending), SD_COMMON_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    DebugHelper::checkErrorCode(stream, "sort(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByKey(Pointer *extraPointers, void *x, LongType const *xShapeInfo, void *dX, LongType const *dXShapeInfo, void *y, LongType const *yShapeInfo, void *dy, LongType const *dyShapeInfo, bool descending) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;

    if (xLength != yLength) THROW_EXCEPTION("sortByKey: keys and values must have the same size");

    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      dim3 launchDims = getSortFullDims(xLength);

      for (int k = 2; k <= xLength; k = 2 * k) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericKey,
                                (launchDims, stream, dX, dXShapeInfo, dy, dyShapeInfo, j, k, xLength, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
        }
      }
    } else {
      int numThreads = sd::math::sd_min<int>(512, xLength);
      int numBlocks = xLength / numThreads;
      if (xLength % numThreads > 0 || numBlocks == 0) numBlocks++;

      numBlocks = sd::math::sd_min<int>(512, numBlocks);
      dim3 launchDims(numBlocks, numThreads, 32768);

      int max = 2, dg = 0;
      while (max < xLength) {
        max <<= 1;
        dg++;
      }
      max <<= 1;

      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          int half = n >> 1;
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericKey,
                                (launchDims, stream, dX, dXShapeInfo, dy, dyShapeInfo, n, xLength, rev, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortByValue(Pointer *extraPointers, void *x, LongType const *xShapeInfo, void *dX, LongType const *dXShapeInfo, void *y, LongType const *yShapeInfo, void *dy, LongType const *dyShapeInfo, bool descending) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);
    auto xEWS = shape::elementWiseStride(xShapeInfo);
    auto xType = ArrayOptions::dataType(yShapeInfo);
    auto yType = ArrayOptions::dataType(xShapeInfo);

    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;

    if (xLength != yLength) THROW_EXCEPTION("sortByValue: keys and values must have the same size");

    // check if xLength is a power of 2, and use bitonic sort, if that's the case
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      dim3 launchDims = getSortFullDims(xLength);

      for (int k = 2; k <= xLength; k = 2 * k) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericKey,
                                (launchDims, stream, dy, dyShapeInfo, dX, dXShapeInfo, j, k, xLength, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
        }
      }
    } else {
      dim3 launchDims = getSortFullDims(xLength);

      int max = 2, dg = 0;
      while (max < xLength) {
        max <<= 1;
        dg++;
      }
      max <<= 1;

      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          int half = n >> 1;
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericKey,
                                (launchDims, stream, dy, dyShapeInfo, dX, dXShapeInfo, n, xLength, rev, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByKey(Pointer *extraPointers, void *x, LongType const *xShapeInfo, void *dX, LongType const *dXShapeInfo, void *y, LongType const *yShapeInfo, void *dy, LongType const *dyShapeInfo, LongType *dimension, LongType dimensionLength, bool descending) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto context =
        extraPointers[0] == 0 ? LaunchContext::defaultContext() : reinterpret_cast<LaunchContext *>(extraPointers[0]);
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
    dim3 launchDims = getSortTadDims(tadPack->numberOfTads());
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, dX, dXShapeInfo, dy, dyShapeInfo, dimension, dimensionLength,
                              tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    DebugHelper::checkErrorCode(stream, "sortTadKey(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTadByValue(Pointer *extraPointers, void *x, LongType const *xShapeInfo, void *dx, LongType const *dxShapeInfo, void *y, LongType const *yShapeInfo, void *dy, LongType const *dyShapeInfo, LongType *dimension, LongType dimensionLength, bool descending) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto context =
        extraPointers[0] == 0 ? LaunchContext::defaultContext() : reinterpret_cast<LaunchContext *>(extraPointers[0]);
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
    dim3 launchDims = getSortTadDims(tadPack->numberOfTads());
    auto xType = ArrayOptions::dataType(yShapeInfo);
    auto yType = ArrayOptions::dataType(xShapeInfo);

    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, dy, dyShapeInfo, dx, dxShapeInfo, dimension, dimensionLength,
                              tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    DebugHelper::checkErrorCode(stream, "sortTadValue(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortTad(Pointer *extraPointers, void *x, LongType const *xShapeInfo, void *dX, LongType const *dXShapeInfo,
             LongType *dimension, LongType dimensionLength, LongType const *tadShapeInfo, LongType const *tadOffsets, bool descending) {
  try {
    // to be implemented
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto context =
        extraPointers[0] == 0 ? LaunchContext::defaultContext() : reinterpret_cast<LaunchContext *>(extraPointers[0]);
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
    dim3 launchDims = getSortTadLarge(tadPack->numberOfTads());
    auto xType = ArrayOptions::dataType(xShapeInfo);
    BUILD_SINGLE_SELECTOR(
        xType, oesTadGeneric,
        (launchDims, stream, dX, dXShapeInfo, nullptr, dimensionLength, tadShapeInfo, tadOffsets, descending),
        SD_COMMON_TYPES);

    DebugHelper::checkErrorCode(stream, "sortTad(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sortCooIndices(Pointer *extraPointers, LongType *indices, void *values, LongType length,
                    const LongType *xShapeInfo) {
  THROW_EXCEPTION("sortCooIndices:: Not implemented yet");
}

void ravelMultiIndex(Pointer *extraPointers, LongType *indices, LongType *flatIndices, LongType length,
                     LongType *shapeInfo, int mode) {
  THROW_EXCEPTION("ravelMultiIndex:: Not implemented yet");
}

void unravelIndex(Pointer *extraPointers, LongType *indices, LongType *flatIndices, LongType length,
                  LongType *shapeInfo) {
  THROW_EXCEPTION("unravelIndex:: Not implemented yet");
}

LongType *mmapFile(Pointer *extraPointers, const char *fileName, LongType length) { return nullptr; }

void munmapFile(Pointer *extraPointers, LongType *ptrMap, LongType length) {}

ResultWrapper *executeFlatGraph(Pointer *extraPointers, Pointer flatBufferPointer) {
  try {
    return GraphExecutioner::executeFlatBuffer(flatBufferPointer);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType getResultWrapperSize(ResultWrapper *ptr) { return ptr->size(); }
Pointer getResultWrapperPointer(ResultWrapper *ptr) { return ptr->pointer(); }

const char *getAllCustomOps() { return ops::OpRegistrator::getInstance().getAllCustomOperations(); }

ShapeList *_calculateOutputShapes(Pointer *extraPointers, ops::DeclarableOp *op, Pointer *inputBuffers,
                                  Pointer *inputShapes, int numInputShapes, double *tArgs, int numTArgs,
                                  LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, int *dArgs, int numDArgs) {
  VariableSpace varSpace;
  Context block(2, &varSpace);
  ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numBArgs; e++) block.getBArguments()->push_back(bArgs[e]);

  for (int e = 0; e < numDArgs; e++) block.getDArguments()->push_back((DataType)dArgs[e]);

  for (int e = 0; e < numInputShapes; e++) {
    if (inputShapes[e] == nullptr) {
      std::string errorMessage;
      errorMessage += "Input shape at index ";
      errorMessage += std::to_string(e);
      errorMessage += " was null!";
      THROW_EXCEPTION(errorMessage.c_str());
    }
    auto shape_ = reinterpret_cast<LongType *>(inputShapes[e]);

    /*
     * Doesn't seem to be a null pointer but an out of bounds? Is it empty then?
     */
    // we shouldn't copy buffer if that's empty array
    void *buffer_ = ArrayOptions::arrayType(shape_) == EMPTY ? nullptr : inputBuffers[e];
    void *bufferD_ = ArrayOptions::arrayType(shape_) == EMPTY ? nullptr : inputBuffers[e + numInputShapes];

    auto array = new NDArray(buffer_, bufferD_, shape_);
    // block should contain references to proper variable
    varSpace.putVariable(1, e, array);
    block.pickInput(1, e);

    inShapes.push_back(shape_);
  }

  auto shapeList = op->calculateOutputShape(&inShapes, block);
  if (varSpace.launchContext()->getWorkspace() != nullptr) shapeList->detach();
  return shapeList;
}

ShapeList *calculateOutputShapes2(Pointer *extraPointers, LongType hash, Pointer *inputBuffers, Pointer *inputShapes,
                                  int numInputShapes, double *tArgs, int numTArgs, LongType *iArgs, int numIArgs,
                                  bool *bArgs, int numBArgs, int *dArgs, int numDArgs) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);
    return _calculateOutputShapes(extraPointers, op, inputBuffers, inputShapes, numInputShapes, tArgs, numTArgs, iArgs,
                                  numIArgs, bArgs, numBArgs, dArgs, numDArgs);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

ShapeList *_calculateOutputShapes(Pointer *extraPointers, ops::DeclarableOp *op, Pointer *inputShapes,
                                  int numInputShapes, double *tArgs, int numTArgs, LongType *iArgs, int numIArgs) {
  Context block(1);
  ShapeList inShapes;

  for (int e = 0; e < numIArgs; e++) block.getIArguments()->push_back(iArgs[e]);

  for (int e = 0; e < numTArgs; e++) block.getTArguments()->push_back(tArgs[e]);

  for (int e = 0; e < numInputShapes; e++) inShapes.push_back(reinterpret_cast<LongType *>(inputShapes[e]));

  auto shapeList = op->calculateOutputShape(&inShapes, block);

  return shapeList;
}

ShapeList *calculateOutputShapes(Pointer *extraPointers, LongType hash, Pointer *inputShapes, int numInputShapes,
                                 double *tArgs, int numTArgs, LongType *iArgs, int numIArgs) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);

    return _calculateOutputShapes(extraPointers, op, inputShapes, numInputShapes, tArgs, numTArgs, iArgs, numIArgs);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType getShapeListSize(ShapeList *list) { return list->size(); }

LongType const *getShape(ShapeList *list, LongType i) { return list->at(i); }

static SD_INLINE Status realExec(ops::DeclarableOp *op, Pointer *extraPointers, LongType hash, Pointer *inputBuffers,
                                 Pointer *inputShapes, int numInputs, Pointer *outputBuffers, Pointer *outputShapes, int numOutputs,
                                     double *tArgs, int numTArgs, LongType *iArgs, int numIArgs, bool *bArgs,
                                     int numBArgs, bool isInplace) {
  if (op == nullptr) sd_printf("Can't find requested operation: [%lld]\n", hash);

// we're using the same fake nodeId everywhere here

  std::vector<NDArray *> inputs(numInputs);
  std::vector<NDArray *> outputs(numOutputs);
  std::vector<double> ttArgs(numTArgs);
  std::vector<bool> bbArgs(numBArgs);
  std::vector<LongType> iiArgs(numIArgs);

// filling block now with inputs
  for (int e = 0; e < numInputs; e++) {
    auto shape = reinterpret_cast<LongType *>(inputShapes[e]);
    void *buffer = ArrayOptions::arrayType(shape) == EMPTY ? nullptr : inputBuffers[e];
    void *bufferD = ArrayOptions::arrayType(shape) == EMPTY ? nullptr : inputBuffers[e + numInputs];

    inputs[e] = new NDArray(buffer, bufferD, shape);
  }

// if not inplace - transferring output arrays

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
// we want to keep original output shape intact
      auto shape = shape::copyShape(reinterpret_cast<LongType *>(outputShapes[e]));
      void *buffer = ArrayOptions::arrayType(shape) == EMPTY ? nullptr : outputBuffers[e];
      void *bufferD = ArrayOptions::arrayType(shape) == EMPTY ? nullptr : outputBuffers[e + numOutputs];

// FIXME: revisit this.
      bool canNullify = true;
      for (int i = 0; i < numInputs; i++) {
        void *ibuffer = ArrayOptions::arrayType(shape) == EMPTY ? nullptr : inputBuffers[i];
        if (ibuffer == buffer) {
          canNullify = false;
          break;
        }
      }

      if (canNullify && buffer != nullptr)
        memset((uint8_t *)buffer, '\0',
               shape::length(shape) * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape)));

      auto array = new NDArray(buffer, bufferD, shape);
      outputs[e] = array;
    }

  for (int e = 0; e < numIArgs; e++) iiArgs[e] = iArgs[e];

  for (int e = 0; e < numTArgs; e++) ttArgs[e] = tArgs[e];

  for (int e = 0; e < numBArgs; e++) bbArgs[e] = bArgs[e];

// hypothetically at this point we have everything filled
  auto dZ = op->execute(inputs, outputs, ttArgs, iiArgs, bbArgs, std::vector<DataType>(), isInplace);

  if (!isInplace)
    for (int e = 0; e < numOutputs; e++) {
      if (outputs[e]->ordering() != shape::order(reinterpret_cast<LongType *>(outputShapes[e])))
        outputs[e]->streamline(shape::order(reinterpret_cast<LongType *>(outputShapes[e])));
    }

  for (auto v : inputs) delete v;

  for (auto v : outputs) delete v;

  return Status::OK;
}

Status execCustomOp(Pointer *extraPointers, LongType hash, Pointer *inputBuffers, Pointer *inputShapes,
                    int numInputs,
                    Pointer *outputBuffers, Pointer *outputShapes, int numOutputs, double *tArgs,
                    int numTArgs,
                    LongType *iArgs, int numIArgs, bool *bArgs, int numBArgs, bool isInplace) {
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

Status execCustomOp2(Pointer *extraPointers, LongType hash, Pointer opContext) {
  try {
    auto op = ops::OpRegistrator::getInstance().getOperation(hash);
    auto context = reinterpret_cast<Context *>(opContext);

    auto result = op->execute(context);

    auto res = cudaStreamSynchronize(*context->launchContext()->getCudaStream());
    if (res != 0) throw cuda_exception::build("customOp execution failed", res);

    for (auto v : context->fastpath_in()) {
      if (!v->isEmpty()) v->syncToDevice();
    }

    for (auto v : context->fastpath_out()) {
      if (!v->isEmpty()) v->syncToDevice();
    }


    return result;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::BAD_INPUT;
  }
}

Status registerGraph(Pointer *extraPointers, LongType graphId, Pointer flatBufferPointer) {
  try {
    auto graph = GraphExecutioner::importFromFlatPointer(flatBufferPointer);

    GraphHolder::getInstance().registerGraph(graphId, graph);

    return Status::OK;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::BAD_INPUT;
  }
}

static VariablesSet *executeStoredGraphT(Pointer *extraPointers, LongType graphId, Pointer *inputBuffers,
                                         Pointer *inputShapes, int *inputIndices, int numInputs) {
  auto graph = GraphHolder::getInstance().pullGraph(graphId);
  auto varSpace = graph->getVariableSpace()->clone();

  std::vector<NDArray *> handles;

  for (int e = 0; e < numInputs; e++) {
    auto idx = inputIndices[e];

    // we'll delete this array later, together with cloned VariableSpace
    auto array = new NDArray(inputBuffers[e], reinterpret_cast<LongType *>(inputShapes[e]));
    handles.emplace_back(array);

    if (varSpace->hasVariable(idx)) {
      auto var = varSpace->getVariable(idx);
      if (var->hasNDArray()) delete var->getNDArray();

      var->setNDArray(array);
    } else
      varSpace->putVariable(idx, array);
  }

  auto dZ = GraphExecutioner::execute(graph, varSpace);
  auto varSet = new VariablesSet(dZ);

  if (dZ == Status::OK) {
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

  delete varSpace;

  return varSet;
}

VariablesSet *executeStoredGraph(Pointer *extraPointers, LongType graphId, Pointer *inputBuffers, Pointer *inputShapes,
                                 int *inputIndices, int numInputs) {
  try {
    return executeStoredGraphT(extraPointers, graphId, inputBuffers, inputShapes, inputIndices, numInputs);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType getVariablesSetSize(VariablesSet *set) { return set->size(); }

Status getVariablesSetStatus(VariablesSet *set) { return set->status(); }

Variable *getVariable(VariablesSet *set, LongType i) { return set->at(i); }

int getVariableId(Variable *variable) { return variable->id(); }

int getVariableIndex(Variable *variable) { return variable->index(); }

const char *getVariableName(Variable *variable) { return variable->getName()->c_str(); }

LongType const *getVariableShape(Variable *variable) { return variable->getNDArray()->shapeInfo(); }

void *getVariableBuffer(Variable *variable) { return variable->getNDArray()->buffer(); }

Status unregisterGraph(Pointer *extraPointers, LongType graphId) {
  try {
    GraphHolder::getInstance().dropGraphAny(graphId);

    return Status::OK;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return Status::BAD_INPUT;
  }
}

void deletePointerArray(Pointer pointer) {
  Pointer *ptr = reinterpret_cast<Pointer *>(pointer);
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

void deleteVariablesSet(VariablesSet *pointer) {
  delete pointer;
}

void deleteShapeList(Pointer shapeList) {
  ShapeList *list = reinterpret_cast<ShapeList *>(shapeList);
  delete list;
}

const char *getAllOperations() { return OpTracker::getInstance().exportOperations(); }

Pointer getGraphState(LongType id) { return (Pointer) new GraphState(id); }

void deleteGraphState(Pointer state) {
  auto stateP = reinterpret_cast<GraphState *>(state);
  delete stateP;
}

Status execCustomOpWithScope(Pointer *extraPointers, GraphState *state, LongType opHash, LongType *scopes,
                             int numScopes, Pointer *inputBuffers, Pointer *inputShapes, int numInputs,
                             Pointer *outputBuffers, Pointer *outputShapes, int numOutputs) {
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

    auto array = new NDArray(buffer, shapeInfo, varSpace->launchContext());

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

  auto dZ = LogicExecutor::processNode(graph, &node);
  if (dZ != Status::OK) return dZ;

  // mapping outputs

  for (int e = 0; e < numOutputs; e++) {
    auto buffer = outputBuffers[e];
    auto shapeInfo = reinterpret_cast<LongType *>(outputShapes[e]);

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
  return Status::OK;
}

Status execCustomOpWithScope(Pointer *extraPointers, Pointer state, LongType opHash, LongType *scopes, int numScopes,
                             Pointer *inputBuffers, Pointer *inputShapes, int numInputs, Pointer *outputBuffers,
                             Pointer *outputShapes, int numOutputs) {
  try {
    return execCustomOpWithScope(extraPointers, reinterpret_cast<GraphState *>(state), opHash, scopes,
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
  auto p = reinterpret_cast<ResultWrapper *>(ptr);
  delete p;
}

int estimateThreshold(Pointer *extraPointers, Pointer dX, LongType const *dXShapeInfo, int N,
                      float threshold) {
  THROW_EXCEPTION("estimateThreshold: Not implemented yet");
}

/*
 * TypeDef:
 *     void convertTypes(sd::Pointer *extras, int srcType, sd::Pointer dX, long N, int dstType, sd::Pointer dZ);
 */
void convertTypes(Pointer *extras, int srcType, Pointer dX, LongType N, int dstType, Pointer dZ) {
  try {
    auto dx = reinterpret_cast<void *>(dX);
    auto dz = reinterpret_cast<void *>(dZ);

    if (srcType == ND4J_FLOAT8) {
      if (dstType == ND4J_FLOAT8) {
        // convertKernel<double, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        // sd::TypeCast::convertGenericCuda<sd::float8, sd::int8>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        // sd::TypeCast::convertGenericCuda<sd::float8, sd::uint8>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        // sd::TypeCast::convertGenericCuda<sd::float8, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        // sd::TypeCast::convertGenericCuda<sd::float8, sd::int16>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        // sd::TypeCast::convertGenericCuda<sd::float8, sd::uint16>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
      } else if (dstType == ND4J_FLOAT32) {
        // sd::TypeCast::convertGenericCuda<sd::float8, float>(extras, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        // sd::TypeCast::convertGenericCuda<sd::float8, double>(extras, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_INT8) {
      if (dstType == ND4J_FLOAT8) {
        // sd::TypeCast::convertGenericCuda<sd::int8, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        // convertKernel<sd::int8, sd::int8>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        TypeCast::convertGenericCuda<int8_t, uint8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        TypeCast::convertGenericCuda<int8_t, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        TypeCast::convertGenericCuda<int8_t, int16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        TypeCast::convertGenericCuda<int8_t, uint16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
        // TODO: eventually we might want to add it
      } else if (dstType == ND4J_FLOAT32) {
        TypeCast::convertGenericCuda<int8_t, float>(extras, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        TypeCast::convertGenericCuda<int8_t, double>(extras, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_UINT8) {
      if (dstType == ND4J_FLOAT8) {
        // sd::TypeCast::convertGenericCuda<uint8_t, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        TypeCast::convertGenericCuda<uint8_t, int8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        TypeCast::convertGenericCuda<uint8_t, uint8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        TypeCast::convertGenericCuda<uint8_t, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        TypeCast::convertGenericCuda<uint8_t, int16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        TypeCast::convertGenericCuda<uint8_t, uint16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
        // TODO: still might want to add
      } else if (dstType == ND4J_FLOAT32) {
        TypeCast::convertGenericCuda<uint8_t, float>(extras, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        TypeCast::convertGenericCuda<uint8_t, double>(extras, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_FLOAT16) {
      if (dstType == ND4J_FLOAT8) {
        // sd::TypeCast::convertGenericCuda<float16, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        TypeCast::convertGenericCuda<float16, int8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        TypeCast::convertGenericCuda<float16, uint8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        TypeCast::convertGenericCuda<float16, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        TypeCast::convertGenericCuda<float16, int16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        TypeCast::convertGenericCuda<float16, uint16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
        // TODO: .... ^^^
      } else if (dstType == ND4J_FLOAT32) {
        TypeCast::convertGenericCuda<float16, float>(extras, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        TypeCast::convertGenericCuda<float16, double>(extras, dx, N, dz);
      } else if (dstType == ND4J_THRESHOLD) {
        // sd::convertToThreshold<float16>(nullptr, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_INT16) {
      if (dstType == ND4J_FLOAT8) {
        // sd::TypeCast::convertGenericCuda<int16_t, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        TypeCast::convertGenericCuda<int16_t, int8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        TypeCast::convertGenericCuda<int16_t, uint8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        TypeCast::convertGenericCuda<int16_t, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        TypeCast::convertGenericCuda<int16_t, int16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        TypeCast::convertGenericCuda<int16_t, uint16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
        // TODO...
      } else if (dstType == ND4J_FLOAT32) {
        TypeCast::convertGenericCuda<int16_t, float>(extras, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        TypeCast::convertGenericCuda<int16_t, double>(extras, dx, N, dz);
      } else {
        printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_FLOAT24) {
    } else if (srcType == ND4J_FLOAT32) {
      if (dstType == ND4J_FLOAT8) {
        // sd::TypeCast::convertGenericCuda<float, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        TypeCast::convertGenericCuda<float, int8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        TypeCast::convertGenericCuda<float, uint8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        TypeCast::convertGenericCuda<float, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        TypeCast::convertGenericCuda<float, int16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        TypeCast::convertGenericCuda<float, uint16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
      } else if (dstType == ND4J_DOUBLE) {
        TypeCast::convertGenericCuda<float, double>(extras, dx, N, dz);
      } else if (dstType == ND4J_THRESHOLD) {
        // sd::convertToThreshold<float>(nullptr, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_DOUBLE) {
      if (dstType == ND4J_FLOAT8) {
        // sd::TypeCast::convertGenericCuda<double, sd::float8>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT8) {
        TypeCast::convertGenericCuda<double, int8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT8) {
        TypeCast::convertGenericCuda<double, uint8_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT16) {
        TypeCast::convertGenericCuda<double, float16>(extras, dx, N, dz);
      } else if (dstType == ND4J_INT16) {
        TypeCast::convertGenericCuda<double, int16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_UINT16) {
        TypeCast::convertGenericCuda<double, uint16_t>(extras, dx, N, dz);
      } else if (dstType == ND4J_FLOAT24) {
      } else if (dstType == ND4J_FLOAT32) {
        TypeCast::convertGenericCuda<double, float>(extras, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        //
      } else if (dstType == ND4J_THRESHOLD) {
        // sd::convertToThreshold<double>(nullptr, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else if (srcType == ND4J_THRESHOLD) {
      if (dstType == ND4J_FLOAT16) {
        // sd::convertFromThreshold<float16>(nullptr, dx, N, dz);
      } else if (dstType == ND4J_FLOAT32) {
        // sd::convertFromThreshold<float>(nullptr, dx, N, dz);
      } else if (dstType == ND4J_DOUBLE) {
        // sd::convertFromThreshold<double>(nullptr, dx, N, dz);
      } else {
        sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
      }
    } else {
      sd_printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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

void deleteUtf8String(Pointer *extraPointers, Pointer ptr) { delete (reinterpret_cast<utf8string *>(ptr)); }

///////////////////////////////////////////////////////////////////
template <typename T, typename I>
SD_KERNEL static void scatterUpdateCuda(const int opCode, const int numOfSubArrs, void *vx,
                                        const LongType *xShapeInfo, const LongType *xOffsets, void *vy,
                                        const LongType *yShapeInfo, const LongType *yOffsets,
                                        const void *vindexes) {
  __shared__ T *x, *y;
  __shared__ LongType arrLenX, arrLenY;
  auto indexes = reinterpret_cast<const I *>(vindexes);

  for (int e = 0; e < numOfSubArrs; e++) {
    const auto xIndex = indexes[e];
    const bool isOwner = xIndex < gridDim.x ? blockIdx.x == xIndex : blockIdx.x == xIndex % gridDim.x;

    if (!isOwner) continue;

    if (threadIdx.x == 0) {
      x = reinterpret_cast<T *>(vx) + xOffsets[xIndex];
      y = reinterpret_cast<T *>(vy) + yOffsets[e];
      arrLenX = shape::length(xShapeInfo);
      arrLenY = shape::length(yShapeInfo);
    }
    __syncthreads();

    if (arrLenX != arrLenY) return;

    for (LongType i = threadIdx.x; i < arrLenX; i += blockDim.x) {
      const auto xOffset = shape::getIndexOffset(i, xShapeInfo);
      const auto yOffset = shape::getIndexOffset(i, yShapeInfo);

      switch (opCode) {
        case 0:
          x[xOffset] += y[yOffset];
          break;
        case 1:
          x[xOffset] -= y[yOffset];
          break;
        case 2:
          x[xOffset] *= y[yOffset];
          break;
        case 3:
          x[xOffset] /= y[yOffset];
          break;
        case 4:
          x[xOffset] = y[yOffset] - x[xOffset];
          break;
        case 5:
          x[xOffset] = y[yOffset] / x[xOffset];
          break;
        case 6:
          x[xOffset] = y[yOffset];
          break;
        default:
          continue;
      }
    }
    __syncthreads();
  }
}

template <typename T, typename I>
SD_HOST static void scatterUpdateCudaLauncher(const cudaStream_t *stream, const int opCode, const int numOfSubArrs,
                                              void *vx, const LongType const *xShapeInfo,
                                              const LongType *xOffsets, void *vy, const LongType *yShapeInfo,
                                              const LongType *yOffsets, const void *indexes) {
  scatterUpdateCuda<T, I><<<512, 256, SD_MAX_NUM_THREADS, *stream>>>(opCode, numOfSubArrs, vx, xShapeInfo, xOffsets, vy,
                                                                     yShapeInfo, yOffsets, indexes);
}

//////////////////////////////////////////////////////////////////////////
void scatterUpdate(Pointer *extraPointers, int opCode, int numOfSubArrs, void *hX, LongType const *hXShapeInfo,
                   LongType const *hXOffsets, void *dX, LongType const *dXShapeInfo, LongType const *dXOffsets, void *hY, LongType const *hYShapeInfo, LongType const *hYOffsets, void *dY,
                   LongType const *dYShapeInfo, LongType const *dYOffsets, void *hIindexes,
                   LongType const *hIndicesShapeInfo,
                   void *dIindexes, LongType const *dIndicesShapeInfo) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto type = ArrayOptions::dataType(hXShapeInfo);
    auto iType = ArrayOptions::dataType(hIndicesShapeInfo);

    BUILD_DOUBLE_SELECTOR(
        type, iType, scatterUpdateCudaLauncher,
        (stream, opCode, numOfSubArrs, dX, dXShapeInfo, dXOffsets, dY, dYShapeInfo, dYOffsets, dIindexes),
        SD_COMMON_TYPES, SD_INDEXING_TYPES);

    DebugHelper::checkErrorCode(stream, "scatterUpdate(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void inspectArray(Pointer *extraPointers, Pointer buffer, LongType *shapeInfo, Pointer specialBuffer,
                  LongType *specialShapeInfo, Pointer debugInfo) {
  try {
    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    auto p = reinterpret_cast<DebugInfo *>(debugInfo);
    NDArray array(buffer, specialBuffer, shapeInfo, &lc);
    DebugHelper::retrieveDebugStatistics(p, &array);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void SD_KERNEL tryPointerKernel(void *p, int len) {
  auto buf = reinterpret_cast<int8_t *>(p);
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int b;
  if (tid < len) atomicAdd(&b, buf[tid]);

  __syncthreads();

}

void tryPointer(Pointer extra, Pointer p, int len) {
  try {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    tryPointerKernel<<<256, 512, len + 64, stream>>>(p, len);
    DebugHelper::checkGlobalErrorCode("try pointer failed(...) failed");

    auto e = cudaStreamSynchronize(stream);

    if (e != 0) throw cuda_exception::build("tryPointer failed", e);

    cudaStreamDestroy(stream);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

int dataTypeFromNpyHeader(void *header) { return (int)cnpy::dataTypeFromHeader(reinterpret_cast<char *>(header)); }

OpaqueConstantShapeBuffer *shapeBuffer(int rank, LongType *shape, LongType *strides, DataType dtype,
                                       char order,
                                       LongType ews, bool empty) {
  return shapeBufferEx(rank, shape, strides, dtype, order, ews, empty ? ARRAY_EMPTY : 0);
}

OpaqueConstantShapeBuffer *shapeBufferEx(int rank, LongType *shape, LongType *strides, DataType dtype,
                                         char order,
                                         LongType ews, LongType extras) {
  try {

    auto desc = new ShapeDescriptor(dtype, order, shape, strides, rank, extras);
    auto buffer = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    return buffer;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void deleteConstantShapeBuffer(OpaqueConstantShapeBuffer *ptr) { }

void deleteConstantDataBuffer(OpaqueConstantDataBuffer *ptr) {
  delete ptr;
}

void deleteTadPack(TadPack *ptr) {
  delete ptr;
}

bool isBlasVersionMatches(int major, int minor, int build) {
  auto result = major == Environment::getInstance()._blasMajorVersion &&
                minor == Environment::getInstance()._blasMinorVersion &&
                build == Environment::getInstance()._blasPatchVersion;

  if (!result) {
    sd_printf("CUDA/cuBLAS version mismatch. Expected: %i.%i.%i but got %i.%i.%i instead\n",
              Environment::getInstance()._blasMajorVersion, Environment::getInstance()._blasMinorVersion,
              Environment::getInstance()._blasPatchVersion, major, minor, build);
    LaunchContext::defaultContext()->errorReference()->setErrorCode(152);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage("CUDA/cuBLAS version mismatch");
  }

  return result;
}

ConstantDataBuffer *constantBufferLong(DataType dtype, LongType const *data, int length) {
  return ConstantHelper::getInstance().constantBuffer(ConstantDescriptor(data, length), dtype);
}

ConstantDataBuffer *constantBufferDouble(DataType dtype, double *data, int length) {
  return ConstantHelper::getInstance().constantBuffer(ConstantDescriptor(data, length), dtype);
}

ConstantDataBuffer *constantBuffer(DataType dtype, ConstantDescriptor *descriptor) {
  return ConstantHelper::getInstance().constantBuffer(*descriptor, dtype);
}

Pointer getConstantDataBufferPrimary(ConstantDataBuffer *dbf) { return dbf->primary(); }
Pointer getConstantDataBufferSpecial(ConstantDataBuffer *dbf) { return dbf->special(); }
LongType getConstantDataBufferLength(ConstantDataBuffer *dbf) { return dbf->length(); }
LongType getConstantDataBufferSizeOf(ConstantDataBuffer *dbf) { return dbf->sizeOf(); }

Pointer getConstantShapeBufferPrimary(OpaqueConstantShapeBuffer *dbf) { return const_cast<LongType *>(dbf->primary()); }

Pointer getConstantShapeBufferSpecial(OpaqueConstantShapeBuffer *dbf) { return const_cast<LongType *>(dbf->special()); }

Context *createGraphContext(int nodeId) { return new Context(nodeId); }

RandomGenerator *getGraphContextRandomGenerator(Context *ptr) { return &ptr->randomGenerator(); }

void markGraphContextInplace(Context *ptr, bool reallyInplace) { ptr->markInplace(reallyInplace); }

void setGraphContextCudaContext(Context *ptr, void *stream, void *reductionPointer,
                                void *allocationPointer) {
  ptr->setCudaContext(stream, reductionPointer, allocationPointer);
}

void setGraphContextInputArray(Context *ptr, int index, void *buffer, void *shapeInfo, void *specialBuffer,
                               void *specialShapeInfo) {
  ptr->setInputArray(index, buffer, shapeInfo, specialBuffer, specialShapeInfo);
}

void setGraphContextOutputArray(Context *ptr, int index, void *buffer, void *shapeInfo, void *specialBuffer,
                                void *specialShapeInfo) {
  ptr->setOutputArray(index, buffer, shapeInfo, specialBuffer, specialShapeInfo);
}

void setGraphContextInputBuffer(OpaqueContext *ptr, int index, OpaqueDataBuffer *buffer, InteropDataBuffer *shapeInfo,
                                InteropDataBuffer *specialShapeInfo) {
  ptr->setInputArray(index, buffer, shapeInfo, specialShapeInfo);
}

void setGraphContextOutputBuffer(OpaqueContext *ptr, int index, OpaqueDataBuffer *buffer, InteropDataBuffer *shapeInfo,
                                 InteropDataBuffer *specialShapeInfo) {
  ptr->setOutputArray(index, buffer, shapeInfo, specialShapeInfo);
}

void setGraphContextTArguments(Context *ptr, double *arguments, int numberOfArguments) {
  ptr->setTArguments(arguments, numberOfArguments);
}

void setGraphContextIArguments(Context *ptr, LongType *arguments, int numberOfArguments) {
  ptr->setIArguments(arguments, numberOfArguments);
}

void setGraphContextBArguments(Context *ptr, bool *arguments, int numberOfArguments) {
  ptr->setBArguments(arguments, numberOfArguments);
}

void setGraphContextDArguments(OpaqueContext *ptr, int *arguments, int numberOfArguments) {
  std::vector<DataType> dtypes(numberOfArguments);
  for (int e = 0; e < numberOfArguments; e++) dtypes[e] = (DataType)arguments[e];

  ptr->setDArguments(dtypes);
}

void deleteGraphContext(Context *ptr) {}

RandomGenerator *createRandomGenerator(LongType rootSeed, LongType nodeSeed) {
  try {
    return new RandomGenerator(rootSeed, nodeSeed);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType getRandomGeneratorRootState(RandomGenerator *ptr) { return ptr->rootState(); }

LongType getRandomGeneratorNodeState(RandomGenerator *ptr) { return ptr->nodeState(); }

void setRandomGeneratorStates(RandomGenerator *ptr, LongType rootSeed, LongType nodeSeed) {
  ptr->setStates(rootSeed, nodeSeed);
}

float getRandomGeneratorRelativeFloat(RandomGenerator *ptr, LongType index) {
  return ptr->relativeT<float>(index);
}

double getRandomGeneratorRelativeDouble(RandomGenerator *ptr, LongType index) {
  return ptr->relativeT<double>(index);
}

int getRandomGeneratorRelativeInt(RandomGenerator *ptr, LongType index) { return ptr->relativeInt(index); }

LongType getRandomGeneratorRelativeLong(RandomGenerator *ptr, LongType index) {
  return ptr->relativeLong(index);
}

int getRandomGeneratorNextInt(RandomGenerator *ptr) {
  // to nullify  _nodeState._long ^= (steps ^ 0xdeadbeef);
  // we will use step = 0xdeadbeef
  auto result = ptr->relativeInt(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

LongType getRandomGeneratorNextLong(RandomGenerator *ptr) {
  auto result = ptr->relativeLong(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

float getRandomGeneratorNextFloat(RandomGenerator *ptr) {
  auto result = ptr->relativeT<float>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

double getRandomGeneratorNextDouble(RandomGenerator *ptr) {
  auto result = ptr->relativeT<double>(1);
  ptr->rewindH(0xdeadbeef);
  return result;
}

void deleteRandomGenerator(RandomGenerator *ptr) { delete ptr; }

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
    return (Pointer)(ConstantShapeHelper::getInstance().createFromExisting(
        shapeBuffer, true));  // TO DO: this can lead to unpleasant crash sometimes
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

LongType getCachedMemory(int deviceId) { return ConstantHelper::getInstance().getCachedAmount(deviceId); }

LaunchContext *defaultLaunchContext() { return LaunchContext::defaultContext(); }

Pointer lcScalarPointer(OpaqueLaunchContext *lc) { return lc->getScalarPointer(); }

Pointer lcReductionPointer(OpaqueLaunchContext *lc) { return lc->getReductionPointer(); }

Pointer lcAllocationPointer(OpaqueLaunchContext *lc) { return lc->getAllocationPointer(); }

Pointer lcExecutionStream(OpaqueLaunchContext *lc) { return lc->getCudaStream(); }

Pointer lcCopyStream(OpaqueLaunchContext *lc) { return lc->getCudaSpecialStream(); }

Pointer lcBlasHandle(OpaqueLaunchContext *lc) { return lc->getCublasHandle(); }

Pointer lcSolverHandle(OpaqueLaunchContext *lc) { return lc->getCusolverHandle(); }

int lastErrorCode() { return LaunchContext::defaultContext()->errorReference()->errorCode(); }

const char *lastErrorMessage() { return LaunchContext::defaultContext()->errorReference()->errorMessage(); }

void ctxShapeFunctionOverride(OpaqueContext *ptr, bool reallyOverride) {
  ptr->setShapeFunctionOverride(reallyOverride);
}

void ctxPurge(OpaqueContext *ptr) { ptr->clearFastPath(); }

int binaryLevel() { return 0; }

int optimalLevel() { return 0; }

bool isMinimalRequirementsMet() { return true; }

bool isOptimalRequirementsMet() { return true; }

void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) { ptr->allowHelpers(reallyAllow); }

void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) {
  if (execMode < 0 || execMode > 2) execMode = 0;

  ptr->setExecutionMode((samediff::ExecutionMode)execMode);
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(LongType elements, int dataType, Pointer primary, Pointer special) {
  auto buffer = dbAllocateDataBuffer(0, dataType, false);
  buffer->markOwner(false);

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) buffer->setSpecial(special, elements);

  return buffer;
}

OpaqueDataBuffer *dbAllocateDataBuffer(LongType elements, int dataType, bool allocateBoth) {
  return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer *allocateDataBuffer(LongType elements, int dataType, bool allocateBoth) {
  try {
    auto dtype = DataTypeUtils::fromInt(dataType);
    LongType totalElementSize = elements == 0 ? DataTypeUtils::sizeOf(dtype) : elements * DataTypeUtils::sizeOf(dtype);
    return new InteropDataBuffer(totalElementSize, dtype, allocateBoth);
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is null");
  return dataBuffer->primary();
}

Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSpecialBuffer: dataBuffer is null");
  return dataBuffer->special();
}

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is null");
  delete dataBuffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, Pointer primaryBuffer, LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetPrimaryBuffer: dataBuffer is null");
  dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, Pointer specialBuffer, LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetSpecialBuffer: dataBuffer is null");
  dataBuffer->setSpecial(specialBuffer, numBytes);
}

void dbAllocatePrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbAllocatePrimaryBuffer: dataBuffer is null");
  dataBuffer->dataBuffer()->allocatePrimary();
}

void dbAllocateSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbAllocateSpecialBuffer: dataBuffer is null");
  dataBuffer->dataBuffer()->allocateSpecial();
}

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, LongType elements) {
  try {
    if(dataBuffer == nullptr)
      THROW_EXCEPTION("dbExpandBuffer: dataBuffer is null");
    dataBuffer->dataBuffer()->expand(elements * DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueDataBuffer *dbCreateView(OpaqueDataBuffer *dataBuffer, LongType length, LongType offset) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbCreateView: dataBuffer is null");
  return new InteropDataBuffer(*dataBuffer, length, offset);
}

int dbUseCount(OpaqueDataBuffer* dataBuffer){
  if(dataBuffer) return dataBuffer->useCount();
  return 0;
}

void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToSpecial: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr &&  dataBuffer->dataBuffer().get() != nullptr && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToSpecial();
}

void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToPrimary: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr &&  dataBuffer->dataBuffer().get() != nullptr && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToPrimary(LaunchContext::defaultContext(),false);

}

void dbTickHostRead(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickHostRead: dataBuffer is null");
  dataBuffer->dataBuffer()->readPrimary();
}

void dbTickHostWrite(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickHostWrite: dataBuffer is null");
  dataBuffer->dataBuffer()->writePrimary();
}

void dbTickDeviceRead(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickDeviceRead: dataBuffer is null");
  dataBuffer->dataBuffer()->readSpecial();
}

void dbTickDeviceWrite(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbTickDeviceWrite: dataBuffer is null");
  dataBuffer->dataBuffer()->writeSpecial();

}

void dbExpand(OpaqueDataBuffer *dataBuffer, LongType elements) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbExpand: dataBuffer is null");
  dataBuffer->expand(elements);
}

void dbClose(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbClose: dataBuffer is null");

  auto ret = dataBuffer->getDataBuffer();
  if(ret != nullptr)
    dataBuffer->getDataBuffer()->close();
}

int dbDeviceId(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbDeviceId: dataBuffer is null");
  return dataBuffer->deviceId();
}

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetDeviceId: dataBuffer is null");
  dataBuffer->setDeviceId(deviceId);
}

int dbLocality(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbLocality: dataBuffer is null");
  auto p = dataBuffer->dataBuffer()->isPrimaryActual();
  auto d = dataBuffer->dataBuffer()->isSpecialActual();

  if (p && d)
    return 0;
  else if (p)
    return -1;
  else
    return 1;
}

void setVedaDeviceLibFolder(std::string path){

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


  auto len = shape::shapeInfoLength(rank);
  auto descriptor = ShapeDescriptor(dt,order,shape.data(),strides.data(),rank,isEmpty ? ARRAY_EMPTY : 0);

  auto buffer = descriptor.toShapeInfo();
  for(LongType i = 0; i < len; i++) {
    bufferToSet[i] = buffer[i];
  }




  delete[] buffer;
}

void setGraphContextInputArrays(OpaqueContext* ptr, int numArrays, Pointer * buffer, Pointer * shapeInfo,
                                Pointer * specialBuffer, Pointer * specialShapeInfo) {

  auto inputBuffers = (void **) buffer;
  auto inputShapeBuffers = (void **) shapeInfo;
  for(int i = 0; i < numArrays; i++) {
    ptr->setInputArray(i,inputBuffers != nullptr && inputBuffers[i] != nullptr ? inputBuffers[i] : nullptr,inputShapeBuffers[i],specialBuffer != nullptr ? specialBuffer[i] : nullptr,specialShapeInfo != nullptr ? specialShapeInfo[i] : nullptr);
  }

}
void setGraphContextOutputArrays(OpaqueContext* ptr, int numArrays, void** buffer, Pointer * shapeInfo,
                                 Pointer * specialBuffer, Pointer * specialShapeInfo) {
  auto inputBuffers = (void **) buffer;
  auto inputShapeBuffers = (void **) shapeInfo;
  for(int i = 0; i < numArrays; i++) {
    ptr->setOutputArray(i,inputBuffers != nullptr && inputBuffers[i] != nullptr  ? inputBuffers[i] : nullptr,inputShapeBuffers[i],specialBuffer != nullptr ? specialBuffer[i] : nullptr,specialShapeInfo != nullptr ? specialShapeInfo[i] : nullptr);
  }

}
void  setGraphContextInputBuffers(OpaqueContext* ptr, int numArrays,void** buffer,
                                  void **shapeInfo, void **specialShapeInfo) {
  if(shapeInfo == nullptr)
    THROW_EXCEPTION("Input shape info was null!");


  OpaqueDataBuffer  **buffers = (OpaqueDataBuffer **) buffer;
  OpaqueDataBuffer **shapeBuffers = (OpaqueDataBuffer **) shapeInfo;
  OpaqueDataBuffer **specialShapeBuffers = (OpaqueDataBuffer **) specialShapeInfo;

  for(int i = 0; i < numArrays; i++) {
    if(shapeInfo[i] == nullptr)
      THROW_EXCEPTION("Input shape at index was null!");

    LongType *primary = (LongType *) shapeBuffers[i]->primary();
    if(buffer != nullptr && buffer[i] != nullptr) {
      setGraphContextInputBuffer(ptr,i,buffers[i],shapeBuffers[i],specialShapeBuffers != nullptr ? specialShapeBuffers[i] : nullptr);
    }
    else {
      setGraphContextInputBuffer(ptr,i, nullptr,shapeBuffers[i],specialShapeInfo != nullptr ? specialShapeBuffers[i] : nullptr);
    }
  }

}
void setGraphContextOutputBuffers(OpaqueContext* ptr, int numArrays, void** buffer,
                                  void **shapeInfo, void **specialShapeInfo) {

  OpaqueDataBuffer **buffers = (OpaqueDataBuffer **) buffer;
  OpaqueDataBuffer **shapeBuffers = (OpaqueDataBuffer **) shapeInfo;
  OpaqueDataBuffer **specialShapeBuffers = (OpaqueDataBuffer **) specialShapeInfo;
  for(int i = 0; i < numArrays; i++) {

    if(buffer != nullptr && buffer[i] != nullptr) {
      setGraphContextOutputBuffer(ptr, i, buffers[i], shapeBuffers[i],
                                  specialShapeBuffers != nullptr ? specialShapeBuffers[i] : nullptr);
    } else {
      setGraphContextOutputBuffer(ptr,i, nullptr,shapeBuffers[i],specialShapeBuffers != nullptr ? specialShapeBuffers[i] : nullptr);
    }

  }

}