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
#include <execution/cuda/LaunchDims.h>
#include <loops/special_kernels.h>

#include "../../array/ShapeList.h"
#include "../../ops/declarable/DeclarableOp.h"
#include "../../system/common.h"
#include "../NativeOps.h"
cudaDeviceProp *deviceProperties;
cudaFuncAttributes *funcAttributes = new cudaFuncAttributes[64];
int blockLimit = 128;
int maxThreads = 512;
bool allowedP2P = false;
bool supportedP2P = false;



//note we only include this if we're running gcc linux
//and should not be enabled in default builds.
#if defined(SD_GCC_FUNCTRACE)
#include <cxxabi.h>  // needed  __cxa_demangle
#include <dlfcn.h>   // needed for dladdr

#include "exceptions/backward.hpp"
#include "execution/cuda/LaunchDims.h"


//note this is outside extern C. This is fine.


#endif





int minThreads = 32;

__constant__ char deviceConstantMemory[49152];



// this method just does type conversion in fancy way
int getDeviceId(Pointer ptrToDeviceId) { return (int)(LongType)ptrToDeviceId; }
// Function to execute a custom operation with context
sd::Status execCustomOp2(sd::Pointer *extraPointers, sd::LongType  hash, Context *opContext) {
  try {
    // Retrieve the operation based on the hash
    auto op = sd::ops::OpRegistrator::getInstance().getOperation(hash);
    if (op == nullptr) {
      throw std::invalid_argument("Operation not found for the given hash.");
    }

    // Execute the custom operation with the provided context
    auto result = op->execute(opContext);

    // Synchronize the CUDA stream to ensure operation completion
    auto res = cudaStreamSynchronize(*opContext->launchContext()->getCudaStream());
    if (res != cudaSuccess) {
      std::string errorMessage;
      errorMessage += "CUDA stream synchronization failed with error code: ";
      errorMessage += std::to_string(res);
      THROW_EXCEPTION(errorMessage.c_str());
    }

    // Synchronize fastpath inputs
    for (auto v : opContext->fastpath_in()) {
      if (!v->isEmpty()) v->syncToDevice();
    }

    // Synchronize fastpath outputs
    for (auto v : opContext->fastpath_out()) {
      if (!v->isEmpty()) v->syncToDevice();
    }

    return result;
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return sd::Status::KERNEL_FAILURE;
  }
}
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


}



void execPairwiseTransform(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execPairwiseTransform(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        extraParams);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execPairwiseTransformBool(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execPairwiseBoolTransform(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        extraParams);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


////////////////////////////////////////////////////////////////////////
void execSummaryStatsScalar(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, bool biasCorrected) {
  try {
    x->prepareSpecialUse({z}, {x});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execSummaryStatsScalar(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        biasCorrected);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execBroadcastBool(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, y, dimension});

    auto dimensionBuffer = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto hTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[9]);
    auto tadOnlyShapeInfo = reinterpret_cast<LongType *>(extraPointers[10]);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers[11]);
    auto tadOnlyShapeInfoZ = reinterpret_cast<LongType *>(extraPointers[12]);
    auto tadOffsetsZ = reinterpret_cast<LongType *>(extraPointers[13]);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execBroadcastBool(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        extraParams,
        dimensionBuffer,
        dimensionLength,
        tadOnlyShapeInfo,
        tadOffsets,
        tadOnlyShapeInfoZ,
        tadOffsetsZ);

    x->registerSpecialUse({z}, {x, y, dimension});
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
void execBroadcast(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, y, dimension});

    auto dimensionBuffer = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto hTADShapeInfo = reinterpret_cast<LongType *>(extraPointers[9]);
    auto tadOnlyShapeInfo = reinterpret_cast<LongType *>(extraPointers[10]);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers[11]);
    auto tadOnlyShapeInfoZ = reinterpret_cast<LongType *>(extraPointers[12]);
    auto tadOffsetsZ = reinterpret_cast<LongType *>(extraPointers[13]);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execBroadcast(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        dimensionBuffer,
        dimensionLength,
        tadOnlyShapeInfo,
        tadOffsets,
        tadOnlyShapeInfoZ,
        tadOffsetsZ);

    x->registerSpecialUse({z}, {x, y, dimension});
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
void execReduceFloat(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceFloatScalar(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceSame(Pointer *extraPointers,
                    int opNum,
                    OpaqueNDArray x,
                    void *extraParams,
                    OpaqueNDArray z) {
  try {


    x->prepareSpecialUse({z}, {x});
    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceSameScalar(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr: x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo())  ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

   x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceSame2(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<LongType>();
    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceSame(&lc,
                                        opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong2(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceLong(&lc, opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceLong(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceLong(&lc, opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceBool2(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceBool(&lc,
                                        opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

    delete dims;
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execReduceBool(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceBool(&lc,
                                        opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(), extraParams,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        zShapeInfoH,
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});

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
 * @param dZ
 * @param dZShapeInfo
 * @param dimension
 * @param dimensionLength
 */
////////////////////////////////////////////////////////////////////////
void execIndexReduce(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionData, dimensionLength);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execIndexReduce(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        dimensionData, dimensionLength, tadPack->specialShapeInfo(), tadPack->specialOffsets());

    x->registerSpecialUse({z}, {x});
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
void execReduceFloat2(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    const auto zLen = shape::length(z->shapeInfo());

    std::vector<LongType> dimensions(dimensionData, dimensionData + dimensionLength);

    const LongType *zShapeInfoH = z->shapeInfo();

    if (shape::rank(x->shapeInfo()) - dimensionLength != shape::rank(z->shapeInfo()) && zLen != 1) {
      auto zPack = ConstantShapeHelper::getInstance().createShapeInfoWithNoUnitiesForReduce(z->shapeInfo(), &dimensions);
      zShapeInfoH = reinterpret_cast<LongType const *>(zPack->primary());
    }

    std::vector<LongType> *dims =
        (zLen != 1) ? ShapeUtils::evalDimsForReduceOp(shape::rank(x->shapeInfo()), &dimensions) : new std::vector<LongType>();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduceFloat(&lc,
                                         opNum,
                                         shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                         x->shapeInfo(),
                                         shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                         x->specialShapeInfo(), extraParams,
                                         shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                         zShapeInfoH,
                                         shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                         z->specialShapeInfo(),
                                         dims->data(), dims->size());

    x->registerSpecialUse({z}, {x});
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
void execIndexReduceScalar(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execIndexReduceScalar(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execTransformSame(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformSame(&lc, opNum,
                                           shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                           x->shapeInfo(),
                                           shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                           x->specialShapeInfo(),
                                           shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                           z->shapeInfo(),
                                           shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                           z->specialShapeInfo(),
                                           extraParams, tadShapeInfo, tadOffsets);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execTransformBool(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[0] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[1] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformBool(&lc,
                                           opNum,
                                           shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                           x->shapeInfo(),
                                           shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                           x->specialShapeInfo(),
                                           shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                           z->shapeInfo(),
                                           shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                           z->specialShapeInfo(),
                                           extraParams,
                                           tadShapeInfo,
                                           tadOffsets);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execTransformAny(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    auto streamSpecial = reinterpret_cast<cudaStream_t &>(extraPointers[4]);
    LaunchContext lc(stream, streamSpecial, extraPointers[5], extraPointers[3], reinterpret_cast<int *>(extraPointers[6]));

    NativeOpExecutioner::execTransformAny(&lc,
                                          opNum,
                                          shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                          x->shapeInfo(),
                                          shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                          x->specialShapeInfo(),
                                          shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                          z->shapeInfo(),
                                          shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                          z->specialShapeInfo(),
                                          extraParams, nullptr, nullptr);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformStrict(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[10] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[11] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformStrict(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(), extraParams,
        tadShapeInfo, tadOffsets);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execTransformFloat(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x});

    auto tadShapeInfo = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[10] : nullptr);
    auto tadOffsets = reinterpret_cast<LongType *>(extraPointers != nullptr ? extraPointers[11] : nullptr);

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execTransformFloat(
        &lc,
        opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(), extraParams,
        tadShapeInfo,
        tadOffsets);

    x->registerSpecialUse({z}, {x});
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

void pullRows(Pointer *extraPointers, OpaqueNDArray x, OpaqueNDArray z, LongType n, OpaqueNDArray indexes, sd::LongType dimension) {
  try {
    x->prepareSpecialUse({z}, {x});


    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    dim3 launchDims = getLaunchDims("pullRows");
    auto xType = x->dataType();

    std::vector<void*> xBuffers(n);
    std::vector<const LongType*> tadShapeInfoBuffers(n);
    std::vector<const LongType*> tadOffsetsBuffers(n);

    for (int i = 0; i < n; ++i) {
      // Calculate TADs for each x
      auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dimension, 1);
      tadShapeInfoBuffers[i] = const_cast<LongType*>(tadPackX->specialShapeInfo());
      tadOffsetsBuffers[i] = const_cast<LongType*>(tadPackX->specialOffsets());
    }

    void* zBuffer = z->specialBuffer();
    LongType* zShapeInfo = const_cast<LongType*>(z->specialShapeInfo());

    // Calculate TADs for z
    auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), &dimension, 1);
    LongType* zTadShapeInfoBuffer = const_cast<LongType*>(tadPackZ->specialShapeInfo());
    LongType* zTadOffsetsBuffer = const_cast<LongType*>(tadPackZ->specialOffsets());

    // Use the special buffer for indexes
    LongType* indexesBuffer = reinterpret_cast<LongType*>(indexes->specialBuffer());

    BUILD_SINGLE_SELECTOR(xType, pullRowsKernelGeneric,
                          (launchDims, stream, x->specialBuffer(), zBuffer, n, indexesBuffer,
                           *tadShapeInfoBuffers.data(), *tadOffsetsBuffers.data(), zTadShapeInfoBuffer, zTadOffsetsBuffer),
                          SD_COMMON_TYPES);

    DEBUG_KERNEL(stream, -1);

    for (int i = 0; i < n; ++i) {
      x->registerSpecialUse({z}, {x});
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void average(Pointer *extras,
             OpaqueNDArrayArr x,
             OpaqueNDArray z,int n, LongType length, bool propagate) {
  try {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extras[1]);
    int mode = getDeviceId(extras[3]);

    if (Environment::getInstance().isDebugAndVerbose()) printf("averageFloat called\n");

    auto xType = x[0]->dataType();

    // launching on gpu
    if (mode == 0) {
      dim3 launchDims = getLaunchDims("average");
      std::vector<void*> xBuffers(n);
      for (int i = 0; i < n; ++i) {
        xBuffers[i] = x[i]->specialBuffer();
      }

      BUILD_SINGLE_SELECTOR(xType, averagingKernelGeneric, (launchDims, stream, xBuffers.data(), z->specialBuffer(), n, length, propagate), SD_COMMON_TYPES);
      DebugHelper::checkErrorCode(stream, "AverageFloat(...) failed");
    } else {
      // launching on host memory
      BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::averageGeneric(x, z, n, length, propagate), SD_COMMON_TYPES);
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void accumulate(Pointer *extras, OpaqueNDArrayArr x,  OpaqueNDArray z, int n, LongType length) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extras[1]);
    int mode = getDeviceId(extras[3]);

    if (Environment::getInstance().isDebugAndVerbose()) printf("accumulateFloat called\n");
    auto xType = x[0]->dataType();


    // launching on gpu
    if (mode == 0) {
      // Extract buffers from each NDArray in the array
      std::vector<void*> xBuffers(n);
      for (int i = 0; i < n; ++i) {
        xBuffers[i] = x[i]->specialBuffer();
      }

      void* zBuffer = z->specialBuffer();

      dim3 launchDims = getAccumDims(n);
      BUILD_SINGLE_SELECTOR(xType, accumulateKernelGeneric, (launchDims, stream, xBuffers.data(), zBuffer, n, length), SD_COMMON_TYPES);
      DebugHelper::checkErrorCode(stream, "AccumulateFloat(...) failed");
    } else {
      std::vector<NDArray*> xBuffers(n);
      for (int i = 0; i < n; ++i) {
        xBuffers[i] = x[i];
      }


      // launching on host memory
      BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::accumulateGeneric(xBuffers.data(), z, n, length), SD_COMMON_TYPES);
    }
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


bool isExperimentalEnabled() { return Environment::getInstance().isExperimentalBuild(); }
void shuffle(Pointer *extras,
             OpaqueNDArrayArr x,
             OpaqueNDArrayArr z,
             int N,
             OpaqueNDArray dimension,
             OpaqueNDArray shuffleMap) {
  try {
    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extras[1]);

    auto xType = x[0]->dataType();
    dim3 launchDims = getLaunchDims("shuffle");

    // Extract buffers from each NDArray in the array
    std::vector<void*> xBuffers(N);
    std::vector<LongType*> xShapeInfos(N);
    std::vector<LongType*> tadShapeInfoBuffers(N);
    std::vector<LongType*> tadOffsetsBuffers(N);
    std::vector<void*> zBuffers(N);
    std::vector<LongType*> zShapeInfos(N);
    std::vector<LongType*> zTadShapeInfoBuffers(N);
    std::vector<LongType*> zTadOffsetsBuffers(N);

    for (int i = 0; i < N; ++i) {
      xBuffers[i] = x[i]->specialBuffer();
      xShapeInfos[i] = const_cast<LongType*>(x[i]->specialShapeInfo());

      zBuffers[i] = z[i]->specialBuffer();
      zShapeInfos[i] = const_cast<LongType*>(z[i]->specialShapeInfo());

      // Extract dimensions for each x[i] and z[i] from the array of arrays
      LongType* dimensions = reinterpret_cast<LongType*>(dimension->buffer());
      LongType dimLength = shape::length(dimension->shapeInfo());

      // Calculate TADs for each x
      auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(x[i]->shapeInfo(), dimensions, dimLength);
      tadShapeInfoBuffers[i] = const_cast<LongType*>(tadPackX->specialShapeInfo());
      tadOffsetsBuffers[i] = const_cast<LongType*>(tadPackX->specialOffsets());

      // Calculate TADs for each z
      auto tadPackZ = sd::ConstantTadHelper::getInstance().tadForDimensions(z[i]->shapeInfo(), dimensions, dimLength);
      zTadShapeInfoBuffers[i] = const_cast<LongType*>(tadPackZ->specialShapeInfo());
      zTadOffsetsBuffers[i] = const_cast<LongType*>(tadPackZ->specialOffsets());
    }

    BUILD_SINGLE_SELECTOR(xType, shuffleKernelGeneric,
                          (launchDims, stream, xBuffers.data(), xShapeInfos.data(), zBuffers.data(), N, reinterpret_cast<int*>(shuffleMap->buffer()), tadShapeInfoBuffers.data(), tadOffsetsBuffers.data(), zTadShapeInfoBuffers.data(), zTadOffsetsBuffers.data()),
                          SD_COMMON_TYPES);

    DebugHelper::checkErrorCode(stream, "shuffle(...) failed");
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void setOmpMinThreads(int threads) {
  minThreads = sd::math::sd_max<int>(32, threads);
  minThreads = sd::math::sd_min<int>(maxThreads, minThreads);
}

int getDevice() { return AffinityManager::currentDeviceId(); }

////////////////////////////////////////////////////////////////////////
void execSummaryStats(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z, bool biasCorrected) {
  try {
    x->prepareSpecialUse({z}, {x});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execSummaryStats(&lc,
                                          opNum,
                                          shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                          x->shapeInfo(),
                                          shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                          ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
                                          extraParams,
                                          shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                          z->shapeInfo(),
                                          shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                          ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
                                          biasCorrected);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execSummaryStatsTad(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray z,
                         OpaqueNDArray dimension, bool biasCorrected) {
  try {
    x->prepareSpecialUse({z}, {x});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionData = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    int dimensionLength = static_cast<int>(shape::length(dimension->shapeInfo()));

    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionData, dimensionLength);
    auto tadShapeInfo = tadPack->primaryShapeInfo();
    auto tadOffsets = tadPack->primaryOffsets();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execSummaryStats(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        dimensionData, dimensionLength, tadShapeInfo, tadOffsets, biasCorrected);

    x->registerSpecialUse({z}, {x});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execReduce3(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduce3(
        &lc,
        opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(),
        extraParams,
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo());

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execReduce3Tad(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, y});
    dimension->preparePrimaryUse({}, {dimension});

    auto dim = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dim, dimensionLength);
    auto xTadShapeInfo = xTadPack->primaryShapeInfo();
    auto xOffsets = xTadPack->primaryOffsets();

    auto yTadPack = ConstantTadHelper::getInstance().tadForDimensions(y->shapeInfo(), dim, dimensionLength);
    auto yTadShapeInfo = yTadPack->primaryShapeInfo();
    auto yOffsets = yTadPack->primaryOffsets();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);

    NativeOpExecutioner::execReduce3TAD(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        x->specialShapeInfo(), extraParams,
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        y->specialShapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        z->specialShapeInfo(),
        dim, dimensionLength,
        xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execReduce3Scalar(Pointer *extraPointers, int opNum, OpaqueNDArray x, void *extraParams, OpaqueNDArray y, OpaqueNDArray z) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduce3Scalar(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(), extraParams,
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
        y->shapeInfo(),
        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(y->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special());

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarBool(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execScalarBool(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->buffer(),
        scalar->shapeInfo(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special(), extraParams);

    x->registerSpecialUse({z}, {x, scalar});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarBoolTad(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});
    dimension->preparePrimaryUse({}, {dimension});

    auto dim = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dim, dimensionLength);
    auto xTadShapeInfo = xTadPack->primaryShapeInfo();
    auto xOffsets = xTadPack->primaryOffsets();

    auto zTadPack = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), dim, dimensionLength);
    auto zTadShapeInfo = zTadPack->primaryShapeInfo();
    auto zOffsets = zTadPack->primaryOffsets();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execScalarBool(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        extraParams,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->buffer(),
        scalar->shapeInfo(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special(),
        dim, dimensionLength,
        xTadShapeInfo, xOffsets, zTadShapeInfo, zOffsets);

    x->registerSpecialUse({z}, {x, scalar});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
  }
}

////////////////////////////////////////////////////////////////////////
void execScalar(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execScalar(
        &lc, opNum,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->buffer(),
        scalar->shapeInfo(),
        shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(scalar->shapeInfo())->special(), extraParams);

    x->registerSpecialUse({z}, {x, scalar});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execScalarTad(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray z, OpaqueNDArray scalar, void *extraParams, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, scalar});
    dimension->preparePrimaryUse({}, {dimension});

    auto dimensionPtr = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionPtr, dimensionLength);
    auto xTadShapeInfo = xTadPack->primaryShapeInfo();
    auto xOffsets = xTadPack->primaryOffsets();

    auto zTadPack = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), dimensionPtr, dimensionLength);
    auto zTadShapeInfo = zTadPack->primaryShapeInfo();
    auto zOffsets = zTadPack->primaryOffsets();

    cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto xType = ArrayOptions::dataType(x->shapeInfo());
    auto yType = ArrayOptions::dataType(scalar->shapeInfo());
    auto zType = ArrayOptions::dataType(z->shapeInfo());

    if (yType != xType && yType != BOOL && !isExperimentalEnabled())
      throw datatype_exception::build("execScalar both operands must have same data type", xType, yType);

    dim3 launchDims = getLaunchDims("scalarTad");

    BUILD_SINGLE_SELECTOR_THRICE(
        xType, functions::scalar::ScalarTransform,
        ::executeCudaAlongDimension(
            launchDims, stream, opNum,
            shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
            xTadShapeInfo,
            shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
            zTadShapeInfo,
            shape::isEmptyConst(scalar->shapeInfo()) ? nullptr : scalar->specialBuffer(),
            extraParams, dimensionPtr, dimensionLength, xTadShapeInfo, xOffsets, zTadShapeInfo, zOffsets),
        SD_COMMON_TYPES);

    DEBUG_KERNEL(stream, opNum);

    x->registerSpecialUse({z}, {x, scalar});
    dimension->registerSpecialUse({}, {dimension});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execRandom(Pointer *extraPointers, int opNum, Pointer stateHost, OpaqueNDArray z, void *extraArguments) {
  try {
    z->prepareSpecialUse({}, {z});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);

    NativeOpExecutioner::execRandom(
        &lc, opNum, stateHost,
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(),
        extraArguments);

    z->registerSpecialUse({}, {z});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

////////////////////////////////////////////////////////////////////////
void execRandom2(Pointer *extraPointers, int opNum, Pointer stateHost, OpaqueNDArray x, OpaqueNDArray z, void *extraArguments) {
  try {
    x->prepareSpecialUse({z}, {x});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);

    NativeOpExecutioner::execRandom(
        &lc, opNum, stateHost,
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
        x->shapeInfo(),
        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(x->shapeInfo())->special(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
        z->shapeInfo(),
        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
        ConstantShapeHelper::getInstance().bufferForShapeInfo(z->shapeInfo())->special(), extraArguments);

    x->registerSpecialUse({z}, {x});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}
////////////////////////////////////////////////////////////////////////
void execRandom3(Pointer *extraPointers, int opNum, Pointer stateHost, OpaqueNDArray x,
                 OpaqueNDArray y, OpaqueNDArray z, void *extraArguments) {
  try {
    x->prepareSpecialUse({z}, {x, y});

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execRandom(&lc, opNum, stateHost,
                                    shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                    x->shapeInfo(),
                                    shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                    x->specialShapeInfo(),
                                    shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
                                    y->shapeInfo(),
                                    shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
                                    y->specialShapeInfo(),
                                    shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                    z->shapeInfo(),
                                    shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                    z->specialShapeInfo(),
                                    extraArguments);

    x->registerSpecialUse({z}, {x, y});
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
void execReduce3All(Pointer *extraPointers, int opNum, OpaqueNDArray x, OpaqueNDArray y, OpaqueNDArray z, void *extraParamsVals, OpaqueNDArray dimension) {
  try {
    x->prepareSpecialUse({z}, {x, y, dimension});
    x->preparePrimaryUse({}, {dimension});

    auto dimensionPtr = dimension != nullptr ? reinterpret_cast<LongType *>(dimension->buffer()) : nullptr;
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), dimensionPtr, dimensionLength);
    auto xTadShapeInfo = xTadPack->primaryShapeInfo();
    auto xOffsets = xTadPack->primaryOffsets();

    auto yTadPack = ConstantTadHelper::getInstance().tadForDimensions(y->shapeInfo(), dimensionPtr, dimensionLength);
    auto yTadShapeInfo = yTadPack->primaryShapeInfo();
    auto yOffsets = yTadPack->primaryOffsets();

    LaunchContext lc(extraPointers[1], extraPointers[4], extraPointers[5], extraPointers[3]);
    NativeOpExecutioner::execReduce3All(&lc, opNum,
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->buffer(),
                                        x->shapeInfo(),
                                        shape::isEmptyConst(x->shapeInfo()) ? nullptr : x->specialBuffer(),
                                        x->specialShapeInfo(),
                                        extraParamsVals,
                                        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->buffer(),
                                        y->shapeInfo(),
                                        shape::isEmptyConst(y->shapeInfo()) ? nullptr : y->specialBuffer(),
                                        y->specialShapeInfo(),
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->buffer(),
                                        z->shapeInfo(),
                                        shape::isEmptyConst(z->shapeInfo()) ? nullptr : z->specialBuffer(),
                                        z->specialShapeInfo(),
                                        dimensionPtr,
                                        dimensionLength, xTadShapeInfo,
                                        xOffsets, yTadShapeInfo, yOffsets);

    x->registerSpecialUse({z}, {x, y});
  } catch (std::exception &e) {
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

void sort(Pointer *extraPointers, OpaqueNDArray x, bool descending) {
  try {
    // Retrieve the CUDA stream from extraPointers
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      // If no stream is provided, use the default stream
      stream = LaunchContext::defaultContext()->getCudaStream();
    }

    // Extract shape information from NDArray*
    const LongType *xShapeInfo = x->shapeInfo();
    const LongType *dXShapeInfo = x->specialShapeInfo();

    // Calculate the length of the array
    auto xLength = shape::length(xShapeInfo);

    // Get element-wise stride (not used in original logic but retrieved for consistency)
    auto xEWS = shape::elementWiseStride(xShapeInfo);

    // Determine the data type of the array
    auto xType = ArrayOptions::dataType(xShapeInfo);

    // Check if xLength is a power of 2 and within the specified limit
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      // Get the launch dimensions for full sort
      dim3 launchDims = getSortFullDims(xLength);

      // Perform bitonic sort steps
      for (int k = 2; k <= xLength; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
          BUILD_SINGLE_SELECTOR(xType, bitonicSortStepGeneric,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, j, k, xLength, descending),
                                SD_COMMON_TYPES);
        }
      }
    } else {
      // Get the launch dimensions for arbitrary sort
      dim3 launchDims = getSortFullDims(xLength);

      // Determine the maximum window size
      int max = 2, dg = 0;
      while (max < xLength) {
        max <<= 1;
        dg++;
      }
      max <<= 1;

      // Perform bitonic sort steps for arbitrary window sizes
      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          int half = n >> 1;
          BUILD_SINGLE_SELECTOR(xType, bitonicArbitraryStepGeneric,
                                (launchDims, stream, x->specialBuffer(), dXShapeInfo, n, xLength, rev, descending),
                                SD_COMMON_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    // Check for CUDA errors after sort execution
    DebugHelper::checkErrorCode(stream, "sort(...) failed");
  } catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages in the LaunchContext
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void sortByKey(Pointer *extraPointers, NDArray *x,
               NDArray *y, bool descending) {
  try {
    // Retrieve the CUDA stream from extraPointers[1]
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      // If no stream is provided, use the default stream from LaunchContext
      stream = LaunchContext::defaultContext()->getCudaStream();
    }

    // Extract shape information from NDArray* objects
    const LongType *xShapeInfo = x->shapeInfo();
    const LongType *dXShapeInfo = x->specialShapeInfo();
    const LongType *yShapeInfo = y->shapeInfo();
    const LongType *dyShapeInfo = y->specialShapeInfo();

    // Calculate the lengths of the arrays
    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);

    // Get element-wise strides (optional, based on original logic)
    auto xEWS = shape::elementWiseStride(xShapeInfo);

    // Determine the data types of the arrays
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    // Check if either array is empty
    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;

    // Ensure that keys and values have the same length
    if (xLength != yLength) THROW_EXCEPTION("sortByKey: keys and values must have the same size");

    // Check if xLength is a power of 2 and within the specified limit
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      // Get the launch dimensions for full sort
      dim3 launchDims = getSortFullDims(xLength);

      // Perform bitonic sort steps
      for (int k = 2; k <= xLength; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericKey,
                                (launchDims, stream, x->specialBuffer(),
                                    dXShapeInfo, y->specialBuffer(), dyShapeInfo, j, k, xLength, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
        }
      }
    } else {
      // Determine the number of threads and blocks
      int numThreads = sd::math::sd_min<int>(512, xLength);
      int numBlocks = xLength / numThreads;
      if (xLength % numThreads > 0 || numBlocks == 0) numBlocks++;
      numBlocks = sd::math::sd_min<int>(512, numBlocks);
      dim3 launchDims(numBlocks, numThreads, 32768);

      // Determine the maximum window size
      int max = 2;
      while (max < xLength) {
        max <<= 1;
      }
      max <<= 1;

      // Perform bitonic sort steps for arbitrary window sizes
      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericKey,
                                (launchDims, stream, x->specialBuffer(),
                                    dXShapeInfo, y->specialBuffer(), dyShapeInfo, n, xLength, rev, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    // Check for CUDA errors after sort execution
    DebugHelper::checkErrorCode(stream, "sortByKey(...) failed");
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages in the LaunchContext
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void sortByValue(Pointer *extraPointers,OpaqueNDArray x,
                 OpaqueNDArray y, bool descending) {
  try {
    // Retrieve the CUDA stream from extraPointers[1]
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      // If no stream is provided, use the default stream from LaunchContext
      stream = LaunchContext::defaultContext()->getCudaStream();
    }

    // Extract shape information from NDArray* objects
    const LongType *xShapeInfo = x->shapeInfo();
    const LongType *dXShapeInfo = x->specialShapeInfo();
    const LongType *yShapeInfo = y->shapeInfo();
    const LongType *dyShapeInfo = y->specialShapeInfo();

    // Calculate the lengths of the arrays
    auto xLength = shape::length(xShapeInfo);
    auto yLength = shape::length(yShapeInfo);

    // Get element-wise strides (optional, based on original logic)
    auto xEWS = shape::elementWiseStride(xShapeInfo);

    // Determine the data types of the arrays
    auto xType = ArrayOptions::dataType(yShapeInfo); // Note the swapped types in original code
    auto yType = ArrayOptions::dataType(xShapeInfo);

    // Check if either array is empty
    if (shape::isEmptyConst(xShapeInfo) || shape::isEmptyConst(yShapeInfo)) return;

    // Ensure that keys and values have the same length
    if (xLength != yLength) THROW_EXCEPTION("sortByValue: keys and values must have the same size");

    // Check if xLength is a power of 2 and within the specified limit
    if ((xLength != 0) && ((xLength & (xLength - 1)) == 0) && (xLength <= 1024 * 1024 * 10)) {
      // Get the launch dimensions for full sort
      dim3 launchDims = getSortFullDims(xLength);

      // Perform bitonic sort steps
      for (int k = 2; k <= xLength; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicSortStepGenericKey,
                                (launchDims, stream, y->specialBuffer(),
                                    dyShapeInfo, x->specialBuffer(),
                                    dXShapeInfo, j, k, xLength, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
        }
      }
    } else {
      // Determine the number of threads and blocks
      dim3 launchDims = getSortFullDims(xLength);

      // Determine the maximum window size
      int max = 2;
      while (max < xLength) {
        max <<= 1;
      }
      max <<= 1;

      // Perform bitonic sort steps for arbitrary window sizes
      for (int window = 2; window < max; window <<= 1) {
        int n = window;
        int rev = 0;
        do {
          BUILD_DOUBLE_SELECTOR(xType, yType, bitonicArbitraryStepGenericKey,
                                (launchDims, stream, y->specialBuffer(),
                                    dyShapeInfo, x->specialBuffer(), dXShapeInfo, n, xLength, rev, descending),
                                SD_COMMON_TYPES, SD_COMMON_TYPES);
          n >>= 1;
          rev = 1;
        } while (n > 1);
      }
    }

    // Check for CUDA errors after sort execution
    DebugHelper::checkErrorCode(stream, "sortByValue(...) failed");
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages in the LaunchContext
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void sortTadByKey(Pointer *extraPointers,
                  OpaqueNDArray x,
                  OpaqueNDArray y,
                  OpaqueNDArray dimension,
                  bool descending) {
  try {
    // Retrieve the CUDA stream from extraPointers[1]
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      // If no stream is provided, use the default stream from LaunchContext
      stream = LaunchContext::defaultContext()->getCudaStream();
    }

    // Extract shape information from NDArray* objects
    const LongType *xShapeInfo = x->shapeInfo();
    const LongType *dXShapeInfo = x->specialShapeInfo();
    const LongType *yShapeInfo = y->shapeInfo();
    const LongType *dyShapeInfo = y->specialShapeInfo();

    // Determine the data types of the arrays
    auto xType = ArrayOptions::dataType(xShapeInfo);
    auto yType = ArrayOptions::dataType(yShapeInfo);

    // Get the dimension buffer and length
    auto dimensionPtr = reinterpret_cast<LongType *>(dimension->buffer());
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    // Get the TAD pack for the given dimensions
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimensionPtr, dimensionLength);

    // Get the number of TADs
    auto numTads = tadPack->numberOfTads();

    // Get the launch dimensions for sorting TADs
    dim3 launchDims = getSortTadDims(numTads);

    // Execute the sortTadByKey operation based on data types
    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, x->specialBuffer(),
                           dXShapeInfo, y->specialBuffer(), dyShapeInfo,
                           dimensionPtr, dimensionLength, tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    // Check for CUDA errors after sort execution
    DebugHelper::checkErrorCode(stream, "sortTadByKey(...) failed");
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages in the LaunchContext
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void sortTadByValue(Pointer *extraPointers,
                    OpaqueNDArray x,
                    OpaqueNDArray y,
                    OpaqueNDArray dimension,
                    bool descending) {
  try {
    // Retrieve the CUDA stream from extraPointers[1]
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      // If no stream is provided, use the default stream from LaunchContext
      stream = LaunchContext::defaultContext()->getCudaStream();
    }

    // Extract shape information from NDArray* objects
    const LongType *xShapeInfo = x->shapeInfo();
    const LongType *dXShapeInfo = x->specialShapeInfo();
    const LongType *yShapeInfo = y->shapeInfo();
    const LongType *dyShapeInfo = y->specialShapeInfo();

    // Determine the data types of the arrays
    auto xType = ArrayOptions::dataType(yShapeInfo); // Note the swapped types in original code
    auto yType = ArrayOptions::dataType(xShapeInfo);

    // Get the dimension buffer and length
    auto dimensionPtr = reinterpret_cast<LongType *>(dimension->buffer());
    LongType dimensionLength = static_cast<LongType>(shape::length(dimension->shapeInfo()));

    // Get the TAD pack for the given dimensions
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimensionPtr, dimensionLength);

    // Get the number of TADs
    auto numTads = tadPack->numberOfTads();

    // Get the launch dimensions for sorting TADs
    dim3 launchDims = getSortTadDims(numTads);

    // Execute the sortTadByValue operation based on data types
    BUILD_DOUBLE_SELECTOR(xType, yType, oesTadGenericKey,
                          (launchDims, stream, y->specialBuffer(), dyShapeInfo, x->specialBuffer(), dXShapeInfo,
                           dimensionPtr, dimensionLength, tadPack->platformShapeInfo(), tadPack->platformOffsets(), descending),
                          SD_COMMON_TYPES, SD_COMMON_TYPES);

    // Check for CUDA errors after sort execution
    DebugHelper::checkErrorCode(stream, "sortTadByValue(...) failed");
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages in the LaunchContext
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void sortTad(Pointer *extraPointers, OpaqueNDArray  x,
             LongType *dimension, LongType dimensionLength,
              LongType *tadShapeInfo,  LongType *tadOffsets, bool descending) {
  try {
    // Retrieve the CUDA stream from extraPointers[1]
    cudaStream_t *stream = nullptr;
    if (extraPointers != nullptr && extraPointers[1] != nullptr) {
      stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);
    } else {
      // If no stream is provided, use the default stream from LaunchContext
      stream = LaunchContext::defaultContext()->getCudaStream();
    }

    // Extract shape information from NDArray* objects
    const LongType *xShapeInfo = x->shapeInfo();
    const LongType *dXShapeInfo = x->specialShapeInfo();

    // Determine the data type of the array
    auto xType = ArrayOptions::dataType(xShapeInfo);

    // Get the TAD pack for the given dimensions
    auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);

    // Get the number of TADs
    auto numTads = tadPack->numberOfTads();

    // Get the launch dimensions for sorting TADs
    dim3 launchDims = getSortTadLarge(numTads);

    // Execute the sortTad operation based on data type
    BUILD_SINGLE_SELECTOR(
        xType, oesTadGeneric,
        (launchDims, stream, x->specialBuffer(), dXShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending),
        SD_COMMON_TYPES
    );

    // Check for CUDA errors after sort execution
    DebugHelper::checkErrorCode(stream, "sortTad(...) failed");
  }
  catch (std::exception &e) {
    // Handle exceptions by setting error codes and messages in the LaunchContext
    LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}


void sortCooIndices(sd::Pointer *extraPointers, OpaqueNDArray indices, OpaqueNDArray values) {
  THROW_EXCEPTION("sortCooIndices:: Not implemented yet");
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
SD_HOST static void scatterUpdateCudaLauncher(const cudaStream_t *stream,
                                              int opCode, OpaqueNDArray array, OpaqueNDArray indices, OpaqueNDArray updates, LongType *axis, LongType axisLength) {
    // Calculate TADs for x and y
    auto tadPackX = sd::ConstantTadHelper::getInstance().tadForDimensions(array->shapeInfo(),axis, axisLength);
    auto tadPackY = sd::ConstantTadHelper::getInstance().tadForDimensions(updates->shapeInfo(), axis,axisLength);

    auto xTadOffsets = tadPackX->specialOffsets();
    auto yTadOffsets = tadPackY->specialOffsets();

    scatterUpdateCuda<T, I><<<512, 256, SD_MAX_NUM_THREADS, *stream>>>(
        opCode, tadPackX->numberOfTads(), array->specialBuffer(), tadPackX->specialShapeInfo(), xTadOffsets, updates->specialBuffer(),
        tadPackY->specialShapeInfo(), yTadOffsets, indices->specialBuffer());
}
//////////////////////////////////////////////////////////////////////////
void scatterUpdate(Pointer *extraPointers, int opCode, OpaqueNDArray array, OpaqueNDArray indices, OpaqueNDArray updates, OpaqueNDArray axis) {
  try {
    auto stream = reinterpret_cast<cudaStream_t *>(extraPointers[1]);

    auto type = ArrayOptions::dataType(array->shapeInfo());
    auto iType = ArrayOptions::dataType(indices->shapeInfo());

    // Extract axis data
    LongType* axisData = reinterpret_cast<LongType*>(axis->buffer());
    LongType axisLength = shape::length(axis->shapeInfo());

    BUILD_DOUBLE_SELECTOR(
        type, iType, scatterUpdateCudaLauncher,
        (stream, opCode, array, indices, updates, axisData, axisLength),
        SD_COMMON_TYPES, SD_INDEXING_TYPES);

    DebugHelper::checkErrorCode(stream, "scatterUpdate(...) failed");
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


void setGraphContextCudaContext(Context *ptr, void *stream, void *reductionPointer,
                                void *allocationPointer) {
  ptr->setCudaContext(stream, reductionPointer, allocationPointer);
}




int binaryLevel() { return 0; }

int optimalLevel() { return 0; }

bool isMinimalRequirementsMet() { return true; }

bool isOptimalRequirementsMet() { return true; }








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
  for(int i = 0; i < len; i++) {
    bufferToSet[i] = inputShapeData[i];
  }

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
