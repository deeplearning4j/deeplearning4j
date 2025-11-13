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
void* mapFromNpzFile(std::string path) {
  cnpy::npz_t* mapPtr = new cnpy::npz_t();
  cnpy::npz_t map = cnpy::npzLoad(path);
  mapPtr->insert(map.begin(), map.end());
  return reinterpret_cast<void*>(mapPtr);
}

int getNumNpyArraysInMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  int n = arrays->size();
  return n;
}

const char* getNpyArrayNameFromMap(void* map, int index, char* nameBuffer) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      size_t len_of_str = strlen(it->first.c_str());
      memcpy(nameBuffer, it->first.c_str(), len_of_str);
    }
  }
  return "";
}

void* getNpyArrayFromMap(void* map, int index) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  cnpy::npz_t::iterator it = arrays->begin();
  cnpy::npz_t::iterator end = arrays->end();
  cnpy::NpyArray* arr = new cnpy::NpyArray();
  int cnt = 0;
  for (; it != end; ++it, ++cnt) {
    if (cnt == index) {
      *arr = it->second;
      return arr;
    }
  }

 return nullptr;
}


void* getNpyArrayData(void* npArray) {
  cnpy::NpyArray* npyArray2 = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return reinterpret_cast<void*>(npyArray2->data);
}

int getNpyArrayRank(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int rank = arr->shape.size();
  return rank;
}

sd::LongType* getNpyArrayShape(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  int ndim = arr->shape.size();
  sd::LongType* shape = new sd::LongType[ndim];
  for (int i = 0; i < ndim; i++) {
    shape[i] = arr->shape.at(i);
  }
  return shape;
}

char getNpyArrayOrder(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return (arr->fortranOrder) ? 'f' : 'c';
}

int getNpyArrayElemSize(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  return arr->wordSize;
}

void deleteNPArrayStruct(void* npArray) {
  cnpy::NpyArray* arr = reinterpret_cast<cnpy::NpyArray*>(npArray);
  delete arr;
}

void deleteNPArrayMap(void* map) {
  cnpy::npz_t* arrays = reinterpret_cast<cnpy::npz_t*>(map);
  delete arrays;
}
//////

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
int elementSizeForNpyArray(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromPointer(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  // arrPointer->destruct();
  return size;
}

/**
 * Get the element size for a numpy array
 * @param npyArray  the numpy array's address
 * to get the length for
 * @return
 */
int elementSizeForNpyArrayHeader(sd::Pointer npyArray) {
  cnpy::NpyArray arr = cnpy::loadNpyFromHeader(reinterpret_cast<char*>(npyArray));
  cnpy::NpyArray* arrPointer = &arr;
  int size = arrPointer->wordSize;
  return size;
}

void releaseNumpy(sd::Pointer npyArray) { free(reinterpret_cast<void*>(npyArray)); }

#if defined(SD_GCC_FUNCTRACE)
// this is mainly a c based function.
extern "C" {

//note this is a c++ 17 feature
#ifndef INSTRUMENT_FILE_DEF
#define INSTRUMENT_FILE_DEF 1
FILE* instrumentFile = nullptr;
#endif





}

#endif

void ctxAllowHelpers(OpaqueContext *ptr, bool reallyAllow) { ptr->allowHelpers(reallyAllow); }

void ctxSetExecutionMode(OpaqueContext *ptr, int execMode) {
  if (execMode < 0 || execMode > 2) execMode = 0;

  ptr->setExecutionMode((samediff::ExecutionMode)execMode);
}

sd::LongType getCachedMemory(int deviceId) { return sd::ConstantHelper::getInstance().getCachedAmount(deviceId); }


void ctxShapeFunctionOverride(OpaqueContext *ptr, bool reallyOverride) {
  ptr->setShapeFunctionOverride(reallyOverride);
}

void ctxPurge(OpaqueContext *ptr) { ptr->clearFastPath(); }

int lastErrorCode() { return sd::LaunchContext::defaultContext()->errorReference()->errorCode(); }

const char *lastErrorMessage() { return sd::LaunchContext::defaultContext()->errorReference()->errorMessage(); }


sd::LaunchContext *defaultLaunchContext() { return sd::LaunchContext::defaultContext(); }





void setIntermediateResult(OpaqueContext *contextPointer,
                           int index,
                           OpaqueDataBuffer *buffer,
                           OpaqueDataBuffer *shapeInfo,
                           sd::LongType dataOffset) {
  if(shapeInfo == nullptr) {
    THROW_EXCEPTION("Set Intermediate Result: shapeInfo is null");
  }
  auto casted = reinterpret_cast<sd::LongType *>(shapeInfo->primary());
  auto desc = new sd::ShapeDescriptor(casted, false);
  auto arr = new sd::NDArray(buffer->dataBuffer(),
                             desc,
                             sd::LaunchContext::defaultContext(),
                             dataOffset);
  contextPointer->setIntermediateResult(index, arr);
}


std::vector<const sd::LongType *> intermediateResultsShapeInfo(OpaqueContext *contextPointer) {
  std::vector<const sd::LongType *> intermediates;
  for (auto v: contextPointer->intermediateResults()) {
    const sd::LongType *buff = v->shapeInfo();
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
                            sd::LongType offset) {
  auto shapeInfoCast = reinterpret_cast<sd::LongType *>(shapeInfo->primary());
  auto desc = new sd::ShapeDescriptor(shapeInfoCast, false);
  auto arr = new sd::NDArray(buffer->dataBuffer(), desc, sd::LaunchContext::defaultContext(), offset);
  contextPointer->pushIntermediateResult(arr);
}

OpaqueDataBuffer  * intermediateResultDataAt(int index, OpaqueContext *contextPointer) {
  auto arr = contextPointer->intermediateResult(index);
  return new OpaqueDataBuffer(arr->dataBuffer());
}

const sd::LongType * intermediateResultShapeInfoAt(int index, OpaqueContext *contextPointer) {
  auto context = reinterpret_cast<sd::graph::Context *>(contextPointer);
  auto arr = context->intermediateResult(index);
  return arr->shapeInfo();
}


sd::LongType const *getPrimaryShapeInfo(sd::TadPack *pack) {
  return const_cast<sd::LongType *>(pack->primaryShapeInfo());
}

sd::LongType const *getPrimaryOffsets(sd::TadPack *pack) {
  if(pack->primaryOffsets() == nullptr)
    THROW_EXCEPTION("getPrimaryOffsets: primaryOffsets is nullptr!");
  return const_cast<sd::LongType *>(pack->primaryOffsets());
}

sd::LongType const *getSpecialShapeInfo(sd::TadPack *pack) {
  return const_cast<sd::LongType *>(pack->specialShapeInfo());
}

sd::LongType const *getSpecialOffsets(sd::TadPack *pack) { return const_cast<sd::LongType *>(pack->specialOffsets()); }

sd::LongType getNumberOfTads(sd::TadPack *pack) { return pack->numberOfTads(); }

int getShapeInfoLength(sd::TadPack *pack) { return pack->shapeInfoLength(); }


sd::TadPack *tadOnlyShapeInfo(OpaqueDataBuffer *hXShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength) {
  try {
    if(hXShapeInfo->primary() == nullptr) {
      THROW_EXCEPTION("tadOnlyShapeInfo: hXShapeInfo->primary() is nullptr!");
    }

    auto buffPrim = reinterpret_cast<sd::LongType *>(hXShapeInfo->primary());
    auto shapeFromCache = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(buffPrim)->primary();
    auto rankVal = shapeFromCache[0];
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




    auto pack = sd::ConstantTadHelper::getInstance().tadForDimensions(
        shapeFromCache, dimension, dimensionLength);
    return pack;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    THROW_EXCEPTION(e.what());
  }

  return nullptr;

}


OpaqueConstantShapeBuffer shapeBuffer(int rank, sd::LongType *shape, sd::LongType *strides, sd::DataType dtype,
                                      char order, sd::LongType ews, bool empty) {
  return shapeBufferEx(rank, shape, strides, dtype, order, ews, empty ? ARRAY_EMPTY : 0);
}

void dbPrintAllocationTrace(OpaqueDataBuffer *db) { db->dataBuffer()->printAllocationTrace(); }

sd::LongType dbBufferLength(OpaqueDataBuffer *dataBuffer) {
  return dataBuffer->dataBuffer()->getNumElements();
}


OpaqueDataBuffer *dbAllocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  return allocateDataBuffer(elements, dataType, allocateBoth);
}

OpaqueDataBuffer *allocateDataBuffer(sd::LongType elements, int dataType, bool allocateBoth) {
  try {
    auto dtype = sd::DataTypeUtils::fromInt(dataType);
    sd::LongType totalElementSize = elements == 0 ? sd::DataTypeUtils::sizeOf(dtype) : elements * sd::DataTypeUtils::sizeOf(dtype);
    auto buffer = new sd::InteropDataBuffer(totalElementSize, dtype, allocateBoth);

    // Track allocation
    if (buffer != nullptr) {
      size_t bytes = totalElementSize;
      g_dataBufferCount.fetch_add(1, std::memory_order_relaxed);
      g_dataBufferBytes.fetch_add(bytes, std::memory_order_relaxed);

      if(sd::Environment::getInstance().isVerbose()) {
        sd_printf("allocateDataBuffer: allocated buffer at %p, count=%zu, total_bytes=%zu, this_bytes=%zu\n",
                  buffer, g_dataBufferCount.load(), g_dataBufferBytes.load(), bytes);
      }
    }

    return buffer;
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) {
  auto buffer = dbAllocateDataBuffer(0, dataType, false);
  buffer->markOwner(false);

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) buffer->setSpecial(special, elements);

  return buffer;
}

sd::Pointer dbPrimaryBuffer(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbPrimaryBuffer: dataBuffer is null");
  return dataBuffer->primary();
}

sd::Pointer dbSpecialBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSpecialBuffer: dataBuffer is null");
  return dataBuffer->special();
}

void deleteDataBuffer(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("deleteDataBuffer: dataBuffer is null");

  // Close the buffer first to ensure proper cleanup of underlying DataBuffer
  // This updates tracking counters and frees the actual data
  dbClose(dataBuffer);

  // Now delete the wrapper
  delete dataBuffer;
}

void dbSetPrimaryBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer primaryBuffer, sd::LongType numBytes) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetPrimaryBuffer: dataBuffer is null");
  dataBuffer->setPrimary(primaryBuffer, numBytes);
}

void dbSetSpecialBuffer(OpaqueDataBuffer *dataBuffer, sd::Pointer specialBuffer, sd::LongType numBytes) {
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

void dbExpandBuffer(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  try {
    if(dataBuffer == nullptr)
      THROW_EXCEPTION("dbExpandBuffer: dataBuffer is null");
    dataBuffer->dataBuffer()->expand(elements * sd::DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
  }
}

OpaqueDataBuffer *dbCreateView(OpaqueDataBuffer *dataBuffer, sd::LongType length) {
  return new OpaqueDataBuffer(dataBuffer, length);
}


int dbUseCount(OpaqueDataBuffer* dataBuffer) {
  if(dataBuffer) return dataBuffer->useCount();
  return 0;
}

void dbSyncToSpecial(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToSpecial: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr  && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToSpecial();
}

void dbSyncToPrimary(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSyncToPrimary: dataBuffer is null");
  if(dataBuffer->dataBuffer() != nullptr  && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToPrimary(sd::LaunchContext::defaultContext(),false);

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

void dbExpand(OpaqueDataBuffer *dataBuffer, sd::LongType elements) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbExpand: dataBuffer is null");
  dataBuffer->expand(elements);
}

void dbClose(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbClose: dataBuffer is null");

  // Check if already closed - this flag is in InteropDataBuffer, not the freed DataBuffer
  if(dataBuffer->_closed) {
    return;
  }

  // Check constant flag (public field, safe to access)
  if(dataBuffer->isConstant) {
    return;
  }

  // Check if we even have a DataBuffer pointer
  if(!dataBuffer->hasValidDataBuffer()) {
    dataBuffer->_closed = true;
    return;
  }

  // If we don't own it, don't close it
  if(!dataBuffer->isOwner()) {
    return;
  }

  // Track deallocation using cached size - DO NOT touch the DataBuffer as it may be freed
  // Use the cached size from InteropDataBuffer instead of accessing potentially freed memory
  size_t bytes = dataBuffer->_cachedLenInBytes;
  g_dataBufferCount.fetch_sub(1, std::memory_order_relaxed);
  g_dataBufferBytes.fetch_sub(bytes, std::memory_order_relaxed);

  if(sd::Environment::getInstance().isVerbose()) {
    sd_printf("dbClose: deallocating buffer at %p, count=%zu, total_bytes=%zu, freed_bytes=%zu\n",
              dataBuffer, g_dataBufferCount.load(), g_dataBufferBytes.load(), bytes);
  }

  // Do NOT call db->close() - if this InteropDataBuffer wraps a DataBuffer that was owned
  // by an NDArray, that DataBuffer may already be freed. The NDArray destructor will have
  // already called close() on its DataBuffer. Calling it again causes use-after-free.
  // Just update tracking and mark as closed.

  // Mark as closed and invalidate pointer after freeing
  dataBuffer->_closed = true;
  dataBuffer->invalidateDataBuffer();
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

