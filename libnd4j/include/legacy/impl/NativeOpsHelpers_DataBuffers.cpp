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
#include <helpers/TransferMetrics.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>

#include "execution/Threads.h"
#include "helpers/OpTracker.h"

#if defined(SD_GCC_FUNCTRACE)
#include <array/DataBufferLifecycleTracker.h>
#endif

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
#include <sys/types.h>
#include <unordered_map>


extern bool experimentalSupport; // Defined in NativeOpsHelpers_Arrays.cpp

// OpaqueNDArray allocation tracking
static std::atomic<size_t> g_opaqueArrayCount{0};
static std::atomic<size_t> g_opaqueArrayBytes{0};
static std::mutex g_opaqueArrayMutex;

// DataBuffer allocation tracking - non-static so platform-specific dbClose can access
std::atomic<size_t> g_dataBufferCount{0};
std::atomic<size_t> g_dataBufferBytes{0};
static std::mutex g_dataBufferMutex;

// TadPack lifetime registry moved to NativeOpsHelpers_DataBuffers_tad.cpp
// extern reference if needed:
extern std::unordered_map<sd::TadPack*, std::shared_ptr<sd::TadPack>> g_tadPackRegistry;
extern std::mutex g_tadPackMutex;

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




// NPZ/numpy interop functions moved to NativeOpsHelpers_DataBuffers_npz.cpp
// TAD pack functions moved to NativeOpsHelpers_DataBuffers_tad.cpp
// Transfer metrics functions moved to NativeOpsHelpers_DataBuffers_metrics.cpp

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
void ctxPurgeNoSync(OpaqueContext *ptr) { ptr->clearFastPathNoSync(); }

int lastErrorCode() { return sd::LaunchContext::defaultContext()->errorReference()->errorCode(); }

const char *lastErrorMessage() { return sd::LaunchContext::defaultContext()->errorReference()->errorMessage(); }

// For CUDA builds, clearLastError is defined in NativeOps.cu with CUDA-specific error clearing
#ifndef SD_CUDA
void clearLastError() {
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
}
#endif

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
  if (contextPointer == nullptr) {
    THROW_EXCEPTION("intermediateResultDataAt: contextPointer is null");
  }
  auto arr = contextPointer->intermediateResult(index);
  if (arr == nullptr) {
    THROW_EXCEPTION("intermediateResultDataAt: intermediateResult returned null");
  }
  return new OpaqueDataBuffer(arr->dataBuffer());
}

const sd::LongType * intermediateResultShapeInfoAt(int index, OpaqueContext *contextPointer) {
  if (contextPointer == nullptr) {
    THROW_EXCEPTION("intermediateResultShapeInfoAt: contextPointer is null");
  }
  auto context = reinterpret_cast<sd::graph::Context *>(contextPointer);
  auto arr = context->intermediateResult(index);
  if (arr == nullptr) {
    THROW_EXCEPTION("intermediateResultShapeInfoAt: intermediateResult returned null");
  }
  return arr->shapeInfo();
}


// TAD pack functions (tadOnlyShapeInfo, clearTadPackRegistry, etc.) moved to NativeOpsHelpers_DataBuffers_tad.cpp


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
#ifdef __cpp_exceptions
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
    safeSetErrorContext(1, e.what());
    return nullptr;
  }
#else
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
#endif
}

OpaqueDataBuffer *dbCreateExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) {
  // Create an InteropDataBuffer and set external pointers
  // Note: This allocates a small internal buffer which is then overwritten with external pointers.
  // The external pointers (e.g., cached shape info) are NOT owned by this buffer.
  auto buffer = dbAllocateDataBuffer(0, dataType, false);

  // Critical: check for null - allocation can fail under concurrent access or CUDA errors
  if (buffer == nullptr) {
    sd_printf("dbCreateExternalDataBuffer: allocation failed for dataType=%d, elements=%lld\n", dataType, elements);
    return nullptr;
  }

  buffer->markOwner(false);

  // Clean up stale auto-allocated buffers BEFORE setting external pointers.
  // dbAllocateDataBuffer(0,...) creates a small internal buffer on CUDA (device side).
  // If the caller only provides a host pointer (e.g., HOST_ONLY workspace), the stale
  // auto-allocated device buffer must be cleared. Otherwise syncToPrimary will try to
  // cudaMemcpyAsync from the tiny stale device buffer using the full _lenInBytes,
  // causing "invalid argument" errors.
  // We clear BEFORE setPrimary/setSpecial so that the final _lenInBytes is correct
  // (setPrimary/setSpecial update _lenInBytes based on the element count).
  if (special == nullptr && primary != nullptr) {
    // Only host pointer provided (e.g., HOST_ONLY workspace) — clear stale device buffer.
    // setSpecial(nullptr, 0) -> setSpecialBuffer(nullptr, 0) -> frees old device memory,
    // nulls the pointer, and temporarily sets _lenInBytes=0.
    // setPrimary below will restore _lenInBytes to the correct value.
    buffer->setSpecial(nullptr, 0);
  }

  if (primary != nullptr) buffer->setPrimary(primary, elements);

  if (special != nullptr) {
    buffer->setSpecial(special, elements);
  } else if (primary != nullptr) {
    // After clearing the stale special buffer and setting primary, the sync counters
    // still indicate "special is more recent" (from the initial writeSpecial() in the
    // DataBuffer constructor). This causes isSpecialActual() to return true, and
    // syncToSpecial() skips allocation+copy entirely — leaving no device buffer.
    // Fix: mark primary as written so syncToSpecial() knows H2D transfer is needed.
    auto db = buffer->dataBuffer();
    if (db != nullptr) {
      db->writePrimary();
    }
  }

  return buffer;
}

OpaqueDataBuffer *dbCreateConstantExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) {
  // Create an externalized buffer and mark it constant IMMEDIATELY
  // This is critical for preventing race conditions where Java GC can finalize
  // the buffer before setConstant() is called on the Java side.
  //
  // By marking constant HERE (in native code), the buffer is protected
  // before it even returns to Java, eliminating the race window entirely.
  auto buffer = dbCreateExternalDataBuffer(elements, dataType, primary, special);

  if (buffer != nullptr) {
    // Mark constant IMMEDIATELY, before returning to Java
    // Use release memory order to ensure visibility to other threads
    buffer->isConstant.store(true, std::memory_order_release);

    // Also propagate to underlying DataBuffer if it exists
    auto db = buffer->dataBuffer();
    if (db != nullptr) {
      db->markConstant(true);
    }

    if (sd::Environment::getInstance().isVerbose()) {
      sd_printf("dbCreateConstantExternalDataBuffer: created constant buffer at %p\n", buffer);
    }
  }

  return buffer;
}

bool dbSetConstant(OpaqueDataBuffer *dataBuffer, bool isConstant) {
  if (dataBuffer == nullptr) {
    // Null buffer - return false to indicate failure
    // This is a programming error, but we don't throw to maintain backwards compatibility
    if (sd::Environment::getInstance().isVerbose()) {
      sd_printf("dbSetConstant: null buffer passed, returning false%s", "\n");
    }
    return false;
  }

  if (!dataBuffer->isValid()) {
    // Buffer is invalid (freed or closed) - return false to indicate failure
    // This can happen when:
    // 1. DeallocatorService runs dbClose() before setConstant() is called
    // 2. This indicates a race condition between GC and constant flag setting
    // 3. The Java side should use registerPendingConstant() to prevent this
    if (sd::Environment::getInstance().isVerbose()) {
      sd_printf("dbSetConstant: FAILED - buffer %p is invalid (freed or closed). "
                "This indicates a race condition - use registerPendingConstant() to protect buffers.\n",
                dataBuffer);
    }
    return false;
  }

  dataBuffer->isConstant.store(isConstant, std::memory_order_release);

  // Also propagate to the underlying DataBuffer if it exists
  auto db = dataBuffer->dataBuffer();
  if (db != nullptr) {
    db->markConstant(isConstant);
  }

  if (sd::Environment::getInstance().isVerbose()) {
    sd_printf("dbSetConstant: buffer %p marked as constant=%d\n", dataBuffer, isConstant);
  }

  return true;
}

bool dbIsConstant(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) {
    return false;
  }
  // Check if buffer is valid before accessing - return false for freed buffers
  // This prevents use-after-free when checking constant status during cleanup
  if (!dataBuffer->isValid()) {
    return false;
  }
  return dataBuffer->isConstant.load(std::memory_order_acquire);
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
  // Deleting null should be a safe no-op, similar to delete nullptr in C++
  // This is important for cleanup paths handling empty arrays which may have null buffers
  if(dataBuffer == nullptr) {
    return;
  }

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
#ifdef __cpp_exceptions
  try {
    if(dataBuffer == nullptr)
      THROW_EXCEPTION("dbExpandBuffer: dataBuffer is null");
    dataBuffer->dataBuffer()->expand(elements * sd::DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
  } catch (std::exception &e) {
    safeSetErrorContext(1, e.what());
  }
#else
  if(dataBuffer == nullptr) {
    safeSetErrorContext(1, "dbExpandBuffer: dataBuffer is null");
    return;
  }
  dataBuffer->dataBuffer()->expand(elements * sd::DataTypeUtils::sizeOf(dataBuffer->dataBuffer()->getDataType()));
#endif
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
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbSyncToPrimary: dataBuffer is null");
  if (dataBuffer->dataBuffer() != nullptr && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToPrimary(sd::LaunchContext::defaultContext(), false);
}

void dbForceSyncToPrimary(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbForceSyncToPrimary: dataBuffer is null");
  if (dataBuffer->dataBuffer() != nullptr && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToPrimary(sd::LaunchContext::defaultContext(), true);
}

void dbForceSyncToSpecial(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr) THROW_EXCEPTION("dbForceSyncToSpecial: dataBuffer is null");
  if (dataBuffer->dataBuffer() != nullptr && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->syncToSpecial(true);
}

#ifndef SD_CUDA
// CPU implementation - simple sequential fallback
void batchSyncToSpecialAsync(OpaqueDataBuffer **buffers, int bufferCount, int streamCount) {
  // On CPU, just iterate through and sync each buffer sequentially
  // The 'async' and 'streamCount' parameters are ignored as CPU doesn't have streams
  for (int i = 0; i < bufferCount; i++) {
    if (buffers[i] != nullptr) {
      dbSyncToSpecial(buffers[i]);
    }
  }
}
#endif

void dbMigrate(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr)
    THROW_EXCEPTION("dbMigrate: dataBuffer is null");
  if (dataBuffer->dataBuffer() != nullptr && dataBuffer->dataBuffer()->getNumElements() > 0)
    dataBuffer->dataBuffer()->migrate();
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

// dbClose is implemented in cpu/NativeOpsHelpers_DataBuffers_close.cpp
// and cuda/NativeOpsHelpers_DataBuffers_close.cu for platform-specific handling

int dbDeviceId(OpaqueDataBuffer *dataBuffer) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbDeviceId: dataBuffer is null");
  return dataBuffer->deviceId();
}

void dbSetDeviceId(OpaqueDataBuffer *dataBuffer, int deviceId) {
  if(dataBuffer == nullptr)
    THROW_EXCEPTION("dbSetDeviceId: dataBuffer is null");
  if (deviceId < 0) {
    // Sentinel: reset sync counters without changing the device id.
    auto db = dataBuffer->dataBuffer();
    if (db != nullptr) {
      db->resetCounters();
    }
    return;
  }
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

// Transfer Metrics API moved to NativeOpsHelpers_DataBuffers_metrics.cpp
