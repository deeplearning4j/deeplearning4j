/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "ZludaHipMemoryBridge.h"

// This is an isolated AMD-native translation unit in the CUDA/ZLUDA artifact.
// It intentionally includes no ND4J or CUDA headers. ZLUDA's CUstream ABI is a
// direct hipStream_t transmute, so the opaque stream is safe to forward here.
#if defined(__HIP_PLATFORM_NVIDIA__)
#error "ZludaHipMemoryBridge.cpp must use the AMD HIP ABI"
#endif
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime.h>

#include <sstream>
#include <string>

namespace sd {
namespace memory {
namespace zluda_hip {
namespace {

thread_local std::string gLastError;

Status hipStatus(const char* operation, hipError_t status) {
  if (status == hipSuccess) {
    gLastError.clear();
    return Status::SUCCESS;
  }

  std::ostringstream message;
  message << operation << ": ";
  const char* detail = hipGetErrorString(status);
  message << (detail != nullptr ? detail : "unknown HIP error")
          << " (" << static_cast<int>(status) << ")";
  gLastError = message.str();

  switch (status) {
    case hipErrorOutOfMemory:
      return Status::OUT_OF_MEMORY;
    case hipErrorNotSupported:
      return Status::NOT_SUPPORTED;
    case hipErrorInvalidValue:
      return Status::INVALID_ARGUMENT;
    default:
      return Status::RUNTIME_ERROR;
  }
}

Status defaultPool(int deviceId, hipMemPool_t* pool) {
  if (pool == nullptr) return Status::INVALID_ARGUMENT;
  *pool = nullptr;
  return hipStatus("hipDeviceGetDefaultMemPool",
                   hipDeviceGetDefaultMemPool(pool, deviceId));
}

}  // namespace

const char* lastError() { return gLastError.c_str(); }

bool memoryPoolsSupported(int deviceId) {
  int supported = 0;
  if (hipStatus("hipDeviceGetAttribute(memoryPoolsSupported)",
                hipDeviceGetAttribute(
                    &supported, hipDeviceAttributeMemoryPoolsSupported,
                    deviceId)) != Status::SUCCESS) {
    return false;
  }
  return supported != 0;
}

Status configureDefaultPool(int deviceId, std::uint64_t releaseThreshold) {
  hipMemPool_t pool = nullptr;
  Status status = defaultPool(deviceId, &pool);
  if (status != Status::SUCCESS) return status;
  return hipStatus("hipMemPoolSetAttribute(releaseThreshold)",
                   hipMemPoolSetAttribute(
                       pool, hipMemPoolAttrReleaseThreshold,
                       &releaseThreshold));
}

Status mallocAsync(void** ptr, std::size_t bytes, void* stream) {
  if (ptr == nullptr || bytes == 0) return Status::INVALID_ARGUMENT;
  *ptr = nullptr;
  return hipStatus("hipMallocAsync",
                   hipMallocAsync(ptr, bytes,
                                  reinterpret_cast<hipStream_t>(stream)));
}

Status freeAsync(void* ptr, void* stream) {
  if (ptr == nullptr) return Status::SUCCESS;
  return hipStatus("hipFreeAsync",
                   hipFreeAsync(ptr, reinterpret_cast<hipStream_t>(stream)));
}

Status getDefaultPoolStats(int deviceId, std::size_t* usedBytes,
                           std::size_t* reservedBytes) {
  if (usedBytes == nullptr || reservedBytes == nullptr) {
    return Status::INVALID_ARGUMENT;
  }
  *usedBytes = 0;
  *reservedBytes = 0;

  hipMemPool_t pool = nullptr;
  Status status = defaultPool(deviceId, &pool);
  if (status != Status::SUCCESS) return status;

  status = hipStatus("hipMemPoolGetAttribute(used)",
                     hipMemPoolGetAttribute(
                         pool, hipMemPoolAttrUsedMemCurrent, usedBytes));
  if (status != Status::SUCCESS) return status;
  return hipStatus("hipMemPoolGetAttribute(reserved)",
                   hipMemPoolGetAttribute(
                       pool, hipMemPoolAttrReservedMemCurrent,
                       reservedBytes));
}

Status trimDefaultPool(int deviceId, std::size_t minBytesToHold) {
  hipMemPool_t pool = nullptr;
  Status status = defaultPool(deviceId, &pool);
  if (status != Status::SUCCESS) return status;
  return hipStatus("hipMemPoolTrimTo",
                   hipMemPoolTrimTo(pool, minBytesToHold));
}

}  // namespace zluda_hip
}  // namespace memory
}  // namespace sd
