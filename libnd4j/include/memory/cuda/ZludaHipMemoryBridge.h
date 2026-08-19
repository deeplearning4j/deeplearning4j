/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_ZLUDA_HIP_MEMORY_BRIDGE_H
#define LIBND4J_ZLUDA_HIP_MEMORY_BRIDGE_H

#include <cstddef>
#include <cstdint>

namespace sd {
namespace memory {
namespace zluda_hip {

// This interface is deliberately SDK-neutral. CUDA/ZLUDA translation units pass
// their opaque stream handles without including HIP headers. ZLUDA represents a
// CUstream with the underlying hipStream_t handle, so the bridge preserves the
// exact stream ordering used by DSP capture and replay.
enum class Status : int {
  SUCCESS = 0,
  OUT_OF_MEMORY = 1,
  NOT_SUPPORTED = 2,
  INVALID_ARGUMENT = 3,
  RUNTIME_ERROR = 4
};

const char* lastError();

bool memoryPoolsSupported(int deviceId);
Status configureDefaultPool(int deviceId, std::uint64_t releaseThreshold);
Status getDefaultPoolStats(int deviceId, std::size_t* usedBytes,
                           std::size_t* reservedBytes);
Status trimDefaultPool(int deviceId, std::size_t minBytesToHold);

}  // namespace zluda_hip
}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_ZLUDA_HIP_MEMORY_BRIDGE_H
