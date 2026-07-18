/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_NATIVE_OPS_H
#define LIBND4J_VULKAN_NATIVE_OPS_H

#if !defined(SD_VULKAN)
#error "legacy/vulkan/NativeOpsVulkan.h is only valid for SD_VULKAN"
#endif

#include <legacy/NativeOps.h>

/**
 * Vulkan-only execution, coherence, allocation, and shutdown introspection.
 *
 * These declarations live in the Vulkan source tree so JavaCPP never exposes
 * them through the CPU or CUDA presets. They use the same exported NativeOps
 * ABI as the established CPU and CUDA backends.
 */

/** Return the calling thread's command-pool handle for a device, or 0. */
SD_LIB_EXPORT sd::LongType vulkanGetThreadCommandPoolHandle(int deviceId);

/** Return the current timeline-semaphore value for a device, or 0. */
SD_LIB_EXPORT sd::LongType vulkanGetTimelineValue(int deviceId);

/** Return whether a device exposes a dedicated transfer queue. */
SD_LIB_EXPORT bool vulkanHasDedicatedTransferQueue(int deviceId);

/** Return whether the host-side contents are current. */
SD_LIB_EXPORT bool dbIsPrimaryActual(OpaqueDataBuffer* dataBuffer);

/** Return whether the device-side contents are current. */
SD_LIB_EXPORT bool dbIsSpecialActual(OpaqueDataBuffer* dataBuffer);

/** Allocate the Vulkan special buffer eagerly. */
SD_LIB_EXPORT void dbAllocateSpecial(OpaqueDataBuffer* dataBuffer);

/** Return the Vulkan memory-pool block id for a device pointer, or -1. */
SD_LIB_EXPORT int vulkanGetPoolBlockId(sd::Pointer ptr, int deviceId);

/** Return the VkMemoryPropertyFlags for a device allocation, or -1. */
SD_LIB_EXPORT int vulkanGetAllocationMemoryPropertyFlags(sd::Pointer pointer,
                                                         int deviceId);

/** Return the number of pending Vulkan retire-list entries for a device. */
SD_LIB_EXPORT int vulkanGetRetireListPendingCount(int deviceId);

/** Shut down Vulkan resources in dependency order. Safe to call repeatedly. */
SD_LIB_EXPORT void vulkanShutdown();

#endif  // LIBND4J_VULKAN_NATIVE_OPS_H
