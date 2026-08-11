/* ******************************************************************************
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

#pragma once

//
// Device dispatch boundary for DSP/graph .cpp files.
// CUDA implementations live in DspCudaDispatch.cu; Vulkan implementations live
// in execution/vulkan/VulkanDspDispatch.cpp. Host-only builds use inline stubs.
//
// Portable buffer accessor: use dspBuffer(arr) instead of backend conditionals.
//

#include <system/common.h>
#include <array/NDArray.h>
#include <execution/LaunchContext.h>
#include <graph/GraphBackend.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace sd {
namespace graph {

#if defined(SD_CUDA)
namespace cuda {
#elif defined(SD_VULKAN)
namespace vulkan {
#endif

#if defined(SD_CUDA) || defined(SD_VULKAN)

// ═══════════════════════════════════════════════════════════════════════════════
// Portable buffer accessor — replaces the DSP_BUF macro
// ═══════════════════════════════════════════════════════════════════════════════

void* dspBuffer(NDArray* arr);
const void* dspBufferConst(const NDArray* arr);

// Null-safe variant
static inline void* dspBufferSafe(NDArray* arr) {
  return arr != nullptr ? dspBuffer(arr) : nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error management
// ═══════════════════════════════════════════════════════════════════════════════

/** Clear the backend dispatch error. Compatibility spelling retained for callers. */
int dspClearLastCudaError();

/** Peek at the backend dispatch error without clearing. Zero means success. */
int dspPeekLastCudaError();

/** Get the backend-specific text for a dispatch error code. */
const char* dspCudaErrorString(int errorCode);

// ═══════════════════════════════════════════════════════════════════════════════
// Device queries
// ═══════════════════════════════════════════════════════════════════════════════

/** Get the current backend device ID. Returns -1 on failure. */
int dspGetCurrentDevice();

/** Set the current backend device. */
void dspSetCurrentDevice(int deviceId);

/** Get device count. Returns 0 on failure. */
int dspGetDeviceCount();

// ═══════════════════════════════════════════════════════════════════════════════
// Stream capture queries
// ═══════════════════════════════════════════════════════════════════════════════

/** Check whether a backend-native stream is currently in capture mode. */
bool dspStreamIsCapturing(void* stream);

/**
 * End a stale capture on a stream and destroy the resulting graph.
 * Used for cleanup when a capture was poisoned by an error.
 * @param stream  backend-native stream value encoded as void*
 * @param label   diagnostic label for logging
 * @return true if a stale capture was ended
 */
bool dspEndStaleCapture(void* stream, const char* label);

// ═══════════════════════════════════════════════════════════════════════════════
// Memory operations
// ═══════════════════════════════════════════════════════════════════════════════

/** Release an allocation through the active device backend. */
void dspDeviceFree(void* ptr);

/** Enqueue host-to-device copy on a backend-native stream. */
int dspMemcpyH2DAsync(void* dst, const void* src, size_t bytes, void* stream);

/** Enqueue device-to-device copy on the default backend stream. */
void dspMemcpyD2DDefaultStream(void* dst, const void* src, size_t bytes);

// ═══════════════════════════════════════════════════════════════════════════════
// Memory pool management
// ═══════════════════════════════════════════════════════════════════════════════

/** Trim the device memory pool to minBytes. Returns true on success. */
bool dspMemPoolTrim(int deviceId, size_t minBytes);

/** Query total memory for the current backend device, in bytes. */
size_t dspGetDeviceTotalMemory();

/** True when arrays have a separately managed device-memory domain. */
bool dspHasDeviceMemory();

/** Compatibility gate for code that genuinely requires CUDA APIs. */
bool dspIsCudaBuild();

/** True when execution uses host memory rather than a device backend. */
bool dspIsHostBuild();

/**
 * Release a plan-owned capture workspace through the active backend memory pool.
 * Called only after dependent DataBuffers have been destroyed.
 */
void dspFreeWorkspaceOnPool(void* ptr);

/**
 * Returns true if ptr is the process-lifetime global capture workspace.
 * Backends without a process-global workspace return false.
 */
bool dspIsGlobalCaptureWorkspace(void* ptr);

// ═══════════════════════════════════════════════════════════════════════════════
// Event management (for cross-stream sync in execute paths)
// ═══════════════════════════════════════════════════════════════════════════════

/** Create a backend synchronization event. */
void* dspCreateEvent();

/** Destroy a backend synchronization event. Handles nullptr safely. */
void dspDestroyEvent(void* event);

/** Record event on stream. Both are void*. */
void dspEventRecord(void* event, void* stream);

/** Make stream wait for event. Both are void*. */
void dspStreamWaitEvent(void* stream, void* event);

// ═══════════════════════════════════════════════════════════════════════════════
// Stream representation convention  (READ THIS before passing a `void* stream` around)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Execute/dispatch entry points receive a STREAM-POINTER representation, while DSP
// helpers and TLS store a canonical STREAM-VALUE. CUDA canonicalization dereferences
// cudaStream_t*; Vulkan validates and returns its opaque VulkanExecutionStream*. Never
// pass an entry-point representation directly to a stream-value consumer.

/** Canonicalize an execute/dispatch stream pointer for the active backend. */
void* dspStreamPtrToValue(void* streamPtr);

/**
 * Block until all work queued on a canonical backend stream value completes.
 * Returns false when the backend reports a synchronization failure.
 */
bool dspSynchronizeStream(void* stream);

// ═══════════════════════════════════════════════════════════════════════════════
// TLS stream access
// ═══════════════════════════════════════════════════════════════════════════════

/** Get tl_dspExecutionStream as void*. */
void* dspGetExecutionStream();

/** Set tl_dspExecutionStream from void*. */
void dspSetExecutionStream(void* stream);

/** Get tl_dspGapStream as void*. */
void* dspGetGapStream();

/** Get tl_graphCaptureStream as void*. */
void* dspGetGraphCaptureStream();

/** Set tl_graphCaptureStream. */
void dspSetGraphCaptureStream(void* stream);

/**
 * Get the LaunchContext default stream as void*.
 * On CPU builds returns nullptr.
 */
void* dspGetLcDefaultStream();

/**
 * Synchronize the active backend's default execution stream.
 * No-op on CPU builds.
 */
void dspSyncDefaultStream();

// ═══════════════════════════════════════════════════════════════════════════════
// Thread completion event
// ═══════════════════════════════════════════════════════════════════════════════

void dspPublishThreadCompletionEvent(void* streamPtr);

// ═══════════════════════════════════════════════════════════════════════════════
// DSP replay active flag (tl_dspReplayActive TLS accessor)
// ═══════════════════════════════════════════════════════════════════════════════

/** Get tl_dspReplayActive thread-local flag. Returns false on CPU. */
bool dspGetReplayActive();

/** Set tl_dspReplayActive thread-local flag. No-op on CPU. */
void dspSetReplayActive(bool active);

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA compiler-backend compatibility accessors (nullptr on other backends)
// ═══════════════════════════════════════════════════════════════════════════════

/** Return NvrtcGraphBackend when active, otherwise nullptr. */
GraphBackend* dspGetNvrtcBackend();

/** Return PtxGraphBackend when active, otherwise nullptr. */
GraphBackend* dspGetPtxBackend();

/**
 * Copy cuBLAS handle from src LaunchContext to dst LaunchContext.
 * No-op on CPU builds.
 */
void dspCopyCublasHandle(LaunchContext* dst, LaunchContext* src);

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA Triton compiler-backend compatibility accessors
// ═══════════════════════════════════════════════════════════════════════════════

/** Clear the CUDA Triton backend cache when that compiler backend is active. */
void dspTritonClearFailedCache();

/** Return the compiled TritonGraphBackend, or nullptr when Triton is not built. */
GraphBackend* dspGetTritonBackend();

#if defined(SD_CUDA)
}  // namespace cuda
using cuda::dspBuffer;
using cuda::dspBufferConst;
using cuda::dspBufferSafe;
using cuda::dspClearLastCudaError;
using cuda::dspPeekLastCudaError;
using cuda::dspCudaErrorString;
using cuda::dspGetCurrentDevice;
using cuda::dspSetCurrentDevice;
using cuda::dspGetDeviceCount;
using cuda::dspStreamIsCapturing;
using cuda::dspEndStaleCapture;
using cuda::dspDeviceFree;
using cuda::dspMemcpyH2DAsync;
using cuda::dspMemcpyD2DDefaultStream;
using cuda::dspMemPoolTrim;
using cuda::dspGetDeviceTotalMemory;
using cuda::dspHasDeviceMemory;
using cuda::dspIsCudaBuild;
using cuda::dspIsHostBuild;
using cuda::dspFreeWorkspaceOnPool;
using cuda::dspIsGlobalCaptureWorkspace;
using cuda::dspCreateEvent;
using cuda::dspDestroyEvent;
using cuda::dspEventRecord;
using cuda::dspStreamWaitEvent;
using cuda::dspStreamPtrToValue;
using cuda::dspSynchronizeStream;
using cuda::dspGetExecutionStream;
using cuda::dspSetExecutionStream;
using cuda::dspGetGapStream;
using cuda::dspGetGraphCaptureStream;
using cuda::dspSetGraphCaptureStream;
using cuda::dspGetLcDefaultStream;
using cuda::dspSyncDefaultStream;
using cuda::dspPublishThreadCompletionEvent;
using cuda::dspGetReplayActive;
using cuda::dspSetReplayActive;
using cuda::dspGetNvrtcBackend;
using cuda::dspGetPtxBackend;
using cuda::dspCopyCublasHandle;
using cuda::dspTritonClearFailedCache;
using cuda::dspGetTritonBackend;
#elif defined(SD_VULKAN)
}  // namespace vulkan
using vulkan::dspBuffer;
using vulkan::dspBufferConst;
using vulkan::dspBufferSafe;
using vulkan::dspClearLastCudaError;
using vulkan::dspPeekLastCudaError;
using vulkan::dspCudaErrorString;
using vulkan::dspGetCurrentDevice;
using vulkan::dspSetCurrentDevice;
using vulkan::dspGetDeviceCount;
using vulkan::dspStreamIsCapturing;
using vulkan::dspEndStaleCapture;
using vulkan::dspDeviceFree;
using vulkan::dspMemcpyH2DAsync;
using vulkan::dspMemcpyD2DDefaultStream;
using vulkan::dspMemPoolTrim;
using vulkan::dspGetDeviceTotalMemory;
using vulkan::dspHasDeviceMemory;
using vulkan::dspIsCudaBuild;
using vulkan::dspIsHostBuild;
using vulkan::dspFreeWorkspaceOnPool;
using vulkan::dspIsGlobalCaptureWorkspace;
using vulkan::dspCreateEvent;
using vulkan::dspDestroyEvent;
using vulkan::dspEventRecord;
using vulkan::dspStreamWaitEvent;
using vulkan::dspStreamPtrToValue;
using vulkan::dspSynchronizeStream;
using vulkan::dspGetExecutionStream;
using vulkan::dspSetExecutionStream;
using vulkan::dspGetGapStream;
using vulkan::dspGetGraphCaptureStream;
using vulkan::dspSetGraphCaptureStream;
using vulkan::dspGetLcDefaultStream;
using vulkan::dspSyncDefaultStream;
using vulkan::dspPublishThreadCompletionEvent;
using vulkan::dspGetReplayActive;
using vulkan::dspSetReplayActive;
using vulkan::dspGetNvrtcBackend;
using vulkan::dspGetPtxBackend;
using vulkan::dspCopyCublasHandle;
using vulkan::dspTritonClearFailedCache;
using vulkan::dspGetTritonBackend;
#endif

#else  // ═══════════════ Host-only dispatch ═══════════════════════════════════

static inline void* dspBuffer(NDArray* arr) { return arr->buffer(); }
static inline const void* dspBufferConst(const NDArray* arr) {
  return const_cast<NDArray*>(arr)->buffer();
}
static inline void* dspBufferSafe(NDArray* arr) {
  return arr != nullptr ? arr->buffer() : nullptr;
}

static inline int  dspClearLastCudaError() { return 0; }
static inline int  dspPeekLastCudaError() { return 0; }
static inline const char* dspCudaErrorString(int) { return "n/a (CPU)"; }

static inline int  dspGetCurrentDevice() { return -1; }
static inline void dspSetCurrentDevice(int) {}
static inline int  dspGetDeviceCount() { return 0; }

static inline bool dspStreamIsCapturing(void*) { return false; }
static inline bool dspEndStaleCapture(void*, const char*) { return false; }

static inline void dspDeviceFree(void*) {}
static inline int  dspMemcpyH2DAsync(void*, const void*, size_t, void*) { return 0; }
static inline void dspMemcpyD2DDefaultStream(void*, const void*, size_t) {}

static inline bool dspMemPoolTrim(int, size_t) { return false; }
static inline size_t dspGetDeviceTotalMemory() { return 0; }
static inline bool dspHasDeviceMemory() { return false; }
static inline bool dspIsCudaBuild() { return false; }
static inline bool dspIsHostBuild() { return true; }
static inline void dspFreeWorkspaceOnPool(void*) {}
static inline bool dspIsGlobalCaptureWorkspace(void*) { return false; }

static inline void* dspCreateEvent() { return nullptr; }
static inline void  dspDestroyEvent(void*) {}
static inline void  dspEventRecord(void*, void*) {}
static inline void  dspStreamWaitEvent(void*, void*) {}

static inline void* dspStreamPtrToValue(void* streamPtr) { return streamPtr; }
static inline bool  dspSynchronizeStream(void*) { return true; }
static inline void* dspGetExecutionStream() { return nullptr; }
static inline void  dspSetExecutionStream(void*) {}
static inline void* dspGetGapStream() { return nullptr; }
static inline void* dspGetGraphCaptureStream() { return nullptr; }
static inline void  dspSetGraphCaptureStream(void*) {}
static inline void* dspGetLcDefaultStream() { return nullptr; }

static inline void  dspPublishThreadCompletionEvent(void*) {}
static inline void  dspSyncDefaultStream() {}

static inline bool dspGetReplayActive() { return false; }
static inline void dspSetReplayActive(bool) {}

static inline GraphBackend* dspGetNvrtcBackend() { return nullptr; }
static inline GraphBackend* dspGetPtxBackend() { return nullptr; }
static inline void dspCopyCublasHandle(LaunchContext*, LaunchContext*) {}

static inline void dspTritonClearFailedCache() {}
static inline GraphBackend* dspGetTritonBackend() { return nullptr; }

#endif

}  // namespace graph
}  // namespace sd
