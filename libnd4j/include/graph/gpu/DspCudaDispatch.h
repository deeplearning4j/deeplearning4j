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
// Unified dispatch header for ALL CUDA API calls in DSP/graph .cpp files.
// .cu implementations in DspCudaDispatch.cu.
// CPU builds get inline stubs — no #ifdef SD_CUDA needed at call sites.
//
// Portable buffer accessor: use dspBuffer(arr) instead of
//   #ifdef SD_CUDA / specialBuffer() / #else / buffer() / #endif
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

#ifdef SD_CUDA

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

/** Clear sticky CUDA error. Returns the error code that was cleared. */
int dspClearLastCudaError();

/** Peek at last CUDA error without clearing. Returns 0 (cudaSuccess) or error code. */
int dspPeekLastCudaError();

/** Get error string for a CUDA error code. */
const char* dspCudaErrorString(int errorCode);

// ═══════════════════════════════════════════════════════════════════════════════
// Device queries
// ═══════════════════════════════════════════════════════════════════════════════

/** Get current CUDA device ID. Returns -1 on failure. */
int dspGetCurrentDevice();

/** Set current CUDA device. */
void dspSetCurrentDevice(int deviceId);

/** Get device count. Returns 0 on failure. */
int dspGetDeviceCount();

// ═══════════════════════════════════════════════════════════════════════════════
// Stream capture queries
// ═══════════════════════════════════════════════════════════════════════════════

/** Check if a stream is currently in capture mode. stream is void* (cudaStream_t). */
bool dspStreamIsCapturing(void* stream);

/**
 * End a stale capture on a stream and destroy the resulting graph.
 * Used for cleanup when a capture was poisoned by an error.
 * @param stream  void* (cudaStream_t)
 * @param label   diagnostic label for logging
 * @return true if a stale capture was ended
 */
bool dspEndStaleCapture(void* stream, const char* label);

// ═══════════════════════════════════════════════════════════════════════════════
// Memory operations
// ═══════════════════════════════════════════════════════════════════════════════

/** cudaFree wrapper. */
void dspDeviceFree(void* ptr);

/** cudaMemcpyAsync H2D. stream is void* (cudaStream_t), nullptr = default stream. */
int dspMemcpyH2DAsync(void* dst, const void* src, size_t bytes, void* stream);

/** cudaMemcpyAsync D2D on the default context stream. No-op if either ptr is null. */
void dspMemcpyD2DDefaultStream(void* dst, const void* src, size_t bytes);

// ═══════════════════════════════════════════════════════════════════════════════
// Memory pool management
// ═══════════════════════════════════════════════════════════════════════════════

/** Trim the device memory pool to minBytes. Returns true on success. */
bool dspMemPoolTrim(int deviceId, size_t minBytes);

/** Query total device memory in bytes (GPU: cudaMemGetInfo totalMem, CPU: 0). */
size_t dspGetDeviceTotalMemory();

/** Returns true if running on a CUDA-capable build with devices present. */
bool dspIsCudaBuild();

/**
 * Unregister a capture workspace from CudaMemoryPool and free the device buffer.
 * On CPU this is a no-op. On CUDA, calls CudaMemoryPool::unregisterCaptureWorkspace
 * then cudaFree. Used by ~NativeDynamicShapePlan to free the plan-owned workspace
 * AFTER all DataBuffers are destroyed (so isInCaptureWorkspace guards stay valid).
 */
void dspFreeWorkspaceOnPool(void* ptr);

/**
 * Returns true if ptr is the process-lifetime global capture workspace.
 * On CPU always returns false (no global workspace exists).
 */
bool dspIsGlobalCaptureWorkspace(void* ptr);

// ═══════════════════════════════════════════════════════════════════════════════
// Event management (for cross-stream sync in execute paths)
// ═══════════════════════════════════════════════════════════════════════════════

/** Create a CUDA event (disable-timing). Returns void* (cudaEvent_t). */
void* dspCreateEvent();

/** Destroy a CUDA event. Handles nullptr safely. */
void dspDestroyEvent(void* event);

/** Record event on stream. Both are void*. */
void dspEventRecord(void* event, void* stream);

/** Make stream wait for event. Both are void*. */
void dspStreamWaitEvent(void* stream, void* event);

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
 * Synchronize the LaunchContext default CUDA stream.
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
// CUDA-only graph backend accessors (return GraphBackend*, nullptr on CPU)
// ═══════════════════════════════════════════════════════════════════════════════

/** Return NvrtcGraphBackend singleton if available, nullptr on CPU. */
GraphBackend* dspGetNvrtcBackend();

/** Return PtxGraphBackend singleton if available, nullptr on CPU. */
GraphBackend* dspGetPtxBackend();

/**
 * Copy cuBLAS handle from src LaunchContext to dst LaunchContext.
 * No-op on CPU builds.
 */
void dspCopyCublasHandle(LaunchContext* dst, LaunchContext* src);

// ═══════════════════════════════════════════════════════════════════════════════
// Triton GPU backend dispatch (TritonGraphBackend is CUDA-only)
// ═══════════════════════════════════════════════════════════════════════════════

/** Clear TritonGraphBackend's failed segment cache. No-op on CPU. */
void dspTritonClearFailedCache();

/** Get TritonGraphBackend singleton as GraphBackend*. Returns nullptr if unavailable. */
GraphBackend* dspTritonGetBackendIfAvailable();

#else  // ═══════════════ CPU stubs ════════════════════════════════════════════

static inline void* dspBuffer(NDArray* arr) { return arr->buffer(); }
static inline const void* dspBufferConst(const NDArray* arr) { return const_cast<NDArray*>(arr)->buffer(); }
static inline void* dspBufferSafe(NDArray* arr) { return arr != nullptr ? arr->buffer() : nullptr; }

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
static inline bool dspIsCudaBuild() { return false; }
static inline void dspFreeWorkspaceOnPool(void*) {}
static inline bool dspIsGlobalCaptureWorkspace(void*) { return false; }

static inline void* dspCreateEvent() { return nullptr; }
static inline void  dspDestroyEvent(void*) {}
static inline void  dspEventRecord(void*, void*) {}
static inline void  dspStreamWaitEvent(void*, void*) {}

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
static inline GraphBackend* dspTritonGetBackendIfAvailable() { return nullptr; }

#endif

}  // namespace graph
}  // namespace sd
