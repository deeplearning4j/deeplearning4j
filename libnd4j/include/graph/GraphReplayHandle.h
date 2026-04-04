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

#ifndef LIBND4J_GRAPH_REPLAY_HANDLE_H
#define LIBND4J_GRAPH_REPLAY_HANDLE_H

#include <array/NDArray.h>
#include <system/common.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * State of a graph replay handle.
 * Platform-agnostic equivalent of cuda::GraphState.
 */
enum class ReplayState {
  EMPTY = 0,     // No graph captured
  CAPTURING,     // Currently capturing
  CAPTURED,      // Capture complete, not yet finalized
  READY,         // Ready for replay (instantiated/finalized)
  ERRORED        // Error state
};

/**
 * Platform-agnostic statistics for a captured graph replay.
 */
struct ReplayStatistics {
  int numOperations = 0;
  int numMemoryOps = 0;
  size_t estimatedMemory = 0;
  double captureTimeMs = 0.0;
  double lastReplayTimeMs = 0.0;
  int replayCount = 0;
};

/**
 * Platform-agnostic capture buffer descriptor.
 *
 * Replaces the CUDA-only GraphSegment::CaptureBuffer struct.
 * CUDA graphs record exact GPU memory addresses during capture.
 * External inputs get recreated each decoder step with new GPU addresses.
 * Capture buffers provide fixed-address staging areas that the graph always
 * references, with data copied from real inputs before each replay.
 */
struct ReplayCaptureBuffer {
  NDArray* buffer = nullptr;            // Fixed-address buffer (owned unless directReference)
  int externalInputIndex = -1;          // Which external input this maps to (-1 = cross-segment)
  int crossSegmentSlotIdx = -1;         // Which output slot this maps to (for cross-segment inputs)
  size_t capturedSize = 0;              // Size in bytes at capture time
  const void* lastSourcePtr = nullptr;  // Last source address — skip copy if unchanged
  bool initialCopyDone = false;         // True after first successful copy
  bool neverSkipCopy = false;           // Always copy even if pointer unchanged (KV cache buffers)
  bool directReference = false;         // Buffer points to external input directly (NOT owned)
};

/**
 * Abstract interface for platform-agnostic graph replay.
 *
 * Abstracts the warmup -> capture -> replay lifecycle into a portable
 * interface that can be implemented for different hardware backends:
 *   - CUDA: wraps CudaGraphHandle (cudaGraph_t / cudaGraphExec_t)
 *   - CPU: FunctionalReplayHandle (cached op dispatch, skip shape inference)
 *   - Future: Metal command buffer, Vulkan command buffer, ROCm HIP graph
 *
 * The handle owns capture buffers, address snapshots, and pinned host
 * pointers that must persist for the graph's lifetime.
 */
class SD_LIB_EXPORT GraphReplayHandle {
 public:
  virtual ~GraphReplayHandle() = default;

  // ── Capture lifecycle ─────────────────────────────────────────────────

  /**
   * Begin capturing operations into the replay handle.
   * @param stream Platform stream (cudaStream_t* for CUDA, nullptr for CPU)
   * @return true if capture started successfully
   */
  virtual bool beginCapture(void* stream) = 0;

  /**
   * End capturing and store the captured graph.
   * @param stream Platform stream used for capture
   * @return true if capture ended successfully
   */
  virtual bool endCapture(void* stream) = 0;

  /**
   * Finalize the captured graph for replay (e.g., instantiate CUDA graph exec).
   * @return true if finalization succeeded
   */
  virtual bool finalize() = 0;

  /**
   * Replay the captured graph.
   * @param stream Platform stream for execution
   * @return true if replay launched successfully
   */
  virtual bool replay(void* stream) = 0;

  // ── State queries ─────────────────────────────────────────────────────

  /** Get the current state of the replay handle. */
  virtual ReplayState getState() const = 0;

  /** Convenience: true when the handle is ready for replay. */
  bool isReady() const { return getState() == ReplayState::READY; }

  // ── Diagnostics ───────────────────────────────────────────────────────

  /** Get replay statistics. */
  virtual ReplayStatistics getStatistics() const = 0;

  /** Get the backend name (e.g., "CUDA", "CPU", "Metal"). */
  virtual const char* backendName() const = 0;

  // ── Capture buffers ───────────────────────────────────────────────────
  // Base implementation stores capture buffers. Subclasses may override
  // for platform-specific buffer management.

  virtual void addCaptureBuffer(ReplayCaptureBuffer&& buf);
  virtual std::vector<ReplayCaptureBuffer>& getCaptureBuffers();
  virtual const std::vector<ReplayCaptureBuffer>& getCaptureBuffers() const;

  // ── Address snapshot for graph invalidation ───────────────────────────
  // Captures external input device buffer addresses at graph capture time.
  // On replay, mismatch indicates the graph has stale addresses baked in.

  virtual void snapshotExternalAddresses(NDArray** externalInputs, int numInputs);
  virtual bool externalAddressesMatch(NDArray** externalInputs, int numInputs) const;
  virtual const std::vector<void*>& getCapturedExternalAddresses() const;
  virtual void clearExternalAddresses();

  // ── Pinned host pointer lifetime management ───────────────────────────
  // Host pointers allocated during capture that must persist for graph
  // lifetime (graph replay reads from recorded host addresses).

  virtual void addCapturedHostPtr(void* ptr);
  virtual std::vector<void*>& getCapturedHostPtrs();
  virtual const std::vector<void*>& getCapturedHostPtrs() const;

  // ── Capture workspace management ─────────────────────────────────────
  // Pre-allocated buffer for PointersManager temporaries during capture.
  // Uses CUDA memory pool (CaptureBufferRegistry) when available,
  // falls back to raw cudaMalloc on CUDA or no-op on CPU.

  /**
   * Allocate capture workspace. On CUDA, routes through CaptureBufferRegistry
   * (pool) when available, falls back to raw cudaMalloc.
   *
   * @param bytes     Requested workspace size
   * @param deviceId  Target GPU device (ignored on CPU)
   * @param registryPtr  Opaque pointer to CaptureBufferRegistry (nullptr = raw alloc)
   * @param segIdx    Segment index for registry ownership tracking
   * @return true if allocation succeeded (or was already allocated)
   */
  virtual bool allocateWorkspace(size_t bytes, int deviceId = 0,
                                 void* registryPtr = nullptr, int segIdx = 0);

  /**
   * Release capture workspace. Handles pool-based or raw deallocation.
   * Called automatically by destructor. Safe to call multiple times.
   *
   * @param registryPtr  Opaque pointer to CaptureBufferRegistry (nullptr = raw free)
   * @param segIdx    Segment index for registry release
   */
  virtual void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0);

  /**
   * Free all captured host pointers (pinned memory from graph capture).
   * Called automatically by destructor. Safe to call multiple times.
   */
  virtual void freeHostPointers();

  /** Get capture workspace pointer (read-only). */
  void* getWorkspacePtr() const { return captureWorkspacePtr_; }

  /** Get capture workspace size in bytes. */
  size_t getWorkspaceBytes() const { return captureWorkspaceBytes_; }

  /**
   * Use an externally-owned workspace instead of allocating per-handle.
   * The caller retains ownership — this handle will NOT free the pointer.
   * All segments sharing a workspace must capture with the same pointer
   * address (guaranteed since we allocate once and reuse).
   */
  void useExternalWorkspace(void* ptr, size_t bytes) {
    captureWorkspacePtr_ = ptr;
    captureWorkspaceBytes_ = bytes;
    workspaceIsExternal_ = true;
  }

  /** Returns true if this handle's workspace is externally owned. */
  bool isWorkspaceExternal() const { return workspaceIsExternal_; }

 protected:
  void* captureWorkspacePtr_ = nullptr;
  size_t captureWorkspaceBytes_ = 0;
  bool workspaceIsExternal_ = false;
  std::vector<ReplayCaptureBuffer> captureBuffers_;
  std::vector<void*> capturedExternalAddrs_;
  std::vector<void*> capturedHostPtrs_;
};

/**
 * Factory for creating platform-appropriate GraphReplayHandle instances.
 */
class SD_LIB_EXPORT GraphReplayFactory {
 public:
  /**
   * Create a replay handle for the current platform.
   * CUDA builds: CudaGraphReplayHandle
   * CPU builds: FunctionalReplayHandle
   *
   * @param deviceId GPU device ID (ignored on CPU)
   * @return Owning pointer to the new handle
   */
  static std::unique_ptr<GraphReplayHandle> create(int deviceId = 0);

  /**
   * Returns true if the current platform supports hardware command replay
   * (CUDA graphs, Metal command buffers, etc.) as opposed to software-only
   * functional replay.
   */
  static bool hasHardwareReplay();
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPH_REPLAY_HANDLE_H
