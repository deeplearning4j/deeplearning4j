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
#include <cstdint>
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

  // Vulkan-specific fields (empty on other backends)
  std::string deviceName;          // GPU device name (e.g., "Adreno 8 Gen 3")
  uint32_t apiVersion = 0;         // VK_MAKE_VERSION result (e.g., VK_API_VERSION_1_2)
  size_t memoryBudgetBytes = 0;    // Allocated workspace size in bytes
};

/** Replay implementations known to the portable graph replay layer. */
enum class ReplayBackend : uint8_t {
  NONE = 0,
  FUNCTIONAL,
  CUDA,
  HIP,
  LEVEL_ZERO,
  VULKAN,
  METAL,
  TPU,
  HEXAGON
};

/**
 * A handle can exist before the DSP plan has a recorder capable of populating it.
 * Keeping these capabilities separate prevents a compiled scaffold from being
 * selected as an executable replay backend.
 */
struct ReplayBackendCapability {
  bool handleAvailable = false;
  bool recorderAvailable = false;

  bool executable() const { return handleAvailable && recorderAvailable; }
};

/** Build-time replay capability matrix used by portable replay selection. */
struct ReplayCapabilityMatrix {
  ReplayBackendCapability functional;
  ReplayBackendCapability cuda;
  ReplayBackendCapability hip;
  ReplayBackendCapability levelZero;
  ReplayBackendCapability vulkan;
  ReplayBackendCapability metal;
  ReplayBackendCapability tpu;
  ReplayBackendCapability hexagon;

  ReplayBackendCapability capability(ReplayBackend backend) const {
    switch (backend) {
      case ReplayBackend::FUNCTIONAL: return functional;
      case ReplayBackend::CUDA: return cuda;
      case ReplayBackend::HIP: return hip;
      case ReplayBackend::LEVEL_ZERO: return levelZero;
      case ReplayBackend::VULKAN: return vulkan;
      case ReplayBackend::METAL: return metal;
      case ReplayBackend::TPU: return tpu;
      case ReplayBackend::HEXAGON: return hexagon;
      case ReplayBackend::NONE:
      default: return {};
    }
  }

  bool canCreate(ReplayBackend backend) const {
    return capability(backend).handleAvailable;
  }

  bool canExecute(ReplayBackend backend) const {
    return capability(backend).executable();
  }

  bool hasExecutableHardwareReplay() const {
    return cuda.executable() || hip.executable() || levelZero.executable() ||
           vulkan.executable() || metal.executable() || tpu.executable() ||
           hexagon.executable();
  }

  ReplayBackend preferredExecutable() const {
    if (cuda.executable()) return ReplayBackend::CUDA;
    if (hip.executable()) return ReplayBackend::HIP;
    if (levelZero.executable()) return ReplayBackend::LEVEL_ZERO;
    if (vulkan.executable()) return ReplayBackend::VULKAN;
    if (metal.executable()) return ReplayBackend::METAL;
    if (tpu.executable()) return ReplayBackend::TPU;
    if (hexagon.executable()) return ReplayBackend::HEXAGON;
    if (functional.executable()) return ReplayBackend::FUNCTIONAL;
    return ReplayBackend::NONE;
  }
};

/**
 * Abstract interface for platform-agnostic graph replay.
 *
 * Abstracts the warmup -> capture -> replay lifecycle into a portable
 * interface that can be implemented for different hardware backends:
 *   - CUDA: wraps CudaGraphHandle (cudaGraph_t / cudaGraphExec_t)
 *   - Vulkan: records and resubmits native compute command buffers
 *   - Functional: cached typed op dispatch with late-bound operands
 *   - Handle scaffolds: HIP, Level Zero, Metal, TPU, and Hexagon
 *
 * The handle owns address snapshots, pinned host pointers, and optional
 * capture workspace that must persist for the graph's lifetime.
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

  /**
   * Device that owns this replay handle, or -1 for host-only replay.
   * Teardown must use this identity rather than the caller's current device.
   */
  virtual int getDeviceId() const { return -1; }

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

  // ── Captured GPU module lifetime management ──────────────────────────
  // Modules whose kernels a captured graph references. Unloaded only at
  // handle death (CUDA backend); unloading earlier invalidates the baked
  // kernel nodes (CUDA error 98 'invalid device function' at replay).
  virtual void addCapturedModule(void* module) { capturedModules_.push_back(module); }
  virtual std::vector<void*>& getCapturedModules() { return capturedModules_; }

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
  std::vector<void*> capturedExternalAddrs_;
  std::vector<void*> capturedHostPtrs_;
  std::vector<void*> capturedModules_;
};

/**
 * Factory for creating platform-appropriate GraphReplayHandle instances.
 */
class SD_LIB_EXPORT GraphReplayFactory {
 public:
  /**
   * Create the highest-priority executable replay handle for this build. A
   * backend is eligible only when both its handle and plan recorder are present;
   * otherwise selection continues to functional replay.
   *
   * @param deviceId GPU device ID (ignored by host-only replay)
   * @return Owning pointer to the new handle, or nullptr when none is executable
   */
  static std::unique_ptr<GraphReplayHandle> create(int deviceId = 0);

  /**
   * Create a specific raw replay handle. Returns nullptr only when the handle
   * implementation is absent; callers that intend to execute it must first
   * require capabilities().canExecute(backend). The default create(deviceId)
   * performs that recorder-aware selection automatically.
   */
  static std::unique_ptr<GraphReplayHandle> create(
      ReplayBackend backend, int deviceId = 0);

  /** Return the build's handle/recorder capability matrix. */
  static ReplayCapabilityMatrix capabilities();

  /**
   * Create a software functional replay handle explicitly, independent of the
   * platform's hardware replay support. Used by EMULATED_REPLAY and CPU native
   * ranges, which record logical slot commands even in CUDA-enabled builds.
   *
   * Vulkan-only builds return nullptr because their segment recorder owns a
   * complete hardware command stream and does not compile the host slot path.
   */
  static std::unique_ptr<GraphReplayHandle> createFunctional();

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
