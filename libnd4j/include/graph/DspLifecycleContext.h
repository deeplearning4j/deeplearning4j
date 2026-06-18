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

//
// DspLifecycleContext — inline-only shim that lets the standard (slot-by-slot)
// execution path query the current DSP (DynamicShapePlan) lifecycle state
// WITHOUT depending on NativeDynamicShapePlan.h.
//
// The slot-by-slot path historically issued syncs, ticks, and closes with no
// awareness that a DSP capture or replay may be live on the same thread. That
// caused subtle bugs: ticks that poisoned captured graphs' actuality counters,
// stream-0 H2D syncs that broke CUDA graph capture, and eager close() calls on
// DataBuffers still referenced by a replay handle.
//
// This header centralises the gates so the call sites only need a single
// include. All accessors are inline and only consult state that is already
// defined elsewhere (DataBuffer.h thread-locals, DataBuffer::isFrozenPlanRegistered,
// a process-wide DspExecutionMode atomic stored here). No .cpp file needed.
//

#ifndef LIBND4J_DSP_LIFECYCLE_CONTEXT_H
#define LIBND4J_DSP_LIFECYCLE_CONTEXT_H

#include <array/DataBuffer.h>
#include <system/common.h>

#include <atomic>
#include <cstdint>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

namespace sd {
namespace graph {

/**
 * DspExecutionMode — process-wide policy controlling how the standard
 * slot-by-slot execution path coexists with an in-flight DSP plan.
 *
 * The three levels are ordered by strictness:
 *
 *   LEGACY_UNAWARE : slot-by-slot behaves exactly as before this shim existed.
 *                    All gates are no-ops. Use only for bisecting regressions.
 *
 *   COEXIST_SAFE   : slot-by-slot defers any state mutation that would race
 *                    with a live DSP capture/replay. This is the default —
 *                    it fixes the observed bugs without changing the fast
 *                    path when no DSP activity is present.
 *
 *   STRICT_ISOLATED: same as COEXIST_SAFE plus hard refusal to touch any
 *                    DSP-protected buffer while DSP owns the thread. Used
 *                    for tests that want a clean error instead of silent
 *                    corruption when slot-by-slot and DSP overlap.
 *
 * Mode is a SINGLE global atomic. Threads do not own independent modes — the
 * decision is a policy knob, not a thread-local state.
 */
enum class DspExecutionMode : int {
  LEGACY_UNAWARE  = 0,
  COEXIST_SAFE    = 1,
  STRICT_ISOLATED = 2,
};

/**
 * DspLifecycleContext — static-only query surface. All methods are inline and
 * safe to call from any translation unit that already pulls in DataBuffer.h.
 *
 * Callers:
 *   - NativeOps_customOp.cu  — guards the post-exec tick{Read,Write}Device loop.
 *   - NativeOps.cpp/.cu      — exposes the mode and gates as JNI entry points.
 *   - InferenceSession.java  — via those JNI entry points — gates data-dependent
 *                              resync and the post-exec orphan close().
 *   - DataBuffer::syncTo*    — consults isOwned() before inserting stream-0
 *                              syncs that would break capture.
 */
class DspLifecycleContext {
 public:
  // ---- mode ----------------------------------------------------------------

  /** Process-wide mode storage. */
  static SD_INLINE std::atomic<int>& modeStorage() {
    // Meyers singleton — thread-safe, zero-init cost, no global constructor.
    static std::atomic<int> mode{static_cast<int>(DspExecutionMode::COEXIST_SAFE)};
    return mode;
  }

  static SD_INLINE DspExecutionMode currentMode() {
    return static_cast<DspExecutionMode>(modeStorage().load(std::memory_order_acquire));
  }

  static SD_INLINE void setCurrentMode(DspExecutionMode mode) {
    modeStorage().store(static_cast<int>(mode), std::memory_order_release);
  }

  static SD_INLINE bool modeIsLegacy() {
    return currentMode() == DspExecutionMode::LEGACY_UNAWARE;
  }

  static SD_INLINE bool modeIsStrict() {
    return currentMode() == DspExecutionMode::STRICT_ISOLATED;
  }

  // ---- lifecycle queries ---------------------------------------------------

  /**
   * True while DSP is recording a CUDA graph (or other graph-backend capture
   * session). Mirrors tl_graphExecutionActive from DataBuffer.h — that flag is
   * set by NativeDynamicShapePlan around every graph segment entry. A host
   * write / stream-0 sync / free inside this window corrupts the captured
   * graph, so every gate defers in this state.
   */
  static SD_INLINE bool isCaptureActive() {
    return tl_graphExecutionActive;
  }

  /**
   * True while DSP is replaying a previously captured graph (or has otherwise
   * taken ownership of this thread's stream via tl_dspExecutionStream). Replay
   * does not set tl_graphExecutionActive, so a distinct check is needed.
   */
  static SD_INLINE bool isReplayActive() {
#ifdef SD_CUDA
    return tl_dspExecutionStream != nullptr;
#else
    return false;
#endif
  }

  /**
   * True when DSP owns the current thread's execution — either capture or
   * replay. Slot-by-slot mutations must defer while this is true.
   */
  static SD_INLINE bool isOwned() {
    return isCaptureActive() || isReplayActive();
  }

  // ---- stream selection ----------------------------------------------------

#ifdef SD_CUDA
  /**
   * Pick the stream DSP wants this work scheduled on, or fall back to the
   * caller's default if DSP has nothing to contribute. Using this instead of
   * raw `stream = 0` is what makes slot-by-slot ops ride the same stream as
   * surrounding DSP graph launches, so ordering is implicit.
   */
  static SD_INLINE cudaStream_t streamOrDefault(cudaStream_t fallback = nullptr) {
    if (tl_dspExecutionStream != nullptr) return reinterpret_cast<cudaStream_t>(tl_dspExecutionStream);
    if (tl_graphCaptureStream != nullptr) return reinterpret_cast<cudaStream_t>(tl_graphCaptureStream);
    return fallback;
  }
#endif

  // ---- buffer protection ---------------------------------------------------

  /**
   * True if `db` is registered in any frozen DSP plan, OR flagged constant.
   * These buffers must not be closed, reallocated, or sync-poisoned by the
   * slot-by-slot path — DSP baked their addresses into replay handles.
   */
  static SD_INLINE bool protectsBuffer(const DataBuffer* db) {
    if (db == nullptr) return false;
    if (db->isConstant) return true;
    return db->isFrozenPlanRegistered();
  }

  // ---- composed gates ------------------------------------------------------

  /**
   * Slot-by-slot close() sites consult this before freeing a DataBuffer it
   * allocated. Returns true iff the close must be deferred (DSP owns the
   * thread, OR the buffer itself is DSP-protected).
   */
  static SD_INLINE bool shouldDeferClose(const DataBuffer* db) {
    if (modeIsLegacy()) return false;
    if (protectsBuffer(db)) return true;
    return isOwned();
  }

  /**
   * NativeOps.execCustomOp2 consults this before calling tick{Read,Write}Device.
   * Ticking during DSP capture can mis-mark buffers as host-authoritative, so
   * the tick must be deferred to DSP's own actuality reconciler.
   */
  static SD_INLINE bool shouldSkipTick() {
    if (modeIsLegacy()) return false;
    return isOwned();
  }

  /**
   * InferenceSession.executeCustomOp consults this before issuing the
   * post-calculateOutputShape `dbSyncToSpecial` resync on data-dependent
   * inputs. When DSP is active and the buffer is protected, the resync is
   * redundant (DSP captured the H2D already) and actively harmful (it inserts
   * a stray stream-0 node into the captured graph).
   */
  static SD_INLINE bool shouldSkipResync(const DataBuffer* db) {
    if (modeIsLegacy()) return false;
    if (!isOwned()) return false;
    return protectsBuffer(db);
  }
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_DSP_LIFECYCLE_CONTEXT_H
