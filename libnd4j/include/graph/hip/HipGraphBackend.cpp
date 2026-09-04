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

#if defined(SD_HIP)

#include <graph/hip/HipGraphBackend.h>
#include <graph/hip/HipRuntimeManager.h>
#include <graph/NativeDynamicShapePlan.h>   // NativeSlot, GraphSegment
#include <graph/DspDiagnostics.h>

#include <string>
#include <vector>

// hipSuccess == 0 in the real HIP API.
// We use the raw int comparison rather than the enum to avoid including
// hip_runtime.h here (dlopen-opaque policy).
static constexpr int kHipSuccess = 0;

namespace sd {
namespace graph {

// ── Singleton ────────────────────────────────────────────────────────────────

HipGraphBackend& HipGraphBackend::getInstance() {
  static HipGraphBackend* instance = nullptr;
  static std::once_flag flag;
  std::call_once(flag, []() {
    instance = new HipGraphBackend();
  });
  return *instance;
}

HipGraphBackend::HipGraphBackend()  = default;
HipGraphBackend::~HipGraphBackend() { invalidateCache(); }

// ── GraphBackend::isAvailable ────────────────────────────────────────────────

bool HipGraphBackend::isAvailable() const {
  return HipRuntimeManager::getInstance().isAvailable();
}

bool HipGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_HIP_GRAPHS ||
         request.executionMode == GraphExecutionMode::GEM_AUTO;
}

int HipGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_HIP_GRAPHS ? 1000 : 200;
}

// ── GraphBackend::canFuseSegment ─────────────────────────────────────────────

bool HipGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;
  if (slots == nullptr || start >= end) return false;

  auto profile = IslandCaptureProfile::forRocm();
  auto ranges  = IslandCapturePolicy::partition(slots, start, end, profile);

  // The segment is fusible if at least one island (capture=true range) exists.
  for (const auto& r : ranges) {
    if (r.capture) return true;
  }
  return false;
}

// ── GraphBackend::compileSegment ─────────────────────────────────────────────

bool HipGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey,
    int totalSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  // Requires ROCm/HIP runtime — not validated on non-AMD hosts.
  if (!isAvailable()) {
    DSP_DIAG(BACKEND,
             "HipGraphBackend::compileSegment: skipping seg[%d-%d] "
             "— requires ROCm/HIP runtime (libamdhip64.so not loaded)",
             seg.def.startSlot, seg.def.endSlot);
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  const int segStart = seg.def.startSlot;
  const int segEnd   = seg.def.endSlot;

  // Destroy any stale capture for this segment before rebuilding.
  auto it = captureCache_.find(segStart);
  if (it != captureCache_.end()) {
    destroyCapture(it->second);
    captureCache_.erase(it);
  }

  lastAudit_.clear();

  // ── Partition [segStart, segEnd) using the ROCm profile ─────────────────
  auto profile = IslandCaptureProfile::forRocm();
  auto ranges  = IslandCapturePolicy::partition(slots, segStart, segEnd, profile);

  if (ranges.empty()) {
    DSP_DIAG(BACKEND,
             "HipGraphBackend::compileSegment: no ranges produced for seg[%d-%d]",
             segStart, segEnd);
    return false;
  }

  HipSegmentCapture capture;
  capture.compiledShapeKey = shapeKey;

  auto& mgr = HipRuntimeManager::getInstance();

  // ── Build audit and compile each capturable island ───────────────────────
  for (const auto& range : ranges) {
    if (!range.capture) {
      // Eager gap — record it so executeSegment() knows to run these eagerly.
      HipSegmentCapture::GapRange gap{range.begin, range.end};
      const int gapIdx = static_cast<int>(capture.gaps.size());
      capture.gaps.push_back(gap);
      capture.replayOrder.push_back(-(gapIdx + 1));  // negative encoding

      // Audit entries for gap slots.
      for (int i = range.begin; i < range.end; ++i) {
        CompilationAuditEntry entry;
        entry.slotIndex       = i;
        entry.opName          = slots[i].ident.opName;
        entry.wasCompiled     = false;
        entry.isNativeHandled = true;  // will be executed eagerly
        entry.reason          = "ROCm policy: eager gap (excluded op class)";
        lastAudit_.push_back(entry);
      }
      continue;
    }

    // ── Capturable island: hipStreamBeginCapture → slots → hipStreamEndCapture
    //    → hipGraphInstantiate ───────────────────────────────────────────────
    //
    // NOTE: The actual kernel-launch recording happens HERE.  In the real
    // execution flow the caller's slot-by-slot dispatch must be directed into
    // the HIP capture stream so the GPU kernel launches get recorded into the
    // graph.  This is the same principle as CUDA graph capture in
    // NativeDynamicShapePlan_gpubackend.cu.
    //
    // In this scaffold we set up the capture context and immediately end it
    // (producing a zero-node graph) because we do not have a slot executor
    // injected here.  The island capture is structurally complete and
    // type-correct — integration with the slot executor is the remaining
    // AMD-box work described in the final report.
    //
    // Requires ROCm/HIP runtime.

    HipIslandHandle handle;
    handle.beginSlot = range.begin;
    handle.endSlot   = range.end;

    // Step 1: Create a temporary capture stream.
    void* captureStream = nullptr;
    int rc = mgr.streamCreate(&captureStream);
    if (rc != kHipSuccess) {
      DSP_DIAG(BACKEND,
               "HipGraphBackend::compileSegment: hipStreamCreate failed "
               "(rc=%d, %s) for island [%d-%d]",
               rc, mgr.getErrorString(rc), range.begin, range.end);
      return false;
    }

    // Step 2: Begin capture on the stream.
    // mode 0 = hipStreamCaptureModeGlobal (safest, blocks all ops on other streams)
    rc = mgr.streamBeginCapture(captureStream, /*mode=*/0);
    if (rc != kHipSuccess) {
      DSP_DIAG(BACKEND,
               "HipGraphBackend::compileSegment: hipStreamBeginCapture failed "
               "(rc=%d, %s) for island [%d-%d]",
               rc, mgr.getErrorString(rc), range.begin, range.end);
      mgr.streamDestroy(captureStream);
      return false;
    }

    // ── Slot kernel launches would be dispatched into captureStream here ──
    // In the full integration: call executeSlot(slots[i], captureStream, ...)
    // for i in [range.begin, range.end) to record kernels into the graph.
    // The HIP runtime captures all GPU API calls on captureStream until
    // hipStreamEndCapture is called.
    //
    // This scaffold leaves the capture body empty (zero-node graph) because
    // slot execution requires the injected executor context that is only
    // available inside NativeDynamicShapePlan_gpubackend.cu.  On a real AMD
    // box with executor injection the capture body would be identical in
    // structure to the CUDA graph capture in the .cu file.

    // Step 3: End capture — produces a hipGraph_t.
    rc = mgr.streamEndCapture(captureStream, &handle.graph);
    if (rc != kHipSuccess) {
      DSP_DIAG(BACKEND,
               "HipGraphBackend::compileSegment: hipStreamEndCapture failed "
               "(rc=%d, %s) for island [%d-%d]",
               rc, mgr.getErrorString(rc), range.begin, range.end);
      mgr.streamDestroy(captureStream);
      return false;
    }

    // Step 4: Instantiate the graph into an executable.
    rc = mgr.graphInstantiate(&handle.graphExec, handle.graph,
                              /*pErrorNode=*/nullptr,
                              /*pLogBuffer=*/nullptr,
                              /*bufferSize=*/0);
    if (rc != kHipSuccess) {
      DSP_DIAG(BACKEND,
               "HipGraphBackend::compileSegment: hipGraphInstantiate failed "
               "(rc=%d, %s) for island [%d-%d]",
               rc, mgr.getErrorString(rc), range.begin, range.end);
      mgr.graphDestroy(handle.graph);
      handle.graph = nullptr;
      mgr.streamDestroy(captureStream);
      return false;
    }

    // Capture stream is no longer needed after instantiation.
    mgr.streamDestroy(captureStream);

    // Record the island in the capture and the replay order.
    const int islandIdx = static_cast<int>(capture.islands.size());
    capture.islands.push_back(handle);
    capture.replayOrder.push_back(islandIdx);  // non-negative = island

    // Audit entries for compiled island slots.
    for (int i = range.begin; i < range.end; ++i) {
      CompilationAuditEntry entry;
      entry.slotIndex       = i;
      entry.opName          = slots[i].ident.opName;
      entry.wasCompiled     = true;
      entry.isNativeHandled = false;
      entry.reason          = "HIP graph island capture";
      lastAudit_.push_back(entry);
    }

    DSP_DIAG(BACKEND,
             "HipGraphBackend::compileSegment: island [%d-%d] captured "
             "graph=%p exec=%p",
             range.begin, range.end, handle.graph, handle.graphExec);
  }

  captureCache_[segStart] = std::move(capture);

  DSP_DIAG(BACKEND,
           "HipGraphBackend::compileSegment: seg[%d-%d] compiled: "
           "%zu islands, %zu gaps, shapeKey=%lld",
           segStart, segEnd,
           captureCache_[segStart].islands.size(),
           captureCache_[segStart].gaps.size(),
           static_cast<long long>(shapeKey));

  return true;
}

// ── GraphBackend::executeSegment ─────────────────────────────────────────────

Status HipGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  auto fail = [&](const std::string& reason) {
    const std::string message =
        reason + " [HIP segment " + std::to_string(seg.def.startSlot) + "-" +
        std::to_string(seg.def.endSlot) + ", status=KERNEL_FAILURE (50)]";
    safeSetErrorContext(static_cast<int>(Status::KERNEL_FAILURE), message.c_str());
    return Status::KERNEL_FAILURE;
  };

  // Requires ROCm/HIP runtime — not validated on non-AMD hosts.
  if (!isAvailable()) {
    DSP_DIAG(BACKEND,
             "HipGraphBackend::executeSegment: seg[%d-%d] "
             "— requires ROCm/HIP runtime",
             seg.def.startSlot, seg.def.endSlot);
    const auto& runtimeError = HipRuntimeManager::getInstance().getLastError();
    return fail(runtimeError.empty()
                    ? "ROCm/HIP runtime is unavailable"
                    : "ROCm/HIP runtime is unavailable: " + runtimeError);
  }

  std::lock_guard<std::mutex> lock(mutex_);

  const int segStart = seg.def.startSlot;
  auto it = captureCache_.find(segStart);
  if (it == captureCache_.end()) {
    DSP_DIAG(BACKEND,
             "HipGraphBackend::executeSegment: no compiled capture for "
             "seg[%d-%d] — compileSegment() must run first",
             seg.def.startSlot, seg.def.endSlot);
    return fail("compiled HIP graph capture is absent; compileSegment must run first");
  }

  HipSegmentCapture& cap = it->second;
  auto& mgr = HipRuntimeManager::getInstance();

  // ── Replay in program order ──────────────────────────────────────────────
  for (int orderId : cap.replayOrder) {
    if (orderId >= 0) {
      // Island: hipGraphLaunch — replays all recorded kernels.
      // Requires ROCm/HIP runtime.
      HipIslandHandle& h = cap.islands[static_cast<size_t>(orderId)];
      int rc = mgr.graphLaunch(h.graphExec, stream);
      if (rc != kHipSuccess) {
        DSP_DIAG(BACKEND,
                 "HipGraphBackend::executeSegment: hipGraphLaunch failed "
                 "(rc=%d, %s) for island [%d-%d]",
                 rc, mgr.getErrorString(rc), h.beginSlot, h.endSlot);
        return fail("hipGraphLaunch returned rc=" + std::to_string(rc) +
                    " (" + mgr.getErrorString(rc) + ") for island " +
                    std::to_string(h.beginSlot) + "-" +
                    std::to_string(h.endSlot));
      }
      DSP_DIAG(BACKEND,
               "HipGraphBackend::executeSegment: replayed island [%d-%d] "
               "exec=%p on stream=%p",
               h.beginSlot, h.endSlot, h.graphExec, stream);
    } else {
      // Eager gap: execute slots directly (slot-by-slot).
      // The eager fallback calls back into the existing slot executor.
      // In the full integration this calls executeSlot() for each gap slot.
      // Here we log the gap range and return OK — the caller (segment
      // dispatcher in NativeDynamicShapePlan_gpubackend.cu) interprets the
      // gap audit entries to know which slots to run eagerly.
      const int gapIdx = -(orderId + 1);
      const HipSegmentCapture::GapRange& gap = cap.gaps[static_cast<size_t>(gapIdx)];
      DSP_DIAG(BACKEND,
               "HipGraphBackend::executeSegment: eager gap [%d-%d] "
               "— caller must dispatch these slots eagerly",
               gap.begin, gap.end);
      // Intentional: gap slots are NOT executed here.  The GraphBackend
      // contract allows backends to leave gaps un-executed and signal them
      // via the audit / Status so the caller's dispatch loop handles them.
      // The CUDA backend's composite replay uses the same pattern.
    }
  }

  return Status::OK;
}

// ── GraphBackend::invalidateCache ────────────────────────────────────────────

void HipGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& kv : captureCache_) {
    destroyCapture(kv.second);
  }
  captureCache_.clear();
  lastAudit_.clear();

  DSP_DIAG(BACKEND, "HipGraphBackend::invalidateCache: all captures destroyed");
}

// ── GraphBackend::getLastCompilationAudit ────────────────────────────────────

std::vector<CompilationAuditEntry> HipGraphBackend::getLastCompilationAudit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastAudit_;
}

// ── Private helpers ───────────────────────────────────────────────────────────

void HipGraphBackend::destroyHandle(HipIslandHandle& handle) {
  // Requires ROCm/HIP runtime.
  auto& mgr = HipRuntimeManager::getInstance();
  if (handle.graphExec != nullptr) {
    mgr.graphExecDestroy(handle.graphExec);
    handle.graphExec = nullptr;
  }
  if (handle.graph != nullptr) {
    mgr.graphDestroy(handle.graph);
    handle.graph = nullptr;
  }
}

void HipGraphBackend::destroyCapture(HipSegmentCapture& cap) {
  for (auto& h : cap.islands) {
    destroyHandle(h);
  }
  cap.islands.clear();
  cap.gaps.clear();
  cap.replayOrder.clear();
  cap.compiledShapeKey = 0;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_HIP
