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

#include <graph/GraphReplayHandle.h>
#include <graph/DspDiagnostics.h>
#include <graph/cpu/FunctionalReplayHandle.h>
#include <graph/gpu/DspCudaDispatch.h>

// ZLUDA: check native targets before SD_CUDA to select the correct replay handle.
#if defined(HAVE_ZLUDA) && defined(ZLUDA_TARGET_AMD)
#include <graph/hip/HipGraphReplayHandle.h>
#elif defined(HAVE_ZLUDA) && defined(ZLUDA_TARGET_INTEL)
#include <graph/levelzero/LevelZeroReplayHandle.h>
#elif defined(SD_CUDA)
#include <graph/cuda/CudaGraphReplayHandle.h>
#endif

#if defined(SD_HIP) && !defined(HAVE_ZLUDA)
#include <graph/hip/HipGraphReplayHandle.h>
#endif

#if defined(HAVE_LEVELZERO) && !defined(HAVE_ZLUDA)
#include <graph/levelzero/LevelZeroReplayHandle.h>
#endif

#if defined(HAVE_VULKAN)
#include <graph/vulkan/VulkanReplayHandle.h>
#endif

#ifdef SD_METAL
#include <graph/metal/MetalReplayHandle.h>
#endif

#ifdef SD_TPU
#include <graph/tpu/TpuReplayHandle.h>
#endif

#ifdef HAVE_HEXAGON_MLIR
#include <graph/hexagon/HexagonReplayHandle.h>
#endif

namespace sd {
namespace graph {

// ═══════════════════════════════════════════════════════════════════════════════
// GraphReplayHandle base class — default implementations
// ═══════════════════════════════════════════════════════════════════════════════

void GraphReplayHandle::snapshotExternalAddresses(NDArray** externalInputs, int numInputs) {
  capturedExternalAddrs_.resize(numInputs);
  int nullCount = 0;
  for (int i = 0; i < numInputs; i++) {
    capturedExternalAddrs_[i] =
        (externalInputs[i] != nullptr) ? externalInputs[i]->specialBuffer() : nullptr;
    if (capturedExternalAddrs_[i] == nullptr) nullCount++;
  }
  DSP_DIAG(EXECUTE, "GraphReplayHandle::snapshotExternalAddresses: %d inputs, %d non-null, %d null (backend=%s)",
           numInputs, numInputs - nullCount, nullCount, backendName());
}

bool GraphReplayHandle::externalAddressesMatch(NDArray** externalInputs, int numInputs) const {
  if (capturedExternalAddrs_.empty()) return false;
  if (numInputs != static_cast<int>(capturedExternalAddrs_.size())) {
    DSP_DIAG(EXECUTE, "externalAddressesMatch: size mismatch (captured=%d current=%d)",
             (int)capturedExternalAddrs_.size(), numInputs);
    return false;
  }
  int mismatches = 0;
  int detailLimit = sd::graph::DspDiagnostics::getInstance().diagDetailLimit();
  for (int i = 0; i < numInputs; i++) {
    void* current = (externalInputs[i] != nullptr) ? externalInputs[i]->specialBuffer() : nullptr;
    if (current != capturedExternalAddrs_[i]) {
      mismatches++;
      if (mismatches <= detailLimit) {
        DSP_DIAG(EXECUTE, "externalAddressesMatch: mismatch at ext[%d] captured=%p current=%p",
                 i, capturedExternalAddrs_[i], current);
      }
    }
  }
  if (mismatches > detailLimit) {
    DSP_DIAG(EXECUTE, "externalAddressesMatch: ... and %d more mismatches", mismatches - detailLimit);
  }
  return mismatches == 0;
}

const std::vector<void*>& GraphReplayHandle::getCapturedExternalAddresses() const {
  return capturedExternalAddrs_;
}

void GraphReplayHandle::clearExternalAddresses() {
  DSP_DIAG(GRAPH_REPLAY, "GraphReplayHandle::clearExternalAddresses: clearing %d entries (backend=%s)",
           (int)capturedExternalAddrs_.size(), backendName());
  capturedExternalAddrs_.clear();
}

void GraphReplayHandle::addCapturedHostPtr(void* ptr) {
  capturedHostPtrs_.push_back(ptr);
}

std::vector<void*>& GraphReplayHandle::getCapturedHostPtrs() {
  return capturedHostPtrs_;
}

const std::vector<void*>& GraphReplayHandle::getCapturedHostPtrs() const {
  return capturedHostPtrs_;
}

// ── Workspace management (base class defaults — no-ops for CPU) ─────────────

bool GraphReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                           void* registryPtr, int segIdx) {
  DSP_DIAG(MEMORY, "GraphReplayHandle::allocateWorkspace: %zuKB device=%d seg=%d (base class no-op)",
           bytes / 1024, deviceId, segIdx);
  // CPU base class: no GPU workspace needed
  return true;
}

void GraphReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  DSP_DIAG(MEMORY, "GraphReplayHandle::releaseWorkspace: seg=%d prevBytes=%zu (base class)",
           segIdx, captureWorkspaceBytes_);
  // CPU base class: nothing to release
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void GraphReplayHandle::freeHostPointers() {
  DSP_DIAG(MEMORY, "GraphReplayHandle::freeHostPointers: clearing %d host ptrs (base class)",
           (int)capturedHostPtrs_.size());
  // CPU base class: host pointers are plain heap — just clear the list.
  // CUDA override uses cudaFreeHost.
  capturedHostPtrs_.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// GraphReplayFactory
// ═══════════════════════════════════════════════════════════════════════════════

std::unique_ptr<GraphReplayHandle> GraphReplayFactory::create(int deviceId) {
  DSP_DIAG_DEV(BACKEND, deviceId,
               "GraphReplayFactory::create: deviceId=%d hwReplay=%d",
               deviceId, hasHardwareReplay() ? 1 : 0);
  // ZLUDA defines SD_CUDA but targets AMD (HIP) or Intel (Level Zero) GPUs.
  // ZLUDA's CUDA graph translation is incomplete — use native graph APIs.
#if defined(HAVE_ZLUDA) && defined(ZLUDA_TARGET_AMD)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating HipGraphReplayHandle (ZLUDA AMD)");
  return std::make_unique<HipGraphReplayHandle>(deviceId);
#elif defined(HAVE_ZLUDA) && defined(ZLUDA_TARGET_INTEL)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating LevelZeroReplayHandle (ZLUDA Intel)");
  return std::make_unique<LevelZeroReplayHandle>(deviceId);

  // ── Native builds: use the platform's own graph replay API ────────────
#elif defined(SD_CUDA)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating CudaGraphReplayHandle");
  return std::make_unique<CudaGraphReplayHandle>(deviceId);
#elif defined(SD_HIP)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating HipGraphReplayHandle");
  return std::make_unique<HipGraphReplayHandle>(deviceId);
#elif defined(SD_METAL)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating MetalReplayHandle");
  return std::make_unique<MetalReplayHandle>(deviceId);
#elif defined(HAVE_LEVELZERO)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating LevelZeroReplayHandle");
  return std::make_unique<LevelZeroReplayHandle>(deviceId);
#elif defined(HAVE_VULKAN)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating VulkanReplayHandle");
  return std::make_unique<VulkanReplayHandle>(deviceId);
#elif defined(SD_TPU)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating TpuReplayHandle");
  return std::make_unique<TpuReplayHandle>(deviceId);
#elif defined(HAVE_HEXAGON_MLIR)
  DSP_DIAG_DEV(GRAPH_REPLAY, deviceId, "GraphReplayFactory: creating HexagonReplayHandle");
  return std::make_unique<HexagonReplayHandle>(deviceId);
#else
  DSP_DIAG(GRAPH_REPLAY, "GraphReplayFactory: creating FunctionalReplayHandle (CPU fallback)");
  return std::make_unique<FunctionalReplayHandle>();
#endif
}

bool GraphReplayFactory::hasHardwareReplay() {
  if (sd::graph::dspIsCudaBuild()) return true;
#if defined(SD_HIP) || defined(SD_METAL) || \
    defined(HAVE_LEVELZERO) || defined(HAVE_VULKAN) || \
    defined(SD_TPU) || defined(HAVE_HEXAGON_MLIR)
  return true;
#else
  return false;
#endif
}

}  // namespace graph
}  // namespace sd
