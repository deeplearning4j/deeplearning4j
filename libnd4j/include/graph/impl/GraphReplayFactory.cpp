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
#include <graph/DspDeviceDispatch.h>

// Include every compiled handle independently. Selection happens through the
// capability matrix below; macro ordering must never choose a handle that the
// active plan recorder cannot populate.
#if defined(SD_CUDA)
#include <graph/cuda/CudaGraphReplayHandle.h>
#endif

#if defined(SD_HIP) || defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN)
#include <graph/hip/HipGraphReplayHandle.h>
#endif

#if defined(HAVE_LEVELZERO)
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

ReplayCapabilityMatrix GraphReplayFactory::capabilities() {
  ReplayCapabilityMatrix matrix;

#if !defined(SD_VULKAN)
  matrix.functional = {true, true};
#endif

#if defined(SD_CUDA)
  matrix.cuda.handleAvailable = true;
  // The monolithic DSP recorder is CUDA-native. ZLUDA builds compile this
  // handle too, but their translated graph path is not yet a supported recorder.
#if !defined(HAVE_ZLUDA)
  matrix.cuda.recorderAvailable = true;
#endif
#endif

#if defined(SD_HIP) || defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN)
  matrix.hip.handleAvailable = true;
  // HipGraphReplayHandle is complete, but NativeDynamicShapePlan does not yet
  // inject slot execution into a HIP capture stream.
#endif

#if defined(HAVE_LEVELZERO)
  matrix.levelZero.handleAvailable = true;
  // LevelZeroReplayHandle requires explicit native kernel recording; the plan
  // does not yet expose Level Zero kernel handles to that recorder.
#endif

#if defined(HAVE_VULKAN)
  matrix.vulkan = {true, true};
#endif

#if defined(SD_METAL)
  matrix.metal.handleAvailable = true;
  // MetalReplayHandle requires explicit MTL pipeline/argument recording.
#endif

#if defined(SD_TPU)
  matrix.tpu.handleAvailable = true;
  // TPU execution is populated by TpuGraphBackend's HLO compiler path, not by
  // the portable replay recorder.
#endif

#if defined(HAVE_HEXAGON_MLIR)
  matrix.hexagon.handleAvailable = true;
  // Hexagon execution is populated by its compiler backend; no replay-only
  // NativeDynamicShapePlan recorder is wired for portable mode.
#endif

  return matrix;
}

std::unique_ptr<GraphReplayHandle> GraphReplayFactory::create(int deviceId) {
  auto matrix = capabilities();
  auto backend = matrix.preferredExecutable();
  DSP_DIAG_DEV(BACKEND, deviceId,
               "GraphReplayFactory::create: deviceId=%d backend=%d hwReplay=%d",
               deviceId, static_cast<int>(backend),
               matrix.hasExecutableHardwareReplay() ? 1 : 0);
  return create(backend, deviceId);
}

std::unique_ptr<GraphReplayHandle> GraphReplayFactory::create(
    ReplayBackend backend, int deviceId) {
  switch (backend) {
    case ReplayBackend::FUNCTIONAL:
      return createFunctional();

    case ReplayBackend::CUDA:
#if defined(SD_CUDA)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating CudaGraphReplayHandle");
      return std::make_unique<CudaGraphReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::HIP:
#if defined(SD_HIP) || defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating HipGraphReplayHandle");
      return std::make_unique<HipGraphReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::LEVEL_ZERO:
#if defined(HAVE_LEVELZERO)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating LevelZeroReplayHandle");
      return std::make_unique<LevelZeroReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::VULKAN:
#if defined(HAVE_VULKAN)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating VulkanReplayHandle");
      return std::make_unique<VulkanReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::METAL:
#if defined(SD_METAL)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating MetalReplayHandle");
      return std::make_unique<MetalReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::TPU:
#if defined(SD_TPU)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating TpuReplayHandle");
      return std::make_unique<TpuReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::HEXAGON:
#if defined(HAVE_HEXAGON_MLIR)
      DSP_DIAG_DEV(GRAPH_REPLAY, deviceId,
                   "GraphReplayFactory: creating HexagonReplayHandle");
      return std::make_unique<HexagonReplayHandle>(deviceId);
#else
      break;
#endif

    case ReplayBackend::NONE:
    default:
      break;
  }

  DSP_DIAG_DEV(FALLBACK, deviceId,
               "GraphReplayFactory: requested replay backend %d is unavailable",
               static_cast<int>(backend));
  return nullptr;
}

std::unique_ptr<GraphReplayHandle> GraphReplayFactory::createFunctional() {
#if defined(SD_VULKAN)
  DSP_DIAG(GRAPH_REPLAY,
           "GraphReplayFactory::createFunctional: unavailable in Vulkan-only builds");
  return nullptr;
#else
  DSP_DIAG(GRAPH_REPLAY,
           "GraphReplayFactory::createFunctional: creating FunctionalReplayHandle");
  return std::make_unique<FunctionalReplayHandle>();
#endif
}

bool GraphReplayFactory::hasHardwareReplay() {
  return capabilities().hasExecutableHardwareReplay();
}

}  // namespace graph
}  // namespace sd
