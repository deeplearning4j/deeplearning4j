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
#include <graph/cpu/FunctionalReplayHandle.h>

// ZLUDA builds define SD_CUDA but the actual GPU is AMD (HIP) or Intel (Level Zero).
// ZLUDA's CUDA graph translation is incomplete — use native APIs instead.
// Check ZLUDA targets BEFORE SD_CUDA so the native handle is selected.
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

void GraphReplayHandle::addCaptureBuffer(ReplayCaptureBuffer&& buf) {
  captureBuffers_.emplace_back(std::move(buf));
}

std::vector<ReplayCaptureBuffer>& GraphReplayHandle::getCaptureBuffers() {
  return captureBuffers_;
}

const std::vector<ReplayCaptureBuffer>& GraphReplayHandle::getCaptureBuffers() const {
  return captureBuffers_;
}

void GraphReplayHandle::snapshotExternalAddresses(NDArray** externalInputs, int numInputs) {
  capturedExternalAddrs_.resize(numInputs);
  for (int i = 0; i < numInputs; i++) {
    capturedExternalAddrs_[i] =
        (externalInputs[i] != nullptr) ? externalInputs[i]->specialBuffer() : nullptr;
  }
}

bool GraphReplayHandle::externalAddressesMatch(NDArray** externalInputs, int numInputs) const {
  if (capturedExternalAddrs_.empty()) return false;
  if (numInputs != static_cast<int>(capturedExternalAddrs_.size())) return false;
  for (int i = 0; i < numInputs; i++) {
    void* current = (externalInputs[i] != nullptr) ? externalInputs[i]->specialBuffer() : nullptr;
    if (current != capturedExternalAddrs_[i]) return false;
  }
  return true;
}

const std::vector<void*>& GraphReplayHandle::getCapturedExternalAddresses() const {
  return capturedExternalAddrs_;
}

void GraphReplayHandle::clearExternalAddresses() {
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
  // CPU base class: no GPU workspace needed
  return true;
}

void GraphReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  // CPU base class: nothing to release
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void GraphReplayHandle::freeHostPointers() {
  // CPU base class: host pointers are plain heap — just clear the list.
  // CUDA override uses cudaFreeHost.
  capturedHostPtrs_.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// GraphReplayFactory
// ═══════════════════════════════════════════════════════════════════════════════

std::unique_ptr<GraphReplayHandle> GraphReplayFactory::create(int deviceId) {
  // ── ZLUDA builds: bypass CUDA graph translation, use native APIs ──────
  // ZLUDA defines SD_CUDA but the GPU is actually AMD or Intel.
  // ZLUDA's CUDA graph API coverage is incomplete (missing cuGraphLaunch
  // in official v5, partial in lshqqytiger fork). Native HIP/L0 graph
  // APIs are production-ready, so use them directly.
#if defined(HAVE_ZLUDA) && defined(ZLUDA_TARGET_AMD)
  return std::make_unique<HipGraphReplayHandle>(deviceId);
#elif defined(HAVE_ZLUDA) && defined(ZLUDA_TARGET_INTEL)
  return std::make_unique<LevelZeroReplayHandle>(deviceId);

  // ── Native builds: use the platform's own graph replay API ────────────
#elif defined(SD_CUDA)
  return std::make_unique<CudaGraphReplayHandle>(deviceId);
#elif defined(SD_HIP)
  return std::make_unique<HipGraphReplayHandle>(deviceId);
#elif defined(SD_METAL)
  return std::make_unique<MetalReplayHandle>(deviceId);
#elif defined(HAVE_LEVELZERO)
  return std::make_unique<LevelZeroReplayHandle>(deviceId);
#elif defined(HAVE_VULKAN)
  return std::make_unique<VulkanReplayHandle>(deviceId);
#elif defined(SD_TPU)
  return std::make_unique<TpuReplayHandle>(deviceId);
#elif defined(HAVE_HEXAGON_MLIR)
  return std::make_unique<HexagonReplayHandle>(deviceId);
#else
  return std::make_unique<FunctionalReplayHandle>();
#endif
}

bool GraphReplayFactory::hasHardwareReplay() {
#if defined(SD_CUDA) || defined(SD_HIP) || defined(SD_METAL) || \
    defined(HAVE_LEVELZERO) || defined(HAVE_VULKAN) || \
    defined(SD_TPU) || defined(HAVE_HEXAGON_MLIR)
  return true;
#else
  return false;
#endif
}

}  // namespace graph
}  // namespace sd
