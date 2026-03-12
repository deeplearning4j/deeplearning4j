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

#ifdef SD_CUDA
#include <graph/cuda/CudaGraphReplayHandle.h>
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
#ifdef SD_CUDA
  return std::make_unique<CudaGraphReplayHandle>(deviceId);
#else
  return std::make_unique<FunctionalReplayHandle>();
#endif
}

bool GraphReplayFactory::hasHardwareReplay() {
#ifdef SD_CUDA
  return true;
#else
  return false;
#endif
}

}  // namespace graph
}  // namespace sd
