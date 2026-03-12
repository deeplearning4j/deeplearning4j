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

#include <graph/gpu/CaptureBufferRegistry.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <execution/LaunchContext.h>
#include <graph/DspDiagnostics.h>

#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

// Entry tracking a single pool allocation owned by a segment.
struct RegistryEntry {
  void* ptr;
  size_t bytes;
  int deviceId;
};

// Implementation hidden from header to avoid CUDA includes in .h
class CaptureBufferRegistryImpl {
 public:
  std::mutex mtx_;
  std::unordered_map<int, std::vector<RegistryEntry>> entries_;

  void* allocate(int segIdx, size_t bytes, int deviceId) {
    // Get current CUDA stream for pool allocation
    cudaStream_t stream = nullptr;
    auto* ctx = LaunchContext::defaultContext();
    if (ctx != nullptr && ctx->getCudaStream() != nullptr) {
      stream = *ctx->getCudaStream();
    }

    void* ptr = memory::CudaMemoryPool::getInstance().allocate(bytes, deviceId, stream);
    if (ptr != nullptr) {
      std::lock_guard<std::mutex> lock(mtx_);
      entries_[segIdx].push_back({ptr, bytes, deviceId});
      DSP_DIAG(MEMORY, "CaptureBufferRegistry: allocated %zuMB for seg %d on device %d (ptr=%p)",
               bytes / (1024 * 1024), segIdx, deviceId, ptr);
    } else {
      DSP_DIAG(MEMORY, "CaptureBufferRegistry: FAILED to allocate %zuMB for seg %d on device %d",
               bytes / (1024 * 1024), segIdx, deviceId);
    }
    return ptr;
  }

  void releaseSegment(int segIdx) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = entries_.find(segIdx);
    if (it == entries_.end()) return;

    cudaStream_t stream = nullptr;
    auto* ctx = LaunchContext::defaultContext();
    if (ctx != nullptr && ctx->getCudaStream() != nullptr) {
      stream = *ctx->getCudaStream();
    }

    size_t totalFreed = 0;
    for (auto& e : it->second) {
      memory::CudaMemoryPool::getInstance().free(e.ptr, e.deviceId, stream);
      totalFreed += e.bytes;
    }

    DSP_DIAG(MEMORY, "CaptureBufferRegistry: released %zuMB from seg %d (%d buffers)",
             totalFreed / (1024 * 1024), segIdx, static_cast<int>(it->second.size()));
    entries_.erase(it);
  }

  void releaseAll() {
    std::lock_guard<std::mutex> lock(mtx_);

    cudaStream_t stream = nullptr;
    auto* ctx = LaunchContext::defaultContext();
    if (ctx != nullptr && ctx->getCudaStream() != nullptr) {
      stream = *ctx->getCudaStream();
    }

    size_t totalFreed = 0;
    for (auto& [segIdx, vec] : entries_) {
      for (auto& e : vec) {
        memory::CudaMemoryPool::getInstance().free(e.ptr, e.deviceId, stream);
        totalFreed += e.bytes;
      }
    }
    DSP_DIAG(MEMORY, "CaptureBufferRegistry: released ALL %zuMB", totalFreed / (1024 * 1024));
    entries_.clear();
  }

  size_t segmentBytes(int segIdx) const {
    // Note: caller should hold lock or accept approximate result
    auto it = entries_.find(segIdx);
    if (it == entries_.end()) return 0;
    size_t total = 0;
    for (auto& e : it->second) total += e.bytes;
    return total;
  }

  size_t totalBytes() const {
    size_t total = 0;
    for (auto& [segIdx, vec] : entries_) {
      for (auto& e : vec) total += e.bytes;
    }
    return total;
  }
};

// ── Public API delegates to impl ────────────────────────────────────────────

CaptureBufferRegistry::CaptureBufferRegistry() : impl_(new CaptureBufferRegistryImpl()) {}

CaptureBufferRegistry::~CaptureBufferRegistry() {
  if (impl_) {
    impl_->releaseAll();
    delete impl_;
  }
}

void* CaptureBufferRegistry::allocate(int segIdx, size_t bytes, int deviceId) {
  return impl_->allocate(segIdx, bytes, deviceId);
}

void CaptureBufferRegistry::releaseSegment(int segIdx) {
  impl_->releaseSegment(segIdx);
}

void CaptureBufferRegistry::releaseAll() {
  impl_->releaseAll();
}

size_t CaptureBufferRegistry::segmentBytes(int segIdx) const {
  std::lock_guard<std::mutex> lock(impl_->mtx_);
  return impl_->segmentBytes(segIdx);
}

size_t CaptureBufferRegistry::totalBytes() const {
  std::lock_guard<std::mutex> lock(impl_->mtx_);
  return impl_->totalBytes();
}

}  // namespace graph
}  // namespace sd
