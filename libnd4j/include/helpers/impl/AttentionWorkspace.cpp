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
// @author Adam Gibson
//

#include <helpers/AttentionWorkspace.h>
#include <array/NDArrayFactory.h>
#include <graph/DspDiagnostics.h>
#include <algorithm>

namespace sd {

// RAII wrapper to ensure thread-local AttentionWorkspace is deleted on thread exit.
// Without this, the raw pointer leaks all cached GPU buffers when the thread terminates.
struct AttentionWorkspaceTLGuard {
  AttentionWorkspace* ptr = nullptr;
  ~AttentionWorkspaceTLGuard() {
    delete ptr;
    ptr = nullptr;
  }
};

// Thread-local instance accessor (function-local thread_local avoids MSVC C2492)
AttentionWorkspace*& AttentionWorkspace::instanceRef() {
  static thread_local AttentionWorkspaceTLGuard guard;
  return guard.ptr;
}

void*& AttentionWorkspace::activeScopeRef() {
  static thread_local void* scope = nullptr;
  return scope;
}

void* AttentionWorkspace::setActiveScope(void* scope) {
  void* previous = activeScopeRef();
  activeScopeRef() = scope;
  return previous;
}

void* AttentionWorkspace::getActiveScope() {
  return activeScopeRef();
}

AttentionWorkspace* AttentionWorkspace::getInstance() {
  auto& inst = instanceRef();
  if (inst == nullptr) {
    inst = new AttentionWorkspace();
  }
  return inst;
}

AttentionWorkspace::AttentionWorkspace() = default;

AttentionWorkspace::~AttentionWorkspace() {
  clear();
}

NDArray* AttentionWorkspace::getBuffer(const std::string& key,
                                        const std::vector<LongType>& shape,
                                        DataType dtype,
                                        LaunchContext* context) {
  std::lock_guard<std::mutex> lock(mutex_);
  void* scope = getActiveScope();
  auto& buffers = buffersByScope_[scope];

  // Calculate required capacity
  LongType requiredElements = 1;
  for (auto dim : shape) {
    requiredElements *= dim;
  }
  size_t requiredBytes = requiredElements * DataTypeUtils::sizeOf(dtype);
  const bool traceForward4d = key.compare(0, 10, "forward4d_") == 0;
  auto requestedDim = [&shape](size_t index) -> LongType {
    return index < shape.size() ? shape[index] : -1;
  };
  auto rawSpecial = [](NDArray* array) -> void* {
    return array != nullptr && array->dataBuffer() != nullptr
        ? array->dataBuffer()->special() : nullptr;
  };

  auto it = buffers.find(key);
  if (it != buffers.end()) {
    auto& existing = it->second;

    // Check if existing buffer is compatible
    if (existing.buffer->dataType() == dtype && existing.capacity >= requiredElements) {
      // Check actual NDArray shape, not tracked currentShape — external reshapei can desync
      auto* actualShapeVec = existing.buffer->getShapeAsVector();
      bool shapeMatches = (*actualShapeVec == shape);
      delete actualShapeVec;

      if (shapeMatches) {
        existing.lastUsed = ++accessCounter_;
        if (traceForward4d) {
          DSP_DIAG(MEMORY,
                   "ATTENTION_WORKSPACE event=HIT seq=%llu key=%s rank=%zu "
                   "shape=[%lld,%lld,%lld,%lld,%lld] elements=%lld capacity=%lld "
                   "array=%p special=%p",
                   static_cast<unsigned long long>(accessCounter_), key.c_str(), shape.size(),
                   static_cast<long long>(requestedDim(0)), static_cast<long long>(requestedDim(1)),
                   static_cast<long long>(requestedDim(2)), static_cast<long long>(requestedDim(3)),
                   static_cast<long long>(requestedDim(4)), static_cast<long long>(requiredElements),
                   static_cast<long long>(existing.capacity), static_cast<void*>(existing.buffer.get()),
                   rawSpecial(existing.buffer.get()));
        }
        // Return view if active (element count differs from buffer), otherwise return buffer
        if (existing.view) {
          return existing.view.get();
        }
        return existing.buffer.get();
      }

      if (traceForward4d) {
        DSP_DIAG(MEMORY,
                 "ATTENTION_WORKSPACE event=EVICT_SHAPE seq=%llu key=%s rank=%zu "
                 "shape=[%lld,%lld,%lld,%lld,%lld] elements=%lld oldElements=%lld "
                 "oldCapacity=%lld array=%p special=%p",
                 static_cast<unsigned long long>(accessCounter_ + 1), key.c_str(), shape.size(),
                 static_cast<long long>(requestedDim(0)), static_cast<long long>(requestedDim(1)),
                 static_cast<long long>(requestedDim(2)), static_cast<long long>(requestedDim(3)),
                 static_cast<long long>(requestedDim(4)), static_cast<long long>(requiredElements),
                 static_cast<long long>(existing.buffer->lengthOf()),
                 static_cast<long long>(existing.capacity), static_cast<void*>(existing.buffer.get()),
                 rawSpecial(existing.buffer.get()));
      }

      // Shape mismatch — evict old buffer and reallocate.
      // Previously this tried to reshape/reuse the buffer, but that breaks CUDA graphs:
      // the graph records kernel launches with the old shape, and replay with reshaped
      // buffers causes shape mismatches and crashes. Reallocation ensures fresh buffers
      // with correct shape info are used.
      currentMemory_ -= existing.capacity * DataTypeUtils::sizeOf(existing.buffer->dataType());
      buffers.erase(it);
    } else {
      if (traceForward4d) {
        DSP_DIAG(MEMORY,
                 "ATTENTION_WORKSPACE event=EVICT_INCOMPATIBLE seq=%llu key=%s "
                 "requestedType=%d oldType=%d elements=%lld oldCapacity=%lld array=%p special=%p",
                 static_cast<unsigned long long>(accessCounter_ + 1), key.c_str(),
                 static_cast<int>(dtype), static_cast<int>(existing.buffer->dataType()),
                 static_cast<long long>(requiredElements), static_cast<long long>(existing.capacity),
                 static_cast<void*>(existing.buffer.get()), rawSpecial(existing.buffer.get()));
      }
      // Buffer exists but wrong type or too small - need to reallocate
      currentMemory_ -= existing.capacity * DataTypeUtils::sizeOf(existing.buffer->dataType());
      buffers.erase(it);
    }
  }

  // Evict old buffers if needed
  evictIfNeeded(requiredBytes);

  // Allocate new buffer - need mutable copy since NDArray constructor takes non-const ref
  std::vector<LongType> mutableShape(shape);
  auto buffer = std::make_unique<NDArray>('c', mutableShape, dtype, context);

  BufferEntry entry;
  entry.buffer = std::move(buffer);
  entry.currentShape = shape;
  entry.capacity = requiredElements;
  entry.lastUsed = ++accessCounter_;

  currentMemory_ += requiredBytes;
  buffers[key] = std::move(entry);

  if (traceForward4d) {
    auto* allocated = buffers[key].buffer.get();
    DSP_DIAG(MEMORY,
             "ATTENTION_WORKSPACE event=ALLOC seq=%llu key=%s rank=%zu "
             "shape=[%lld,%lld,%lld,%lld,%lld] elements=%lld capacity=%lld "
             "array=%p special=%p workspaceBytes=%zu entries=%zu",
             static_cast<unsigned long long>(accessCounter_), key.c_str(), shape.size(),
             static_cast<long long>(requestedDim(0)), static_cast<long long>(requestedDim(1)),
             static_cast<long long>(requestedDim(2)), static_cast<long long>(requestedDim(3)),
             static_cast<long long>(requestedDim(4)), static_cast<long long>(requiredElements),
             static_cast<long long>(buffers[key].capacity), static_cast<void*>(allocated),
             rawSpecial(allocated), currentMemory_, buffers.size());
  }

  return buffers[key].buffer.get();
}

NDArray* AttentionWorkspace::getScratchBuffer(const std::string& key,
                                               LongType numElements,
                                               DataType dtype,
                                               LaunchContext* context) {
  std::vector<LongType> shape = {numElements};
  return getBuffer(key, shape, dtype, context);
}

void AttentionWorkspace::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t entries = 0;
  for (const auto& scope : buffersByScope_) {
    entries += scope.second.size();
  }
  if (entries > 0) {
    DSP_DIAG(MEMORY, "ATTENTION_WORKSPACE event=CLEAR_ALL scopes=%zu entries=%zu workspaceBytes=%zu",
             buffersByScope_.size(), entries, currentMemory_);
  }
  buffersByScope_.clear();
  currentMemory_ = 0;
}

void AttentionWorkspace::clearScope(void* scope) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto scopeIt = buffersByScope_.find(scope);
  if (scopeIt == buffersByScope_.end()) {
    return;
  }

  size_t scopeBytes = 0;
  for (const auto& pair : scopeIt->second) {
    scopeBytes += pair.second.capacity * DataTypeUtils::sizeOf(pair.second.buffer->dataType());
  }
  DSP_DIAG(MEMORY,
           "ATTENTION_WORKSPACE event=CLEAR_SCOPE scope=%p entries=%zu scopeBytes=%zu workspaceBytes=%zu",
           scope, scopeIt->second.size(), scopeBytes, currentMemory_);

  currentMemory_ = scopeBytes <= currentMemory_ ? currentMemory_ - scopeBytes : 0;
  buffersByScope_.erase(scopeIt);
}

void AttentionWorkspace::clearPrefix(const std::string& prefix) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto scopeIt = buffersByScope_.find(getActiveScope());
  if (scopeIt == buffersByScope_.end()) {
    return;
  }

  auto& buffers = scopeIt->second;
  auto it = buffers.begin();
  while (it != buffers.end()) {
    if (it->first.find(prefix) == 0) {
      currentMemory_ -= it->second.capacity * DataTypeUtils::sizeOf(it->second.buffer->dataType());
      it = buffers.erase(it);
    } else {
      ++it;
    }
  }
  if (buffers.empty()) {
    buffersByScope_.erase(scopeIt);
  }
}

size_t AttentionWorkspace::getMemoryUsage() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return currentMemory_;
}

size_t AttentionWorkspace::getBufferCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (const auto& scope : buffersByScope_) {
    count += scope.second.size();
  }
  return count;
}

void AttentionWorkspace::setMemoryLimit(size_t maxBytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  memoryLimit_ = maxBytes;
  if (memoryLimit_ > 0 && currentMemory_ > memoryLimit_) {
    evictIfNeeded(0);
  }
}

void AttentionWorkspace::evictIfNeeded(size_t requiredBytes) {
  // Already holding lock from caller

  if (memoryLimit_ == 0) {
    return;  // No limit set
  }

  size_t targetMemory = requiredBytes >= memoryLimit_ ? 0 : memoryLimit_ - requiredBytes;
  if (currentMemory_ <= targetMemory) {
    return;  // Already under limit
  }

  struct EvictionCandidate {
    void* scope;
    std::string key;
    uint64_t lastUsed;
  };
  std::vector<EvictionCandidate> candidates;
  for (const auto& scope : buffersByScope_) {
    for (const auto& pair : scope.second) {
      candidates.push_back({scope.first, pair.first, pair.second.lastUsed});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.lastUsed < b.lastUsed; });

  for (const auto& candidate : candidates) {
    if (currentMemory_ <= targetMemory) {
      break;
    }

    auto scopeIt = buffersByScope_.find(candidate.scope);
    if (scopeIt == buffersByScope_.end()) {
      continue;
    }
    auto it = scopeIt->second.find(candidate.key);
    if (it != scopeIt->second.end()) {
      currentMemory_ -= it->second.capacity * DataTypeUtils::sizeOf(it->second.buffer->dataType());
      scopeIt->second.erase(it);
      if (scopeIt->second.empty()) {
        buffersByScope_.erase(scopeIt);
      }
    }
  }
}

}  // namespace sd
