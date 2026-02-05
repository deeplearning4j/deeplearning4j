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
#include <algorithm>

namespace sd {

// Thread-local instance
thread_local AttentionWorkspace* AttentionWorkspace::instance_ = nullptr;

AttentionWorkspace* AttentionWorkspace::getInstance() {
  if (instance_ == nullptr) {
    instance_ = new AttentionWorkspace();
  }
  return instance_;
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

  // Calculate required capacity
  LongType requiredElements = 1;
  for (auto dim : shape) {
    requiredElements *= dim;
  }
  size_t requiredBytes = requiredElements * DataTypeUtils::sizeOf(dtype);

  auto it = buffers_.find(key);
  if (it != buffers_.end()) {
    auto& entry = it->second;

    // Check if existing buffer is compatible
    if (entry.buffer->dataType() == dtype && entry.capacity >= requiredElements) {
      // Reshape existing buffer to new shape
      // Note: This is a view operation if shapes are compatible
      if (entry.currentShape != shape) {
        entry.buffer->reshapei(shape);
        entry.currentShape = shape;
      }
      entry.lastUsed = ++accessCounter_;
      return entry.buffer.get();
    }

    // Buffer exists but wrong type or too small - need to reallocate
    currentMemory_ -= entry.capacity * DataTypeUtils::sizeOf(entry.buffer->dataType());
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
  buffers_[key] = std::move(entry);

  return buffers_[key].buffer.get();
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
  buffers_.clear();
  currentMemory_ = 0;
}

void AttentionWorkspace::clearPrefix(const std::string& prefix) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = buffers_.begin();
  while (it != buffers_.end()) {
    if (it->first.find(prefix) == 0) {
      currentMemory_ -= it->second.capacity * DataTypeUtils::sizeOf(it->second.buffer->dataType());
      it = buffers_.erase(it);
    } else {
      ++it;
    }
  }
}

size_t AttentionWorkspace::getMemoryUsage() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return currentMemory_;
}

size_t AttentionWorkspace::getBufferCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffers_.size();
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

  size_t targetMemory = memoryLimit_ - requiredBytes;
  if (currentMemory_ <= targetMemory) {
    return;  // Already under limit
  }

  // Build list of buffers sorted by last used time (LRU)
  std::vector<std::pair<std::string, uint64_t>> candidates;
  for (const auto& pair : buffers_) {
    candidates.push_back({pair.first, pair.second.lastUsed});
  }

  // Sort by last used (oldest first)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Evict until under limit
  for (const auto& candidate : candidates) {
    if (currentMemory_ <= targetMemory) {
      break;
    }

    auto it = buffers_.find(candidate.first);
    if (it != buffers_.end()) {
      currentMemory_ -= it->second.capacity * DataTypeUtils::sizeOf(it->second.buffer->dataType());
      buffers_.erase(it);
    }
  }
}

}  // namespace sd
