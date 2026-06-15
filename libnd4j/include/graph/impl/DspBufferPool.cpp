/* ******************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
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

#include <graph/DspBufferPool.h>
#include <graph/DspDiagnostics.h>
#include <array/DataTypeUtils.h>
#include <execution/AffinityManager.h>

#include <cstring>
#include <stdexcept>

namespace sd {
namespace graph {

// ─── Static storage ──────────────────────────────────────────────────────────

std::mutex DspBufferPool::globalMutex_;
std::unordered_map<int, DspBufferPool*> DspBufferPool::pools_;

// ─── Constructor / Destructor ────────────────────────────────────────────────

DspBufferPool::DspBufferPool(int deviceId) : deviceId_(deviceId) {}

DspBufferPool::~DspBufferPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& kv : available_) {
    for (auto* buf : kv.second) {
      delete buf;
    }
  }
  available_.clear();
  pooledSet_.clear();
  pooledBytes_.store(0, std::memory_order_relaxed);
  pooledCount_.store(0, std::memory_order_relaxed);
}

// ─── Singleton per device ────────────────────────────────────────────────────

DspBufferPool& DspBufferPool::forDevice(int deviceId) {
  std::lock_guard<std::mutex> lock(globalMutex_);
  auto it = pools_.find(deviceId);
  if (it != pools_.end()) return *it->second;

  auto* pool = new DspBufferPool(deviceId);
  pools_[deviceId] = pool;
  return *pool;
}

DspBufferPool& DspBufferPool::forCurrentDevice() {
  return forDevice(AffinityManager::currentDeviceId());
}

void DspBufferPool::destroyAll() {
  std::lock_guard<std::mutex> lock(globalMutex_);
  for (auto& kv : pools_) {
    delete kv.second;
  }
  pools_.clear();
}

// ─── Key computation ─────────────────────────────────────────────────────────

uint64_t DspBufferPool::keyFor(LongType numElements, DataType dtype) {
  // FNV-1a style hash combining numElements and dtype
  uint64_t h = 14695981039346656037ULL;
  auto ne = static_cast<uint64_t>(numElements);
  auto dt = static_cast<uint64_t>(dtype);
  h ^= ne;
  h *= 1099511628211ULL;
  h ^= dt;
  h *= 1099511628211ULL;
  return h;
}

// ─── Acquire ─────────────────────────────────────────────────────────────────

DataBuffer* DspBufferPool::acquire(LongType numElements, DataType dtype) {
  totalAcquired_.fetch_add(1, std::memory_order_relaxed);

  uint64_t key = keyFor(numElements, dtype);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = available_.find(key);
    if (it != available_.end() && !it->second.empty()) {
      DataBuffer* buf = it->second.back();
      it->second.pop_back();
      if (it->second.empty()) {
        available_.erase(it);
      }
      pooledSet_.erase(buf);

      size_t bufBytes = buf->getLenInBytes();
      pooledBytes_.fetch_sub(bufBytes, std::memory_order_relaxed);
      pooledCount_.fetch_sub(1, std::memory_order_relaxed);
      totalReused_.fetch_add(1, std::memory_order_relaxed);

      DSP_DIAG(MEMORY, "POOL_ACQUIRE reused: numElements=%lld dtype=%s device=%d pooledBytes=%zu",
               (long long)numElements,
               DataTypeUtils::asString(dtype).c_str(),
               deviceId_,
               pooledBytes_.load(std::memory_order_relaxed));

      return buf;
    }
  }

  // No match in pool — allocate fresh
  size_t sizeInBytes = numElements * DataTypeUtils::sizeOfElement(dtype);
  auto* buf = new DataBuffer(sizeInBytes, dtype);

  DSP_DIAG(MEMORY, "POOL_ACQUIRE fresh: numElements=%lld dtype=%s device=%d bytes=%zu",
           (long long)numElements,
           DataTypeUtils::asString(dtype).c_str(),
           deviceId_, sizeInBytes);

  return buf;
}

// ─── Release ─────────────────────────────────────────────────────────────────

void DspBufferPool::release(DataBuffer* buffer, LongType numElements, DataType dtype) {
  if (buffer == nullptr) {
    THROW_EXCEPTION("DspBufferPool::release(): buffer is null");
  }

  uint64_t key = keyFor(numElements, dtype);
  size_t bufBytes = buffer->getLenInBytes();

  // Validate buffer size matches claimed
  size_t expectedBytes = numElements * DataTypeUtils::sizeOfElement(dtype);
  if (bufBytes < expectedBytes) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "DspBufferPool::release(): buffer size %zu < expected %zu for %lld elements of %s",
             bufBytes, expectedBytes, (long long)numElements,
             DataTypeUtils::asString(dtype).c_str());
    THROW_EXCEPTION(msg);
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Double-release detection
  if (pooledSet_.count(buffer) > 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "DspBufferPool::release(): double-release of DataBuffer %p",
             (void*)buffer);
    THROW_EXCEPTION(msg);
  }

  available_[key].push_back(buffer);
  pooledSet_.insert(buffer);
  pooledBytes_.fetch_add(bufBytes, std::memory_order_relaxed);
  pooledCount_.fetch_add(1, std::memory_order_relaxed);

  DSP_DIAG(MEMORY, "POOL_RELEASE: numElements=%lld dtype=%s device=%d pooledBytes=%zu",
           (long long)numElements,
           DataTypeUtils::asString(dtype).c_str(),
           deviceId_,
           pooledBytes_.load(std::memory_order_relaxed));
}

// ─── Trim ────────────────────────────────────────────────────────────────────

size_t DspBufferPool::trim(size_t targetFreeBytes) {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t freedBytes = 0;

  // Free buffers from the pool until we've freed enough or pool is empty
  // Iterate over shape classes and free from each
  auto it = available_.begin();
  while (it != available_.end() && (targetFreeBytes == 0 || freedBytes < targetFreeBytes)) {
    auto& buffers = it->second;
    while (!buffers.empty() && (targetFreeBytes == 0 || freedBytes < targetFreeBytes)) {
      DataBuffer* buf = buffers.back();
      size_t bufBytes = buf->getLenInBytes();
      buffers.pop_back();
      pooledSet_.erase(buf);
      delete buf;
      freedBytes += bufBytes;
      pooledBytes_.fetch_sub(bufBytes, std::memory_order_relaxed);
      pooledCount_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (buffers.empty()) {
      it = available_.erase(it);
    } else {
      ++it;
    }
  }

  DSP_DIAG(MEMORY, "POOL_TRIM: freed %zuMB, %d buffers remaining",
           freedBytes / (1024 * 1024),
           pooledCount_.load(std::memory_order_relaxed));

  return freedBytes;
}

// ─── Introspection ───────────────────────────────────────────────────────────

size_t DspBufferPool::pooledBytes() const {
  return pooledBytes_.load(std::memory_order_relaxed);
}

int DspBufferPool::pooledCount() const {
  return pooledCount_.load(std::memory_order_relaxed);
}

DspBufferPool::PoolReport DspBufferPool::report() const {
  std::lock_guard<std::mutex> lock(mutex_);

  PoolReport r;
  r.deviceId = deviceId_;
  r.pooledBytes = pooledBytes_.load(std::memory_order_relaxed);
  r.pooledCount = pooledCount_.load(std::memory_order_relaxed);
  r.numShapeClasses = static_cast<int>(available_.size());
  r.totalAcquired = totalAcquired_.load(std::memory_order_relaxed);
  r.totalReused = totalReused_.load(std::memory_order_relaxed);
  r.reuseRate = (r.totalAcquired > 0)
      ? static_cast<float>(r.totalReused) / static_cast<float>(r.totalAcquired)
      : 0.0f;
  return r;
}

}  // namespace graph
}  // namespace sd
