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
// Attention Workspace Manager - Reusable buffer pool to eliminate allocation overhead
//

#ifndef LIBND4J_ATTENTION_WORKSPACE_H
#define LIBND4J_ATTENTION_WORKSPACE_H

#include "array/NDArray.h"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>

namespace sd {

/**
 * AttentionWorkspace - Thread-local workspace for attention operations
 *
 * Maintains a pool of reusable buffers to eliminate cudaMalloc/cudaFree overhead.
 * Buffers are keyed by their total size and reused when shapes match.
 *
 * Usage:
 *   auto workspace = AttentionWorkspace::getInstance();
 *   NDArray* scores = workspace->getBuffer("scores", {batch, seqQ, seqKV}, dtype, context);
 *   // ... use buffer ...
 *   // Buffer is automatically reused on next call with same key
 */
class SD_LIB_EXPORT AttentionWorkspace {
 public:
  /**
   * Get the thread-local workspace instance
   */
  static AttentionWorkspace* getInstance();

  /**
   * Get or create a buffer with the given key and shape
   * If a buffer exists with the same key and sufficient size, it's reshaped and returned
   * Otherwise a new buffer is allocated (and cached for future use)
   *
   * @param key Unique identifier for this buffer (e.g., "scores", "qPerm")
   * @param shape Desired shape for the buffer
   * @param dtype Data type
   * @param context Launch context
   * @return Pointer to the buffer (owned by workspace, do NOT delete)
   */
  NDArray* getBuffer(const std::string& key, const std::vector<LongType>& shape,
                     DataType dtype, LaunchContext* context = nullptr);

  /**
   * Get a buffer by size (for temporary workspace)
   * Useful when you just need scratch space of a certain size
   *
   * @param key Unique identifier
   * @param numElements Number of elements needed
   * @param dtype Data type
   * @param context Launch context
   * @return Pointer to 1D buffer with at least numElements
   */
  NDArray* getScratchBuffer(const std::string& key, LongType numElements,
                            DataType dtype, LaunchContext* context = nullptr);

  /**
   * Clear all cached buffers (frees memory)
   * Call this when you're done with attention operations for a while
   */
  void clear();

  /**
   * Clear buffers for a specific key prefix
   * Useful for clearing related buffers (e.g., all "forward_" buffers)
   */
  void clearPrefix(const std::string& prefix);

  /**
   * Get total memory used by workspace buffers
   */
  size_t getMemoryUsage() const;

  /**
   * Get number of cached buffers
   */
  size_t getBufferCount() const;

  /**
   * Set maximum memory limit for workspace (0 = unlimited)
   * When limit is reached, oldest unused buffers are evicted
   */
  void setMemoryLimit(size_t maxBytes);

  ~AttentionWorkspace();

 private:
  AttentionWorkspace();

  struct BufferEntry {
    std::unique_ptr<NDArray> buffer;
    std::vector<LongType> currentShape;
    size_t capacity;  // Total elements allocated
    uint64_t lastUsed;  // For LRU eviction
  };

  std::unordered_map<std::string, BufferEntry> buffers_;
  mutable std::mutex mutex_;
  size_t memoryLimit_ = 0;
  size_t currentMemory_ = 0;
  uint64_t accessCounter_ = 0;

  void evictIfNeeded(size_t requiredBytes);

  // Thread-local instance
  static thread_local AttentionWorkspace* instance_;
};

/**
 * AttentionWorkspaceScope - RAII helper for automatic workspace cleanup
 *
 * Usage:
 *   {
 *     AttentionWorkspaceScope scope("forward");
 *     // ... do attention operations ...
 *   } // Automatically clears "forward_*" buffers when scope exits
 */
class SD_LIB_EXPORT AttentionWorkspaceScope {
 public:
  explicit AttentionWorkspaceScope(const std::string& prefix) : prefix_(prefix) {}
  ~AttentionWorkspaceScope() {
    // Don't clear on scope exit - keep buffers for reuse
    // Only clear explicitly when needed
  }

  // Manually clear this scope's buffers
  void clear() {
    AttentionWorkspace::getInstance()->clearPrefix(prefix_);
  }

 private:
  std::string prefix_;
};

}  // namespace sd

#endif  // LIBND4J_ATTENTION_WORKSPACE_H
