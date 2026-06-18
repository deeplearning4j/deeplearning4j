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

#ifndef LIBND4J_CAPTURE_BUFFER_REGISTRY_H
#define LIBND4J_CAPTURE_BUFFER_REGISTRY_H

#include <system/common.h>
#include <cstddef>

namespace sd {
namespace graph {

// Forward declaration — opaque, implementation in .cu file
class CaptureBufferRegistryImpl;

/**
 * Thin registry that tracks {segIdx → vector<{ptr, bytes, deviceId}>} mappings.
 * All actual allocation/deallocation goes through CudaMemoryPool.
 *
 * When capture pool is enabled, capture workspace allocations are routed
 * through CudaMemoryPool instead of raw cudaMalloc. This allows freed memory
 * from invalidated segments to be automatically reused by other segments
 * via CUDA's native pool.
 */
class CaptureBufferRegistry {
 public:
  CaptureBufferRegistry();
  ~CaptureBufferRegistry();

  /**
   * Allocate via CudaMemoryPool, register ownership to segment.
   * @param segIdx  Segment index (used as ownership key)
   * @param bytes   Allocation size
   * @param deviceId  Target CUDA device
   * @return Pointer to allocated memory, or nullptr on failure
   */
  void* allocate(int segIdx, size_t bytes, int deviceId);

  /**
   * Release all buffers owned by segment back to CudaMemoryPool.
   * @param segIdx  Segment to release
   */
  void releaseSegment(int segIdx);

  /**
   * Release ALL registered buffers back to CudaMemoryPool.
   */
  void releaseAll();

  /**
   * Get total bytes registered to a specific segment.
   */
  size_t segmentBytes(int segIdx) const;

  /**
   * Get total bytes across all registered segments.
   */
  size_t totalBytes() const;

 private:
  CaptureBufferRegistryImpl* impl_;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_CAPTURE_BUFFER_REGISTRY_H
