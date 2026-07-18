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

//
// This class implements Workspace functionality in c++
//
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_WORKSPACE_H
#define LIBND4J_WORKSPACE_H

#include <memory/ExternalWorkspace.h>
#include <memory/MemoryType.h>
#include <system/common.h>
#include <types/float16.h>

#include <atomic>
#include <mutex>
#include <vector>

namespace sd {
namespace memory {

class MultiBackendWorkspace;

class SD_LIB_EXPORT Workspace {
 public:
  // Size of canary region appended after workspace host buffer (debug mode only)
  static constexpr sd::LongType CANARY_SIZE = 4096;
  static constexpr unsigned char CANARY_BYTE = 0xDE;

 protected:
  char* _ptrHost = nullptr;
  char* _ptrDevice = nullptr;

  bool _allocatedHost = false;
  bool _allocatedDevice = false;
  bool _canaryEnabled = false;  // true when canary region was written

  std::atomic<sd::LongType> _offset;
  std::atomic<sd::LongType> _offsetSecondary;

  sd::LongType _initialSize = 0L;
  sd::LongType _initialSizeSecondary = 0L;

  sd::LongType _currentSize = 0L;
  sd::LongType _currentSizeSecondary = 0L;

  std::mutex _mutexAllocation;
  std::mutex _mutexSpills;

  bool _externalized = false;

  std::vector<void*> _spills;
  std::vector<void*> _spillsSecondary;

  std::atomic<sd::LongType> _spillsSize;
  std::atomic<sd::LongType> _cycleAllocations;

  std::atomic<sd::LongType> _spillsSizeSecondary;
  std::atomic<sd::LongType> _cycleAllocationsSecondary;

  int _deviceId = -1;  // Device where device memory was allocated (-1 = not set)

  // When true, the secondary (host) buffer is allocated via plain malloc/free rather
  // than cudaHostAlloc/cudaFreeHost.  Set by the CUDA MultiBackendWorkspace for
  // CPU-device workspaces so that no CUDA context is required for the allocation.
  bool _secondaryUsePlainMalloc = false;

  void init(sd::LongType primaryBytes, sd::LongType secondaryBytes = 0L);
  void freeSpills();

  /**
   * Snapshot separately owned primary allocation identities. The returned
   * pointers are borrowed and remain valid only while workspace allocation
   * lifetime is externally stable.
   */
  std::vector<void*> snapshotPrimaryAllocations() {
    std::lock_guard<std::mutex> lock(_mutexSpills);
    return _spills;
  }

  friend class MultiBackendWorkspace;

 public:
  explicit Workspace(ExternalWorkspace* external);
  Workspace(sd::LongType initialSize = 0L, sd::LongType secondaryBytes = 0L,
            bool secondaryUsePlainMalloc = false);
  ~Workspace();

  sd::LongType getAllocatedSize();
  sd::LongType getCurrentSize();
  sd::LongType getCurrentOffset();
  sd::LongType getSpilledSize();
  sd::LongType getUsedSize();

  sd::LongType getAllocatedSecondarySize();
  sd::LongType getCurrentSecondarySize();
  sd::LongType getCurrentSecondaryOffset();
  sd::LongType getSpilledSecondarySize();
  sd::LongType getUsedSecondarySize();

  void expandBy(sd::LongType primaryBytes, sd::LongType secondaryBytes = 0L);
  void expandTo(sd::LongType primaryBytes, sd::LongType secondaryBytes = 0L);

  //            bool resizeSupported();

  void* allocateBytes(sd::LongType numBytes);
  void* allocateBytes(MemoryType type, sd::LongType numBytes);

  void scopeIn();
  void scopeOut();

  /**
   * Enable canary region after the host buffer. Only call once after construction.
   * Fills CANARY_SIZE bytes after the host buffer with CANARY_BYTE pattern.
   * Only active in debug/verbose mode.
   */
  void enableCanary();

  /**
   * Check if the canary region has been corrupted.
   * Returns the byte offset of the first corrupted byte, or -1 if canary is intact.
   * Only checks if canary was previously enabled.
   */
  sd::LongType checkCanary() const;

  /**
   * Get the raw host (secondary) memory pointer for this workspace.
   * On CPU backend this is the primary pointer; on CUDA it is the pinned host pointer.
   */
  void* getHostPointer() const { return _ptrHost; }

  /**
   * Get the raw device (primary) memory pointer for this workspace.
   * On CPU backend this may be nullptr; on CUDA it is the device pointer.
   */
  void* getDevicePointer() const { return _ptrDevice; }

  /*
   * This method creates NEW workspace of the same memory size and returns pointer to it
   */
  Workspace* clone();
};
}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_WORKSPACE_H
