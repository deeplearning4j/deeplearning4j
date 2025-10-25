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

class SD_LIB_EXPORT Workspace {
 protected:
  char* _ptrHost = nullptr;
  char* _ptrDevice = nullptr;

  bool _allocatedHost = false;
  bool _allocatedDevice = false;

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

  void init(sd::LongType primaryBytes, sd::LongType secondaryBytes = 0L);
  void freeSpills();

 public:
  explicit Workspace(ExternalWorkspace* external);
  Workspace(sd::LongType initialSize = 0L, sd::LongType secondaryBytes = 0L);
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

  /*
   * This method creates NEW workspace of the same memory size and returns pointer to it
   */
  Workspace* clone();
};
}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_WORKSPACE_H
