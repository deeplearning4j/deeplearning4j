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
// Created by raver119 on 07.05.19.
//

#ifndef DEV_TESTS_MEMORYTRACKER_H
#define DEV_TESTS_MEMORYTRACKER_H
#include <system/common.h>

#include <map>
#include <mutex>
#include <string>

#include "AllocationEntry.h"

namespace sd {
namespace memory {
/**
 * This class is used for tracking memory allocation wrt their allocation points in code
 */
class SD_LIB_EXPORT MemoryTracker {
 private:
  std::map<sd::LongType, AllocationEntry> _allocations;
  std::map<sd::LongType, AllocationEntry> _released;
  std::mutex _locker;

  MemoryTracker();
  ~MemoryTracker() = default;

 public:
  static MemoryTracker& getInstance();

  void countIn(MemoryType type, sd::Pointer ptr, sd::LongType numBytes);
  void countOut(sd::Pointer ptr);

  void summarize();
  void reset();
};
}  // namespace memory
}  // namespace sd

#endif  // DEV_TESTS_MEMORYTRACKER_H
