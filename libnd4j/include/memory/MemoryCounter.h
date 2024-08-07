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
// @author raver119@gmail.com
//

#ifndef SD_MEMORYCOUNTER_H
#define SD_MEMORYCOUNTER_H

#include <memory/MemoryType.h>
#include <system/common.h>

#include <map>
#include <mutex>

namespace sd {
namespace memory {
/**
 * This class provides simple per-device counter
 */
class SD_LIB_EXPORT MemoryCounter {
 private:
  // used for synchronization
  std::mutex _locker;

  // per-device counters
  std::map<int, LongType> _deviceCounters;

  // TODO: change this wrt heterogenous stuff on next iteration
  // per-group counters
  std::map<MemoryType, LongType> _groupCounters;

  // per-device limits
  std::map<int, LongType> _deviceLimits;

  // per-group limits
  std::map<MemoryType, LongType> _groupLimits;

  MemoryCounter();
  ~MemoryCounter() = default;

 public:
  static MemoryCounter& getInstance();

  /**
   * This method checks if allocation of numBytes won't break through per-group or per-device limit
   * @param numBytes
   * @return TRUE if allocated ammount will keep us below limit, FALSE otherwise
   */
  bool validate(LongType numBytes);

  /**
   * This method checks if allocation of numBytes won't break through  per-device limit
   * @param deviceId
   * @param numBytes
   * @return TRUE if allocated ammount will keep us below limit, FALSE otherwise
   */
  bool validateDevice(int deviceId, LongType numBytes);

  /**
   * This method checks if allocation of numBytes won't break through per-group limit
   * @param deviceId
   * @param numBytes
   * @return TRUE if allocated ammount will keep us below limit, FALSE otherwise
   */
  bool validateGroup(MemoryType group, LongType numBytes);

  /**
   * This method adds specified number of bytes to specified counter
   * @param deviceId
   * @param numBytes
   */
  void countIn(int deviceId, LongType numBytes);
  void countIn(MemoryType group, LongType numBytes);

  /**
   * This method subtracts specified number of bytes from specified counter
   * @param deviceId
   * @param numBytes
   */
  void countOut(int deviceId, LongType numBytes);
  void countOut(MemoryType group, LongType numBytes);

  /**
   * This method returns amount of memory allocated on specified device
   * @param deviceId
   * @return
   */
  LongType allocatedDevice(int deviceId);

  /**
   * This method returns amount of memory allocated in specified group of devices
   * @param group
   * @return
   */
  LongType allocatedGroup(MemoryType group);

  /**
   * This method allows to set per-device memory limits
   * @param deviceId
   * @param numBytes
   */
  void setDeviceLimit(int deviceId, LongType numBytes);

  /**
   * This method returns current device limit in bytes
   * @param deviceId
   * @return
   */
  LongType deviceLimit(int deviceId);

  /**
   * This method allows to set per-group memory limits
   * @param group
   * @param numBytes
   */
  void setGroupLimit(MemoryType group, LongType numBytes);

  /**
   * This method returns current group limit in bytes
   * @param group
   * @return
   */
  LongType groupLimit(MemoryType group);
};
}  // namespace memory
}  // namespace sd

#endif  // SD_MEMORYCOUNTER_H
