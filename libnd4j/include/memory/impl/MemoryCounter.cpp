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
#include "../MemoryCounter.h"

#include <execution/AffinityManager.h>
#include <helpers/logger.h>
#include <system/Environment.h>

namespace sd {
namespace memory {

MemoryCounter::MemoryCounter() {
  auto numDevices = AffinityManager::numberOfDevices();

  // setting default 0s
  for (int e = 0; e < numDevices; e++) {
    _deviceLimits[e] = 0;
    _deviceCounters[e] = 0;
  }

  // setting initial values for limits
  _groupLimits[HOST] = Environment::getInstance().maxPrimaryMemory();
  _groupLimits[DEVICE] = Environment::getInstance().maxSpecialMemory();

  // setting initial counter values
  _groupCounters[HOST] = 0;
  _groupCounters[DEVICE] = 0;
}

MemoryCounter& MemoryCounter::getInstance() {
  static MemoryCounter instance;
  return instance;
}

void MemoryCounter::countIn(int deviceId, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  _deviceCounters[deviceId] += numBytes;
}

void MemoryCounter::countIn(MemoryType group, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  _groupCounters[group] += numBytes;
}

void MemoryCounter::countOut(int deviceId, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  _deviceCounters[deviceId] -= numBytes;
}

void MemoryCounter::countOut(MemoryType group, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  _groupCounters[group] -= numBytes;
}

bool MemoryCounter::validate(LongType numBytes) {
  auto deviceId = AffinityManager::currentDeviceId();
  return validateDevice(deviceId, numBytes);
}

bool MemoryCounter::validateDevice(int deviceId, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  auto dLimit = _deviceLimits[deviceId];
  if (dLimit <= 0) return true;

  auto dAlloc = _deviceCounters[deviceId];

  return numBytes + dAlloc <= dLimit;
}

bool MemoryCounter::validateGroup(MemoryType group, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  auto gLimit = _groupLimits[group];
  if (gLimit <= 0) return true;

  auto gAlloc = _groupCounters[group];

  return numBytes + gAlloc <= gLimit;
}

LongType MemoryCounter::allocatedDevice(int deviceId) {
  std::lock_guard<std::mutex> lock(_locker);
  return _deviceCounters[deviceId];
}

LongType MemoryCounter::allocatedGroup(MemoryType group) {
  std::lock_guard<std::mutex> lock(_locker);
  return _groupCounters[group];
}

void MemoryCounter::setDeviceLimit(int deviceId, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  _deviceLimits[deviceId] = numBytes;
}

void MemoryCounter::setGroupLimit(MemoryType group, LongType numBytes) {
  std::lock_guard<std::mutex> lock(_locker);
  _groupLimits[group] = numBytes;
}

LongType MemoryCounter::deviceLimit(int deviceId) {
  std::lock_guard<std::mutex> lock(_locker);
  return _deviceLimits[deviceId];
}

LongType MemoryCounter::groupLimit(MemoryType group) {
  std::lock_guard<std::mutex> lock(_locker);
  return _groupLimits[group];
}
}  // namespace memory
}  // namespace sd
