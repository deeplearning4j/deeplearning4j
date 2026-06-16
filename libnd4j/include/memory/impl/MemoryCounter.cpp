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
#include <memory/MemoryUtils.h>
#include <mutex>
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

  // init soft limit from Environment (reads SD_CPU_SOFT_LIMIT_PERCENT env var)
  int softLimit = Environment::getInstance().cpuSoftLimitPercent();
  if (softLimit > 0 && softLimit <= 100) {
    _softLimitPercent.store(softLimit, std::memory_order_relaxed);
  }
}

MemoryCounter& MemoryCounter::getInstance() {
  static MemoryCounter* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new MemoryCounter();
  });
  return *instance;
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

void MemoryCounter::setSoftLimitPercent(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  _softLimitPercent.store(percent, std::memory_order_relaxed);
}

int MemoryCounter::getSoftLimitPercent() {
  return _softLimitPercent.load(std::memory_order_relaxed);
}

bool MemoryCounter::validateSoftLimit(LongType numBytes) {
  int softLimit = _softLimitPercent.load(std::memory_order_relaxed);
  if (softLimit <= 0) return true;  // disabled

  size_t freeBytes = MemoryUtils::getSystemFreeMemoryBytes();
  if (freeBytes == 0) return true;  // query failed, don't block

  // Estimate total system RAM from free + our tracked allocations.
  // This is approximate but avoids an extra syscall — MemoryUtils only
  // provides free memory, not total. We use HOST group counter as a
  // proxy for our consumption.
  LongType ourUsage = 0;
  {
    std::lock_guard<std::mutex> lock(_locker);
    ourUsage = _groupCounters[HOST];
  }
  size_t estimatedTotal = freeBytes + static_cast<size_t>(ourUsage > 0 ? ourUsage : 0);
  if (estimatedTotal == 0) return true;

  double usagePercent = 100.0 * (1.0 - static_cast<double>(freeBytes) / static_cast<double>(estimatedTotal));

  if (usagePercent >= static_cast<double>(softLimit)) {
    if (sd::Environment::getInstance().isVerbose()) {
      sd_printf("MemoryCounter: CPU soft limit hit — system at %.1f%% usage "
                "(soft limit %d%%), free: %zu MB, rejecting %lld bytes\n",
                usagePercent, softLimit, freeBytes / (1024 * 1024), (long long)numBytes);
    }
    return false;
  }

  return true;
}

}  // namespace memory
}  // namespace sd
