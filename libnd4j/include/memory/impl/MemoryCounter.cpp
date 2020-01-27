/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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
#include <Environment.h>
#include <helpers/logger.h>

namespace nd4j {
    namespace memory {

        MemoryCounter::MemoryCounter() {
            auto numDevices = nd4j::AffinityManager::numberOfDevices();

            // setting default 0s
            for (int e = 0; e < numDevices; e++) {
                _deviceLimits[e] = 0;
                _deviceCounters[e] = 0;
            }

            // setting initial values for limits
            _groupLimits[nd4j::memory::MemoryType::HOST] = nd4j::Environment::getInstance()->maxPrimaryMemory();
            _groupLimits[nd4j::memory::MemoryType::DEVICE] = nd4j::Environment::getInstance()->maxSpecialMemory();

            // setting initial counter values
            _groupCounters[nd4j::memory::MemoryType::HOST] = 0;
            _groupCounters[nd4j::memory::MemoryType::DEVICE] = 0;
        }

        MemoryCounter* MemoryCounter::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new MemoryCounter();

            return _INSTANCE;
        }

        void MemoryCounter::countIn(int deviceId, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            _deviceCounters[deviceId] += numBytes;
        }

        void MemoryCounter::countIn(nd4j::memory::MemoryType group, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            _groupCounters[group] += numBytes;
        }

        void MemoryCounter::countOut(int deviceId, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            _deviceCounters[deviceId] -= numBytes;
        }

        void MemoryCounter::countOut(nd4j::memory::MemoryType group, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            _groupCounters[group] -= numBytes;
        }

        bool MemoryCounter::validate(Nd4jLong numBytes) {
            auto deviceId = nd4j::AffinityManager::currentDeviceId();
            return validateDevice(deviceId, numBytes);
        }

        bool MemoryCounter::validateDevice(int deviceId, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            auto dLimit = _deviceLimits[deviceId];
            if (dLimit <= 0)
                return true;

            auto dAlloc = _deviceCounters[deviceId];

            return numBytes + dAlloc <= dLimit;
        }

        bool MemoryCounter::validateGroup(nd4j::memory::MemoryType group, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            auto gLimit = _groupLimits[group];
            if (gLimit <= 0)
                return true;

            auto gAlloc = _groupCounters[group];

            return numBytes + gAlloc <= gLimit;
        }

        Nd4jLong MemoryCounter::allocatedDevice(int deviceId) {
            std::lock_guard<std::mutex> lock(_locker);
            return _deviceCounters[deviceId];
        }

        Nd4jLong MemoryCounter::allocatedGroup(nd4j::memory::MemoryType group) {
            std::lock_guard<std::mutex> lock(_locker);
            return _groupCounters[group];
        }

        void MemoryCounter::setDeviceLimit(int deviceId, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            _deviceLimits[deviceId] = numBytes;
        }

        void MemoryCounter::setGroupLimit(nd4j::memory::MemoryType group, Nd4jLong numBytes) {
            std::lock_guard<std::mutex> lock(_locker);
            _groupLimits[group] = numBytes;
        }

        Nd4jLong MemoryCounter::deviceLimit(int deviceId) {
            std::lock_guard<std::mutex> lock(_locker);
            return _deviceLimits[deviceId];
        }

        Nd4jLong MemoryCounter::groupLimit(nd4j::memory::MemoryType group) {
            std::lock_guard<std::mutex> lock(_locker);
            return _groupLimits[group];
        }

        MemoryCounter* MemoryCounter::_INSTANCE = 0;
    }
}