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

#ifndef SD_MEMORYCOUNTER_H
#define SD_MEMORYCOUNTER_H

#include <system/pointercast.h>
#include <system/dll.h>
#include <map>
#include <memory/MemoryType.h>
#include <mutex>

namespace sd {
    namespace memory {
        /**
         * This class provides simple per-device counter
         */
        class ND4J_EXPORT MemoryCounter {
        private:
            static MemoryCounter* _INSTANCE;

            // used for synchronization
            std::mutex _locker;

            // per-device counters
            std::map<int, Nd4jLong> _deviceCounters;

            // TODO: change this wrt heterogenous stuff on next iteration
            // per-group counters
            std::map<sd::memory::MemoryType, Nd4jLong> _groupCounters;

            // per-device limits
            std::map<int, Nd4jLong> _deviceLimits;

            // per-group limits
            std::map<sd::memory::MemoryType, Nd4jLong> _groupLimits;

            MemoryCounter();
            ~MemoryCounter() = default;

        public:
            static MemoryCounter *getInstance();

            /**
             * This method checks if allocation of numBytes won't break through per-group or per-device limit
             * @param numBytes
             * @return TRUE if allocated ammount will keep us below limit, FALSE otherwise
             */
            bool validate(Nd4jLong numBytes);

            /**
             * This method checks if allocation of numBytes won't break through  per-device limit
             * @param deviceId
             * @param numBytes
             * @return TRUE if allocated ammount will keep us below limit, FALSE otherwise
             */
            bool validateDevice(int deviceId, Nd4jLong numBytes);

            /**
             * This method checks if allocation of numBytes won't break through per-group limit
             * @param deviceId
             * @param numBytes
             * @return TRUE if allocated ammount will keep us below limit, FALSE otherwise
             */
            bool validateGroup(sd::memory::MemoryType group, Nd4jLong numBytes);

            /**
             * This method adds specified number of bytes to specified counter
             * @param deviceId
             * @param numBytes
             */
            void countIn(int deviceId, Nd4jLong numBytes);
            void countIn(sd::memory::MemoryType group, Nd4jLong numBytes);

            /**
             * This method subtracts specified number of bytes from specified counter
             * @param deviceId
             * @param numBytes
             */
            void countOut(int deviceId, Nd4jLong numBytes);
            void countOut(sd::memory::MemoryType group, Nd4jLong numBytes);

            /**
             * This method returns amount of memory allocated on specified device
             * @param deviceId
             * @return
             */
            Nd4jLong allocatedDevice(int deviceId);

            /**
             * This method returns amount of memory allocated in specified group of devices
             * @param group
             * @return
             */
            Nd4jLong allocatedGroup(sd::memory::MemoryType group);

            /**
             * This method allows to set per-device memory limits
             * @param deviceId
             * @param numBytes
             */
            void setDeviceLimit(int deviceId, Nd4jLong numBytes);

            /**
             * This method returns current device limit in bytes
             * @param deviceId
             * @return
             */
            Nd4jLong deviceLimit(int deviceId);

            /**
             * This method allows to set per-group memory limits
             * @param group
             * @param numBytes
             */
            void setGroupLimit(sd::memory::MemoryType group, Nd4jLong numBytes);

            /**
             * This method returns current group limit in bytes
             * @param group
             * @return
             */
            Nd4jLong groupLimit(sd::memory::MemoryType group);
        };
    }
}


#endif //SD_MEMORYCOUNTER_H
