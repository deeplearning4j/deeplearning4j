/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// Created by raver119 on 07.05.19.
//

#ifndef DEV_TESTS_MEMORYTRACKER_H
#define DEV_TESTS_MEMORYTRACKER_H

#include <map>
#include <string>
#include <pointercast.h>
#include <mutex>
#include "AllocationEntry.h"

namespace nd4j {
    namespace memory {
        class MemoryTracker {
        private:
            static MemoryTracker* _INSTANCE;
            std::map<Nd4jLong, AllocationEntry> _allocations;
            std::map<Nd4jLong, AllocationEntry> _released;
            std::mutex _locker;

            MemoryTracker();
            ~MemoryTracker() = default;
        public:
            static MemoryTracker* getInstance();

            void countIn(MemoryType type, Nd4jPointer ptr, Nd4jLong numBytes);
            void countOut(Nd4jPointer ptr);

            void summarize();
            void reset();
        };
    }
}


#endif //DEV_TESTS_MEMORYTRACKER_H
