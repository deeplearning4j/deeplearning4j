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

#include <memory/MemoryTracker.h>
#include <stdexcept>
#include <helpers/logger.h>

// FIXME: linux only!!!
#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>

namespace nd4j {
    namespace memory {

        MemoryTracker::MemoryTracker() {
            //
        }

        MemoryTracker* MemoryTracker::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new MemoryTracker();

            return _INSTANCE;
        }

        void MemoryTracker::countIn(MemoryType type, Nd4jPointer ptr, Nd4jLong numBytes) {
            Nd4jLong lptr = reinterpret_cast<Nd4jLong>(ptr);

            void *array[50];
            size_t size;
            char **messages;
            size = backtrace(array, 50);

            std::string stack("");
            messages = backtrace_symbols(array, size);
            for (int i = 1; i < size && messages != NULL; ++i) {
                stack += std::string(messages[i]) + "\n";
            }

            free(messages);

            std::pair<Nd4jLong, AllocationEntry> pair(lptr, AllocationEntry(type, lptr, numBytes, stack));
            _allocations.insert(pair);
        }

        void MemoryTracker::countOut(Nd4jPointer ptr) {
            Nd4jLong lptr = reinterpret_cast<Nd4jLong>(ptr);

            if (_released.count(lptr) > 0) {
                throw std::runtime_error("Double free!");
            }

            if (_allocations.count(lptr) > 0) {
                auto entry = _allocations[lptr];
                std::string stack("new stack");
                std::pair<Nd4jLong, AllocationEntry> pair(lptr, entry);
                _released.insert(pair);


                _allocations.erase(lptr);
            }

        }

        void MemoryTracker::summarize() {
            if (!_allocations.empty()) {
                nd4j_printf("%i leaked allocations\n", (int) _allocations.size());

                for (auto &v: _allocations) {
                    nd4j_printf("Leak of %i bytes\n%s\n\n", (int) v.second.numBytes(), v.second.stackTrace().c_str());
                }

                throw std::runtime_error("Non-released allocations found");
            }
        }

        void MemoryTracker::reset() {
            _allocations.clear();
            _released.clear();
        }

        MemoryTracker* MemoryTracker::_INSTANCE = 0;
    }
}
