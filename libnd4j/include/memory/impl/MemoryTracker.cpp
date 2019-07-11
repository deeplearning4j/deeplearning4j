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


#include <stdlib.h>

#if defined(__GNUC__) && !defined(__MINGW64__) && !defined(ANDROID_BUILD) && !defined(IOS_BUILD)  && !defined(APPLE_BUILD)

#include <unistd.h>
#include <execinfo.h>
#include <cxxabi.h>

#endif

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

#if defined(__GNUC__) && !defined(__MINGW64__) && !defined(ANDROID_BUILD) && !defined(IOS_BUILD)  && !defined(APPLE_BUILD)
        std::string demangle(char *message) {
            char *mangled_name = 0, *offset_begin = 0, *offset_end = 0;

            // find parantheses and +address offset surrounding mangled name
            for (char *p = message; *p; ++p)
            {
                if (*p == '(')
                {
                    mangled_name = p;
                }
                else if (*p == '+')
                {
                    offset_begin = p;
                }
                else if (*p == ')')
                {
                    offset_end = p;
                    break;
                }
            }

            // if the line could be processed, attempt to demangle the symbol
            if (mangled_name && offset_begin && offset_end && mangled_name < offset_begin) {
                *mangled_name++ = '\0';
                *offset_begin++ = '\0';
                *offset_end++ = '\0';

                int status;
                char * real_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);

                // if demangling is successful, output the demangled function name
                if (status == 0) {
                    std::string result(real_name);
                    free(real_name);
                    return result;
                } else {
                    // otherwise, output the mangled function name
                    std::string result (message);
                    free(real_name);
                    return result;
                }
            }
        }

#endif

        void MemoryTracker::countIn(MemoryType type, Nd4jPointer ptr, Nd4jLong numBytes) {
#if defined(__GNUC__) && !defined(__MINGW64__) && !defined(ANDROID_BUILD) && !defined(IOS_BUILD)  && !defined(APPLE_BUILD)
            if (Environment::getInstance()->isDetectingLeaks()) {
                auto lptr = reinterpret_cast<Nd4jLong>(ptr);

                _locker.lock();

                void *array[50];
                size_t size;
                char **messages;
                size = backtrace(array, 50);

                std::string stack("");
                messages = backtrace_symbols(array, size);
                for (int i = 1; i < size && messages != NULL; ++i) {
                    stack += demangle(messages[i]) + "\n";
                }

                free(messages);

                if (stack.find("ConstantTad") != std::string::npos ||
                    stack.find("ConstantShape") != std::string::npos) {
                    _locker.unlock();
                    return;
                }

                std::pair<Nd4jLong, AllocationEntry> pair(lptr, AllocationEntry(type, lptr, numBytes, stack));
                _allocations.insert(pair);

                _locker.unlock();
            }
#endif
        }

        void MemoryTracker::countOut(Nd4jPointer ptr) {
#if defined(__GNUC__) && !defined(__MINGW64__) && !defined(ANDROID_BUILD) && !defined(IOS_BUILD)  && !defined(APPLE_BUILD)
            if (Environment::getInstance()->isDetectingLeaks()) {
                auto lptr = reinterpret_cast<Nd4jLong>(ptr);

                _locker.lock();
                if (_released.count(lptr) > 0) {
                    //throw std::runtime_error("Double free!");
                }

                if (_allocations.count(lptr) > 0) {
                    //auto entry = _allocations[lptr];
                    //std::string stack("new stack");
                    //std::pair<Nd4jLong, AllocationEntry> pair(lptr, entry);
                    //_released.insert(pair);


                    _allocations.erase(lptr);
                }

                _locker.unlock();
            }
#endif
        }

        void MemoryTracker::summarize() {
            if (!_allocations.empty()) {
                nd4j_printf("\n%i leaked allocations\n", (int) _allocations.size());

                for (auto &v: _allocations) {
                    nd4j_printf("Leak of %i [%s] bytes\n%s\n\n", (int) v.second.numBytes(), v.second.memoryType() == MemoryType::HOST ? "HOST" : "DEVICE", v.second.stackTrace().c_str());
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
