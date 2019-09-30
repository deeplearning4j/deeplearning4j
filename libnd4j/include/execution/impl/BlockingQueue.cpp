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
// @author raver119@gmail.com
//

#include <execution/BlockingQueue.h>
#include <CallableWithArguments.h>
#include <thread>

namespace samediff {
    template <typename T>
    BlockingQueue<T>::BlockingQueue(int queueSize) {
        _size = 0;
        _available = true;
    }

    template <typename T>
    T BlockingQueue<T>::poll() {
        // locking untill there's something within queue
        std::unique_lock<std::mutex> lock(_lock);
        _condition.wait(lock, [=]{ return !this->_queue.empty(); });

        T t(std::move(_queue.front()));
        _queue.pop();
        return t;
    }

    template <typename T>
    void BlockingQueue<T>::put(const T &t) {
        {
            // locking before push, unlocking after
            std::unique_lock<std::mutex> lock(_lock);
            _queue.push(t);
            _available = false;
        }

        // notifying condition
        _condition.notify_one();
    }

    template <typename T>
    bool BlockingQueue<T>::available() {
        return _available.load();
    }

    template <typename T>
    void BlockingQueue<T>::markAvailable() {
        _available = true;
    }

    template class BlockingQueue<CallableWithArguments*>;
}
