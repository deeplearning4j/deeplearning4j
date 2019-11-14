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

#ifndef SAMEDIFF_BLOCKINGQUEUE_H
#define SAMEDIFF_BLOCKINGQUEUE_H

#include <functional>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace samediff {
    template <typename T>
    class BlockingQueue {
    private:
        std::queue<T> _queue;
        std::mutex _lock;
        std::atomic<int> _size;
        std::atomic<bool> _available;

        std::condition_variable _condition;
    public:
        BlockingQueue(int queueSize);
        ~BlockingQueue() = default;
        T poll();
        void put(const T &t);

        bool available();
        void markAvailable();
        void markUnavailable();
    };
}

#endif //DEV_TESTS_BLOCKINGQUEUE_H
