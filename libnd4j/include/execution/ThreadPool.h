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

#ifndef SAMEDIFF_THREADPOOL_H
#define SAMEDIFF_THREADPOOL_H

#include <list>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <execution/BlockingQueue.h>
#include <execution/CallableWithArguments.h>
#include <execution/CallableInterface.h>
#include <execution/Ticket.h>
#include <queue>

namespace samediff {
    class ThreadPool {
    private:
        static ThreadPool* _INSTANCE;

        std::vector<std::thread*> _threads;
        std::vector<BlockingQueue<CallableWithArguments*>*> _queues;
        std::vector<CallableInterface*> _interfaces;

        std::mutex _lock;
        std::atomic<int> _available;
        std::queue<Ticket*> _tickets;
    protected:
        ThreadPool();
        ~ThreadPool();
    public:
        static ThreadPool* getInstance();

        /**
         * This method returns list of pointers to threads ONLY if num_threads of threads were available upon request, returning empty list otherwise
         * @param num_threads
         * @return
         */
        Ticket* tryAcquire(int num_threads);

        /**
         * This method marks specified number of threads as released, and available for use
         * @param num_threads
         */
        void release(int num_threads = 1);

        void release(Ticket *ticket);
    };
}


#endif //DEV_TESTS_THREADPOOL_H
