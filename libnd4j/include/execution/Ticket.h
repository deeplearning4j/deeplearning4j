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

#ifndef SAMEDIFF_TICKET_H
#define SAMEDIFF_TICKET_H

#include <vector>
#include <execution/BlockingQueue.h>
#include <execution/CallableWithArguments.h>
#include <execution/CallableInterface.h>
#include <atomic>
#include <mutex>

namespace samediff {
    class Ticket {
    private:
        bool _acquired = false;
        std::vector<BlockingQueue<CallableWithArguments*>*> _queues;
        std::vector<CallableWithArguments*> _callables;
        std::vector<CallableInterface*> _interfaces;

        uint32_t _acquiredThreads = 0;
    public:
        explicit Ticket(const std::vector<BlockingQueue<CallableWithArguments*>*> &queues);
        Ticket();
        ~Ticket() = default;

        bool acquired();

        void acquiredThreads(uint32_t threads);

        void attach(uint32_t thread_id, CallableInterface *interface);

        // deprecated one
        void enqueue(int thread_id, CallableWithArguments* callable);

        void enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_DO func);
        void enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_1D func, int64_t start_x, int64_t stop_x, int64_t inc_x);
        void enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_2D func, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y);
        void enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_3D func, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_, int64_t stop_z, int64_t inc_z);

        void waitAndRelease();
    };
}


#endif //DEV_TESTS_TICKET_H
