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
#include <atomic>
#include <mutex>

namespace samediff {
    class Ticket {
    private:
        bool _acquired = false;
        std::vector<BlockingQueue<CallableWithArguments*>*> _queues;
        std::vector<CallableWithArguments*> _callables;

    public:
        explicit Ticket(const std::vector<BlockingQueue<CallableWithArguments*>*> &queues);
        Ticket() = default;
        ~Ticket() = default;

        bool acquired();

        void enqueue(int thread_id, CallableWithArguments *callable);

        void waitAndRelease();
    };
}


#endif //DEV_TESTS_TICKET_H
