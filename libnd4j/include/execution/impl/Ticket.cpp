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

#include <execution/Ticket.h>
#include <execution/ThreadPool.h>

namespace samediff {
    Ticket::Ticket(const std::vector<BlockingQueue<CallableWithArguments*>*> &queues) {
        _acquired = true;
        _queues = queues;
    }

    bool Ticket::acquired() {
        return _acquired;
    }

    void Ticket::enqueue(int thread_id, samediff::CallableWithArguments *callable) {
        _queues[thread_id]->put(callable);
        _callables.emplace_back(callable);
    }

    void Ticket::waitAndRelease() {
        // we need to wait till all chunks finished
        for (auto c:_callables) {
            // blocking on the current callable, till it finishes
            c->waitUntilFinished();

            // release callable
            delete c;

            // notify that queue is available
            _queues[c->threadId()]->markAvailable();

            // notify ThreadPool that at least one thread finished
            ThreadPool::getInstance()->release();
        }
    }
}