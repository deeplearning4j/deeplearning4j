/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <helpers/logger.h>
#include <array>

namespace samediff {
    Ticket::Ticket(const std::vector<BlockingQueue<CallableWithArguments*>*> &queues) {
        _acquired = true;
        _queues = queues;
    }

    Ticket::Ticket() {
        _acquired = true;
        _interfaces.resize(sd::Environment::getInstance().maxThreads());
    }

    bool Ticket::acquired() {
        return _acquired;
    }

    void Ticket::enqueue(int thread_id, samediff::CallableWithArguments *callable) {
        _queues[thread_id]->put(callable);
        _callables.emplace_back(callable);
    }

    void Ticket::enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_DO func) {
        _interfaces[thread_id]->fill(thread_id, num_threads, func);
    }

    void Ticket::enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_1D func, int64_t start_x, int64_t stop_x, int64_t inc_x) {
        _interfaces[thread_id]->fill(thread_id, num_threads, func, start_x, stop_x, inc_x);
    }

    void Ticket::enqueue(uint32_t thread_id, uint32_t num_threads, int64_t *lpt, FUNC_RL func, int64_t start_x, int64_t stop_x, int64_t inc_x) {
        _interfaces[thread_id]->fill(thread_id, num_threads, lpt, func, start_x, stop_x, inc_x);
    }

    void Ticket::enqueue(uint32_t thread_id, uint32_t num_threads, double *dpt, FUNC_RD func, int64_t start_x, int64_t stop_x, int64_t inc_x) {
        _interfaces[thread_id]->fill(thread_id, num_threads, dpt, func, start_x, stop_x, inc_x);
    }

    void Ticket::enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_2D func, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y) {
        _interfaces[thread_id]->fill(thread_id, num_threads, std::move(func), start_x, stop_x, inc_x, start_y, stop_y, inc_y);
    }

    void Ticket::enqueue(uint32_t thread_id, uint32_t num_threads, FUNC_3D func, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z) {
        _interfaces[thread_id]->fill(thread_id, num_threads, func, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);
    }

    void Ticket::acquiredThreads(uint32_t threads) {
        _acquiredThreads = threads;
    }

    void Ticket::waitAndRelease() {
        for (uint32_t e = 0; e < this->_acquiredThreads; e++) {
            // block until finished
            _interfaces[e]->waitForCompletion();

            // mark available
            _interfaces[e]->markAvailable();

            // increment availability counter
            ThreadPool::getInstance().release();
        }

        // return this ticket back to the pool
        ThreadPool::getInstance().release(this);
    }


    void Ticket::attach(uint32_t thread_id, samediff::CallableInterface *interface) {
        _interfaces[thread_id] = interface;
    }
}