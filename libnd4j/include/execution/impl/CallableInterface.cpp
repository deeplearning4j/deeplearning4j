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

#include <execution/CallableInterface.h>
#include <helpers/logger.h>

namespace samediff {
    CallableInterface::CallableInterface() {
        // initial state is available
        _available = true;
        _filled = false;
        _finished = false;
    }

    bool CallableInterface::available() {
        return _available.load();
    }

    void CallableInterface::markUnavailable() {
        _available = false;
    }

    void CallableInterface::markAvailable() {
        _available = true;
    }

    void CallableInterface::fill(int threadID, int numThreads, FUNC_DO func) {
        _function_do = std::move(func);

        _branch = 0;
        _num_threads = numThreads;
        _thread_id = threadID;
        _finished = false;
        {
            std::unique_lock<std::mutex> l(_ms);
            _filled = true;
        }
        _starter.notify_one();
    }

    void CallableInterface::fill(int threadID, int numThreads, FUNC_1D func, int64_t startX, int64_t stopX, int64_t incX) {
        _function_1d = std::move(func);
        _arguments[0] = startX;
        _arguments[1] = stopX;
        _arguments[2] = incX;

        _branch = 1;
        _num_threads = numThreads;
        _thread_id = threadID;
        _finished = false;

        {
            std::unique_lock<std::mutex> l(_ms);
            _filled = true;
        }
        _starter.notify_one();
    }

    void CallableInterface::fill(int threadID, int numThreads, FUNC_2D func, int64_t startX, int64_t stopX, int64_t incX, int64_t start_y, int64_t stop_y, int64_t inc_y) {
        _function_2d = std::move(func);
        _arguments[0] = startX;
        _arguments[1] = stopX;
        _arguments[2] = incX;
        _arguments[3] = start_y;
        _arguments[4] = stop_y;
        _arguments[5] = inc_y;

        _branch = 2;
        _num_threads = numThreads;
        _thread_id = threadID;
        _finished = false;

        {
            std::unique_lock<std::mutex> l(_ms);
            _filled = true;
        }
        _starter.notify_one();
    }

    void CallableInterface::fill(int threadID, int numThreads, FUNC_3D func, int64_t startX, int64_t stopX, int64_t incX, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z) {
        _function_3d = std::move(func);
        _arguments[0] = startX;
        _arguments[1] = stopX;
        _arguments[2] = incX;
        _arguments[3] = start_y;
        _arguments[4] = stop_y;
        _arguments[5] = inc_y;
        _arguments[6] = start_z;
        _arguments[7] = stop_z;
        _arguments[8] = inc_z;

        _branch = 3;
        _num_threads = numThreads;
        _thread_id = threadID;
        _finished = false;

        {
            std::unique_lock<std::mutex> l(_ms);
            _filled = true;
        }
        _starter.notify_one();
    }

    void CallableInterface::fill(int threadID, int numThreads, int64_t *lptr, FUNC_RL func, int64_t startX, int64_t stopX, int64_t incX) {
        _function_rl = std::move(func);
        _arguments[0] = startX;
        _arguments[1] = stopX;
        _arguments[2] = incX;

        _lptr = lptr;

        _branch = 4;
        _num_threads = numThreads;
        _thread_id = threadID;
        _finished = false;

        {
            std::unique_lock<std::mutex> l(_ms);
            _filled = true;
        }
        _starter.notify_one();
    }

    void CallableInterface::fill(int threadID, int numThreads, double *dptr, FUNC_RD func, int64_t startX, int64_t stopX, int64_t incX) {
        _function_rd = std::move(func);
        _arguments[0] = startX;
        _arguments[1] = stopX;
        _arguments[2] = incX;

        _dptr = dptr;

        _branch = 5;
        _num_threads = numThreads;
        _thread_id = threadID;
        _finished = false;

        {
            std::unique_lock<std::mutex> l(_ms);
            _filled = true;
        }
        _starter.notify_one();
    }

    void CallableInterface::waitForTask() {
        // block until task is available
        std::unique_lock<std::mutex> lock(_ms);
        _starter.wait(lock, [&]{ return _filled.load(); });
    }

    void CallableInterface::waitForCompletion() {
        //while (!_finished.load());

        // block until finished
        std::unique_lock<std::mutex> lock(_mf);
        _finisher.wait(lock, [&] { return _finished.load(); });
    }

    void CallableInterface::finish() {
        // mark as finished
        {
            std::unique_lock<std::mutex> l(_mf);
            _finished.store(true);
        }
        _finisher.notify_one();
    }

    void CallableInterface::execute() {
        // mark it as consumed
        _filled = false;

        // actually executing op
        switch (_branch) {
            case 0:
                _function_do(_thread_id, _num_threads);
                break;
            case 1:
                _function_1d(_thread_id, _arguments[0], _arguments[1], _arguments[2]);
                break;
            case 2:
                _function_2d(_thread_id, _arguments[0], _arguments[1], _arguments[2], _arguments[3], _arguments[4], _arguments[5]);
                break;
            case 3:
                _function_3d(_thread_id, _arguments[0], _arguments[1], _arguments[2], _arguments[3], _arguments[4], _arguments[5], _arguments[6], _arguments[7], _arguments[8]);
                break;
            case 4:
                _lptr[0] = _function_rl(_thread_id, _arguments[0], _arguments[1], _arguments[2]);
                break;
            case 5:
                _dptr[0] = _function_rd(_thread_id, _arguments[0], _arguments[1], _arguments[2]);
                break;
        }

        // notify that thread finished the job
        this->finish();
    }
}