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

#ifndef SAMEDIFF_CALLABLEINTERFACE_H
#define SAMEDIFF_CALLABLEINTERFACE_H

#include <openmp_pragmas.h>
#include <cstdint>
#include <functional>
#include <atomic>
#include <array>
#include <mutex>
#include <condition_variable>

namespace samediff {
    /**
     * This class is suited for passing functions to execution threads without queues
     */
    class CallableInterface {
    private:
        FUNC_1D _function_1d;
        FUNC_2D _function_2d;
        FUNC_3D _function_3d;
        FUNC_DO _function_do;

        std::array<int64_t, 9> _arguments;

        volatile int _branch = 0;
        volatile uint32_t _thread_id = 0;
        volatile uint32_t _num_threads = 0;

        std::atomic<bool> _finished;
        std::atomic<bool> _filled;
        std::atomic<bool> _available;

        std::condition_variable _starter;
        std::condition_variable _finisher;

        std::mutex _ms;
        std::mutex _mf;
    public:
        CallableInterface();
        ~CallableInterface() = default;

        void waitForTask();
        void waitForCompletion();

        void fill(int thread_id, int num_threads, FUNC_DO func);
        void fill(int thread_id, int num_threads, FUNC_1D func, int64_t start_x, int64_t stop_x, int64_t inc_x);
        void fill(int thread_id, int num_threads, FUNC_2D func, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y);
        void fill(int thread_id, int num_threads, FUNC_3D func, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z);

        bool available();
        void markAvailable();
        void markUnavailable();

        void finish();

        void execute();
    };
}


#endif //DEV_TESTS_CALLABLEINTERFACE_H
