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

#ifndef DEV_TESTS_CALLABLEWITHARGUMENTS_H
#define DEV_TESTS_CALLABLEWITHARGUMENTS_H

#include <functional>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <op_boilerplate.h>

namespace samediff {
    class CallableWithArguments {
        FUNC_DO _function_do;
        FUNC_1D _function_1d;
        FUNC_2D _function_2d;
        FUNC_3D _function_3d;

        std::vector<int64_t> _arguments;

        std::atomic<bool> _finished;

        std::condition_variable _condition;

        std::mutex _lock;

        int _dimensions = 0;

        uint64_t _threadId;
        uint64_t _numThreads;
    public:
        CallableWithArguments(FUNC_DO func, uint64_t thread_id, uint64_t numThreads);
        CallableWithArguments(FUNC_1D func, uint64_t thread_id, int64_t start_x, int64_t stop_x, int64_t increment_x);
        CallableWithArguments(FUNC_2D func, uint64_t thread_id, int64_t start_x, int64_t stop_x, int64_t increment_x, int64_t start_y, int64_t stop_y, int64_t increment_y);
        CallableWithArguments(FUNC_3D func, uint64_t thread_id, int64_t start_x, int64_t stop_x, int64_t increment_x, int64_t start_y, int64_t stop_y, int64_t increment_y, int64_t start_z, int64_t stop_z, int64_t increment_z);


        /**
         * This method returns number of dimensions
         * @return
         */
        int dimensions();

        /**
         * This method checks if this callable is finished
         * @return
         */
        bool finished();

        /**
         * this method marks this Callable as finished
         */
        void finish();

        /**
         * This method blocks until callable is finished
         */
        void waitUntilFinished();

        std::vector<int64_t>& arguments();
        FUNC_DO function_do();
        FUNC_1D function_1d();
        FUNC_2D function_2d();
        FUNC_3D function_3d();


        uint64_t threadId();

        uint64_t numThreads();
    };
}


#endif //DEV_TESTS_CALLABLEWITHARGUMENTS_H
