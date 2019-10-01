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
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <vector>
#include <thread>
#include <helpers/logger.h>


namespace samediff {
    int ThreadsHelper::numberOfThreads(int maxThreads, uint64_t numberOfElements) {
        // no sense launching more threads than elements
        if (numberOfElements < maxThreads)
            return numberOfElements;

        return maxThreads;
    }

    int Threads::parallel_for(FUNC_1D function, uint64_t start, uint64_t stop, uint64_t increment, uint32_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_for got start > stop");

        auto delta = (stop - start);

        // in some cases we just fire func as is
        if (delta == 0 || numThreads == 1) {
            function(0, start, stop, increment);
            return 1;
        }

        auto numElements = delta / increment;

        numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
        // we don't want to launch single thread, just call function in place
        if (numThreads == 1) {
            function(0, start, stop, increment);
            return 1;
        }

        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket.acquired()) {
            // if we got our threads - we'll run our jobs here
            auto span = delta / numThreads;

            for (int e = 0; e < numThreads; e++) {
                auto _start = span * e + start;
                auto _stop = span * (e + 1) + start;

                // last thread will process tail
                if (e == numThreads - 1)
                    _stop = stop;

                // FIXME: make callables served from pool, rather than from creating new one
                // putting the task into the queue for a given thread
                ticket.enqueue(e, new CallableWithArguments(function, e, _start, _stop, increment));
            }

            // block and wait till all threads finished the job
            ticket.waitAndRelease();

            // we tell that parallelism request succeeded
            return numThreads;
        } else {
            // if there were no threads available - we'll execute function right within current thread
            function(0, start, stop, increment);

            // we tell that parallelism request declined
            return 1;
        }
    }

    int Threads::parallel_for(FUNC_2D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t numThreads) {
        if (start_x > stop_x)
            throw std::runtime_error("Threads::parallel_for got start_x > stop_x");

        if (start_y > stop_y)
            throw std::runtime_error("Threads::parallel_for got start_y > stop_y");

        function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y);
        return 1;
        /*
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket.acquired()) {

            for (int e = 0; e < numThreads; e++) {

            }

            // we tell that parallelism request succeeded
            return numThreads;
        } else {
            nd4j_printf("Running one thread\n","");
            // if there were no threads available - we'll execute function right within current thread
            function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y);

            // we tell that parallelism request declined
            return 1;
        }
         */
    }


    int Threads::parallel_for(FUNC_3D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t start_z, uint64_t stop_z, uint64_t inc_z, uint64_t numThreads) {
        if (start_x > stop_x)
            throw std::runtime_error("Threads::parallel_for got start_x > stop_x");

        if (start_y > stop_y)
            throw std::runtime_error("Threads::parallel_for got start_y > stop_y");

        if (start_z > stop_z)
            throw std::runtime_error("Threads::parallel_for got start_z > stop_z");

        function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);
        return 1;
        /*
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket.acquired()) {

            for (int e = 0; e < numThreads; e++) {

            }

            // we tell that parallelism request succeeded
            return numThreads;
        } else {
            nd4j_printf("Running one thread\n","");
            // if there were no threads available - we'll execute function right within current thread
            function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);

            // we tell that parallelism request declined
            return 1;
        }
        */
    }

    int Threads::parallel_do(FUNC_DO function, uint64_t numThreads) {
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket.acquired()) {

            // submit tasks one by one
            for (uint64_t e = 0; e < numThreads; e++)
                ticket.enqueue(e, new CallableWithArguments(function, e));

            ticket.waitAndRelease();

            return numThreads;
        } else {
            // if there's no threads available - we'll execute function sequentially one by one
            for (uint64_t e = 0; e < numThreads; e++)
                function(e);

            return 1;
        }
    }
}