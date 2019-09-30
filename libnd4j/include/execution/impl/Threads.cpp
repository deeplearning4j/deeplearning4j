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
    int Threads::parallel_for(FUNC_1D function, uint64_t start, uint64_t stop, uint64_t increment, uint32_t numThreads) {
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket.acquired()) {
            // if we got our threads - we'll run our jobs here
            int span = (stop - start) / numThreads;

            for (int e = 0; e < numThreads; e++) {
                int _start = span * e + start;
                int _stop = span * (e + 1) + start;
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
            nd4j_printf("Running one thread\n","");
            // if there were no threads available - we'll execute function right within current thread
            function(0, start, stop, increment);

            // we tell that parallelism request declined
            return 1;
        }
    }

    int Threads::parallel_for(FUNC_2D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t numThreads) {
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
        // TODO: to be implemented
        function(0);

        return 1;
    }
}