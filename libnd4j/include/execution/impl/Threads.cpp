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
#include <templatemath.h>
#include <shape.h>


namespace samediff {
    int ThreadsHelper::numberOfThreads(int maxThreads, uint64_t numberOfElements) {
        // let's see how many threads we actually need first
        auto optimalThreads = nd4j::math::nd4j_max<uint64_t>(1, numberOfElements / 1024);

        // now return the smallest value
        return nd4j::math::nd4j_min<int>(optimalThreads, maxThreads);
    }

    Span2::Span2(int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y) {
        _startX = start_x;
        _startY = start_y;
        _stopX = stop_x;
        _stopY = stop_y;
        _incX = inc_x;
        _incY = inc_y;
    }


    Span2 Span2::build(int loop, uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y) {
        if (loop == 1) {
            auto span = (stop_x - start_x) / num_threads;
            auto s = span * thread_id;
            auto e = span * (thread_id + 1);
            if (thread_id == num_threads - 1)
                e = stop_x;

            return Span2(s, e, inc_x, start_y, stop_y, inc_y);
        } else {
            auto span = (stop_y - start_y) / num_threads;
            auto s = span * thread_id;
            auto e = span * (thread_id + 1);
            if (thread_id == num_threads - 1)
                e = stop_y;

            return Span2(start_x, stop_x, inc_x, s, e, inc_y);
        }
    }

    int64_t Span2::startX() {
        return _startX;
    }

    int64_t Span2::startY() {
        return _startY;
    }

    int64_t Span2::stopX() {
        return _stopX;
    }

    int64_t Span2::stopY() {
        return _stopY;
    }

    int64_t Span2::incX() {
        return _incX;
    }

    int64_t Span2::incY() {
        return _incY;
    }

    int ThreadsHelper::pickLoop2d(int numThreads, uint64_t iters_x, uint64_t iters_y) {
        // if one of dimensions is definitely too small - we just pick the other one
        if (iters_x < numThreads)
            return 2;
        else if (iters_y < numThreads)
            return 1;

        // next step - we pick the most balanced dimension
        auto rem_x = iters_x % numThreads;
        auto rem_y = iters_y % numThreads;
        auto split_y = iters_y / numThreads;

        // if there's no remainder left in some dimension - we're picking that dimension, because it'll be the most balanced work distribution
        if (rem_x == 0)
            return 1;
        else if (rem_y == 0)
            return 2;


        // if there's no loop without a remainder - we're picking one with smaller remainder
        if (rem_x < rem_y)
            return 1;
        else if (rem_y < rem_x && split_y >= 64) // we don't want too small splits over last dimension, or vectorization will fail
            return 2;
        else // if loops are equally sized - give the preference to the first thread
            return 1;
    }

    int Threads::parallel_tad(FUNC_1D function, uint64_t start, uint64_t stop, uint64_t increment, uint32_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_for got start > stop");

        function(0, start, stop, increment);
        return 1;
    }

    int Threads::parallel_for(FUNC_1D function, uint64_t start, uint64_t stop, uint64_t increment, uint32_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_for got start > stop");

        if (1 > 0) {
            function(0, start, stop, increment);
            return 1;
        }

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

    int Threads::parallel_for(FUNC_2D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t numThreads, bool debug) {
        if (start_x > stop_x)
            throw std::runtime_error("Threads::parallel_for got start_x > stop_x");

        if (start_y > stop_y)
            throw std::runtime_error("Threads::parallel_for got start_y > stop_y");

        // number of elements per loop
        auto delta_x = (stop_x - start_x);
        auto delta_y = (stop_y - start_y);

        // number of iterations per loop
        auto iters_x = delta_x / inc_x;
        auto iters_y = delta_y / inc_y;

        // total number of iterations
        auto iters_t = iters_x * iters_y;

        // we are checking the case of number of requested threads was smaller
        numThreads = ThreadsHelper::numberOfThreads(numThreads, iters_t);

        // basic shortcut for no-threading cases
        if (numThreads == 1) {
            function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y);
            return 1;
        }

        // We have couple of scenarios:
        // either we split workload along 1st loop, or 2nd
        auto splitLoop = ThreadsHelper::pickLoop2d(numThreads, iters_x, iters_y);

        // for debug mode we execute things inplace, without any threada
        if (debug) {
            for (int e = 0; e < numThreads; e++) {
                auto span = Span2::build(splitLoop, e, numThreads, start_x, stop_x, inc_x, start_y, stop_y, inc_y);

                function(e, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY());
            }
        } else {
            auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
            if (ticket.acquired()) {
                //nd4j_printf("Threads: starting with %i threads\n", numThreads);
                for (int e = 0; e < numThreads; e++) {
                    auto span = Span2::build(splitLoop, e, numThreads, start_x, stop_x, inc_x, start_y, stop_y, inc_y);

                    ticket.enqueue(e, new CallableWithArguments(function, e, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY()));
                }

                // block until all threads finish their job
                ticket.waitAndRelease();
                return numThreads;
            } else {
                // if there were no threads available - we'll execute function right within current thread
                function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y);

                // we tell that parallelism request declined
                return 1;
            }
        };
    }


    int Threads::parallel_for(FUNC_3D function, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t start_z, uint64_t stop_z, uint64_t inc_z, uint64_t numThreads) {
        if (start_x > stop_x)
            throw std::runtime_error("Threads::parallel_for got start_x > stop_x");

        if (start_y > stop_y)
            throw std::runtime_error("Threads::parallel_for got start_y > stop_y");

        if (start_z > stop_z)
            throw std::runtime_error("Threads::parallel_for got start_z > stop_z");

        if (1 > 0) {
            function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);
            return 1;
        }

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

    }

    int Threads::parallel_do(FUNC_DO function, uint64_t numThreads) {
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket.acquired()) {

            // submit tasks one by one
            for (uint64_t e = 0; e < numThreads; e++)
                ticket.enqueue(e, new CallableWithArguments(function, e, numThreads));

            ticket.waitAndRelease();

            return numThreads;
        } else {
            // if there's no threads available - we'll execute function sequentially one by one
            for (uint64_t e = 0; e < numThreads; e++)
                function(e, numThreads);

            return numThreads;
        }


        return numThreads;
    }

}