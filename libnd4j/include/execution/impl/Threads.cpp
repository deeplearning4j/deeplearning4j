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

    Span3::Span3(int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z) {
        _startX = start_x;
        _startY = start_y;
        _startZ = start_z;
        _stopX = stop_x;
        _stopY = stop_y;
        _stopZ = stop_z;
        _incX = inc_x;
        _incY = inc_y;
        _incZ = inc_z;
    }

    Span3 Span3::build(int loop, uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z) {
        switch (loop) {
            case 1: {
                    auto span = (stop_x - start_x) / num_threads;
                    auto s = span * thread_id;
                    auto e = span * (thread_id + 1);
                    if (thread_id == num_threads - 1)
                        e = stop_x;

                    return Span3(s, e, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);
                }
                break;
            case 2: {
                    auto span = (stop_y - start_y) / num_threads;
                    auto s = span * thread_id;
                    auto e = span * (thread_id + 1);
                    if (thread_id == num_threads - 1)
                        e = stop_y;

                    return Span3(start_x, stop_x, inc_x, s, e, inc_y, start_z, stop_z, inc_z);
                }
                break;
            case 3: {
                    auto span = (stop_z - start_z) / num_threads;
                    auto s = span * thread_id;
                    auto e = span * (thread_id + 1);
                    if (thread_id == num_threads - 1)
                        e = stop_z;

                    return Span3(start_x, stop_x, inc_x, start_y, stop_y, inc_y, s, e, inc_z);
                }
                break;
            default:
                throw std::runtime_error("");
        }
        return Span3(start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);
    }

    Span::Span(int64_t start_x, int64_t stop_x, int64_t inc_x) {
        _startX = start_x;
        _stopX = stop_x;
        _incX = inc_x;
    }

    Span Span::build(uint64_t thread_id, uint64_t num_threads, int64_t start_x, int64_t stop_x, int64_t inc_x) {
        auto span = (stop_x - start_x) / num_threads;
        auto s = span * thread_id;
        auto e = span * (thread_id + 1);
        if (thread_id == num_threads - 1)
            e = stop_x;

        return Span(s, e, inc_x);
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

    int64_t Span::startX() const {
        return _startX;
    }

    int64_t Span::stopX() const {
        return _stopX;
    }

    int64_t Span::incX() const {
        return _incX;
    }

    int64_t Span2::startX() const {
        return _startX;
    }

    int64_t Span2::startY() const {
        return _startY;
    }

    int64_t Span2::stopX() const {
        return _stopX;
    }

    int64_t Span2::stopY() const {
        return _stopY;
    }

    int64_t Span2::incX() const {
        return _incX;
    }

    int64_t Span2::incY() const {
        return _incY;
    }

    int64_t Span3::startX() const {
        return _startX;
    }

    int64_t Span3::startY() const {
        return _startY;
    }

    int64_t Span3::startZ() const {
        return _startZ;
    }

    int64_t Span3::stopX() const {
        return _stopX;
    }

    int64_t Span3::stopY() const {
        return _stopY;
    }

    int64_t Span3::stopZ() const {
        return _stopZ;
    }

    int64_t Span3::incX() const {
        return _incX;
    }

    int64_t Span3::incY() const {
        return _incY;
    }

    int64_t Span3::incZ() const {
        return _incZ;
    }

    int ThreadsHelper::pickLoop2d(int numThreads, uint64_t iters_x, uint64_t iters_y) {
        // if one of dimensions is definitely too small - we just pick the other one
        if (iters_x < numThreads && iters_y >= numThreads)
            return 2;
        else if (iters_y < numThreads && iters_x >= numThreads)
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


    static int _threads(int maxThreads, uint64_t elements) {
        if (elements == maxThreads) {
            return maxThreads;
        } else if (elements > maxThreads) {
            // if we have full load across thread, or at least half of threads can be utilized
            auto rem = elements % maxThreads;
            if (rem == 0 || rem >= maxThreads / 3) {
                return maxThreads;
            } else {
                return _threads(maxThreads - 1, elements);
            }
        } else if (elements < maxThreads) {
            return elements;
        }

        return 1;
    }

    int ThreadsHelper::numberOfThreads3d(int maxThreads, uint64_t iters_x, uint64_t iters_y, uint64_t iters_z) {
        // we don't want to run underloaded threads
        if (iters_x * iters_y * iters_z <= 128)
            return 1;

        auto rem_x = iters_x % maxThreads;
        auto rem_y = iters_y % maxThreads;
        auto rem_z = iters_z % maxThreads;

        // if we have perfect balance across one of dimensions - just go for it
        if ((iters_x >= maxThreads && rem_x == 0) || (iters_y >= maxThreads && rem_y == 0) || (iters_z >= maxThreads && rem_z == 0))
            return maxThreads;

        int threads_x = 0, threads_y = 0, threads_z = 0;

        // now we look into possible number of
        threads_x = _threads(maxThreads, iters_x);
        threads_y = _threads(maxThreads, iters_y);
        threads_z = _threads(maxThreads, iters_z);

        // we want to split as close to outer loop as possible, so checking it out first
        if (threads_x >= threads_y && threads_x >= threads_z)
            return threads_x;
        else if (threads_y >= threads_x && threads_y >= threads_z)
            return threads_y;
        else if (threads_z >= threads_x && threads_z >= threads_y)
            return threads_z;

        return 1;
    }

    int ThreadsHelper::pickLoop3d(int numThreads, uint64_t iters_x, uint64_t iters_y, uint64_t iters_z) {
        auto rem_x = iters_x % numThreads;
        auto rem_y = iters_y % numThreads;
        auto rem_z = iters_z % numThreads;

        auto split_x = iters_x / numThreads;
        auto split_y = iters_y / numThreads;
        auto split_z = iters_z / numThreads;

        // if there's no remainder left in some dimension - we're picking that dimension, because it'll be the most balanced work distribution
        if (rem_x == 0)
            return 1;
        else if (rem_y == 0)
            return 2;
        else if (rem_z == 0) // TODO: we don't want too smal splits over last dimension? or we do?
            return 3;

        if (iters_x > numThreads)
            return 1;
        else if (iters_y > numThreads)
            return 2;
        else if (iters_z > numThreads)
            return 3;

        return 1;
    }

    int Threads::parallel_tad(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, uint32_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_for got start > stop");

        auto delta = (stop - start);

        if (numThreads > delta)
            numThreads = delta;

        if (numThreads == 0)
            return 0;

        // shortcut
        if (numThreads == 1) {
            function(0, start, stop, increment);
            return 1;
        }


        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
        if (ticket != nullptr) {
            // if we got our threads - we'll run our jobs here
            auto span = delta / numThreads;

            for (uint32_t e = 0; e < numThreads; e++) {
                auto _start = span * e + start;
                auto _stop = span * (e + 1) + start;

                // last thread will process tail, and will process it in-place
                if (e == numThreads - 1) {
                    _stop = stop;
                    function(e, _start, _stop, increment);
                } else {
                    // putting the task into the queue for a given thread
                    ticket->enqueue(e, numThreads, function, _start, _stop, increment);
                }
            }

            // block and wait till all threads finished the job
            ticket->waitAndRelease();

            // we tell that parallelism request succeeded
            return numThreads;
        } else {
            // if there were no threads available - we'll execute function right within current thread
            function(0, start, stop, increment);

            // we tell that parallelism request declined
            return 1;
        }
    }

    int Threads::parallel_for(FUNC_1D function, int64_t start, int64_t stop, int64_t increment, uint32_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_for got start > stop");

        auto delta = (stop - start);

        // in some cases we just fire func as is
        if (delta == 0 || numThreads == 1) {
            function(0, start, stop, increment);
            return 1;
        }

        auto numElements = delta / increment;

        // we decide what's optimal number of threads we need here, and execute it in parallel_tad.
        numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
        return parallel_tad(function, start, stop, increment, numThreads);
    }

    int Threads::parallel_for(FUNC_2D function, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, uint64_t numThreads, bool debug) {
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
            if (ticket != nullptr) {
                //nd4j_printf("Threads: starting with %i threads\n", numThreads);
                for (int e = 0; e < numThreads; e++) {
                    auto span = Span2::build(splitLoop, e, numThreads, start_x, stop_x, inc_x, start_y, stop_y, inc_y);

                    ticket->enqueue(e, numThreads, function, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY());
                }

                // block until all threads finish their job
                ticket->waitAndRelease();
                return numThreads;
            } else {
                // if there were no threads available - we'll execute function right within current thread
                function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y);

                // we tell that parallelism request declined
                return 1;
            }
        };
    }


    int Threads::parallel_for(FUNC_3D function, int64_t start_x, int64_t stop_x, int64_t inc_x, int64_t start_y, int64_t stop_y, int64_t inc_y, int64_t start_z, int64_t stop_z, int64_t inc_z, uint64_t numThreads) {
        if (start_x > stop_x)
            throw std::runtime_error("Threads::parallel_for got start_x > stop_x");

        if (start_y > stop_y)
            throw std::runtime_error("Threads::parallel_for got start_y > stop_y");

        if (start_z > stop_z)
            throw std::runtime_error("Threads::parallel_for got start_z > stop_z");

        auto delta_x = stop_x - start_x;
        auto delta_y = stop_y - start_y;
        auto delta_z = stop_z - start_z;

        auto iters_x = delta_x / inc_x;
        auto iters_y = delta_y / inc_y;
        auto iters_z = delta_z / inc_z;

        numThreads = ThreadsHelper::numberOfThreads3d(numThreads, iters_x, iters_y, iters_z);
        if (numThreads == 1) {
            // loop is too small - executing function as is
            function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);
            return 1;
        }

        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads);
        if (ticket != nullptr) {
            auto splitLoop = ThreadsHelper::pickLoop3d(numThreads, iters_x, iters_y, iters_z);

            for (int e = 0; e < numThreads; e++) {
                auto span = Span3::build(splitLoop, e, numThreads, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);

                ticket->enqueue(e, numThreads, function, span.startX(), span.stopX(), span.incX(), span.startY(), span.stopY(), span.incY(), span.startZ(), span.stopZ(), span.incZ());
            }

            // block until we're done
            ticket->waitAndRelease();

            // we tell that parallelism request succeeded
            return numThreads;
        } else {
            // if there were no threads available - we'll execute function right within current thread
            function(0, start_x, stop_x, inc_x, start_y, stop_y, inc_y, start_z, stop_z, inc_z);

            // we tell that parallelism request declined
            return 1;
        }

    }

    int Threads::parallel_do(FUNC_DO function, uint64_t numThreads) {
        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
        if (ticket != nullptr) {

            // submit tasks one by one
            for (uint64_t e = 0; e < numThreads - 1; e++)
                ticket->enqueue(e, numThreads, function);

            function(numThreads - 1, numThreads);

            ticket->waitAndRelease();

            return numThreads;
        } else {
            // if there's no threads available - we'll execute function sequentially one by one
            for (uint64_t e = 0; e < numThreads; e++)
                function(e, numThreads);

            return numThreads;
        }


        return numThreads;
    }

    int64_t Threads::parallel_long(FUNC_RL function, FUNC_AL aggregator, int64_t start, int64_t stop, int64_t increment, uint64_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_long got start > stop");

        auto delta = (stop - start);
        if (delta == 0 || numThreads == 1)
            return function(0, start, stop, increment);

        auto numElements = delta / increment;

        // we decide what's optimal number of threads we need here, and execute it
        numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
        if (numThreads == 1)
            return function(0, start, stop, increment);

        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
        if (ticket == nullptr)
            return function(0, start, stop, increment);

        // create temporary array
        int64_t intermediatery[256];
        auto span = delta / numThreads;

        // execute threads in parallel
        for (uint32_t e = 0; e < numThreads; e++) {
            auto _start = span * e + start;
            auto _stop = span * (e + 1) + start;

            if (e == numThreads - 1)
                intermediatery[e] = function(e, _start, stop, increment);
            else
                ticket->enqueue(e, numThreads, &intermediatery[e], function, _start, _stop, increment);
        }

        ticket->waitAndRelease();

        // aggregate results in single thread
        for (uint64_t e = 1; e < numThreads; e++)
            intermediatery[0] = aggregator(intermediatery[0], intermediatery[e]);

        // return accumulated result
        return intermediatery[0];
    }

    double Threads::parallel_double(FUNC_RD function, FUNC_AD aggregator, int64_t start, int64_t stop, int64_t increment, uint64_t numThreads) {
        if (start > stop)
            throw std::runtime_error("Threads::parallel_long got start > stop");

        auto delta = (stop - start);
        if (delta == 0 || numThreads == 1)
            return function(0, start, stop, increment);

        auto numElements = delta / increment;

        // we decide what's optimal number of threads we need here, and execute it
        numThreads = ThreadsHelper::numberOfThreads(numThreads, numElements);
        if (numThreads == 1)
            return function(0, start, stop, increment);

        auto ticket = ThreadPool::getInstance()->tryAcquire(numThreads - 1);
        if (ticket == nullptr)
            return function(0, start, stop, increment);

        // create temporary array
        double intermediatery[256];
        auto span = delta / numThreads;

        // execute threads in parallel
        for (uint32_t e = 0; e < numThreads; e++) {
            auto _start = span * e + start;
            auto _stop = span * (e + 1) + start;

            if (e == numThreads - 1)
                intermediatery[e] = function(e, _start, stop, increment);
            else
                ticket->enqueue(e, numThreads, &intermediatery[e], function, _start, _stop, increment);
        }

        ticket->waitAndRelease();

        // aggregate results in single thread
        for (uint64_t e = 1; e < numThreads; e++)
            intermediatery[0] = aggregator(intermediatery[0], intermediatery[e]);

        // return accumulated result
        return intermediatery[0];
    }

}