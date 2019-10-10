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

#include <execution/ThreadPool.h>
#include <stdexcept>
#include <helpers/logger.h>

namespace samediff {

    // this function executed once per thread, it polls functions from queue, and executes them via wrapper
    static void _executionLoop(int thread_id, BlockingQueue<CallableWithArguments*> *queue) {
        while (true) {
            // this method blocks until there's something within queue
            auto c = queue->poll();
            //nd4j_printf("ThreadPool: starting thread %i\n", c->threadId());
            switch (c->dimensions()) {
                case 0: {
                        c->function_do()(c->threadId(), c->numThreads());
                        c->finish();
                    }
                    break;
                case 1: {
                        auto args = c->arguments();
                        c->function_1d()(c->threadId(), args[0], args[1], args[2]);
                        c->finish();
                    }
                    break;
                case 2: {
                        auto args = c->arguments();
                        c->function_2d()(c->threadId(), args[0], args[1], args[2], args[3], args[4], args[5]);
                        c->finish();
                        //nd4j_printf("ThreadPool: finished thread %i\n", c->threadId());
                    }
                    break;
                case 3: {
                        auto args = c->arguments();
                        c->function_3d()(c->threadId(), args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
                        c->finish();
                    }
                    break;
                default:
                    throw std::runtime_error("Don't know what to do with provided Callable");
            }
        }
    }

    static void _executionLoopWithInterface(int thread_id, CallableInterface *c) {
        while (true) {
            // blocking here until there's something to do
            c->waitForTask();

            // execute whatever we have
            c->execute();
        }
    }

    ThreadPool::ThreadPool() {
        // TODO: number of threads must reflect number of cores for UMA system. In case of NUMA it should be per-device pool
        // FIXME: on mobile phones this feature must NOT be used
        _available = nd4j::Environment::getInstance()->maxThreads();

        _queues.resize(_available.load());
        _threads.resize(_available.load());
        _interfaces.resize(_available.load());

        // creating threads here
        for (int e = 0; e < _available.load(); e++) {
            _queues[e] = new BlockingQueue<CallableWithArguments*>(2);
            _interfaces[e] = new CallableInterface();
            _threads[e] = new std::thread(_executionLoopWithInterface, e, _interfaces[e]);
            _tickets.push(new Ticket());
            // _threads[e] = new std::thread(_executionLoop, e, _queues[e]);

            // TODO: add other platforms here as well
            // now we must set affinity, and it's going to be platform-specific thing
#ifdef LINUX_BUILD
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(e, &cpuset);
            int rc = pthread_setaffinity_np(_threads[e]->native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0)
                throw std::runtime_error("Failed to set pthread affinity");
#endif
        }
    }

    ThreadPool::~ThreadPool() {
        // TODO: implement this one properly
        for (int e = 0; e < _queues.size(); e++) {
            // stop each and every thread

            // release queue and thread
            //delete _queues[e];
            //delete _threads[e];
        }
    }

    static std::mutex _lmutex;

    ThreadPool* ThreadPool::getInstance() {
        std::unique_lock<std::mutex> lock(_lmutex);
        if (!_INSTANCE)
            _INSTANCE = new ThreadPool();

        return _INSTANCE;
    }

    void ThreadPool::release(int num_threads) {
        _available += num_threads;
    }

    Ticket* ThreadPool::tryAcquire(int num_threads) {
        //std::vector<BlockingQueue<CallableWithArguments*>*> queues;

        Ticket *t = nullptr;
        // we check for threads availability first
        bool threaded = false;
        {
            // we lock before checking availability
            std::unique_lock<std::mutex> lock(_lock);
            if (_available >= num_threads) {
                threaded = true;
                _available -= num_threads;

                // getting a ticket from the queue
                t = _tickets.front();
                _tickets.pop();

                // ticket must contain information about number of threads for the current session
                t->acquiredThreads(num_threads);

                // filling ticket with executable interfaces
                for (int e = 0, i = 0; e < _queues.size() && i < num_threads; e++) {
                    if (_interfaces[e]->available()) {
                        t->attach(i++, _interfaces[e]);
                        _interfaces[e]->markUnavailable();
                    }
                }
            }
        }

        // we either dispatch tasks to threads, or run single-threaded
        if (threaded) {
            return t;
        } else {
            // if there's no threads available - return nullptr
            return nullptr;
        }
    }

    void ThreadPool::release(samediff::Ticket *ticket) {
        // returning ticket back to the queue
        std::unique_lock<std::mutex> lock(_lock);
        _tickets.push(ticket);
    }


    ThreadPool* ThreadPool::_INSTANCE = 0;
}
