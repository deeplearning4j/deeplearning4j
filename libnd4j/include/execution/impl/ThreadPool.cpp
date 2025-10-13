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
#include <execution/ThreadPool.h>
#include <helpers/logger.h>

#include <stdexcept>


namespace samediff {

// this function executed once per thread, it polls functions from queue, and executes them via wrapper
static void executionLoop_(int thread_id, BlockingQueue<CallableWithArguments *> *queue) {
  while (true) {
    // this method blocks until there's something within queue
    auto c = queue->poll();
    switch (c->dimensions()) {
      case 0: {
        c->function_do()(c->threadId(), c->numThreads());
        c->finish();
      } break;
      case 1: {
        auto args = c->arguments();
        c->function_1d()(c->threadId(), args[0], args[1], args[2]);
        c->finish();
      } break;
      case 2: {
        auto args = c->arguments();
        c->function_2d()(c->threadId(), args[0], args[1], args[2], args[3], args[4], args[5]);
        c->finish();
      } break;
      case 3: {
        auto args = c->arguments();
        c->function_3d()(c->threadId(), args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                         args[8]);
        c->finish();
      } break;
      default:
        THROW_EXCEPTION("Don't know what to do with provided Callable");
    }
  }
}

static void executionLoopWithInterface_(int thread_id, CallableInterface *c) {
  while (true) {
    // blocking here until there's something to do
    c->waitForTask();

    // Check if shutdown was requested
    if (c->isShutdown()) {
      break;
    }

    // execute whatever we have
    c->execute();
  }
}

ThreadPool::ThreadPool() {
  // TODO: number of threads must reflect number of cores for UMA system. In case of NUMA it should be per-device pool
  // FIXME: on mobile phones this feature must NOT be used
  _available = sd::Environment::getInstance().maxThreads();

  _queues.resize(_available.load());
  _threads.resize(_available.load());
  _interfaces.resize(_available.load());

  // we're not creating threadpool on aurora
  // creating threads here
  for (int e = 0; e < _available.load(); e++) {
    _queues[e] = new BlockingQueue<CallableWithArguments *>(2);
    _interfaces[e] = new CallableInterface();
    _threads[e] = std::thread(executionLoopWithInterface_, e, _interfaces[e]);
    _tickets.push(new Ticket());
    // _threads[e] = new std::thread(executionLoop_, e, _queues[e]);

    // TODO: add other platforms here as well
    // now we must set affinity, and it's going to be platform-specific thing
#ifdef LINUX_BUILD
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(e, &cpuset);
    int rc = pthread_setaffinity_np(_threads[e].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) THROW_EXCEPTION("Failed to set pthread affinity");
#endif

  }
  //add an extra ticket to minimize the risk of running out of tickets due to race conditions
  _tickets.push(new Ticket());
}

ThreadPool::~ThreadPool() {
  // Signal all threads to shutdown BEFORE destroying anything
  for (size_t e = 0; e < _interfaces.size(); e++) {
    _interfaces[e]->shutdown();
  }

  // Wait for all threads to finish
  for (size_t e = 0; e < _threads.size(); e++) {
    if (_threads[e].joinable()) {
      _threads[e].join();
    }
  }

  // Now it's safe to delete resources
  for (size_t e = 0; e < _queues.size(); e++) {
    delete _queues[e];
    delete _interfaces[e];
  }

  while (!_tickets.empty()) {
    auto t = _tickets.front();
    _tickets.pop();
    delete t;
  }
}

ThreadPool &ThreadPool::getInstance() {
  static ThreadPool instance;
  return instance;
}

void ThreadPool::release(int numThreads) { _available += numThreads; }

Ticket *ThreadPool::tryAcquire(int numThreads) {
  if (numThreads <= 0) return nullptr;
  Ticket *t = nullptr;
  // we check for threads availability first
  bool threaded = false;
  {
    // we lock before checking availability
    std::unique_lock<std::mutex> lock(_lock);
    //test for both _available and _tickets in order to deal with race conditions caused by the
    //fact that marking threads as available AND releasing tickets does not happen atomically
    if (_available >= numThreads && !_tickets.empty()) {
      threaded = true;
      _available -= numThreads;

      // getting a ticket from the queue
      t = _tickets.front();
      _tickets.pop();

      // ticket must contain information about number of threads for the current session
      t->acquiredThreads(numThreads);

      // filling ticket with executable interfaces
      for (size_t e = 0, i = 0; e < _queues.size() && i < static_cast<size_t>(numThreads); e++) {
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
}  // namespace samediff
