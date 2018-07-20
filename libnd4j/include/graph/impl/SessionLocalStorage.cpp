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

#include <VariableSpace.h>
#include <Stash.h>
#include <graph/SessionLocalStorage.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        SessionLocalStorage<T>::SessionLocalStorage(VariableSpace<T>* variableSpace, Stash<T>* stash) {
            // we start from 1, since key 0 holds original VariableSpace
            _sessionCounter.store(1);
            _variableSpace = variableSpace;
            _stash = stash;
        }

        template <typename T>
        VariableSpace<T>* SessionLocalStorage<T>::localVariableSpace(Nd4jLong sessionId) {
            _mutex.lock();
            auto varSpace = _threadVariableSpace.at(sessionId);
            _mutex.unlock();

            return varSpace;
        }

        template <typename T>
        VariableSpace<T>* SessionLocalStorage<T>::localVariableSpace() {
            return localVariableSpace(getSessionId());
        }

        template <typename T>
        SessionLocalStorage<T>::~SessionLocalStorage() {
            for (const auto & v: _threadVariableSpace) {
                delete v.second;
            }
        }


        template <typename T>
        Nd4jLong SessionLocalStorage<T>::getThreadId() {
#ifdef __APPLE__
            // syscall?
#elif _WIN32
            // some win32api
#else
    // syscall!
#endif
            auto id=std::this_thread::get_id();
            uint64_t* ptr=(uint64_t*) &id;
            return (*ptr);
        }

        template <typename T>
        int SessionLocalStorage<T>::numberOfSessions() {
            _mutex.lock();
            int size = (int) _threadSession.size();
            _mutex.unlock();
            return size;
        }

        template <typename T>
        void SessionLocalStorage<T>::endSession(Nd4jLong sessionId) {
            // we should delete specific holders here
            _mutex.lock();
            auto vs = _threadVariableSpace[sessionId];
            _threadVariableSpace.erase(sessionId);

            delete vs;
            _mutex.unlock();
        }

        template <typename T>
        void SessionLocalStorage<T>::endSession() {
            auto tid = getThreadId();

            _mutex.lock();

            auto ntid = _threadSession[tid];
            _threadSession.erase(tid);

            _mutex.unlock();

            endSession(ntid);
        }

        template <typename T>
        Nd4jLong SessionLocalStorage<T>::getSessionId() {
            auto tid = getThreadId();

            _mutex.lock();
            auto ntid = _threadSession[tid];

            _mutex.unlock();

            return ntid;
        }

        template <typename T>
        Nd4jLong nd4j::graph::SessionLocalStorage<T>::startSession() {
            auto tid = getThreadId();

            nd4j_debug("Adding ThreadId: %i;\n", (int) tid);
            Nd4jLong ntid = _sessionCounter++;
            _mutex.lock();

            _threadSession[tid] = ntid;
            _threadVariableSpace[ntid] = _variableSpace->clone();

            _mutex.unlock();

            return ntid;
        }

        template class ND4J_EXPORT SessionLocalStorage<float>;
        template class ND4J_EXPORT SessionLocalStorage<float16>;
        template class ND4J_EXPORT SessionLocalStorage<double>;
    }
}

