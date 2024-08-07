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
#include <graph/SessionLocalStorage.h>
#include <graph/Stash.h>
#include <graph/VariableSpace.h>

namespace sd {
namespace graph {
SessionLocalStorage::SessionLocalStorage(VariableSpace* variableSpace, Stash* stash) {
  // we start from 1, since key 0 holds original VariableSpace
  _sessionCounter.store(1);
  _variableSpace = variableSpace;
  _stash = stash;
}

VariableSpace* SessionLocalStorage::localVariableSpace(LongType sessionId) {
  _mutex.lock();
  auto varSpace = _threadVariableSpace.at(sessionId);
  _mutex.unlock();

  return varSpace;
}

VariableSpace* SessionLocalStorage::localVariableSpace() { return localVariableSpace(getSessionId()); }

SessionLocalStorage::~SessionLocalStorage() {
  for (const auto& v : _threadVariableSpace) {
    delete v.second;
  }
}

LongType SessionLocalStorage::getThreadId() {
#ifdef __APPLE__
  // syscall?
#elif _WIN32
  // some win32api
#else
  // syscall!
#endif
  auto id = std::this_thread::get_id();
  uint64_t* ptr = (uint64_t*)&id;
  return (*ptr);
}

int SessionLocalStorage::numberOfSessions() {
  _mutex.lock();
  int size = (int)_threadSession.size();
  _mutex.unlock();
  return size;
}

void SessionLocalStorage::endSession(LongType sessionId) {
  // we should delete specific holders here
  _mutex.lock();
  auto vs = _threadVariableSpace[sessionId];
  _threadVariableSpace.erase(sessionId);

  delete vs;
  _mutex.unlock();
}

void SessionLocalStorage::endSession() {
  auto tid = getThreadId();

  _mutex.lock();

  auto ntid = _threadSession[tid];
  _threadSession.erase(tid);

  _mutex.unlock();

  endSession(ntid);
}

LongType SessionLocalStorage::getSessionId() {
  auto tid = getThreadId();

  _mutex.lock();
  auto ntid = _threadSession[tid];

  _mutex.unlock();

  return ntid;
}

LongType SessionLocalStorage::startSession() {
  auto tid = getThreadId();

  sd_debug("Adding ThreadId: %i;\n", (int)tid);
  LongType ntid = _sessionCounter++;
  _mutex.lock();

  _threadSession[tid] = ntid;
  _threadVariableSpace[ntid] = _variableSpace->clone();

  _mutex.unlock();

  return ntid;
}
}  // namespace graph
}  // namespace sd
