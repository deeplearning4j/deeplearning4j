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

#ifndef LIBND4J_SESSIONLOCALSTORAGE_H
#define LIBND4J_SESSIONLOCALSTORAGE_H
#include <memory/Workspace.h>

#include <map>
#include <thread>
#include <unordered_map>

#include "Context.h"
#include "Stash.h"
#include "VariableSpace.h"

namespace sd {
namespace graph {
class SD_LIB_EXPORT SessionLocalStorage {
 protected:
  std::atomic<sd::LongType> _sessionCounter;
  SD_MAP_IMPL<sd::LongType, sd::LongType> _threadSession;
  SD_MAP_IMPL<sd::LongType, VariableSpace*> _threadVariableSpace;

  VariableSpace* _variableSpace;
  Stash* _stash;

  std::mutex _mutex;

  sd::LongType getSessionId();
  sd::LongType getThreadId();

 public:
  SessionLocalStorage(VariableSpace* variableSpace = nullptr, Stash* stash = nullptr);

  ~SessionLocalStorage();

  VariableSpace* localVariableSpace();
  VariableSpace* localVariableSpace(sd::LongType sessionId);

  sd::LongType startSession();
  void endSession(sd::LongType sessionId);
  void endSession();

  int numberOfSessions();
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_SESSIONLOCALSTORAGE_H
