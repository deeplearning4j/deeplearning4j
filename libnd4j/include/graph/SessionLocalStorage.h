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

#ifndef LIBND4J_SESSIONLOCALSTORAGE_H
#define LIBND4J_SESSIONLOCALSTORAGE_H

#include <thread>
#include <unordered_map>
#include <map>
#include "VariableSpace.h"
#include "Context.h"
#include "Stash.h"
#include <memory/Workspace.h>

namespace sd{
    namespace graph {
        class ND4J_EXPORT SessionLocalStorage {
        protected:
            std::atomic<Nd4jLong> _sessionCounter;
            MAP_IMPL<Nd4jLong, Nd4jLong> _threadSession;
            MAP_IMPL<Nd4jLong, VariableSpace*> _threadVariableSpace;

            VariableSpace* _variableSpace;
            Stash* _stash;

            std::mutex _mutex;

            Nd4jLong getSessionId();
            Nd4jLong getThreadId();
        public:
            SessionLocalStorage(VariableSpace* variableSpace = nullptr, Stash* stash = nullptr);

            ~SessionLocalStorage();

            VariableSpace* localVariableSpace();
            VariableSpace* localVariableSpace(Nd4jLong sessionId);


            Nd4jLong startSession();
            void endSession(Nd4jLong sessionId);
            void endSession();

            int numberOfSessions();
        };
    }
}

#endif //LIBND4J_SESSIONLOCALSTORAGE_H
