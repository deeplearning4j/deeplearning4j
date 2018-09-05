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
//  @author raver119@gmail.com
//

#include <helpers/logger.h>
#include <pointercast.h>
#include <map>
#include <graph/Graph.h>
#include <helpers/SimpleReadWriteLock.h>
#include <graph/exceptions/unknown_graph_exception.h>

namespace nd4j {
    namespace graph {
        class GraphHolder {
        private:
            static GraphHolder *_INSTANCE;
            std::map<Nd4jLong, Graph *> _graphF;

            std::map<Nd4jLong, SimpleReadWriteLock> _locks;

            GraphHolder() = default;
            ~GraphHolder() = default;
        public:
            static GraphHolder* getInstance();

            void registerGraph(Nd4jLong graphId, Graph *graph);
            
            Graph* cloneGraph(Nd4jLong graphId);

            Graph* pullGraph(Nd4jLong graphId);

            void forgetGraph(Nd4jLong graphId);

            void dropGraph(Nd4jLong graphId);

            void dropGraphAny(Nd4jLong graphId);

            bool hasGraph(Nd4jLong graphId);

            bool hasGraphAny(Nd4jLong graphId);

            flatbuffers::Offset<FlatResult> execute(Nd4jLong graphId, flatbuffers::FlatBufferBuilder &builder, const FlatInferenceRequest* request);

            void replaceGraph(Nd4jLong graphId, Graph *graph);

            /////////////////////////////

            FORCEINLINE void lockWrite(Nd4jLong graphId) {
                if (_locks.count(graphId) == 0)
                    return;

                _locks[graphId].lockWrite();
            }

            FORCEINLINE void unlockWrite(Nd4jLong graphId) {
                if (_locks.count(graphId) == 0)
                    return;

                _locks[graphId].unlockWrite();
            }

            FORCEINLINE void lockRead(Nd4jLong graphId) {
                if (_locks.count(graphId) == 0)
                    return;

                _locks[graphId].lockRead();
            }

            FORCEINLINE void unlockRead(Nd4jLong graphId) {
                if (_locks.count(graphId) == 0)
                    return;

                _locks[graphId].unlockRead();
            }
        };
    }
}