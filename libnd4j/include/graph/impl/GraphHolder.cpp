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

#include <graph/GraphHolder.h>
#include <GraphExecutioner.h>
#include <graph/exceptions/graph_exists_exception.h>
#include <graph/exceptions/graph_execution_exception.h>

namespace nd4j {
    namespace graph {
        GraphHolder* GraphHolder::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new GraphHolder();

            return _INSTANCE;
        };

        void GraphHolder::registerGraph(Nd4jLong graphId, Graph* graph) {
            if (hasGraphAny(graphId))
                throw graph_exists_exception(graphId);

            _graphF[graphId] = graph;

            nd4j::SimpleReadWriteLock lock;
            _locks[graphId] = lock;
        }

        Graph* GraphHolder::cloneGraph(Nd4jLong graphId) {
            if (!this->hasGraph(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = _graphF[graphId]->cloneWithProxy();

            return graph;
        }

        Graph* GraphHolder::pullGraph(Nd4jLong graphId) {
            if (!this->hasGraph(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = _graphF[graphId];

            return graph;
        }

        void GraphHolder::forgetGraph(Nd4jLong graphId) {
            if (this->hasGraph(graphId))
                _graphF.erase(graphId);
        }

        void GraphHolder::dropGraph(Nd4jLong graphId) {
            if (this->hasGraph(graphId)) {
                auto g = _graphF[graphId];
                forgetGraph(graphId);
                delete g;
            }
        }

        void GraphHolder::dropGraphAny(Nd4jLong graphId) {
            if (!hasGraphAny(graphId))
                return;

            this->lockWrite(graphId);

            this->dropGraph(graphId);

            this->unlockWrite(graphId);
        }

        bool GraphHolder::hasGraphAny(Nd4jLong graphId) {
            return this->hasGraph(graphId);
        }

        bool GraphHolder::hasGraph(Nd4jLong graphId) {
                return _graphF.count(graphId) > 0;
        }

        void GraphHolder::replaceGraph(Nd4jLong graphId, Graph* graph) {
            if (!hasGraph(graphId)) {
                registerGraph(graphId, graph);
                return;
            }

            this->lockWrite(graphId);

            _graphF[graphId] = graph;

            this->unlockWrite(graphId);
        }




        flatbuffers::Offset<FlatResult> GraphHolder::execute(Nd4jLong graphId, flatbuffers::FlatBufferBuilder &builder, const FlatInferenceRequest* request) {
            if (!hasGraph(graphId))
                throw unknown_graph_exception(graphId);

            lockRead(graphId);

            auto graph = cloneGraph(graphId);
            auto res = GraphExecutioner::execute(graph, builder, request);
            delete graph;

            unlockRead(graphId);

            return res;
        }

        GraphHolder* GraphHolder::_INSTANCE = 0;
    }
}
