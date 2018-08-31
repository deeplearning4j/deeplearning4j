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

namespace nd4j {
    namespace graph {

        GraphHolder* GraphHolder::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new GraphHolder();

            return _INSTANCE;
        };

        template <>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<float>* graph) {
            if (hasGraphAny(graphId))
                throw graph_exists_exception(graphId);

            _graphF[graphId] = graph;

            nd4j::SimpleReadWriteLock lock;
            _locks[graphId] = lock;
        }

        template <>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<float16>* graph) {
            if (hasGraphAny(graphId))
                throw graph_exists_exception(graphId);

            _graphH[graphId] = graph;

            nd4j::SimpleReadWriteLock lock;
            _locks[graphId] = lock;
        }

        template <>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<double>* graph) {
            if (hasGraphAny(graphId))
                throw graph_exists_exception(graphId);

            _graphD[graphId] = graph;

            nd4j::SimpleReadWriteLock lock;
            _locks[graphId] = lock;
        }
            
        template <>
        Graph<float>* GraphHolder::cloneGraph(Nd4jLong graphId) {
            if (!this->hasGraph<float>(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = _graphF[graphId]->cloneWithProxy();

            return graph;
        }

        template <>
        Graph<float>* GraphHolder::pullGraph(Nd4jLong graphId) {
            if (!this->hasGraph<float>(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = _graphF[graphId];

            return graph;
        }

        template <>
        Graph<float16>* GraphHolder::pullGraph(Nd4jLong graphId) {
            if (!this->hasGraph<float16>(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = _graphH[graphId];

            return graph;
        }

        template <>
        Graph<double>* GraphHolder::pullGraph(Nd4jLong graphId) {
            if (!this->hasGraph<double>(graphId)) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = _graphD[graphId];

            return graph;
        }

        template <typename T>
        void GraphHolder::forgetGraph(Nd4jLong graphId) {
            if (std::is_same<T, float>::value) {
                if (this->hasGraph<float>(graphId))
                    _graphF.erase(graphId);
            } else if (std::is_same<T, double>::value) {
                if (this->hasGraph<double>(graphId))
                    _graphD.erase(graphId);
            } else if (std::is_same<T, float16>::value) {
                if (this->hasGraph<float16>(graphId))
                    _graphH.erase(graphId);
            }
        }

        template <typename T>
        void GraphHolder::dropGraph(Nd4jLong graphId) {
            // FIXME: we don't want this sizeof(T) here really. especially once we add multi-dtype support
            if (std::is_same<T, float>::value) {
                if (this->hasGraph<float>(graphId)) {
                    auto g = _graphF[graphId];
                    forgetGraph<float>(graphId);
                    delete g;
                }
            } else if (std::is_same<T, double>::value) {
                if (this->hasGraph<double>(graphId)) {
                    auto g = _graphD[graphId];
                    forgetGraph<double>(graphId);
                    delete g;
                }
            } else if (std::is_same<T, float16>::value) {
                if (this->hasGraph<float16>(graphId)) {
                    auto g = _graphH[graphId];
                    forgetGraph<float16>(graphId);
                    delete g;
                }
            }
        }

        void GraphHolder::dropGraphAny(Nd4jLong graphId) {
            if (!hasGraphAny(graphId))
                return;

            this->lockWrite(graphId);

            this->dropGraph<float>(graphId);
            this->dropGraph<float16>(graphId);
            this->dropGraph<double>(graphId);

            this->unlockWrite(graphId);
        }

        bool GraphHolder::hasGraphAny(Nd4jLong graphId) {
            return this->hasGraph<float>(graphId) || this->hasGraph<double>(graphId) || this->hasGraph<float16>(graphId);
        }

        template<typename T>
        bool GraphHolder::hasGraph(Nd4jLong graphId) {
            if (std::is_same<T, float>::value) {
                return _graphF.count(graphId) > 0;
            } else if (std::is_same<T, double>::value) {
                return _graphD.count(graphId) > 0;
            } else if (std::is_same<T, float16>::value) {
                return _graphH.count(graphId) > 0;
            } else {
                nd4j_printf("Unsupported dtype was requested for GraphHolder::hasGraph","");
                return false;
            }
        }

        template <typename T>
        void GraphHolder::replaceGraph(Nd4jLong graphId, Graph<T>* graph) {
            if (!hasGraph<T>(graphId)) {
                registerGraph<T>(graphId, graph);
                return;
            }

            this->lockWrite(graphId);

            if (std::is_same<T, float>::value) {
                _graphF[graphId] = graph;
            } else if (std::is_same<T, double>::value) {
                _graphD[graphId] = graph;
            } else if (std::is_same<T, float16>::value) {
                _graphH[graphId] = graph;
            } else {
                nd4j_printf("Unsupported dtype was requested for GraphHolder::replaceGraph","");
            }

            this->unlockWrite(graphId);
        }


        flatbuffers::Offset<FlatResult> GraphHolder::execute(Nd4jLong graphId, flatbuffers::FlatBufferBuilder &builder, const FlatInferenceRequest* request) {
            if (!hasGraph<float>(graphId))
                throw unknown_graph_exception(graphId);

            return 0;
        }


        template bool GraphHolder::hasGraph<float>(Nd4jLong graphId);
        template bool GraphHolder::hasGraph<double>(Nd4jLong graphId);
        template bool GraphHolder::hasGraph<float16>(Nd4jLong graphId);

        template void GraphHolder::forgetGraph<float>(Nd4jLong graphId);
        template void GraphHolder::forgetGraph<float16>(Nd4jLong graphId);
        template void GraphHolder::forgetGraph<double>(Nd4jLong graphId);

        template void GraphHolder::dropGraph<float>(Nd4jLong graphId);


        GraphHolder* GraphHolder::_INSTANCE = 0;
    }
}