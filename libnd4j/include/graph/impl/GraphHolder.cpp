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

namespace nd4j {
    namespace graph {

        GraphHolder* GraphHolder::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new GraphHolder();

            return _INSTANCE;
        };

        template <>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<float>* graph) {
            _graphF[graphId] = graph;
        }

        template <>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<float16>* graph) {
            _graphH[graphId] = graph;
        }

        template <>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<double>* graph) {
            _graphD[graphId] = graph;
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
            if (sizeof(T) == 4) {
                if (this->hasGraph<float>(graphId))
                    _graphF.erase(graphId);
            } else if (sizeof(T) == 2) {
                if (this->hasGraph<double>(graphId))
                    _graphD.erase(graphId);
            } else if (sizeof(T) == 8) {
                if (this->hasGraph<float16>(graphId))
                    _graphH.erase(graphId);
            }
        }

        template <typename T>
        void GraphHolder::dropGraph(Nd4jLong graphId) {
            // FIXME: we don't want this sizeof(T) here really. especially once we add multi-dtype support
            if (sizeof(T) == 4) {
                if (this->hasGraph<float>(graphId)) {
                    auto g = _graphF[graphId];
                    forgetGraph<float>(graphId);
                    delete g;
                }
            } else if (sizeof(T) == 8) {
                if (this->hasGraph<double>(graphId)) {
                    auto g = _graphD[graphId];
                    forgetGraph<double>(graphId);
                    delete g;
                }
            } else if (sizeof(T) == 2) {
                if (this->hasGraph<float16>(graphId)) {
                    auto g = _graphH[graphId];
                    forgetGraph<float16>(graphId);
                    delete g;
                }
            }
        }

        void GraphHolder::dropGraphAny(Nd4jLong graphId) {
            this->dropGraph<float>(graphId);
            this->dropGraph<float16>(graphId);
            this->dropGraph<double>(graphId);
        }

        template<typename T>
        bool GraphHolder::hasGraph(Nd4jLong graphId) {
            return _graphF.count(graphId) > 0;
        }


        template bool GraphHolder::hasGraph<float>(Nd4jLong graphId);
        template void GraphHolder::forgetGraph<float>(Nd4jLong graphId);
        template void GraphHolder::dropGraph<float>(Nd4jLong graphId);


        GraphHolder* GraphHolder::_INSTANCE = 0;
    }
}