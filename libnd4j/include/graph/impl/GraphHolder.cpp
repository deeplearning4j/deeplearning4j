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

        template <typename T>
        void GraphHolder::registerGraph(Nd4jLong graphId, Graph<T>* graph) {
            std::map<Nd4jLong, Graph<T>*> &graphmap = getGraphMap<T>();
            graphmap[graphId] = graph;
        }

        template <>
        std::map<Nd4jLong, Graph<float>*>& GraphHolder::getGraphMap() {
            std::map<Nd4jLong, Graph<float>*> &mapref = _graphF;
            return mapref;
        }

        template <>
        std::map<Nd4jLong, Graph<float16>*>& GraphHolder::getGraphMap() {
            std::map<Nd4jLong, Graph<float16>*> &mapref = _graphH;
            return mapref;
        }

        template <>
        std::map<Nd4jLong, Graph<double>*>& GraphHolder::getGraphMap() {
            std::map<Nd4jLong, Graph<double>*> &mapref = _graphD;
            return mapref;
        }

        template <>
        std::map<Nd4jLong, Graph<int>*>& GraphHolder::getGraphMap() {
            std::map<Nd4jLong, Graph<int>*> &mapref = _graphI;
            return mapref;
        }

        template <>
        std::map<Nd4jLong, Graph<Nd4jLong>*>& GraphHolder::getGraphMap() {
            std::map<Nd4jLong, Graph<Nd4jLong>*> &mapref = _graphL;
            return mapref;
        }

        template <typename T>
        Graph<T>* GraphHolder::cloneGraph(Nd4jLong graphId) {
            std::map<Nd4jLong, Graph<T>*> &graphmap = getGraphMap<T>();
            auto it = graphmap.find(graphId);

            if (it == graphmap.end()) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = it->second->clone();
            return graph;
        }

        template <typename T>
        Graph<T>* GraphHolder::pullGraph(Nd4jLong graphId) {
            std::map<Nd4jLong, Graph<T>*> &graphmap = getGraphMap<T>();
            auto it = graphmap.find(graphId);

            if (it == graphmap.end()) {
                nd4j_printf("GraphHolder doesn't have graph stored for [%lld]\n", graphId);
                throw std::runtime_error("Bad argument");
            }

            auto graph = it->second;
            return graph;
        }

        template <typename T>
        void GraphHolder::forgetGraph(Nd4jLong graphId) {
            std::map<Nd4jLong, Graph<T>*> &graphmap = getGraphMap<T>();
            graphmap.erase(graphId);
        }

        template <typename T>
        void GraphHolder::dropGraph(Nd4jLong graphId) {
            std::map<Nd4jLong, Graph<T>*> &graphmap = getGraphMap<T>();
            auto it = graphmap.find(graphId);

            if (it == graphmap.end()) {
                return;
            }
            auto graph = it->second;
            graphmap.erase(it);
            delete graph;
        }

        void GraphHolder::dropGraphAny(Nd4jLong graphId) {
            this->dropGraph<float>(graphId);
            this->dropGraph<float16>(graphId);
            this->dropGraph<double>(graphId);
            this->dropGraph<int>(graphId);
            this->dropGraph<Nd4jLong>(graphId);
        }

        template<typename T>
        bool GraphHolder::hasGraph(Nd4jLong graphId) {
            return getGraphMap<T>().count(graphId) > 0;
        }

        GraphHolder* GraphHolder::_INSTANCE = 0;

        template void GraphHolder::registerGraph<float>(Nd4jLong, Graph<float>*);
        template void GraphHolder::registerGraph<float16>(Nd4jLong, Graph<float16>*);
        template void GraphHolder::registerGraph<double>(Nd4jLong, Graph<double>*);
        template void GraphHolder::registerGraph<int>(Nd4jLong, Graph<int>*);
        template void GraphHolder::registerGraph<Nd4jLong>(Nd4jLong, Graph<Nd4jLong>*);

        template bool GraphHolder::hasGraph<float>(Nd4jLong);
        template bool GraphHolder::hasGraph<float16>(Nd4jLong);
        template bool GraphHolder::hasGraph<double>(Nd4jLong);
        template bool GraphHolder::hasGraph<int>(Nd4jLong);
        template bool GraphHolder::hasGraph<Nd4jLong>(Nd4jLong);

        template Graph<float>* GraphHolder::cloneGraph<float>(Nd4jLong);
        template Graph<float16>* GraphHolder::cloneGraph<float16>(Nd4jLong);
        template Graph<double>* GraphHolder::cloneGraph<double>(Nd4jLong);
        template Graph<int>* GraphHolder::cloneGraph<int>(Nd4jLong);
        template Graph<Nd4jLong>* GraphHolder::cloneGraph<Nd4jLong>(Nd4jLong);

        template Graph<float>* GraphHolder::pullGraph<float>(Nd4jLong);
        template Graph<float16>* GraphHolder::pullGraph<float16>(Nd4jLong);
        template Graph<double>* GraphHolder::pullGraph<double>(Nd4jLong);
        template Graph<int>* GraphHolder::pullGraph<int>(Nd4jLong);
        template Graph<Nd4jLong>* GraphHolder::pullGraph<Nd4jLong>(Nd4jLong);

        template void GraphHolder::forgetGraph<float>(Nd4jLong);
        template void GraphHolder::forgetGraph<float16>(Nd4jLong);
        template void GraphHolder::forgetGraph<double>(Nd4jLong);
        template void GraphHolder::forgetGraph<int>(Nd4jLong);
        template void GraphHolder::forgetGraph<Nd4jLong>(Nd4jLong);       

        template void GraphHolder::dropGraph<float>(Nd4jLong);
        template void GraphHolder::dropGraph<float16>(Nd4jLong);
        template void GraphHolder::dropGraph<double>(Nd4jLong);
        template void GraphHolder::dropGraph<int>(Nd4jLong);
        template void GraphHolder::dropGraph<Nd4jLong>(Nd4jLong);                   
    }
}