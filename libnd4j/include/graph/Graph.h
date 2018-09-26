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

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H

#include <list>
#include <algorithm>
#include <map>
//#include <NDArray.h>
#include <graph/Node.h>
#include <graph/Stash.h>
#include <graph/Scope.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/config_generated.h>
#include <graph/ExecutorConfiguration.h>

namespace nd4j {
    namespace graph {

        class ND4J_EXPORT Graph {
        protected:
            ExecutorConfiguration *_configuration;
            VariableSpace *_variableSpace;
            Stash* _stash;

            // this list holds references to Node ptrs, which should be free'd in Graph destructor
            std::vector<Node*> _handles;

            // vector holds ID's of top nodes only
            std::vector<int > *_nodes;
            std::map<int, nd4j::graph::Node*> *_mapped;

            std::map<int, std::vector<nd4j::graph::Node*> *> *_onion;
            std::map<int, nd4j::graph::Node*> _unmapped;
            std::vector<int> _unmappedMap; // macOS?

            std::mutex _mutexPreprocessing;
            std::atomic<bool> _built;

            std::vector<int> _output;
            std::vector<int> _autos;


            std::map<int, Scope*> _mappedScopes;
            std::vector<Scope*> _scopes;

////////////////////////////////////////
            Nd4jStatus validateNode(nd4j::graph::Node *node);

            void expandOnion(int newLayer);

            void injectNode(nd4j::graph::Node *node);

            void pushToOutputOnce(int id);

            void printOutNode(Node* node);

            void prepareOutputs();
        public:
            Graph(const FlatGraph *flatGraph = nullptr, VariableSpace *variableSpace = nullptr);

            ~Graph();

            // method that'll print out graph
            Nd4jStatus validate();

            // this method will build structured representation of graph
            Nd4jStatus buildGraph();

            // this method will return estimated memory size (in bytes) required for 1 full graph execution round
            Nd4jLong estimateRequiredMemory();

            // this method returns number of root nodes in this graph
            int rootNodes();

            // this method returns total number of nodes in this graph
            int totalNodes();

            int numberOfPlaceholders();

            std::vector<nd4j::graph::Variable*>* getPlaceholders();

            /**
             * This method returns pointer to thread_local VariableSpace
             * @return
             */
            nd4j::graph::VariableSpace *getVariableSpace();

            /**
             * This method adds given node to the graph
             *
             * @param node
             */
            void addNode(nd4j::graph::Node *node);

            /**
             * This method returns layered representation of the graph
             *
             * @return
             */
            std::map<int, std::vector<nd4j::graph::Node*> *> *getOnion();

            /**
             * This method returns map of all nodes of the graph
             * @return
             */
            std::map<int, nd4j::graph::Node*> *getMapped();

            /**
             * This method returns outputs of of this graph
             * @return
             */
            std::vector<nd4j::graph::Variable*> *fetchOutputs();

            /**
             * This method returns pointer to ExecutorConfiguration
             *
             * @return
             */
            ExecutorConfiguration *getExecutorConfiguration();

            /**
             * This method adds specified node (by ID) to de
             * @param id
             */
            void addOutput(int id);

            /**
             * This method returns all nodes at once (order is NOT guaranteed)
             * @return
             */
            std::vector<nd4j::graph::Node*> *getAllNodes();

            /**
             * This method prints out Graph op-by-op, and respective inputs
             */
            void printOut();

            /**
             * This method collect all ops from the graph into ops vector
             */
            std::vector<OpDescriptor> getOperations();

            /**
             * This method returns Scope ptr specified with id
             *
             * @param id
             * @return
             */
            Scope* scopeById(int id);

            /**
             * This method returns TRUE if specified ID refers to Scope, and false otherwise
             * @param id
             * @return
             */
            bool hasScope(int id);

            /**
             * This method returns clone of the graph
             */
            Graph* clone();

            /**
             * This method returns clone of the graph, backed by VariableProxy instead of VariableSpace
             */
            Graph* cloneWithProxy();

            /**
             * This method removes reference to VariableSpace from this Graph
             */
            void forgetVariableSpace();

            /**
             * This method returns Node with given Id
             */
            Node* nodeById(int nodeId);

            /**
             * This method returns True if node with given ID exists, False otherwise
             * @param nodeId
             * @return
             */
            bool hasNode(int nodeId);

            /**
             * This method returns hash of given Graph instance
             */
            Nd4jLong hashCode();

            /**
             * PLEASE NOTE: This method will be moved to private section
             */
            void tagInplaceNodes();

            void replaceState(VariableSpace *state, ExecutorConfiguration *configuration);

            FORCEINLINE std::vector<int>* nodes() {
                return _nodes;
            }

            FORCEINLINE std::vector<int>* autos() {
                return &_autos;
            }

            FORCEINLINE std::vector<int>* output() {
                return &_output;
            }

            FORCEINLINE std::map<int, Scope*>* scopes() {
                return &_mappedScopes;
            }

            FORCEINLINE bool built() {
                return _built.load();
            }

            FORCEINLINE void pullState(Graph *other) {
                for (int e = 0; e < other->nodes()->size(); e++)
                    this->_nodes->emplace_back(other->nodes()->at(e));

                for (int e = 0; e < other->output()->size(); e++)
                    this->_output.emplace_back(other->output()->at(e));
                
                for (int e = 0; e < other->autos()->size(); e++)
                    this->_autos.emplace_back(other->autos()->at(e));

                for (auto &v: *other->scopes()) {
                    auto scp = v.second->clone();
                    this->_mappedScopes[v.first] = scp;
                    this->_scopes.emplace_back(scp);
                }
                
                for (auto &v: *other->getOnion()) {
                    auto vec = this->_onion->count(v.first) > 0 ? this->_onion->at(v.first) : new std::vector<Node*>();

                    auto ovec = (*other->getOnion())[v.first];
                    for (auto x: *(ovec)) {
                        auto n = x->clone();
                        vec->emplace_back(n);
                        _handles.emplace_back(n);
                        (*this->_mapped)[n->id()] = n;
                    }

                    if (this->_onion->count(v.first) < 1)
                        (*this->_onion)[v.first] = vec;
                }

                this->_built.store(other->built());
            }
        };
    }
}

#endif //LIBND4J_GRAPH_H
