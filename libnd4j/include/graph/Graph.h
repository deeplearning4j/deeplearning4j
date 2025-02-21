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

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H
#include <algorithm>
#include <list>
#include <map>
#include <unordered_map>
#include <graph/ExecutorConfiguration.h>
#include <graph/Node.h>
#include <graph/Scope.h>
#include <graph/Stash.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/generated/config_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/node_generated.h>
#include <ops/declarable/OpDescriptor.h>

namespace sd {
namespace graph {

class SD_LIB_EXPORT Graph {
 protected:
  ExecutorConfiguration *_configuration;
  VariableSpace *_variableSpace;
  Stash *_stash;

  // this list holds references to Node ptrs, which should be free'd in Graph destructor
  std::vector<Node *> _handles;

  // vector holds ID's of top nodes only
  std::vector<int> *_nodes;
  SD_MAP_IMPL<int, Node *> *_mapped;

  SD_MAP_IMPL<int, std::vector<Node *> *> *_onion;
  SD_MAP_IMPL<int, Node *> _unmapped;
  std::vector<int> _unmappedMap;  // macOS?

  std::mutex _mutexPreprocessing;
  std::atomic<bool> _built;

  std::vector<int> _output;
  std::vector<int> _autos;

  SD_MAP_IMPL<int, Scope *> _mappedScopes;
  std::vector<Scope *> _scopes;

  void expandOnion(int newLayer);

  void injectNode(Node *node);

  void pushToOutputOnce(int id);

  void printOutNode(Node *node);

  void prepareOutputs();

 public:
  Graph(const ::graph::FlatGraph *flatGraph = nullptr, VariableSpace *variableSpace = nullptr);

  ~Graph();

  // this method applies toposort to nodes
  void toposortNodes();

  // method that'll print out graph
  Status validate();

  // this method will build structured representation of graph
  Status buildGraph();

  // this method will return estimated memory size (in bytes) required for 1 full graph execution round
  LongType estimateRequiredMemory();

  // this method returns number of root nodes in this graph
  int rootNodes();

  // this method returns total number of nodes in this graph
  int totalNodes();

  int numberOfPlaceholders();

  std::vector<Variable *> *getPlaceholders();

  /**
   * This method returns pointer to thread_local VariableSpace
   * @return
   */
  VariableSpace *getVariableSpace();

  /**
   * This method adds given node to the graph
   *
   * @param node
   */
  void addNode(Node *node);

  /**
   * This method returns layered representation of the graph
   *
   * @return
   */
  SD_MAP_IMPL<int, std::vector<Node *> *> *getOnion();

  /**
   * This method returns map of all nodes of the graph
   * @return
   */
  SD_MAP_IMPL<int, Node *> *getMapped();

  /**
   * This method returns outputs of this graph
   * @return
   */
  std::vector<Variable *> *fetchOutputs();

  /**
   * This method returns pointer to ExecutorConfiguration
   *
   * @return
   */
  ExecutorConfiguration *getExecutorConfiguration();

  /**
   * This method returns all nodes at once (order is NOT guaranteed)
   * @return
   */
  std::vector<Node *> *getAllNodes();

  /**
   * This method prints out Graph op-by-op, and respective inputs
   */
  void printOut();

  /**
   * This method returns OpScope ptr specified with id
   *
   * @param id
   * @return
   */
  Scope *scopeById(int id);

  /**
   * This method returns TRUE if specified ID refers to OpScope, and false otherwise
   * @param id
   * @return
   */
  bool hasScope(int id);

  /**
   * This method returns clone of the graph
   */
  Graph *clone();

  /**
   * This method returns clone of the graph, backed by VariableProxy instead of VariableSpace
   */
  Graph *cloneWithProxy();

  /**
   * This method removes reference to VariableSpace from this Graph
   */
  void forgetVariableSpace();

  /**
   * This method returns Node with given Id
   */
  Node *nodeById(int nodeId);

  /**
   * This method returns True if node with given ID exists, False otherwise
   * @param nodeId
   * @return
   */
  bool hasNode(int nodeId);

  /**
   * This method returns hash of given Graph instance
   */
  LongType hashCode();

  /**
   * PLEASE NOTE: This method will be moved to private section
   */
  void tagInplaceNodes();

  void replaceState(VariableSpace *state, ExecutorConfiguration *configuration);

  SD_INLINE std::vector<int> *nodes() { return _nodes; }


  SD_INLINE std::vector<int> *output() { return &_output; }


};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPH_H
