/* ******************************************************************************
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
// @author Adam Gibson
//
// Implementation of convex subgraph analysis for mega-kernel fusion.
//

#include <graph/analysis/ConvexSubgraphAnalyzer.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <algorithm>
#include <queue>
#include <stack>

namespace sd {
namespace graph {

struct ConvexSubgraphAnalyzer::AnalyzerState {
    std::vector<int> nodeOrder;
    std::unordered_map<int, std::unordered_set<int>> predecessors;
    std::unordered_map<int, std::unordered_set<int>> successors;
    bool closureComputed = false;
};

ConvexSubgraphAnalyzer::ConvexSubgraphAnalyzer()
    : state_(std::make_unique<AnalyzerState>()),
      minSubgraphSize_(2),
      maxSubgraphSize_(100),
      minBenefitThreshold_(0.1f) {
    seedOpTypes_ = getDefaultSeedOpTypes();
}

ConvexSubgraphAnalyzer::~ConvexSubgraphAnalyzer() = default;

void ConvexSubgraphAnalyzer::setSeedOpTypes(const std::unordered_set<std::string>& opTypes) {
    seedOpTypes_ = opTypes;
}

void ConvexSubgraphAnalyzer::setMinSubgraphSize(int minSize) {
    minSubgraphSize_ = minSize;
}

void ConvexSubgraphAnalyzer::setMaxSubgraphSize(int maxSize) {
    maxSubgraphSize_ = maxSize;
}

void ConvexSubgraphAnalyzer::setMinBenefitThreshold(float threshold) {
    minBenefitThreshold_ = threshold;
}

// Helper function to get operation name from a Node
static std::string getNodeOpName(Node* node) {
    if (node == nullptr) return "";

    // Try to get name from custom op if available
    if (node->hasCustomOp() && node->getCustomOp() != nullptr) {
        std::string* opName = node->getCustomOp()->getOpName();
        if (opName != nullptr) {
            return *opName;
        }
    }

    // Fall back to node name
    std::string* nodeName = node->name();
    if (nodeName != nullptr && !nodeName->empty()) {
        return *nodeName;
    }

    // Return string based on opNum
    return "op_" + std::to_string(node->opNum());
}

void ConvexSubgraphAnalyzer::computeTransitiveClosure(Graph* graph) {
    reachableFrom_.clear();
    reachableTo_.clear();

    // Build adjacency lists from graph structure using getAllNodes
    std::vector<Node*>* allNodes = graph->getAllNodes();
    if (allNodes == nullptr) return;

    for (Node* node : *allNodes) {
        if (node == nullptr) continue;
        int nodeId = node->id();

        // Get inputs (predecessors)
        auto* inputs = node->input();
        if (inputs != nullptr) {
            for (size_t j = 0; j < inputs->size(); j++) {
                auto& input = inputs->at(j);
                int predId = input.first;
                state_->predecessors[nodeId].insert(predId);
                state_->successors[predId].insert(nodeId);
            }
        }
    }

    // Compute transitive closure using BFS from each node
    for (Node* node : *allNodes) {
        if (node == nullptr) continue;
        int nodeId = node->id();

        // BFS from this node
        std::queue<int> q;
        std::unordered_set<int> visited;
        q.push(nodeId);
        visited.insert(nodeId);

        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            auto it = state_->successors.find(curr);
            if (it != state_->successors.end()) {
                for (int succ : it->second) {
                    if (visited.find(succ) == visited.end()) {
                        visited.insert(succ);
                        q.push(succ);
                        reachableFrom_[nodeId].insert(succ);
                        reachableTo_[succ].insert(nodeId);
                    }
                }
            }
        }
    }

    state_->closureComputed = true;
}

bool ConvexSubgraphAnalyzer::canReach(int nodeA, int nodeB) const {
    auto it = reachableFrom_.find(nodeA);
    if (it == reachableFrom_.end()) return false;
    return it->second.find(nodeB) != it->second.end();
}

std::vector<int> ConvexSubgraphAnalyzer::findSeedNodes(Graph* graph) {
    std::vector<int> seeds;
    auto excludedOps = getExcludedOpTypes();

    std::vector<Node*>* allNodes = graph->getAllNodes();
    if (allNodes == nullptr) return seeds;

    for (Node* node : *allNodes) {
        if (node == nullptr) continue;

        std::string opName = getNodeOpName(node);

        // Check if this is a seed operation
        if (seedOpTypes_.find(opName) != seedOpTypes_.end() &&
            excludedOps.find(opName) == excludedOps.end()) {
            seeds.push_back(node->id());
        }
    }

    return seeds;
}

bool ConvexSubgraphAnalyzer::wouldMaintainConvexity(Graph* graph, const ConvexSubgraph& subgraph, int nodeId) const {
    // A subgraph remains convex if adding nodeId doesn't create a path
    // from any node inside to another node inside that goes through an outside node

    // Check 1: For all nodes in subgraph, if nodeId can reach them,
    // there should be a direct edge or a path entirely within the subgraph
    for (int existingNode : subgraph.nodeSet) {
        if (canReach(nodeId, existingNode)) {
            // There's a path from nodeId to existingNode
            // Check if all intermediate nodes are in the subgraph
            auto it = state_->successors.find(nodeId);
            bool directEdge = (it != state_->successors.end()) &&
                              (it->second.find(existingNode) != it->second.end());
            if (!directEdge) {
                // Non-direct path - need to check if intermediate nodes are in subgraph
                // For simplicity, require direct edge for now
                return false;
            }
        }

        if (canReach(existingNode, nodeId)) {
            auto it = state_->successors.find(existingNode);
            bool directEdge = (it != state_->successors.end()) &&
                              (it->second.find(nodeId) != it->second.end());
            if (!directEdge) {
                return false;
            }
        }
    }

    return true;
}

ConvexSubgraph ConvexSubgraphAnalyzer::growSubgraph(Graph* graph, int seedNode) {
    ConvexSubgraph subgraph;
    subgraph.nodeSet.insert(seedNode);
    subgraph.nodeIds.push_back(seedNode);

    auto excludedOps = getExcludedOpTypes();
    bool changed = true;

    while (changed && static_cast<int>(subgraph.nodeSet.size()) < maxSubgraphSize_) {
        changed = false;

        // Make a copy of nodeSet for iteration since we're modifying it
        std::unordered_set<int> currentNodes = subgraph.nodeSet;

        // Try to add predecessors
        for (int nodeId : currentNodes) {
            auto it = state_->predecessors.find(nodeId);
            if (it == state_->predecessors.end()) continue;

            for (int pred : it->second) {
                if (subgraph.nodeSet.find(pred) != subgraph.nodeSet.end()) continue;

                // Check if predecessor can be added
                Node* predNode = graph->nodeById(pred);
                if (predNode == nullptr) continue;

                std::string opName = getNodeOpName(predNode);
                if (excludedOps.find(opName) != excludedOps.end()) continue;

                if (wouldMaintainConvexity(graph, subgraph, pred)) {
                    subgraph.nodeSet.insert(pred);
                    changed = true;
                }
            }
        }

        // Try to add successors
        for (int nodeId : currentNodes) {
            auto it = state_->successors.find(nodeId);
            if (it == state_->successors.end()) continue;

            for (int succ : it->second) {
                if (subgraph.nodeSet.find(succ) != subgraph.nodeSet.end()) continue;

                Node* succNode = graph->nodeById(succ);
                if (succNode == nullptr) continue;

                std::string opName = getNodeOpName(succNode);
                if (excludedOps.find(opName) != excludedOps.end()) continue;

                if (wouldMaintainConvexity(graph, subgraph, succ)) {
                    subgraph.nodeSet.insert(succ);
                    changed = true;
                }
            }
        }
    }

    // Sort nodes in topological order
    subgraph.nodeIds = topologicalSort(graph, subgraph.nodeSet);

    // Identify boundary
    identifyBoundary(graph, subgraph);

    // Estimate benefit
    subgraph.estimatedBenefit = estimateBenefit(graph, subgraph);

    return subgraph;
}

std::vector<int> ConvexSubgraphAnalyzer::topologicalSort(Graph* graph, const std::unordered_set<int>& nodes) {
    std::vector<int> result;
    std::unordered_set<int> visited;
    std::stack<int> stack;

    std::function<void(int)> dfs = [&](int node) {
        if (visited.find(node) != visited.end()) return;
        visited.insert(node);

        auto it = state_->successors.find(node);
        if (it != state_->successors.end()) {
            for (int succ : it->second) {
                if (nodes.find(succ) != nodes.end()) {
                    dfs(succ);
                }
            }
        }

        stack.push(node);
    };

    for (int node : nodes) {
        dfs(node);
    }

    while (!stack.empty()) {
        result.push_back(stack.top());
        stack.pop();
    }

    return result;
}

void ConvexSubgraphAnalyzer::identifyBoundary(Graph* graph, ConvexSubgraph& subgraph) {
    subgraph.inputs.clear();
    subgraph.outputs.clear();

    for (int nodeId : subgraph.nodeSet) {
        // Check inputs
        auto it = state_->predecessors.find(nodeId);
        if (it != state_->predecessors.end()) {
            for (int pred : it->second) {
                if (subgraph.nodeSet.find(pred) == subgraph.nodeSet.end()) {
                    // External input
                    if (std::find(subgraph.inputs.begin(), subgraph.inputs.end(), pred) == subgraph.inputs.end()) {
                        subgraph.inputs.push_back(pred);
                    }
                }
            }
        }

        // Check outputs
        auto succIt = state_->successors.find(nodeId);
        if (succIt != state_->successors.end()) {
            for (int succ : succIt->second) {
                if (subgraph.nodeSet.find(succ) == subgraph.nodeSet.end()) {
                    // External output
                    if (std::find(subgraph.outputs.begin(), subgraph.outputs.end(), nodeId) == subgraph.outputs.end()) {
                        subgraph.outputs.push_back(nodeId);
                    }
                }
            }
        }
    }
}

float ConvexSubgraphAnalyzer::estimateBenefit(Graph* graph, const ConvexSubgraph& subgraph) {
    // Simple heuristic: benefit proportional to number of fused ops
    // minus overhead for boundary crossings

    float benefit = static_cast<float>(subgraph.nodeSet.size()) * 0.5f;

    // Penalty for boundary crossings
    benefit -= static_cast<float>(subgraph.inputs.size()) * 0.1f;
    benefit -= static_cast<float>(subgraph.outputs.size()) * 0.1f;

    // Bonus for containing seed ops
    int seedCount = 0;
    for (int nodeId : subgraph.nodeSet) {
        Node* node = graph->nodeById(nodeId);
        if (node != nullptr) {
            std::string opName = getNodeOpName(node);
            if (seedOpTypes_.find(opName) != seedOpTypes_.end()) {
                seedCount++;
            }
        }
    }
    benefit += static_cast<float>(seedCount) * 0.3f;

    return benefit;
}

std::vector<ConvexSubgraph> ConvexSubgraphAnalyzer::analyze(Graph* graph) {
    if (!state_->closureComputed) {
        computeTransitiveClosure(graph);
    }

    std::vector<ConvexSubgraph> subgraphs;
    std::unordered_set<int> usedNodes;

    // Find seed nodes
    std::vector<int> seeds = findSeedNodes(graph);

    // Grow subgraph from each seed
    for (int seed : seeds) {
        if (usedNodes.find(seed) != usedNodes.end()) continue;

        ConvexSubgraph subgraph = growSubgraph(graph, seed);

        // Check minimum size and benefit thresholds
        if (static_cast<int>(subgraph.nodeSet.size()) >= minSubgraphSize_ &&
            subgraph.estimatedBenefit >= minBenefitThreshold_) {

            // Mark nodes as used
            for (int nodeId : subgraph.nodeSet) {
                usedNodes.insert(nodeId);
            }

            subgraphs.push_back(std::move(subgraph));
        }
    }

    // Sort by estimated benefit (descending)
    std::sort(subgraphs.begin(), subgraphs.end(),
              [](const ConvexSubgraph& a, const ConvexSubgraph& b) {
                  return a.estimatedBenefit > b.estimatedBenefit;
              });

    return subgraphs;
}

}  // namespace graph
}  // namespace sd
