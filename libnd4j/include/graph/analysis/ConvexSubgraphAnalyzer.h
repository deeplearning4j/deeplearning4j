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
// Convex subgraph analysis for mega-kernel fusion.
//

#ifndef LIBND4J_CONVEX_SUBGRAPH_ANALYZER_H
#define LIBND4J_CONVEX_SUBGRAPH_ANALYZER_H

#include <graph/Graph.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>

namespace sd {
namespace graph {

/**
 * A convex subgraph is a set of nodes where there are no paths from
 * a node inside the subgraph to another node inside the subgraph that
 * pass through a node outside the subgraph.
 *
 * This property is essential for fusion: we can execute the entire
 * subgraph as a single kernel without needing intermediate memory
 * allocation for values that don't cross the subgraph boundary.
 */
struct ConvexSubgraph {
    std::vector<int> nodeIds;           // Nodes in topological order
    std::unordered_set<int> nodeSet;    // For O(1) membership testing
    std::vector<int> inputs;            // External inputs to the subgraph
    std::vector<int> outputs;           // Outputs consumed outside the subgraph
    float estimatedBenefit;             // Estimated performance gain from fusion
};

/**
 * Analyzes a computation graph to identify convex subgraphs suitable
 * for mega-kernel fusion.
 *
 * The algorithm:
 * 1. Build dependency graph from the Graph's layer structure (_onion)
 * 2. Compute transitive closure for reachability queries
 * 3. Identify seed nodes (operations that benefit from fusion)
 * 4. Grow convex regions greedily while maintaining convexity
 * 5. Rank subgraphs by estimated fusion benefit
 */
class ConvexSubgraphAnalyzer {
public:
    ConvexSubgraphAnalyzer();
    ~ConvexSubgraphAnalyzer();

    /**
     * Set of operation types that are good candidates for fusion.
     */
    void setSeedOpTypes(const std::unordered_set<std::string>& opTypes);

    /**
     * Set the minimum subgraph size to consider for fusion.
     * Smaller subgraphs may not provide enough benefit.
     */
    void setMinSubgraphSize(int minSize);

    /**
     * Set the maximum subgraph size.
     * Larger subgraphs may have diminishing returns or exceed device resources.
     */
    void setMaxSubgraphSize(int maxSize);

    /**
     * Set the minimum estimated benefit threshold.
     * Subgraphs below this threshold will not be returned.
     */
    void setMinBenefitThreshold(float threshold);

    /**
     * Analyze the graph and return all convex subgraphs.
     *
     * @param graph The computation graph to analyze
     * @return List of convex subgraphs, sorted by estimated benefit (descending)
     */
    std::vector<ConvexSubgraph> analyze(Graph* graph);

    /**
     * Check if adding a node to an existing subgraph maintains convexity.
     *
     * @param graph The computation graph
     * @param subgraph Current subgraph
     * @param nodeId Node to potentially add
     * @return true if the resulting subgraph would be convex
     */
    bool wouldMaintainConvexity(Graph* graph, const ConvexSubgraph& subgraph, int nodeId) const;

    /**
     * Compute the transitive closure of the dependency graph.
     * This enables O(1) reachability queries.
     */
    void computeTransitiveClosure(Graph* graph);

    /**
     * Check if there is a path from nodeA to nodeB.
     */
    bool canReach(int nodeA, int nodeB) const;

private:
    struct AnalyzerState;
    std::unique_ptr<AnalyzerState> state_;

    int minSubgraphSize_;
    int maxSubgraphSize_;
    float minBenefitThreshold_;
    std::unordered_set<std::string> seedOpTypes_;

    // Transitive closure represented as adjacency lists
    std::unordered_map<int, std::unordered_set<int>> reachableFrom_;
    std::unordered_map<int, std::unordered_set<int>> reachableTo_;

    // Internal methods
    std::vector<int> findSeedNodes(Graph* graph);
    ConvexSubgraph growSubgraph(Graph* graph, int seedNode);
    float estimateBenefit(Graph* graph, const ConvexSubgraph& subgraph);
    std::vector<int> topologicalSort(Graph* graph, const std::unordered_set<int>& nodes);
    void identifyBoundary(Graph* graph, ConvexSubgraph& subgraph);
};

/**
 * Default seed operation types that benefit from fusion.
 * These are typically memory-bound elementwise operations.
 */
inline std::unordered_set<std::string> getDefaultSeedOpTypes() {
    return {
        // Activations
        "relu", "sigmoid", "tanh", "gelu", "silu", "softmax",
        // Elementwise
        "add", "sub", "mul", "div", "neg", "exp", "log", "sqrt", "rsqrt",
        // Normalization
        "layer_norm", "rms_norm", "batch_norm",
        // LLM specific
        "rope", "rms_norm_swiglu",
        // Reductions (can be fused if followed by elementwise)
        "sum", "mean"
    };
}

/**
 * Operations that should NOT be included in mega-kernels.
 * These are typically compute-bound operations better served by specialized kernels.
 */
inline std::unordered_set<std::string> getExcludedOpTypes() {
    return {
        // Large matrix operations
        "matmul", "mmul", "tensormmul", "conv2d", "conv3d",
        // Operations with complex control flow
        "while", "if", "cond",
        // Operations that need special handling
        "random", "dropout_inverted"
    };
}

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_CONVEX_SUBGRAPH_ANALYZER_H
