/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.torchscript.ir;

import lombok.Builder;
import lombok.Data;

import java.util.*;

/**
 * Parsed representation of a TorchScript computation graph.
 */
@Data
@Builder
public class TorchScriptGraph {

    /**
     * Name of the graph/module
     */
    private String name;

    /**
     * List of nodes (operations) in the graph
     */
    @Builder.Default
    private List<TorchScriptNode> nodes = new ArrayList<>();

    /**
     * Graph input values
     */
    @Builder.Default
    private List<TorchScriptValue> inputs = new ArrayList<>();

    /**
     * Graph output values
     */
    @Builder.Default
    private List<TorchScriptValue> outputs = new ArrayList<>();

    /**
     * Map of all values by name
     */
    @Builder.Default
    private Map<String, TorchScriptValue> values = new HashMap<>();

    /**
     * Whether this is from a traced model (vs scripted)
     */
    @Builder.Default
    private boolean isTraced = true;

    /**
     * Subgraphs (for control flow in scripted models)
     */
    @Builder.Default
    private List<TorchScriptGraph> subgraphs = new ArrayList<>();

    /**
     * Get nodes in topological order.
     *
     * @return topologically sorted nodes
     */
    public List<TorchScriptNode> topologicalSort() {
        // Build dependency graph
        Map<String, Set<String>> dependencies = new HashMap<>();
        Map<String, TorchScriptNode> nodeByOutput = new HashMap<>();

        for (TorchScriptNode node : nodes) {
            for (String output : node.getOutputNames()) {
                nodeByOutput.put(output, node);
                dependencies.put(output, new HashSet<>(node.getInputNames()));
            }
        }

        // Kahn's algorithm
        List<TorchScriptNode> sorted = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        // Start with inputs (no dependencies)
        for (TorchScriptValue input : inputs) {
            queue.add(input.getName());
            visited.add(input.getName());
        }

        // Also add constants/parameters that have no producer
        for (TorchScriptValue value : values.values()) {
            if (value.isParameter() || value.isBuffer()) {
                queue.add(value.getName());
                visited.add(value.getName());
            }
        }

        while (!queue.isEmpty()) {
            String current = queue.poll();
            TorchScriptNode node = nodeByOutput.get(current);

            if (node != null && !sorted.contains(node)) {
                // Check if all inputs are satisfied
                boolean allInputsSatisfied = true;
                for (String input : node.getInputNames()) {
                    if (!visited.contains(input)) {
                        allInputsSatisfied = false;
                        break;
                    }
                }

                if (allInputsSatisfied) {
                    sorted.add(node);
                    for (String output : node.getOutputNames()) {
                        visited.add(output);
                        queue.add(output);
                    }
                }
            }

            // Find nodes whose dependencies are now satisfied
            for (TorchScriptNode n : nodes) {
                if (!sorted.contains(n)) {
                    boolean satisfied = true;
                    for (String input : n.getInputNames()) {
                        if (!visited.contains(input)) {
                            satisfied = false;
                            break;
                        }
                    }
                    if (satisfied) {
                        sorted.add(n);
                        for (String output : n.getOutputNames()) {
                            visited.add(output);
                        }
                    }
                }
            }
        }

        return sorted;
    }

    /**
     * Get a node by name.
     *
     * @param name node name
     * @return the node or null
     */
    public TorchScriptNode getNode(String name) {
        for (TorchScriptNode node : nodes) {
            if (name.equals(node.getName())) {
                return node;
            }
        }
        return null;
    }

    /**
     * Get nodes by operation type.
     *
     * @param opType operation type (e.g., "aten::conv2d")
     * @return list of matching nodes
     */
    public List<TorchScriptNode> getNodesByOp(String opType) {
        List<TorchScriptNode> result = new ArrayList<>();
        for (TorchScriptNode node : nodes) {
            if (opType.equals(node.getOpType())) {
                result.add(node);
            }
        }
        return result;
    }

    /**
     * Get nodes by schema name (without namespace).
     *
     * @param schemaName schema name (e.g., "conv2d")
     * @return list of matching nodes
     */
    public List<TorchScriptNode> getNodesBySchema(String schemaName) {
        List<TorchScriptNode> result = new ArrayList<>();
        for (TorchScriptNode node : nodes) {
            if (schemaName.equals(node.getSchemaName())) {
                result.add(node);
            }
        }
        return result;
    }

    /**
     * Get a value by name.
     *
     * @param name value name
     * @return the value or null
     */
    public TorchScriptValue getValue(String name) {
        return values.get(name);
    }

    /**
     * Add a node to the graph.
     *
     * @param node the node to add
     */
    public void addNode(TorchScriptNode node) {
        nodes.add(node);
    }

    /**
     * Add a value to the graph.
     *
     * @param value the value to add
     */
    public void addValue(TorchScriptValue value) {
        values.put(value.getName(), value);
    }

    /**
     * Get all parameter values.
     *
     * @return list of parameter values
     */
    public List<TorchScriptValue> getParameters() {
        List<TorchScriptValue> params = new ArrayList<>();
        for (TorchScriptValue value : values.values()) {
            if (value.isParameter()) {
                params.add(value);
            }
        }
        return params;
    }

    /**
     * Get all buffer values.
     *
     * @return list of buffer values
     */
    public List<TorchScriptValue> getBuffers() {
        List<TorchScriptValue> buffers = new ArrayList<>();
        for (TorchScriptValue value : values.values()) {
            if (value.isBuffer()) {
                buffers.add(value);
            }
        }
        return buffers;
    }

    /**
     * Count nodes by operation type.
     *
     * @return map of op type to count
     */
    public Map<String, Integer> countNodesByOp() {
        Map<String, Integer> counts = new HashMap<>();
        for (TorchScriptNode node : nodes) {
            String op = node.getOpType();
            counts.put(op, counts.getOrDefault(op, 0) + 1);
        }
        return counts;
    }

    /**
     * Get the total number of nodes.
     *
     * @return node count
     */
    public int getNodeCount() {
        return nodes.size();
    }

    /**
     * Check if the graph has control flow (subgraphs).
     *
     * @return true if there are subgraphs
     */
    public boolean hasControlFlow() {
        return !subgraphs.isEmpty();
    }
}
