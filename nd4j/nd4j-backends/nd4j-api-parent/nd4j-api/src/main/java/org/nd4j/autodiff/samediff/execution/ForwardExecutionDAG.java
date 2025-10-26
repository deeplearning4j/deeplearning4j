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

package org.nd4j.autodiff.samediff.execution;

import lombok.Data;

import java.util.*;
import java.util.stream.Collectors;

/**
 * The corrected forward execution DAG that properly separates operations from variables.
 * This replaces the broken mixed dependency approach with clean operation-to-operation dependencies.
 * 
 * Key features:
 * - Operations depend on other operations (not variables)
 * - Variables are inputs/outputs of operations 
 * - Proper topological ordering
 * - Frame-aware execution for control flow
 * - Multi-output operation support
 * 
 * @author Alex Gibson
 */
@Data
public class ForwardExecutionDAG {
    
    private final List<ExecutionNode> executionOrder;           // Topological order for execution
    private final Map<String, ExecutionNode> operationNodes;   // Operation name -> ExecutionNode
    private final Map<String, String> variableProducers;       // Variable name -> Operation that produces it
    private final Map<String, Set<String>> variableConsumers;  // Variable name -> Operations that consume it
    private final Set<String> requiredPlaceholders;
    private final Set<String> constants;
    private final Set<String> variables;
    
    /**
     * Get execution order respecting frame boundaries and iterations.
     * This ensures control flow operations execute in the correct order.
     */
    public List<ExecutionNode> getFrameAwareExecutionOrder() {
        // Group by frame, then sort by dependencies within each frame
        Map<FrameInfo, List<ExecutionNode>> frameGroups = executionOrder.stream()
            .collect(Collectors.groupingBy(node -> 
                node.getFrameInfo() != null ? node.getFrameInfo() : FrameInfo.OUTER_FRAME));
        
        List<ExecutionNode> frameAwareOrder = new ArrayList<>();
        
        // Execute outer frame first, then nested frames
        executeFrameGroup(FrameInfo.OUTER_FRAME, frameGroups, frameAwareOrder, new HashSet<>());
        
        return frameAwareOrder;
    }
    
    /**
     * Recursively execute frame groups in proper order
     */
    private void executeFrameGroup(FrameInfo frame, Map<FrameInfo, List<ExecutionNode>> frameGroups, 
                                 List<ExecutionNode> result, Set<FrameInfo> visited) {
        if (visited.contains(frame)) return;
        visited.add(frame);
        
        List<ExecutionNode> nodesInFrame = frameGroups.getOrDefault(frame, Collections.emptyList());
        result.addAll(nodesInFrame);
        
        // Execute child frames
        for (FrameInfo childFrame : frameGroups.keySet()) {
            if (childFrame.getParentFrame() != null && childFrame.getParentFrame().equals(frame)) {
                executeFrameGroup(childFrame, frameGroups, result, visited);
            }
        }
    }
    
    /**
     * Get the operation that produces a specific variable
     */
    public ExecutionNode getProducerNode(String variableName) {
        String producerOpName = variableProducers.get(variableName);
        return producerOpName != null ? operationNodes.get(producerOpName) : null;
    }
    
    /**
     * Get all operations that consume a specific variable
     */
    public Set<ExecutionNode> getConsumerNodes(String variableName) {
        Set<String> consumerOpNames = variableConsumers.get(variableName);
        if (consumerOpNames == null) {
            return Collections.emptySet();
        }
        
        return consumerOpNames.stream()
            .map(operationNodes::get)
            .filter(Objects::nonNull)
            .collect(Collectors.toSet());
    }
    
    /**
     * Get all nodes that have no dependencies (can execute immediately)
     */
    public List<ExecutionNode> getInitialNodes() {
        return executionOrder.stream()
            .filter(ExecutionNode::hasNoDependencies)
            .collect(Collectors.toList());
    }
    
    /**
     * Get all nodes of a specific type
     */
    public List<ExecutionNode> getNodesByType(ExecutionNode.ExecutionNodeType type) {
        return executionOrder.stream()
            .filter(node -> node.getNodeType() == type)
            .collect(Collectors.toList());
    }
    
    /**
     * Get all operations in a specific frame
     */
    public List<ExecutionNode> getNodesInFrame(FrameInfo frame) {
        return executionOrder.stream()
            .filter(node -> Objects.equals(node.getFrameInfo(), frame))
            .collect(Collectors.toList());
    }
    
    /**
     * Check if a variable is required as input (placeholder)
     */
    public boolean isRequiredPlaceholder(String variableName) {
        return requiredPlaceholders.contains(variableName);
    }
    
    /**
     * Check if a variable is a constant
     */
    public boolean isConstant(String variableName) {
        return constants.contains(variableName);
    }
    
    /**
     * Check if a variable is a trainable variable
     */
    public boolean isVariable(String variableName) {
        return variables.contains(variableName);
    }
    
    /**
     * Get a summary of the DAG structure
     */
    public DAGSummary getSummary() {
        Map<ExecutionNode.ExecutionNodeType, Long> nodeTypeCounts = executionOrder.stream()
            .collect(Collectors.groupingBy(ExecutionNode::getNodeType, Collectors.counting()));
        
        Map<FrameInfo, Long> frameCounts = executionOrder.stream()
            .collect(Collectors.groupingBy(
                node -> node.getFrameInfo() != null ? node.getFrameInfo() : FrameInfo.OUTER_FRAME,
                Collectors.counting()));
        
        return new DAGSummary(
            executionOrder.size(),
            nodeTypeCounts,
            frameCounts,
            requiredPlaceholders.size(),
            constants.size(),
            variables.size()
        );
    }
    
    /**
     * Validate the DAG structure for consistency
     */
    public void validate() {
        Set<String> issues = new HashSet<>();
        
        // Check that all variable producers exist
        for (Map.Entry<String, String> entry : variableProducers.entrySet()) {
            String variable = entry.getKey();
            String producer = entry.getValue();
            
            if (!operationNodes.containsKey(producer)) {
                issues.add("Variable " + variable + " producer " + producer + " not found in operation nodes");
            }
        }
        
        // Check that all variable consumers exist
        for (Map.Entry<String, Set<String>> entry : variableConsumers.entrySet()) {
            String variable = entry.getKey();
            Set<String> consumers = entry.getValue();
            
            for (String consumer : consumers) {
                if (!operationNodes.containsKey(consumer)) {
                    issues.add("Variable " + variable + " consumer " + consumer + " not found in operation nodes");
                }
            }
        }
        
        // Check for cycles in operation dependencies
        Set<String> visited = new HashSet<>();
        Set<String> visiting = new HashSet<>();
        
        for (ExecutionNode node : executionOrder) {
            if (!visited.contains(node.getOperationName())) {
                if (hasCycle(node, visited, visiting, issues)) {
                    issues.add("Cycle detected involving operation: " + node.getOperationName());
                }
            }
        }
        
        if (!issues.isEmpty()) {
            throw new IllegalStateException("DAG validation failed:\n" + String.join("\n", issues));
        }
    }
    
    private boolean hasCycle(ExecutionNode node, Set<String> visited, Set<String> visiting, Set<String> issues) {
        String nodeName = node.getOperationName();
        
        if (visiting.contains(nodeName)) {
            return true; // Cycle detected
        }
        
        if (visited.contains(nodeName)) {
            return false;
        }
        
        visiting.add(nodeName);
        
        // Check dependencies
        for (String depName : node.getDependsOnOperations()) {
            ExecutionNode depNode = operationNodes.get(depName);
            if (depNode != null && hasCycle(depNode, visited, visiting, issues)) {
                return true;
            }
        }
        
        visiting.remove(nodeName);
        visited.add(nodeName);
        return false;
    }
    
    /**
     * Summary information about the DAG structure
     */
    @Data
    public static class DAGSummary {
        private final int totalNodes;
        private final Map<ExecutionNode.ExecutionNodeType, Long> nodeTypeCounts;
        private final Map<FrameInfo, Long> frameCounts;
        private final int placeholderCount;
        private final int constantCount;
        private final int variableCount;
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("DAG Summary:\n");
            sb.append("  Total nodes: ").append(totalNodes).append("\n");
            sb.append("  Node types: ").append(nodeTypeCounts).append("\n");
            sb.append("  Frames: ").append(frameCounts.size()).append("\n");
            sb.append("  Placeholders: ").append(placeholderCount).append("\n");
            sb.append("  Constants: ").append(constantCount).append("\n");
            sb.append("  Variables: ").append(variableCount).append("\n");
            return sb.toString();
        }
    }
}
