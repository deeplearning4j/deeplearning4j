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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.shape.Split;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Builder for creating corrected forward execution DAGs.
 * This fixes the fundamental issue where variables and operations were mixed in dependency chains.
 * 
 * PROBLEM SOLVED: Instead of broken chains like:
 *   1492 → /pooler/dense/Gemm_output_0 → pooler.dense.weight → pooler.dense.bias → last_hidden_state → add_85 → split_11:2 → softmax_11
 * 
 * We now create proper operation dependencies:
 *   TanhOp → DenseOp → GatherOp (where operations depend on operations, variables are just inputs/outputs)
 * 
 * Supports both TensorFlow and ONNX import patterns.
 * 
 * @author Alex Gibson
 */
@Slf4j
public class ForwardExecutionDAGBuilder {
    
    private final SameDiff sameDiff;
    
    public ForwardExecutionDAGBuilder(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }
    
    /**
     * Build the corrected forward execution DAG.
     * This is the main entry point that fixes the variable/operation mixing problem.
     * 
     * @param requestedOutputs Variables we want to compute
     * @return Properly structured DAG with operation-to-operation dependencies
     */
    public ForwardExecutionDAG buildForwardDAG(Collection<String> requestedOutputs) {
        log.info("Building forward execution DAG for outputs: {}", requestedOutputs);

        Set<String> requiredOperations = new HashSet<>();
        Set<String> requiredVariables = new HashSet<>();

        Set<String> allPlaceholders = findAllPlaceholders();
        requiredVariables.addAll(allPlaceholders);
        requiredVariables.addAll(requestedOutputs);

        boolean foundNewNodes = true;
        int iteration = 0;

        while (foundNewNodes && iteration < 50) {
            int prevOps = requiredOperations.size();
            int prevVars = requiredVariables.size();

            findRequiredSubgraph(new ArrayList<>(requiredVariables), requiredOperations, requiredVariables);

            foundNewNodes = (requiredOperations.size() > prevOps) || (requiredVariables.size() > prevVars);
            iteration++;

            if (iteration % 10 == 0) {
                log.debug("Iteration {}: {} ops, {} vars", iteration, requiredOperations.size(), requiredVariables.size());
            }
        }

        log.info("Converged after {} iterations: {} operations, {} variables",
                iteration, requiredOperations.size(), requiredVariables.size());

        validateCompleteness(requestedOutputs, allPlaceholders, requiredVariables);

        Map<String, ExecutionNode> operationNodes = buildExecutionNodes(requiredOperations, requiredVariables);
        establishOperationDependencies(operationNodes);
        List<ExecutionNode> executionOrder = createTopologicalOrder(operationNodes);

        Map<String, String> variableProducers = buildVariableProducerMap(operationNodes);
        Map<String, Set<String>> variableConsumers = buildVariableConsumerMap(operationNodes);

        Set<String> constants = findConstants();
        Set<String> variables = findVariables();

        ForwardExecutionDAG dag = new ForwardExecutionDAG(
                executionOrder, operationNodes, variableProducers, variableConsumers,
                allPlaceholders, constants, variables
        );

        dag.validate();
        return dag;
    }
    
    /**
     * FIXED: Find required subgraph using proper operation traversal.
     * This replaces the broken mixed variable/operation approach.
     * 
     * Key fix: We traverse by finding producer OPERATIONS for variables,
     * not mixing variables and operations in a single dependency chain.
     */
    private void findRequiredSubgraph(Collection<String> startingVariables,
                                      Set<String> requiredOperations,
                                      Set<String> requiredVariables) {

        Queue<String> variablesToProcess = new LinkedList<>(startingVariables);
        Set<String> processedVariables = new HashSet<>();

        while (!variablesToProcess.isEmpty()) {
            String currentVariable = variablesToProcess.poll();

            if (processedVariables.contains(currentVariable)) {
                continue;
            }
            processedVariables.add(currentVariable);
            requiredVariables.add(currentVariable);

            String producerOpName = findProducerOperation(currentVariable);

            if (producerOpName != null) {
                requiredOperations.add(producerOpName);

                SameDiffOp producerOp = sameDiff.getOps().get(producerOpName);
                if (producerOp != null && producerOp.getInputsToOp() != null) {
                    for (String inputVar : producerOp.getInputsToOp()) {
                        if (!processedVariables.contains(inputVar)) {
                            variablesToProcess.offer(inputVar);
                        }
                    }
                }

                if (producerOp != null && producerOp.getControlDeps() != null) {
                    for (String controlDep : producerOp.getControlDeps()) {
                        requiredOperations.add(controlDep);

                        SameDiffOp controlDepOp = sameDiff.getOps().get(controlDep);
                        if (controlDepOp != null && controlDepOp.getOutputsOfOp() != null) {
                            for (String controlDepOutput : controlDepOp.getOutputsOfOp()) {
                                if (!processedVariables.contains(controlDepOutput)) {
                                    variablesToProcess.offer(controlDepOutput);
                                }
                            }
                        }
                    }
                }
            }

            // Handle placeholder multi-use: find ALL operations that consume this variable
            if (isPlaceholder(currentVariable)) {
                Set<String> consumerOps = findAllConsumerOperations(currentVariable);
                for (String consumerOp : consumerOps) {
                    if (!requiredOperations.contains(consumerOp)) {
                        requiredOperations.add(consumerOp);

                        // Add outputs of consumer operations to ensure they're processed
                        SameDiffOp op = sameDiff.getOps().get(consumerOp);
                        if (op != null && op.getOutputsOfOp() != null) {
                            for (String output : op.getOutputsOfOp()) {
                                if (!processedVariables.contains(output)) {
                                    variablesToProcess.offer(output);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private boolean isPlaceholder(String variableName) {
        Variable var = sameDiff.getVariables().get(variableName);
        return var != null && var.getVariable().getVariableType() == VariableType.PLACEHOLDER;
    }

    private Set<String> findAllConsumerOperations(String variableName) {
        Set<String> consumers = new HashSet<>();

        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            if (op.getInputsToOp() != null) {
                for (String input : op.getInputsToOp()) {
                    if (input.equals(variableName) || stripVariableSuffix(input).equals(stripVariableSuffix(variableName))) {
                        consumers.add(entry.getKey());
                        break;
                    }
                }
            }
        }

        return consumers;
    }


    private Set<String> findAllPlaceholders() {
        return sameDiff.getVariables().values().stream()
                .filter(v -> v.getVariable().getVariableType() == VariableType.PLACEHOLDER)
                .map(v -> v.getVariable().name())
                .collect(Collectors.toSet());
    }

    private void validateCompleteness(Collection<String> requestedOutputs,
                                      Set<String> allPlaceholders,
                                      Set<String> requiredVariables) {

        for (String output : requestedOutputs) {
            if (!requiredVariables.contains(output)) {
                throw new IllegalStateException("Required output not in graph: " + output);
            }
        }

        for (String placeholder : allPlaceholders) {
            if (!requiredVariables.contains(placeholder)) {
                throw new IllegalStateException("Placeholder not in graph: " + placeholder);
            }
        }

        log.info("Validation passed: {} outputs, {} placeholders included",
                requestedOutputs.size(), allPlaceholders.size());
    }

    /**
     * Build execution nodes with proper separation of concerns.
     * Each node represents either an operation or a variable initialization.
     */
    private Map<String, ExecutionNode> buildExecutionNodes(Set<String> requiredOperations, 
                                                          Set<String> requiredVariables) {
        Map<String, ExecutionNode> nodes = new HashMap<>();
        
        // Create nodes for actual operations
        for (String opName : requiredOperations) {
            SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
            if (sameDiffOp == null) {
                log.warn("Operation {} not found in SameDiff ops", opName);
                continue;
            }
            
            DifferentialFunction op = sameDiffOp.getOp();
            
            // Get input/output variables for this operation
            List<String> inputs = sameDiffOp.getInputsToOp() != null ? 
                sameDiffOp.getInputsToOp() : Collections.emptyList();
            List<String> outputs = sameDiffOp.getOutputsOfOp() != null ? 
                sameDiffOp.getOutputsOfOp() : Collections.emptyList();
            
            // Determine node type and frame info
            ExecutionNode.ExecutionNodeType nodeType = determineNodeType(op);
            FrameInfo frameInfo = determineFrameInfo(op, opName);
            
            ExecutionNode node = new ExecutionNode(
                opName, op, inputs, outputs, new HashSet<>(), nodeType, frameInfo
            );
            
            nodes.put(opName, node);
            log.trace("Created execution node: {}", node.getDescription());
        }
        
        // Add special nodes for constants, variables, and placeholders
        addSpecialVariableNodes(nodes, requiredVariables);
        
        return nodes;
    }
    
    /**
     * Establish dependencies between OPERATIONS, not variables.
     * This is the key fix that solves the mixed dependency problem.
     * 
     * For each operation, we find the operations that produce its input variables,
     * creating clean operation-to-operation dependencies.
     */
    private void establishOperationDependencies(Map<String, ExecutionNode> operationNodes) {
        for (ExecutionNode node : operationNodes.values()) {
            Set<String> dependencies = new HashSet<>();
            
            // For each input variable, find the operation that produces it
            for (String inputVar : node.getInputVariables()) {
                String producerOpName = findProducerOperation(inputVar);
                
                if (producerOpName != null && operationNodes.containsKey(producerOpName)) {
                    dependencies.add(producerOpName);
                    log.trace("Operation {} depends on operation {} (via variable {})", 
                            node.getOperationName(), producerOpName, inputVar);
                }
            }
            
            // Add control dependencies
            SameDiffOp sameDiffOp = sameDiff.getOps().get(node.getOperationName());
            if (sameDiffOp != null && sameDiffOp.getControlDeps() != null) {
                for (String controlDep : sameDiffOp.getControlDeps()) {
                    if (operationNodes.containsKey(controlDep)) {
                        dependencies.add(controlDep);
                        log.trace("Operation {} has control dependency on operation {}", 
                                node.getOperationName(), controlDep);
                    }
                }
            }
            
            // Handle special cases for control flow operations
            addControlFlowDependencies(node, dependencies, operationNodes);
            
            node.getDependsOnOperations().addAll(dependencies);
        }
    }
    
    /**
     * Handle special dependency cases for control flow operations.
     * These operations have unique execution semantics that require special handling.
     */
    private void addControlFlowDependencies(ExecutionNode node, Set<String> dependencies, 
                                          Map<String, ExecutionNode> operationNodes) {
        DifferentialFunction op = node.getOperation();
        
        if (op instanceof Switch) {
            // Switch depends on both input tensor and predicate
            // Dependencies already handled by input variables
            log.trace("Switch operation {} requires both input and predicate", node.getOperationName());
        } else if (op instanceof Merge) {
            // Merge can execute when ANY of its inputs is ready
            // This requires special handling in the execution engine
            log.trace("Merge operation {} requires special OR-dependency handling", node.getOperationName());
        } else if (op instanceof Enter) {
            // Enter creates new frame context
            Enter enter = (Enter) op;
            log.trace("Enter operation {} creates frame: {}", node.getOperationName(), enter.getFrameName());
        } else if (op instanceof Exit) {
            // Exit returns to parent frame
            log.trace("Exit operation {} returns to parent frame", node.getOperationName());
        } else if (op instanceof NextIteration) {
            // NextIteration advances iteration counter
            log.trace("NextIteration operation {} advances iteration", node.getOperationName());
        } else if (op instanceof LoopCond) {
            // LoopCond determines loop continuation
            log.trace("LoopCond operation {} controls loop execution", node.getOperationName());
        }
    }
    
    /**
     * Create topological execution order respecting dependencies.
     * This ensures operations execute in the correct order.
     */
    private List<ExecutionNode> createTopologicalOrder(Map<String, ExecutionNode> operationNodes) {
        List<ExecutionNode> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Set<String> visiting = new HashSet<>();
        
        // Start with nodes that have no dependencies (constants, variables, placeholders)
        for (ExecutionNode node : operationNodes.values()) {
            if (node.getDependsOnOperations().isEmpty()) {
                topologicalSort(node, operationNodes, visited, visiting, result);
            }
        }
        
        // Then process remaining nodes
        for (ExecutionNode node : operationNodes.values()) {
            if (!visited.contains(node.getOperationName())) {
                topologicalSort(node, operationNodes, visited, visiting, result);
            }
        }
        
        log.info("Created topological execution order with {} nodes", result.size());
        return result;
    }
    
    /**
     * Recursive topological sort implementation
     */
    private void topologicalSort(ExecutionNode node, Map<String, ExecutionNode> allNodes,
                               Set<String> visited, Set<String> visiting, List<ExecutionNode> result) {
        
        String nodeName = node.getOperationName();
        
        if (visiting.contains(nodeName)) {
            log.warn("Cycle detected involving node: {}", nodeName);
            return;
        }
        
        if (visited.contains(nodeName)) {
            return;
        }
        
        visiting.add(nodeName);
        
        // Visit all dependencies first
        for (String depName : node.getDependsOnOperations()) {
            ExecutionNode depNode = allNodes.get(depName);
            if (depNode != null) {
                topologicalSort(depNode, allNodes, visited, visiting, result);
            }
        }
        
        visiting.remove(nodeName);
        visited.add(nodeName);
        result.add(node);
    }
    
    /**
     * Find the operation that produces a given variable.
     * This fixes the core lookup logic that was causing the mixed dependencies.
     * 
     * Handles both TensorFlow and ONNX import patterns:
     * - TensorFlow: multi-output ops use ":0", ":1" suffixes
     * - ONNX: different naming patterns
     */
    private String findProducerOperation(String variableName) {
        // Strip variable suffix for multi-output operations (e.g., "split:1" -> "split")
        String baseVarName = stripVariableSuffix(variableName);
        
        // Check all operations to find the producer
        for (Map.Entry<String, SameDiffOp> entry : sameDiff.getOps().entrySet()) {
            SameDiffOp op = entry.getValue();
            List<String> outputs = op.getOutputsOfOp();
            
            if (outputs != null) {
                // Direct match
                if (outputs.contains(variableName)) {
                    return entry.getKey();
                }
                
                // Base name match (for multi-output ops)
                if (outputs.contains(baseVarName)) {
                    return entry.getKey();
                }
                
                // Handle TensorFlow import patterns where output might be "op_name:index"
                for (String output : outputs) {
                    if (variableName.startsWith(output + ":")) {
                        return entry.getKey();
                    }
                }
            }
        }
        
        // Check if it's a variable/constant/placeholder
        Variable var = sameDiff.getVariables().get(variableName);
        if (var != null) {
            VariableType type = var.getVariable().getVariableType();
            if (type == VariableType.CONSTANT || type == VariableType.VARIABLE || type == VariableType.PLACEHOLDER) {
                return null; // These don't have producer operations
            }
        }
        
        log.trace("No producer operation found for variable: {}", variableName);
        return null;
    }
    
    /**
     * Strip variable suffixes for multi-output operations.
     * Handles both TensorFlow (:0, :1, :2) and ONNX patterns.
     */
    private String stripVariableSuffix(String variableName) {
        // TensorFlow pattern: "operation:0", "operation:1", etc.
        int colonIndex = variableName.lastIndexOf(':');
        if (colonIndex > 0) {
            String suffix = variableName.substring(colonIndex + 1);
            try {
                Integer.parseInt(suffix);
                return variableName.substring(0, colonIndex);
            } catch (NumberFormatException e) {
                // Not a numeric suffix, return as-is (could be ONNX pattern)
            }
        }
        
        return variableName;
    }
    
    /**
     * Determine the type of execution node based on the operation
     */
    private ExecutionNode.ExecutionNodeType determineNodeType(DifferentialFunction op) {
        if (op instanceof Switch || op instanceof Merge || op instanceof Enter || 
            op instanceof Exit || op instanceof NextIteration || op instanceof LoopCond) {
            return ExecutionNode.ExecutionNodeType.CONTROL_FLOW_OP;
        } else if (op instanceof Split) {
            return ExecutionNode.ExecutionNodeType.MULTI_OUTPUT_OP;
        } else {
            return ExecutionNode.ExecutionNodeType.STANDARD_OP;
        }
    }
    
    /**
     * Determine frame information for control flow operations.
     * This is important for TensorFlow-style control flow.
     */
    private FrameInfo determineFrameInfo(DifferentialFunction op, String opName) {
        if (op instanceof Enter) {
            Enter enter = (Enter) op;
            return new FrameInfo(enter.getFrameName(), 0, FrameInfo.OUTER_FRAME);
        }
        
        // For now, most operations are in the outer frame
        // This would need to be enhanced for full control flow support
        return FrameInfo.OUTER_FRAME;
    }
    
    /**
     * Add special nodes for constants, variables, and placeholders.
     * These don't correspond to operations but need to be available for execution.
     */
    private void addSpecialVariableNodes(Map<String, ExecutionNode> nodes, Set<String> requiredVariables) {
        for (String varName : requiredVariables) {
            Variable var = sameDiff.getVariables().get(varName);
            if (var != null) {
                VariableType type = var.getVariable().getVariableType();
                
                if (type == VariableType.CONSTANT) {
                    ExecutionNode node = new ExecutionNode(
                        "CONST_" + varName, null, Collections.emptyList(), 
                        Arrays.asList(varName), new HashSet<>(),
                        ExecutionNode.ExecutionNodeType.VARIABLE_INIT, FrameInfo.OUTER_FRAME
                    );
                    nodes.put("CONST_" + varName, node);
                } else if (type == VariableType.VARIABLE) {
                    ExecutionNode node = new ExecutionNode(
                        "VAR_" + varName, null, Collections.emptyList(), 
                        Arrays.asList(varName), new HashSet<>(),
                        ExecutionNode.ExecutionNodeType.VARIABLE_INIT, FrameInfo.OUTER_FRAME
                    );
                    nodes.put("VAR_" + varName, node);
                } else if (type == VariableType.PLACEHOLDER) {
                    ExecutionNode node = new ExecutionNode(
                        "PH_" + varName, null, Collections.emptyList(), 
                        Arrays.asList(varName), new HashSet<>(),
                        ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET, FrameInfo.OUTER_FRAME
                    );
                    nodes.put("PH_" + varName, node);
                }
            }
        }
    }
    
    /**
     * Build variable producer mapping for quick lookups
     */
    private Map<String, String> buildVariableProducerMap(Map<String, ExecutionNode> operationNodes) {
        Map<String, String> producers = new HashMap<>();
        
        for (ExecutionNode node : operationNodes.values()) {
            for (String outputVar : node.getOutputVariables()) {
                producers.put(outputVar, node.getOperationName());
            }
        }
        
        return producers;
    }
    
    /**
     * Build variable consumer mapping for dependency tracking
     */
    private Map<String, Set<String>> buildVariableConsumerMap(Map<String, ExecutionNode> operationNodes) {
        Map<String, Set<String>> consumers = new HashMap<>();
        
        for (ExecutionNode node : operationNodes.values()) {
            for (String inputVar : node.getInputVariables()) {
                consumers.computeIfAbsent(inputVar, k -> new HashSet<>()).add(node.getOperationName());
            }
        }
        
        return consumers;
    }
    
    private Set<String> findConstants() {
        return sameDiff.getVariables().values().stream()
            .filter(v -> v.getVariable().getVariableType() == VariableType.CONSTANT)
            .map(v -> v.getVariable().name())
            .collect(Collectors.toSet());
    }
    
    private Set<String> findVariables() {
        return sameDiff.getVariables().values().stream()
            .filter(v -> v.getVariable().getVariableType() == VariableType.VARIABLE)
            .map(v -> v.getVariable().name())
            .collect(Collectors.toSet());
    }
    
    private Set<String> findPlaceholders(Set<String> requiredVariables) {
        return requiredVariables.stream()
            .filter(varName -> {
                Variable var = sameDiff.getVariables().get(varName);
                return var != null && var.getVariable().getVariableType() == VariableType.PLACEHOLDER;
            })
            .collect(Collectors.toSet());
    }
}
