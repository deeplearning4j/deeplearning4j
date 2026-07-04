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

package org.nd4j.autodiff.samediff.internal;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.execution.ExecutionNode;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.*;

@Slf4j
public class ControlFlowExecutor {
    /** Bounded diagnostic counter for switch-routing visibility. */
    private int switchLogCount = 0;

    private final InferenceSession session;

    public ControlFlowExecutor(InferenceSession session) {
        this.session = session;
    }

    public void executeIdentityNode(ExecutionNode node, Map<String, SDValue> variableValues) {
        String opName = node.getOperationName();
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("Identity operation " + opName + " has no inputs or outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            // Dead-path passthrough: propagate the null marker so downstream
            // null-skip discipline (and eventually Merge) resolves the branch.
            if (variableValues.containsKey(inputVar)) {
                variableValues.put(outputVar, null);
                return;
            }
            throw new IllegalStateException("Input variable " + inputVar + " not found for Identity operation " + opName);
        }

        variableValues.put(outputVar, inputValue);
    }

    public void executeSwitchNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        Switch switchOp = (Switch) op;
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.size() < 2) {
            throw new IllegalStateException("Switch operation requires at least 2 inputs");
        }

        String dataInput = inputs.get(0);
        String predicateInput = inputs.get(1);

        SDValue dataValue = variableValues.get(dataInput);
        SDValue predicateValue = variableValues.get(predicateInput);

        if (predicateValue == null) {
            throw new IllegalStateException("Switch inputs not available: data=" + (dataValue != null) +
                    ", predicate=false");
        }
        if (dataValue == null) {
            // Null data = a value from a dead path (e.g. deliberately-unfed inputs of
            // an ONNX If's never-evaluated branch). A Switch with nothing to route
            // emits null on BOTH sides so downstream null-propagation skips cleanly —
            // throwing here would make dead-branch inputs fatal, contradicting the
            // engine's null-skip discipline that Merge relies on.
            if (outputs.size() >= 2) {
                variableValues.put(outputs.get(0), null);
                variableValues.put(outputs.get(1), null);
            }
            return;
        }

        INDArray predicate = predicateValue.getTensorValue();
        boolean condition = predicate.getDouble(0) != 0.0;
        if (log.isInfoEnabled() && condition && switchLogCount < 12) {
            switchLogCount++;
            log.info("Switch '{}': predicate '{}' = TRUE -> routing data to output[1] ('{}')",
                    node.getOperationName(), predicateInput,
                    outputs.size() >= 2 ? outputs.get(1) : "?");
        }

        // Switch outputs: [false_output, true_output]
        if (outputs.size() >= 2) {
            if (condition) {
                variableValues.put(outputs.get(1), dataValue); // true branch
                variableValues.put(outputs.get(0), null);       // false branch (null)
            } else {
                variableValues.put(outputs.get(0), dataValue);  // false branch
                variableValues.put(outputs.get(1), null);       // true branch (null)
            }
        }
    }

    public void executeEnterNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        Enter enterOp = (Enter) op;
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("Enter operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for Enter operation");
        }

        // Enter just forwards the input to the output (entering a new frame)
        variableValues.put(outputVar, inputValue);
    }

    public void executeExitNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("Exit operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for Exit operation");
        }

        // Exit forwards the input to the parent frame
        variableValues.put(outputVar, inputValue);
    }

    public void executeNextIterationNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("NextIteration operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for NextIteration operation");
        }

        // NextIteration forwards input to next iteration
        variableValues.put(outputVar, inputValue);
    }

    public void executeMergeNode(ExecutionNode node, Map<String, SDValue> variableValues,
                                  DifferentialFunction op, Set<String> allRequired) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.size() < 2 || outputs.isEmpty()) {
            throw new IllegalStateException("Merge operation requires at least 2 inputs and 1 output");
        }

        String outputVar = outputs.get(0);

        // Standard Merge picks the first AVAILABLE input — but dead control-flow
        // paths can surface as EMPTY arrays rather than nulls (shape-derived
        // empties from branches whose data inputs are absent, e.g. the compute
        // side of a nested If whose captures live only in the other outer branch).
        // An empty must never beat a real value: prefer the first non-null,
        // NON-EMPTY input; fall back to an empty only when nothing real exists.
        SDValue emptyFallback = null;
        String emptyFallbackVar = null;
        for (String inputVar : inputs) {
            SDValue inputValue = variableValues.get(inputVar);
            if (inputValue == null) continue;
            org.nd4j.linalg.api.ndarray.INDArray t = inputValue.getTensorValue();
            if (t != null && (t.isEmpty() || t.length() == 0)) {
                if (emptyFallback == null) {
                    emptyFallback = inputValue;
                    emptyFallbackVar = inputVar;
                }
                continue;
            }
            if (log.isInfoEnabled() && outputVar.startsWith("present.")) {
                log.info("Merge '{}' picked input '{}' -> {}", outputVar, inputVar,
                        t == null ? "NULL_TENSOR" : java.util.Arrays.toString(t.shape()));
            }
            variableValues.put(outputVar, inputValue);

            // Add dependency tracking for the Merge output
            // This is crucial: the Merge shares the SDValue with its input,
            // so we need to add a dependency to prevent the array from being
            // freed when the input's dependency is satisfied
            session.addConsumerDependencies(inputValue, outputVar, allRequired);
            return;
        }
        if (emptyFallback != null) {
            log.debug("Merge '{}': only EMPTY input available ('{}') — passing it through",
                    outputVar, emptyFallbackVar);
            variableValues.put(outputVar, emptyFallback);
            session.addConsumerDependencies(emptyFallback, outputVar, allRequired);
            return;
        }

        throw new IllegalStateException("No inputs available for Merge operation " + node.getOperationName());
    }

    public void executeLoopCondNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.isEmpty() || outputs.isEmpty()) {
            throw new IllegalStateException("LoopCond operation requires inputs and outputs");
        }

        String inputVar = inputs.get(0);
        String outputVar = outputs.get(0);

        SDValue inputValue = variableValues.get(inputVar);
        if (inputValue == null) {
            throw new IllegalStateException("Input variable " + inputVar + " not found for LoopCond operation");
        }

        // LoopCond forwards boolean condition
        variableValues.put(outputVar, inputValue);
    }

    /**
     * Execute a while-loop region iteratively until the condition is false.
     */
    public void executeWhileLoop(InferenceSession.WhileLoopRegion region, ForwardExecutionDAG dag,
                                  Map<String, SDValue> variableValues,
                                  Set<String> completedOps,
                                  Set<String> allRequired,
                                  List<Listener> listeners, At at, MultiDataSet batch) {
        Map<String, ExecutionNode> opNodes = dag.getOperationNodes();
        int maxIterations = 10000;

        // Build a map from NextIteration output var → Merge input var
        // so we can find which Merge input comes from NextIteration
        Set<String> nextIterOutputVars = new HashSet<>();
        for (String nextIterOp : region.nextIterOps) {
            ExecutionNode nextIterNode = opNodes.get(nextIterOp);
            if (nextIterNode != null) {
                nextIterOutputVars.addAll(nextIterNode.getOutputVariables());
            }
        }

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            // 1. Execute Merge ops
            // On iteration 0: pick the Enter input (first non-null)
            // On iteration 1+: pick the NextIteration input (which was just updated)
            for (String mergeOp : region.mergeOps) {
                ExecutionNode mergeNode = opNodes.get(mergeOp);
                if (mergeNode == null) continue;

                if (iteration == 0) {
                    // First iteration: use standard Merge (picks Enter input)
                    session.executeNode(mergeNode, variableValues, allRequired, listeners, at, batch);
                } else {
                    // Subsequent iterations: prefer the NextIteration input
                    List<String> inputs = mergeNode.getInputVariables();
                    List<String> outputs = mergeNode.getOutputVariables();
                    String outputVar = outputs.get(0);

                    SDValue nextIterValue = null;
                    for (String input : inputs) {
                        if (nextIterOutputVars.contains(input)) {
                            nextIterValue = variableValues.get(input);
                            break;
                        }
                    }

                    if (nextIterValue != null) {
                        variableValues.put(outputVar, nextIterValue);
                    } else {
                        // Fallback to standard Merge
                        session.executeNode(mergeNode, variableValues, allRequired, listeners, at, batch);
                    }
                }
                completedOps.add(mergeOp);
            }

            // 2. Execute condition ops (between Merge and LoopCond)
            for (String condOp : region.condOps) {
                ExecutionNode condNode = opNodes.get(condOp);
                if (condNode != null) {
                    session.executeNode(condNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(condOp);
                }
            }

            // 3. Execute LoopCond
            if (region.loopCondOp != null) {
                ExecutionNode loopCondNode = opNodes.get(region.loopCondOp);
                if (loopCondNode != null) {
                    session.executeNode(loopCondNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(region.loopCondOp);
                }
            }

            // 4. Execute Switch ops
            for (String switchOp : region.switchOps) {
                ExecutionNode switchNode = opNodes.get(switchOp);
                if (switchNode != null) {
                    session.executeNode(switchNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(switchOp);
                }
            }

            // 5. Check condition: if any Switch routed to false (exit), stop looping
            boolean conditionTrue = true;
            for (String switchOp : region.switchOps) {
                ExecutionNode switchNode = opNodes.get(switchOp);
                if (switchNode == null) continue;
                List<String> outputs = switchNode.getOutputVariables();
                if (outputs.size() >= 2) {
                    // If true output is null, condition is false
                    SDValue trueOutput = variableValues.get(outputs.get(1));
                    if (trueOutput == null) {
                        conditionTrue = false;
                        break;
                    }
                }
            }

            if (!conditionTrue) {
                // Execute Exit ops to forward false-branch values out of the loop
                for (String exitOp : region.exitOps) {
                    ExecutionNode exitNode = opNodes.get(exitOp);
                    if (exitNode != null) {
                        session.executeNode(exitNode, variableValues, allRequired, listeners, at, batch);
                        completedOps.add(exitOp);
                    }
                }
                log.debug("While loop '{}' exited after {} iterations", region.frameName, iteration + 1);
                break;
            }

            // 6. Execute body ops (with nested control flow support)
            // Body ops may contain nested if-conditionals (Switch/Merge pairs).
            // After executing a nested Switch, mark inactive branch ops for skipping.
            // Reset skip set each iteration since if-condition may change between iterations.
            Set<String> bodySkipOps = new HashSet<>();
            for (String bodyOp : region.bodyOps) {
                if (bodySkipOps.contains(bodyOp)) {
                    continue; // Skip inactive branch ops from nested if
                }
                ExecutionNode bodyNode = opNodes.get(bodyOp);
                if (bodyNode != null) {
                    // For nested Merge ops (from if-conditionals), relax readiness:
                    // only need at least one non-null input
                    DifferentialFunction bodyNodeOp = bodyNode.getOperation();
                    if (bodyNodeOp instanceof Merge) {
                        boolean hasInput = false;
                        for (String input : bodyNode.getInputVariables()) {
                            if (variableValues.get(input) != null) {
                                hasInput = true;
                                break;
                            }
                        }
                        if (!hasInput) {
                            log.warn("Nested Merge {} in while body has no available inputs, skipping", bodyOp);
                            continue;
                        }
                    }

                    session.executeNode(bodyNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(bodyOp);

                    // After nested Switch, mark inactive branch for skipping
                    if (bodyNodeOp instanceof Switch) {
                        markInactiveBranchForSkipping(bodyNode, dag, variableValues, bodySkipOps);
                    }
                }
            }

            // 7. Execute NextIteration ops (feed values back to Merge)
            for (String nextIterOp : region.nextIterOps) {
                ExecutionNode nextIterNode = opNodes.get(nextIterOp);
                if (nextIterNode != null) {
                    session.executeNode(nextIterNode, variableValues, allRequired, listeners, at, batch);
                    completedOps.add(nextIterOp);
                }
            }
        }
    }

    /**
     * After a Switch op executes, mark all ops on the inactive branch for skipping.
     * BFS from null Switch outputs through consumer ops. Stop at Merge nodes.
     */
    public void markInactiveBranchForSkipping(ExecutionNode switchNode, ForwardExecutionDAG dag,
                                               Map<String, SDValue> variableValues,
                                               Set<String> skipOps) {
        List<String> outputs = switchNode.getOutputVariables();
        if (outputs.size() < 2) return;

        for (String output : outputs) {
            SDValue val = variableValues.get(output);
            if (val != null) continue; // Active branch

            // This output is null (inactive). BFS through consumers.
            Queue<String> queue = new LinkedList<>();
            Set<String> consumers = dag.getVariableConsumers().get(output);
            // NOTE: do NOT fall back to the ":N"-stripped base name on a lookup miss —
            // the base name is the OTHER (live) switch output, and marking ITS
            // consumers kills the active branch (observed: entire else-frame encoder
            // chain marked dead through 'switch' when 'switch:1' missed). A missed
            // lookup is safe to leave unmarked: skipped-op marker publication makes
            // dead-side propagation correct without BFS; BFS is only an optimization.
            if (consumers == null || consumers.isEmpty()) {
                log.debug("markInactiveBranchForSkipping: dead switch output '{}' has no consumers " +
                        "in the DAG map — relying on null-marker propagation", output);
            }
            int before = skipOps.size();
            if (consumers != null) {
                queue.addAll(consumers);
            }

            while (!queue.isEmpty()) {
                String consumerOp = queue.poll();
                if (skipOps.contains(consumerOp)) continue;

                ExecutionNode consumerNode = dag.getOperationNodes().get(consumerOp);
                if (consumerNode == null) continue;

                // Stop at Merge nodes — they handle null inputs by picking the other.
                // Stop at Switch nodes too: a Switch is a re-gating point that handles
                // dead data itself (emits null markers on both sides); marking it
                // skipped would swallow its LIVE other input's routing.
                if (consumerNode.getOperation() instanceof Merge
                        || consumerNode.getOperation() instanceof Switch) continue;

                skipOps.add(consumerOp);

                // Continue BFS through this op's outputs
                for (String consumerOutput : consumerNode.getOutputVariables()) {
                    Set<String> nextConsumers = dag.getVariableConsumers().get(consumerOutput);
                    if (nextConsumers != null) {
                        queue.addAll(nextConsumers);
                    }
                }
            }
            if (log.isDebugEnabled()) {
                log.debug("markInactiveBranchForSkipping: dead output '{}' -> marked {} ops inactive",
                        output, skipOps.size() - before);
            }
        }
    }
}
