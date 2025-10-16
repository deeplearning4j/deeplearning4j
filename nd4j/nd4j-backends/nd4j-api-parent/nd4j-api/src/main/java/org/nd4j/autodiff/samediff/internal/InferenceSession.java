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

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.ExecutionResult;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.execution.ExecutionNode;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAG;
import org.nd4j.autodiff.samediff.execution.DAGCache;
import org.nd4j.autodiff.samediff.execution.ForwardExecutionDAGBuilder;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.autodiff.samediff.internal.memory.HashDependencyTracker;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.imports.VariableUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.custom.Invoke;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.Concat;
import org.nd4j.linalg.api.ops.impl.shape.CreateView;
import org.nd4j.linalg.api.ops.impl.shape.Stack;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.*;
import org.nd4j.linalg.api.ops.impl.transforms.Assert;
import org.nd4j.linalg.api.ops.impl.transforms.custom.Assign;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.ops.impl.transforms.same.Identity;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.shade.wstx.util.StringUtil;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
public class InferenceSession extends AbstractSession<INDArray, Pair<SameDiffOp,OpContext>> {
    private static final String SCOPE_PANIC_MSG = "If required, arrays in workspaces can be detached using INDArray.detach() before being passed to the SameDiff instance.\n" +
            "Alternatively, arrays defined in a workspace must be replaced after the workspace has been closed.";

    protected static final String KERAS_TRAIN_TEST = "keras_learning_phase";
    //freed array ids to track for allocation, sometimes SDValues contain dup arrays that get freed twice.
    //we track the ids to avoid double frees
    protected Set<Long> freedArrays = new LinkedHashSet<>();

    @Getter
    @Setter
    private SessionMemMgr mmgr;     //Used for allocating and deallocating memory
    /**
     * Array use tracker: What needs to happen before the array can be closed/released?
     * As the name suggests, the INDArrays are tracked using object identity, not equality
     */
    @Getter
    @Setter
    private AbstractDependencyTracker<SDValue, Dep> arrayUseTracker = new HashDependencyTracker<>();


    @Getter
    private Map<String,OpContext> opContexts = new LinkedHashMap<>();

    // DAG cache for avoiding expensive convergence process
    private final DAGCache dagCache = new DAGCache();

    public InferenceSession(@NonNull SameDiff sameDiff) {
        super(sameDiff);
        mmgr = new ArrayCacheMemoryMgr();
    }


    @SneakyThrows
    public ExecutionResult output(@NonNull List<String> variables,
                                  Map<String, INDArray> placeholderValues,
                                  Map<String, SDValue> otherPlaceHolderValues,
                                  MultiDataSet batch,
                                  Collection<String> requiredActivations,
                                  List<Listener> listeners, At at) {

        // Clear freed arrays tracking from previous execution
        freedArrays.clear();

        // Clear DAG cache to prevent unbounded growth with different output sets
        dagCache.clear();

        log.info("Executing forward pass for {} variables", variables.size());

        // Prepare all required outputs
        Set<String> allRequired = new LinkedHashSet<>(variables);
        if (requiredActivations != null) {
            allRequired.addAll(requiredActivations);
        }

        // Build corrected DAG with caching (replaces broken initSubgraph)
        ForwardExecutionDAG dag = dagCache.getOrCompute(allRequired, () -> {
            ForwardExecutionDAGBuilder builder = new ForwardExecutionDAGBuilder(sameDiff);
            return builder.buildForwardDAG(allRequired);
        });



        // Preprocess placeholders using existing logic
        Map<String, INDArray> processedPlaceholders = preprocessPlaceholders(placeholderValues, at);
        Map<String, SDValue> processedOtherPlaceholders = preprocessValuePlaceholders(otherPlaceHolderValues, at);
        // Execute with corrected ordering
        Map<String, SDValue> results = executeOperations(dag, processedPlaceholders,
                processedOtherPlaceholders, allRequired, listeners, at, batch);

        // Post-process results using existing logic
        Map<String, SDValue> finalResults = postProcessOutputValues(results);

        // Return only requested variables
        Map<String, SDValue> filteredResults = new HashMap<>();
        for (String var : variables) {
            if (finalResults.containsKey(var)) {
                filteredResults.put(var, finalResults.get(var));
            }
        }

        log.info("Forward pass completed: {} results", filteredResults.size());

        // Mark output dependencies as satisfied so buffers can be released before clearing tracker
        for (String outputVar : variables) {
            arrayUseTracker.markSatisfied(new ReqOutputDep(outputVar), true);
        }

        // Release any arrays that became releasable (will NOT release outputs we're returning)
        if (arrayUseTracker.hasNewAllSatisfied()) {
            arrayUseTracker.getNewAllSatisfiedList(); // Just pop them off the queue
        }

        // Clear array use tracker to prevent stale dependencies from accumulating
        arrayUseTracker.clear();

        // Close OpContext instances to prevent native memory leak
        for (OpContext ctx : opContexts.values()) {
            if (ctx != null) {
                ctx.close();
            }
        }
        opContexts.clear();

        // Close memory manager to release cached arrays
        if (mmgr != null) {
            try {
                mmgr.close();
            } catch (Exception e) {
                log.warn("Error closing memory manager: {}", e.getMessage());
            }
        }

        return ExecutionResult.builder().valueOutputs(filteredResults).build();
    }


    private Map<String, SDValue> executeOperations(ForwardExecutionDAG dag,
                                                   Map<String, INDArray> placeholderValues,
                                                   Map<String, SDValue> otherPlaceholderValues,
                                                   Set<String> allRequired,
                                                   List<Listener> listeners,
                                                   At at,
                                                   MultiDataSet batch) {

        Map<String, SDValue> variableValues = new LinkedHashMap<>();
        Map<String, SDValue> results = new LinkedHashMap<>();
        Set<String> completedOps = new LinkedHashSet<>();

        // Initialize constants, variables, and placeholders
        initializeValues(variableValues, dag, placeholderValues, otherPlaceholderValues);

        // Execute operations in corrected topological order
        List<ExecutionNode> executionOrder = dag.getFrameAwareExecutionOrder();

        for (ExecutionNode node : executionOrder) {
            if (node.getNodeType() == ExecutionNode.ExecutionNodeType.VARIABLE_INIT ||
                    node.getNodeType() == ExecutionNode.ExecutionNodeType.PLACEHOLDER_SET) {
                // Skip - already handled in initialization
                continue;
            }

            // Check dependencies are satisfied
            if (!node.isReadyToExecute(completedOps)) {
                Set<String> missing = new HashSet<>(node.getDependsOnOperations());
                missing.removeAll(completedOps);
                throw new IllegalStateException("Operation " + node.getOperationName() +
                        " not ready. Missing dependencies: " + missing);
            }

            // Execute the operation
            executeNode(node, variableValues, allRequired, listeners, at, batch);

            // Mark as completed
            completedOps.add(node.getOperationName());

            // Store results for requested outputs
            // After each operation execution, sync to nodeValueOutputs
            for (String outputVar : node.getOutputVariables()) {
                if (variableValues.containsKey(outputVar)) {
                    VarId vid = new VarId(outputVar, currentFrame, currentFrameIter, currParentFrame);
                    putNodeValue(variableValues.get(outputVar), vid);
                }
            }

            // Mark operation dependency as satisfied in arrayUseTracker
            Dep opDep = new OpDep(node.getOperationName(), OUTER_FRAME, 0, null);
            arrayUseTracker.markSatisfied(opDep, true);

            // Release any arrays that are no longer needed
            if (arrayUseTracker.hasNewAllSatisfied()) {
                List<SDValue> canClose = arrayUseTracker.getNewAllSatisfiedList();
                for (SDValue value : canClose) {
                    if (log.isTraceEnabled()) {
                        if (value.getSdValueType() == SDValueType.TENSOR) {
                            INDArray arr = value.getTensorValue();
                            log.trace("Releasing array after op {}: id={}, shape={}", node.getOperationName(), arr.getId(), Arrays.toString(arr.shape()));
                        }
                    }

                    switch(value.getSdValueType()) {
                        case TENSOR:
                            if (!freedArrays.contains(value.getTensorValue().getId())) {
                                mmgr.release(value.getTensorValue());
                                freedArrays.add(value.getTensorValue().getId());
                            }
                            break;
                        case LIST:
                            for (INDArray arr : value.getListValue()) {
                                if (arr != null && !freedArrays.contains(arr.getId())) {
                                    mmgr.release(arr);
                                    freedArrays.add(arr.getId());
                                }
                            }
                            break;
                    }
                }
            }
        }

        for(String output : allRequired) {
            if(!variableValues.containsKey(output)) {
                throw new IllegalStateException("Output: " + output + " missing from the final output!");
            }
            results.put(output,variableValues.get(output));
        }

        return results;
    }



    /**
     * Get all dependent values for a variable as a formatted string.
     *
     * @param variableValues Current variable values from execution
     * @param variableName Variable to get dependencies for
     * @return Formatted string with dependent values
     */
    public String getDependentValuesString(Map<String, SDValue> variableValues, String variableName) {
        Map<String, String> deps = getDependentValuesMap(variableValues, variableName);
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> entry : deps.entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Get all dependent values for a variable as a map.
     *
     * @param variableValues Current variable values from execution
     * @param variableName Variable to get dependencies for
     * @return Map of variable names to their values
     */
    public Map<String, String> getDependentValuesMap(Map<String, SDValue> variableValues, String variableName) {
        Map<String, String> result = new LinkedHashMap<>();
        Set<String> visited = new HashSet<>();
        collectDependentValues(variableValues, variableName, result, visited);
        return result;
    }

    private void collectDependentValues(Map<String, SDValue> variableValues, String varName,
                                        Map<String, String> result, Set<String> visited) {
        if (visited.contains(varName)) {
            return;
        }
        visited.add(varName);

        // Add this variable's value
        SDValue value = variableValues.get(varName);
        if (value != null) {
            result.put(varName, formatValue(value));
        }

        // Find the op that produces this variable
        for (SameDiffOp op : sameDiff.getOps().values()) {
            if (op.getOutputsOfOp() != null && op.getOutputsOfOp().contains(varName)) {
                // Collect values from all inputs
                if (op.getInputsToOp() != null) {
                    for (String input : op.getInputsToOp()) {
                        collectDependentValues(variableValues, input, result, visited);
                    }
                }
                break;
            }
        }
    }

    private String formatValue(SDValue value) {
        if (value.getSdValueType() == SDValueType.TENSOR) {
            INDArray arr = value.getTensorValue();
            if (arr == null) return "null";
            if (arr.isScalar()) return String.valueOf(arr.getDouble(0));
            return Arrays.toString(arr.shape()) + " = " + arr.toString().replaceAll("\\s+", " ").trim();
        } else if (value.getSdValueType() == SDValueType.LIST) {
            return "List[" + value.getListValue().size() + "]";
        }
        return value.toString();
    }
    private void initializeValues(Map<String, SDValue> variableValues,
                                  ForwardExecutionDAG dag,
                                  Map<String, INDArray> placeholderValues,
                                  Map<String, SDValue> otherPlaceholderValues) {

        // Initialize constants
        for (String constName : dag.getConstants()) {
            INDArray constValue = getConstantOrVariable(constName);
            if (constValue != null) {
                variableValues.put(constName, SDValue.create(constValue));
            }
        }

        // Initialize variables
        for (String varName : dag.getVariables()) {
            INDArray varValue = getConstantOrVariable(varName);
            if (varValue != null) {
                variableValues.put(varName, SDValue.create(varValue));
            }
        }

        // Initialize placeholders
        if (placeholderValues != null) {
            for (Map.Entry<String, INDArray> entry : placeholderValues.entrySet()) {
                variableValues.put(entry.getKey(), SDValue.create(entry.getValue()));
            }
        }

        if (otherPlaceholderValues != null) {
            variableValues.putAll(otherPlaceholderValues);
        }
    }

// =============================================================================
// OVERRIDE 4: Execute single node (add this to InferenceSession)
// =============================================================================

    private void executeNode(ExecutionNode node,
                             Map<String, SDValue> variableValues,
                             Set<String> allRequired,
                             List<Listener> listeners,
                             At at,
                             MultiDataSet batch) {

        String opName = node.getOperationName();
        log.trace("Executing operation: {}", opName);

        try {
            // Get the operation
            SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
            if (sameDiffOp == null) {
                throw new IllegalStateException("Operation not found: " + opName);
            }

            DifferentialFunction op = sameDiffOp.getOp();

            // Handle special control flow operations directly
            if (op instanceof Identity) {
                executeIdentityNode(node, variableValues);
                return;
            }

            if (op instanceof Switch) {
                executeSwitchNode(node, variableValues, op);
                return;
            }

            if (op instanceof Enter) {
                executeEnterNode(node, variableValues, op);
                return;
            }

            if (op instanceof Exit) {
                executeExitNode(node, variableValues, op);
                return;
            }

            if (op instanceof NextIteration) {
                executeNextIterationNode(node, variableValues, op);
                return;
            }

            if (op instanceof Merge) {
                executeMergeNode(node, variableValues, op);
                return;
            }

            if (op instanceof LoopCond) {
                executeLoopCondNode(node, variableValues, op);
                return;
            }

            if (op instanceof BaseTensorOp) {
                executeTensorArrayNode(node, variableValues, op);
                return;
            }

            // For regular operations, use the existing doExec infrastructure
            executeRegularOperation(node, variableValues, allRequired, listeners, at, batch);


        } catch (Exception e) {
            log.error("Failed to execute operation: {}", opName, e);
            throw new RuntimeException("Operation execution failed: " + opName, e);
        }
    }

    private void executeIdentityNode(ExecutionNode node, Map<String, SDValue> variableValues) {
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
            throw new IllegalStateException("Input variable " + inputVar + " not found for Identity operation " + opName);
        }

        variableValues.put(outputVar, inputValue);
    }

    private void executeSwitchNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
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

        if (dataValue == null || predicateValue == null) {
            throw new IllegalStateException("Switch inputs not available: data=" + (dataValue != null) +
                    ", predicate=" + (predicateValue != null));
        }

        INDArray predicate = predicateValue.getTensorValue();
        boolean condition = predicate.getDouble(0) != 0.0;

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

    private void executeEnterNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
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

    private void executeExitNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
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

    private void executeNextIterationNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
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

    private void executeMergeNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        List<String> inputs = node.getInputVariables();
        List<String> outputs = node.getOutputVariables();

        if (inputs.size() < 2 || outputs.isEmpty()) {
            throw new IllegalStateException("Merge operation requires at least 2 inputs and 1 output");
        }

        String outputVar = outputs.get(0);

        // Find the first available input (standard Merge behavior)
        for (String inputVar : inputs) {
            SDValue inputValue = variableValues.get(inputVar);
            if (inputValue != null) {
                variableValues.put(outputVar, inputValue);
                return;
            }
        }

        throw new IllegalStateException("No inputs available for Merge operation " + node.getOperationName());
    }

    private void executeLoopCondNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
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

    private void executeTensorArrayNode(ExecutionNode node, Map<String, SDValue> variableValues, DifferentialFunction op) {
        // Convert to VarId-based approach for tensor array operations
        // This maintains compatibility with existing tensor array handling

        FrameIter frameIter = new FrameIter(OUTER_FRAME, 0, null);
        Set<VarId> opInputs = new HashSet<>();
        Set<VarId> allIterInputs = new HashSet<>();

        // Convert string-based inputs to VarIds for tensor array compatibility
        for (String inputVar : node.getInputVariables()) {
            VarId vid = new VarId(inputVar, OUTER_FRAME, 0, null);
            // Store the variable value in nodeValueOutputs for tensor array ops
            SDValue value = variableValues.get(inputVar);
            if (value != null) {
                putNodeValue(value, vid);
            }
            opInputs.add(vid);
        }

        try {
            ExecutionResult result = getOutputsHelperTensorArrayOps(op, frameIter, opInputs, allIterInputs, variableValues);

            // Store results back in variableValues
            if (result.hasValues()) {
                Map<String, SDValue> outputs = result.getValueOutputs();
                for (Map.Entry<String, SDValue> entry : outputs.entrySet()) {
                    variableValues.put(entry.getKey(), entry.getValue());
                }
            } else if (result.hasSingle()) {
                List<String> outputNames = node.getOutputVariables();
                for (int i = 0; i < outputNames.size() && i < result.numResults(); i++) {
                    INDArray resultArray = result.resultAt(i);
                    if (resultArray != null) {
                        variableValues.put(outputNames.get(i), SDValue.create(resultArray));
                    }
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to execute tensor array operation " + node.getOperationName(), e);
        }
    }

    private void executeRegularOperation(ExecutionNode node, Map<String, SDValue> variableValues,
                                         Set<String> allRequired, List<Listener> listeners,
                                         At at, MultiDataSet batch) {
        String opName = node.getOperationName();
        SameDiffOp sameDiffOp = sameDiff.getOps().get(opName);
        DifferentialFunction op = sameDiffOp.getOp();

        // Create OpContext for the operation
        OpContext opContext = opContexts.get(opName);
        if (opContext == null) {
            opContext = Nd4j.getExecutioner().buildContext();
            opContexts.put(opName, opContext);
        }

        // Prepare inputs
        String[] argNames = op.argNames();
        if (argNames != null && argNames.length > 0) {
            INDArray[] inputArrays = new INDArray[argNames.length];

            for (int i = 0; i < argNames.length; i++) {
                String argName = argNames[i];
                SDValue argValue = variableValues.get(argName);

                if (argValue == null) {
                    // Try to get from constants/variables
                    SDVariable variable = sameDiff.getVariable(argName);
                    if (variable != null) {
                        if (variable.isConstant() || variable.getVariableType() == VariableType.VARIABLE) {
                            INDArray arr = getConstantOrVariable(argName);
                            inputArrays[i] = arr;
                            continue;
                        }
                    }
                    throw new IllegalStateException("Input " + argName + " not found for operation " + opName);
                }

                switch (argValue.getSdValueType()) {
                    case TENSOR:
                        inputArrays[i] = argValue.getTensorValue();
                        break;
                    case LIST:
                        // For list values, try to use the first non-null element
                        List<INDArray> list = argValue.getListValue();
                        if (!list.isEmpty()) {
                            for (INDArray arr : list) {
                                if (arr != null) {
                                    inputArrays[i] = arr;
                                    break;
                                }
                            }
                        }
                        if (inputArrays[i] == null) {
                            throw new IllegalStateException("No valid array found in list for input " + argName);
                        }
                        break;
                    default:
                        throw new IllegalStateException("Unsupported SDValue type: " + argValue.getSdValueType());
                }
            }

            opContext.setInputArrays(inputArrays);
        }

        // Hanle different operation types
        if (op instanceof CustomOp) {
            executeCustomOp((CustomOp) op, opContext, node, variableValues, allRequired);
        } else if (op instanceof Op) {
            executeStandardOp((Op) op, opContext, node, variableValues, allRequired);
        } else {
            throw new UnsupportedOperationException("Unsupported operation type: " + op.getClass().getName());
        }
    }

    private void executeCustomOp(CustomOp customOp, OpContext opContext, ExecutionNode node,
                                 Map<String, SDValue> variableValues, Set<String> allRequired) {

        DynamicCustomOp dynOp = (DynamicCustomOp) customOp;

        // Set op arguments
        opContext.setInputArrays(customOp.inputArguments());
        opContext.setIArguments(dynOp.iArgs());
        opContext.setTArguments(dynOp.tArgs());
        opContext.setDArguments(dynOp.dArgs());
        opContext.setBArguments(dynOp.bArgs());
        // Calculate output shapes
        List<DataBuffer> outShape = dynOp.calculateOutputShape(opContext);
        if (outShape == null || outShape.isEmpty()) {
            throw new IllegalStateException("No output shapes calculated for op: " + customOp.opName());
        }

        // Allocate output arrays
        List<String> outputNames = node.getOutputVariables();
        INDArray[] outputArrays = new INDArray[outShape.size()];

        for (int i = 0; i < outShape.size(); i++) {
            DataBuffer shapeBuffer = outShape.get(i);
            long[] shape = shapeBuffer.asLong();

            // Get output datatype from variable definition
            DataType dt = DataType.FLOAT; // default
            if (i < outputNames.size()) {
                SDVariable outVar = sameDiff.getVariable(outputNames.get(i));
                if (outVar != null) {
                    dt = outVar.dataType();
                }
            }

            boolean isOutput = allRequired.contains(outputNames.get(i));
            outputArrays[i] = mmgr.allocate(isOutput, dt, Shape.shape(shape));
        }

        opContext.setOutputArrays(outputArrays);

        // Execute the operation
        Nd4j.exec(dynOp, opContext);

        // Store results and track for deallocation
        for (int i = 0; i < outputArrays.length && i < outputNames.size(); i++) {
            SDValue outputValue = SDValue.create(outputArrays[i]);
            variableValues.put(outputNames.get(i), outputValue);

            // Add to arrayUseTracker so it can be released when no longer needed
            String varName = outputNames.get(i);
            if (allRequired.contains(varName)) {
                // This is a final output, don't deallocate
                arrayUseTracker.addDependency(outputValue, new ReqOutputDep(varName));
            } else {
                // Check what ops need this array
                Variable v = sameDiff.getVariables().get(varName);
                if (v != null && v.getInputsForOp() != null) {
                    for (String consumerOp : v.getInputsForOp()) {
                        arrayUseTracker.addDependency(outputValue, new OpDep(consumerOp, OUTER_FRAME, 0, null));
                    }
                } else {
                    // No consumers, can release immediately after this op
                    if (!freedArrays.contains(outputArrays[i].getId())) {
                        mmgr.release(outputArrays[i]);
                        freedArrays.add(outputArrays[i].getId());
                    }
                }
            }
        }
    }

    private void executeStandardOp(Op op, OpContext opContext, ExecutionNode node,
                                   Map<String, SDValue> variableValues, Set<String> allRequired) {

        // Handle reduction operations with axis
        if (op instanceof ReduceOp && ((ReduceOp) op).getOpType() != Op.Type.REDUCE3) {
            handleReduceOpAxis(op, opContext);
        }

        // Handle scalar operations
        if (op instanceof ScalarOp && opContext.getInputArrays().size() >= 2) {
            INDArray scalar = opContext.getInputArray(1);
            if (scalar.isScalar()) {
                ((ScalarOp) op).setScalar(scalar);
            }
        }

        // Calculate output shape
        List<DataBuffer> outputShape = ((BaseOp) op).calculateOutputShape(opContext);
        if (outputShape == null || outputShape.isEmpty()) {
            throw new IllegalStateException("No output shape calculated for op: " + op.opName());
        }

        // Allocate output array
        DataBuffer shapeBuffer = outputShape.get(0);
        List<String> outputNames = node.getOutputVariables();
        boolean isOutput = !outputNames.isEmpty() && allRequired.contains(outputNames.get(0));
        INDArray outputArray = mmgr.allocateFromDescriptor(isOutput, shapeBuffer);

        opContext.setOutputArray(0, outputArray);

        // Execute the operation
        Nd4j.exec(op, opContext);

        // Store result and track for deallocation
        if (!outputNames.isEmpty()) {
            SDValue outputValue = SDValue.create(outputArray);
            String varName = outputNames.get(0);
            variableValues.put(varName, outputValue);

            // Add to arrayUseTracker so it can be released when no longer needed
            if (allRequired.contains(varName)) {
                // This is a final output, don't deallocate
                arrayUseTracker.addDependency(outputValue, new ReqOutputDep(varName));
            } else {
                // Check what ops need this array
                Variable v = sameDiff.getVariables().get(varName);
                if (v != null && v.getInputsForOp() != null) {
                    for (String consumerOp : v.getInputsForOp()) {
                        arrayUseTracker.addDependency(outputValue, new OpDep(consumerOp, OUTER_FRAME, 0, null));
                    }
                } else {
                    // No consumers, can release immediately after this op
                    if (!freedArrays.contains(outputArray.getId())) {
                        mmgr.release(outputArray);
                        freedArrays.add(outputArray.getId());
                    }
                }
            }
        }
    }

    private void handleReduceOpAxis(Op op, OpContext opContext) {
        if (opContext.getInputArrays().size() >= 2) {
            INDArray axisArray = opContext.getInputArray(1);
            if (!axisArray.isEmpty()) {
                long[] axis = axisArray.toLongVector();
                int rank = opContext.getInputArray(0).rank();
                axis = Shape.normalizeAxis(rank, axis);
                ((DifferentialFunction) op).setDimensions(axis);
                ((BaseReduceOp) op).setEmptyReduce(false);
            } else {
                ((DifferentialFunction) op).setDimensions(null);
                ((BaseReduceOp) op).setEmptyReduce(true);
            }
        }
    }


    @Override
    protected Map<String, INDArray> preprocessPlaceholders(Map<String, INDArray> placeholders, At at) {
        arrayUseTracker.clear();

        //We'll also use this method as a "pre execution" hook-in, to mark variables as something we should never deallocate
        //This occurs by never marking these "ConstantDep" and "VariableDep" instances as satisfied, so there's always
        // an unsatisfied dependency for them in the array use tracker
        //TODO we shouldn't be clearing this on every single iteration, in 99.5% of cases variables will be same as last iteration...
        for (SDVariable v : sameDiff.variables()) {
            if (v.getVariableType() == VariableType.CONSTANT) {
                arrayUseTracker.addDependency(SDValue.create(v.getArr()), new ConstantDep(v.name()));
            } else if (v.getVariableType() == VariableType.VARIABLE) {
                arrayUseTracker.addDependency(SDValue.create(v.getArr()), new VariableDep(v.name()));
            }
        }

        //Workaround for some TF/Keras based models that require explicit train/test as a placeholder
        boolean kerasWorkaround = false;
        List<String> phs = sameDiff.inputs();
        if (phs != null && !phs.isEmpty()) {
            for (String s : phs) {
                if (s.endsWith(KERAS_TRAIN_TEST) && !placeholders.containsKey(s)) {
                    // The behaviour of some Keras layers (like GRU) differs depending on whether the model is training.
                    // We provide this value directly, unless the user has provided this manually
                    INDArray scalar = mmgr.allocate(false, DataType.BOOL).assign(at.operation().isTrainingPhase());
                    placeholders = new HashMap<>(placeholders); //Array might be singleton, or otherwise unmodifiable
                    placeholders.put(s, scalar);
                    kerasWorkaround = true;
                }
            }
        }


        if (placeholders == null || placeholders.isEmpty()) {
            return placeholders;
        }

        //Handle casting of the input array automatically.
        //The idea here is to avoid unexpected errors if the user (for example) tries to perform inference with a double
        // array for a float placeholder
        //TODO eventually we might have ops that support multiple input types, and hence won't need this casting
        Map<String, INDArray> out = new HashMap<>();
        for (Map.Entry<String, INDArray> e : placeholders.entrySet()) {
            Preconditions.checkState(sameDiff.hasVariable(e.getKey()), "Invalid placeholder passed for execution: " +
                    "No variable/placeholder with name %s exists", e.getKey());
            INDArray arr = e.getValue();
            SDValue arrValue = SDValue.create(arr);
            //First: check workspaces
            if (arr.isAttached()) {
                MemoryWorkspace ws = arr.data() == null ? null : arr.data().getParentWorkspace();
                if (ws != null && ws.getWorkspaceType() != MemoryWorkspace.Type.CIRCULAR) {
                    if (!ws.isScopeActive()) {
                        throw new ND4JIllegalStateException("Placeholder \"" + e.getKey() + "\" array uses leaked workspace pointer from workspace ["
                                + ws.getId() + "]: Workspace the array was defined in is no longer open.\nAll open workspaces: " + DefaultOpExecutioner.allOpenWorkspaces()
                                + "\n" + SCOPE_PANIC_MSG);
                    }

                    if (ws.getGenerationId() != arr.data().getGenerationId())
                        throw new ND4JIllegalStateException("Placeholder \"" + e.getKey() + "\" array uses outdated workspace pointer from workspace ["
                                + ws.getId() + "]: Workspace array was defined in has been closed and reopened at least once since array creation. Array WS iteration: " +
                                arr.data().getGenerationId() + ". Workspace current iteration: " +
                                ws.getGenerationId() + "\nAll open workspaces: " + DefaultOpExecutioner.allOpenWorkspaces() + "\n" + SCOPE_PANIC_MSG);
                }
            }


            //Second: cast the input to the required type
            //TODO For the casting case, we SHOULD actually deallocate this when we're done with it, which is usually sooner than "exec done"
            DataType dt = sameDiff.getVariable(e.getKey()).dataType();
            if (kerasWorkaround && e.getKey().endsWith(KERAS_TRAIN_TEST)) {
                arrayUseTracker.addDependency(arrValue, new ExecDoneDep());
            } else if (arr.dataType() == dt) {
                //Mark as a placeholder array in the array use tracker, so we never deallocate this array...
                arrayUseTracker.addDependency(arrValue,
                        PlaceholderDep.builder()
                        .phName(e.getKey())
                        .frame(currentFrame).parentFrame(currParentFrame)
                        .build());
            } else {
                INDArray cast = mmgr.allocate(false, dt, arr.shape());
                cast.assign(arr);
                arr = cast;
                //This array CAN be deallocated once consumed, because of the cast
                //TODO we can likely close this sooner
                arrayUseTracker.addDependency(arrValue, ExecDoneDep.builder()
                        .frame(currentFrame)
                        .parentFrame(currParentFrame)
                        .build());
            }
            out.put(e.getKey(), arr);
        }

        return out;
    }

    @Override
    protected Map<String, SDValue> postProcessOutputValues(Map<String, SDValue> output) {
        //For any queued (not yet processed) ops - mark them as satisfied, so we can deallocate any arrays
        // that are waiting on them
        if (dt.hasNewAllSatisfied()) {
            List<ExecStep> execSteps = dt.getNewAllSatisfiedList();
            for (ExecStep es : execSteps) {
                if (es.getType() == ExecType.OP) {
                    OpDep od = new OpDep(es.getName(), es.getFrameIter().getFrame(), es.getFrameIter().getIteration(), es.getFrameIter().getParentFrame());
                    arrayUseTracker.markSatisfied(od, true);
                }
            }
        }

        //Also mark "end of execution" for array dependency tracker. Mainly used for TensorArray arrays at present.
        //TODO Optimize for reduced memory for some TensorArray operations - i.e., close/deallocate earlier
        arrayUseTracker.markSatisfied(new ExecDoneDep(), true);
        if (arrayUseTracker.hasNewAllSatisfied()) {
            List<SDValue> l = arrayUseTracker.getNewAllSatisfiedList();
            for (SDValue value : l) {
                switch(value.getSdValueType()) {
                    case LIST:
                        for(INDArray arr : value.getListValue())
                            if(arr != null && !freedArrays.contains(arr.getId())) {
                                mmgr.release(arr);
                                freedArrays.add(arr.getId());
                            }
                        break;
                    case TENSOR:
                        if(!freedArrays.contains(value.getTensorValue().getId())) {
                            mmgr.release(value.getTensorValue());
                            freedArrays.add(value.getTensorValue().getId());
                        }
                        break;
                }
            }
        }

        return output;
    }



    @Override
    public ExecutionResult getOutputs(Pair<SameDiffOp, OpContext> opPair,
                                      FrameIter outputFrameIter,
                                      Set<VarId> opInputs,
                                      Set<VarId> allIterInputs,
                                      Set<String> constAndPhInputs,
                                      List<Listener> listeners,
                                      At at, MultiDataSet batch,
                                      Set<String> allReqVariables,
                                      Map<String, SDValue> otherPlaceHolders) {
        SameDiffOp op = opPair.getFirst();
        at.setFrameIter(outputFrameIter);
        if (listeners != null && listeners.size() > 0) {
            SameDiffOp sdOp = sameDiff.getOps().get(op.getOp().getOwnName());
            for (Listener l : listeners) {
                if (l.isActive(at.operation()))
                    l.preOpExecution(sameDiff, at, sdOp, opPair.getSecond());
            }
        }

        if(sameDiff.isDebugMode()) {
            log.info("Executing samediff op: " + op.getName());
        }

        ExecutionResult out = doExec(
                op.getOp(),
                opPair.getRight(),
                outputFrameIter, opInputs,
                allIterInputs,
                constAndPhInputs,
                otherPlaceHolders);
        List<String> opOutNames = op.getOutputsOfOp();

        if (log.isTraceEnabled()) {
            StringBuilder sb = new StringBuilder();
            sb.append(op.getName()).append(" - ").append(outputFrameIter).append(" outputs: ");
            for (int i = 0; i < out.numResults(); i++) {
                if (i > 0)
                    sb.append(", ");
                if(out.hasSingle())
                    sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                            out.resultAt(i) == null ? null :  out.resultAt(i) .getId()).append(")");

                else if(out.hasValues()) {
                    SDValue value = out.valueWithKeyAtIndex(i, false);
                    //append either the list of associated array ids or the singular one similar to the singular array case
                    String append = value != null && value.getSdValueType() == SDValueType.LIST ? StringUtil.concatEntries(value.getListValue().stream()
                            .map(input -> input == null ? "" : input.getId()).collect(Collectors.toList()),",",",") : value != null ? String.valueOf(value.getTensorValue().getId()) : null;
                    sb.append("(").append(i).append(" - ").append(opOutNames.get(i)).append(" = ").append(
                            value == null ? null : append).append(")");

                }
            }
            log.trace(sb.toString());
        }

        //Call listeners, before we (maybe) deallocate input arrays
        if (listeners != null && listeners.size() > 0) {
            Map<String, INDArray> namedOuts = null;

            for (Listener l : listeners) {
                if (l.isActive(at.operation())) {
                    //Lazily create map, only if required
                    if (namedOuts == null) {
                        Map<String, INDArray> namedOutsBuilder = new HashMap<>();

                        for (int i = 0; i < out.numResults(); i++)
                            namedOutsBuilder.put(op.outputsOfOp.get(i), out.resultAt(i));
                        namedOuts = Collections.unmodifiableMap(namedOutsBuilder);
                    }


                    l.opExecution(sameDiff, at, batch, op, opPair.getSecond(), out.outputsToArray(opOutNames));

                    for (String varName : namedOuts.keySet()) {
                        l.activationAvailable(sameDiff, at, batch, op, varName, namedOuts.get(varName));
                    }
                }
            }
        }
        op.getOp().clearArrays();
        if(opPair.getSecond() != null)
            opPair.getSecond().purge();


        //Record array uses for memory management/deallocation
        SameDiffOp o = sameDiff.getOps().get(op.getName());
        List<String> outVarNames = o.getOutputsOfOp();
        for (int i = 0; i < out.numResults(); i++) {
            if (out.hasSingle() && out.resultAt(i) == null   || out.hasValues()
                    && out.valueWithKeyAtIndex(i, false) == null
                    && o.getOp() instanceof Switch)
                continue;   //Switch case: we only ever get one of 2 outputs, other is null (branch not executed)
            String name = outVarNames.get(i);
            Variable v = sameDiff.getVariables().get(name);
            List<String> inputsForOps = v.getInputsForOp();
            if (inputsForOps != null) {
                for (String opName : inputsForOps) {
                    //Only add dependencies if we actually need the op this feeds into, otherwise the dependency
                    // will never be marked as satisfied
                    if (!subgraphOps.contains(opName))
                        continue;

                    SameDiffOp forOp = sameDiff.getOps().get(opName);

                    //TODO do switch or merge need special handling also?
                    if (forOp.getOp() instanceof Enter) {
                        Enter e = (Enter) forOp.getOp();
                        if (e.isConstant()) {
                        /*
                        Constant enter case: Need to keep this array around for the entire duration of the frame, including
                        any nested frames, and all iterations.
                        Unfortunately, we don't know exactly when we're done with a frame for good
                        This isn't a great solution, but other possibilities (frame close, trying to detect all exit ops,
                        detecting return to parent frame, etc all fail in certain circumstances, such as due to control dependencies
                        on variables).
                         */
                            Dep d = new ExecDoneDep();
                            addToArrayTracker(out,i,d);
                        } else {
                            Dep d = new OpDep(opName, e.getFrameName(), 0, outputFrameIter);
                            addToArrayTracker(out,i,d);
                        }
                    } else if (forOp.getOp() instanceof NextIteration) {
                        //The array is needed by the NEXT iteration op, not the current one
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration() + 1, outputFrameIter.getParentFrame());
                        addToArrayTracker(out,i,d);
                    } else if (forOp.getOp() instanceof Exit) {
                        //The array is needed at the EXIT frame (i.e., parent frame), not the inner/just executed one
                        FrameIter fi = outputFrameIter.getParentFrame();
                        Dep d = new OpDep(opName, fi.getFrame(), fi.getIteration(), fi.getParentFrame());
                        addToArrayTracker(out,i,d);
                    } else {
                        //All other ops...
                        Dep d = new OpDep(opName, outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
                        addToArrayTracker(out,i,d);
                    }
                }
            }

            if (OUTER_FRAME.equals(outputFrameIter.getFrame()) && allReqVariables.contains(name)) {
                //This variable is an output, record that in the array use tracker, so we don't deallocate it
                //the specific value here
                addToArrayTracker(out,i,new ReqOutputDep(name));
            } else if ((inputsForOps == null || inputsForOps.isEmpty()) && out.getValueOutputs() != null && !arrayUseTracker.hasDependency(out.valueWithKeyAtIndex(i,false))) {
                //This particular array is not actually needed anywhere, so we can deallocate in immediately
                //Possibly only a control dependency, or only one of the outputs of a multi-output op is used
                SDValue array = out.valueWithKeyAtIndex(i, false);
                if (log.isTraceEnabled()) {
                    if(array != null && array.getTensorValue() != null)
                        log.trace("Found array id {} (output of {}) not required anywhere, deallocating", array.getTensorValue().getId(), o.getName());
                }

                if(!outVarNames.contains(name) && array != null && array.getTensorValue() != null && !freedArrays.contains(array.getTensorValue().getId())) {
                    mmgr.release(array.getTensorValue());
                    freedArrays.add(array.getTensorValue().getId());
                }
            } else if ((inputsForOps == null || inputsForOps.isEmpty()) && out.getOutputs() != null && !arrayUseTracker.hasDependency(SDValue.create(out.resultAt(i)))) {
                //This particular array is not actually needed anywhere, so we can deallocate in immediately
                //Possibly only a control dependency, or only one of the outputs of a multi-output op is used
                INDArray array = out.resultAt(i);
                if (log.isTraceEnabled()) {
                    if(array != null && array != null)
                        log.trace("Found array id {} (output of {}) not required anywhere, deallocating", array.getId(), o.getName());
                }

                if(!outVarNames.contains(name) && array != null && !freedArrays.contains(array.getId())) {
                    mmgr.release(array);
                    freedArrays.add(array.getId());
                }
            }
        }

        //Mark current op dependency as satisfied...
        Dep d = new OpDep(op.getName(), outputFrameIter.getFrame(), outputFrameIter.getIteration(), outputFrameIter.getParentFrame());
        arrayUseTracker.markSatisfied(d, true);


        //Close any no longer required arrays
        if (arrayUseTracker.hasNewAllSatisfied()) {
            List<SDValue> canClose = arrayUseTracker.getNewAllSatisfiedList();
            for (SDValue value : canClose) {
                if (log.isTraceEnabled()) {
                    if(value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray arr = value.getTensorValue();
                        log.trace("Closing array... id={}, {}", arr.getId(), arr.shapeInfoToString());

                    }
                }

                //don't free anything that's an output
                boolean containsOutput = false;
                for(String output : outVarNames) {
                    if(op.getOutputsOfOp().contains(output)) {
                        containsOutput = true;
                    }
                }

                if(!(op.getOp() instanceof Switch))
                    switch(value.getSdValueType()) {
                        case TENSOR:
                            if(!freedArrays.contains(value.getTensorValue().getId()) &&
                                    !containsOutput) {
                                mmgr.release(value.getTensorValue());
                                freedArrays.add(value.getTensorValue().getId());
                            }
                            break;
                        case LIST:
                            for(INDArray arr : value.getListValue())
                                if(arr != null && !freedArrays.contains(arr.getId()) && !containsOutput) {
                                    mmgr.release(arr);
                                    freedArrays.add(arr.getId());
                                }
                            break;
                    }

            }
        }

        return out;
    }


    private void addToArrayTracker(ExecutionResult out,int i,Dep d) {
        if(out.hasSingle()) {
            arrayUseTracker.addDependency(SDValue.create(out.resultOrValueAt(i,false)), d);       //Op defined by "d" needs to be executed before specified array can be closed
        } else {
            arrayUseTracker.addDependency(out.valueWithKeyAtIndex(i,false),d);
        }
    }

    public ExecutionResult doExec(DifferentialFunction op,
                                  OpContext opContext,
                                  FrameIter outputFrameIter,
                                  Set<VarId> opInputs, Set<VarId> allIterInputs,
                                  Set<String> constAndPhInputs,
                                  Map<String, SDValue> otherPlaceHolders) {

        int totalInputs = (opInputs == null ? 0 : opInputs.size()) + (constAndPhInputs == null ? 0 : constAndPhInputs.size())
                + (allIterInputs == null ? 0 : allIterInputs.size());

        boolean constPhInput = (opInputs == null || opInputs.size() == 0) && (allIterInputs == null || allIterInputs.size() == 0);

        // Initialize visualization if enabled - mimic output() method
        if (visualizationEnabled && visualizer != null) {
            // Prepare visualization data
            List<String> stepInputs = getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs);
            List<String> stepOutputs = getStepOutputsForVisualization(op);
            String executionStatus = "INITIALIZING";
            String detailedStatus = String.format("Operation: %s, Type: %s, Inputs: %d",
                    op.getOwnName(), op.getClass().getSimpleName(), totalInputs);

            visualizer.recordStep(
                    ExecType.OP,
                    op.getOwnName(),
                    outputFrameIter,
                    stepInputs,
                    stepOutputs,
                    executionStatus + " | " + detailedStatus
            );
        }

        // Execution status tracking for visualization
        String executionStatus = "SUCCESS";
        String detailedStatus = "";
        List<String> outputNames = new ArrayList<>();

        try {
            if (op instanceof Identity) {
                Identity i = (Identity) op;
                String[] argNames = i.argNames();
                Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in identity op, got %s", (Object) argNames);
                VarId vid = outputFrameIter.toVarId(argNames[0]);
                SDValue orig = getSdValue(vid);

                executionStatus = "SUCCESS";
                detailedStatus = "Identity passthrough";
                outputNames.add(vid.getVariable());

                ExecutionResult result = ExecutionResult.createValue(vid.getVariable(), orig);

                // Record successful execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames[0]),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Switch) {
                Switch s = (Switch) op;
                String[] argNames = s.argNames();       //Order: input, boolean array
                VarId vidPredicate = outputFrameIter.toVarId(argNames[1]);
                SDValue sdValuePred = getSdValue(vidPredicate);
                INDArray predicate = sdValuePred.getSdValueType() == SDValueType.LIST ? sdValuePred.getListValue().get(0) :
                        sdValuePred.getTensorValue();
                if(predicate != null && predicate.isEmpty()) {
                    predicate = Nd4j.scalar(false);
                }
                if(predicate == null && !constAndPhInputs.isEmpty() && constAndPhInputs.contains(argNames[1])) {
                    //Constant predicate...
                    predicate = getTensorFromOutputs(new VarId(argNames[1], OUTER_FRAME, 0, null));
                }
                Preconditions.checkNotNull(predicate, "Error during graph execution: Predicate array was null. VarId=%s", vidPredicate);
                Preconditions.checkState(predicate.isScalar() && predicate.dataType() == DataType.BOOL, "Expected boolean predicate: got %ndSInfo", predicate);
                VarId vid = outputFrameIter.toVarId(argNames[0]);
                SDValue sdValue = getSdValue(vid);
                Map<String,SDValue> values = new LinkedHashMap<>();
                ExecutionResult.ExecutionResultBuilder executionResultBuilder = ExecutionResult.builder()
                        .valueOutputs(values);

                boolean predicateValue = predicate.getDouble(0) != 0.0;
                String branchTaken = predicateValue ? "RIGHT" : "LEFT";
                executionStatus = "SWITCH_" + branchTaken;
                detailedStatus = String.format("SWITCH decision: %s branch taken (frame: %s, iter: %d) | Exec count: 1, Switches: 1",
                        branchTaken, outputFrameIter.getFrame(), outputFrameIter.getIteration());

                if (predicate.getDouble(0) == 0.0) {
                    //tensorflow import case
                    if(vid.getVariable().equals(vidPredicate.getVariable())) {
                        SDValue sdValue1 = SDValue.create(Arrays.asList(sdValue.getTensorValue(), null));
                        values.put(vidPredicate.getVariable(),sdValue1);
                        putNodeValue(sdValue1,vid);
                        VarId varId1 = new VarId(vid.getVariable() + ":1", vid.getFrame(), vid.getIteration(),vid.getParentFrame());
                        putNodeValue(sdValue1,varId1);
                        outputNames.add(vid.getVariable());
                    } else {
                        values.put(vid.getVariable(),sdValue);
                        values.put(vidPredicate.getVariable(),null);
                        outputNames.add(vid.getVariable());
                    }
                } else {
                    //tensorflow import case
                    if(vid.getVariable().equals(vidPredicate.getVariable())) {
                        SDValue sdValue1 = SDValue.create(Arrays.asList(null,sdValue.getTensorValue()));
                        values.put(vidPredicate.getVariable(),sdValue1);
                        values.put(vidPredicate.getVariable() + ":1",sdValue1);
                        outputNames.add(vidPredicate.getVariable());
                    } else {
                        values.put(vid.getVariable(),null);
                        values.put(vidPredicate.getVariable(),sdValue);
                        outputNames.add(vidPredicate.getVariable());
                    }
                }

                ExecutionResult result = executionResultBuilder.build();

                // Record switch execution with enhanced details
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Enter) {
                Enter e = (Enter) op;
                String[] input = e.argNames();
                Preconditions.checkState(input.length == 1, "Expected only 1 arg name for enter op: got %s", (Object) input);
                Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for Enter op \"%s\", got %s+%s", e.getOwnName(), opInputs, constAndPhInputs);

                VarId inputVarId;
                if (constPhInput) {
                    inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
                } else if (allIterInputs != null && allIterInputs.size() > 0) {
                    inputVarId = allIterInputs.iterator().next();
                } else {
                    inputVarId = opInputs.iterator().next();
                }

                inputVarId.setVariable(VariableUtils.stripVarSuffix(inputVarId.getVariable()));

                // Create explicit dependency aliases for cross-frame access
                SameDiffOp enterOp = sameDiff.getOps().get(e.getOwnName());
                List<String> enterOutputs = enterOp.getOutputsOfOp();
                String outFrame = e.getFrameName();
                FrameIter enterOutFrameIter = new FrameIter(outFrame, 0, outputFrameIter);

                if (enterOutputs != null) {
                    for (String outputVar : enterOutputs) {
                        String inputVar = enterOp.getInputsToOp().get(0);

                        ExecStep expectedStep = new ExecStep(ExecType.OP, outputVar, enterOutFrameIter);
                        ExecStep actualStep = new ExecStep(ExecType.OP, e.getOwnName(), enterOutFrameIter);
                        dt.createDependeeAlias(expectedStep, actualStep);

                        log.debug("Created Enter dependency alias: {} -> {} (frame: {} -> {})",
                                outputVar, inputVar, outputFrameIter.getFrame(), outFrame);
                    }

                    // Handle frame transition to ensure cross-frame dependencies are properly established
                    handleFrameTransition(e.getOwnName(), outputFrameIter, enterOutFrameIter, enterOutputs);

                    // Validate that dependent Merge operations will be able to find the Enter outputs
                    validateMergeDependencies(enterOutFrameIter, enterOutputs);
                }

                ExecutionResult result;
                if(nodeValueOutputs.containsKey(inputVarId)) {
                    SDValue value = getSdValue(inputVarId);
                    if(value != null && value.getSdValueType() == SDValueType.LIST) {
                        result = ExecutionResult.createValue(inputVarId.getVariable(), value);
                    } else if(value != null &&  value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray inArr = getTensorFromOutputs(inputVarId);
                        if (inArr == null) {
                            Preconditions.throwStateEx("Could not find array for Enter operation %s with output %s (frame=%s, iteration=%s)",
                                    op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), enterOutFrameIter.getFrame(), enterOutFrameIter.getIteration());
                        }
                        result = ExecutionResult.createFrom(Arrays.asList(inputVarId.getVariable()),new INDArray[]{inArr});
                    } else {
                        throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + inputVarId);
                    }
                } else {
                    INDArray inArr = getTensorFromOutputs(inputVarId);
                    if (inArr == null) {
                        Preconditions.throwStateEx("Could not find array for Enter operation %s with output %s (frame=%s, iteration=%s)",
                                op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), enterOutFrameIter.getFrame(), enterOutFrameIter.getIteration());
                    }
                    result = ExecutionResult.createFrom(Arrays.asList(inputVarId.getVariable()),new INDArray[]{inArr});
                }

                return result;
            }else if (op instanceof Exit) {
                //Exit node forwards input to parent frame
                VarId inputVarId;
                if (constPhInput) {
                    //Constant or placeholder
                    inputVarId = new VarId(constAndPhInputs.iterator().next(), OUTER_FRAME, 0, null);
                } else if (allIterInputs != null && allIterInputs.size() > 0) {
                    inputVarId = allIterInputs.iterator().next();
                } else {
                    inputVarId = opInputs.iterator().next();
                }

                executionStatus = "EXIT_FRAME";
                detailedStatus = String.format("EXIT from frame '%s' (iter: %d) | Variables exiting: %d",
                        outputFrameIter.getFrame(), outputFrameIter.getIteration(), 1);
                outputNames.add(inputVarId.getVariable());

                SDValue sdValue = getSdValue(inputVarId);
                ExecutionResult result = ExecutionResult.createValue(inputVarId.getVariable(), sdValue);

                // Record exit execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(inputVarId.getVariable()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof NextIteration) {
                //NextIteration op: forwards its single input to the output of the current frame, but increments the iteration number
                Preconditions.checkState(totalInputs == 1, "Expected exactly 1 op input for NextIteration: got %s+%s", opInputs, constAndPhInputs);
                VarId in = (allIterInputs != null && !allIterInputs.isEmpty() ? allIterInputs.iterator().next() : opInputs.iterator().next());
                Preconditions.checkState(outputFrameIter.getFrame().equals(in.getFrame()), "Expected same frame for NextIteration input vs. output:" +
                        " got input %s, output %s", in, outputFrameIter);
                Preconditions.checkState(outputFrameIter.getIteration() == in.getIteration() + 1, "Expected output iteration for NextIteration output to" +
                        " be 1 larger than the input iteration. Input: %s, output %s", in, outputFrameIter);

                executionStatus = "NEXT_ITERATION";
                detailedStatus = String.format("NEXT_ITERATION in frame '%s' (iter: %d -> %d)",
                        outputFrameIter.getFrame(), in.getIteration(), outputFrameIter.getIteration());
                outputNames.add(in.getVariable());

                ExecutionResult result;
                if(nodeValueOutputs.containsKey(in) && getSdValue(in) != null) {
                    SDValue value = getSdValue(in);
                    if(value != null && value.getSdValueType() == SDValueType.LIST) {
                        result = ExecutionResult.createValue(in.getVariable(),value);
                    } else if(value != null && value.getSdValueType() == SDValueType.TENSOR) {
                        INDArray inArr = getTensorFromOutputs(in);
                        if (inArr == null) {
                            Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                                    op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                        }
                        result = ExecutionResult.createFrom(Arrays.asList(in.getVariable()),new INDArray[]{inArr});
                    } else {
                        throw new IllegalStateException("Illegal value type " + value.getSdValueType() + " for input " + in);
                    }
                } else {
                    INDArray inArr = getTensorFromOutputs(in);
                    if (inArr == null) {
                        Preconditions.throwStateEx("Could not find array for NextIteration operation %s with output %s (frame=%s, iteration=%s)",
                                op.getOwnName(), sameDiff.getOps().get(op.getOwnName()).getOutputsOfOp().get(0), outputFrameIter.getFrame(), outputFrameIter.getIteration());
                    }
                    result = ExecutionResult.createFrom(Arrays.asList(in.getVariable()),new INDArray[]{inArr});
                }

                // Record next iteration execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(in.getVariable()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Merge) {
                Merge m = (Merge) op;
                String[] in = sameDiff.getInputsForOp(op);

                // Multi-frame input resolution for Merge operations
                List<VarId> candidateInputs = new ArrayList<>();
                List<SDValue> availableValues = new ArrayList<>();

                for (String inputName : in) {
                    SDValue foundValue = null;
                    VarId foundVarId = null;

                    // Strategy 1: Current frame lookup
                    VarId currentFrameVid = outputFrameIter.toVarId(inputName);
                    foundValue = getSdValue(currentFrameVid);
                    if (foundValue != null) {
                        candidateInputs.add(currentFrameVid);
                        availableValues.add(foundValue);
                        continue;
                    }

                    // Strategy 2: Cross-frame lookup for Enter operations
                    for (Map.Entry<VarId, SDValue> entry : nodeValueOutputs.entrySet()) {
                        VarId storedVid = entry.getKey();

                        if (storedVid.getFrame().equals(outputFrameIter.getFrame())) {
                            String producerOp = findVariableProducer(storedVid.getVariable());
                            if (producerOp != null) {
                                SameDiffOp producer = sameDiff.getOps().get(producerOp);
                                if (producer != null && producer.getOp() instanceof Enter) {
                                    List<String> enterOutputs = producer.getOutputsOfOp();
                                    if (enterOutputs != null && enterOutputs.contains(inputName)) {
                                        foundValue = entry.getValue();
                                        foundVarId = storedVid;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    // Strategy 3: Look for alias mappings in dependency tracker
                    if (foundValue == null) {
                        VarId aliasVid = resolveVarIdAlias(currentFrameVid);
                        if (aliasVid != null && !aliasVid.equals(currentFrameVid)) {
                            foundValue = getSdValue(aliasVid);
                            foundVarId = aliasVid;
                        }
                    }

                    if (foundValue != null) {
                        candidateInputs.add(foundVarId != null ? foundVarId : currentFrameVid);
                        availableValues.add(foundValue);
                    }
                }

                if (log.isDebugEnabled()) {
                    log.debug("Merge operation {} found {} available inputs out of {} expected",
                            m.getOwnName(), availableValues.size(), in.length);
                    for (int i = 0; i < candidateInputs.size(); i++) {
                        log.debug("  Input {}: {} -> {}", i, candidateInputs.get(i),
                                availableValues.get(i) != null ? "AVAILABLE" : "NULL");
                    }
                }

                if (availableValues.isEmpty()) {
                    throw new IllegalStateException(String.format(
                            "Merge node %s has no available inputs (expected: %s, frame: %s) - cross-frame dependency resolution failure",
                            m.getOwnName(), Arrays.toString(in), outputFrameIter));
                }

                // Use first available input (standard Merge behavior)
                SDValue selectedValue = availableValues.get(0);
                VarId selectedVarId = candidateInputs.get(0);

                log.trace("Merge {} selected input: {} from frame {}",
                        m.getOwnName(), selectedVarId.getVariable(), selectedVarId.getFrame());

                ExecutionResult result;
                if(selectedValue.getSdValueType() == SDValueType.LIST) {
                    result = ExecutionResult.createValue(selectedVarId.getVariable(), selectedValue);
                } else if(selectedValue.getSdValueType() == SDValueType.TENSOR) {
                    INDArray inArr = getTensorFromOutputs(selectedVarId);
                    if (inArr == null) {
                        inArr = selectedValue.getTensorValue();
                    }
                    if (inArr == null) {
                        throw new IllegalStateException(String.format(
                                "Could not resolve tensor for Merge operation %s input %s (frame=%s, iteration=%s)",
                                op.getOwnName(), selectedVarId.getVariable(), outputFrameIter.getFrame(), outputFrameIter.getIteration()));
                    }
                    result = ExecutionResult.createFrom(Arrays.asList(selectedVarId.getVariable()), new INDArray[]{inArr});
                } else {
                    throw new IllegalStateException("Illegal value type " + selectedValue.getSdValueType() + " for Merge input " + selectedVarId);
                }

                return result;
            }else if (op instanceof LoopCond) {
                //LoopCond just forwards scalar boolean to output
                LoopCond lc = (LoopCond) op;
                String[] argNames = lc.argNames();
                Preconditions.checkState(argNames.length == 1, "Expected only 1 arg name in LoopCond op, got %s", (Object) argNames);
                VarId vid = outputFrameIter.toVarId(argNames[0]);
                SDValue getValue = getSdValue(vid);
                if(getValue.getTensorValue() == null) {
                    throw new IllegalStateException("Node value output at " + vid.getVariable() + " was not a boolean tensor!");
                }
                Preconditions.checkNotNull(getValue, "Input to LoopCond op must not be null");
                Preconditions.checkState(getValue.getTensorValue().isScalar() && getValue.getTensorValue().dataType() == DataType.BOOL, "LoopCond input must be a scalar boolean, got %ndShape");

                boolean conditionValue = getValue.getTensorValue().getDouble(0) != 0.0;
                executionStatus = "SUCCESS";
                detailedStatus = String.format("LoopCond forwarded: %s", conditionValue);
                outputNames.add(vid.getVariable());

                ExecutionResult result = ExecutionResult.createValue(vid.getVariable(), getValue);

                // Record loop condition execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames[0]),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof BaseTensorOp) {
                //TensorOps - special cases...
                executionStatus = "SUCCESS";
                detailedStatus = "TensorArray operation";

                ExecutionResult result = getOutputsHelperTensorArrayOps(op, outputFrameIter, opInputs, allIterInputs, otherPlaceHolders);

                // Record tensor op execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof Identity) {
                List<VarId> orderedInputs = new ArrayList<>(opInputs);
                SDValue sdValue = getSdValue(orderedInputs.get(0));
                executionStatus = "SUCCESS";
                detailedStatus = "Identity operation";
                outputNames.add(op.outputVariablesNames()[0]);

                ExecutionResult result = ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue);

                // Record identity execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(orderedInputs.get(0).getVariable()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof Assign) {
                List<VarId> orderedInputs = new ArrayList<>(opInputs);
                executionStatus = "SUCCESS";
                detailedStatus = "Assign operation";
                outputNames.add(op.outputVariablesNames()[0]);

                ExecutionResult result;
                if(orderedInputs.size() > 1) {
                    SDValue sdValue = getSdValue(orderedInputs.get(0));
                    SDValue sdValue1 = getSdValue(orderedInputs.get(1));
                    switch(sdValue.getSdValueType()) {
                        case TENSOR:
                            Assign c = (Assign) op;
                            Nd4j.exec(c, opContext);
                            result = ExecutionResult.createFrom(c,opContext);
                            break;
                        case LIST:
                            result = ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue1);
                            break;
                        default:
                            throw new IllegalStateException("Unknown SDValue type: " + sdValue.getSdValueType());
                    }
                } else {
                    SDValue sdValue = getSdValue(orderedInputs.get(0));
                    result = ExecutionResult.createValue(op.outputVariablesNames()[0], sdValue);
                }

                // Record assign execution
                if (visualizationEnabled && visualizer != null) {
                    List<String> inputs = new ArrayList<>();
                    for (VarId vid : orderedInputs) {
                        inputs.add(vid.getVariable());
                    }
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            inputs,
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof GradientBackwardsMarker) {
                INDArray out = mmgr.allocate(false, DataType.FLOAT).assign(1.0f);
                executionStatus = "SUCCESS";
                detailedStatus = "Gradient backwards marker";
                outputNames.add("gradientbackwardsmarker");

                ExecutionResult result = ExecutionResult.createFrom(Arrays.asList("gradientbackwardsmarker"), new INDArray[]{out});

                // Track for deallocation
                if (!freedArrays.contains(out.getId())) {
                    mmgr.release(out);
                    freedArrays.add(out.getId());
                }

                // Record gradient marker execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Collections.emptyList(),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof CreateView) {
                Map<String,VarId> inputVars = new LinkedHashMap<>();
                String[] argNames = op.argNames();
                for(Iterator<VarId> iter = opInputs.iterator(); iter.hasNext();) {
                    VarId varId  = iter.next();
                    inputVars.put(varId.getVariable(),varId);
                }

                executionStatus = "SUCCESS";
                detailedStatus = String.format("CreateView with %d indices", argNames.length - 1);
                outputNames.add(op.outputVariablesNames()[0]);

                SDValue sdValue = getSdValue(inputVars.get(argNames[0]));
                if(sdValue == null) {
                    sdValue = SDValue.create(opContext.getInputArray(0));
                }
                INDArray[] indices = new INDArray[argNames.length - 1];
                for(int i = 1; i < argNames.length; i++) {
                    indices[i - 1] = getSdValue(inputVars.get(argNames[i])).getTensorValue();
                }

                INDArray from = CreateView.createFrom(sdValue.getTensorValue(), indices);
                from.setCloseable(false);
                sdValue.getTensorValue().setCloseable(false);
                for(INDArray arr : indices)
                    arr.setCloseable(false);
                ExecutionResult result = ExecutionResult.createFrom(op.outputVariablesNames()[0], from);

                // Record create view execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(argNames),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof ExternalErrorsFunction) {
                ExternalErrorsFunction fn = (ExternalErrorsFunction) op;
                String n = fn.getGradPlaceholderName();
                INDArray arr = getTensorFromOutputs(new VarId(n, OUTER_FRAME, 0, null));
                Preconditions.checkState(arr != null, "Could not find external errors placeholder array: %s", arr);

                executionStatus = "SUCCESS";
                detailedStatus = "External errors function";
                outputNames.add(n);

                INDArray out = mmgr.allocate(false, arr.dataType(), arr.shape());
                out.assign(arr);
                ExecutionResult result = ExecutionResult.createFrom(Arrays.asList(n), new INDArray[]{out});

                // Track for deallocation - this is typically an output array
                arrayUseTracker.addDependency(SDValue.create(out), new ReqOutputDep(n));

                // Record external errors execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(n),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if(op instanceof Invoke) {
                Invoke invoke = (Invoke) op;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("Invoke with %d inputs", invoke.getInputVarNames().length);
                outputNames.addAll(Arrays.asList(invoke.getOutputVarNames()));

                boolean hasValues = false;
                for(VarId varId : opInputs) {
                    //need to invoke with values
                    if(nodeValueOutputs.containsKey(varId)) {
                        hasValues = true;
                        break;
                    }
                }

                //no need to check placeholders if other values are present
                if(!hasValues)
                    for(Map.Entry<String,SDValue> entry : otherPlaceHolders.entrySet()) {
                        if(constAndPhInputs.contains(entry.getKey())) {
                            hasValues = true;
                            break;
                        }
                    }

                Map<String,INDArray> inputs = new LinkedHashMap<>();
                Map<String,SDValue> valueInputs = new LinkedHashMap<>();
                //need to pull from tensor arrays
                if(!hasValues) {
                    //simple linear scan of inputs over inputs
                    int currInput = 0;
                    for(VarId opInput : opInputs) {
                        inputs.put(opInput.getVariable(),opContext.getInputArray(currInput));
                        currInput++;
                    }
                } else {
                    //simple linear scan of inputs over inputs
                    Map<String,VarId> varIdsByVariable = new HashMap<>();
                    for(VarId opInput : opInputs) {
                        varIdsByVariable.put(opInput.getVariable(),opInput);
                    }

                    for(int i = 0; i < invoke.getInputVarNames().length; i++) {
                        VarId opInput = varIdsByVariable.get(invoke.getInputVarNames()[i]);
                        if(constAndPhInputs.contains(invoke.getInputVarNames()[i])) {
                            if(otherPlaceHolders.containsKey(invoke.getInputVarNames()[i]))
                                valueInputs.put(invoke.getInputVarNames()[i],otherPlaceHolders.get(invoke.getInputVarNames()[i]));
                            else if(inputs.containsKey(invoke.getInputVarNames()[i]))
                                valueInputs.put(invoke.getInputVarNames()[i],SDValue.create(inputs.get(invoke.getInputVarNames()[i])));
                        }else if(sameDiff.getArrForVarName(invoke.getInputVarNames()[i]) != null) {
                            valueInputs.put(invoke.getInputVarNames()[i],SDValue.create(sameDiff.getArrForVarName(invoke.getInputVarNames()[i])));
                        }  else if(nodeValueOutputs.containsKey(opInput)) {
                            valueInputs.put(opInput.getVariable(), getSdValue(opInput));
                        } else {
                            valueInputs.put(opInput.getVariable(),SDValue.create(opContext.getInputArray(i)));
                        }
                    }
                }

                if(valueInputs.size() + inputs.size() != op.args().length) {
                    throw new IllegalArgumentException("Value inputs and inputs combined did not fulfill all arguments. Inputs were: " + Arrays.toString(op.argNames()) + " for op name " + op.getOwnName());
                }

                ExecutionResult result = Invoke.doInvoke(invoke,inputs,valueInputs);

                // Record invoke execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            Arrays.asList(invoke.getInputVarNames()),
                            outputNames,
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Assert) {
                Assert a = (Assert) op;
                boolean condition = !opContext.getInputArray(0).isEmpty() && opContext.getInputArray(0).getDouble(0) != 0.0;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("Assert condition: %s", condition);

                if(!condition) {
                    //Assertion failed
                    String s = "Assertion failed for operation \"" + op.getOwnName() + "\" during execution";
                    if(a.numInputArguments() >= 3) {
                        INDArray msg = opContext.getInputArray(2);
                        if (msg != null && msg.dataType() == DataType.UTF8) {
                            s += ": " + msg.getString(0);
                        }
                    }
                    if(a.numInputArguments() >= 5) {
                        INDArray arr = opContext.getInputArray(4);
                        s += "\n" + arr;
                    }

                    // Record failed assertion
                    if (visualizationEnabled && visualizer != null) {
                        visualizer.recordStep(
                                ExecType.OP,
                                op.getOwnName(),
                                outputFrameIter,
                                getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                                Collections.emptyList(),
                                "EXECUTION_FAILED | Assertion failed: " + s
                        );
                    }

                    throw new IllegalStateException(s);
                }

                ExecutionResult result = ExecutionResult.createFrom(a,opContext);

                // Record successful assertion
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof CustomOp) {
                CustomOp c = (CustomOp) op;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("CustomOp: %s", c.opName());

                Nd4j.exec(c, opContext);
                ExecutionResult result = ExecutionResult.createFrom((DifferentialFunction) c,opContext);

                // Record custom op execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else if (op instanceof Op) {
                Op o = (Op) op;

                executionStatus = "SUCCESS";
                detailedStatus = String.format("Op: %s", o.opName());

                Nd4j.exec(o, opContext);
                ExecutionResult result = ExecutionResult.createFrom((DifferentialFunction)o,opContext);

                // Record op execution
                if (visualizationEnabled && visualizer != null) {
                    visualizer.recordStep(
                            ExecType.OP,
                            op.getOwnName(),
                            outputFrameIter,
                            getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                            getStepOutputsForVisualization(op),
                            executionStatus + " | " + detailedStatus
                    );
                }

                return result;

            } else {
                throw new UnsupportedOperationException("Execution not yet implemented for: " + op.getClass().getName());
            }

        } catch (Exception e) {
            // Enhanced error visualization - mimic output() method error handling
            executionStatus = "EXECUTION_FAILED";
            detailedStatus = String.format("Exception: %s - %s", e.getClass().getSimpleName(), e.getMessage());

            // Record failed execution with comprehensive context
            if (visualizationEnabled && visualizer != null) {
                String failureContext = generateOperationFailureContext(op, opInputs, constAndPhInputs, allIterInputs, e);

                visualizer.recordStep(
                        ExecType.OP,
                        op.getOwnName(),
                        outputFrameIter,
                        getStepInputsForVisualization(opInputs, constAndPhInputs, allIterInputs),
                        getStepOutputsForVisualization(op),
                        executionStatus + " | " + detailedStatus + " | " + failureContext
                );

                // Enhanced failure analysis for control flow operations
                if (op instanceof Switch || op instanceof Merge || op instanceof Enter ||
                        op instanceof Exit || op instanceof NextIteration || op instanceof LoopCond) {

                    visualizer.analyzeControlFlowFailure(op, opInputs, allIterInputs, constAndPhInputs,
                            outputFrameIter, nodeValueOutputs, e);
                }
            }

            throw e; // Re-throw the exception
        }
    }



    /**
     * Initialize variable values from constants, variables, and placeholders
     */
    private void initializeVariableValues(Map<String, SDValue> variableValues,
                                          ForwardExecutionDAG dag,
                                          Map<String, INDArray> placeholderValues,
                                          Map<String, SDValue> otherPlaceholderValues) {

        // Initialize constants
        for (String constName : dag.getConstants()) {
            INDArray constValue = getConstantOrVariable(constName);
            if (constValue != null) {
                variableValues.put(constName, SDValue.create(constValue));
            }
        }

        // Initialize variables
        for (String varName : dag.getVariables()) {
            INDArray varValue = getConstantOrVariable(varName);
            if (varValue != null) {
                variableValues.put(varName, SDValue.create(varValue));
            }
        }

        // Initialize placeholders
        if (placeholderValues != null) {
            for (Map.Entry<String, INDArray> entry : placeholderValues.entrySet()) {
                variableValues.put(entry.getKey(), SDValue.create(entry.getValue()));
            }
        }

        if (otherPlaceholderValues != null) {
            variableValues.putAll(otherPlaceholderValues);
        }
    }

    // Helper methods for visualization - mimic the output() method approach

    private List<String> getStepInputsForVisualization(Set<VarId> opInputs, Set<String> constAndPhInputs, Set<VarId> allIterInputs) {
        List<String> stepInputs = new ArrayList<>();

        if (opInputs != null) {
            for (VarId vid : opInputs) {
                stepInputs.add(vid.getVariable());
            }
        }

        if (constAndPhInputs != null) {
            stepInputs.addAll(constAndPhInputs);
        }

        if (allIterInputs != null) {
            for (VarId vid : allIterInputs) {
                stepInputs.add(vid.getVariable());
            }
        }

        return stepInputs;
    }

    private List<String> getStepOutputsForVisualization(DifferentialFunction op) {
        if (op.outputVariablesNames() != null) {
            return Arrays.asList(op.outputVariablesNames());
        }

        // For ops that might not have standard output variable names
        SameDiffOp sdOp = sameDiff.getOps().get(op.getOwnName());
        if (sdOp != null && sdOp.getOutputsOfOp() != null) {
            return sdOp.getOutputsOfOp();
        }

        return Collections.emptyList();
    }

    private String generateOperationFailureContext(DifferentialFunction op, Set<VarId> opInputs,
                                                   Set<String> constAndPhInputs, Set<VarId> allIterInputs,
                                                   Exception e) {
        StringBuilder context = new StringBuilder();

        context.append("Op: ").append(op.getClass().getSimpleName());
        context.append(", Inputs: ").append(opInputs != null ? opInputs.size() : 0);
        context.append(", ConstPh: ").append(constAndPhInputs != null ? constAndPhInputs.size() : 0);
        context.append(", AllIter: ").append(allIterInputs != null ? allIterInputs.size() : 0);

        // Add input availability analysis
        if (opInputs != null) {
            int availableInputs = 0;
            for (VarId vid : opInputs) {
                if (getSdValue(vid) != null) {
                    availableInputs++;
                }
            }
            context.append(", Available: ").append(availableInputs).append("/").append(opInputs.size());
        }

        return context.toString();
    }

    private SDValue getPreviousValue(VarId varId) {
        return getPreviousValue(varId,1);
    }

    private SDValue getPreviousValue(VarId varId,int offset) {
        VarId ret = new VarId(varId.getVariable(), varId.getFrame(), varId.getIteration() - offset,varId.getParentFrame());
        return nodeValueOutputs.get(ret);
    }

    private SDValue getValueAtIteration(String var,String frame, int iteration,FrameIter parentFrame) {
        VarId varId = new VarId(var,frame,iteration,parentFrame);
        return nodeValueOutputs.get(varId);
    }

    /**
     * Forward pass for TensorArray ops
     */
    public ExecutionResult getOutputsHelperTensorArrayOps(DifferentialFunction op, FrameIter outputFrameIter, Set<VarId> opInputs, Set<VarId> allIterInputs, Map<String, SDValue> otherPlaceHolders) {
        /*
        TODO: TensorArray memory management note: For now, we'll close any INDArrays stored in the TensorArray at the end of
        graph execution. This uses more memory than necessary for an earlier close strategy, but simplifies memory management.
        This should be revisited and optimized later
         */

        if (op instanceof TensorArray) {
            //Create a TensorArray
            VarId vid = outputFrameIter.toVarId(op.outputVariable().name());
            if(nodeValueOutputs.containsKey(vid)) {
                // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
                return ExecutionResult.createValue(vid.getVariable(),nodeValueOutputs.get(vid));
            }
            Preconditions.checkState(!nodeValueOutputs.containsKey(vid), "TensorArray already exists for %s when executing TensorArrayV3", vid);
            List<INDArray> createList = new ArrayList<>();

            if(op.args().length > 0) {
                SDVariable size = op.arg(0);
                INDArray arr = size.getArr();
                TensorArray tensorArray = (TensorArray) op;
                long[] requiredShape = tensorArray.args().length > 1 ? tensorArray.requiredShape() : null;
                for(int i = 0; i  < arr.getInt(0); i++) {
                    createList.add(null);
                }

            }


            SDValue listValue = SDValue.create(createList);
            putNodeValue(listValue, vid);

            // Note that TensorArray has 2 outputs - a 'dummy' SDVariable that represents it, and a second output (return a scalar 0.0)
            return ExecutionResult.createValue(vid.getVariable(),listValue);
        } else if (op instanceof TensorArrayRead) {
            //Do lookup and return
            //Input 0 is the TensorArray (or dummy variable that represents it). Sometimes (for import) this can be like (TensorArray -> Enter -> TensorArrayRead)
            //Input 1 is the index
            SDVariable idxSDV = op.arg(1);
            INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
            Preconditions.checkState(idxArr.isScalar(), "TensorArrayRead input argument 1 should be scalar - has shape %ndShape", idxArr);
            int i = idxArr.getInt(0);

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array

            //Work out the frame/iteration:
            VarId v = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (v == null && allIterInputs != null) {
                v = lookup(inTensorArray.name(), allIterInputs, false);
            }


            Preconditions.checkState(v != null, "Could not find input %s", inTensorArray.name());

            TensorArray tensorArray1 = TensorArray.getTensorArray(sameDiff, inTensorArray);

            List<INDArray> list = null;
            if(!nodeValueOutputs.containsKey(v)) {
                TensorArray tensorArray = TensorArray.getTensorArray(sameDiff,inTensorArray);
                SDVariable output = tensorArray.getVar();
                list = getTensorArraysInSession(output.name());

            } else {
                list = getSdValue(v).getListValue();
            }

            //we specify a shape every element should be and validate it
            if(tensorArray1.args().length > 1) {
                long[] inputShapeArr = tensorArray1.requiredShape();
                for(int j = 0; j < list.size(); j++) {
                    if(list.get(j) != null)
                        if(!Arrays.equals(inputShapeArr,list.get(j).shape()) && inputShapeArr.length > 0) {
                            throw new IllegalArgumentException("Element " + j  + " of list " + v.getVariable() + " did not have correct shape of " + Arrays.toString(inputShapeArr) + " was shape " + Arrays.toString(list.get(j).shape()));
                        }

                }
            }
            Preconditions.checkState(list != null, "Could not find TensorList for %s", v);
            Preconditions.checkState(list.size() > i, "Cannot get index %s from TensorList of size %s (array not present?) - VarId=%s", i, list.size(), v);

            INDArray out = list.get(i);

            log.trace("Reading item at index " + i + " for list " + v + " with value " + out + " with list of " + list);
            return ExecutionResult.createFrom(v.getVariable(),out);
        } else if (op instanceof TensorArrayWrite) {
            //TensorArrayWrite - also has a scalar 0.0 that it returns...
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            //Work out the varid (frame/iteration) of the tensor array:
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }



            //create new tensor array for placeholder referencing a passed in variable
            if(tArr == null && inTensorArray.getVariableType() == VariableType.PLACEHOLDER) {
                VarId varId = new VarId(inTensorArray.name(),outputFrameIter.getFrame(),outputFrameIter.getIteration(),outputFrameIter.getParentFrame());
                tArr = varId;
                SDValue sdValue = otherPlaceHolders.get(inTensorArray.name());
                //putNodeValue(sdValue, tArr);
            }

            Preconditions.checkState(tArr != null, "Could not find input %s", inTensorArray.name());



            //Input 0 is the TensorArray (or dummy variable that represents it) - but sometimes Enter, in TensorArray -> Enter -> TensorARrayRead
            //Input 1 is the index
            //Input 2 is the value to write

            String idxName = op.arg(1).name();
            SDVariable idxSDV = sameDiff.getVariable(idxName);
            INDArray idxArr = getArray(idxSDV, opInputs, allIterInputs);
            Preconditions.checkState(idxArr.isScalar(), "Index variable ID for TensorArrayWrite should be a scalar, got %ndShape", idxArr);
            int idx = idxArr.getInt(0);

            String inName = op.arg(2).name();
            SDVariable inSDV = sameDiff.getVariable(inName);
            INDArray arr = getArray(inSDV, opInputs, allIterInputs);
            Preconditions.checkState(arr != null, "Could not find array for %s", inName);
            TensorArray tArrOp = TensorArray.getTensorArray(sameDiff,inTensorArray);
            tArr = new VarId(tArrOp.outputVariable().name(),OUTER_FRAME,0,null);
            if(tArrOp.args().length > 1) {
                long[] shape = tArrOp.arg(1).getArr().toLongVector();
                if(!Arrays.equals(arr.shape(),shape) && shape.length > 0) {
                    throw new IllegalArgumentException("Unable to write array of shape " + Arrays.toString(arr.shape()) + " must be " + Arrays.toString(shape) + " for op " + op.getOwnName() + " and tensor array " + tArrOp.getOwnName());
                }
            }


            Preconditions.checkState(nodeValueOutputs.containsKey(tArr), "Tensor array does not exist for %s", tArr);
            //TODO is this always safe to insert by index for all execution orders?
            SDValue sdValue1 = getSdValue(tArr);
            List<INDArray> l = sdValue1.getListValue(); //.set(idx, arr);
            if(idx < 0 && l != null && !l.isEmpty()) {
                idx += l.size() + 1;
            } else if(idx < 0) {
                idx = 0;
            }
            while (l.size() <= idx) {
                //Can't use set(int, E) if index >= size
                l.add(null);
            }

            setArrayAtIndex(l, idx, arr);
            log.trace("Setting item at index " + idx + " for list " + tArr + " with value " + arr + " with whole list of after write " + l + " and value array " + arr);
            log.trace("Writing value " + inSDV + " to list " + tArr.getVariable() + " at iteration " + tArr.getIteration());

            //Add a dependency
            Dep d = new ExecDoneDep();
            arrayUseTracker.addDependency(sdValue1, d);
            return ExecutionResult.createValue(op.outputVariable().name(),sdValue1);
        } else if (op instanceof TensorArraySize) {
            //Index 0 is the TensorArray (or dummy variable that represents it)
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            TensorArray tensorArray = TensorArray.getTensorArray(sameDiff,inTensorArray);
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            List<INDArray> l = getSdValue(tArr).getListValue();
            int size = l == null ? 0 : l.size();
            INDArray scalar = mmgr.allocate(false, DataType.INT).assign(size);
            return ExecutionResult.createFrom(tensorArray.getVar().name(),scalar);
        } else if (op instanceof TensorArrayConcat) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }
            List<INDArray> l = getSdValue(tArr).getListValue();

            Concat c = new Concat(0, l.stream().filter(input -> input != null).collect(Collectors.toList())
                    .toArray(new INDArray[0]));
            List<DataBuffer> shape = c.calculateOutputShape();
            INDArray out = mmgr.allocateFromDescriptor(false, shape.get(0));
            c.setOutputArgument(0, out);
            Nd4j.exec(c);
            return ExecutionResult.createFrom(tArr.getVariable(),out);
        } else if (op instanceof TensorArrayGather) {
            //Input 0: the TensorArray
            //Input 1: the indices (1d integer vector)

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            List<INDArray> l = getSdValue(tArr).getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String indicesName = op.arg(1).name();
            SDVariable indicesSDV = sameDiff.getVariable(indicesName);
            INDArray idxArr = indicesSDV.getArr();
            Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayGather should be a vector, got %ndShape for %s", idxArr, indicesName);
            Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayGather should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);

            int[] idxArrInt = idxArr.toIntVector();
            log.trace("Gathering op " + op.getOwnName() + " from indices " + Arrays.toString(idxArrInt) + " named " + indicesName + " from list " + tArr.getVariable());
            if(idxArrInt.length > 0) {
                //Edge case: -1 means "all"
                List<INDArray> newList = new ArrayList<>();
                if (idxArrInt.length == 1 || idxArrInt.length > 0 &&  idxArrInt[0]  < 0) {
                    newList.addAll(l);
                } else {
                    for (int id : idxArrInt) {
                        Preconditions.checkState(id >= 0, "Index for TensorArrayGather must be >= 0, got %s", id);
                        if(l.get(id) != null) {
                            log.trace("Gathering op " + op.getOwnName() + " at index " + id + " adding value " + l.get(id).toStringFull() + " from full list " + l);
                            newList.add(l.get(id));

                        }
                    }
                }

                Stack s = new Stack(newList.stream().filter(input -> input != null).collect(Collectors.toList())
                        .toArray(new INDArray[0]), null, 0);
                List<DataBuffer> shape = s.calculateOutputShape();
                INDArray out = mmgr.allocateFromDescriptor(false, shape.get(0));
                s.setOutputArgument(0, out);
                Nd4j.exec(s);
                return ExecutionResult.createFrom(tArr.getVariable(),out);
            } else {
                return ExecutionResult.createFrom(tArr.getVariable(),Nd4j.zeros(op.arg().dataType(),0));
            }

        } else if (op instanceof TensorArrayScatter) {
            //Scatter values from a rank (N+1)d tensor into specific indices of the TensorArray
            //Input 0: the TensorArray
            //Input 1: the indices (1d integer vector)
            //Input 2: The values to scatter

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            TensorArray ta = TensorArray.getTensorArray(sameDiff,inTensorArray);
            VarId tArr = (opInputs == null ? null : lookup(ta.outputVariablesNames()[0], opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(ta.outputVariablesNames()[0], allIterInputs, false);
            }

            SDValue retValue = getSdValue(tArr);
            List<INDArray> l = retValue.getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String indicesName = op.arg(1).name();
            SDVariable indicesSDV = sameDiff.getVariable(indicesName);
            INDArray idxArr = indicesSDV.getArr();
            Preconditions.checkState(idxArr.isVector(), "Indices variable for TensorArrayScatter should be a vector, got %ndShape for %s", idxArr, indicesName);
            Preconditions.checkState(idxArr.dataType().isIntType(), "Indices variable for TensorArrayScatter should be an integer type, got %s for array %s", idxArr.dataType(), indicesName);
            int[] idxs = idxArr.toIntVector();

            String valuesName = op.arg(2).name();
            SDVariable valuesSDV = sameDiff.getVariable(valuesName);
            INDArray valuesArr = getArray(valuesSDV, opInputs, allIterInputs);

            while (l.size() < idxs.length) { //Can't use set(int, E) if index >= size
                l.add(null);
            }


            //Edge case: idxs being [-1] means "all sub arrays" (i.e., "unstack" case)
            if (idxs.length == 1 && idxs[0] == -1) {
                idxs = ArrayUtil.range(0, (int) valuesArr.size(0));
            }

            for(int i = 0; i < idxs.length; i++) {
                if(valuesArr.size(0) < idxs[i]) {
                    throw new IllegalArgumentException("Unable to obtain slice from values array named " + valuesName +  " with shape " + Arrays.toString(valuesArr.shape()) + " at index " + idxs[i] + " at node named " + op.getOwnName()  + " with inputs " + Arrays.toString(op.argNames()));
                }
            }

            for (int i = 0; i < idxs.length; i++) {
                if(idxs[i] >= valuesArr.size(0)) {
                    throw new IllegalStateException("Unable to pull slice from value array " + valuesSDV.name() + " of shape " + Arrays.toString(valuesArr.shape()) + " index was" + idxs[i]  + " all indices were " + Arrays.toString(idxs));
                }
                INDArray getView = valuesArr.slice(idxs[i]);
                INDArray get = mmgr.dup(getView);
                if(ta.args().length > 1) {
                    long[] shape = ta.arg(1).getArr().toLongVector();
                    if(!Arrays.equals(get.shape(),shape) && shape.length > 0) {
                        throw new IllegalArgumentException("Unable to write array of shape " + Arrays.toString(get.shape()) + " must be " + shape + " for op " + op.getOwnName() + " and tensor array " + ta.getOwnName());
                    }
                }
                SDValue newValue = SDValue.create(get);
                int outIdx = idxs[i];
                if (valuesArr.rank() == 1 && get.rank() > 0) {
                    get = get.reshape();
                }

                //reflect the expanded storage
                if(outIdx >= l.size()) {
                    while(l.size() <= outIdx) {
                        l.add(null);
                    }
                }

                log.trace("Scattering item at index " + i + " for list " + tArr + " with value " + get + " from whole list of " + l + " from values array " + valuesArr.toStringFull() + " named " + valuesSDV.name());
                setArrayAtIndex(l, outIdx, get);

                //Add dependency for values array until end of execution
                arrayUseTracker.addDependency(newValue, new ExecDoneDep());
            }


            return ExecutionResult.createValue(valuesName,retValue);
        } else if (op instanceof TensorArraySplit) {
            //Split values from a rank (N+1)d tensor into sequential indices of the TensorArray
            //For example, orig=[8,2] sizearray with split (4,4) means TensorArray[0] = orig[0:4,:] and TensorArray[1] = orig[4:8,:]
            //Input 0: the TensorArray
            //Input 1: The values to split
            //Input 2: the size of each split (1d integer vector)

            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }


            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            SDValue sdValue = getSdValue(tArr);
            List<INDArray> l = sdValue.getListValue();
            Preconditions.checkState(l != null, "Could not find TensorArray: %s", tArr);

            String splitName = op.arg(1).name();
            INDArray splitArr = getArray(sameDiff.getVariable(splitName), opInputs, allIterInputs);


            String sizeName = op.arg(2).name();
            SDVariable sizeSDV = sameDiff.getVariable(sizeName);
            INDArray sizeArr = getArray(sizeSDV, opInputs, allIterInputs);
            Preconditions.checkState(sizeArr.isVector(), "Indices variable for TensorArraySplit should be a vector, got %ndShape for %s", sizeArr, sizeName);
            Preconditions.checkState(sizeArr.dataType().isIntType(), "Indices variable for TensorArraySplit should be an integer type, got %s for array %s", sizeArr.dataType(), sizeName);
            int[] sizes = sizeArr.toIntVector();

            while (l.size() <= sizes.length) { //Can't use set(int, E) if index >= size
                l.add(null);
            }

            INDArrayIndex[] idx = ArrayUtil.nTimes(splitArr.rank(), NDArrayIndex.all(), INDArrayIndex.class);
            int soFar = 0;
            for (int i = 0; i < sizes.length; i++) {
                idx[0] = NDArrayIndex.interval(soFar, soFar + sizes[i]);
                INDArray sub = mmgr.dup(splitArr.get(idx));
                SDValue subValue = SDValue.create(sub);
                setArrayAtIndex(l, i, sub);
                soFar += sizes[i];

                //Add dependency for values array until end of execution
                arrayUseTracker.addDependency(subValue, new ExecDoneDep());
            }

            return ExecutionResult.createValue(sizeName,sdValue);
        } else if (op instanceof TensorArrayRemove) {
            SDVariable inTensorArray = op.arg(0);   //Dummy variable representing the tensor array
            SDVariable index = op.arg(1);
            List<INDArray> l = getTensorArraysInSession(inTensorArray.name());
            if(l == null)
                l = new ArrayList<>();
            else if(l != null)
                l.remove(index.getArr(true).getInt(0));
            VarId tArr = (opInputs == null ? null : lookup(inTensorArray.name(), opInputs, false));
            if (tArr == null && allIterInputs != null) {
                tArr = lookup(inTensorArray.name(), allIterInputs, false);
            }

            while (sameDiff.getVariableOutputOp(inTensorArray.name()) instanceof Enter) {
                //Handle the Enter case: this is like TensorArray -> Enter -> TensorArrayWrite
                //TODO also TensorArrayScatter, etc??
                inTensorArray = sameDiff.getVariableOutputOp(inTensorArray.name()).arg();
                tArr = tArr.getParentFrame().toVarId(inTensorArray.name());
            }

            //setup an extra reference to the removed list
            putNodeValue(SDValue.create(l), tArr);
            return ExecutionResult.createValue(tArr.getVariable(),l);
        }

        else {
            throw new IllegalStateException("Execution support not yet implemented for: " + op.getClass().getName());
        }
    }


    private Map<Pair<String,Integer>,SDValue> valuesFor(String varName) {
        Map<Pair<String,Integer>,SDValue> ret = new HashMap<>();
        for(Map.Entry<VarId,SDValue> values : nodeValueOutputs.entrySet()) {
            if(values.getKey().getVariable().equals(varName)) {
                ret.put(Pair.of(values.getKey().getVariable(),values.getKey().getIteration()),values.getValue());
            }
        }

        return ret;
    }


    @Override
    public INDArray getConstantOrVariable(String variableName) {
        SDVariable v = sameDiff.getVariable(variableName);
        Preconditions.checkState(sameDiff.getVariable(variableName).isConstant() || v.getVariableType() == VariableType.VARIABLE,
                "Variable %s is not a constant", variableName);
        return sameDiff.getArrForVarName(variableName);
    }

    @Override
    public Pair<SameDiffOp,OpContext> getAndParameterizeOp(String opName, FrameIter frameIter, Set<VarId> opInputs, Set<VarId> allIterInputs,
                                                           Set<String> constAndPhInputs, Map<String, INDArray> placeholderValues, Set<String> allReqVariables, Map<String, SDValue> otherPlaceholders) {
        SameDiffOp sdo = sameDiff.getOps().get(opName);
        DifferentialFunction df = sdo.getOp();

        //TODO Switch to OpContext - and make sure executing like that is thread safe (i.e., array fields in ops are not used etc)

        Preconditions.checkNotNull(df, "No differential function found with name \"%s\"", opName);

        if (df instanceof LoopCond || df instanceof Enter || df instanceof Exit || df instanceof NextIteration ||
                df instanceof Merge || df instanceof Switch || df instanceof BaseTensorOp || df instanceof Invoke) {
            //Control dependencies and tensor ops (like TensorArray, TensorArrayRead etc) don't need inputs set, execution is a special case
            return new Pair<>(sdo, null);
        }

        //Infer the args based on the inputs (variable + frame + iteration)
        String[] argNames = df.argNames();
        int numArgs = (argNames == null ? 0 : argNames.length);
        int numNonConstIns = (opInputs == null ? 0 : opInputs.size());
        int numNonConstInsAllIters = (allIterInputs == null ? 0 : allIterInputs.size());
        int numConstPhIns = (constAndPhInputs == null ? 0 : constAndPhInputs.size());

        if (numArgs != (numNonConstIns + numConstPhIns + numNonConstInsAllIters)) {
            if (numArgs > 1) {
                //Might be due to repeated inputs
                Set<String> uniqueArgNames = new LinkedHashSet<>();
                Collections.addAll(uniqueArgNames, argNames);

            } else {
                Preconditions.checkState(numArgs == (numNonConstIns + numConstPhIns),
                        "Different number of arg names as op inputs for op %s (%s): arg names %s vs. op inputs %s+%s", df.getClass().getSimpleName(),
                        opName, argNames, opInputs, constAndPhInputs);
            }
        }

        INDArray[] args = null;
        if (argNames != null && argNames.length > 0) {
            args = new INDArray[argNames.length];
            int i = 0;
            for (String s : argNames) {
                SDVariable v = sameDiff.getVariable(s);
                if (v.isConstant()) {
                    args[i] = v.getArr();
                } else if (v.getVariableType() == VariableType.VARIABLE) {
                    args[i] = v.getArr();
                } else if (v.isPlaceHolder()) {
                    if(placeholderValues != null && placeholderValues.containsKey(s))
                        args[i] = placeholderValues.get(s);
                    else if(otherPlaceholders != null && otherPlaceholders.containsKey(s)) {
                        args[i] = otherPlaceholders.get(s).getTensorValue();
                    }
                    else
                        throw new IllegalArgumentException("No array was provided for required placeholder variable \"%s\"".format(s));
                } else {
                    VarId vid = lookup(s, opInputs, allIterInputs, true);
                    SDValue getValue = getSdValue(vid);
                    if(getValue != null)
                        switch(getValue.getSdValueType()) {
                            case TENSOR:
                                args[i] = getValue.getTensorValue();
                                break;
                            case LIST:
                                DifferentialFunction variableOutputOp = sameDiff.getVariableOutputOp(s);
                                //tensorflow import case: when switch is imported and 2 are input names are equal
                                //we output a list with 1 value that's null and 1 that's not
                                if(variableOutputOp instanceof Switch && variableOutputOp.argNames().length == 2 && variableOutputOp.argNames()[0].equals(variableOutputOp.argNames()[1])) {
                                    //find the non null value
                                    for(int j = 0; j < getValue.getListValue().size(); j++) {
                                        if(getValue.getListValue().get(j) !=  null) {
                                            args[i] = getValue.getListValue().get(j);
                                            break;
                                        }
                                    }
                                }
                                else
                                    args[i] = Nd4j.empty(DataType.FLOAT);
                                break;

                        }
                }


                Preconditions.checkNotNull(args[i], "Could not parameterize op %s: array %s (variable %s) is null", opName, i, v.name());
                i++;
            }
        }

        if(df.needsConfigure()) {
            SDVariable[] vars = df.args();
            for(int i = 0; i < vars.length; i++) {
                vars[i].setShape(args[i].shape());
            }

            df.configureWithSameDiff(sameDiff);
        }


        //Set the op inputs and output arguments
        //Note that when we are in a loop (and non-first iteration), we want to allocate new arrays even if shapes are
        // ok: this is because we need the values in past iterations for backprop (potentially)
        //TODO let's find a way to use in-place modification for loops where possible to reduce memory requirements
        boolean isLoop = !frameIter.getFrame().equals(OUTER_FRAME) && frameIter.getIteration() > 0;

        OpContext oc = opContexts.get(opName);
        if(oc == null) {
            oc = Nd4j.getExecutioner().buildContext();
            opContexts.put(opName, oc);
        }

        if (df instanceof CustomOp) {
            DynamicCustomOp customOp = (DynamicCustomOp) df;
            if (df instanceof Identity || df instanceof CreateView) {
                if (args != null) {
                    oc.setInputArrays(args);
                }

                //set a dummy result to be replaced
                oc.setOutputArrays(args[0]);
                //We don't need to allocate an output array for Identity, we pass through the input array without copying
                return new Pair<>(sdo, oc);
            }

            oc.setArgs(args, customOp.iArgs(), customOp.dArgs() , customOp.tArgs(), customOp.bArgs() );

            //input and output should be same for assign
            if((df instanceof Assign)) {
                oc.setOutputArray(0, oc.getInputArray(0));

            } else {
                List<DataBuffer> outShape = customOp.calculateOutputShape(oc);
                Preconditions.checkState(outShape != null && outShape.size() > 0, "Failed to calculate output shapes for op %s (%s) - no shapes were returned by calculateOutputShape()", customOp.opName(), customOp.getOwnName());
                String[] outNames = df.outputVariablesNames();
                Preconditions.checkState(outNames.length == outShape.size(), "Error in operation shape calculation for op \"%s\": Got %s op output shapes for an operation" +
                        " with %s outputs (number of shapes and outputs must be equal)", df.opName(), outShape.size(), outNames.length);
                for (int i = 0; i < outShape.size(); i++) {
                    DataBuffer reqShape = outShape.get(i);
                    long[] asJava = reqShape.asLong();;
                    //Issue: many ops have multiple valid output datatypes, and output shape calc can't at present know which: https://github.com/eclipse/deeplearning4j/issues/6872
                    //As a workaround, we'll use the output variable datatype instead.
                    DataType dt = sameDiff.getVariable(outNames[i]).dataType();
                    DataType currDT = reqShape.dataType();
                    if (dt != currDT) {
                        Shape.setExtras(asJava,Shape.extras(asJava));
                    }

                    //Always allocate new output array, rely on memory manager for efficient memory management and array reuse etc
                    boolean isOutput = allReqVariables.contains(outNames[i]);
                    reqShape = Nd4j.createBuffer(asJava);
                    INDArray out = mmgr.allocateFromDescriptor(false, reqShape);
                    if(Shape.isEmpty(asJava) && !out.isEmpty()) {
                        throw new IllegalStateException("Output shape was empty, but created array was not.");
                    }

                    oc.setOutputArray(i, out);
                }
            }


        } else if (df instanceof Op) {
            Op op = (Op) df;

            boolean axisArg = false;
            boolean emptyReduce = false;
            if (op instanceof ReduceOp && ((ReduceOp) op).getOpType() != Op.Type.REDUCE3 && df.argNames().length == 2) {
                //2nd input should be treated as integer axis arg...
                SDVariable axisArgVar = df.arg(1);
                Preconditions.checkState(axisArgVar.dataType().isIntType(), "Legacy op %s input 1 (axis) was expected to be an integer type, is %s", df.getClass(), axisArgVar.dataType());

                INDArray arr = getArray(axisArgVar, opInputs, allIterInputs);
                Preconditions.checkState(arr != null, "Could not get axis argument for op %s: %s", df.getOwnName(), df.getClass());
                if (!arr.isEmpty()) {
                    long[] axis = arr.toLongVector();
                    int rank = args[0].rank();
                    axis = Shape.normalizeAxis(rank, axis);
                    df.setDimensions(axis);
                    ((BaseReduceOp) op).setEmptyReduce(false);
                } else {
                    df.setDimensions(null);
                    emptyReduce = true;
                    //Note: edge case: [x,y].sum(empty) = [x,y] for TF import compatibility.
                    //Note also that empty is not the same as int[0] as in INDArray.sum(new int[0])
                    ((BaseReduceOp) op).setEmptyReduce(true);
                }
                axisArg = true;
            } else if (op instanceof ScalarOp && df.argNames().length == 2) {
                //Scalar ops: 2nd input should be treated as scalar...
                SDVariable scalarVar = df.arg(1);
                INDArray scalar = getArray(scalarVar, opInputs, allIterInputs);
                Preconditions.checkState(scalar != null, "Could not get scalar argument for op %s: %s", df.getOwnName(), df.getClass());
                Preconditions.checkState(scalar.isScalar(), "Scalar argument for op %s (%s) is not a scalar: has shape %ndShape", df.getOwnName(), df.getClass(), scalar);
                ((ScalarOp) op).setScalar(scalar);
            }

            if (args != null && args.length > 0) {
                oc.setInputArray(0, args[0]);
                if (args.length == 2 && !axisArg)
                    oc.setInputArray(1, args[1]);
            }


            //Check output shape; allocate a new Z if required
            //For example, if minibatch size has changed since last op execution
            boolean isOutput = allReqVariables.contains(((BaseOp) op).outputVariablesNames()[0]);
            if (emptyReduce) {
                //Always allocate new output array, rely on memory manager for efficient memory management and array reuse etc
                INDArray z = mmgr.allocate(false, oc.getInputArray(0).dataType(), oc.getInputArray(0).shape());
                oc.setOutputArray(0, z);
            } else {
                List<DataBuffer> outputShape = ((BaseOp) op).calculateOutputShape(oc);
                Preconditions.checkState(outputShape != null && outputShape.size() == 1, "Could not calculate output shape for op: %s", op.getClass());
                DataBuffer lsd = outputShape.get(0);
                INDArray z = mmgr.allocateFromDescriptor(isOutput, lsd);
                oc.setOutputArray(0, z);
            }
        }

        return new Pair<>(sdo, oc);
    }


    protected INDArray getArray(SDVariable sdv, Collection<VarId> opInputs, Collection<VarId> allIterInputs) {
        String n = sdv.name();
        if (sdv.getVariableType() == VariableType.CONSTANT || sdv.getVariableType() == VariableType.VARIABLE) {
            return getConstantOrVariable(n);

        }   else {
            VarId inVarId = lookup(n, opInputs, allIterInputs, false);
            Preconditions.checkState(inVarId != null, "Could not find array for variable %s", sdv.name());
            return getTensorFromOutputs(inVarId);
        }
    }

}
