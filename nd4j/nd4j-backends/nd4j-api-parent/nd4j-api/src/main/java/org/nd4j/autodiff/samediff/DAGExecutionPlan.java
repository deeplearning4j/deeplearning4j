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

package org.nd4j.autodiff.samediff;

import org.nd4j.linalg.api.buffer.DataType;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Class to hold the results of a dry run execution analysis with enhanced frame granularity
 */
public class DAGExecutionPlan {
    // Basic execution plan data
    private List<String> executionOrder = new ArrayList<>();
    private Map<String, Set<String>> dependencies = new HashMap<>();
    private List<String> requestedOutputs = new ArrayList<>();
    private Map<String, VariableInfo> variables = new HashMap<>();
    private Map<String, OperationInfo> operations = new HashMap<>();
    private Set<String> leafVariables = new HashSet<>();
    private Set<String> trainableVariables = new HashSet<>();
    private Set<String> sequenceVariables = new HashSet<>();
    private List<String> missingVariables = new ArrayList<>();
    private List<String> orphanedVariables = new ArrayList<>();
    private List<String> cycles = new ArrayList<>();
    private Map<String, List<String>> controlDependencies = new HashMap<>();
    private Map<String, List<String>> variableControlDependencies = new HashMap<>();

    // Enhanced frame management structures
    private Set<String> enterOperations = new HashSet<>();
    private Set<String> exitOperations = new HashSet<>();
    private Set<String> switchOperations = new HashSet<>();
    private Set<String> mergeOperations = new HashSet<>();
    private Set<String> nextIterationOperations = new HashSet<>();
    private Set<String> loopConditionOperations = new HashSet<>();

    // Frame hierarchy and relationships
    private Map<String, String> frameHierarchy = new HashMap<>(); // frame -> parent frame
    private Map<String, Set<String>> framesUsed = new HashMap<>(); // frame -> operations in frame
    private Map<String, Set<String>> frameChildren = new HashMap<>(); // frame -> child frames
    private Map<String, Integer> frameDepth = new HashMap<>(); // frame -> nesting depth
    private Map<String, FrameMetadata> frameMetadata = new HashMap<>(); // frame -> detailed metadata

    // Frame execution analysis
    private Map<String, List<String>> frameExecutionOrder = new HashMap<>(); // frame -> ops in execution order
    private Map<String, Set<String>> frameVariables = new HashMap<>(); // frame -> variables in frame
    private Map<String, Set<String>> frameBoundaryOperations = new HashMap<>(); // frame -> boundary ops
    private Map<String, Map<String, Integer>> frameIterationCounts = new HashMap<>(); // frame -> var -> max iteration

    // Frame transitions and dependencies
    private Map<String, List<FrameTransitionInfo>> frameTransitions = new HashMap<>(); // from frame -> transitions
    private Map<String, Set<String>> frameDependencies = new HashMap<>(); // frame -> dependent frames
    private Map<String, Set<String>> frameProducers = new HashMap<>(); // frame -> frames that produce data for this frame
    private Map<String, Set<String>> frameConsumers = new HashMap<>(); // frame -> frames that consume data from this frame

    // Cross-frame data flow
    private Map<String, List<CrossFrameReference>> crossFrameReferences = new HashMap<>(); // var -> cross-frame refs
    private Map<String, Set<String>> frameInputVariables = new HashMap<>(); // frame -> input variables from other frames
    private Map<String, Set<String>> frameOutputVariables = new HashMap<>(); // frame -> output variables to other frames

    // Basic getters and setters
    public List<String> getExecutionOrder() {
        return executionOrder;
    }

    public void setExecutionOrder(List<String> executionOrder) {
        this.executionOrder = executionOrder;
    }

    public Map<String, Set<String>> getDependencies() {
        return dependencies;
    }

    public void setDependencies(Map<String, Set<String>> dependencies) {
        this.dependencies = dependencies;
    }

    public List<String> getRequestedOutputs() {
        return requestedOutputs;
    }

    public void setRequestedOutputs(List<String> requestedOutputs) {
        this.requestedOutputs = requestedOutputs;
    }

    // Frame management methods

    /**
     * Add a frame with detailed metadata
     */
    public void addFrame(String frameName, String parentFrame, FrameType type) {
        int depth = parentFrame == null ? 0 : frameDepth.getOrDefault(parentFrame, 0) + 1;
        FrameMetadata metadata = new FrameMetadata(frameName, parentFrame, depth, type);
        frameMetadata.put(frameName, metadata);
        frameDepth.put(frameName, depth);

        if (parentFrame != null) {
            frameHierarchy.put(frameName, parentFrame);
            frameChildren.computeIfAbsent(parentFrame, k -> new HashSet<>()).add(frameName);
            frameMetadata.get(parentFrame).childFrames.add(frameName);
        }

        framesUsed.computeIfAbsent(frameName, k -> new HashSet<>());
        frameExecutionOrder.computeIfAbsent(frameName, k -> new ArrayList<>());
        frameVariables.computeIfAbsent(frameName, k -> new HashSet<>());
        frameBoundaryOperations.computeIfAbsent(frameName, k -> new HashSet<>());
        frameIterationCounts.computeIfAbsent(frameName, k -> new HashMap<>());
    }

    /**
     * Add frame transition information
     */
    public void addFrameTransition(String fromFrame, String toFrame, FrameTransition transitionType,
                                   String operationName, List<String> affectedVariables, int iterationChange) {
        FrameTransitionInfo transition = new FrameTransitionInfo(fromFrame, toFrame, transitionType, operationName);
        if (affectedVariables != null) {
            transition.affectedVariables.addAll(affectedVariables);
        }
        transition.iterationChange = iterationChange;

        frameTransitions.computeIfAbsent(fromFrame, k -> new ArrayList<>()).add(transition);

        // Update frame dependencies
        if (toFrame != null && !fromFrame.equals(toFrame)) {
            frameDependencies.computeIfAbsent(toFrame, k -> new HashSet<>()).add(fromFrame);
            frameProducers.computeIfAbsent(toFrame, k -> new HashSet<>()).add(fromFrame);
            frameConsumers.computeIfAbsent(fromFrame, k -> new HashSet<>()).add(toFrame);
        }

        // Update metadata
        FrameMetadata fromMeta = frameMetadata.get(fromFrame);
        if (fromMeta != null) {
            fromMeta.transitionCounts.merge(transitionType, 1, Integer::sum);
            if (transitionType == FrameTransition.NEXT_ITERATION) {
                fromMeta.hasLoops = true;
            } else if (transitionType == FrameTransition.SWITCH) {
                fromMeta.hasConditionals = true;
            }
        }
    }

    /**
     * Add cross-frame variable reference
     */
    public void addCrossFrameReference(String variableName, String sourceFrame, String targetFrame,
                                       int sourceIteration, int targetIteration, String mediatingOperation,
                                       CrossFrameReferenceType type) {
        CrossFrameReference ref = new CrossFrameReference();
        ref.variableName = variableName;
        ref.sourceFrame = sourceFrame;
        ref.targetFrame = targetFrame;
        ref.sourceIteration = sourceIteration;
        ref.targetIteration = targetIteration;
        ref.mediatingOperation = mediatingOperation;
        ref.referenceType = type;

        crossFrameReferences.computeIfAbsent(variableName, k -> new ArrayList<>()).add(ref);

        // Update frame input/output tracking
        if (targetFrame != null) {
            frameInputVariables.computeIfAbsent(targetFrame, k -> new HashSet<>()).add(variableName);
        }
        if (sourceFrame != null) {
            frameOutputVariables.computeIfAbsent(sourceFrame, k -> new HashSet<>()).add(variableName);
        }
    }

    public void addFrameOperation(String opName, FrameTransition transition, String frame, String parentFrame) {
        switch (transition) {
            case ENTER:
                enterOperations.add(opName);
                if (frame != null) {
                    frameBoundaryOperations.computeIfAbsent(frame, k -> new HashSet<>()).add(opName);
                    FrameMetadata meta = frameMetadata.get(frame);
                    if (meta != null) {
                        meta.entryPoints.add(opName);
                    }
                }
                break;
            case EXIT:
                exitOperations.add(opName);
                if (frame != null) {
                    frameBoundaryOperations.computeIfAbsent(frame, k -> new HashSet<>()).add(opName);
                    FrameMetadata meta = frameMetadata.get(frame);
                    if (meta != null) {
                        meta.exitPoints.add(opName);
                    }
                }
                break;
            case SWITCH:
                switchOperations.add(opName);
                break;
            case MERGE:
                mergeOperations.add(opName);
                break;
            case NEXT_ITERATION:
                nextIterationOperations.add(opName);
                break;
            case LOOP_CONDITION:
                loopConditionOperations.add(opName);
                break;
        }

        if (parentFrame != null && frame != null) {
            frameHierarchy.put(frame, parentFrame);
        }
    }

    public void addOperationWithFrame(String opName, String opType, String className,
                                      List<String> inputs, List<String> outputs, FrameInfo frameInfo) {
        OperationInfo opInfo = new OperationInfo(opName, opType, className, inputs, outputs);
        opInfo.setFrameInfo(frameInfo);
        operations.put(opName, opInfo);

        // Track frame usage and execution order
        String frame = frameInfo.outputFrame;
        if (frame != null) {
            framesUsed.computeIfAbsent(frame, k -> new HashSet<>()).add(opName);
            frameExecutionOrder.computeIfAbsent(frame, k -> new ArrayList<>()).add(opName);

            // Update frame metadata
            FrameMetadata meta = frameMetadata.get(frame);
            if (meta != null) {
                meta.totalOperations++;
            }
        }
    }

    public void addVariableWithFrame(String name, VariableType type, DataType dataType,
                                     String frame, int iteration, String parentFrame) {
        VariableInfo varInfo = new VariableInfo(name, type, dataType);
        varInfo.frame = frame;
        varInfo.iteration = iteration;
        varInfo.parentFrame = parentFrame;
        variables.put(name, varInfo);

        // Track frame variables and iterations
        if (frame != null) {
            frameVariables.computeIfAbsent(frame, k -> new HashSet<>()).add(name);
            frameIterationCounts.computeIfAbsent(frame, k -> new HashMap<>())
                    .merge(name, iteration, Integer::max);

            // Update frame metadata
            FrameMetadata meta = frameMetadata.get(frame);
            if (meta != null) {
                meta.totalVariables++;
                meta.maxIterations = Math.max(meta.maxIterations, iteration);
            }
        }
    }

    // Production utility methods

    /**
     * Get all operations within a specific frame
     */
    public List<String> getOperationsInFrame(String frameName) {
        return frameExecutionOrder.getOrDefault(frameName, Collections.emptyList());
    }

    /**
     * Get all variables within a specific frame
     */
    public Set<String> getVariablesInFrame(String frameName) {
        return frameVariables.getOrDefault(frameName, Collections.emptySet());
    }



    /**
     * Get frame for a specific operation
     */
    public String getOperationFrame(String operationName) {
        OperationInfo opInfo = operations.get(operationName);
        if (opInfo != null && opInfo.getFrameInfo() != null) {
            return opInfo.getFrameInfo().outputFrame;
        }
        return null;
    }

    /**
     * Get frame for a specific variable
     */
    public String getVariableFrame(String variableName) {
        VariableInfo varInfo = variables.get(variableName);
        if (varInfo != null) {
            return varInfo.frame;
        }
        return null;
    }

    /**
     * Find all variables that are shared between frames
     */
    public Map<String, List<String>> getSharedVariables() {
        Map<String, List<String>> sharedVars = new HashMap<>();
        Map<String, Set<String>> varToFrames = new HashMap<>();

        // Build mapping of variables to frames
        for (Map.Entry<String, Set<String>> entry : frameVariables.entrySet()) {
            String frame = entry.getKey();
            for (String var : entry.getValue()) {
                varToFrames.computeIfAbsent(var, k -> new HashSet<>()).add(frame);
            }
        }

        // Find variables in multiple frames
        for (Map.Entry<String, Set<String>> entry : varToFrames.entrySet()) {
            if (entry.getValue().size() > 1) {
                sharedVars.put(entry.getKey(), new ArrayList<>(entry.getValue()));
            }
        }

        return sharedVars;
    }

    /**
     * Get operations that produce outputs for a specific frame
     */
    public List<String> getFrameProducerOperations(String frameName) {
        Set<String> producers = frameProducers.getOrDefault(frameName, Collections.emptySet());
        List<String> producerOps = new ArrayList<>();

        for (String producerFrame : producers) {
            producerOps.addAll(getOperationsInFrame(producerFrame));
        }

        return producerOps;
    }

    /**
     * Get operations that consume outputs from a specific frame
     */
    public List<String> getFrameConsumerOperations(String frameName) {
        Set<String> consumers = frameConsumers.getOrDefault(frameName, Collections.emptySet());
        List<String> consumerOps = new ArrayList<>();

        for (String consumerFrame : consumers) {
            consumerOps.addAll(getOperationsInFrame(consumerFrame));
        }

        return consumerOps;
    }

    /**
     * Get all child frames for a given parent frame
     */
    public Set<String> getChildFrames(String parentFrame) {
        return frameChildren.getOrDefault(parentFrame, Collections.emptySet());
    }

    /**
     * Get parent frame for a given frame
     */
    public String getParentFrame(String frameName) {
        return frameHierarchy.get(frameName);
    }

    /**
     * Check if a frame contains loops
     */
    public boolean frameHasLoops(String frameName) {
        FrameMetadata meta = frameMetadata.get(frameName);
        return meta != null && meta.hasLoops;
    }

    /**
     * Check if a frame contains conditionals
     */
    public boolean frameHasConditionals(String frameName) {
        FrameMetadata meta = frameMetadata.get(frameName);
        return meta != null && meta.hasConditionals;
    }

    /**
     * Get maximum iteration count for variables in a frame
     */
    public int getFrameMaxIterations(String frameName) {
        FrameMetadata meta = frameMetadata.get(frameName);
        return meta != null ? meta.maxIterations : 0;
    }

    /**
     * Get all frames that this frame depends on (transitive closure)
     */
    public Set<String> getAllFrameDependencies(String frameName) {
        Set<String> allDeps = new HashSet<>();
        collectFrameDependencies(frameName, allDeps, new HashSet<>());
        return allDeps;
    }

    /**
     * Get operations by frame transition type
     */
    public Set<String> getOperationsByTransition(FrameTransition transitionType) {
        switch (transitionType) {
            case ENTER: return enterOperations;
            case EXIT: return exitOperations;
            case SWITCH: return switchOperations;
            case MERGE: return mergeOperations;
            case NEXT_ITERATION: return nextIterationOperations;
            case LOOP_CONDITION: return loopConditionOperations;
            default: return Collections.emptySet();
        }
    }

    /**
     * Find operations that execute before entering a specific frame
     */
    public List<String> getPreFrameOperations(String frameName) {
        List<String> preOps = new ArrayList<>();
        FrameMetadata meta = frameMetadata.get(frameName);

        if (meta != null && !meta.entryPoints.isEmpty()) {
            String firstEntryOp = meta.entryPoints.get(0);
            int entryIndex = executionOrder.indexOf(firstEntryOp);

            if (entryIndex > 0) {
                for (int i = 0; i < entryIndex; i++) {
                    String opName = executionOrder.get(i);
                    String opFrame = getOperationFrame(opName);
                    if (opFrame == null || !opFrame.equals(frameName)) {
                        preOps.add(opName);
                    }
                }
            }
        }

        return preOps;
    }

    /**
     * Find operations that execute after exiting a specific frame
     */
    public List<String> getPostFrameOperations(String frameName) {
        List<String> postOps = new ArrayList<>();
        FrameMetadata meta = frameMetadata.get(frameName);

        if (meta != null && !meta.exitPoints.isEmpty()) {
            String lastExitOp = meta.exitPoints.get(meta.exitPoints.size() - 1);
            int exitIndex = executionOrder.indexOf(lastExitOp);

            if (exitIndex >= 0 && exitIndex < executionOrder.size() - 1) {
                for (int i = exitIndex + 1; i < executionOrder.size(); i++) {
                    String opName = executionOrder.get(i);
                    String opFrame = getOperationFrame(opName);
                    if (opFrame == null || !opFrame.equals(frameName)) {
                        postOps.add(opName);
                    }
                }
            }
        }

        return postOps;
    }

    // Analysis methods

    /**
     * Get all frames at a specific nesting depth
     */
    public Set<String> getFramesAtDepth(int depth) {
        return frameDepth.entrySet().stream()
                .filter(e -> e.getValue() == depth)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }

    /**
     * Get the execution path through frames
     */
    public List<String> getFrameExecutionPath() {
        List<String> path = new ArrayList<>();
        Map<String, Integer> frameFirstOp = new HashMap<>();

        // Find first operation in each frame
        for (int i = 0; i < executionOrder.size(); i++) {
            String opName = executionOrder.get(i);
            OperationInfo opInfo = operations.get(opName);
            if (opInfo != null && opInfo.getFrameInfo() != null && opInfo.getFrameInfo().outputFrame != null) {
                String frame = opInfo.getFrameInfo().outputFrame;
                if (!frameFirstOp.containsKey(frame)) {
                    frameFirstOp.put(frame, i);
                }
            }
        }

        // Sort frames by first operation order
        return frameFirstOp.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    /**
     * Analyze frame dependencies and detect circular dependencies
     */
    public Map<String, Set<String>> analyzeFrameDependencies() {
        Map<String, Set<String>> result = new HashMap<>();

        for (String frame : frameMetadata.keySet()) {
            Set<String> allDeps = new HashSet<>();
            collectFrameDependencies(frame, allDeps, new HashSet<>());
            result.put(frame, allDeps);
        }

        return result;
    }

    private void collectFrameDependencies(String frame, Set<String> allDeps, Set<String> visited) {
        if (visited.contains(frame)) {
            // Circular dependency detected
            cycles.add("Frame circular dependency involving: " + frame);
            return;
        }

        visited.add(frame);
        Set<String> directDeps = frameDependencies.getOrDefault(frame, Collections.emptySet());
        allDeps.addAll(directDeps);

        for (String dep : directDeps) {
            collectFrameDependencies(dep, allDeps, visited);
        }

        visited.remove(frame);
    }

    /**
     * Get detailed frame statistics
     */
    public Map<String, Object> getDetailedFrameStats() {
        Map<String, Object> stats = new HashMap<>();

        stats.put("totalFrames", frameMetadata.size());
        stats.put("maxFrameDepth", frameDepth.values().stream().mapToInt(Integer::intValue).max().orElse(0));
        stats.put("framesWithLoops", frameMetadata.values().stream().mapToLong(m -> m.hasLoops ? 1 : 0).sum());
        stats.put("framesWithConditionals", frameMetadata.values().stream().mapToLong(m -> m.hasConditionals ? 1 : 0).sum());
        stats.put("totalCrossFrameReferences", crossFrameReferences.values().stream().mapToInt(List::size).sum());

        // Frame type distribution
        Map<FrameType, Long> typeDistribution = frameMetadata.values().stream()
                .collect(Collectors.groupingBy(m -> m.frameType, Collectors.counting()));
        stats.put("frameTypeDistribution", typeDistribution);

        // Transition statistics
        Map<FrameTransition, Integer> totalTransitions = new HashMap<>();
        for (FrameMetadata meta : frameMetadata.values()) {
            for (Map.Entry<FrameTransition, Integer> entry : meta.transitionCounts.entrySet()) {
                totalTransitions.merge(entry.getKey(), entry.getValue(), Integer::sum);
            }
        }
        stats.put("frameTransitions", totalTransitions);

        return stats;
    }

    // Legacy methods (maintained for compatibility)

    public void addVariable(String name, VariableType type, DataType dataType) {
        variables.put(name, new VariableInfo(name, type, dataType));
    }

    public void addOperation(String opName, String opType, String className, List<String> inputs, List<String> outputs) {
        operations.put(opName, new OperationInfo(opName, opType, className, inputs, outputs));
    }

    public void addLeafVariable(String name) {
        leafVariables.add(name);
    }

    public void addTrainableVariable(String name) {
        trainableVariables.add(name);
    }

    public void addSequenceVariable(String name) {
        sequenceVariables.add(name);
    }

    public void addMissingVariable(String name) {
        missingVariables.add(name);
    }

    public void addOrphanedVariable(String varName, String missingOp) {
        orphanedVariables.add(varName + (missingOp != null ? " (op: " + missingOp + ")" : ""));
    }

    public void addCycle(String name) {
        cycles.add(name);
    }

    public void addControlDependencies(String opName, List<String> deps) {
        controlDependencies.put(opName, new ArrayList<>(deps));
    }

    public void addVariableControlDependencies(String opName, List<String> deps) {
        variableControlDependencies.put(opName, new ArrayList<>(deps));
    }

    public boolean isOperationProcessed(String opName) {
        return operations.containsKey(opName);
    }

    /**
     * Enhanced format summary with detailed frame information
     */
    public String formatSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== SameDiff Execution DAG Analysis ===\n\n");

        // Basic summary statistics
        sb.append("Summary:\n");
        sb.append(String.format("  Requested outputs: %d\n", requestedOutputs.size()));
        sb.append(String.format("  Total operations: %d\n", operations.size()));
        sb.append(String.format("  Total variables: %d\n", variables.size()));
        sb.append(String.format("  Leaf variables: %d\n", leafVariables.size()));
        sb.append(String.format("  Trainable variables: %d\n", trainableVariables.size()));

        // Enhanced frame analysis
        sb.append("\nFrame Analysis:\n");
        sb.append(String.format("  Total frames: %d\n", frameMetadata.size()));
        sb.append(String.format("  Max frame depth: %d\n", frameDepth.values().stream().mapToInt(Integer::intValue).max().orElse(0)));
        sb.append(String.format("  Frames with loops: %d\n", frameMetadata.values().stream().mapToLong(m -> m.hasLoops ? 1 : 0).sum()));
        sb.append(String.format("  Frames with conditionals: %d\n", frameMetadata.values().stream().mapToLong(m -> m.hasConditionals ? 1 : 0).sum()));
        sb.append(String.format("  Cross-frame references: %d\n", crossFrameReferences.values().stream().mapToInt(List::size).sum()));

        // Frame hierarchy
        if (!frameMetadata.isEmpty()) {
            sb.append("\nFrame Hierarchy:\n");
            for (String rootFrame : getFramesAtDepth(0)) {
                printFrameHierarchy(sb, rootFrame, 0);
            }
        }

        // Frame execution path
        List<String> framePath = getFrameExecutionPath();
        if (!framePath.isEmpty()) {
            sb.append("\nFrame Execution Path:\n");
            for (int i = 0; i < framePath.size(); i++) {
                String frame = framePath.get(i);
                FrameMetadata meta = frameMetadata.get(frame);
                sb.append(String.format("  %d. %s (%s) - %d ops, %d vars\n",
                        i + 1, frame, meta.frameType, meta.totalOperations, meta.totalVariables));
            }
        }

        // Issues
        if (!missingVariables.isEmpty() || !orphanedVariables.isEmpty() || !cycles.isEmpty()) {
            sb.append("\nISSUES DETECTED:\n");
            if (!missingVariables.isEmpty()) {
                sb.append(String.format("  Missing variables: %s\n", missingVariables));
            }
            if (!orphanedVariables.isEmpty()) {
                sb.append(String.format("  Orphaned variables: %s\n", orphanedVariables));
            }
            if (!cycles.isEmpty()) {
                sb.append(String.format("  Potential cycles: %s\n", cycles));
            }
        }

        return sb.toString();
    }

    private void printFrameHierarchy(StringBuilder sb, String frameName, int depth) {
        FrameMetadata meta = frameMetadata.get(frameName);
        if (meta == null) return;

        String indentStr = "  ".repeat(depth);
        sb.append(String.format("%s- %s (%s): %d ops, %d vars, max iter: %d\n",
                indentStr, frameName, meta.frameType, meta.totalOperations, meta.totalVariables, meta.maxIterations));

        for (String child : meta.childFrames) {
            printFrameHierarchy(sb, child, depth + 1);
        }
    }

    /**
     * Get control flow statistics (enhanced)
     */
    public Map<String, Integer> getControlFlowStats() {
        Map<String, Integer> stats = new HashMap<>();
        stats.put("enterOps", enterOperations.size());
        stats.put("exitOps", exitOperations.size());
        stats.put("switchOps", switchOperations.size());
        stats.put("mergeOps", mergeOperations.size());
        stats.put("nextIterationOps", nextIterationOperations.size());
        stats.put("loopConditionOps", loopConditionOperations.size());
        stats.put("totalFrames", framesUsed.size());
        stats.put("maxFrameDepth", frameDepth.values().stream().mapToInt(Integer::intValue).max().orElse(0));
        return stats;
    }

    /**
     * Get operations that directly produce the requested outputs
     */
    public List<String> getOutputProducingOperations() {
        List<String> result = new ArrayList<>();
        for (String output : requestedOutputs) {
            for (OperationInfo op : operations.values()) {
                if (op.getOutputs() != null && op.getOutputs().contains(output)) {
                    result.add(op.getOperationName());
                    break;
                }
            }
        }
        return result;
    }

    /**
     * Get the depth of the execution DAG (longest path)
     */
    public int getExecutionDepth() {
        return executionOrder.size();
    }

    // Getters for frame-related data structures
    public Map<String, FrameMetadata> getFrameMetadata() {
        return Collections.unmodifiableMap(frameMetadata);
    }

    public Map<String, List<CrossFrameReference>> getCrossFrameReferences() {
        return Collections.unmodifiableMap(crossFrameReferences);
    }

    public Map<String, List<FrameTransitionInfo>> getFrameTransitions() {
        return Collections.unmodifiableMap(frameTransitions);
    }

    public Map<String, Set<String>> getFrameDependencies() {
        return Collections.unmodifiableMap(frameDependencies);
    }

    public Map<String, List<String>> getFrameExecutionOrder() {
        return Collections.unmodifiableMap(frameExecutionOrder);
    }

    public Map<String, Set<String>> getFrameInputVariables() {
        return Collections.unmodifiableMap(frameInputVariables);
    }

    public Map<String, Set<String>> getFrameOutputVariables() {
        return Collections.unmodifiableMap(frameOutputVariables);
    }

    public Map<String, Set<String>> getFrameBoundaryOperations() {
        return Collections.unmodifiableMap(frameBoundaryOperations);
    }

    public Map<String, Set<String>> getFrameProducers() {
        return Collections.unmodifiableMap(frameProducers);
    }

    public Map<String, Set<String>> getFrameConsumers() {
        return Collections.unmodifiableMap(frameConsumers);
    }

    public Map<String, Integer> getFrameDepth() {
        return Collections.unmodifiableMap(frameDepth);
    }

    public Map<String, String> getFrameHierarchy() {
        return Collections.unmodifiableMap(frameHierarchy);
    }

    public Map<String, Set<String>> getFrameChildren() {
        return Collections.unmodifiableMap(frameChildren);
    }

    public Map<String, Map<String, Integer>> getFrameIterationCounts() {
        return Collections.unmodifiableMap(frameIterationCounts);
    }

    public Set<String> getEnterOperations() {
        return Collections.unmodifiableSet(enterOperations);
    }

    public Set<String> getExitOperations() {
        return Collections.unmodifiableSet(exitOperations);
    }

    public Set<String> getSwitchOperations() {
        return Collections.unmodifiableSet(switchOperations);
    }

    public Set<String> getMergeOperations() {
        return Collections.unmodifiableSet(mergeOperations);
    }

    public Set<String> getNextIterationOperations() {
        return Collections.unmodifiableSet(nextIterationOperations);
    }

    public Set<String> getLoopConditionOperations() {
        return Collections.unmodifiableSet(loopConditionOperations);
    }

    public Map<String, VariableInfo> getVariables() {
        return Collections.unmodifiableMap(variables);
    }

    public Map<String, OperationInfo> getOperations() {
        return Collections.unmodifiableMap(operations);
    }

    public Set<String> getLeafVariables() {
        return Collections.unmodifiableSet(leafVariables);
    }

    public Set<String> getTrainableVariables() {
        return Collections.unmodifiableSet(trainableVariables);
    }

    public Set<String> getSequenceVariables() {
        return Collections.unmodifiableSet(sequenceVariables);
    }

    public List<String> getMissingVariables() {
        return Collections.unmodifiableList(missingVariables);
    }

    public List<String> getOrphanedVariables() {
        return Collections.unmodifiableList(orphanedVariables);
    }

    public List<String> getCycles() {
        return Collections.unmodifiableList(cycles);
    }

    public Map<String, List<String>> getControlDependencies() {
        return Collections.unmodifiableMap(controlDependencies);
    }

    public Map<String, List<String>> getVariableControlDependencies() {
        return Collections.unmodifiableMap(variableControlDependencies);
    }
}