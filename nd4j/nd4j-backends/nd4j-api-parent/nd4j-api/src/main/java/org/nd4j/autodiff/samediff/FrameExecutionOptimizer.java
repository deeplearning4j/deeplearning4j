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

import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility class for optimizing frame execution order and identifying optimization opportunities
 */
public class FrameExecutionOptimizer {
    
    /**
     * Optimize frame execution order to minimize frame transitions
     */
    public static List<String> optimizeFrameExecutionOrder(DAGExecutionPlan plan) {
        List<String> originalOrder = plan.getExecutionOrder();
        Map<String, String> opToFrame = new HashMap<>();
        
        // Build operation to frame mapping
        for (String opName : originalOrder) {
            String frame = plan.getOperationFrame(opName);
            if (frame != null) {
                opToFrame.put(opName, frame);
            }
        }
        
        // Group operations by frame while preserving dependencies
        Map<String, List<String>> frameOps = new LinkedHashMap<>();
        String currentFrame = null;
        
        for (String opName : originalOrder) {
            String opFrame = opToFrame.get(opName);
            if (opFrame != null) {
                if (!opFrame.equals(currentFrame)) {
                    currentFrame = opFrame;
                }
                frameOps.computeIfAbsent(currentFrame, k -> new ArrayList<>()).add(opName);
            }
        }
        
        // Rebuild execution order with optimized frame grouping
        List<String> optimizedOrder = new ArrayList<>();
        for (List<String> ops : frameOps.values()) {
            optimizedOrder.addAll(ops);
        }
        
        return optimizedOrder;
    }
    
    /**
     * Identify operations that could be moved to reduce frame switches
     */
    public static Map<String, String> identifyMovableOperations(DAGExecutionPlan plan) {
        Map<String, String> suggestions = new HashMap<>();
        List<String> execOrder = plan.getExecutionOrder();
        
        for (int i = 1; i < execOrder.size() - 1; i++) {
            String currentOp = execOrder.get(i);
            String prevOp = execOrder.get(i - 1);
            String nextOp = execOrder.get(i + 1);
            
            String currentFrame = plan.getOperationFrame(currentOp);
            String prevFrame = plan.getOperationFrame(prevOp);
            String nextFrame = plan.getOperationFrame(nextOp);
            
            // If current operation is in different frame but could be moved
            if (currentFrame != null && prevFrame != null && nextFrame != null) {
                if (!currentFrame.equals(prevFrame) && !currentFrame.equals(nextFrame) && 
                    prevFrame.equals(nextFrame)) {
                    
                    // Check if operation has dependencies that would allow moving
                    if (canMoveOperation(plan, currentOp, prevFrame)) {
                        suggestions.put(currentOp, "Could move to frame: " + prevFrame + " to reduce frame switches");
                    }
                }
            }
        }
        
        return suggestions;
    }
    
    /**
     * Find frame boundaries that could be optimized
     */
    public static List<FrameBoundaryOptimization> findBoundaryOptimizations(DAGExecutionPlan plan) {
        List<FrameBoundaryOptimization> optimizations = new ArrayList<>();
        
        for (String frameName : plan.getFrameMetadata().keySet()) {
            Set<String> boundaryOps = plan.getFrameBoundaryOperations().getOrDefault(frameName, Collections.emptySet());
            
            for (String boundaryOp : boundaryOps) {
                FrameBoundaryOptimization opt = analyzeBoundaryOperation(plan, frameName, boundaryOp);
                if (opt != null) {
                    optimizations.add(opt);
                }
            }
        }
        
        return optimizations;
    }
    
    /**
     * Calculate frame execution cost based on transitions and operations
     */
    public static Map<String, Integer> calculateFrameExecutionCost(DAGExecutionPlan plan) {
        Map<String, Integer> costs = new HashMap<>();
        
        for (String frameName : plan.getFrameMetadata().keySet()) {
            int cost = 0;
            
            // Base cost from number of operations
            List<String> frameOps = plan.getOperationsInFrame(frameName);
            cost += frameOps.size();
            
            // Additional cost from frame transitions
            FrameMetadata meta = plan.getFrameMetadata().get(frameName);
            for (Map.Entry<FrameTransition, Integer> entry : meta.transitionCounts.entrySet()) {
                cost += entry.getValue() * getTransitionCost(entry.getKey());
            }
            
            // Cost from cross-frame references
            Set<String> inputs = plan.getFrameInputVariables().getOrDefault(frameName, Collections.emptySet());
            Set<String> outputs = plan.getFrameOutputVariables().getOrDefault(frameName, Collections.emptySet());
            cost += (inputs.size() + outputs.size()) * 2;
            
            costs.put(frameName, cost);
        }
        
        return costs;
    }
    
    /**
     * Suggest frame consolidation opportunities
     */
    public static List<FrameConsolidationSuggestion> suggestFrameConsolidations(DAGExecutionPlan plan) {
        List<FrameConsolidationSuggestion> suggestions = new ArrayList<>();
        
        // Find small frames that could be merged with their parent
        for (FrameMetadata meta : plan.getFrameMetadata().values()) {
            if (meta.totalOperations < 5 && meta.parentFrame != null) {
                String parentFrame = meta.parentFrame;
                FrameMetadata parentMeta = plan.getFrameMetadata().get(parentFrame);
                
                if (parentMeta != null && !meta.hasLoops && !meta.hasConditionals) {
                    FrameConsolidationSuggestion suggestion = new FrameConsolidationSuggestion();
                    suggestion.sourceFrame = meta.frameName;
                    suggestion.targetFrame = parentFrame;
                    suggestion.reason = "Small frame (" + meta.totalOperations + " ops) without control flow";
                    suggestion.estimatedSavings = calculateConsolidationSavings(plan, meta.frameName, parentFrame);
                    suggestions.add(suggestion);
                }
            }
        }
        
        return suggestions;
    }
    
    /**
     * Analyze parallel execution opportunities within frames
     */
    public static Map<String, List<ParallelizationOpportunity>> analyzeParallelizationOpportunities(DAGExecutionPlan plan) {
        Map<String, List<ParallelizationOpportunity>> opportunities = new HashMap<>();
        
        for (String frameName : plan.getFrameMetadata().keySet()) {
            List<String> frameOps = plan.getOperationsInFrame(frameName);
            List<ParallelizationOpportunity> frameOpportunities = new ArrayList<>();
            
            // Find operations that can be parallelized within the frame
            Map<String, Set<String>> opDependencies = buildOperationDependencies(plan, frameOps);
            List<Set<String>> parallelGroups = findParallelOperationGroups(opDependencies);
            
            for (Set<String> group : parallelGroups) {
                if (group.size() > 1) {
                    ParallelizationOpportunity opp = new ParallelizationOpportunity();
                    opp.frameName = frameName;
                    opp.parallelOperations = new ArrayList<>(group);
                    opp.estimatedSpeedup = calculateEstimatedSpeedup(group.size());
                    frameOpportunities.add(opp);
                }
            }
            
            if (!frameOpportunities.isEmpty()) {
                opportunities.put(frameName, frameOpportunities);
            }
        }
        
        return opportunities;
    }
    
    /**
     * Identify frames that could benefit from batching
     */
    public static List<BatchingOpportunity> identifyBatchingOpportunities(DAGExecutionPlan plan) {
        List<BatchingOpportunity> opportunities = new ArrayList<>();
        
        for (String frameName : plan.getFrameMetadata().keySet()) {
            FrameMetadata meta = plan.getFrameMetadata().get(frameName);
            
            // Look for frames with similar operation patterns
            if (meta.hasLoops && meta.maxIterations > 1) {
                List<String> frameOps = plan.getOperationsInFrame(frameName);
                
                // Analyze operation patterns for batching potential
                Map<String, Integer> opTypeCounts = countOperationTypes(plan, frameOps);
                
                if (hasBatchingPotential(opTypeCounts)) {
                    BatchingOpportunity opp = new BatchingOpportunity();
                    opp.frameName = frameName;
                    opp.iterationCount = meta.maxIterations;
                    opp.operationTypes = opTypeCounts;
                    opp.estimatedBenefit = calculateBatchingBenefit(meta.maxIterations, frameOps.size());
                    opportunities.add(opp);
                }
            }
        }
        
        return opportunities;
    }
    
    /**
     * Generate optimization report for the entire execution plan
     */
    public static OptimizationReport generateOptimizationReport(DAGExecutionPlan plan) {
        OptimizationReport report = new OptimizationReport();
        
        // Calculate current execution costs
        report.currentExecutionCosts = calculateFrameExecutionCost(plan);
        report.totalCurrentCost = report.currentExecutionCosts.values().stream().mapToInt(Integer::intValue).sum();
        
        // Find optimization opportunities
        report.movableOperations = identifyMovableOperations(plan);
        report.boundaryOptimizations = findBoundaryOptimizations(plan);
        report.consolidationSuggestions = suggestFrameConsolidations(plan);
        report.parallelizationOpportunities = analyzeParallelizationOpportunities(plan);
        report.batchingOpportunities = identifyBatchingOpportunities(plan);
        
        // Calculate potential savings
        report.estimatedSavings = calculateTotalSavings(report);
        report.optimizationPriority = prioritizeOptimizations(report);
        
        return report;
    }
    
    private static boolean canMoveOperation(DAGExecutionPlan plan, String opName, String targetFrame) {
        OperationInfo opInfo = plan.getOperations().get(opName);
        if (opInfo == null) return false;
        
        // Check if all input variables are available in target frame
        for (String input : opInfo.getInputs()) {
            String inputFrame = plan.getVariableFrame(input);
            if (inputFrame != null && !inputFrame.equals(targetFrame)) {
                // Check if input is available via cross-frame reference
                List<CrossFrameReference> refs = plan.getCrossFrameReferences().get(input);
                if (refs == null || refs.stream().noneMatch(ref -> ref.targetFrame.equals(targetFrame))) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    private static FrameBoundaryOptimization analyzeBoundaryOperation(DAGExecutionPlan plan, String frameName, String opName) {
        OperationInfo opInfo = plan.getOperations().get(opName);
        if (opInfo == null || opInfo.getFrameInfo() == null) return null;
        
        FrameTransition transition = opInfo.getFrameInfo().frameTransition;
        if (transition == FrameTransition.ENTER || transition == FrameTransition.EXIT) {
            FrameBoundaryOptimization opt = new FrameBoundaryOptimization();
            opt.frameName = frameName;
            opt.operationName = opName;
            opt.transitionType = transition;
            
            // Check if this boundary operation is necessary
            if (transition == FrameTransition.ENTER) {
                // Check if variables being entered are actually used in frame
                Set<String> frameVars = plan.getVariablesInFrame(frameName);
                boolean allInputsUsed = opInfo.getInputs().stream().allMatch(frameVars::contains);
                if (!allInputsUsed) {
                    opt.suggestion = "Some input variables may not be used in frame";
                    opt.priority = OptimizationPriority.LOW;
                }
            }
            
            return opt;
        }
        
        return null;
    }
    
    private static int getTransitionCost(FrameTransition transition) {
        switch (transition) {
            case ENTER: return 3;
            case EXIT: return 3;
            case SWITCH: return 5;
            case MERGE: return 4;
            case NEXT_ITERATION: return 6;
            case LOOP_CONDITION: return 4;
            default: return 1;
        }
    }
    
    private static int calculateConsolidationSavings(DAGExecutionPlan plan, String sourceFrame, String targetFrame) {
        int savings = 0;
        
        // Savings from eliminated frame transitions
        FrameMetadata sourceMeta = plan.getFrameMetadata().get(sourceFrame);
        if (sourceMeta != null) {
            for (Map.Entry<FrameTransition, Integer> entry : sourceMeta.transitionCounts.entrySet()) {
                savings += entry.getValue() * getTransitionCost(entry.getKey());
            }
        }
        
        // Savings from reduced cross-frame references
        Set<String> inputs = plan.getFrameInputVariables().getOrDefault(sourceFrame, Collections.emptySet());
        Set<String> outputs = plan.getFrameOutputVariables().getOrDefault(sourceFrame, Collections.emptySet());
        savings += (inputs.size() + outputs.size()) * 2;
        
        return savings;
    }
    
    private static Map<String, Set<String>> buildOperationDependencies(DAGExecutionPlan plan, List<String> frameOps) {
        Map<String, Set<String>> dependencies = new HashMap<>();
        
        for (String op : frameOps) {
            dependencies.put(op, plan.getDependencies().getOrDefault(op, Collections.emptySet()));
        }
        
        return dependencies;
    }
    
    private static List<Set<String>> findParallelOperationGroups(Map<String, Set<String>> dependencies) {
        List<Set<String>> parallelGroups = new ArrayList<>();
        Set<String> processed = new HashSet<>();
        
        for (String op : dependencies.keySet()) {
            if (!processed.contains(op)) {
                Set<String> group = new HashSet<>();
                findParallelGroup(op, dependencies, group, processed);
                if (group.size() > 1) {
                    parallelGroups.add(group);
                }
            }
        }
        
        return parallelGroups;
    }
    
    private static void findParallelGroup(String op, Map<String, Set<String>> dependencies, 
                                        Set<String> group, Set<String> processed) {
        if (processed.contains(op)) return;
        
        processed.add(op);
        group.add(op);
        
        // Find operations that don't depend on this one and this one doesn't depend on
        for (String otherOp : dependencies.keySet()) {
            if (!processed.contains(otherOp)) {
                Set<String> opDeps = dependencies.get(op);
                Set<String> otherDeps = dependencies.get(otherOp);
                
                if (!opDeps.contains(otherOp) && !otherDeps.contains(op)) {
                    findParallelGroup(otherOp, dependencies, group, processed);
                }
            }
        }
    }
    
    private static double calculateEstimatedSpeedup(int parallelOperations) {
        // Simple model: speedup = min(parallelOperations, available_cores) * efficiency
        int availableCores = Runtime.getRuntime().availableProcessors();
        double efficiency = 0.8; // Assume 80% parallel efficiency
        return Math.min(parallelOperations, availableCores) * efficiency;
    }
    
    private static Map<String, Integer> countOperationTypes(DAGExecutionPlan plan, List<String> operations) {
        Map<String, Integer> counts = new HashMap<>();
        
        for (String opName : operations) {
            OperationInfo opInfo = plan.getOperations().get(opName);
            if (opInfo != null) {
                counts.merge(opInfo.getOpType(), 1, Integer::sum);
            }
        }
        
        return counts;
    }
    
    private static boolean hasBatchingPotential(Map<String, Integer> opTypeCounts) {
        // Heuristic: if there are many similar operations, batching might help
        return opTypeCounts.values().stream().anyMatch(count -> count > 3);
    }
    
    private static double calculateBatchingBenefit(int iterations, int opsPerIteration) {
        // Simple model: benefit increases with iterations and operations
        return Math.log(iterations) * opsPerIteration * 0.1;
    }
    
    private static int calculateTotalSavings(OptimizationReport report) {
        int savings = 0;
        
        savings += report.consolidationSuggestions.stream()
            .mapToInt(s -> s.estimatedSavings).sum();
        
        savings += report.batchingOpportunities.stream()
            .mapToInt(b -> (int) b.estimatedBenefit).sum();
        
        return savings;
    }
    
    private static List<String> prioritizeOptimizations(OptimizationReport report) {
        List<String> priorities = new ArrayList<>();
        
        // High impact optimizations first
        if (!report.consolidationSuggestions.isEmpty()) {
            priorities.add("Frame consolidation: " + report.consolidationSuggestions.size() + " opportunities");
        }
        
        if (!report.batchingOpportunities.isEmpty()) {
            priorities.add("Batching optimization: " + report.batchingOpportunities.size() + " opportunities");
        }
        
        if (!report.parallelizationOpportunities.isEmpty()) {
            int totalParallelOps = report.parallelizationOpportunities.values().stream()
                .mapToInt(List::size).sum();
            priorities.add("Parallelization: " + totalParallelOps + " opportunities");
        }
        
        return priorities;
    }
    
    /**
     * Frame boundary optimization suggestion
     */
    public static class FrameBoundaryOptimization {
        public String frameName;
        public String operationName;
        public FrameTransition transitionType;
        public String suggestion;
        public OptimizationPriority priority = OptimizationPriority.MEDIUM;
    }
    
    /**
     * Frame consolidation suggestion
     */
    public static class FrameConsolidationSuggestion {
        public String sourceFrame;
        public String targetFrame;
        public String reason;
        public int estimatedSavings;
    }
    
    /**
     * Parallelization opportunity within a frame
     */
    public static class ParallelizationOpportunity {
        public String frameName;
        public List<String> parallelOperations = new ArrayList<>();
        public double estimatedSpeedup;
    }
    
    /**
     * Batching opportunity for repetitive operations
     */
    public static class BatchingOpportunity {
        public String frameName;
        public int iterationCount;
        public Map<String, Integer> operationTypes = new HashMap<>();
        public double estimatedBenefit;
    }
    
    /**
     * Comprehensive optimization report
     */
    public static class OptimizationReport {
        public Map<String, Integer> currentExecutionCosts = new HashMap<>();
        public int totalCurrentCost;
        public Map<String, String> movableOperations = new HashMap<>();
        public List<FrameBoundaryOptimization> boundaryOptimizations = new ArrayList<>();
        public List<FrameConsolidationSuggestion> consolidationSuggestions = new ArrayList<>();
        public Map<String, List<ParallelizationOpportunity>> parallelizationOpportunities = new HashMap<>();
        public List<BatchingOpportunity> batchingOpportunities = new ArrayList<>();
        public int estimatedSavings;
        public List<String> optimizationPriority = new ArrayList<>();
        
        public String generateSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Optimization Report Summary\n");
            sb.append("==========================\n\n");
            
            sb.append("Current Execution Cost: ").append(totalCurrentCost).append("\n");
            sb.append("Estimated Savings: ").append(estimatedSavings).append("\n");
            sb.append("Potential Improvement: ").append(String.format("%.1f%%", 
                (double) estimatedSavings / totalCurrentCost * 100)).append("\n\n");
            
            sb.append("Optimization Opportunities:\n");
            sb.append("- Movable Operations: ").append(movableOperations.size()).append("\n");
            sb.append("- Boundary Optimizations: ").append(boundaryOptimizations.size()).append("\n");
            sb.append("- Consolidation Suggestions: ").append(consolidationSuggestions.size()).append("\n");
            sb.append("- Parallelization Opportunities: ").append(parallelizationOpportunities.size()).append("\n");
            sb.append("- Batching Opportunities: ").append(batchingOpportunities.size()).append("\n\n");
            
            if (!optimizationPriority.isEmpty()) {
                sb.append("Recommended Priority:\n");
                for (int i = 0; i < optimizationPriority.size(); i++) {
                    sb.append((i + 1)).append(". ").append(optimizationPriority.get(i)).append("\n");
                }
            }
            
            return sb.toString();
        }
    }
    
    /**
     * Optimization priority levels
     */
    public enum OptimizationPriority {
        LOW, MEDIUM, HIGH, CRITICAL
    }
}