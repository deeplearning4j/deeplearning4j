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

/**
 * Utility class for analyzing frame structures and dependencies in DAG execution plans
 */
public class FrameAnalyzer {
    
    /**
     * Analyze frame execution patterns and detect potential issues
     */
    public static FrameAnalysisResult analyzeFrameExecution(DAGExecutionPlan plan) {
        FrameAnalysisResult result = new FrameAnalysisResult();
        
        // Analyze frame depth and nesting
        analyzeFrameNesting(plan, result);
        
        // Detect frame dependency cycles
        detectFrameCycles(plan, result);
        
        // Analyze frame transition patterns
        analyzeTransitionPatterns(plan, result);
        
        // Check for frame isolation issues
        checkFrameIsolation(plan, result);
        
        return result;
    }
    
    /**
     * Find the critical path through frames
     */
    public static List<String> findFrameCriticalPath(DAGExecutionPlan plan) {
        Map<String, Integer> frameOperationCounts = new HashMap<>();
        
        for (Map.Entry<String, List<String>> entry : plan.getFrameExecutionOrder().entrySet()) {
            frameOperationCounts.put(entry.getKey(), entry.getValue().size());
        }
        
        return frameOperationCounts.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .map(Map.Entry::getKey)
            .collect(ArrayList::new, (list, item) -> list.add(item), ArrayList::addAll);
    }
    
    /**
     * Calculate frame execution efficiency metrics
     */
    public static Map<String, Double> calculateFrameEfficiency(DAGExecutionPlan plan) {
        Map<String, Double> efficiency = new HashMap<>();
        
        for (String frameName : plan.getFrameMetadata().keySet()) {
            List<String> frameOps = plan.getOperationsInFrame(frameName);
            Set<String> frameVars = plan.getVariablesInFrame(frameName);
            
            if (!frameOps.isEmpty()) {
                double opsToVarsRatio = frameVars.isEmpty() ? 0.0 : (double) frameOps.size() / frameVars.size();
                efficiency.put(frameName, opsToVarsRatio);
            }
        }
        
        return efficiency;
    }
    
    /**
     * Find frames that could be parallelized
     */
    public static Set<Set<String>> findParallelizableFrames(DAGExecutionPlan plan) {
        Set<Set<String>> parallelGroups = new HashSet<>();
        Map<String, Set<String>> frameDeps = plan.analyzeFrameDependencies();
        
        // Find frames at the same depth with no dependencies between them
        Map<Integer, Set<String>> framesByDepth = new HashMap<>();
        for (Map.Entry<String, FrameMetadata> entry : plan.getFrameMetadata().entrySet()) {
            framesByDepth.computeIfAbsent(entry.getValue().depth, k -> new HashSet<>()).add(entry.getKey());
        }
        
        for (Set<String> framesAtDepth : framesByDepth.values()) {
            if (framesAtDepth.size() > 1) {
                Set<String> parallelizable = new HashSet<>();
                for (String frame : framesAtDepth) {
                    boolean canParallelize = true;
                    for (String otherFrame : framesAtDepth) {
                        if (!frame.equals(otherFrame)) {
                            Set<String> deps = frameDeps.getOrDefault(frame, Collections.emptySet());
                            if (deps.contains(otherFrame)) {
                                canParallelize = false;
                                break;
                            }
                        }
                    }
                    if (canParallelize) {
                        parallelizable.add(frame);
                    }
                }
                if (parallelizable.size() > 1) {
                    parallelGroups.add(parallelizable);
                }
            }
        }
        
        return parallelGroups;
    }
    
    private static void analyzeFrameNesting(DAGExecutionPlan plan, FrameAnalysisResult result) {
        int maxDepth = 0;
        Map<Integer, Integer> depthCounts = new HashMap<>();
        
        for (FrameMetadata meta : plan.getFrameMetadata().values()) {
            maxDepth = Math.max(maxDepth, meta.depth);
            depthCounts.merge(meta.depth, 1, Integer::sum);
        }
        
        result.maxNestingDepth = maxDepth;
        result.frameCountByDepth = depthCounts;
        
        // Flag deeply nested frames as potential issues
        if (maxDepth > 5) {
            result.warnings.add("Deep frame nesting detected (depth: " + maxDepth + "). Consider flattening.");
        }
    }
    
    private static void detectFrameCycles(DAGExecutionPlan plan, FrameAnalysisResult result) {
        Map<String, Set<String>> frameDeps = plan.getFrameDependencies();
        Set<String> visited = new HashSet<>();
        Set<String> recursionStack = new HashSet<>();
        
        for (String frame : plan.getFrameMetadata().keySet()) {
            if (!visited.contains(frame)) {
                if (hasCycleDFS(frame, frameDeps, visited, recursionStack, result.frameCycles)) {
                    result.hasCycles = true;
                }
            }
        }
    }
    
    private static boolean hasCycleDFS(String frame, Map<String, Set<String>> deps, 
                                      Set<String> visited, Set<String> stack, List<String> cycles) {
        visited.add(frame);
        stack.add(frame);
        
        Set<String> frameDeps = deps.getOrDefault(frame, Collections.emptySet());
        for (String dep : frameDeps) {
            if (!visited.contains(dep)) {
                if (hasCycleDFS(dep, deps, visited, stack, cycles)) {
                    return true;
                }
            } else if (stack.contains(dep)) {
                cycles.add("Cycle detected: " + frame + " -> " + dep);
                return true;
            }
        }
        
        stack.remove(frame);
        return false;
    }
    
    private static void analyzeTransitionPatterns(DAGExecutionPlan plan, FrameAnalysisResult result) {
        Map<FrameTransition, Integer> transitionCounts = new HashMap<>();
        
        for (FrameMetadata meta : plan.getFrameMetadata().values()) {
            for (Map.Entry<FrameTransition, Integer> entry : meta.transitionCounts.entrySet()) {
                transitionCounts.merge(entry.getKey(), entry.getValue(), Integer::sum);
            }
        }
        
        result.transitionPatterns = transitionCounts;
        
        // Analyze patterns for potential optimizations
        int enterExitRatio = transitionCounts.getOrDefault(FrameTransition.ENTER, 0) - 
                            transitionCounts.getOrDefault(FrameTransition.EXIT, 0);
        if (Math.abs(enterExitRatio) > 5) {
            result.warnings.add("Unbalanced ENTER/EXIT transitions (difference: " + enterExitRatio + ")");
        }
    }
    
    private static void checkFrameIsolation(DAGExecutionPlan plan, FrameAnalysisResult result) {
        for (String frameName : plan.getFrameMetadata().keySet()) {
            Set<String> inputs = plan.getFrameInputVariables().getOrDefault(frameName, Collections.emptySet());
            Set<String> outputs = plan.getFrameOutputVariables().getOrDefault(frameName, Collections.emptySet());
            
            if (inputs.isEmpty() && outputs.isEmpty()) {
                List<String> frameOps = plan.getOperationsInFrame(frameName);
                if (!frameOps.isEmpty()) {
                    result.isolatedFrames.add(frameName);
                }
            }
        }
        
        if (!result.isolatedFrames.isEmpty()) {
            result.warnings.add("Found " + result.isolatedFrames.size() + " isolated frames with no external I/O");
        }
    }
    
    /**
     * Result of frame analysis
     */
    public static class FrameAnalysisResult {
        public int maxNestingDepth;
        public Map<Integer, Integer> frameCountByDepth = new HashMap<>();
        public boolean hasCycles = false;
        public List<String> frameCycles = new ArrayList<>();
        public Map<FrameTransition, Integer> transitionPatterns = new HashMap<>();
        public List<String> isolatedFrames = new ArrayList<>();
        public List<String> warnings = new ArrayList<>();
        
        public boolean hasIssues() {
            return hasCycles || !isolatedFrames.isEmpty() || !warnings.isEmpty();
        }
        
        public String getSummary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Frame Analysis Summary:\n");
            sb.append("  Max nesting depth: ").append(maxNestingDepth).append("\n");
            sb.append("  Has cycles: ").append(hasCycles).append("\n");
            sb.append("  Isolated frames: ").append(isolatedFrames.size()).append("\n");
            sb.append("  Warnings: ").append(warnings.size()).append("\n");
            
            if (!warnings.isEmpty()) {
                sb.append("\nWarnings:\n");
                for (String warning : warnings) {
                    sb.append("  - ").append(warning).append("\n");
                }
            }
            
            return sb.toString();
        }
    }
}