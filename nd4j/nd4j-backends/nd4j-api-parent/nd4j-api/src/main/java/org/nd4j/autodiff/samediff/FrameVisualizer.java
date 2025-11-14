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
 * Utility class for visualizing frame structures and generating textual representations
 */
public class FrameVisualizer {
    
    /**
     * Generate a detailed textual representation of the frame hierarchy
     */
    public static String generateFrameHierarchyDiagram(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Hierarchy Diagram\n");
        sb.append("======================\n\n");
        
        Set<String> rootFrames = plan.getFramesAtDepth(0);
        if (rootFrames.isEmpty()) {
            sb.append("No frames detected\n");
            return sb.toString();
        }
        
        for (String rootFrame : rootFrames) {
            printFrameTree(plan, rootFrame, 0, sb, new HashSet<>());
        }
        
        return sb.toString();
    }
    
    /**
     * Generate execution flow diagram showing frame transitions
     */
    public static String generateExecutionFlowDiagram(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Execution Flow Diagram\n");
        sb.append("=====================\n\n");
        
        List<String> executionOrder = plan.getExecutionOrder();
        String currentFrame = null;
        int frameCounter = 1;
        
        for (String opName : executionOrder) {
            String opFrame = plan.getOperationFrame(opName);
            
            if (opFrame != null && !opFrame.equals(currentFrame)) {
                if (currentFrame != null) {
                    sb.append("    └─ End Frame: ").append(currentFrame).append("\n");
                    sb.append("       ↓\n");
                }
                sb.append("[").append(frameCounter++).append("] Start Frame: ").append(opFrame).append("\n");
                currentFrame = opFrame;
            }
            
            if (opFrame != null) {
                sb.append("    ├─ ").append(opName);
                OperationInfo opInfo = plan.getOperations().get(opName);
                if (opInfo != null && opInfo.getFrameInfo() != null &&
                    opInfo.getFrameInfo().frameTransition != FrameTransition.NONE) {
                    sb.append(" [").append(opInfo.getFrameInfo().frameTransition).append("]");
                }
                sb.append("\n");
            }
        }
        
        if (currentFrame != null) {
            sb.append("    └─ End Frame: ").append(currentFrame).append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Generate cross-frame dependency matrix
     */
    public static String generateDependencyMatrix(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Dependency Matrix\n");
        sb.append("======================\n\n");
        
        List<String> frames = new ArrayList<>(plan.getFrameMetadata().keySet());
        frames.sort(String::compareTo);
        
        if (frames.isEmpty()) {
            sb.append("No frames to analyze\n");
            return sb.toString();
        }
        
        // Header
        sb.append("     ");
        for (String frame : frames) {
            sb.append(String.format("%8s", frame.substring(0, Math.min(7, frame.length()))));
        }
        sb.append("\n");
        
        // Matrix rows
        Map<String, Set<String>> dependencies = plan.getFrameDependencies();
        for (String fromFrame : frames) {
            sb.append(String.format("%-8s", fromFrame.substring(0, Math.min(7, fromFrame.length()))));
            Set<String> deps = dependencies.getOrDefault(fromFrame, Collections.emptySet());
            
            for (String toFrame : frames) {
                if (fromFrame.equals(toFrame)) {
                    sb.append("    -   ");
                } else if (deps.contains(toFrame)) {
                    sb.append("    X   ");
                } else {
                    sb.append("    .   ");
                }
            }
            sb.append("\n");
        }
        
        sb.append("\nLegend: X = depends on, . = no dependency, - = self\n");
        return sb.toString();
    }
    
    /**
     * Generate frame statistics summary
     */
    public static String generateFrameStatistics(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Statistics Summary\n");
        sb.append("=======================\n\n");
        
        Map<String, Object> stats = plan.getDetailedFrameStats();
        
        sb.append("Overall Statistics:\n");
        sb.append("  Total Frames: ").append(stats.get("totalFrames")).append("\n");
        sb.append("  Max Depth: ").append(stats.get("maxFrameDepth")).append("\n");
        sb.append("  Frames with Loops: ").append(stats.get("framesWithLoops")).append("\n");
        sb.append("  Frames with Conditionals: ").append(stats.get("framesWithConditionals")).append("\n");
        sb.append("  Cross-frame References: ").append(stats.get("totalCrossFrameReferences")).append("\n\n");
        
        // Frame type distribution
        @SuppressWarnings("unchecked")
        Map<FrameType, Long> typeDistribution = 
            (Map<FrameType, Long>) stats.get("frameTypeDistribution");
        
        if (typeDistribution != null && !typeDistribution.isEmpty()) {
            sb.append("Frame Type Distribution:\n");
            for (Map.Entry<FrameType, Long> entry : typeDistribution.entrySet()) {
                sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
            sb.append("\n");
        }
        
        // Transition statistics
        @SuppressWarnings("unchecked")
        Map<FrameTransition, Integer> transitions = 
            (Map<FrameTransition, Integer>) stats.get("frameTransitions");
        
        if (transitions != null && !transitions.isEmpty()) {
            sb.append("Frame Transitions:\n");
            for (Map.Entry<FrameTransition, Integer> entry : transitions.entrySet()) {
                sb.append("  ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
            sb.append("\n");
        }
        
        // Individual frame details
        sb.append("Individual Frame Details:\n");
        sb.append("------------------------\n");
        
        List<FrameMetadata> sortedFrames = plan.getFrameMetadata().values().stream()
            .sorted(Comparator.comparing(m -> m.frameName))
            .collect(Collectors.toList());
        
        for (FrameMetadata meta : sortedFrames) {
            sb.append("Frame: ").append(meta.frameName).append("\n");
            sb.append("  Type: ").append(meta.frameType).append("\n");
            sb.append("  Depth: ").append(meta.depth).append("\n");
            sb.append("  Parent: ").append(meta.parentFrame != null ? meta.parentFrame : "None").append("\n");
            sb.append("  Operations: ").append(meta.totalOperations).append("\n");
            sb.append("  Variables: ").append(meta.totalVariables).append("\n");
            sb.append("  Max Iterations: ").append(meta.maxIterations).append("\n");
            sb.append("  Has Loops: ").append(meta.hasLoops).append("\n");
            sb.append("  Has Conditionals: ").append(meta.hasConditionals).append("\n");
            
            if (!meta.entryPoints.isEmpty()) {
                sb.append("  Entry Points: ").append(meta.entryPoints).append("\n");
            }
            if (!meta.exitPoints.isEmpty()) {
                sb.append("  Exit Points: ").append(meta.exitPoints).append("\n");
            }
            if (!meta.childFrames.isEmpty()) {
                sb.append("  Child Frames: ").append(meta.childFrames).append("\n");
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Generate a compact frame execution timeline
     */
    public static String generateExecutionTimeline(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Execution Timeline\n");
        sb.append("=======================\n\n");
        
        List<String> executionOrder = plan.getExecutionOrder();
        Map<String, Integer> frameStartPos = new HashMap<>();
        Map<String, Integer> frameEndPos = new HashMap<>();
        
        // Find frame start and end positions
        for (int i = 0; i < executionOrder.size(); i++) {
            String opName = executionOrder.get(i);
            String frame = plan.getOperationFrame(opName);
            if (frame != null) {
                frameStartPos.putIfAbsent(frame, i);
                frameEndPos.put(frame, i);
            }
        }
        
        // Create timeline visualization
        List<String> frames = frameStartPos.keySet().stream()
            .sorted(Comparator.comparing(frameStartPos::get))
            .collect(Collectors.toList());
        
        int timelineLength = executionOrder.size();
        char[] timeline = new char[timelineLength];
        Arrays.fill(timeline, '.');
        
        for (String frame : frames) {
            int start = frameStartPos.get(frame);
            int end = frameEndPos.get(frame);
            char symbol = getFrameSymbol(frame);
            
            for (int i = start; i <= end; i++) {
                timeline[i] = symbol;
            }
        }
        
        // Print timeline with labels
        sb.append("Timeline (each character = 1 operation):\n");
        sb.append("Position: ");
        for (int i = 0; i < Math.min(timelineLength, 100); i += 10) {
            sb.append(String.format("%-10d", i));
        }
        sb.append("\n");
        
        sb.append("Frames:   ");
        for (int i = 0; i < Math.min(timelineLength, 100); i++) {
            sb.append(timeline[i]);
        }
        sb.append("\n\n");
        
        // Frame legend
        sb.append("Frame Legend:\n");
        for (String frame : frames) {
            char symbol = getFrameSymbol(frame);
            int start = frameStartPos.get(frame);
            int end = frameEndPos.get(frame);
            sb.append("  ").append(symbol).append(" = ").append(frame)
              .append(" (ops ").append(start).append("-").append(end).append(")\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Generate cross-frame variable flow diagram
     */
    public static String generateVariableFlowDiagram(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Cross-Frame Variable Flow\n");
        sb.append("========================\n\n");
        
        Map<String, List<CrossFrameReference>> crossRefs = plan.getCrossFrameReferences();
        
        if (crossRefs.isEmpty()) {
            sb.append("No cross-frame variable references found\n");
            return sb.toString();
        }
        
        // Group by source frame
        Map<String, List<CrossFrameReference>> bySourceFrame = new HashMap<>();
        for (List<CrossFrameReference> refs : crossRefs.values()) {
            for (CrossFrameReference ref : refs) {
                bySourceFrame.computeIfAbsent(ref.sourceFrame, k -> new ArrayList<>()).add(ref);
            }
        }
        
        for (Map.Entry<String, List<CrossFrameReference>> entry : bySourceFrame.entrySet()) {
            String sourceFrame = entry.getKey();
            List<CrossFrameReference> refs = entry.getValue();
            
            sb.append("From Frame: ").append(sourceFrame).append("\n");
            
            Map<String, List<CrossFrameReference>> byTarget = refs.stream()
                .collect(Collectors.groupingBy(ref -> ref.targetFrame));
            
            for (Map.Entry<String, List<CrossFrameReference>> targetEntry : byTarget.entrySet()) {
                String targetFrame = targetEntry.getKey();
                List<CrossFrameReference> targetRefs = targetEntry.getValue();
                
                sb.append("  └─> To Frame: ").append(targetFrame).append("\n");
                for (CrossFrameReference ref : targetRefs) {
                    sb.append("      ├─ Variable: ").append(ref.variableName)
                      .append(" (").append(ref.referenceType).append(")");
                    if (ref.mediatingOperation != null) {
                        sb.append(" via ").append(ref.mediatingOperation);
                    }
                    sb.append("\n");
                }
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Generate frame transition flow diagram
     */
    public static String generateFrameTransitionDiagram(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Transition Flow\n");
        sb.append("====================\n\n");
        
        Map<String, List<FrameTransitionInfo>> transitions = plan.getFrameTransitions();
        
        if (transitions.isEmpty()) {
            sb.append("No frame transitions found\n");
            return sb.toString();
        }
        
        for (Map.Entry<String, List<FrameTransitionInfo>> entry : transitions.entrySet()) {
            String fromFrame = entry.getKey();
            List<FrameTransitionInfo> frameTransitions = entry.getValue();
            
            sb.append("From Frame: ").append(fromFrame).append("\n");
            
            for (FrameTransitionInfo transition : frameTransitions) {
                sb.append("  ├─ ").append(transition.transitionType)
                  .append(" → ").append(transition.toFrame)
                  .append(" (").append(transition.operationName).append(")");
                
                if (transition.iterationChange != 0) {
                    sb.append(" [iter: ").append(transition.iterationChange > 0 ? "+" : "")
                      .append(transition.iterationChange).append("]");
                }
                
                if (!transition.affectedVariables.isEmpty()) {
                    sb.append(" vars: ").append(transition.affectedVariables);
                }
                
                sb.append("\n");
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Generate frame execution performance analysis
     */
    public static String generateFramePerformanceAnalysis(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Performance Analysis\n");
        sb.append("=========================\n\n");
        
        Map<String, Integer> frameCosts = FrameExecutionOptimizer.calculateFrameExecutionCost(plan);
        int totalCost = frameCosts.values().stream().mapToInt(Integer::intValue).sum();
        
        sb.append("Total Execution Cost: ").append(totalCost).append("\n\n");
        
        // Sort frames by cost (descending)
        List<Map.Entry<String, Integer>> sortedCosts = frameCosts.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .collect(Collectors.toList());
        
        sb.append("Frame Cost Analysis:\n");
        sb.append("Frame Name                    Cost    % of Total   Operations   Variables\n");
        sb.append("------------------------------------------------------------------------\n");
        
        for (Map.Entry<String, Integer> entry : sortedCosts) {
            String frameName = entry.getKey();
            int cost = entry.getValue();
            double percentage = (double) cost / totalCost * 100;
            
            FrameMetadata meta = plan.getFrameMetadata().get(frameName);
            int ops = meta != null ? meta.totalOperations : 0;
            int vars = meta != null ? meta.totalVariables : 0;
            
            sb.append(String.format("%-28s %6d %8.1f%% %12d %10d\n", 
                frameName.substring(0, Math.min(28, frameName.length())), 
                cost, percentage, ops, vars));
        }
        
        return sb.toString();
    }
    
    /**
     * Generate comprehensive frame analysis report
     */
    public static String generateComprehensiveFrameReport(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("═══════════════════════════════════════════════════════════════\n");
        sb.append("                    COMPREHENSIVE FRAME ANALYSIS REPORT\n");
        sb.append("═══════════════════════════════════════════════════════════════\n\n");
        
        // Executive summary
        sb.append("EXECUTIVE SUMMARY\n");
        sb.append("================\n");
        Map<String, Object> stats = plan.getDetailedFrameStats();
        sb.append("• Total Frames: ").append(stats.get("totalFrames")).append("\n");
        sb.append("• Maximum Nesting Depth: ").append(stats.get("maxFrameDepth")).append("\n");
        sb.append("• Frames with Control Flow: ").append(
            (Long) stats.get("framesWithLoops") + (Long) stats.get("framesWithConditionals")).append("\n");
        sb.append("• Cross-frame Data References: ").append(stats.get("totalCrossFrameReferences")).append("\n");
        
        // Get optimization opportunities
        FrameExecutionOptimizer.OptimizationReport optReport = FrameExecutionOptimizer.generateOptimizationReport(plan);
        sb.append("• Optimization Opportunities: ").append(
            optReport.consolidationSuggestions.size() + 
            optReport.parallelizationOpportunities.size() + 
            optReport.batchingOpportunities.size()).append("\n");
        sb.append("• Potential Performance Improvement: ").append(
            String.format("%.1f%%", (double) optReport.estimatedSavings / optReport.totalCurrentCost * 100)).append("\n\n");
        
        // Section divider
        sb.append("─────────────────────────────────────────────────────────────────\n\n");
        
        // Frame hierarchy
        sb.append("1. FRAME HIERARCHY\n");
        sb.append("==================\n");
        sb.append(generateFrameHierarchyDiagram(plan)).append("\n");
        
        // Frame statistics
        sb.append("2. DETAILED FRAME STATISTICS\n");
        sb.append("============================\n");
        sb.append(generateFrameStatistics(plan)).append("\n");
        
        // Performance analysis
        sb.append("3. PERFORMANCE ANALYSIS\n");
        sb.append("=======================\n");
        sb.append(generateFramePerformanceAnalysis(plan)).append("\n");
        
        // Variable flow analysis
        sb.append("4. CROSS-FRAME DATA FLOW\n");
        sb.append("========================\n");
        sb.append(generateVariableFlowDiagram(plan)).append("\n");
        
        // Frame transitions
        sb.append("5. FRAME TRANSITIONS\n");
        sb.append("===================\n");
        sb.append(generateFrameTransitionDiagram(plan)).append("\n");
        
        // Optimization recommendations
        sb.append("6. OPTIMIZATION RECOMMENDATIONS\n");
        sb.append("===============================\n");
        sb.append(optReport.generateSummary()).append("\n");
        
        // Timeline visualization
        sb.append("7. EXECUTION TIMELINE\n");
        sb.append("=====================\n");
        sb.append(generateExecutionTimeline(plan)).append("\n");
        
        sb.append("═══════════════════════════════════════════════════════════════\n");
        sb.append("                            END OF REPORT\n");
        sb.append("═══════════════════════════════════════════════════════════════\n");
        
        return sb.toString();
    }
    
    /**
     * Generate a simple ASCII art visualization of frame structure
     */
    public static String generateFrameStructureAscii(DAGExecutionPlan plan) {
        StringBuilder sb = new StringBuilder();
        sb.append("Frame Structure (ASCII)\n");
        sb.append("======================\n\n");
        
        Set<String> rootFrames = plan.getFramesAtDepth(0);
        for (String rootFrame : rootFrames) {
            generateFrameAscii(plan, rootFrame, sb, "", true);
        }
        
        return sb.toString();
    }
    
    private static void generateFrameAscii(DAGExecutionPlan plan, String frameName, 
                                         StringBuilder sb, String prefix, boolean isLast) {
        FrameMetadata meta = plan.getFrameMetadata().get(frameName);
        if (meta == null) return;
        
        // Draw current frame
        sb.append(prefix);
        sb.append(isLast ? "└── " : "├── ");
        sb.append(frameName);
        sb.append(" [").append(meta.frameType).append("]");
        sb.append(" (").append(meta.totalOperations).append(" ops, ");
        sb.append(meta.totalVariables).append(" vars)");
        if (meta.hasLoops) sb.append(" 🔄");
        if (meta.hasConditionals) sb.append(" 🔀");
        sb.append("\n");
        
        // Draw children
        List<String> children = new ArrayList<>(meta.childFrames);
        children.sort(String::compareTo);
        
        for (int i = 0; i < children.size(); i++) {
            boolean childIsLast = (i == children.size() - 1);
            String childPrefix = prefix + (isLast ? "    " : "│   ");
            generateFrameAscii(plan, children.get(i), sb, childPrefix, childIsLast);
        }
    }
    
    private static void printFrameTree(DAGExecutionPlan plan, String frameName, int depth, 
                                      StringBuilder sb, Set<String> visited) {
        if (visited.contains(frameName)) {
            sb.append("  ".repeat(depth)).append("└─ ").append(frameName).append(" [CYCLE DETECTED]\n");
            return;
        }
        
        visited.add(frameName);
        FrameMetadata meta = plan.getFrameMetadata().get(frameName);
        
        if (meta != null) {
            String prefix = depth == 0 ? "┌─ " : "├─ ";
            sb.append("  ".repeat(depth)).append(prefix).append(frameName);
            sb.append(" (").append(meta.frameType).append(", ");
            sb.append("ops: ").append(meta.totalOperations).append(", ");
            sb.append("vars: ").append(meta.totalVariables).append(")");
            
            if (meta.hasLoops) sb.append(" [LOOPS]");
            if (meta.hasConditionals) sb.append(" [CONDITIONALS]");
            sb.append("\n");
            
            Set<String> children = plan.getChildFrames(frameName);
            for (String child : children) {
                printFrameTree(plan, child, depth + 1, sb, visited);
            }
        }
        
        visited.remove(frameName);
    }
    
    private static char getFrameSymbol(String frameName) {
        // Simple hash-based symbol assignment
        int hash = frameName.hashCode();
        char[] symbols = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
        return symbols[Math.abs(hash) % symbols.length];
    }
}