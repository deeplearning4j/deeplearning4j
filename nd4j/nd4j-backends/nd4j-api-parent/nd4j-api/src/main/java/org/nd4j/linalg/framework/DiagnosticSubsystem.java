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

package org.nd4j.linalg.framework;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.framework.stats.DiagnosticIssue;
import org.nd4j.linalg.framework.stats.MemoryStats;
import org.nd4j.linalg.framework.leak.LeakDetector;
import org.nd4j.linalg.framework.leak.LeakReport;

import java.util.ArrayList;
import java.util.List;

/**
 * Diagnostic subsystem - provides leak detection and health monitoring.
 * 
 * @author ND4J Team
 */
@Slf4j
public final class DiagnosticSubsystem {
    
    /**
     * Run leak detection
     */
    public LeakReport runLeakDetection() {
        return LeakDetector.getInstance().runDetection();
    }
    
    /**
     * Get active diagnostic issues
     */
    public List<DiagnosticIssue> getActiveIssues() {
        List<DiagnosticIssue> issues = new ArrayList<>();
        
        // Check for memory pressure
        MemoryStats memStats = getMemoryStats();
        if (memStats.getHeapUtilizationPercent() > 90) {
            issues.add(DiagnosticIssue.builder()
                .issueId("MEM_HIGH_HEAP")
                .severity(DiagnosticIssue.Severity.WARNING)
                .category(DiagnosticIssue.Category.MEMORY)
                .title("High heap utilization")
                .description(String.format("Heap utilization at %.1f%%", memStats.getHeapUtilizationPercent()))
                .recommendedAction("Consider increasing heap size or reducing memory usage")
                .firstDetectedMs(System.currentTimeMillis())
                .lastUpdatedMs(System.currentTimeMillis())
                .build());
        }
        
        // Check for leaks
        LeakReport leakReport = runLeakDetection();
        if (leakReport.isLeaksDetected()) {
            issues.add(DiagnosticIssue.builder()
                .issueId("LEAK_DETECTED")
                .severity(leakReport.getSeverity() == LeakReport.Severity.CRITICAL ? 
                         DiagnosticIssue.Severity.CRITICAL : DiagnosticIssue.Severity.ERROR)
                .category(DiagnosticIssue.Category.LEAK)
                .title("Potential memory leak detected")
                .description(leakReport.getSummary())
                .recommendedAction("Review allocation patterns and ensure arrays are properly deallocated")
                .firstDetectedMs(System.currentTimeMillis())
                .lastUpdatedMs(System.currentTimeMillis())
                .build());
        }
        
        return issues;
    }
    
    /**
     * Get framework health status
     */
    public Framework.HealthStatus health() {
        List<DiagnosticIssue> issues = getActiveIssues();
        
        boolean hasCritical = issues.stream().anyMatch(i -> i.getSeverity() == DiagnosticIssue.Severity.CRITICAL);
        boolean hasError = issues.stream().anyMatch(i -> i.getSeverity() == DiagnosticIssue.Severity.ERROR);
        
        return Framework.HealthStatus.builder()
            .healthy(!hasCritical && !hasError)
            .issueCount(issues.size())
            .criticalCount((int) issues.stream().filter(i -> i.getSeverity() == DiagnosticIssue.Severity.CRITICAL).count())
            .errorCount((int) issues.stream().filter(i -> i.getSeverity() == DiagnosticIssue.Severity.ERROR).count())
            .warningCount((int) issues.stream().filter(i -> i.getSeverity() == DiagnosticIssue.Severity.WARNING).count())
            .build();
    }
    
    /**
     * Get memory stats (helper method)
     */
    private MemoryStats getMemoryStats() {
        // This would normally come from MemorySubsystem
        // For now, return a basic stats object
        Runtime runtime = Runtime.getRuntime();
        long heapUsed = runtime.totalMemory() - runtime.freeMemory();
        long heapMax = runtime.maxMemory();
        
        return MemoryStats.builder()
            .heapUsedBytes(heapUsed)
            .heapMaxBytes(heapMax)
            .build();
    }
}
