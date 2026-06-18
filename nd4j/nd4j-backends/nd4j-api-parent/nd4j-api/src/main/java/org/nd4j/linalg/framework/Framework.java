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
import org.nd4j.allocator.impl.MemoryTracker;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.framework.constant.ConstantCacheSubsystem;
import org.nd4j.linalg.framework.device.DevicePinningManager;
import org.nd4j.linalg.framework.device.DeviceSubsystem;
import org.nd4j.linalg.framework.device.TransferSubsystem;
import org.nd4j.linalg.framework.exec.ExecutionSubsystem;
import org.nd4j.lifecycle.LifecycleSubsystem;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.framework.leak.LeakDetector;
import org.nd4j.linalg.framework.leak.LeakReport;
import org.nd4j.linalg.framework.memory.MemoryState;
import org.nd4j.linalg.framework.memory.MemorySubsystem;
import org.nd4j.linalg.framework.stats.*;
import org.nd4j.linalg.framework.workspace.WorkspaceSubsystem;
import org.nd4j.linalg.framework.profiling.ProfilingSubsystem;

import java.util.ArrayList;
import java.util.List;

/**
 * Unified accessor for ND4J framework internals.
 * 
 * Provides transparent access to memory management, execution, device management,
 * op profiling, array lifecycle tracking, and diagnostic facilities.
 * 
 * <p>Hierarchy:</p>
 * <pre>
 * Nd4j.framework
 * ├── execution()
 * │   ├── dsp()           - DSP diagnostics (COMPILE, EXECUTE, MEMORY, etc.)
 * │   ├── opExecutioner() - Op execution control
 * │   └── kernels()       - Kernel selection and plugins
 * ├── memory()
 * │   ├── manager()       - MemoryManager access
 * │   ├── deallocator()   - DeallocatorService access
 * │   ├── tracker()       - MemoryTracker access
 * │   └── samples()       - Historical memory sampling
 * ├── device()
 * │   ├── affinity()      - AffinityManager access
 * │   └── info()          - Device information
 * ├── workspaces()        - Workspace management
 * ├── profiling()         - Op timing and profiling
 * ├── lifecycle()         - Array lifecycle tracking
 * └── diagnostics()       - Leak detection and health monitoring
 * </pre>
 * 
 * <p>Usage examples:</p>
 * <pre>
 *   // DSP diagnostics
 *   Nd4j.framework.execution().dsp().enableAll();
 *   Nd4j.framework.execution().dsp().getPlanReport();
 *   
 *   // Memory management
 *   Nd4j.framework.memory().manager().get();
 *   Nd4j.framework.memory().tracker().getFreeMemory(0);
 *   Nd4j.framework.memory().samples().start();
 *   
 *   // Device management
 *   Nd4j.framework.device().affinity().getDeviceForCurrentThread();
 *   Nd4j.framework.device().info().count();
 *   
 *   // Leak detection
 *   Nd4j.framework.diagnostics().runLeakDetection();
 * </pre>
 * 
 * @author ND4J Team
 */
@Slf4j
public final class Framework {
    
    /**
     * Singleton instance
     */
    private static final Framework INSTANCE = new Framework();
    
    /**
     * Execution subsystem (DSP, op executioner, kernels)
     */
    private final ExecutionSubsystem execution;
    
    /**
     * Memory subsystem (manager, deallocator, tracker, sampling)
     */
    private final MemorySubsystem memory;
    
    /**
     * Device subsystem (affinity, device info)
     */
    private final DeviceSubsystem device;
    
    /**
     * Workspace subsystem
     */
    private final WorkspaceSubsystem workspaces;
    
    /**
     * Profiling subsystem
     */
    private final ProfilingSubsystem profiling;
    
    /**
     * Lifecycle subsystem
     */
    private final LifecycleSubsystem lifecycle;
    
    /**
     * Diagnostic subsystem
     */
    private final DiagnosticSubsystem diagnostics;
    
    /**
     * Constant cache subsystem
     */
    private final ConstantCacheSubsystem constants;

    private Framework() {
        this.execution = new ExecutionSubsystem();
        this.memory = new MemorySubsystem();
        this.device = new DeviceSubsystem();
        this.workspaces = new WorkspaceSubsystem();
        this.profiling = new ProfilingSubsystem();
        this.lifecycle = new LifecycleSubsystem();
        this.diagnostics = new DiagnosticSubsystem();
        this.constants = new ConstantCacheSubsystem();
    }
    
    /**
     * Get singleton instance
     */
    public static Framework getInstance() {
        return INSTANCE;
    }
    
    /**
     * Execution subsystem - DSP, op executioner, kernel management
     */
    public ExecutionSubsystem execution() {
        return execution;
    }
    
    /**
     * Memory subsystem - memory management, allocation tracking, sampling
     */
    public MemorySubsystem memory() {
        return memory;
    }
    
    /**
     * Device subsystem - device management, affinity control
     */
    public DeviceSubsystem device() {
        return device;
    }
    
    /**
     * Workspace subsystem - workspace management
     */
    public WorkspaceSubsystem workspaces() {
        return workspaces;
    }
    
    /**
     * Profiling subsystem - op timing and performance tracking
     */
    public ProfilingSubsystem profiling() {
        return profiling;
    }
    
    /**
     * Lifecycle subsystem - array lifecycle tracking
     */
    public LifecycleSubsystem lifecycle() {
        return lifecycle;
    }
    
    /**
     * Diagnostic subsystem - leak detection, health monitoring
     */
    public DiagnosticSubsystem diagnostics() {
        return diagnostics;
    }
    
    /**
     * Constant cache subsystem - constant buffers, shape buffers, TAD helpers
     */
    public ConstantCacheSubsystem constants() {
        return constants;
    }
    
    /**
     * Snapshot of entire framework state for debugging
     */
    public FrameworkSnapshot snapshot() {
        return new FrameworkSnapshot(
            System.currentTimeMillis(),
            memory.stats(),
            profiling.stats(),
            lifecycle.stats(),
            workspaces.stats(),
            diagnostics.getActiveIssues()
        );
    }
    
    /**
     * Get comprehensive state summary of all framework subsystems.
     * 
     * Returns a POJO with human-readable toString() containing:
     * - Memory state (heap, off-heap, device, workspace, arrays)
     * - Device state (count, current, routing, fallback)
     * - Execution state (DSP, profiling, backend, ops)
     * - Workspace state (count, memory, spill, learning)
     * - Profiling state (enabled, frequency, ops, timing)
     * - Lifecycle state (created, destroyed, live, tracking)
     * - Diagnostic state (enabled, issues, health)
     * 
     * @return Complete framework state summary
     */
    public FrameworkSummary summary() {
        return FrameworkSummary.builder()
            .timestampMs(System.currentTimeMillis())
            .memory(captureMemoryState())
            .device(captureDeviceState())
            .execution(captureExecutionState())
            .workspace(captureWorkspaceState())
            .profiling(captureProfilingState())
            .lifecycle(captureLifecycleState())
            .diagnostics(captureDiagnosticState())
            .constants(captureConstantCacheState())
            .build();
    }
    
    /**
     * Capture memory subsystem state
     */
    private MemoryState captureMemoryState() {
        Runtime runtime = Runtime.getRuntime();
        long heapUsed = runtime.totalMemory() - runtime.freeMemory();
        long heapMax = runtime.maxMemory();

        MemoryTracker tracker = MemoryTracker.getInstance();
        long offHeapUsed = tracker.getAllocatedHostAmount() + tracker.getCachedHostAmount();

        long deviceUsed = 0;
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int i = 0; i < numDevices; i++) {
                deviceUsed += tracker.getActiveMemory(i);
            }
        } catch (Exception e) {
            // Ignore
        }

        long workspaceUsed = 0;
        try {
            MemoryWorkspace ws = Nd4j.getMemoryManager().getCurrentWorkspace();
            if (ws != null) {
                workspaceUsed = ws.getCurrentSize();
            }
        } catch (Exception e) {
            // Ignore
        }

        long liveArrays = 0;
        try {
            liveArrays = DeallocatorService.getInstance().getReferenceMap().size();
        } catch (Exception e) {
            // Ignore
        }

        return MemoryState.builder()
            .heapUsedBytes(heapUsed)
            .heapMaxBytes(heapMax)
            .offHeapUsedBytes(offHeapUsed)
            .deviceUsedBytes(deviceUsed)
            .workspaceUsedBytes(workspaceUsed)
            .liveArrayCount(liveArrays)
            .gcFrequency(Nd4j.getMemoryManager().getFrequency())
            .periodicGcEnabled(Nd4j.getMemoryManager().isPeriodicGcActive())
            .noArrayGc(Boolean.getBoolean(ND4JSystemProperties.NO_ARRAY_GC))
            .memoryPressureThreshold(Nd4j.getEnvironment().heapPressureThreshold())
            .build();
    }
    
    /**
     * Capture device subsystem state
     */
    private org.nd4j.linalg.framework.device.DeviceState captureDeviceState() {
        org.nd4j.linalg.api.concurrency.AffinityManager affinity =
            org.nd4j.linalg.factory.Nd4j.getAffinityManager();

        int numDevices = affinity.getNumberOfDevices();
        int currentDevice = affinity.getDeviceForCurrentThread();

        org.nd4j.linalg.api.device.DeviceMemoryManager deviceMemMgr =
            org.nd4j.linalg.api.device.DeviceMemoryManager.getInstance();

        int defaultDevice = 0;
        try {
            defaultDevice = deviceMemMgr.getCurrentDeviceId();
        } catch (Exception e) {
            // Ignore
        }

        String routingPolicy = "UNKNOWN";
        boolean autoFallback = false;
        double pressureThreshold = 0.9;
        try {
            routingPolicy = deviceMemMgr.getDeviceRoutingPolicy().name();
            autoFallback = deviceMemMgr.isAutoFallbackEnabled();
            pressureThreshold = deviceMemMgr.getMemoryPressureThreshold();
        } catch (Exception e) {
            // Ignore
        }

        int cudaPoolSize = 0;
        try {
            cudaPoolSize = org.nd4j.linalg.factory.Nd4j.getEnvironment().cudaMemoryPoolSize();
        } catch (Exception e) {
            // Ignore
        }

        boolean hasGpu = false;
        try {
            for (int i = 0; i < numDevices; i++) {
                if (affinity.getDeviceType(i) == org.nd4j.linalg.api.device.DeviceType.GPU) {
                    hasGpu = true;
                    break;
                }
            }
        } catch (Exception e) {
            // Ignore
        }

        // Get transfer stats
        long totalTransfers = 0;
        long totalTransferBytes = 0;
        try {
            TransferSubsystem transfers = device.transfers();
            if (transfers != null && transfers.isEnabled()) {
                totalTransfers = transfers.getTotalTransferCount();
                totalTransferBytes = transfers.getTotalBytes();
            }
        } catch (Exception e) {
            // Ignore
        }

        // Get pinning stats
        int pinnedVariableCount = 0;
        try {
            DevicePinningManager pinning = device.pinning();
            if (pinning != null && pinning.isEnabled()) {
                pinnedVariableCount = pinning.getPinnedVariableCount();
            }
        } catch (Exception e) {
            // Ignore
        }

        // Count frozen buffers
        int frozenBufferCount = 0;
        try {
            // This would require iterating all arrays - skip for now
            // Could be added later if needed
        } catch (Exception e) {
            // Ignore
        }

        return org.nd4j.linalg.framework.device.DeviceState.builder()
            .numDevices(numDevices)
            .currentDeviceId(currentDevice)
            .defaultDeviceId(defaultDevice)
            .routingPolicy(routingPolicy)
            .autoFallbackEnabled(autoFallback)
            .memoryPressureThreshold(pressureThreshold)
            .cudaMemoryPoolSizeMb(cudaPoolSize)
            .hasGpu(hasGpu)
            .totalTransfers(totalTransfers)
            .totalTransferBytes(totalTransferBytes)
            .pinnedVariableCount(pinnedVariableCount)
            .frozenBufferCount(frozenBufferCount)
            .build();
    }
    
    /**
     * Capture execution subsystem state
     */
    private org.nd4j.linalg.framework.exec.ExecutionState captureExecutionState() {
        boolean dspEnabled = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED);
        
        String dspDiag = System.getProperty(
            org.nd4j.common.config.ND4JSystemProperties.DSP_DIAGNOSTICS, "none");
        
        String dspLevel = System.getProperty(
            org.nd4j.common.config.ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "summary");
        
        boolean opProfiling = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.PROFILING_ENABLE);
        
        String preferredBackend = System.getProperty(
            org.nd4j.common.config.ND4JSystemProperties.PREFERRED_BACKEND, "auto");
        
        long opsExecuted = 0;
        long totalExecTime = 0;
        String executionerType = "UNKNOWN";
        
        try {
            org.nd4j.linalg.api.ops.executioner.OpExecutioner exec = 
                org.nd4j.linalg.factory.Nd4j.getExecutioner();
            if (exec != null) {
                opsExecuted = exec.getOperationsExecuted();
                totalExecTime = exec.getTotalExecutionTime();
                executionerType = exec.getClass().getSimpleName();
            }
        } catch (Exception e) {
            // Ignore
        }
        
        return org.nd4j.linalg.framework.exec.ExecutionState.builder()
            .dspEnabled(dspEnabled)
            .dspDiagnostics(dspDiag)
            .dspDiagnosticsLevel(dspLevel)
            .opProfilingEnabled(opProfiling)
            .preferredBackend(preferredBackend)
            .operationsExecuted(opsExecuted)
            .totalExecutionTimeNanos(totalExecTime)
            .executionerType(executionerType)
            .build();
    }
    
    /**
     * Capture workspace subsystem state
     */
    private org.nd4j.linalg.framework.workspace.WorkspaceState captureWorkspaceState() {
        int numWorkspaces = 0;
        long totalWsBytes = 0;
        long totalSpilled = 0;
        
        try {
            // Try to get workspace info from manager
            org.nd4j.linalg.api.memory.MemoryWorkspaceManager wsMgr = 
                org.nd4j.linalg.factory.Nd4j.getWorkspaceManager();
            
            org.nd4j.linalg.api.memory.MemoryWorkspace ws = 
                org.nd4j.linalg.factory.Nd4j.getMemoryManager().getCurrentWorkspace();
            
            if (ws != null) {
                numWorkspaces = 1;
                totalWsBytes = ws.getCurrentSize();
                
                if (ws instanceof org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace) {
                    org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace nd4jWs = 
                        (org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace) ws;
                    totalSpilled = nd4jWs.getSpilledSize();
                }
            }
        } catch (Exception e) {
            // Ignore
        }
        
        long initialSize = Long.getLong(
            org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_INITIAL_SIZE, 
            1024 * 1024 * 1024);
        
        boolean learningEnabled = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_LEARNING_MODE);
        
        String debugMode = System.getProperty(
            org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_DEBUG_MODE, "DISABLED");
        
        return org.nd4j.linalg.framework.workspace.WorkspaceState.builder()
            .numWorkspaces(numWorkspaces)
            .totalWorkspaceBytes(totalWsBytes)
            .totalSpilledBytes(totalSpilled)
            .initialSizeBytes(initialSize)
            .learningEnabled(learningEnabled)
            .debugMode(debugMode)
            .build();
    }
    
    /**
     * Capture profiling subsystem state
     */
    private org.nd4j.linalg.framework.profiling.ProfilingState captureProfilingState() {
        boolean enabled = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.PROFILING_ENABLE);
        
        int frequency = Integer.getInteger(
            org.nd4j.common.config.ND4JSystemProperties.PROFILING_FREQUENCY, 1000);
        
        return org.nd4j.linalg.framework.profiling.ProfilingState.builder()
            .enabled(enabled)
            .frequency(frequency)
            .totalOps(0)
            .totalTimeNanos(0)
            .helperUsagePercent(0.0)
            .build();
    }
    
    /**
     * Capture lifecycle subsystem state
     */
    private org.nd4j.linalg.framework.lifecycle.LifecycleState captureLifecycleState() {
        long totalCreated = 0;
        long totalDestroyed = 0;
        long liveArrays = 0;
        
        try {
            org.nd4j.linalg.api.memory.deallocation.DeallocatorService ds =
                org.nd4j.linalg.api.memory.deallocation.DeallocatorService.getInstance();

            totalCreated = (long) ds.getAllocated().get("INDArray");
            totalDestroyed = (long) ds.getDeallocated().get("INDArray");
            liveArrays = ds.getReferenceMap().size();
        } catch (Exception e) {
            // Ignore
        }
        
        boolean trackingEnabled = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.LIFECYCLE_TRACKING_ENABLE);
        
        boolean stackTraceCapture = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.LIFECYCLE_STACK_TRACE_CAPTURE);
        
        long leakAgeThreshold = Long.getLong(
            org.nd4j.common.config.ND4JSystemProperties.LEAK_DETECTION_AGE_THRESHOLD, 
            5 * 60 * 1000);
        
        return org.nd4j.linalg.framework.lifecycle.LifecycleState.builder()
            .totalCreated(totalCreated)
            .totalDestroyed(totalDestroyed)
            .liveArrays(liveArrays)
            .trackingEnabled(trackingEnabled)
            .stackTraceCapture(stackTraceCapture)
            .leakDetectionAgeThresholdMs(leakAgeThreshold)
            .build();
    }
    
    /**
     * Capture diagnostic subsystem state
     */
    private DiagnosticState captureDiagnosticState() {
        boolean enabled = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.DIAGNOSTICS_ENABLE);
        
        String level = System.getProperty(
            org.nd4j.common.config.ND4JSystemProperties.DIAGNOSTICS_LEVEL, "WARNING");
        
        boolean healthMonitoring = Boolean.getBoolean(
            org.nd4j.common.config.ND4JSystemProperties.HEALTH_MONITORING_ENABLE);
        
        long healthCheckInterval = Long.getLong(
            org.nd4j.common.config.ND4JSystemProperties.HEALTH_CHECK_INTERVAL, 60000);
        
        // Get issue counts from diagnostics
        int issueCount = 0;
        int criticalIssues = 0;
        int errorIssues = 0;
        int warningIssues = 0;
        
        try {
            java.util.List<org.nd4j.linalg.framework.stats.DiagnosticIssue> issues = 
                diagnostics.getActiveIssues();
            
            issueCount = issues.size();
            for (org.nd4j.linalg.framework.stats.DiagnosticIssue issue : issues) {
                switch (issue.getSeverity()) {
                    case CRITICAL: criticalIssues++; break;
                    case ERROR: errorIssues++; break;
                    case WARNING: warningIssues++; break;
                    default: break;
                }
            }
        } catch (Exception e) {
            // Ignore
        }
        
        boolean healthy = criticalIssues == 0 && errorIssues == 0;
        
        return DiagnosticState.builder()
            .enabled(enabled)
            .level(level)
            .issueCount(issueCount)
            .criticalIssues(criticalIssues)
            .errorIssues(errorIssues)
            .warningIssues(warningIssues)
            .healthMonitoringEnabled(healthMonitoring)
            .healthCheckIntervalMs(healthCheckInterval)
            .healthy(healthy)
            .build();
    }
    
    /**
     * Capture constant cache subsystem state
     */
    private org.nd4j.linalg.framework.constant.ConstantCacheState captureConstantCacheState() {
        try {
            return constants.state();
        } catch (Exception e) {
            // Return default state on error
            return org.nd4j.linalg.framework.constant.ConstantCacheState.builder()
                .totalConstantBuffers(0)
                .totalShapeBuffers(0)
                .totalTadHelpers(0)
                .constantBufferCacheBytes(0)
                .shapeBufferCacheBytes(0)
                .tadHelperCacheBytes(0)
                .constantBufferHits(0)
                .constantBufferMisses(0)
                .hitPercentage(0.0)
                .cachingEnabled(false)
                .build();
        }
    }
    
    /**
     * Print framework status to log
     */
    public void printStatus() {
        FrameworkSnapshot snapshot = snapshot();
        log.info("=== ND4J Framework Status ===");
        log.info(snapshot.getSummary());
        
        if (snapshot.getDiagnosticIssues() != null && !snapshot.getDiagnosticIssues().isEmpty()) {
            log.warn("Active Issues:");
            for (DiagnosticIssue issue : snapshot.getDiagnosticIssues()) {
                log.warn("  - {}", issue.getSummary());
            }
        }
    }
    
    // =========================================================================
    // Legacy compatibility methods (for backward compatibility)
    // =========================================================================
    
    /**
     * @deprecated Use {@link #memory()} for new code
     */
    @Deprecated
    public MemorySubsystem getMemorySubsystem() {
        return memory();
    }
    
    /**
     * @deprecated Use {@link #execution()} for new code
     */
    @Deprecated
    public ExecutionSubsystem getExecutionSubsystem() {
        return execution();
    }

    /**
     * Framework health status.
     */
    public static class HealthStatus {
        private final boolean healthy;
        private final int issueCount;
        private final int criticalCount;
        private final int errorCount;
        private final int warningCount;

        private HealthStatus(Builder builder) {
            this.healthy = builder.healthy;
            this.issueCount = builder.issueCount;
            this.criticalCount = builder.criticalCount;
            this.errorCount = builder.errorCount;
            this.warningCount = builder.warningCount;
        }

        public boolean isHealthy() { return healthy; }
        public int getIssueCount() { return issueCount; }
        public int getCriticalCount() { return criticalCount; }
        public int getErrorCount() { return errorCount; }
        public int getWarningCount() { return warningCount; }

        public static Builder builder() { return new Builder(); }

        public static class Builder {
            private boolean healthy = true;
            private int issueCount = 0;
            private int criticalCount = 0;
            private int errorCount = 0;
            private int warningCount = 0;

            public Builder healthy(boolean healthy) { this.healthy = healthy; return this; }
            public Builder issueCount(int issueCount) { this.issueCount = issueCount; return this; }
            public Builder criticalCount(int criticalCount) { this.criticalCount = criticalCount; return this; }
            public Builder errorCount(int errorCount) { this.errorCount = errorCount; return this; }
            public Builder warningCount(int warningCount) { this.warningCount = warningCount; return this; }

            public HealthStatus build() { return new HealthStatus(this); }
        }

        @Override
        public String toString() {
            return String.format("HealthStatus{healthy=%s, issues=%d, critical=%d, errors=%d, warnings=%d}",
                healthy, issueCount, criticalCount, errorCount, warningCount);
        }
    }
}
