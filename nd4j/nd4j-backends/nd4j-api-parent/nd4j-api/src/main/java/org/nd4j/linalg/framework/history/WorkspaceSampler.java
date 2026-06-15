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

package org.nd4j.linalg.framework.history;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.WorkspaceAllocationsTracker;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Collects time-series samples of workspace usage.
 * 
 * @author ND4J Team
 */
@Slf4j
public class WorkspaceSampler {
    
    /**
     * Maximum samples per workspace
     */
    private static final int DEFAULT_MAX_SAMPLES_PER_WORKSPACE = 100;
    
    /**
     * Default sampling interval in milliseconds
     */
    private static final long DEFAULT_INTERVAL_MS = 1000;
    
    /**
     * Samples per workspace ID
     */
    private final ConcurrentHashMap<String, ConcurrentLinkedDeque<WorkspaceSample>> samplesByWorkspace;
    
    /**
     * Maximum samples per workspace
     */
    private final AtomicInteger maxSamplesPerWorkspace;
    
    /**
     * Flag indicating if sampling is active
     */
    private final AtomicBoolean samplingActive;
    
    /**
     * Background sampler thread
     */
    private volatile Thread samplerThread;
    
    /**
     * Sampling interval in milliseconds
     */
    private volatile long samplingIntervalMs;
    
    /**
     * Singleton instance
     */
    private static final WorkspaceSampler INSTANCE = new WorkspaceSampler();
    
    private WorkspaceSampler() {
        this.samplesByWorkspace = new ConcurrentHashMap<>();
        this.maxSamplesPerWorkspace = new AtomicInteger(DEFAULT_MAX_SAMPLES_PER_WORKSPACE);
        this.samplingActive = new AtomicBoolean(false);
        this.samplingIntervalMs = DEFAULT_INTERVAL_MS;
    }
    
    /**
     * Get singleton instance
     */
    public static WorkspaceSampler getInstance() {
        return INSTANCE;
    }
    
    /**
     * Start background sampling
     */
    public void startSampling() {
        startSampling(DEFAULT_INTERVAL_MS);
    }
    
    /**
     * Start background sampling with custom interval
     */
    public void startSampling(long intervalMs) {
        if (samplingActive.getAndSet(true)) {
            log.debug("Workspace sampling already active");
            return;
        }
        
        this.samplingIntervalMs = intervalMs;
        this.samplerThread = new Thread(this::samplingLoop, "ND4J-Workspace-Sampler");
        this.samplerThread.setDaemon(true);
        this.samplerThread.start();
        
        log.info("Workspace sampling started with interval {}ms", intervalMs);
    }
    
    /**
     * Stop background sampling
     */
    public void stopSampling() {
        if (!samplingActive.getAndSet(false)) {
            return;
        }
        
        if (samplerThread != null) {
            samplerThread.interrupt();
            try {
                samplerThread.join(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            samplerThread = null;
        }
        
        log.info("Workspace sampling stopped");
    }
    
    /**
     * Check if sampling is active
     */
    public boolean isSamplingActive() {
        return samplingActive.get();
    }
    
    /**
     * Set maximum samples per workspace
     */
    public void setMaxSamplesPerWorkspace(int maxSamples) {
        this.maxSamplesPerWorkspace.set(maxSamples);
    }
    
    /**
     * Manually record a sample for current workspace
     */
    public void recordSample() {
        try {
            MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
            if (workspace == null) {
                return;
            }
            
            WorkspaceSample sample = collectSample(workspace);
            addSample(sample);
        } catch (Exception e) {
            log.warn("Failed to record workspace sample", e);
        }
    }
    
    /**
     * Get recent samples for a workspace
     */
    public List<WorkspaceSample> getRecentSamples(String workspaceId, int numSamples) {
        ConcurrentLinkedDeque<WorkspaceSample> workspaceSamples = samplesByWorkspace.get(workspaceId);
        if (workspaceSamples == null) {
            return new ArrayList<>();
        }

        List<WorkspaceSample> result = new ArrayList<>(Math.min(numSamples, workspaceSamples.size()));
        int count = 0;

        Iterator<WorkspaceSample> it = workspaceSamples.descendingIterator();
        while (it.hasNext() && count < numSamples) {
            result.add(it.next());
            count++;
        }

        return result;
    }
    
    /**
     * Get all workspace IDs being tracked
     */
    public List<String> getAllWorkspaceIds() {
        return new ArrayList<>(samplesByWorkspace.keySet());
    }
    
    /**
     * Clear samples for a specific workspace
     */
    public void clearSamples(String workspaceId) {
        ConcurrentLinkedDeque<WorkspaceSample> workspaceSamples = samplesByWorkspace.remove(workspaceId);
        if (workspaceSamples != null) {
            workspaceSamples.clear();
        }
    }
    
    /**
     * Clear all samples
     */
    public void clearAllSamples() {
        for (ConcurrentLinkedDeque<WorkspaceSample> deque : samplesByWorkspace.values()) {
            deque.clear();
        }
        samplesByWorkspace.clear();
    }
    
    /**
     * Get current sample count for workspace
     */
    public int getSampleCount(String workspaceId) {
        ConcurrentLinkedDeque<WorkspaceSample> workspaceSamples = samplesByWorkspace.get(workspaceId);
        return workspaceSamples != null ? workspaceSamples.size() : 0;
    }
    
    /**
     * Main sampling loop
     */
    private void samplingLoop() {
        while (samplingActive.get() && !Thread.interrupted()) {
            try {
                recordSample();
                Thread.sleep(samplingIntervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                log.warn("Error during workspace sampling", e);
            }
        }
    }
    
    /**
     * Collect workspace statistics into a sample
     */
    private WorkspaceSample collectSample(MemoryWorkspace workspace) {
        long timestampMs = System.currentTimeMillis();
        
        long currentSizeBytes = 0;
        long maxSizeBytes = 0;
        long spilledBytes = 0;
        long pinnedBytes = 0;
        long allocationCount = 0;
        String workspaceId = "unknown";
        
        try {
            if (workspace instanceof Nd4jWorkspace) {
                Nd4jWorkspace nd4jWorkspace = (Nd4jWorkspace) workspace;
                currentSizeBytes = nd4jWorkspace.getPrimaryOffset();
                maxSizeBytes = nd4jWorkspace.getInitialSize();
                spilledBytes = nd4jWorkspace.getSpilledSize();
                pinnedBytes = nd4jWorkspace.getPinnedSize();
                allocationCount = nd4jWorkspace.getNumberOfExternalAllocations();
                workspaceId = nd4jWorkspace.getId() != null ? nd4jWorkspace.getId() : "unknown";
            }
        } catch (Exception e) {
            log.debug("Failed to collect workspace statistics", e);
        }
        
        return WorkspaceSample.builder()
            .timestampMs(timestampMs)
            .workspaceId(workspaceId)
            .currentSizeBytes(currentSizeBytes)
            .maxSizeBytes(maxSizeBytes)
            .spilledBytes(spilledBytes)
            .pinnedBytes(pinnedBytes)
            .allocationCount(allocationCount)
            .build();
    }
    
    /**
     * Add sample to workspace history
     */
    private void addSample(WorkspaceSample sample) {
        String workspaceId = sample.getWorkspaceId();
        
        ConcurrentLinkedDeque<WorkspaceSample> workspaceSamples = 
            samplesByWorkspace.computeIfAbsent(workspaceId, k -> new ConcurrentLinkedDeque<>());
        
        workspaceSamples.addLast(sample);
        
        // Trim if needed
        int maxSamples = maxSamplesPerWorkspace.get();
        while (workspaceSamples.size() > maxSamples) {
            workspaceSamples.pollFirst();
        }
    }
    
    /**
     * Get spill statistics across all workspaces
     */
    public SpillStats getSpillStats() {
        long totalSpilledBytes = 0;
        long totalAllocations = 0;
        int workspacesWithSpill = 0;
        
        for (ConcurrentLinkedDeque<WorkspaceSample> deque : samplesByWorkspace.values()) {
            if (!deque.isEmpty()) {
                WorkspaceSample latest = deque.peekLast();
                if (latest.getSpilledBytes() > 0) {
                    workspacesWithSpill++;
                    totalSpilledBytes += latest.getSpilledBytes();
                }
                totalAllocations += latest.getAllocationCount();
            }
        }
        
        return SpillStats.builder()
            .totalSpilledBytes(totalSpilledBytes)
            .totalAllocations(totalAllocations)
            .workspacesWithSpill(workspacesWithSpill)
            .build();
    }
}
