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
import org.nd4j.allocator.impl.MemoryTracker;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.WorkspaceAllocationsTracker;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Collects time-series samples of memory usage.
 * 
 * Runs on a background thread at configurable intervals
 * to capture memory statistics for trend analysis and
 * leak detection.
 * 
 * @author ND4J Team
 */
@Slf4j
public class MemorySampler {
    
    /**
     * Maximum number of samples to retain in history
     */
    private static final int DEFAULT_MAX_SAMPLES = 1000;
    
    /**
     * Default sampling interval in milliseconds
     */
    private static final long DEFAULT_INTERVAL_MS = 1000;
    
    /**
     * Thread-safe deque for storing samples
     */
    private final ConcurrentLinkedDeque<MemorySample> samples;
    
    /**
     * Maximum number of samples to retain
     */
    private final AtomicInteger maxSamples;
    
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
    private static final MemorySampler INSTANCE = new MemorySampler();
    
    private MemorySampler() {
        this.samples = new ConcurrentLinkedDeque<>();
        this.maxSamples = new AtomicInteger(DEFAULT_MAX_SAMPLES);
        this.samplingActive = new AtomicBoolean(false);
        this.samplingIntervalMs = DEFAULT_INTERVAL_MS;
    }
    
    /**
     * Get singleton instance
     */
    public static MemorySampler getInstance() {
        return INSTANCE;
    }
    
    /**
     * Start background sampling thread
     */
    public void startSampling() {
        startSampling(DEFAULT_INTERVAL_MS);
    }
    
    /**
     * Start background sampling thread with custom interval
     * 
     * @param intervalMs Sampling interval in milliseconds
     */
    public void startSampling(long intervalMs) {
        if (samplingActive.getAndSet(true)) {
            log.debug("Memory sampling already active");
            return;
        }
        
        this.samplingIntervalMs = intervalMs;
        this.samplerThread = new Thread(this::samplingLoop, "ND4J-Memory-Sampler");
        this.samplerThread.setDaemon(true);
        this.samplerThread.start();
        
        log.info("Memory sampling started with interval {}ms", intervalMs);
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
        
        log.info("Memory sampling stopped");
    }
    
    /**
     * Check if sampling is active
     */
    public boolean isSamplingActive() {
        return samplingActive.get();
    }
    
    /**
     * Set maximum number of samples to retain
     */
    public void setMaxSamples(int maxSamples) {
        this.maxSamples.set(maxSamples);
        
        // Trim if needed
        while (samples.size() > maxSamples) {
            samples.pollFirst();
        }
    }
    
    /**
     * Get current maximum sample count
     */
    public int getMaxSamples() {
        return maxSamples.get();
    }
    
    /**
     * Manually record a sample (useful for on-demand sampling)
     */
    public void recordSample() {
        try {
            MemorySample sample = collectSample();
            addSample(sample);
        } catch (Exception e) {
            log.warn("Failed to record memory sample", e);
        }
    }
    
    /**
     * Get recent samples (most recent first)
     *
     * @param numSamples Number of samples to return
     */
    public List<MemorySample> getRecentSamples(int numSamples) {
        List<MemorySample> result = new ArrayList<>(Math.min(numSamples, samples.size()));
        int count = 0;

        // Iterate from most recent
        Iterator<MemorySample> it = samples.descendingIterator();
        while (it.hasNext() && count < numSamples) {
            result.add(it.next());
            count++;
        }

        return result;
    }

    /**
     * Get all samples (most recent first)
     */
    public List<MemorySample> getAllSamples() {
        List<MemorySample> result = new ArrayList<>(samples.size());
        Iterator<MemorySample> it = samples.descendingIterator();
        while (it.hasNext()) {
            result.add(it.next());
        }
        return result;
    }
    
    /**
     * Clear all samples
     */
    public void clearSamples() {
        samples.clear();
    }
    
    /**
     * Get current sample count
     */
    public int getSampleCount() {
        return samples.size();
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
                log.warn("Error during memory sampling", e);
            }
        }
    }
    
    /**
     * Collect current memory statistics into a sample
     */
    private MemorySample collectSample() {
        long timestampMs = System.currentTimeMillis();
        
        // JVM heap
        Runtime runtime = Runtime.getRuntime();
        long heapUsedBytes = runtime.totalMemory() - runtime.freeMemory();
        long heapMaxBytes = runtime.maxMemory();
        
        // Off-heap (from MemoryTracker)
        MemoryTracker tracker = MemoryTracker.getInstance();
        long offHeapUsedBytes = tracker.getAllocatedHostAmount() + tracker.getCachedHostAmount();
        
        // Device memory (sum across all devices)
        long deviceUsedBytes = 0;
        long poolReservedBytes = 0;
        
        try {
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int i = 0; i < numDevices; i++) {
                deviceUsedBytes += tracker.getActiveMemory(i);
                // Pool stats would come from native calls (implemented later)
            }
        } catch (Exception e) {
            log.debug("Failed to get device memory stats", e);
        }
        
        // Workspace memory
        long workspaceUsedBytes = 0;
        try {
            MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
            if (workspace != null) {
                workspaceUsedBytes = workspace.getCurrentSize();
            }
        } catch (Exception e) {
            log.debug("Failed to get workspace memory stats", e);
        }
        
        // Live array count
        long liveArrayCount = 0;
        try {
            liveArrayCount = DeallocatorService.getInstance().getReferenceMap().size();
        } catch (Exception e) {
            log.debug("Failed to get live array count", e);
        }
        
        return MemorySample.builder()
            .timestampMs(timestampMs)
            .heapUsedBytes(heapUsedBytes)
            .heapMaxBytes(heapMaxBytes)
            .offHeapUsedBytes(offHeapUsedBytes)
            .deviceUsedBytes(deviceUsedBytes)
            .workspaceUsedBytes(workspaceUsedBytes)
            .poolReservedBytes(poolReservedBytes)
            .liveArrayCount(liveArrayCount)
            .build();
    }
    
    /**
     * Add sample to history, trimming if needed
     */
    private void addSample(MemorySample sample) {
        samples.addLast(sample);
        
        // Trim oldest if over limit
        while (samples.size() > maxSamples.get()) {
            samples.pollFirst();
        }
    }
    
    /**
     * Get memory growth trend (bytes per second)
     * 
     * Positive values indicate memory growth (potential leak)
     * Negative values indicate memory shrinking
     * 
     * @param windowSeconds Time window for trend calculation
     * @return bytes per second growth rate
     */
    public long getMemoryGrowthTrend(int windowSeconds) {
        List<MemorySample> recent = getRecentSamples(windowSeconds);
        
        if (recent.size() < 2) {
            return 0;
        }
        
        MemorySample oldest = recent.get(recent.size() - 1);
        MemorySample newest = recent.get(0);
        
        long timeDeltaSeconds = (newest.getTimestampMs() - oldest.getTimestampMs()) / 1000;
        if (timeDeltaSeconds <= 0) {
            return 0;
        }
        
        long memoryDelta = newest.getTotalUsedBytes() - oldest.getTotalUsedBytes();
        return memoryDelta / timeDeltaSeconds;
    }
}
