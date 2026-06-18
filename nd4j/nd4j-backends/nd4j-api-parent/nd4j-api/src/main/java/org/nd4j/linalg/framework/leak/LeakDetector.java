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

package org.nd4j.linalg.framework.leak;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.framework.history.ArrayLifecycleTracker;
import org.nd4j.linalg.framework.history.LifecycleStats;
import org.nd4j.linalg.framework.history.MemorySampler;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Detects potential memory leaks in ND4J array allocations.
 * 
 * Analyzes:
 * - Memory growth trends
 * - Old objects that haven't been freed
 * - Large allocations
 * - Allocation patterns
 * 
 * @author ND4J Team
 */
@Slf4j
public class LeakDetector {
    
    /**
     * Default age threshold for leak detection (5 minutes)
     */
    private static final long DEFAULT_AGE_THRESHOLD_MS = 5 * 60 * 1000;
    
    /**
     * Default size threshold for large allocation detection (10 MB)
     */
    private static final long DEFAULT_SIZE_THRESHOLD_BYTES = 10 * 1024 * 1024;
    
    /**
     * Default memory growth rate threshold (1 MB/s)
     */
    private static final long DEFAULT_GROWTH_THRESHOLD_BYTES_PER_SEC = 1024 * 1024;
    
    /**
     * Singleton instance
     */
    private static final LeakDetector INSTANCE = new LeakDetector();
    
    /**
     * Age threshold for considering objects as potential leaks
     */
    private volatile long ageThresholdMs;
    
    /**
     * Size threshold for flagging large allocations
     */
    private volatile long sizeThresholdBytes;
    
    /**
     * Growth rate threshold for leak detection
     */
    private volatile long growthThresholdBytesPerSec;
    
    /**
     * Cache of previously reported leaks to avoid duplicates
     */
    private final ConcurrentHashMap<Long, Long> reportedLeaks;
    
    private LeakDetector() {
        this.ageThresholdMs = DEFAULT_AGE_THRESHOLD_MS;
        this.sizeThresholdBytes = DEFAULT_SIZE_THRESHOLD_BYTES;
        this.growthThresholdBytesPerSec = DEFAULT_GROWTH_THRESHOLD_BYTES_PER_SEC;
        this.reportedLeaks = new ConcurrentHashMap<>();
    }
    
    /**
     * Get singleton instance
     */
    public static LeakDetector getInstance() {
        return INSTANCE;
    }
    
    /**
     * Set age threshold for leak detection
     */
    public void setAgeThresholdMs(long thresholdMs) {
        this.ageThresholdMs = thresholdMs;
    }
    
    /**
     * Set size threshold for large allocation detection
     */
    public void setSizeThresholdBytes(long thresholdBytes) {
        this.sizeThresholdBytes = thresholdBytes;
    }
    
    /**
     * Set memory growth rate threshold
     */
    public void setGrowthThresholdBytesPerSec(long threshold) {
        this.growthThresholdBytesPerSec = threshold;
    }
    
    /**
     * Run comprehensive leak detection analysis
     */
    public LeakReport runDetection() {
        long startTime = System.currentTimeMillis();
        
        List<PotentialLeak> potentialLeaks = new ArrayList<>();
        long totalLeakBytes = 0;

        // Analyze live objects for age-based leaks
        analyzeLiveObjects(potentialLeaks);

        // Analyze replica leaks
        analyzeReplicaLeaks(potentialLeaks);

        // Calculate total leak bytes
        for (PotentialLeak leak : potentialLeaks) {
            totalLeakBytes += leak.getSizeBytes();
        }
        
        // Get memory growth trend
        long growthRate = MemorySampler.getInstance().getMemoryGrowthTrend(60);
        
        // Determine if leaks detected
        boolean leaksDetected = !potentialLeaks.isEmpty() || 
                                growthRate > growthThresholdBytesPerSec;
        
        long duration = System.currentTimeMillis() - startTime;
        
        LeakReport report = LeakReport.builder()
            .timestampMs(System.currentTimeMillis())
            .leaksDetected(leaksDetected)
            .potentialLeakCount(potentialLeaks.size())
            .potentialLeakBytes(totalLeakBytes)
            .potentialLeaks(potentialLeaks)
            .memoryGrowthRateBytesPerSec(growthRate)
            .analysisDurationMs(duration)
            .build();
        
        if (leaksDetected) {
            log.warn("Leak detection: {}", report.getSummary());
        }
        
        return report;
    }
    
    /**
     * Analyze live objects for potential leaks
     */
    private void analyzeLiveObjects(List<PotentialLeak> potentialLeaks) {
        try {
            DeallocatorService deallocatorService = DeallocatorService.getInstance();
            Map<Long, DeallocatableReference> referenceMap = deallocatorService.getReferenceMap();
            
            long currentTime = System.currentTimeMillis();
            
            for (Map.Entry<Long, DeallocatableReference> entry : referenceMap.entrySet()) {
                Long arrayId = entry.getKey();
                DeallocatableReference ref = entry.getValue();
                
                // Skip if already reported recently (avoid spam)
                if (reportedLeaks.containsKey(arrayId)) {
                    long lastReported = reportedLeaks.get(arrayId);
                    if (currentTime - lastReported < 60000) { // 1 minute cooldown
                        continue;
                    }
                }
                
                // Check for old objects
                long ageMs = currentTime - ref.getTimestamp();
                if (ageMs > ageThresholdMs) {
                    PotentialLeak leak = createLeakFromReference(
                        arrayId, ref, ageMs, PotentialLeak.LeakReason.OLD_AGE);
                    potentialLeaks.add(leak);
                    reportedLeaks.put(arrayId, currentTime);
                    continue;
                }
                
                // Check for large allocations
                if (ref.getBytes() > sizeThresholdBytes) {
                    PotentialLeak leak = createLeakFromReference(
                        arrayId, ref, ageMs, PotentialLeak.LeakReason.LARGE_SIZE);
                    potentialLeaks.add(leak);
                    reportedLeaks.put(arrayId, currentTime);
                }
            }
        } catch (Exception e) {
            log.warn("Error analyzing live objects for leaks", e);
        }
    }

    /**
     * Analyze replica leaks from multi-device replication
     */
    private void analyzeReplicaLeaks(List<PotentialLeak> potentialLeaks) {
        try {
            // Get replica leak detector from device subsystem
            org.nd4j.linalg.framework.device.ReplicaLeakDetector replicaDetector =
                org.nd4j.linalg.factory.Nd4j.framework.device().replicaLeaks();

            if (replicaDetector == null || !replicaDetector.isEnabled()) {
                return;
            }

            // Use same age threshold as regular leak detection
            List<org.nd4j.linalg.framework.device.ReplicaLeakDetector.PotentialReplicaLeak> replicaLeaks =
                replicaDetector.detectLeakedReplicas(ageThresholdMs);

            for (org.nd4j.linalg.framework.device.ReplicaLeakDetector.PotentialReplicaLeak replicaLeak : replicaLeaks) {
                PotentialLeak leak = PotentialLeak.builder()
                    .arrayId(replicaLeak.getArrayId())
                    .sizeBytes(replicaLeak.getSizeBytes())
                    .dataType("unknown")
                    .shape("unknown")
                    .ageMs(replicaLeak.getAgeMs())
                    .stackTrace(null)
                    .leakReason(PotentialLeak.LeakReason.LEAKED_REPLICA)
                    .confidence(0.8) // High confidence - replica should have been cleaned up
                    .build();

                potentialLeaks.add(leak);

                if (log.isDebugEnabled()) {
                    log.debug("Detected leaked replica: variable={}, devices={} -> {}, age={}ms, size={} bytes",
                        replicaLeak.getVariableName(),
                        replicaLeak.getSourceDevice(),
                        replicaLeak.getTargetDevice(),
                        replicaLeak.getAgeMs(),
                        replicaLeak.getSizeBytes());
                }
            }
        } catch (Exception e) {
            log.trace("Error analyzing replica leaks", e);
        }
    }

    /**
     * Create PotentialLeak from DeallocatableReference
     */
    private PotentialLeak createLeakFromReference(Long arrayId, 
                                                   DeallocatableReference ref,
                                                   long ageMs,
                                                   PotentialLeak.LeakReason reason) {
        // Try to get additional details from the actual object
        String dataType = "unknown";
        String shape = "unknown";
        StackTraceElement[] stackTrace = null;
        
        try {
            Object obj = ref.get();
            if (obj instanceof INDArray) {
                INDArray array = (INDArray) obj;
                dataType = array.dataType().name();
                shape = arrayShapeToString(array);
            }
        } catch (Exception e) {
            // Object may have been collected
        }
        
        // Calculate confidence based on reason and characteristics
        double confidence = calculateConfidence(reason, ageMs, ref.getBytes());
        
        return PotentialLeak.builder()
            .arrayId(arrayId)
            .sizeBytes(ref.getBytes())
            .dataType(dataType)
            .shape(shape)
            .ageMs(ageMs)
            .stackTrace(stackTrace)
            .leakReason(reason)
            .confidence(confidence)
            .build();
    }
    
    /**
     * Calculate confidence score for potential leak
     */
    private double calculateConfidence(PotentialLeak.LeakReason reason, 
                                       long ageMs, 
                                       long sizeBytes) {
        double confidence = 0.5; // Base confidence
        
        switch (reason) {
            case OLD_AGE:
                // Older = more confident
                if (ageMs > ageThresholdMs * 4) {
                    confidence += 0.4;
                } else if (ageMs > ageThresholdMs * 2) {
                    confidence += 0.2;
                }
                break;
                
            case LARGE_SIZE:
                // Larger = more confident
                if (sizeBytes > sizeThresholdBytes * 10) {
                    confidence += 0.4;
                } else if (sizeBytes > sizeThresholdBytes * 2) {
                    confidence += 0.2;
                }
                break;
                
            default:
                break;
        }
        
        return Math.min(confidence, 1.0);
    }
    
    /**
     * Convert array shape to string
     */
    private String arrayShapeToString(INDArray array) {
        if (array == null) return "unknown";
        
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.rank(); i++) {
            if (i > 0) sb.append(",");
            sb.append(array.size(i));
        }
        return sb.toString();
    }
    
    /**
     * Clear reported leaks cache
     */
    public void clearReportedLeaks() {
        reportedLeaks.clear();
    }
    
    /**
     * Get lifecycle statistics summary
     */
    public LifecycleStats getLifecycleStats() {
        return ArrayLifecycleTracker.getInstance().getStats();
    }
}
