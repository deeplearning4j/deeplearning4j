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
import org.nd4j.linalg.api.memory.Deallocatable;
import org.nd4j.linalg.api.memory.deallocation.DeallocatorService;
import org.nd4j.linalg.api.memory.deallocation.DeallocatableReference;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Tracks INDArray lifecycle events for debugging and leak detection.
 * 
 * Records creation, destruction, and other lifecycle events
 * with optional stack trace capture.
 * 
 * @author ND4J Team
 */
@Slf4j
public class ArrayLifecycleTracker {
    
    /**
     * Maximum number of events to retain in history
     */
    private static final int DEFAULT_MAX_EVENTS = 10000;
    
    /**
     * Event history (thread-safe deque)
     */
    private final ConcurrentLinkedDeque<ArrayLifecycleEvent> eventHistory;
    
    /**
     * Maximum events to retain
     */
    private final AtomicLong maxEvents;
    
    /**
     * Flag indicating if tracking is enabled
     */
    private final AtomicBoolean trackingEnabled;
    
    /**
     * Flag indicating if stack traces should be captured
     */
    private final AtomicBoolean captureStackTraces;
    
    /**
     * Singleton instance
     */
    private static final ArrayLifecycleTracker INSTANCE = new ArrayLifecycleTracker();
    
    private ArrayLifecycleTracker() {
        this.eventHistory = new ConcurrentLinkedDeque<>();
        this.maxEvents = new AtomicLong(DEFAULT_MAX_EVENTS);
        this.trackingEnabled = new AtomicBoolean(false);
        this.captureStackTraces = new AtomicBoolean(false);
    }
    
    /**
     * Get singleton instance
     */
    public static ArrayLifecycleTracker getInstance() {
        return INSTANCE;
    }
    
    /**
     * Enable lifecycle tracking
     */
    public void enable() {
        trackingEnabled.set(true);
        log.info("Array lifecycle tracking enabled");
    }
    
    /**
     * Disable lifecycle tracking
     */
    public void disable() {
        trackingEnabled.set(false);
        log.info("Array lifecycle tracking disabled");
    }
    
    /**
     * Check if tracking is enabled
     */
    public boolean isEnabled() {
        return trackingEnabled.get();
    }
    
    /**
     * Enable stack trace capture (adds overhead)
     */
    public void enableStackTraceCapture() {
        captureStackTraces.set(true);
        log.info("Stack trace capture enabled for lifecycle tracking");
    }
    
    /**
     * Disable stack trace capture
     */
    public void disableStackTraceCapture() {
        captureStackTraces.set(false);
        log.info("Stack trace capture disabled for lifecycle tracking");
    }
    
    /**
     * Check if stack trace capture is enabled
     */
    public boolean isCaptureStackTraces() {
        return captureStackTraces.get();
    }
    
    /**
     * Set maximum number of events to retain
     */
    public void setMaxEvents(long maxEvents) {
        this.maxEvents.set(maxEvents);
        
        // Trim if needed
        while (eventHistory.size() > maxEvents) {
            eventHistory.pollFirst();
        }
    }
    
    /**
     * Record array creation event
     */
    public void recordCreation(INDArray array) {
        if (!trackingEnabled.get()) return;
        
        try {
            ArrayLifecycleEvent event = createEvent(
                ArrayLifecycleEvent.EventType.CREATED,
                array,
                null
            );
            addEvent(event);
        } catch (Exception e) {
            log.debug("Failed to record array creation", e);
        }
    }
    
    /**
     * Record array destruction event
     */
    public void recordDestruction(long arrayId, long sizeBytes) {
        if (!trackingEnabled.get()) return;
        
        try {
            ArrayLifecycleEvent event = ArrayLifecycleEvent.builder()
                .timestampMs(System.currentTimeMillis())
                .eventType(ArrayLifecycleEvent.EventType.DESTROYED)
                .arrayId(arrayId)
                .sizeBytes(sizeBytes)
                .threadName(Thread.currentThread().getName())
                .stackTrace(captureStackTraces.get() ? Thread.currentThread().getStackTrace() : null)
                .build();
            
            addEvent(event);
        } catch (Exception e) {
            log.debug("Failed to record array destruction", e);
        }
    }
    
    /**
     * Record workspace allocation event
     */
    public void recordWorkspaceAllocation(INDArray array, String workspaceId) {
        if (!trackingEnabled.get()) return;
        
        try {
            ArrayLifecycleEvent event = createEvent(
                ArrayLifecycleEvent.EventType.WORKSPACE_ALLOCATED,
                array,
                workspaceId
            );
            addEvent(event);
        } catch (Exception e) {
            log.debug("Failed to record workspace allocation", e);
        }
    }
    
    /**
     * Get recent lifecycle events (most recent first)
     */
    public List<ArrayLifecycleEvent> getRecentEvents(int numEvents) {
        List<ArrayLifecycleEvent> result = new ArrayList<>(Math.min(numEvents, eventHistory.size()));
        int count = 0;

        Iterator<ArrayLifecycleEvent> it = eventHistory.descendingIterator();
        while (it.hasNext() && count < numEvents) {
            result.add(it.next());
            count++;
        }

        return result;
    }

    /**
     * Get all events (most recent first)
     */
    public List<ArrayLifecycleEvent> getAllEvents() {
        List<ArrayLifecycleEvent> result = new ArrayList<>(eventHistory.size());
        Iterator<ArrayLifecycleEvent> it = eventHistory.descendingIterator();
        while (it.hasNext()) {
            result.add(it.next());
        }
        return result;
    }
    
    /**
     * Clear event history
     */
    public void clearHistory() {
        eventHistory.clear();
    }
    
    /**
     * Get current event count
     */
    public int getEventCount() {
        return eventHistory.size();
    }
    
    /**
     * Get statistics summary
     */
    public LifecycleStats getStats() {
        long created = 0;
        long destroyed = 0;
        long workspaceAllocated = 0;
        
        for (ArrayLifecycleEvent event : eventHistory) {
            switch (event.getEventType()) {
                case CREATED:
                    created++;
                    break;
                case DESTROYED:
                    destroyed++;
                    break;
                case WORKSPACE_ALLOCATED:
                    workspaceAllocated++;
                    break;
                default:
                    break;
            }
        }
        
        return LifecycleStats.builder()
            .totalCreated(created)
            .totalDestroyed(destroyed)
            .totalWorkspaceAllocated(workspaceAllocated)
            .currentLive(created - destroyed)
            .build();
    }
    
    /**
     * Create lifecycle event from INDArray
     */
    private ArrayLifecycleEvent createEvent(ArrayLifecycleEvent.EventType type, 
                                            INDArray array, String workspaceId) {
        ArrayLifecycleEvent.ArrayLifecycleEventBuilder builder = ArrayLifecycleEvent.builder()
            .timestampMs(System.currentTimeMillis())
            .eventType(type)
            .threadName(Thread.currentThread().getName())
            .workspaceId(workspaceId);
        
        if (array != null) {
            try {
                // Try to get array ID from deallocator service
                builder.arrayId(getArrayId(array));
                builder.sizeBytes(array.length() * array.dataType().width());
                builder.dataType(array.dataType().name());
                builder.shape(ShapeToString(array));
            } catch (Exception e) {
                log.debug("Failed to get array details for lifecycle event", e);
            }
        }
        
        if (captureStackTraces.get()) {
            builder.stackTrace(Thread.currentThread().getStackTrace());
        }
        
        return builder.build();
    }
    
    /**
     * Get array ID from DeallocatorService
     */
    private long getArrayId(INDArray array) {
        try {
            if (array instanceof Deallocatable) {
                Deallocatable deallocatable = (Deallocatable) array;
                return deallocatable.getUniqueId();
            }
        } catch (Exception e) {
            log.debug("Failed to get array ID", e);
        }
        return -1;
    }
    
    /**
     * Convert array shape to string
     */
    private String ShapeToString(INDArray array) {
        if (array == null || array.shapeInfo() == null) {
            return "";
        }
        
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.rank(); i++) {
            if (i > 0) sb.append(",");
            sb.append(array.size(i));
        }
        return sb.toString();
    }
    
    /**
     * Add event to history, trimming if needed
     */
    private void addEvent(ArrayLifecycleEvent event) {
        eventHistory.addLast(event);
        
        while (eventHistory.size() > maxEvents.get()) {
            eventHistory.pollFirst();
        }
    }
}
