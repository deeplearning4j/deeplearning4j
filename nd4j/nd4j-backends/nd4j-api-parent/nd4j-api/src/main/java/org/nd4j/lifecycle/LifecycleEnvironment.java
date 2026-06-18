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

package org.nd4j.lifecycle;

import org.nd4j.common.config.ND4JSystemProperties;

/**
 * Lifecycle subsystem environment configuration.
 * 
 * @author ND4J Team
 */
public final class LifecycleEnvironment {
    
    private LifecycleEnvironment() {}
    
    /**
     * Check if array GC is disabled
     */
    public boolean isNoArrayGc() {
        return Boolean.getBoolean(ND4JSystemProperties.NO_ARRAY_GC);
    }
    
    /**
     * Enable/disable array GC
     */
    public void setNoArrayGc(boolean noGc) {
        System.setProperty(ND4JSystemProperties.NO_ARRAY_GC, String.valueOf(noGc));
    }
    
    /**
     * Get deallocator service GC thread count
     */
    public int getDeallocatorGcThreads() {
        return Integer.getInteger(ND4JSystemProperties.DEALLOCATOR_SERVICE_GC_THREADS, 1);
    }
    
    /**
     * Set deallocator service GC thread count
     */
    public void setDeallocatorGcThreads(int threads) {
        System.setProperty(ND4JSystemProperties.DEALLOCATOR_SERVICE_GC_THREADS, String.valueOf(threads));
    }
    
    /**
     * Check if lifecycle tracking is enabled
     */
    public boolean isLifecycleTrackingEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.LIFECYCLE_TRACKING_ENABLE);
    }
    
    /**
     * Enable/disable lifecycle tracking
     */
    public void setLifecycleTrackingEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.LIFECYCLE_TRACKING_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Check if stack trace capture is enabled for lifecycle
     */
    public boolean isLifecycleStackTraceCapture() {
        return Boolean.getBoolean(ND4JSystemProperties.LIFECYCLE_STACK_TRACE_CAPTURE);
    }
    
    /**
     * Enable/disable stack trace capture for lifecycle
     */
    public void setLifecycleStackTraceCapture(boolean enabled) {
        System.setProperty(ND4JSystemProperties.LIFECYCLE_STACK_TRACE_CAPTURE, String.valueOf(enabled));
    }
    
    /**
     * Get lifecycle event retention count
     */
    public int getLifecycleEventRetention() {
        return Integer.getInteger(ND4JSystemProperties.LIFECYCLE_EVENT_RETENTION, 10000);
    }
    
    /**
     * Set lifecycle event retention count
     */
    public void setLifecycleEventRetention(int count) {
        System.setProperty(ND4JSystemProperties.LIFECYCLE_EVENT_RETENTION, String.valueOf(count));
    }
    
    /**
     * Check if leak detection is enabled
     */
    public boolean isLeakDetectionEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.LEAK_DETECTION_ENABLE);
    }
    
    /**
     * Enable/disable leak detection
     */
    public void setLeakDetectionEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.LEAK_DETECTION_ENABLE, String.valueOf(enabled));
    }
    
    /**
     * Get leak detection age threshold (ms)
     */
    public long getLeakDetectionAgeThreshold() {
        return Long.getLong(ND4JSystemProperties.LEAK_DETECTION_AGE_THRESHOLD, 5 * 60 * 1000);
    }
    
    /**
     * Set leak detection age threshold (ms)
     */
    public void setLeakDetectionAgeThreshold(long thresholdMs) {
        System.setProperty(ND4JSystemProperties.LEAK_DETECTION_AGE_THRESHOLD, String.valueOf(thresholdMs));
    }
    
    /**
     * Get leak detection size threshold (bytes)
     */
    public long getLeakDetectionSizeThreshold() {
        return Long.getLong(ND4JSystemProperties.LEAK_DETECTION_SIZE_THRESHOLD, 10 * 1024 * 1024);
    }
    
    /**
     * Set leak detection size threshold (bytes)
     */
    public void setLeakDetectionSizeThreshold(long thresholdBytes) {
        System.setProperty(ND4JSystemProperties.LEAK_DETECTION_SIZE_THRESHOLD, String.valueOf(thresholdBytes));
    }
    
    /**
     * Get summary of lifecycle environment settings
     */
    public String getSummary() {
        return String.format(
            "Lifecycle Environment: NoArrayGC=%s, Deallocator threads=%d, " +
            "Tracking=%s, StackTrace=%s, Retention=%d, " +
            "Leak Detection=%s (age=%ds, size=%dMB)",
            isNoArrayGc() ? "yes" : "no",
            getDeallocatorGcThreads(),
            isLifecycleTrackingEnabled() ? "on" : "off",
            isLifecycleStackTraceCapture() ? "on" : "off",
            getLifecycleEventRetention(),
            isLeakDetectionEnabled() ? "on" : "off",
            getLeakDetectionAgeThreshold() / 1000,
            getLeakDetectionSizeThreshold() / (1024 * 1024)
        );
    }
}
