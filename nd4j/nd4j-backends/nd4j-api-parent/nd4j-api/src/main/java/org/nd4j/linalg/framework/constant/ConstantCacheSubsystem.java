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

package org.nd4j.linalg.framework.constant;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Constant cache subsystem - manages caching of constant buffers, shape buffers, and TAD helpers.
 *
 * Provides full access to native cache statistics including current sizes, peak high-water marks,
 * shutdown management, and cache maintenance.
 *
 * @author ND4J Team
 */
@Slf4j
public final class ConstantCacheSubsystem {

    private static final ConstantCacheSubsystem INSTANCE = new ConstantCacheSubsystem();

    /**
     * Get singleton instance
     */
    public static ConstantCacheSubsystem getInstance() {
        return INSTANCE;
    }

    /**
     * Default constructor
     */
    public ConstantCacheSubsystem() {
        // Singleton constructor
    }

    // ==================== Constant buffer accessors ====================

    /**
     * Get constant buffer for int array
     */
    public DataBuffer getConstantBuffer(int[] array, DataType dataType) {
        return constantHandler().getConstantBuffer(array, dataType);
    }

    /**
     * Get constant buffer for long array
     */
    public DataBuffer getConstantBuffer(long[] array, DataType dataType) {
        return constantHandler().getConstantBuffer(array, dataType);
    }

    /**
     * Get constant buffer for double array
     */
    public DataBuffer getConstantBuffer(double[] array, DataType dataType) {
        return constantHandler().getConstantBuffer(array, dataType);
    }

    /**
     * Get constant buffer for float array
     */
    public DataBuffer getConstantBuffer(float[] array, DataType dataType) {
        return constantHandler().getConstantBuffer(array, dataType);
    }

    /**
     * Get constant buffer for boolean array
     */
    public DataBuffer getConstantBuffer(boolean[] array, DataType dataType) {
        return constantHandler().getConstantBuffer(array, dataType);
    }

    /**
     * Relocate constant space for a data buffer
     */
    public DataBuffer relocateConstantSpace(DataBuffer buffer) {
        return constantHandler().relocateConstantSpace(buffer);
    }

    /**
     * Purge all cached constants
     */
    public void purge() {
        constantHandler().purgeConstants();
        log.info("Constant caches purged");
    }

    // ==================== Constant cache stats ====================

    /**
     * Get cached bytes from native ops for current device
     */
    public long getCachedBytes() {
        try {
            int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            return nativeOps().getConstantCacheBytes(deviceId);
        } catch (Exception e) {
            log.debug("Failed to get constant cache bytes", e);
            return 0L;
        }
    }

    /**
     * Get cached bytes in KB
     */
    public double getCachedKb() {
        return getCachedBytes() / 1024.0;
    }

    /**
     * Get cached bytes in MB
     */
    public double getCachedMb() {
        return getCachedBytes() / (1024.0 * 1024.0);
    }

    /**
     * Clear constant cache
     */
    public void clear() {
        try {
            nativeOps().clearConstantCache();
            log.info("Constant cache cleared");
        } catch (Exception e) {
            log.warn("Failed to clear constant cache", e);
        }
    }

    // ==================== Shape cache ====================

    /**
     * Get number of shape cache entries
     */
    public long getShapeCachedEntries() {
        try {
            return nativeOps().getShapeCachedEntries();
        } catch (Exception e) {
            log.debug("Failed to get shape cached entries", e);
            return 0L;
        }
    }

    /**
     * Get shape cache memory usage in bytes
     */
    public long getShapeCachedBytes() {
        try {
            return nativeOps().getShapeCachedBytes();
        } catch (Exception e) {
            log.debug("Failed to get shape cached bytes", e);
            return 0L;
        }
    }

    /**
     * Get peak number of shape cache entries (high-water mark)
     */
    public long getShapePeakCachedEntries() {
        try {
            return nativeOps().getShapePeakCachedEntries();
        } catch (Exception e) {
            log.debug("Failed to get shape peak cached entries", e);
            return 0L;
        }
    }

    /**
     * Get peak shape cache memory in bytes (high-water mark)
     */
    public long getShapePeakCachedBytes() {
        try {
            return nativeOps().getShapePeakCachedBytes();
        } catch (Exception e) {
            log.debug("Failed to get shape peak cached bytes", e);
            return 0L;
        }
    }

    /**
     * Clear the shape cache.
     * NOTE: Will return early without action if shape cache shutdown is in progress.
     */
    public void clearShapeCache() {
        try {
            nativeOps().clearShapeCache();
            log.info("Shape cache cleared");
        } catch (Exception e) {
            log.warn("Failed to clear shape cache", e);
        }
    }

    /**
     * Get a string representation of the shape cache tree for debugging.
     * @param maxDepth maximum tree depth to traverse
     * @param maxEntries maximum entries to include
     * @return formatted string of shape cache contents
     */
    public String getShapeCacheString(int maxDepth, int maxEntries) {
        try {
            return nativeOps().getShapeCacheString(maxDepth, maxEntries);
        } catch (Exception e) {
            log.debug("Failed to get shape cache string", e);
            return "";
        }
    }

    // ==================== TAD cache ====================

    /**
     * Get number of TAD cache entries
     */
    public long getTADCachedEntries() {
        try {
            return nativeOps().getTADCachedEntries();
        } catch (Exception e) {
            log.debug("Failed to get TAD cached entries", e);
            return 0L;
        }
    }

    /**
     * Get TAD cache memory usage in bytes
     */
    public long getTADCachedBytes() {
        try {
            return nativeOps().getTADCachedBytes();
        } catch (Exception e) {
            log.debug("Failed to get TAD cached bytes", e);
            return 0L;
        }
    }

    /**
     * Get peak number of TAD cache entries (high-water mark)
     */
    public long getTADPeakCachedEntries() {
        try {
            return nativeOps().getTADPeakCachedEntries();
        } catch (Exception e) {
            log.debug("Failed to get TAD peak cached entries", e);
            return 0L;
        }
    }

    /**
     * Get peak TAD cache memory in bytes (high-water mark)
     */
    public long getTADPeakCachedBytes() {
        try {
            return nativeOps().getTADPeakCachedBytes();
        } catch (Exception e) {
            log.debug("Failed to get TAD peak cached bytes", e);
            return 0L;
        }
    }

    /**
     * Clear the TAD cache.
     * NOTE: Will return early without action if TAD cache shutdown is in progress.
     */
    public void clearTADCache() {
        try {
            nativeOps().clearTADCache();
            log.info("TAD cache cleared");
        } catch (Exception e) {
            log.warn("Failed to clear TAD cache", e);
        }
    }

    /**
     * Get a string representation of the TAD cache tree for debugging.
     * @param maxDepth maximum tree depth to traverse
     * @param maxEntries maximum entries to include
     * @return formatted string of TAD cache contents
     */
    public String getTADCacheString(int maxDepth, int maxEntries) {
        try {
            return nativeOps().getTADCacheString(maxDepth, maxEntries);
        } catch (Exception e) {
            log.debug("Failed to get TAD cache string", e);
            return "";
        }
    }

    /**
     * @deprecated Use {@link #getTADCachedEntries()} instead
     */
    @Deprecated
    public long getTadCacheEntries() {
        return getTADCachedEntries();
    }

    /**
     * @deprecated Use {@link #getTADCachedBytes()} instead
     */
    @Deprecated
    public long getTadCacheBytes() {
        return getTADCachedBytes();
    }

    /**
     * @deprecated Use {@link #clearTADCache()} instead
     */
    @Deprecated
    public void clearTadCache() {
        clearTADCache();
    }

    // ==================== Shutdown management ====================

    /**
     * Set whether shape cache shutdown is in progress.
     * When true, clearShapeCache() becomes a no-op to avoid segfaults during shutdown.
     */
    public void setShapeCacheShutdownInProgress(boolean inProgress) {
        try {
            nativeOps().setShapeCacheShutdownInProgress(inProgress);
        } catch (Exception e) {
            log.debug("Failed to set shape cache shutdown flag", e);
        }
    }

    /**
     * Check if shape cache shutdown is in progress.
     */
    public boolean isShapeCacheShutdownInProgress() {
        try {
            return nativeOps().isShapeCacheShutdownInProgress();
        } catch (Exception e) {
            log.debug("Failed to check shape cache shutdown flag", e);
            return false;
        }
    }

    /**
     * Set whether TAD cache shutdown is in progress.
     * When true, clearTADCache() becomes a no-op to avoid segfaults during shutdown.
     */
    public void setTADCacheShutdownInProgress(boolean inProgress) {
        try {
            nativeOps().setTADCacheShutdownInProgress(inProgress);
        } catch (Exception e) {
            log.debug("Failed to set TAD cache shutdown flag", e);
        }
    }

    /**
     * Check if TAD cache shutdown is in progress.
     */
    public boolean isTADCacheShutdownInProgress() {
        try {
            return nativeOps().isTADCacheShutdownInProgress();
        } catch (Exception e) {
            log.debug("Failed to check TAD cache shutdown flag", e);
            return false;
        }
    }

    // ==================== Maintenance ====================

    /**
     * Run cache health check and cleanup across all cache subsystems.
     */
    public void checkAndCleanupCaches() {
        try {
            nativeOps().checkAndCleanupCaches();
        } catch (Exception e) {
            log.debug("Failed to check and cleanup caches", e);
        }
    }

    // ==================== State capture ====================

    /**
     * Capture constant cache state snapshot with full statistics from native layer.
     */
    public ConstantCacheState captureState() {
        long bufferBytes = 0;
        long shapeEntries = 0;
        long shapeBytes = 0;
        long shapePeakEntries = 0;
        long shapePeakBytes = 0;
        long tadEntries = 0;
        long tadBytes = 0;
        long tadPeakEntries = 0;
        long tadPeakBytes = 0;

        try {
            int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            bufferBytes = nativeOps().getConstantCacheBytes(deviceId);
        } catch (Exception e) {
            log.debug("Failed to get constant cache bytes", e);
        }

        try {
            shapeEntries = nativeOps().getShapeCachedEntries();
            shapeBytes = nativeOps().getShapeCachedBytes();
            shapePeakEntries = nativeOps().getShapePeakCachedEntries();
            shapePeakBytes = nativeOps().getShapePeakCachedBytes();
        } catch (Exception e) {
            log.debug("Failed to get shape cache stats", e);
        }

        try {
            tadEntries = nativeOps().getTADCachedEntries();
            tadBytes = nativeOps().getTADCachedBytes();
            tadPeakEntries = nativeOps().getTADPeakCachedEntries();
            tadPeakBytes = nativeOps().getTADPeakCachedBytes();
        } catch (Exception e) {
            log.debug("Failed to get TAD cache stats", e);
        }

        boolean enabled = bufferBytes > 0 || shapeBytes > 0 || tadBytes > 0;

        return ConstantCacheState.builder()
            .totalConstantBuffers(estimateBufferCount(bufferBytes))
            .totalShapeBuffers((int) Math.min(shapeEntries, Integer.MAX_VALUE))
            .totalTadHelpers((int) Math.min(tadEntries, Integer.MAX_VALUE))
            .constantBufferCacheBytes(bufferBytes)
            .shapeBufferCacheBytes(shapeBytes)
            .tadHelperCacheBytes(tadBytes)
            .shapePeakCachedEntries(shapePeakEntries)
            .shapePeakCachedBytes(shapePeakBytes)
            .tadPeakCachedEntries(tadPeakEntries)
            .tadPeakCachedBytes(tadPeakBytes)
            .constantPeakCachedBytes(0L)
            .constantBufferHits(0)
            .constantBufferMisses(0)
            .hitPercentage(0.0)
            .cachingEnabled(enabled)
            .build();
    }

    /**
     * Estimate buffer count from bytes (rough estimate when native count unavailable)
     */
    private int estimateBufferCount(long bytes) {
        // Assume average buffer size of ~256 bytes
        return bytes > 0 ? (int) (bytes / 256) : 0;
    }

    /**
     * Get state snapshot (alias for captureState)
     */
    public ConstantCacheState state() {
        return captureState();
    }

    /**
     * Get constant handler (API-neutral)
     */
    private ConstantHandler constantHandler() {
        return Nd4j.getConstantHandler();
    }

    /**
     * Get native ops
     */
    private NativeOps nativeOps() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps();
    }
}
