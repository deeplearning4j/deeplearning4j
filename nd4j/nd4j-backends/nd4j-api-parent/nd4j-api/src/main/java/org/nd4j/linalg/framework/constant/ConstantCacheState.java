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

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Constant cache subsystem state snapshot.
 *
 * Tracks constant buffers, shape buffers, and TAD helpers,
 * including peak statistics for high-water-mark monitoring.
 *
 * @author ND4J Team
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ConstantCacheState {

    /**
     * Total constant buffers cached
     */
    private int totalConstantBuffers;

    /**
     * Total shape buffers cached (actual count from native layer)
     */
    private int totalShapeBuffers;

    /**
     * Total TAD helpers cached
     */
    private int totalTadHelpers;

    /**
     * Constant buffer cache size (bytes)
     */
    private long constantBufferCacheBytes;

    /**
     * Shape buffer cache size (bytes)
     */
    private long shapeBufferCacheBytes;

    /**
     * TAD helper cache size (bytes)
     */
    private long tadHelperCacheBytes;

    /**
     * Peak shape cache entries (high-water mark)
     */
    private long shapePeakCachedEntries;

    /**
     * Peak shape cache bytes (high-water mark)
     */
    private long shapePeakCachedBytes;

    /**
     * Peak TAD cache entries (high-water mark)
     */
    private long tadPeakCachedEntries;

    /**
     * Peak TAD cache bytes (high-water mark)
     */
    private long tadPeakCachedBytes;

    /**
     * Peak constant buffer cache bytes (high-water mark)
     */
    private long constantPeakCachedBytes;

    /**
     * Cache hits for constant buffers
     */
    private long constantBufferHits;

    /**
     * Cache misses for constant buffers
     */
    private long constantBufferMisses;

    /**
     * Cache hit percentage
     */
    private double hitPercentage;

    /**
     * Whether constant caching is enabled
     */
    private boolean cachingEnabled;

    /**
     * Get total cache size in bytes
     */
    public long getTotalCacheBytes() {
        return constantBufferCacheBytes + shapeBufferCacheBytes + tadHelperCacheBytes;
    }

    /**
     * Get total cache size in KB
     */
    public double getTotalCacheKb() {
        return getTotalCacheBytes() / 1024.0;
    }

    /**
     * Get total peak cache bytes across all subsystems
     */
    public long getTotalPeakCacheBytes() {
        return constantPeakCachedBytes + shapePeakCachedBytes + tadPeakCachedBytes;
    }

    /**
     * Get total peak cache in KB
     */
    public double getTotalPeakCacheKb() {
        return getTotalPeakCacheBytes() / 1024.0;
    }

    /**
     * Get total peak cache in MB
     */
    public double getTotalPeakCacheMb() {
        return getTotalPeakCacheBytes() / (1024.0 * 1024.0);
    }

    /**
     * Get constant buffer cache in KB
     */
    public double getConstantBufferCacheKb() {
        return constantBufferCacheBytes / 1024.0;
    }

    /**
     * Get shape buffer cache in KB
     */
    public double getShapeBufferCacheKb() {
        return shapeBufferCacheBytes / 1024.0;
    }

    /**
     * Get TAD helper cache in KB
     */
    public double getTadHelperCacheKb() {
        return tadHelperCacheBytes / 1024.0;
    }

    /**
     * Get cache efficiency (hits / total accesses)
     */
    public double getCacheEfficiency() {
        long total = constantBufferHits + constantBufferMisses;
        if (total <= 0) return 0.0;
        return (constantBufferHits * 100.0) / total;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "Buffers: %d, Shapes: %d, TADs: %d, Cache: %.1f KB (%.1f%% hits), Enabled: %s",
            totalConstantBuffers,
            totalShapeBuffers,
            totalTadHelpers,
            getTotalCacheKb(),
            getCacheEfficiency(),
            cachingEnabled ? "YES" : "NO"
        ));
        long peakBytes = getTotalPeakCacheBytes();
        if (peakBytes > 0) {
            sb.append(String.format(", Peak: %.1f KB", getTotalPeakCacheKb()));
        }
        return sb.toString();
    }
}
