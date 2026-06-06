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

package org.nd4j.linalg.factory.config;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Memory pool and allocator configuration: release thresholds, failover
 * headroom gates, and other memory-management tuning knobs.
 *
 * <p>Java-side mirror of the C++ {@code sd::config::MemoryConfig} class.
 * Accessible via {@code Nd4j.getEnvironment().memory()}.
 *
 * <p>Defaults match the C++ defaults:
 * <ul>
 *   <li>{@code poolReleaseThresholdPercent} = 75</li>
 *   <li>{@code nonPeerHeadroomPercent} = 50</li>
 * </ul>
 */
public final class MemoryConfig {

    private static final MemoryConfig INSTANCE = new MemoryConfig();

    /** Pool release threshold: percentage of total device memory the CUDA memory
     *  pool is allowed to keep reserved. Range: 1–100. Default: 75. */
    private final AtomicInteger poolReleaseThresholdPercent = new AtomicInteger(75);

    /** Non-peer failover headroom: minimum percentage of total GPU memory that
     *  must remain free AFTER a managed-memory allocation for it to use the fast
     *  cudaMallocManaged path on non-peer devices. Range: 1–100. Default: 50. */
    private final AtomicInteger nonPeerHeadroomPercent = new AtomicInteger(50);

    private MemoryConfig() {
        // Read optional system property overrides
        String poolStr = System.getenv("SD_POOL_RELEASE_THRESHOLD_PERCENT");
        if (poolStr == null) poolStr = System.getProperty("nd4j.memory.poolReleaseThresholdPercent");
        if (poolStr != null) {
            try { setPoolReleaseThresholdPercent(Integer.parseInt(poolStr)); } catch (NumberFormatException ignored) {}
        }

        String headStr = System.getenv("SD_NON_PEER_HEADROOM_PERCENT");
        if (headStr == null) headStr = System.getProperty("nd4j.memory.nonPeerHeadroomPercent");
        if (headStr != null) {
            try { setNonPeerHeadroomPercent(Integer.parseInt(headStr)); } catch (NumberFormatException ignored) {}
        }
    }

    /** Return the singleton instance. */
    public static MemoryConfig getInstance() {
        return INSTANCE;
    }

    // ── Pool release threshold ──────────────────────────────────────────────

    /**
     * Returns the pool release threshold as a percentage (1–100).
     * Default: 75.
     */
    public int poolReleaseThresholdPercent() {
        return poolReleaseThresholdPercent.get();
    }

    /**
     * Set the pool release threshold as a percentage.
     * Values below 1 are clamped to 1; values above 100 are clamped to 100.
     */
    public void setPoolReleaseThresholdPercent(int percent) {
        poolReleaseThresholdPercent.set(Math.max(1, Math.min(100, percent)));
    }

    // ── Non-peer failover headroom ──────────────────────────────────────────

    /**
     * Returns the non-peer failover headroom percentage (1–100).
     * Default: 50.
     */
    public int nonPeerHeadroomPercent() {
        return nonPeerHeadroomPercent.get();
    }

    /**
     * Set the non-peer failover headroom percentage.
     * Values below 1 are clamped to 1; values above 100 are clamped to 100.
     */
    public void setNonPeerHeadroomPercent(int percent) {
        nonPeerHeadroomPercent.set(Math.max(1, Math.min(100, percent)));
    }
}
