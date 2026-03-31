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

package org.nd4j.autodiff.samediff.execution;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Cache for compiled DynamicShapePlans, keyed by {output names, input shapes, device placement}.
 *
 * <p>Replaces the single-entry {@code currentPlan} field in DynamicShapePlanExecutor with
 * a proper cache that supports multiple plan configurations simultaneously. This is useful
 * for models that are called with different output sets (e.g., prefill vs decode, or
 * different subgraph extractions).</p>
 *
 * <p>Thread-safe: uses ConcurrentHashMap internally. Each entry includes a hit counter
 * for monitoring cache effectiveness.</p>
 */
@Slf4j
public class PlanCompileCache implements Closeable {

    /**
     * Cache key combining output names, input shapes, and device placement.
     * Two compilations with identical keys produce identical plans.
     */
    public static final class PlanCacheKey {
        private final long outputNamesHash;
        private final long inputShapesHash;
        private final int deviceId;

        public PlanCacheKey(long outputNamesHash, long inputShapesHash, int deviceId) {
            this.outputNamesHash = outputNamesHash;
            this.inputShapesHash = inputShapesHash;
            this.deviceId = deviceId;
        }

        /**
         * Create a cache key from output names and placeholder shapes.
         */
        public static PlanCacheKey of(List<String> outputNames,
                                       Map<String, INDArray> placeholders,
                                       int deviceId) {
            // Hash output names (sorted for consistency)
            List<String> sorted = new ArrayList<>(outputNames);
            Collections.sort(sorted);
            long nameHash = 0;
            for (String name : sorted) {
                nameHash = nameHash * 31 + name.hashCode();
            }

            // Hash input shapes (sorted by key for consistency)
            long shapeHash = 0;
            if (placeholders != null) {
                List<String> sortedKeys = new ArrayList<>(placeholders.keySet());
                Collections.sort(sortedKeys);
                for (String key : sortedKeys) {
                    INDArray arr = placeholders.get(key);
                    if (arr != null) {
                        shapeHash = shapeHash * 31 + key.hashCode();
                        shapeHash = shapeHash * 31 + Arrays.hashCode(arr.shape());
                        shapeHash = shapeHash * 31 + arr.dataType().ordinal();
                    }
                }
            }

            return new PlanCacheKey(nameHash, shapeHash, deviceId);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof PlanCacheKey)) return false;
            PlanCacheKey that = (PlanCacheKey) o;
            return outputNamesHash == that.outputNamesHash
                    && inputShapesHash == that.inputShapesHash
                    && deviceId == that.deviceId;
        }

        @Override
        public int hashCode() {
            long h = outputNamesHash;
            h = h * 31 + inputShapesHash;
            h = h * 31 + deviceId;
            return Long.hashCode(h);
        }

        @Override
        public String toString() {
            return String.format("PlanCacheKey{names=%x, shapes=%x, device=%d}",
                    outputNamesHash, inputShapesHash, deviceId);
        }
    }

    /**
     * Cached compiled plan entry with hit statistics.
     */
    static final class CompiledPlanEntry {
        final DynamicShapePlan plan;
        final long compiledAtNanos;
        long lastAccessNanos;
        int hitCount;

        CompiledPlanEntry(DynamicShapePlan plan) {
            this.plan = plan;
            this.compiledAtNanos = System.nanoTime();
            this.lastAccessNanos = this.compiledAtNanos;
            this.hitCount = 0;
        }
    }

    private final ConcurrentHashMap<PlanCacheKey, CompiledPlanEntry> cache = new ConcurrentHashMap<>();

    /**
     * Get a cached plan, or return null if not cached.
     * Increments hit counter on success.
     */
    public DynamicShapePlan get(PlanCacheKey key) {
        CompiledPlanEntry entry = cache.get(key);
        if (entry != null) {
            entry.hitCount++;
            entry.lastAccessNanos = System.nanoTime();
            return entry.plan;
        }
        return null;
    }

    /**
     * Store a compiled plan in the cache.
     */
    public void put(PlanCacheKey key, DynamicShapePlan plan) {
        cache.put(key, new CompiledPlanEntry(plan));
        log.debug("PlanCompileCache: cached plan for {}", key);
    }

    /**
     * Get a cached plan or compile a new one using the provided compiler.
     *
     * @param key      Cache key
     * @param compiler Function to compile a new plan if not cached
     * @return The cached or newly compiled plan
     */
    public DynamicShapePlan getOrCompile(PlanCacheKey key,
                                          java.util.function.Supplier<DynamicShapePlan> compiler) {
        CompiledPlanEntry entry = cache.get(key);
        if (entry != null) {
            entry.hitCount++;
            entry.lastAccessNanos = System.nanoTime();
            return entry.plan;
        }

        // Compile new plan
        DynamicShapePlan plan = compiler.get();
        if (plan != null) {
            put(key, plan);
        }
        return plan;
    }

    /**
     * Invalidate a specific cache entry.
     */
    public void invalidate(PlanCacheKey key) {
        cache.remove(key);
    }

    /**
     * Invalidate all cache entries.
     */
    public void invalidateAll() {
        cache.clear();
    }

    /**
     * Get the number of cached plans.
     */
    public int size() {
        return cache.size();
    }

    /**
     * Get cache statistics for diagnostics.
     *
     * @return Map of key → {hitCount, ageMs}
     */
    public Map<String, long[]> getStats() {
        Map<String, long[]> stats = new LinkedHashMap<>();
        long now = System.nanoTime();
        for (Map.Entry<PlanCacheKey, CompiledPlanEntry> e : cache.entrySet()) {
            long ageMs = (now - e.getValue().compiledAtNanos) / 1_000_000;
            stats.put(e.getKey().toString(), new long[]{e.getValue().hitCount, ageMs});
        }
        return stats;
    }

    @Override
    public void close() {
        invalidateAll();
    }
}
