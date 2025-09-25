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

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Cache for ForwardExecutionDAG instances to avoid expensive recomputation.
 * The cache is keyed by the set of requested output variables.
 * 
 * This significantly improves performance when the same outputs are requested
 * multiple times, avoiding the expensive convergence process in ForwardExecutionDAGBuilder.
 * 
 * @author Your Name
 */
@Slf4j
public class DAGCache {
    
    private final Map<CacheKey, ForwardExecutionDAG> cache = new ConcurrentHashMap<>();
    private volatile boolean enabled = true;
    
    /**
     * Retrieve a cached DAG or compute and cache a new one.
     * 
     * @param requestedOutputs The requested output variables
     * @param dagSupplier Supplier to compute the DAG if not cached
     * @return The cached or newly computed DAG
     */
    public ForwardExecutionDAG getOrCompute(Collection<String> requestedOutputs, 
                                           java.util.function.Supplier<ForwardExecutionDAG> dagSupplier) {
        if (!enabled) {
            return dagSupplier.get();
        }
        
        CacheKey key = new CacheKey(requestedOutputs);
        
        ForwardExecutionDAG cachedDag = cache.get(key);
        if (cachedDag != null) {
            log.debug("Using cached DAG for outputs: {}", requestedOutputs);
            return cachedDag;
        }
        
        // Compute the DAG
        log.debug("Computing new DAG for outputs: {}", requestedOutputs);
        ForwardExecutionDAG newDag = dagSupplier.get();
        
        // Store in cache
        cache.put(key, newDag);
        log.debug("Cached new DAG. Cache size: {}", cache.size());
        
        return newDag;
    }
    
    /**
     * Clear the entire cache.
     * Should be called when the graph structure changes.
     */
    public void clear() {
        cache.clear();
        log.debug("DAG cache cleared");
    }
    
    /**
     * Remove a specific entry from the cache.
     * 
     * @param requestedOutputs The outputs to remove from cache
     */
    public void invalidate(Collection<String> requestedOutputs) {
        CacheKey key = new CacheKey(requestedOutputs);
        cache.remove(key);
        log.debug("Invalidated cache entry for outputs: {}", requestedOutputs);
    }
    
    /**
     * Check if caching is enabled.
     */
    public boolean isEnabled() {
        return enabled;
    }
    
    /**
     * Enable or disable caching.
     * 
     * @param enabled True to enable caching, false to disable
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) {
            clear();
        }
    }
    
    /**
     * Get the current cache size.
     */
    public int size() {
        return cache.size();
    }
    
    /**
     * Cache key based on requested outputs.
     * Uses a sorted set for consistent hashing regardless of input order.
     */
    @Data
    private static class CacheKey {
        private final Set<String> outputs;
        
        CacheKey(Collection<String> requestedOutputs) {
            // Use TreeSet for consistent ordering
            this.outputs = Collections.unmodifiableSet(new TreeSet<>(requestedOutputs));
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            CacheKey cacheKey = (CacheKey) o;
            return outputs.equals(cacheKey.outputs);
        }
        
        @Override
        public int hashCode() {
            return outputs.hashCode();
        }
    }
}
