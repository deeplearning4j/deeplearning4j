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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Caches compiled {@link ExecutionPlan} instances keyed by
 * (DAG topology, placeholder shape signature).
 *
 * <p>The DAG topology is captured by the set of requested output variable names
 * (same key as {@link DAGCache}).  The shape signature captures all placeholder
 * shapes and data types, so a plan compiled for shape [1, 128] is not reused
 * for shape [1, 256].</p>
 *
 * <p>This mirrors ONNX Runtime's {@code MemoryPatternGroup} cache in
 * {@code SessionState}, which caches execution frame layouts keyed by input shapes.</p>
 *
 * @see ExecutionPlan
 * @see ExecutionPlanCompiler
 */
@Slf4j
public class ExecutionPlanCache {

    private final Map<PlanCacheKey, ExecutionPlan> cache = new ConcurrentHashMap<>();
    private volatile boolean enabled = true;

    /**
     * Retrieve a cached plan or compute and cache a new one.
     *
     * @param requestedOutputs the requested output variables (DAG topology key)
     * @param placeholderArrays the placeholder arrays (shape signature key)
     * @param planSupplier supplier to compile the plan if not cached
     * @return the cached or newly compiled plan
     */
    public ExecutionPlan getOrCompute(Collection<String> requestedOutputs,
                                       Map<String, INDArray> placeholderArrays,
                                       java.util.function.Supplier<ExecutionPlan> planSupplier) {
        if (!enabled) {
            return planSupplier.get();
        }

        PlanCacheKey key = new PlanCacheKey(requestedOutputs, placeholderArrays);

        ExecutionPlan cached = cache.get(key);
        if (cached != null) {
            log.debug("Plan cache hit for outputs={}, shapeSignature={}", requestedOutputs, key.shapeSignature);
            return cached;
        }

        ExecutionPlan plan = planSupplier.get();
        cache.put(key, plan);
        log.debug("Compiled and cached new plan. Cache size: {}", cache.size());
        return plan;
    }

    public void clear() {
        cache.clear();
        log.debug("ExecutionPlan cache cleared");
    }

    public void invalidate(Collection<String> requestedOutputs) {
        cache.keySet().removeIf(k -> k.outputs.equals(new TreeSet<>(requestedOutputs)));
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        if (!enabled) {
            clear();
        }
    }

    public int size() {
        return cache.size();
    }

    @Data
    private static class PlanCacheKey {
        private final Set<String> outputs;
        private final long shapeSignature;

        PlanCacheKey(Collection<String> requestedOutputs, Map<String, INDArray> placeholderArrays) {
            this.outputs = Collections.unmodifiableSet(new TreeSet<>(requestedOutputs));
            this.shapeSignature = ExecutionPlanCompiler.computeShapeSignature(placeholderArrays);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            PlanCacheKey that = (PlanCacheKey) o;
            return shapeSignature == that.shapeSignature && outputs.equals(that.outputs);
        }

        @Override
        public int hashCode() {
            return 31 * outputs.hashCode() + Long.hashCode(shapeSignature);
        }
    }
}
