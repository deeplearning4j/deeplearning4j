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

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Cost-model-driven auto-tuner for DSP execution mode selection.
 *
 * <p>Analyzes a compiled {@link DynamicShapePlan}'s structural properties -- op count,
 * segment layout, op mix, estimated memory pressure -- and recommends the optimal
 * {@link GraphExecutionMode}. Recommendations are cached per shape key so repeated
 * compilations with the same shapes reuse prior decisions.</p>
 *
 * <h3>Cost model heuristics</h3>
 * <ul>
 *   <li><b>Small graphs</b> ({@code <=5} slots): SLOT_BY_SLOT avoids compile overhead</li>
 *   <li><b>Value-dependent shapes</b> (>50% slots): SLOT_BY_SLOT since graph capture cannot
 *       handle dynamic shapes</li>
 *   <li><b>Compute-heavy graphs</b> (high matmul/conv ratio): TRITON benefits from kernel fusion</li>
 *   <li><b>Memory-bound graphs</b> (high element-wise ratio): CUDA_GRAPHS benefits from
 *       reduced launch overhead</li>
 *   <li><b>Mixed graphs</b>: AUTO lets the C++ backend make per-segment decisions</li>
 * </ul>
 *
 * <h3>Profiling integration</h3>
 * <p>After warmup executions, {@link #recordTrialResult(String, GraphExecutionMode, long)}
 * feeds back actual timing data. The tuner overrides the cost model with empirical
 * measurements when available.</p>
 */
@Slf4j
public final class AutoTuner {

    private AutoTuner() {}

    // ─── Configuration thresholds ───────────────────────────────────────────

    /** Graphs with fewer slots than this use SLOT_BY_SLOT (compile overhead not worth it). */
    public static final int SMALL_GRAPH_THRESHOLD = 5;

    /** If more than this fraction of slots have value-dependent output shapes,
     *  graph capture is unlikely to help. */
    public static final double VALUE_DEPENDENT_THRESHOLD = 0.5;

    /** Minimum fraction of compute ops (matmul, conv) for the graph to be classified
     *  as compute-heavy, favoring TRITON for kernel fusion. */
    public static final double COMPUTE_HEAVY_THRESHOLD = 0.15;

    /** Minimum fraction of element-wise ops for the graph to be classified as
     *  memory-bound, favoring CUDA_GRAPHS for launch overhead reduction. */
    public static final double ELEMENTWISE_HEAVY_THRESHOLD = 0.60;

    /** Minimum number of capturable segments needed before CUDA_GRAPHS is preferred
     *  over SLOT_BY_SLOT for memory-bound graphs. A single large segment also qualifies
     *  if it exceeds {@link #LARGE_SEGMENT_THRESHOLD}. */
    public static final int MIN_CAPTURABLE_SEGMENTS = 2;

    /** A single capturable segment with at least this many slots qualifies for
     *  CUDA_GRAPHS even without multiple segments. */
    public static final int LARGE_SEGMENT_THRESHOLD = 10;

    // ─── Op classification sets ─────────────────────────────────────────────

    private static final Set<String> COMPUTE_OPS = Set.of(
            "matmul", "mmul", "batchedgemm", "conv2d", "conv1d", "conv3d",
            "depthwise_conv2d", "gemm", "dot", "tensordot"
    );

    private static final Set<String> ELEMENTWISE_OPS = Set.of(
            "add", "subtract", "multiply", "divide", "abs", "neg", "exp", "log",
            "tanh", "sigmoid", "relu", "softplus", "selu", "gelu", "swish",
            "square", "sqrt", "reciprocal", "pow", "sin", "cos", "identity",
            "cast", "rsub", "rdiv", "squaredsubtract", "minimum", "maximum",
            "assign", "copy"
    );

    private static final Set<String> REDUCTION_OPS = Set.of(
            "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod",
            "reduce_norm2", "reduce_norm1", "reduce_variance", "reduce_stdev",
            "sum", "mean", "max", "min", "prod", "norm2", "norm1"
    );

    // ─── Cache ──────────────────────────────────────────────────────────────

    /** Shape-key -> cached recommendation. Thread-safe. */
    private static final ConcurrentHashMap<String, GraphExecutionMode> recommendationCache =
            new ConcurrentHashMap<>();

    /** Shape-key -> mode -> best observed wall-clock nanos. */
    private static final ConcurrentHashMap<String, Map<GraphExecutionMode, Long>> trialResults =
            new ConcurrentHashMap<>();

    // ─── Public API ─────────────────────────────────────────────────────────

    /**
     * Analyze a compiled plan and recommend the best execution mode.
     *
     * @param plan    the compiled DSP plan (must have slots populated)
     * @param sd      the SameDiff instance (for variable metadata)
     * @param hasCuda true if a CUDA backend is available
     * @return the recommended GraphExecutionMode
     */
    public static GraphExecutionMode recommend(DynamicShapePlan plan, SameDiff sd, boolean hasCuda) {
        if (plan == null || plan.getSlots() == null || plan.getSlots().length == 0) {
            return GraphExecutionMode.SLOT_BY_SLOT;
        }

        GraphAnalysis analysis = analyze(plan);

        // Check trial results first -- empirical data overrides the cost model
        String shapeKey = computeShapeKey(plan);
        Map<GraphExecutionMode, Long> trials = trialResults.get(shapeKey);
        if (trials != null && trials.size() >= 2) {
            GraphExecutionMode best = null;
            long bestTime = Long.MAX_VALUE;
            for (Map.Entry<GraphExecutionMode, Long> entry : trials.entrySet()) {
                if (entry.getValue() < bestTime) {
                    bestTime = entry.getValue();
                    best = entry.getKey();
                }
            }
            if (best != null) {
                log.debug("AutoTuner: using empirical best mode {} ({}ns) for shape key {}",
                        best, bestTime, shapeKey);
                return best;
            }
        }

        // Check recommendation cache
        GraphExecutionMode cached = recommendationCache.get(shapeKey);
        if (cached != null) {
            return cached;
        }

        // Apply cost model
        GraphExecutionMode recommended = applyCostModel(analysis, hasCuda);

        recommendationCache.put(shapeKey, recommended);
        log.debug("AutoTuner: recommend {} for plan ({} slots, {}% compute, {}% elementwise, {}% val-dep)",
                recommended, analysis.totalSlots,
                String.format("%.0f", analysis.computeOpFraction * 100),
                String.format("%.0f", analysis.elementwiseOpFraction * 100),
                String.format("%.0f", analysis.valueDependentFraction * 100));
        return recommended;
    }

    /**
     * Record a profiling trial result for a given shape key and mode.
     * Call after warmup executions to feed empirical timing data back to the tuner.
     *
     * @param shapeKey the plan's shape key (from {@link #computeShapeKey(DynamicShapePlan)})
     * @param mode     the execution mode that was used
     * @param wallNanos wall-clock execution time in nanoseconds
     */
    public static void recordTrialResult(String shapeKey, GraphExecutionMode mode, long wallNanos) {
        trialResults.computeIfAbsent(shapeKey, k -> new ConcurrentHashMap<>())
                .merge(mode, wallNanos, Math::min);
        // Invalidate the cost-model cache so next recommend() uses empirical data
        recommendationCache.remove(shapeKey);
    }

    /**
     * Compute a stable shape key for a plan, suitable for caching recommendations.
     * Uses the plan's slot count, op name sequence hash, and external input count.
     */
    public static String computeShapeKey(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        if (slots == null || slots.length == 0) return "empty";

        long hash = 17;
        for (DynamicShapeSlot slot : slots) {
            hash = hash * 31 + slot.getOpName().hashCode();
        }
        return "plan_" + slots.length + "_ext" + plan.getExternalInputKeys().length +
                "_h" + Long.toHexString(hash);
    }

    /**
     * Analyze a plan's structural properties for cost-model decisions.
     */
    public static GraphAnalysis analyze(DynamicShapePlan plan) {
        DynamicShapeSlot[] slots = plan.getSlots();
        int totalSlots = slots.length;
        int computeOps = 0;
        int elementwiseOps = 0;
        int reductionOps = 0;
        int valueDependentOps = 0;
        int capturableSlots = 0;

        Map<String, Integer> opHistogram = new TreeMap<>();

        for (DynamicShapeSlot slot : slots) {
            String opName = slot.getOpName().toLowerCase();
            opHistogram.merge(opName, 1, Integer::sum);

            if (COMPUTE_OPS.contains(opName)) computeOps++;
            else if (ELEMENTWISE_OPS.contains(opName)) elementwiseOps++;
            else if (REDUCTION_OPS.contains(opName)) reductionOps++;

            if (slot.isOutputShapeDependsOnInputValues()) valueDependentOps++;
            if (!slot.isOutputShapeDependsOnInputValues()) capturableSlots++;
        }

        // Compute segments
        List<PlanIntrospection.SegmentInfo> segments = PlanIntrospection.getSegments(plan);
        int capturableSegments = 0;
        int maxSegmentSize = 0;
        for (PlanIntrospection.SegmentInfo seg : segments) {
            if (seg.isCapturable()) capturableSegments++;
            maxSegmentSize = Math.max(maxSegmentSize, seg.getSlotCount());
        }

        // Estimate memory
        long estimatedPeakMemory = ModelMemoryEstimator.estimatePeakMemory(plan, 0, totalSlots);

        return GraphAnalysis.builder()
                .totalSlots(totalSlots)
                .computeOps(computeOps)
                .elementwiseOps(elementwiseOps)
                .reductionOps(reductionOps)
                .valueDependentOps(valueDependentOps)
                .capturableSlots(capturableSlots)
                .totalSegments(segments.size())
                .capturableSegments(capturableSegments)
                .maxSegmentSize(maxSegmentSize)
                .estimatedPeakMemoryBytes(estimatedPeakMemory)
                .computeOpFraction(totalSlots > 0 ? (double) computeOps / totalSlots : 0)
                .elementwiseOpFraction(totalSlots > 0 ? (double) elementwiseOps / totalSlots : 0)
                .reductionOpFraction(totalSlots > 0 ? (double) reductionOps / totalSlots : 0)
                .valueDependentFraction(totalSlots > 0 ? (double) valueDependentOps / totalSlots : 0)
                .opHistogram(opHistogram)
                .build();
    }

    /**
     * Clear all cached recommendations and trial results.
     * Useful when the model structure changes or for testing.
     */
    public static void clearCache() {
        recommendationCache.clear();
        trialResults.clear();
    }

    /**
     * Get the current number of cached recommendations.
     */
    public static int getCacheSize() {
        return recommendationCache.size();
    }

    /**
     * Get the current number of shape keys with trial data.
     */
    public static int getTrialCount() {
        return trialResults.size();
    }

    // ─── Cost model ─────────────────────────────────────────────────────────

    static GraphExecutionMode applyCostModel(GraphAnalysis analysis, boolean hasCuda) {
        // Rule 1: tiny graphs -- compile overhead dominates
        if (analysis.totalSlots <= SMALL_GRAPH_THRESHOLD) {
            return GraphExecutionMode.SLOT_BY_SLOT;
        }

        // Rule 2: heavily value-dependent -- graph capture can't help
        if (analysis.valueDependentFraction > VALUE_DEPENDENT_THRESHOLD) {
            return GraphExecutionMode.SLOT_BY_SLOT;
        }

        // No GPU -- can't use CUDA_GRAPHS or TRITON effectively
        if (!hasCuda) {
            // On CPU, TRITON can still do code generation via triton-cpu,
            // but only for compute-heavy graphs
            if (analysis.computeOpFraction >= COMPUTE_HEAVY_THRESHOLD &&
                    analysis.totalSlots > 20) {
                return GraphExecutionMode.TRITON;
            }
            return GraphExecutionMode.AUTO;
        }

        // GPU path: choose based on op mix
        // Rule 3: compute-heavy → TRITON for kernel fusion
        if (analysis.computeOpFraction >= COMPUTE_HEAVY_THRESHOLD) {
            return GraphExecutionMode.TRITON;
        }

        // Rule 4: element-wise heavy with capturable segments → CUDA_GRAPHS
        // Either multiple capturable segments or one large capturable segment qualifies.
        if (analysis.elementwiseOpFraction >= ELEMENTWISE_HEAVY_THRESHOLD &&
                (analysis.capturableSegments >= MIN_CAPTURABLE_SEGMENTS ||
                 analysis.maxSegmentSize >= LARGE_SEGMENT_THRESHOLD)) {
            return GraphExecutionMode.CUDA_GRAPHS;
        }

        // Rule 5: mixed workload → AUTO lets C++ backend decide per-segment
        return GraphExecutionMode.AUTO;
    }

    // ─── Analysis data class ────────────────────────────────────────────────

    /**
     * Structural analysis of a compiled DSP plan. Produced by {@link #analyze(DynamicShapePlan)}
     * and consumed by the cost model.
     */
    @Data
    @Builder
    public static class GraphAnalysis {
        private final int totalSlots;
        private final int computeOps;
        private final int elementwiseOps;
        private final int reductionOps;
        private final int valueDependentOps;
        private final int capturableSlots;
        private final int totalSegments;
        private final int capturableSegments;
        private final int maxSegmentSize;
        private final long estimatedPeakMemoryBytes;
        private final double computeOpFraction;
        private final double elementwiseOpFraction;
        private final double reductionOpFraction;
        private final double valueDependentFraction;
        private final Map<String, Integer> opHistogram;
    }
}
