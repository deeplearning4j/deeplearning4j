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

import java.util.Map;

/**
 * Structural analysis of a compiled {@link DynamicShapePlan}, produced by
 * {@link AutoTuner#analyze(DynamicShapePlan)} and consumed by
 * {@link AutoTuner#applyCostModel(GraphAnalysis, boolean)}.
 *
 * <p>Fields are package-accessible for direct use in the cost model without
 * getter overhead.  Use {@link #builder()} to construct instances.</p>
 */
public class GraphAnalysis {

    // ── Raw counts ───────────────────────────────────────────────────────────

    /** Total number of slots (ops) in the plan. */
    final int totalSlots;

    /** Number of compute-heavy ops (matmul, conv, etc.). */
    final int computeOps;

    /** Number of element-wise ops. */
    final int elementwiseOps;

    /** Number of reduction ops. */
    final int reductionOps;

    /** Number of ops whose output shape depends on input values. */
    final int valueDependentOps;

    /** Number of slots that can be captured into a graph segment. */
    final int capturableSlots;

    /** Total number of segments in the plan. */
    final int totalSegments;

    /** Number of segments that are flagged as capturable. */
    final int capturableSegments;

    /** Size of the largest capturable segment in slots. */
    final int maxSegmentSize;

    /** Estimated peak device-memory usage in bytes. */
    final long estimatedPeakMemoryBytes;

    // ── Derived fractions ────────────────────────────────────────────────────

    /** Fraction of slots that are compute ops. */
    final double computeOpFraction;

    /** Fraction of slots that are element-wise ops. */
    final double elementwiseOpFraction;

    /** Fraction of slots that are reduction ops. */
    final double reductionOpFraction;

    /** Fraction of slots whose output shape depends on input values. */
    final double valueDependentFraction;

    // ── Op histogram ─────────────────────────────────────────────────────────

    /** Histogram of op names → occurrence count (sorted by name). */
    final Map<String, Integer> opHistogram;

    // ── Construction ─────────────────────────────────────────────────────────

    private GraphAnalysis(Builder b) {
        this.totalSlots              = b.totalSlots;
        this.computeOps              = b.computeOps;
        this.elementwiseOps          = b.elementwiseOps;
        this.reductionOps            = b.reductionOps;
        this.valueDependentOps       = b.valueDependentOps;
        this.capturableSlots         = b.capturableSlots;
        this.totalSegments           = b.totalSegments;
        this.capturableSegments      = b.capturableSegments;
        this.maxSegmentSize          = b.maxSegmentSize;
        this.estimatedPeakMemoryBytes = b.estimatedPeakMemoryBytes;
        this.computeOpFraction       = b.computeOpFraction;
        this.elementwiseOpFraction   = b.elementwiseOpFraction;
        this.reductionOpFraction     = b.reductionOpFraction;
        this.valueDependentFraction  = b.valueDependentFraction;
        this.opHistogram             = b.opHistogram;
    }

    public static Builder builder() {
        return new Builder();
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    public int getTotalSlots()              { return totalSlots; }
    public int getComputeOps()              { return computeOps; }
    public int getElementwiseOps()          { return elementwiseOps; }
    public int getReductionOps()            { return reductionOps; }
    public int getValueDependentOps()       { return valueDependentOps; }
    public int getCapturableSlots()         { return capturableSlots; }
    public int getTotalSegments()           { return totalSegments; }
    public int getCapturableSegments()      { return capturableSegments; }
    public int getMaxSegmentSize()          { return maxSegmentSize; }
    public long getEstimatedPeakMemoryBytes() { return estimatedPeakMemoryBytes; }
    public double getComputeOpFraction()    { return computeOpFraction; }
    public double getElementwiseOpFraction() { return elementwiseOpFraction; }
    public double getReductionOpFraction()  { return reductionOpFraction; }
    public double getValueDependentFraction() { return valueDependentFraction; }
    public Map<String, Integer> getOpHistogram() { return opHistogram; }

    // ── Builder ──────────────────────────────────────────────────────────────

    public static final class Builder {
        private int totalSlots;
        private int computeOps;
        private int elementwiseOps;
        private int reductionOps;
        private int valueDependentOps;
        private int capturableSlots;
        private int totalSegments;
        private int capturableSegments;
        private int maxSegmentSize;
        private long estimatedPeakMemoryBytes;
        private double computeOpFraction;
        private double elementwiseOpFraction;
        private double reductionOpFraction;
        private double valueDependentFraction;
        private Map<String, Integer> opHistogram;

        public Builder totalSlots(int v)               { this.totalSlots = v; return this; }
        public Builder computeOps(int v)               { this.computeOps = v; return this; }
        public Builder elementwiseOps(int v)           { this.elementwiseOps = v; return this; }
        public Builder reductionOps(int v)             { this.reductionOps = v; return this; }
        public Builder valueDependentOps(int v)        { this.valueDependentOps = v; return this; }
        public Builder capturableSlots(int v)          { this.capturableSlots = v; return this; }
        public Builder totalSegments(int v)            { this.totalSegments = v; return this; }
        public Builder capturableSegments(int v)       { this.capturableSegments = v; return this; }
        public Builder maxSegmentSize(int v)           { this.maxSegmentSize = v; return this; }
        public Builder estimatedPeakMemoryBytes(long v){ this.estimatedPeakMemoryBytes = v; return this; }
        public Builder computeOpFraction(double v)     { this.computeOpFraction = v; return this; }
        public Builder elementwiseOpFraction(double v) { this.elementwiseOpFraction = v; return this; }
        public Builder reductionOpFraction(double v)   { this.reductionOpFraction = v; return this; }
        public Builder valueDependentFraction(double v){ this.valueDependentFraction = v; return this; }
        public Builder opHistogram(Map<String, Integer> v) { this.opHistogram = v; return this; }

        public GraphAnalysis build() {
            return new GraphAnalysis(this);
        }
    }
}
