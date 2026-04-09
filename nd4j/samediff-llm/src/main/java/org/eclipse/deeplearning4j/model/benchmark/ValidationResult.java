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

package org.eclipse.deeplearning4j.model.benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Unified validation result for benchmark-faithful correctness validation.
 *
 * <p>Produced by {@link DecodeValidationFramework} after running a validation
 * comparison. Encapsulates all validation levels, replay metadata, and
 * actionable failure diagnostics in a single record.</p>
 *
 * <h3>Validation Levels (most to least strict)</h3>
 * <ul>
 *   <li><b>TENSOR</b> — Exact tensor equality or strict tolerance for ALL intermediates</li>
 *   <li><b>LOGITS</b> — Logits comparison at each decode step (max abs diff, mean abs diff)</li>
 *   <li><b>TOKEN_PREFIX</b> — First N generated tokens must match oracle exactly</li>
 *   <li><b>TOKEN_TOPK</b> — Top-K token overlap at each step must meet threshold</li>
 *   <li><b>INTERMEDIATE</b> — Targeted intermediate tensor capture/comparison at specific steps</li>
 * </ul>
 *
 * <h3>When to use each level</h3>
 * <ul>
 *   <li><b>TENSOR</b> — Root-cause analysis when you need to find the exact divergent op.
 *       Most expensive, highest diagnostic value. Use {@link DspAccuracyValidator#validatePerOp}.</li>
 *   <li><b>LOGITS</b> — Standard correctness validation for decode. Catches numerical drift
 *       before it cascades into token divergence. Use {@link DecodeStepValidator#compare}.</li>
 *   <li><b>TOKEN_PREFIX</b> — Quick correctness check for benchmark validation. If the first
 *       N tokens match, the replay path is correct. Fastest validation.</li>
 *   <li><b>TOKEN_TOPK</b> — Relaxed correctness for TF32/low-precision configs where exact
 *       token match is unlikely but top-K overlap indicates semantic equivalence.</li>
 *   <li><b>INTERMEDIATE</b> — Targeted validation of specific subgraphs (attention-prep,
 *       KV-prep, mask-reformat). Use {@link CapturedComparison} with specific variable names.</li>
 * </ul>
 *
 * <h3>Replay Metadata</h3>
 * <p>Each result includes replay metadata for introspection:</p>
 * <ul>
 *   <li>Phase progression (COMPILE → CAPTURE → REPLAY)</li>
 *   <li>Replay identity/signature hash stability</li>
 *   <li>Replay unit counts per segment</li>
 *   <li>Execution counts per segment (monotonicity check)</li>
 *   <li>Phase-violation reporting (any unexpected phase transitions)</li>
 * </ul>
 */
public class ValidationResult {

    private final String configName;
    private final String referenceMode;
    private final String testMode;
    private final ValidationLevel level;
    private final boolean passed;
    private final List<String> failures;
    private final ReplayMetadataTracker.ReplayMetadata replayMetadata;
    private final Map<String, Double> metrics;
    private final DivergenceReport divergenceReport;
    private final DecodeStepValidator.DecodeComparisonReport decodeReport;
    private final int firstDivergentStep;
    private final int totalStepsCompared;
    private final double maxDiffObserved;

    private ValidationResult(Builder builder) {
        this.configName = builder.configName;
        this.referenceMode = builder.referenceMode;
        this.testMode = builder.testMode;
        this.level = builder.level;
        this.passed = builder.passed;
        this.failures = Collections.unmodifiableList(new ArrayList<>(builder.failures));
        this.replayMetadata = builder.replayMetadata;
        this.metrics = Collections.unmodifiableMap(new LinkedHashMap<>(builder.metrics));
        this.divergenceReport = builder.divergenceReport;
        this.decodeReport = builder.decodeReport;
        this.firstDivergentStep = builder.firstDivergentStep;
        this.totalStepsCompared = builder.totalStepsCompared;
        this.maxDiffObserved = builder.maxDiffObserved;
    }

    public static Builder builder(String configName, String referenceMode, String testMode,
                                   ValidationLevel level) {
        return new Builder(configName, referenceMode, testMode, level);
    }

    // ─── Getters ─────────────────────────────────────────────────────────────

    public String getConfigName() { return configName; }
    public String getReferenceMode() { return referenceMode; }
    public String getTestMode() { return testMode; }
    public ValidationLevel getLevel() { return level; }
    public boolean isPassed() { return passed; }
    public List<String> getFailures() { return failures; }
    public ReplayMetadataTracker.ReplayMetadata getReplayMetadata() { return replayMetadata; }
    public Map<String, Double> getMetrics() { return metrics; }
    public DivergenceReport getDivergenceReport() { return divergenceReport; }
    public DecodeStepValidator.DecodeComparisonReport getDecodeReport() { return decodeReport; }
    public int getFirstDivergentStep() { return firstDivergentStep; }
    public int getTotalStepsCompared() { return totalStepsCompared; }
    public double getMaxDiffObserved() { return maxDiffObserved; }

    /**
     * Returns the first failure message, or null if validation passed.
     */
    public String getFirstFailure() {
        return failures.isEmpty() ? null : failures.get(0);
    }

    /**
     * Human-readable summary report.
     */
    public String toReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Validation Result ===\n");
        sb.append("Config:       ").append(configName).append("\n");
        sb.append("Reference:    ").append(referenceMode).append("\n");
        sb.append("Test:         ").append(testMode).append("\n");
        sb.append("Level:        ").append(level).append("\n");
        sb.append("Result:       ").append(passed ? "PASSED" : "FAILED").append("\n");

        if (!passed) {
            sb.append("\nFailures:\n");
            for (String f : failures) {
                sb.append("  - ").append(f).append("\n");
            }
        }

        if (firstDivergentStep >= 0) {
            sb.append("\nFirst divergent step: ").append(firstDivergentStep).append("\n");
        }
        sb.append("Steps compared: ").append(totalStepsCompared).append("\n");
        sb.append("Max diff observed: ").append(String.format("%.6e", maxDiffObserved)).append("\n");

        if (decodeReport != null) {
            sb.append("\n").append(decodeReport.toReport());
        }
        if (divergenceReport != null && !divergenceReport.isPassed()) {
            sb.append("\n").append(divergenceReport.toReport());
        }
        if (replayMetadata != null) {
            sb.append("\n").append(replayMetadata.toReport());
        }
        if (!metrics.isEmpty()) {
            sb.append("\nMetrics:\n");
            for (Map.Entry<String, Double> e : metrics.entrySet()) {
                sb.append("  ").append(e.getKey()).append(": ").append(e.getValue()).append("\n");
            }
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("ValidationResult[%s: %s vs %s @ %s → %s, firstDiv=%d, maxDiff=%.6e]",
                configName, referenceMode, testMode, level,
                passed ? "PASSED" : "FAILED", firstDivergentStep, maxDiffObserved);
    }

    // ─── Validation Level Enum ──────────────────────────────────────────────

    public enum ValidationLevel {
        /** Exact or tolerance-based tensor equality for ALL intermediates (most strict). */
        TENSOR,
        /** Logits comparison at each decode step. */
        LOGITS,
        /** First N generated tokens must match oracle exactly. */
        TOKEN_PREFIX,
        /** Top-K token overlap at each step. */
        TOKEN_TOPK,
        /** Targeted intermediate tensor capture/comparison at specific steps. */
        INTERMEDIATE
    }

    // ─── Builder ─────────────────────────────────────────────────────────────

    public static class Builder {
        private final String configName;
        private final String referenceMode;
        private final String testMode;
        private final ValidationLevel level;
        private boolean passed = true;
        private final List<String> failures = new ArrayList<>();
        private ReplayMetadataTracker.ReplayMetadata replayMetadata;
        private final Map<String, Double> metrics = new LinkedHashMap<>();
        private DivergenceReport divergenceReport;
        private DecodeStepValidator.DecodeComparisonReport decodeReport;
        private int firstDivergentStep = -1;
        private int totalStepsCompared = 0;
        private double maxDiffObserved = 0.0;

        Builder(String configName, String referenceMode, String testMode, ValidationLevel level) {
            this.configName = configName;
            this.referenceMode = referenceMode;
            this.testMode = testMode;
            this.level = level;
        }

        public Builder passed(boolean v) { this.passed = v; return this; }
        public Builder failure(String msg) { this.passed = false; this.failures.add(msg); return this; }
        public Builder failures(List<String> msgs) { this.failures.addAll(msgs); return this; }
        public Builder replayMetadata(ReplayMetadataTracker.ReplayMetadata m) { this.replayMetadata = m; return this; }
        public Builder metric(String name, double value) { this.metrics.put(name, value); return this; }
        public Builder divergenceReport(DivergenceReport r) { this.divergenceReport = r; return this; }
        public Builder decodeReport(DecodeStepValidator.DecodeComparisonReport r) { this.decodeReport = r; return this; }
        public Builder firstDivergentStep(int v) { this.firstDivergentStep = v; return this; }
        public Builder totalStepsCompared(int v) { this.totalStepsCompared = v; return this; }
        public Builder maxDiffObserved(double v) { this.maxDiffObserved = v; return this; }

        public ValidationResult build() {
            return new ValidationResult(this);
        }
    }
}
