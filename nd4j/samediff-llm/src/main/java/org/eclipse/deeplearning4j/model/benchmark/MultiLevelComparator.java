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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Multi-level tensor comparison engine for benchmark-faithful correctness validation.
 *
 * <h3>Comparison Levels (most to least strict)</h3>
 * <ul>
 *   <li><b>TENSOR</b> — Full tensor equality with configurable abs/rel tolerance.
 *       Compares shape, dtype, then element-wise absolute and relative differences.
 *       Both must exceed tolerances for a mismatch to be reported.</li>
 *   <li><b>LOGITS</b> — Logits-only comparison at each decode step. Reports max abs diff,
 *       mean abs diff, argmax match, and top-K overlap per step.</li>
 *   <li><b>TOKEN_PREFIX</b> — First N sampled token IDs must match exactly. Fastest
 *       validation — suitable for benchmark smoke tests.</li>
 *   <li><b>TOKEN_TOPK</b> — Top-K token set overlap at each step. Relaxed validation
 *       for low-precision configs where exact match is unlikely.</li>
 *   <li><b>INTERMEDIATE</b> — Targeted intermediate tensor comparison for specific
 *       captured variables at specific decode steps. Use for subgraph-level validation
 *       (attention-prep, KV-prep, mask-reformat).</li>
 * </ul>
 *
 * <h3>How to choose</h3>
 * <ul>
 *   <li>Use <b>TOKEN_PREFIX</b> for quick benchmark validation — "does the replay produce
 *       the same first N tokens?"</li>
 *   <li>Use <b>LOGITS</b> for standard decode correctness — "do the per-step logits match
 *       before sampling?"</li>
 *   <li>Use <b>TENSOR</b> via {@link DspAccuracyValidator#validatePerOp} when you need
 *       to find the exact divergent op in the graph.</li>
 *   <li>Use <b>INTERMEDIATE</b> when you suspect a specific subgraph (e.g., attention-prep
 *       seg[2676-2730]) and want to validate it in isolation.</li>
 * </ul>
 */
@Slf4j
public class MultiLevelComparator {

    private final ValidationConfig config;

    public MultiLevelComparator(ValidationConfig config) {
        this.config = config;
    }

    // ─── Token Prefix Comparison ────────────────────────────────────────────

    /**
     * Compare first N generated tokens for exact match.
     *
     * @param referenceTokens reference token IDs (oracle: SLOT_BY_SLOT or manual baseline)
     * @param testTokens      test token IDs
     * @param prefixLength    number of tokens to compare
     * @return list of failure messages (empty if match)
     */
    public List<String> compareTokenPrefix(List<Integer> referenceTokens,
                                             int[] testTokens,
                                             int prefixLength) {
        List<String> failures = new ArrayList<>();
        int n = Math.min(prefixLength, Math.min(referenceTokens.size(), testTokens.length));

        for (int i = 0; i < n; i++) {
            int refTok = referenceTokens.get(i);
            int testTok = testTokens[i];
            if (refTok != testTok) {
                failures.add(String.format(
                        "Token mismatch at step %d: expected %d, got %d", i, refTok, testTok));
                break; // Report first divergence only
            }
        }

        if (referenceTokens.size() < prefixLength) {
            failures.add(String.format(
                    "Reference produced only %d tokens, need %d for prefix comparison",
                    referenceTokens.size(), prefixLength));
        }
        if (testTokens.length < prefixLength) {
            failures.add(String.format(
                    "Test produced only %d tokens, need %d for prefix comparison",
                    testTokens.length, prefixLength));
        }

        return failures;
    }

    /**
     * Compare first N generated tokens for exact match (both as lists).
     */
    public List<String> compareTokenPrefix(List<Integer> referenceTokens,
                                             List<Integer> testTokens,
                                             int prefixLength) {
        List<String> failures = new ArrayList<>();
        int n = Math.min(prefixLength, Math.min(referenceTokens.size(), testTokens.size()));

        for (int i = 0; i < n; i++) {
            int refTok = referenceTokens.get(i);
            int testTok = testTokens.get(i);
            if (refTok != testTok) {
                failures.add(String.format(
                        "Token mismatch at step %d: expected %d, got %d", i, refTok, testTok));
                break;
            }
        }

        if (referenceTokens.size() < prefixLength) {
            failures.add(String.format(
                    "Reference produced only %d tokens, need %d for prefix comparison",
                    referenceTokens.size(), prefixLength));
        }
        if (testTokens.size() < prefixLength) {
            failures.add(String.format(
                    "Test produced only %d tokens, need %d for prefix comparison",
                    testTokens.size(), prefixLength));
        }

        return failures;
    }

    // ─── Top-K Comparison ───────────────────────────────────────────────────

    /**
     * Compare top-K token overlap between two logit vectors.
     *
     * @return overlap count (0..K)
     */
    public int compareTopKOverlap(INDArray refLogits, INDArray testLogits, int k) {
        return DecodeStepValidator.computeTopKOverlap(
                flattenLogits(refLogits), flattenLogits(testLogits), k);
    }

    /**
     * Compare top-K overlap at each decode step.
     *
     * @return per-step overlap counts
     */
    public List<Integer> compareTopKPerStep(List<INDArray> refLogits,
                                              List<INDArray> testLogits, int k) {
        List<Integer> overlaps = new ArrayList<>();
        int steps = Math.min(refLogits.size(), testLogits.size());
        for (int i = 0; i < steps; i++) {
            overlaps.add(compareTopKOverlap(refLogits.get(i), testLogits.get(i), k));
        }
        return overlaps;
    }

    // ─── Logits Comparison ──────────────────────────────────────────────────

    /**
     * Compare two logit vectors. Returns max abs diff, mean abs diff, and argmax match.
     */
    public LogitsComparison compareLogits(INDArray refLogits, INDArray testLogits) {
        INDArray refFlat = flattenLogits(refLogits);
        INDArray testFlat = flattenLogits(testLogits);

        INDArray diff = org.nd4j.linalg.ops.transforms.Transforms.abs(refFlat.sub(testFlat));
        double maxAbs = diff.maxNumber().doubleValue();
        double meanAbs = diff.meanNumber().doubleValue();

        int refArgmax = Nd4j.argMax(refFlat).getInt(0);
        int testArgmax = Nd4j.argMax(testFlat).getInt(0);

        return new LogitsComparison(maxAbs, meanAbs, refArgmax, testArgmax, refArgmax == testArgmax);
    }

    // ─── Intermediate Capture Comparison ────────────────────────────────────

    /**
     * Compare captured intermediate tensors from two interceptors.
     *
     * <p>Compares variable-by-variable, step-by-step, using the configured tolerances.
     * Returns a divergence report with the first mismatching variable/step.</p>
     *
     * @param refInterceptor  capturing interceptor from reference run (SLOT_BY_SLOT)
     * @param testInterceptor capturing interceptor from test run
     * @param refLabel        label for reference (e.g., "SLOT_BY_SLOT")
     * @param testLabel       label for test (e.g., "TRITON_NO_GC")
     * @return divergence report
     */
    // ─── Tensor Comparison ──────────────────────────────────────────────────

    /**
     * Compare two tensors with configured tolerances.
     *
     * @return null if within tolerance, OpDivergence otherwise
     */
    public OpDivergence compareTensors(INDArray reference, INDArray test,
                                        String varName, String opName, int stepIdx) {
        double absTol = config.getAbsTolForOp(opName);
        return DspAccuracyValidator.compareArrays(
                reference, test, varName, opName, stepIdx,
                absTol, config.getDefaultRelTol(), null);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private INDArray flattenLogits(INDArray logits) {
        INDArray flat = logits.reshape(logits.length());
        if (logits.dataType() == DataType.HALF) {
            INDArray casted = flat.castTo(DataType.FLOAT);
            if (flat != logits && flat.closeable()) flat.close();
            return casted;
        }
        return flat;
    }

    // ─── Result Records ─────────────────────────────────────────────────────

    /**
     * Result of a logits comparison.
     */
    public static class LogitsComparison {
        private final double maxAbsDiff;
        private final double meanAbsDiff;
        private final int referenceArgmax;
        private final int testArgmax;
        private final boolean argmaxMatch;

        public LogitsComparison(double maxAbsDiff, double meanAbsDiff,
                                 int referenceArgmax, int testArgmax, boolean argmaxMatch) {
            this.maxAbsDiff = maxAbsDiff;
            this.meanAbsDiff = meanAbsDiff;
            this.referenceArgmax = referenceArgmax;
            this.testArgmax = testArgmax;
            this.argmaxMatch = argmaxMatch;
        }

        public double getMaxAbsDiff() { return maxAbsDiff; }
        public double getMeanAbsDiff() { return meanAbsDiff; }
        public int getReferenceArgmax() { return referenceArgmax; }
        public int getTestArgmax() { return testArgmax; }
        public boolean isArgmaxMatch() { return argmaxMatch; }

        @Override
        public String toString() {
            return String.format("LogitsComparison[maxAbs=%.6e, meanAbs=%.6e, argmax=%s]",
                    maxAbsDiff, meanAbsDiff, argmaxMatch ? "match" : "MISMATCH");
        }
    }
}
