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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Compares autoregressive decode output step-by-step across two token generation runs.
 *
 * <p>Given two lists of per-step logits and sampled tokens (from two different
 * execution configurations), this validator reports:</p>
 * <ul>
 *   <li>Logits max/mean absolute difference at each step</li>
 *   <li>Whether the greedy argmax matches at each step</li>
 *   <li>Top-K token overlap at each step</li>
 *   <li>The first step where sampled tokens diverge</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <p>Callers collect logits from two decode runs (e.g., one with SLOT_BY_SLOT mode,
 * one with OPTIMAL), then pass them here for comparison:</p>
 * <pre>
 *   DecodeStepValidator validator = new DecodeStepValidator(ValidationConfig.standard());
 *   DecodeComparisonReport report = validator.compare(refLogits, refTokens, testLogits, testTokens,
 *       "SLOT_BY_SLOT", "OPTIMAL");
 *   System.out.println(report.toReport());
 * </pre>
 */
@Slf4j
public class DecodeStepValidator {

    private final ValidationConfig config;

    public DecodeStepValidator(ValidationConfig config) {
        this.config = config;
    }

    /**
     * Per-step comparison result.
     */
    @Getter
    public static class StepComparison {
        private final int stepIdx;
        private final int referenceTokenId;
        private final int testTokenId;
        private final boolean tokenMatch;
        private final double logitMaxAbsDiff;
        private final double logitMeanAbsDiff;
        private final boolean argmaxMatch;
        private final int topKOverlap;
        private final int topK;

        public StepComparison(int stepIdx, int referenceTokenId, int testTokenId,
                              double logitMaxAbsDiff, double logitMeanAbsDiff,
                              boolean argmaxMatch, int topKOverlap, int topK) {
            this.stepIdx = stepIdx;
            this.referenceTokenId = referenceTokenId;
            this.testTokenId = testTokenId;
            this.tokenMatch = (referenceTokenId == testTokenId);
            this.logitMaxAbsDiff = logitMaxAbsDiff;
            this.logitMeanAbsDiff = logitMeanAbsDiff;
            this.argmaxMatch = argmaxMatch;
            this.topKOverlap = topKOverlap;
            this.topK = topK;
        }

    }

    /**
     * Full decode comparison report.
     */
    @Getter
    public static class DecodeComparisonReport {
        private final String referenceMode;
        private final String testMode;
        private final int totalSteps;
        private final int tokenMatchCount;
        private final int argmaxMatchCount;
        private final int firstTokenDivergenceStep;
        private final int firstArgmaxDivergenceStep;
        private final List<StepComparison> steps;

        public DecodeComparisonReport(String referenceMode, String testMode,
                                       List<StepComparison> steps) {
            this.referenceMode = referenceMode;
            this.testMode = testMode;
            this.steps = steps;
            this.totalSteps = steps.size();

            int tokens = 0, argmax = 0;
            int firstToken = -1, firstArgmax = -1;
            for (int i = 0; i < steps.size(); i++) {
                StepComparison sc = steps.get(i);
                if (sc.tokenMatch) tokens++;
                if (sc.argmaxMatch) argmax++;
                if (!sc.tokenMatch && firstToken < 0) firstToken = i;
                if (!sc.argmaxMatch && firstArgmax < 0) firstArgmax = i;
            }
            this.tokenMatchCount = tokens;
            this.argmaxMatchCount = argmax;
            this.firstTokenDivergenceStep = firstToken;
            this.firstArgmaxDivergenceStep = firstArgmax;
        }


        public double getTokenMatchRate() {
            return totalSteps > 0 ? (double) tokenMatchCount / totalSteps : 0;
        }

        public double getArgmaxMatchRate() {
            return totalSteps > 0 ? (double) argmaxMatchCount / totalSteps : 0;
        }

        public String toReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== Decode Step Validation Report ===\n");
            sb.append("Reference: ").append(referenceMode).append("\n");
            sb.append("Test:      ").append(testMode).append("\n");
            sb.append("Steps:     ").append(totalSteps).append("\n");
            sb.append("Token matches:  ").append(tokenMatchCount).append("/").append(totalSteps)
              .append(String.format(" (%.1f%%)", getTokenMatchRate() * 100)).append("\n");
            sb.append("Argmax matches: ").append(argmaxMatchCount).append("/").append(totalSteps)
              .append(String.format(" (%.1f%%)", getArgmaxMatchRate() * 100)).append("\n");
            if (firstTokenDivergenceStep >= 0) {
                sb.append("First token divergence at step ").append(firstTokenDivergenceStep).append("\n");
            }
            if (firstArgmaxDivergenceStep >= 0) {
                sb.append("First argmax divergence at step ").append(firstArgmaxDivergenceStep).append("\n");
            }

            sb.append("\nPer-step details:\n");
            sb.append(String.format("%-6s %-8s %-8s %-7s %-7s %-12s %-12s %-8s\n",
                    "Step", "RefTok", "TestTok", "Match", "Argmax", "MaxAbsDiff", "MeanAbsDiff", "TopK"));
            sb.append("----------------------------------------------------------------------\n");
            for (StepComparison sc : steps) {
                sb.append(String.format("%-6d %-8d %-8d %-7s %-7s %-12.4e %-12.4e %d/%d\n",
                        sc.stepIdx, sc.referenceTokenId, sc.testTokenId,
                        sc.tokenMatch ? "Y" : "N",
                        sc.argmaxMatch ? "Y" : "N",
                        sc.logitMaxAbsDiff, sc.logitMeanAbsDiff,
                        sc.topKOverlap, sc.topK));
            }

            return sb.toString();
        }

        @Override
        public String toString() {
            return String.format("DecodeComparisonReport[%s vs %s: %d steps, " +
                            "token=%.1f%%, argmax=%.1f%%, firstDiv=%d]",
                    referenceMode, testMode, totalSteps,
                    getTokenMatchRate() * 100, getArgmaxMatchRate() * 100,
                    firstTokenDivergenceStep);
        }
    }

    /**
     * Compare two decode runs step-by-step.
     *
     * @param refLogits   per-step logits from reference run (each [1, vocabSize] or [vocabSize])
     * @param refTokens   per-step sampled token IDs from reference run
     * @param testLogits  per-step logits from test run
     * @param testTokens  per-step sampled token IDs from test run
     * @param refLabel    label for reference mode
     * @param testLabel   label for test mode
     * @return comparison report
     */
    public DecodeComparisonReport compare(List<INDArray> refLogits, List<Integer> refTokens,
                                           List<INDArray> testLogits, List<Integer> testTokens,
                                           String refLabel, String testLabel) {
        int steps = Math.min(refLogits.size(), testLogits.size());
        int topK = config.getTopKOverlap();
        List<StepComparison> comparisons = new ArrayList<>();

        for (int i = 0; i < steps; i++) {
            INDArray refL = refLogits.get(i);
            INDArray testL = testLogits.get(i);

            // Ensure float for comparison
            if (refL.dataType() == DataType.HALF) refL = refL.castTo(DataType.FLOAT);
            if (testL.dataType() == DataType.HALF) testL = testL.castTo(DataType.FLOAT);

            // Flatten to 1D
            INDArray refFlat = refL.reshape(refL.length());
            INDArray testFlat = testL.reshape(testL.length());

            // Absolute difference
            INDArray diff = org.nd4j.linalg.ops.transforms.Transforms.abs(refFlat.sub(testFlat));
            double maxAbs = diff.maxNumber().doubleValue();
            double meanAbs = diff.meanNumber().doubleValue();

            // Argmax comparison
            int refArgmax = Nd4j.argMax(refFlat).getInt(0);
            int testArgmax = Nd4j.argMax(testFlat).getInt(0);
            boolean argmaxMatch = (refArgmax == testArgmax);

            // Top-K overlap
            int overlap = computeTopKOverlap(refFlat, testFlat, topK);

            int refTok = i < refTokens.size() ? refTokens.get(i) : -1;
            int testTok = i < testTokens.size() ? testTokens.get(i) : -1;

            comparisons.add(new StepComparison(i, refTok, testTok,
                    maxAbs, meanAbs, argmaxMatch, overlap, topK));
        }

        return new DecodeComparisonReport(refLabel, testLabel, comparisons);
    }

    /**
     * Compute how many of the top-K token indices overlap between two logit vectors.
     */
    static int computeTopKOverlap(INDArray refLogits, INDArray testLogits, int k) {
        Set<Integer> refTopK = getTopKIndices(refLogits, k);
        Set<Integer> testTopK = getTopKIndices(testLogits, k);
        int overlap = 0;
        for (int idx : refTopK) {
            if (testTopK.contains(idx)) overlap++;
        }
        return overlap;
    }

    /**
     * Returns the indices of the top-K values in a 1D array.
     */
    static Set<Integer> getTopKIndices(INDArray logits, int k) {
        long len = logits.length();
        k = (int) Math.min(k, len);

        // Sort indices by value descending
        INDArray sorted = Nd4j.sortWithIndices(logits.dup(), 0, false)[0];
        Set<Integer> topK = new HashSet<>();
        for (int i = 0; i < k; i++) {
            topK.add(sorted.getInt(i));
        }
        return topK;
    }
}
