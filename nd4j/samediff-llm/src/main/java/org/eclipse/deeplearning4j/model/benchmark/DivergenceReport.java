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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Getter;

/**
 * Complete report of all divergences found in a DSP validation run.
 *
 * <p>Contains ordered divergences (by step index), summary statistics,
 * and a human-readable report formatter. Created by {@link DspAccuracyValidator}.</p>
 */
@Getter
public class DivergenceReport {

    private final String referenceMode;
    private final String testMode;
    private final int totalOpsCompared;
    private final int opsMatched;
    private final List<OpDivergence> divergences;
    private final boolean passed;

    public DivergenceReport(String referenceMode, String testMode,
                            int totalOpsCompared, int opsMatched,
                            List<OpDivergence> divergences) {
        this.referenceMode = referenceMode;
        this.testMode = testMode;
        this.totalOpsCompared = totalOpsCompared;
        this.opsMatched = opsMatched;
        this.divergences = Collections.unmodifiableList(new ArrayList<>(divergences));
        this.passed = divergences.isEmpty();
    }

    /**
     * Returns the first divergence, or null if all matched.
     */
    public OpDivergence getFirstDivergence() {
        return divergences.isEmpty() ? null : divergences.get(0);
    }

    /**
     * Summarizes divergences by op type: how many divergences each op type caused.
     */
    public Map<String, Integer> getDivergencesByOpType() {
        Map<String, Integer> counts = new HashMap<>();
        for (OpDivergence d : divergences) {
            counts.merge(d.getOpName(), 1, Integer::sum);
        }
        return counts;
    }

    /**
     * Formats a human-readable validation report.
     */
    public String toReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== DSP Validation Report ===\n");
        sb.append("Reference: ").append(referenceMode).append("\n");
        sb.append("Test:      ").append(testMode).append("\n");
        sb.append("Result:    ").append(passed ? "PASSED" : "FAILED").append("\n");
        sb.append("Ops compared: ").append(totalOpsCompared)
          .append(", matched: ").append(opsMatched)
          .append(", diverged: ").append(divergences.size()).append("\n");

        if (!divergences.isEmpty()) {
            OpDivergence first = divergences.get(0);
            sb.append("\nFirst divergence at step ").append(first.getStepIdx())
              .append(" (").append(first.getOpName()).append(" -> ")
              .append(first.getVarName()).append(")\n");
            sb.append("  maxAbsDiff=").append(String.format("%.6e", first.getMaxAbsDiff()))
              .append(", meanAbsDiff=").append(String.format("%.6e", first.getMeanAbsDiff()))
              .append(", tolerance=").append(String.format("%.2e", first.getAbsTolUsed()))
              .append("\n");
            sb.append("  ref[").append(first.getMaxDiffIndex()).append("]=")
              .append(String.format("%.6e", first.getReferenceValueAtMax()))
              .append(", test[").append(first.getMaxDiffIndex()).append("]=")
              .append(String.format("%.6e", first.getTestValueAtMax())).append("\n");
            if (first.getInputVarNames() != null && first.getInputVarNames().length > 0) {
                sb.append("  inputs: ").append(String.join(", ", first.getInputVarNames())).append("\n");
            }
        }

        if (divergences.size() > 1) {
            sb.append("\nAll divergences:\n");
            for (OpDivergence d : divergences) {
                sb.append("  ").append(d).append("\n");
            }
        }

        if (!divergences.isEmpty()) {
            sb.append("\nDivergence count by op type:\n");
            Map<String, Integer> byOp = getDivergencesByOpType();
            byOp.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> sb.append(String.format("  %-30s %d\n", e.getKey(), e.getValue())));
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("DivergenceReport[%s vs %s: %s, %d/%d matched, %d diverged]",
                referenceMode, testMode, passed ? "PASSED" : "FAILED",
                opsMatched, totalOpsCompared, divergences.size());
    }
}
