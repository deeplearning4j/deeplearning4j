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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the prefill last-position logits optimisation.
 *
 * <p>This optimisation (ADR: TTFT gap analysis) makes generation prefill request
 * {@code lm_logits_last} (shape {@code [B, 1, vocab]}) instead of the full
 * {@code lm_logits} ({@code [B, S, vocab]}).  DSP then naturally prunes the graph
 * to only compute the vocab projection for the final hidden state, skipping the
 * full-S projection — a TTFT win proportional to prompt length * vocab size.</p>
 *
 * <p>Tests in this class use a tiny synthetic SameDiff graph that mimics the lm-head
 * section of {@link org.nd4j.ggml.architecture.LLaMAArchitecture}, so no GGUF file
 * download or real model is required.</p>
 *
 * <p>Run from platform-tests/:</p>
 * <pre>
 * /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *   -Dtest=TestLastPositionLogits* \
 *   2>&amp;1 | tee /tmp/lmhead-tests.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestLastPositionLogits extends BaseNd4jTestWithBackends {

    // ── Synthetic graph dimensions ──────────────────────────────────────────
    static final int HIDDEN   = 8;     // tiny hidden size
    static final int VOCAB    = 16;    // tiny vocabulary
    static final int SEQ      = 5;     // prefill sequence length
    static final int BATCH    = 1;

    // ─────────────────────────────────────────────────────────────────────────
    // Helper: build a tiny synthetic SameDiff graph that mirrors the lm-head
    // section produced by LLaMAArchitecture.buildGraph().
    //
    //   hidden [B, S, H]  →  lm_logits [B, S, V]   (full projection)
    //                      →  lm_logits_last [B, 1, V]  (last-pos slice + project)
    //
    // Both outputs share the SAME lm_head.weight variable, so their values must
    // agree at position S-1.
    // ─────────────────────────────────────────────────────────────────────────
    private SameDiff buildSyntheticLmHeadGraph(int batch, int seq, int hidden, int vocab, DataType dtype) {
        SameDiff sd = SameDiff.create();

        // Placeholder for hidden states [B, S, H]
        SDVariable hiddenVar = sd.placeHolder("hidden_states", dtype, batch, seq, hidden);

        // Fixed lm_head weight [V, H] (stored transposed; matmul will permute)
        INDArray weightData = Nd4j.randn(dtype, vocab, hidden).muli(0.1);
        SDVariable lmHeadW = sd.var("lm_head.weight", weightData);

        // Full logits: hidden @ lm_head^T  →  [B, S, V]
        SDVariable lmHeadT = lmHeadW.permute(1, 0);  // [H, V]
        SDVariable logits = sd.mmul("lm_logits", hiddenVar, lmHeadT);

        // Last-position logits (the optimisation):
        // slice hidden at seq-dim index S-1, then project.
        // begin and size are DYNAMIC (sd.sizeAt) — NOT baked Java constants.
        SDVariable batchSize = sd.sizeAt(hiddenVar, 0);
        SDVariable seqLen    = sd.sizeAt(hiddenVar, 1);
        SDVariable hiddenDim = sd.sizeAt(hiddenVar, 2);
        SDVariable one       = sd.constant(Nd4j.scalar(DataType.INT64, 1L));
        SDVariable seqMinus1 = seqLen.sub(one);
        SDVariable zero      = sd.constant(Nd4j.scalar(DataType.INT64, 0L));
        SDVariable beginVec  = sd.stack("lm_last_begin", 0, zero, seqMinus1, zero);
        SDVariable sizeVec   = sd.stack("lm_last_size",  0, batchSize, one, hiddenDim);
        SDVariable hiddenLast = sd.slice("hidden_last", hiddenVar, beginVec, sizeVec);
        sd.mmul("lm_logits_last", hiddenLast, lmHeadT);

        sd.setOutputs(Arrays.asList("lm_logits", "lm_logits_last"));
        return sd;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1 (parameterized): lm_logits_last == last row of lm_logits
    // ─────────────────────────────────────────────────────────────────────────
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLastPositionLogitsMatchesFullLogitsLastRow(Nd4jBackend backend) {
        log.info("testLastPositionLogitsMatchesFullLogitsLastRow backend={}", backend.getClass().getSimpleName());
        SameDiff sd = buildSyntheticLmHeadGraph(BATCH, SEQ, HIDDEN, VOCAB, DataType.FLOAT);

        INDArray hiddenData = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HIDDEN);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("hidden_states", hiddenData);

        Map<String, INDArray> outputs = sd.output(inputs, "lm_logits", "lm_logits_last");

        INDArray fullLogits = outputs.get("lm_logits");     // [B, S, V]
        INDArray lastLogits = outputs.get("lm_logits_last"); // [B, 1, V]

        assertNotNull(fullLogits,  "lm_logits must be produced");
        assertNotNull(lastLogits,  "lm_logits_last must be produced");

        // Shape checks
        assertArrayEquals(new long[]{BATCH, SEQ, VOCAB},  fullLogits.shape(),
                "Full logits shape mismatch");
        assertArrayEquals(new long[]{BATCH, 1, VOCAB}, lastLogits.shape(),
                "Last-pos logits shape mismatch: expected [B,1,V]");

        // Value check: lastLogits[0,0,:] must equal fullLogits[0, S-1, :]
        INDArray expectedLastRow = fullLogits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(SEQ - 1),
                NDArrayIndex.all()).dup();          // [V]
        INDArray actualLastRow  = lastLogits.get(
                NDArrayIndex.point(0),
                NDArrayIndex.point(0),
                NDArrayIndex.all()).dup();           // [V]

        double maxDiff = expectedLastRow.sub(actualLastRow).amaxNumber().doubleValue();
        log.info("Max difference between full-logits last row and lm_logits_last: {}", maxDiff);
        assertEquals(0.0, maxDiff, 1e-5,
                "lm_logits_last must equal the last row of lm_logits");

        hiddenData.close();
        expectedLastRow.close();
        actualLastRow.close();
        for (INDArray v : outputs.values()) { if (v != null) v.close(); }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2 (parameterized): lm_logits_last is independent of rows 0..S-2
    //   Even if we randomise the first rows of hidden, lm_logits_last stays
    //   equal to the last-row projection only.
    // ─────────────────────────────────────────────────────────────────────────
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLastPositionLogitsOnlyDependsOnLastHiddenRow(Nd4jBackend backend) {
        log.info("testLastPositionLogitsOnlyDependsOnLastHiddenRow backend={}", backend.getClass().getSimpleName());
        SameDiff sd = buildSyntheticLmHeadGraph(BATCH, SEQ, HIDDEN, VOCAB, DataType.FLOAT);

        // Two hidden tensors that share the same last row
        INDArray hiddenA = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HIDDEN);
        INDArray hiddenB = hiddenA.dup();
        // Scramble rows 0..SEQ-2 of B — last row is identical
        INDArray scrambleRows = Nd4j.randn(DataType.FLOAT, BATCH, SEQ - 1, HIDDEN);
        hiddenB.get(NDArrayIndex.all(), NDArrayIndex.interval(0, SEQ - 1), NDArrayIndex.all())
               .assign(scrambleRows);

        Map<String, INDArray> inputsA = new HashMap<>();
        inputsA.put("hidden_states", hiddenA);
        Map<String, INDArray> inputsB = new HashMap<>();
        inputsB.put("hidden_states", hiddenB);

        Map<String, INDArray> outA = sd.output(inputsA, "lm_logits_last");
        Map<String, INDArray> outB = sd.output(inputsB, "lm_logits_last");

        INDArray lastA = outA.get("lm_logits_last");
        INDArray lastB = outB.get("lm_logits_last");

        double maxDiff = lastA.sub(lastB).amaxNumber().doubleValue();
        log.info("Max difference in lm_logits_last when non-last rows differ: {}", maxDiff);
        assertEquals(0.0, maxDiff, 1e-5,
                "lm_logits_last must be identical when only non-last rows differ");

        hiddenA.close();
        hiddenB.close();
        scrambleRows.close();
        for (INDArray v : outA.values()) { if (v != null) v.close(); }
        for (INDArray v : outB.values()) { if (v != null) v.close(); }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3 (unit, no backend): config flag and kill-switch logic
    // ─────────────────────────────────────────────────────────────────────────
    @Test
    public void testConfigFlagAndKillSwitch() {
        // Default: flag is off
        GenerationPipelineConfig defaultConfig = GenerationPipelineConfig.builder()
                .tokenizer(null)  // tokenizer not needed for config-flag test
                .build();
        assertFalse(defaultConfig.isPrefillLastPositionLogitsEnabled(),
                "Default config: prefillLastPositionLogitsEnabled must be false");
        assertFalse(defaultConfig.isEffectivePrefillLastPositionLogitsEnabled(),
                "Default config: effective flag must be false");

        // Enabled explicitly
        GenerationPipelineConfig enabledConfig = GenerationPipelineConfig.builder()
                .tokenizer(null)
                .prefillLastPositionLogitsEnabled(true)
                .build();
        assertTrue(enabledConfig.isPrefillLastPositionLogitsEnabled(),
                "Enabled config: prefillLastPositionLogitsEnabled must be true");

        // Kill-switch via system property (without prop: still on)
        String propName = GenerationPipelineConfig.PREFILL_LAST_POSITION_LOGITS_PROP;
        String savedProp = System.getProperty(propName);
        try {
            System.clearProperty(propName);
            assertTrue(enabledConfig.isEffectivePrefillLastPositionLogitsEnabled(),
                    "With no kill-switch: effective flag must be true when config is true");

            // Kill-switch disables it
            System.setProperty(propName, "false");
            assertFalse(enabledConfig.isEffectivePrefillLastPositionLogitsEnabled(),
                    "With kill-switch=false: effective flag must be false even when config is true");

            // Prop=true: does NOT enable when config flag is false
            System.setProperty(propName, "true");
            assertFalse(defaultConfig.isEffectivePrefillLastPositionLogitsEnabled(),
                    "With prop=true but config flag=false: effective flag must remain false");
        } finally {
            if (savedProp != null) System.setProperty(propName, savedProp);
            else System.clearProperty(propName);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4 (parameterized): DSP compiles a plan for lm_logits_last output
    //   and the plan actually produces correct results (end-to-end DSP path).
    // ─────────────────────────────────────────────────────────────────────────
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDspPlanForLastPositionOutput(Nd4jBackend backend) {
        log.info("testDspPlanForLastPositionOutput backend={}", backend.getClass().getSimpleName());
        SameDiff sd = buildSyntheticLmHeadGraph(BATCH, SEQ, HIDDEN, VOCAB, DataType.FLOAT);
        // Enable DSP auto-compile so execution goes through the plan infrastructure
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray hiddenData = Nd4j.randn(DataType.FLOAT, BATCH, SEQ, HIDDEN);
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("hidden_states", hiddenData);

        // Run twice to exercise plan reuse
        for (int run = 0; run < 2; run++) {
            Map<String, INDArray> outputs = sd.output(inputs, "lm_logits_last");
            INDArray lastLogits = outputs.get("lm_logits_last");
            assertNotNull(lastLogits, "run " + run + ": lm_logits_last must be non-null");
            assertArrayEquals(new long[]{BATCH, 1, VOCAB}, lastLogits.shape(),
                    "run " + run + ": lm_logits_last shape mismatch");

            // Sanity: no NaN in output
            assertFalse(Double.isNaN(lastLogits.meanNumber().doubleValue()),
                    "run " + run + ": lm_logits_last must not contain NaN");
            for (INDArray v : outputs.values()) { if (v != null) v.close(); }
        }

        hiddenData.close();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5 (parameterized): verify lm_logits_last is present in the output
    //   names of the graph — i.e., the architecture registers it correctly.
    // ─────────────────────────────────────────────────────────────────────────
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGraphExposesLastPositionOutput(Nd4jBackend backend) {
        SameDiff sd = buildSyntheticLmHeadGraph(BATCH, SEQ, HIDDEN, VOCAB, DataType.FLOAT);
        List<String> outputs = sd.outputs();
        assertTrue(outputs.contains("lm_logits"),
                "Graph must expose lm_logits; actual outputs=" + outputs);
        assertTrue(outputs.contains("lm_logits_last"),
                "Graph must expose lm_logits_last; actual outputs=" + outputs);
    }
}
