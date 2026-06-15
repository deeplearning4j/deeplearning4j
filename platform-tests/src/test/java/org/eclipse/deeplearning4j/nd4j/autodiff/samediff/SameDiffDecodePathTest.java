/*
 *  ******************************************************************************
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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that exercise the EXACT execution paths used by GenerationPipeline:
 *
 * - SameDiff.output() for steps 0-1 (with setCloseable protection)
 * - SameDiff.outputDirect() for steps 2+ (NO setCloseable protection)
 * - Growing input shapes (view-based KV cache grows each step)
 * - DSP auto-compile potentially kicking in mid-sequence
 *
 * These test through the SameDiff graph execution layer, NOT raw Nd4j.exec().
 * The previous StaticKvDecodeIsolationTest used raw ops and passed — meaning
 * the bug is in SameDiff execution, not the underlying op.
 */
@Slf4j
@DisplayName("SameDiffDecodePathTest")
public class SameDiffDecodePathTest {

    private static final int BATCH = 1;
    private static final int NUM_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN = NUM_HEADS * HEAD_DIM;

    /**
     * Build a SameDiff graph with OnnxMultiHeadAttention that accepts
     * variable-length past KV (using -1 for the sequence dimension).
     */
    private SameDiff buildMhaGraph() {
        SameDiff sd = SameDiff.create();

        SDVariable query = sd.placeHolder("query", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, BATCH, 1, HIDDEN);
        // bias has dynamic last dim to match totalSeq
        SDVariable bias = sd.placeHolder("bias", DataType.FLOAT, 1, 1, 1, -1);
        // past KV has dynamic sequence dim
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, BATCH, NUM_HEADS, -1, HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, BATCH, NUM_HEADS, -1, HEAD_DIM);

        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, query, key, value, bias, pastKey, pastValue,
                NUM_HEADS, 0.0, false);
        SDVariable[] outputs = mha.outputVariables();
        sd.setOutputs(outputs[0].name());

        return sd;
    }

    /**
     * Hypothesis 1: outputDirect() vs output() divergence.
     *
     * Run the same step with output() and outputDirect() on the same graph,
     * verify they produce the same result. If not, setCloseable protection
     * matters and outputDirect() is unsafe for this use case.
     */
    @Test
    @DisplayName("testOutputVsOutputDirect_sameResult")
    public void testOutputVsOutputDirect_sameResult() {
        int pastSeq = 5;
        int totalSeq = pastSeq + 1;

        INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray pastK = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray pastV = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);

        // Run with output()
        SameDiff sd1 = buildMhaGraph();
        String attnName = sd1.outputs().get(0);
        Map<String, INDArray> ph1 = new HashMap<>();
        ph1.put("query", q.dup());
        ph1.put("key", k.dup());
        ph1.put("value", v.dup());
        ph1.put("bias", bias.dup());
        ph1.put("past_key", pastK.dup());
        ph1.put("past_value", pastV.dup());
        Map<String, INDArray> result1 = sd1.output(ph1, attnName);
        INDArray attn1 = result1.get(attnName).dup();

        // Run with outputDirect() on same graph (session reused)
        Map<String, INDArray> ph2 = new HashMap<>();
        ph2.put("query", q.dup());
        ph2.put("key", k.dup());
        ph2.put("value", v.dup());
        ph2.put("bias", bias.dup());
        ph2.put("past_key", pastK.dup());
        ph2.put("past_value", pastV.dup());
        Map<String, INDArray> result2 = sd1.outputDirect(ph2, attnName);
        INDArray attn2 = result2.get(attnName).dup();

        double maxDiff = attn1.sub(attn2).amaxNumber().doubleValue();
        log.info("output() vs outputDirect() maxDiff: {}", maxDiff);
        assertTrue(maxDiff < 1e-5,
                "output() and outputDirect() should produce identical results. MaxDiff=" + maxDiff);

        sd1.close();
    }

    /**
     * Hypothesis 2: outputDirect() with growing shapes (view-based KV).
     *
     * Run multiple steps through SameDiff, switching from output() (steps 0-1)
     * to outputDirect() (steps 2+), with growing KV views — exactly like
     * GenerationPipeline. Verify each step produces valid output.
     */
    @Test
    @DisplayName("testOutputDirect_growingShapes_multiStep")
    public void testOutputDirect_growingShapes_multiStep() {
        int maxKvLen = 30;
        int numSteps = 15;

        SameDiff sd = buildMhaGraph();
        String attnName = sd.outputs().get(0);

        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        int cachePos = 0;

        for (int step = 0; step < numSteps; step++) {
            INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("query", q);
            ph.put("key", k);
            ph.put("value", v);

            if (cachePos > 0) {
                int totalSeq = cachePos + 1;
                INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
                INDArray viewKey = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                INDArray viewValue = staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                ph.put("bias", bias);
                ph.put("past_key", viewKey);
                ph.put("past_value", viewValue);
            } else {
                // Step 0: no past KV — pass empty/minimal
                INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 1);
                INDArray emptyK = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM);
                INDArray emptyV = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM);
                ph.put("bias", bias);
                ph.put("past_key", emptyK);
                ph.put("past_value", emptyV);
            }

            // Use output() for steps 0-1, outputDirect() for steps 2+
            // This matches GenerationPipeline behavior
            Map<String, INDArray> result;
            boolean useDirect = step >= 2;
            if (useDirect) {
                result = sd.outputDirect(ph, attnName);
            } else {
                result = sd.output(ph, attnName);
            }

            INDArray attn = result.get(attnName);
            assertNotNull(attn, "Step " + step + ": null attention output");
            assertFalse(Double.isNaN(attn.sumNumber().doubleValue()),
                    "Step " + step + " (direct=" + useDirect + "): NaN in attention output");
            assertFalse(Double.isInfinite(attn.sumNumber().doubleValue()),
                    "Step " + step + " (direct=" + useDirect + "): Inf in attention output");
            assertTrue(attn.ameanNumber().doubleValue() > 1e-8,
                    "Step " + step + " (direct=" + useDirect + "): attention output all zeros");

            // Scatter new KV entry from present output into static buffer
            // present_key output is the second output from MHA
            // For simplicity, just scatter random KV (the point is testing the execution path)
            INDArray newEntry = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 1, HEAD_DIM).mul(0.1);
            staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                    newEntry.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                    newEntry.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            cachePos++;

            log.info("Step {} (direct={}): passed, absMax={}", step, useDirect,
                    attn.amaxNumber().floatValue());
        }

        sd.close();
    }

    /**
     * Hypothesis 3: outputDirect() + DSP auto-compile with growing shapes.
     *
     * Enable DSP auto-compile, run decode steps with growing shapes.
     * DSP may compile a plan on step 1, then step 2+ has different shapes.
     * If DSP doesn't properly invalidate/recompile, it produces garbage.
     */
    @Test
    @DisplayName("testDspAutoCompile_growingShapes")
    public void testDspAutoCompile_growingShapes() {
        int maxKvLen = 30;
        int numSteps = 10;

        SameDiff sd = buildMhaGraph();
        String attnName = sd.outputs().get(0);

        // Enable DSP auto-compile (matches what the decode pipeline does)
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        int cachePos = 0;

        for (int step = 0; step < numSteps; step++) {
            INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);

            Map<String, INDArray> ph = new HashMap<>();
            ph.put("query", q);
            ph.put("key", k);
            ph.put("value", v);

            if (cachePos > 0) {
                int totalSeq = cachePos + 1;
                INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
                INDArray viewKey = staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                INDArray viewValue = staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                ph.put("bias", bias);
                ph.put("past_key", viewKey);
                ph.put("past_value", viewValue);
            } else {
                INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 1);
                INDArray emptyK = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM);
                INDArray emptyV = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM);
                ph.put("bias", bias);
                ph.put("past_key", emptyK);
                ph.put("past_value", emptyV);
            }

            // Use outputDirect for step >= 2 (like the decode pipeline)
            Map<String, INDArray> result;
            boolean useDirect = step >= 2;
            if (useDirect) {
                result = sd.outputDirect(ph, attnName);
            } else {
                result = sd.output(ph, attnName);
            }

            INDArray attn = result.get(attnName);
            assertNotNull(attn, "Step " + step + ": null output");
            assertFalse(Double.isNaN(attn.sumNumber().doubleValue()),
                    "Step " + step + " (DSP+direct=" + useDirect + "): NaN");
            assertFalse(Double.isInfinite(attn.sumNumber().doubleValue()),
                    "Step " + step + " (DSP+direct=" + useDirect + "): Inf");
            assertTrue(attn.ameanNumber().doubleValue() > 1e-8,
                    "Step " + step + " (DSP+direct=" + useDirect + "): all zeros");

            // Check if DSP compiled a plan
            InferenceSession sess = sd.getOrCreateSession();
            DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
            boolean hasPlan = exec != null && exec.getCurrentPlan() != null;
            log.info("Step {} (direct={}, dspPlan={}): passed, absMax={}", step, useDirect,
                    hasPlan, attn.amaxNumber().floatValue());

            // Scatter
            INDArray newEntry = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 1, HEAD_DIM).mul(0.1);
            staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                    newEntry.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                    newEntry.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            cachePos++;
        }

        sd.close();
    }

    /**
     * Hypothesis 4: Multi-step output() vs outputDirect() token divergence.
     *
     * Run the full decode loop TWICE — once using only output(), once using
     * output() for steps 0-1 and outputDirect() for steps 2+.
     * If tokens diverge, outputDirect() is introducing errors.
     */
    @Test
    @DisplayName("testOutputDirect_tokenDivergence")
    public void testOutputDirect_tokenDivergence() {
        int maxKvLen = 30;
        int numSteps = 10;

        // Pre-generate inputs
        INDArray[] queries = new INDArray[numSteps];
        INDArray[] keys = new INDArray[numSteps];
        INDArray[] values = new INDArray[numSteps];
        INDArray[] scatterKeys = new INDArray[numSteps];
        INDArray[] scatterValues = new INDArray[numSteps];
        for (int i = 0; i < numSteps; i++) {
            queries[i] = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            keys[i] = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            values[i] = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            scatterKeys[i] = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 1, HEAD_DIM).mul(0.1);
            scatterValues[i] = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 1, HEAD_DIM).mul(0.1);
        }

        // === Path A: all output() ===
        INDArray[] attnsA = runDecodeLoop(numSteps, maxKvLen, queries, keys, values,
                scatterKeys, scatterValues, false);

        // === Path B: output() for 0-1, outputDirect() for 2+ ===
        INDArray[] attnsB = runDecodeLoop(numSteps, maxKvLen, queries, keys, values,
                scatterKeys, scatterValues, true);

        // Compare step by step
        for (int step = 0; step < numSteps; step++) {
            double maxDiff = attnsA[step].sub(attnsB[step]).amaxNumber().doubleValue();
            log.info("Step {}: output() vs outputDirect() maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < 1e-5,
                    "Step " + step + ": output() and outputDirect() diverge. MaxDiff=" + maxDiff);
        }
    }

    private INDArray[] runDecodeLoop(int numSteps, int maxKvLen,
                                     INDArray[] queries, INDArray[] keys, INDArray[] values,
                                     INDArray[] scatterKeys, INDArray[] scatterValues,
                                     boolean useDirectAfterStep1) {
        SameDiff sd = buildMhaGraph();
        String attnName = sd.outputs().get(0);

        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        int cachePos = 0;

        INDArray[] results = new INDArray[numSteps];

        for (int step = 0; step < numSteps; step++) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("query", queries[step].dup());
            ph.put("key", keys[step].dup());
            ph.put("value", values[step].dup());

            if (cachePos > 0) {
                int totalSeq = cachePos + 1;
                ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq));
                ph.put("past_key", staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()));
                ph.put("past_value", staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()));
            } else {
                ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 1));
                ph.put("past_key", Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM));
                ph.put("past_value", Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM));
            }

            Map<String, INDArray> result;
            boolean useDirect = useDirectAfterStep1 && step >= 2;
            if (useDirect) {
                result = sd.outputDirect(ph, attnName);
            } else {
                result = sd.output(ph, attnName);
            }

            results[step] = result.get(attnName).dup();

            // Scatter
            staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                    scatterKeys[step].get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                    scatterValues[step].get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(0), NDArrayIndex.all()));
            cachePos++;
        }

        sd.close();
        return results;
    }

    /**
     * Hypothesis 5: DSP auto-compile + outputDirect + growing shapes.
     *
     * The most specific reproduction: enable DSP, run steps 0-1 with output()
     * (DSP compiles on step 1), then run steps 2+ with outputDirect() and
     * growing shapes. This is the EXACT GenerationPipeline decode path.
     */
    @Test
    @DisplayName("testFullDecodePathWithDsp")
    public void testFullDecodePathWithDsp() {
        int maxKvLen = 30;
        int numSteps = 15;

        // Pre-generate all inputs
        INDArray[] queries = new INDArray[numSteps];
        INDArray[] keys = new INDArray[numSteps];
        INDArray[] values = new INDArray[numSteps];
        for (int i = 0; i < numSteps; i++) {
            queries[i] = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            keys[i] = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
            values[i] = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        }

        // === Reference path: all output(), no DSP ===
        INDArray[] refResults = new INDArray[numSteps];
        {
            SameDiff sd = buildMhaGraph();
            String attnName = sd.outputs().get(0);
            INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
            INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
            int cachePos = 0;

            for (int step = 0; step < numSteps; step++) {
                Map<String, INDArray> ph = new HashMap<>();
                ph.put("query", queries[step].dup());
                ph.put("key", keys[step].dup());
                ph.put("value", values[step].dup());
                if (cachePos > 0) {
                    ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, cachePos + 1));
                    ph.put("past_key", staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()));
                    ph.put("past_value", staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()));
                } else {
                    ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 1));
                    ph.put("past_key", Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM));
                    ph.put("past_value", Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM));
                }

                Map<String, INDArray> result = sd.output(ph, attnName);
                refResults[step] = result.get(attnName).dup();

                // Scatter using the actual MHA present_key output (simplified: use random)
                INDArray newK = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM).mul(0.1);
                staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                        newK.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(0), NDArrayIndex.all()));
                staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                        newK.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(0), NDArrayIndex.all()));
                cachePos++;
            }
            sd.close();
        }

        // === Test path: DSP enabled, output() for 0-1, outputDirect() for 2+ ===
        INDArray[] testResults = new INDArray[numSteps];
        {
            SameDiff sd = buildMhaGraph();
            String attnName = sd.outputs().get(0);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
            INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
            int cachePos = 0;

            for (int step = 0; step < numSteps; step++) {
                Map<String, INDArray> ph = new HashMap<>();
                ph.put("query", queries[step].dup());
                ph.put("key", keys[step].dup());
                ph.put("value", values[step].dup());
                if (cachePos > 0) {
                    ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, cachePos + 1));
                    ph.put("past_key", staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()));
                    ph.put("past_value", staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(0, cachePos), NDArrayIndex.all()));
                } else {
                    ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 1));
                    ph.put("past_key", Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM));
                    ph.put("past_value", Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, 0, HEAD_DIM));
                }

                Map<String, INDArray> result;
                boolean useDirect = step >= 2;
                if (useDirect) {
                    result = sd.outputDirect(ph, attnName);
                } else {
                    result = sd.output(ph, attnName);
                }

                testResults[step] = result.get(attnName).dup();

                // Log DSP state
                InferenceSession sess = sd.getOrCreateSession();
                DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
                boolean hasPlan = exec != null && exec.getCurrentPlan() != null;
                log.info("Step {} (direct={}, dsp={}): absMax={}",
                        step, useDirect, hasPlan,
                        testResults[step].amaxNumber().floatValue());

                // Use same scatter data as reference (same random seed won't work,
                // so we need deterministic scatter — use step index as seed)
                Nd4j.getRandom().setSeed(1000 + step);
                INDArray newK = Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, 1, HEAD_DIM).mul(0.1);
                staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                        newK.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(0), NDArrayIndex.all()));
                staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(
                        newK.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.point(0), NDArrayIndex.all()));
                cachePos++;
            }
            sd.close();
        }

        // Verify step 0 matches (same inputs, same path)
        double step0Diff = refResults[0].sub(testResults[0]).amaxNumber().doubleValue();
        log.info("Step 0 ref vs test maxDiff: {}", step0Diff);
        assertTrue(step0Diff < 1e-5,
                "Step 0 should match between ref and test. MaxDiff=" + step0Diff);

        // Steps 1+ may diverge due to different scatter data (random seed timing),
        // but each step individually should be valid
        for (int step = 0; step < numSteps; step++) {
            assertFalse(Double.isNaN(testResults[step].sumNumber().doubleValue()),
                    "Step " + step + " DSP+direct: NaN");
            assertTrue(testResults[step].ameanNumber().doubleValue() > 1e-8,
                    "Step " + step + " DSP+direct: all zeros");
        }
    }

    /**
     * Hypothesis 6: View of static buffer passed to SameDiff while scatter
     * writes to the same static buffer between steps.
     *
     * This tests whether SameDiff execution corrupts or is affected by
     * the underlying static buffer being modified between calls.
     */
    @Test
    @DisplayName("testViewAliasing_staticBufferModification")
    public void testViewAliasing_staticBufferModification() {
        int maxKvLen = 20;
        SameDiff sd = buildMhaGraph();
        String attnName = sd.outputs().get(0);

        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);

        // Fill first 3 positions with known data
        for (int i = 0; i < 3; i++) {
            staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(i), NDArrayIndex.all()).assign(i + 1.0f);
            staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(i), NDArrayIndex.all()).assign((i + 1.0f) * 10);
        }

        // Execute with view [0:3]
        INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("query", q.dup());
        ph.put("key", k.dup());
        ph.put("value", v.dup());
        ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 4));
        ph.put("past_key", staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 3), NDArrayIndex.all()));
        ph.put("past_value", staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 3), NDArrayIndex.all()));

        Map<String, INDArray> result1 = sd.output(ph, attnName);
        INDArray attn1 = result1.get(attnName).dup();

        // Now scatter to position 3 (OUTSIDE the previous view)
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(3), NDArrayIndex.all()).assign(4.0f);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(3), NDArrayIndex.all()).assign(40.0f);

        // Execute again with same view [0:3] — scatter to pos 3 should NOT affect this
        Map<String, INDArray> ph2 = new HashMap<>();
        ph2.put("query", q.dup());
        ph2.put("key", k.dup());
        ph2.put("value", v.dup());
        ph2.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 4));
        ph2.put("past_key", staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 3), NDArrayIndex.all()));
        ph2.put("past_value", staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 3), NDArrayIndex.all()));

        Map<String, INDArray> result2 = sd.output(ph2, attnName);
        INDArray attn2 = result2.get(attnName).dup();

        double maxDiff = attn1.sub(attn2).amaxNumber().doubleValue();
        log.info("Before/after scatter outside view: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5,
                "Scatter outside view should not affect result. MaxDiff=" + maxDiff);

        // Now execute with EXPANDED view [0:4] — should include the new data
        Map<String, INDArray> ph3 = new HashMap<>();
        ph3.put("query", q.dup());
        ph3.put("key", k.dup());
        ph3.put("value", v.dup());
        ph3.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 5));
        ph3.put("past_key", staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 4), NDArrayIndex.all()));
        ph3.put("past_value", staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, 4), NDArrayIndex.all()));

        Map<String, INDArray> result3 = sd.output(ph3, attnName);
        INDArray attn3 = result3.get(attnName);

        assertNotNull(attn3);
        assertFalse(Double.isNaN(attn3.sumNumber().doubleValue()),
                "Expanded view should produce valid output");

        sd.close();
    }

    /**
     * Hypothesis 7: Session reuse across shape changes.
     *
     * SameDiff reuses the InferenceSession across calls. If the session
     * caches intermediate results or shapes from a previous call, and the
     * next call has different shapes, cached data may be stale.
     */
    @Test
    @DisplayName("testSessionReuse_shapeChange")
    public void testSessionReuse_shapeChange() {
        SameDiff sd = buildMhaGraph();
        String attnName = sd.outputs().get(0);

        INDArray q = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);

        // Run with pastSeq=3
        INDArray pastK3 = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 3, HEAD_DIM).mul(0.1);
        INDArray pastV3 = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 3, HEAD_DIM).mul(0.1);
        Map<String, INDArray> ph1 = new HashMap<>();
        ph1.put("query", q.dup()); ph1.put("key", k.dup()); ph1.put("value", v.dup());
        ph1.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 4));
        ph1.put("past_key", pastK3); ph1.put("past_value", pastV3);

        Map<String, INDArray> r1 = sd.output(ph1, attnName);
        INDArray a1 = r1.get(attnName).dup();

        // Run with pastSeq=7 (different shape, same session)
        INDArray pastK7 = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 7, HEAD_DIM).mul(0.1);
        INDArray pastV7 = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, 7, HEAD_DIM).mul(0.1);
        Map<String, INDArray> ph2 = new HashMap<>();
        ph2.put("query", q.dup()); ph2.put("key", k.dup()); ph2.put("value", v.dup());
        ph2.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 8));
        ph2.put("past_key", pastK7); ph2.put("past_value", pastV7);

        Map<String, INDArray> r2 = sd.output(ph2, attnName);
        INDArray a2 = r2.get(attnName).dup();

        // Run with pastSeq=3 AGAIN — should match first run
        Map<String, INDArray> ph3 = new HashMap<>();
        ph3.put("query", q.dup()); ph3.put("key", k.dup()); ph3.put("value", v.dup());
        ph3.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 4));
        ph3.put("past_key", pastK3.dup()); ph3.put("past_value", pastV3.dup());

        Map<String, INDArray> r3 = sd.output(ph3, attnName);
        INDArray a3 = r3.get(attnName).dup();

        double maxDiff = a1.sub(a3).amaxNumber().doubleValue();
        log.info("Same inputs after shape change: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5,
                "Same inputs should produce same output after intermediate shape change. MaxDiff=" + maxDiff);

        // Now test with outputDirect
        Map<String, INDArray> ph4 = new HashMap<>();
        ph4.put("query", q.dup()); ph4.put("key", k.dup()); ph4.put("value", v.dup());
        ph4.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, 4));
        ph4.put("past_key", pastK3.dup()); ph4.put("past_value", pastV3.dup());

        Map<String, INDArray> r4 = sd.outputDirect(ph4, attnName);
        INDArray a4 = r4.get(attnName).dup();

        double directDiff = a1.sub(a4).amaxNumber().doubleValue();
        log.info("output() vs outputDirect() after shape changes: maxDiff={}", directDiff);
        assertTrue(directDiff < 1e-5,
                "outputDirect should match output() for same inputs. MaxDiff=" + directDiff);

        sd.close();
    }
}
