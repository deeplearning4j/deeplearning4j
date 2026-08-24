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
package org.nd4j.autodiff.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspHandle;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * DSP + OpenVINO accuracy isolation tests.
 *
 * Builds small self-contained SameDiff graphs, runs them with DSP+OpenVINO
 * (AUTO mode) and without DSP (EAGER mode), and compares outputs numerically.
 * Each test isolates a single op pattern to pinpoint where accuracy diverges.
 *
 * <p>Run a single test:
 * <pre>
 * cd platform-tests && mvn test \
 *   -Dtest=DspOpenVinoAccuracyTest#testMatmulChain \
 *   -Dbackend.artifactId=nd4j-native \
 *   2>&1 | tee /tmp/ov-accuracy.log
 * </pre>
 */
@Slf4j
@DisplayName("DSP OpenVINO Accuracy Isolation Tests")
@Tag("dsp")
public class DspOpenVinoAccuracyTest {

    private static final double ABS_TOL = 1e-4;
    private static final double REL_TOL = 1e-3;
    // Warm up + freeze iterations: 2 warmup + multiple frozen to catch drift
    private static final int WARMUP_ITERS = 2;
    private static final int FROZEN_ITERS = 5;

    private static boolean isNativeBackend() {
        return Nd4j.getBackend().getClass().getName().toLowerCase().contains("native")
            || Nd4j.getBackend().getClass().getName().toLowerCase().contains("cpu");
    }

    @BeforeEach
    void setUp() {
        assumeTrue(isNativeBackend(), "OpenVINO tests require the CPU/native backend");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, "ALL");
        System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, "summary");
    }

    @AfterEach
    void tearDown() {
        System.clearProperty(ND4JSystemProperties.DSP_DIAGNOSTICS);
        System.clearProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL);
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    /** Run the graph in EAGER mode (no DSP) to get the reference output. */
    private Map<String, INDArray> runEager(SameDiff sd, Map<String, INDArray> inputs, String... outputs) {
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        InferenceSession.setDynamicShapePlanEnabled(false);
        try {
            return sd.output(inputs, outputs);
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(true);
        }
    }

    /** Run the graph in AUTO mode (DSP + OpenVINO) for WARMUP_ITERS + FROZEN_ITERS, return last output. */
    private Map<String, INDArray> runDsp(SameDiff sd, Map<String, INDArray> inputs, String... outputs) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        InferenceSession.setDynamicShapePlanEnabled(true);

        Map<String, INDArray> result = null;
        for (int i = 0; i < WARMUP_ITERS + FROZEN_ITERS; i++) {
            result = sd.output(inputs, outputs);
        }
        return result;
    }

    /** Compare two output maps element-wise with tolerance. */
    private void assertOutputsMatch(String testName,
                                    Map<String, INDArray> expected,
                                    Map<String, INDArray> actual) {
        for (String key : expected.keySet()) {
            INDArray exp = expected.get(key);
            INDArray act = actual.get(key);
            assertNotNull(act, testName + ": missing output '" + key + "' from DSP run");

            // Shape check
            assertTrue(exp.equalShapes(act),
                    String.format("%s output '%s': shape mismatch — expected %s got %s",
                            testName, key, java.util.Arrays.toString(exp.shape()),
                            java.util.Arrays.toString(act.shape())));

            // Numerical check
            double maxAbsDiff = exp.sub(act).amaxNumber().doubleValue();
            double maxExpVal = exp.amaxNumber().doubleValue();
            double relErr = maxExpVal > 1e-10 ? maxAbsDiff / maxExpVal : maxAbsDiff;

            log.info("{} output '{}': maxAbsDiff={} relErr={} maxVal={}",
                    testName, key, maxAbsDiff, relErr, maxExpVal);

            if (maxAbsDiff > ABS_TOL && relErr > REL_TOL) {
                // Print first few values for debugging
                long n = Math.min(10, exp.length());
                INDArray expFlat = exp.dup().reshape(-1);
                INDArray actFlat = act.dup().reshape(-1);
                StringBuilder expStr = new StringBuilder("[");
                StringBuilder actStr = new StringBuilder("[");
                for (long j = 0; j < n; j++) {
                    if (j > 0) { expStr.append(", "); actStr.append(", "); }
                    expStr.append(expFlat.getFloat(j));
                    actStr.append(actFlat.getFloat(j));
                }
                expStr.append("]");
                actStr.append("]");
                log.error("{} ACCURACY FAILURE output '{}': maxAbsDiff={} relErr={}",
                        testName, key, maxAbsDiff, relErr);
                log.error("  expected first {}: {}", n, expStr);
                log.error("  actual   first {}: {}", n, actStr);
                fail(String.format("%s output '%s': accuracy exceeded tolerance — " +
                                "maxAbsDiff=%.6f relErr=%.6f (abs_tol=%.6f rel_tol=%.6f)",
                        testName, key, maxAbsDiff, relErr, ABS_TOL, REL_TOL));
            }
        }
    }

    // ── Test 1: Simple matmul chain ─────────────────────────────────────────

    /**
     * x -> matmul(W0) -> relu -> matmul(W1) -> output
     * Exercises: matmul, relu — basic ops that OpenVINO should handle perfectly.
     */
    @Test
    void testMatmulChain() {
        int batch = 1, inDim = 128, hidden = 256;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, inDim);
        SDVariable w0 = sd.var("w0", Nd4j.randn(DataType.FLOAT, inDim, hidden).muli(0.02));
        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, hidden, inDim).muli(0.02));
        SDVariable h = sd.nn.relu("h", sd.mmul("mm0", x, w0), 0);
        SDVariable y = sd.mmul("y", h, w1);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, batch, inDim));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testMatmulChain", expected, actual);

        DspHandle handle = sd.dsp();
        assertTrue(handle.numSegments() > 0, "OpenVINO test produced no DSP segments");
        assertTrue(handle.segmentCompiledBackend(0).contains("OpenVINO"),
                "AUTO x86 priority did not select OpenVINO: "
                        + handle.segmentCompiledBackend(0));
    }

    @Test
    @DisplayName("OpenVINO topology cache distinguishes equal-shape dataflow")
    void testTopologyCacheIncludesNormalizedWiring() {
        SameDiff first = SameDiff.create();
        SameDiff second = SameDiff.create();
        try {
            SDVariable firstInput = first.placeHolder("x", DataType.FLOAT, 1, 1, 16);
            SDVariable firstW1 = first.var("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
            SDVariable firstW2 = first.var("w2", Nd4j.randn(DataType.FLOAT, 16, 8));
            SDVariable firstFlat = first.reshape("xflat", firstInput, 1, 16);
            SDVariable firstMm = first.mmul("mm1", firstFlat, firstW1);
            SDVariable firstResidual = firstFlat.add("residual", firstMm);
            first.mmul("out", firstResidual, firstW2);
            first.setOutputs("out");

            SDVariable secondInput = second.placeHolder("x", DataType.FLOAT, 1, 1, 16);
            SDVariable secondW1 = second.var("w1", Nd4j.randn(DataType.FLOAT, 16, 16));
            SDVariable secondW2 = second.var("w2", Nd4j.randn(DataType.FLOAT, 16, 8));
            SDVariable secondFlat = second.reshape("xflat", secondInput, 1, 16);
            SDVariable secondMm = second.mmul("mm1", secondFlat, secondW1);
            // Same op sequence and tensor shapes as the first graph, but a
            // different producer edge into add. The old cache key collided.
            SDVariable secondResidual = secondMm.add("residual", secondMm);
            second.mmul("out", secondResidual, secondW2);
            second.setOutputs("out");

            Map<String, INDArray> firstInputs = new LinkedHashMap<>();
            firstInputs.put("x", Nd4j.randn(DataType.FLOAT, 1, 1, 16));
            runDsp(first, firstInputs, "out");

            Map<String, INDArray> secondInputs = new LinkedHashMap<>();
            secondInputs.put("x", Nd4j.randn(DataType.FLOAT, 1, 1, 16));
            Map<String, INDArray> expected = runEager(second, secondInputs, "out");
            Map<String, INDArray> actual = runDsp(second, secondInputs, "out");
            assertOutputsMatch("testTopologyCacheIncludesNormalizedWiring", expected, actual);
            assertTrue(second.dsp().segmentCompiledBackend(0).contains("OpenVINO"));
        } finally {
            second.close();
            first.close();
        }
    }

    @Test
    @DisplayName("OpenVINO layout drift re-resolves before execution")
    void testSameShapeLayoutDriftFallsThroughBeforeExecution() {
        SameDiff sd = SameDiff.create();
        try {
            INDArray weights = Nd4j.randn(DataType.FLOAT, 4, 3);
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, 2, 4);
            SDVariable weight = sd.var("weight", weights.dup());
            sd.mmul("out", input, weight);
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

            INDArray cInput = Nd4j.create(DataType.FLOAT, new long[]{2, 4}, 'c').assign(0.5);
            Map<String, INDArray> cValues = Map.of("input", cInput);
            for (int execution = 0; execution < 7; ++execution) {
                sd.output(cValues, "out");
            }
            assertTrue(sd.dsp().segmentCompiledBackend(0).contains("OpenVINO"));

            INDArray fInput = Nd4j.create(DataType.FLOAT, new long[]{2, 4}, 'f').assign(1.25);
            INDArray expected = fInput.mmul(weights);
            INDArray actual = sd.output(Map.of("input", fInput), "out").get("out");
            assertTrue(expected.equalsWithEps(actual, 1e-5),
                    "same-shape F-order rebind used a stale dense OpenVINO artifact");
            assertTrue(sd.dsp().segmentCompiledBackend(0).contains("OneDNN"),
                    "non-dense input was not transferred to the exact-stride OneDNN tier: "
                            + sd.dsp().segmentCompiledBackend(0));
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("OpenVINO refreshes mutable FP16 promotion at a stable address")
    void testMutableFp16InputIsPromotedEveryExecution() {
        SameDiff sd = SameDiff.create();
        try {
            INDArray weights = Nd4j.ones(DataType.HALF, 4, 3);
            SDVariable input = sd.placeHolder("input", DataType.HALF, 2, 4);
            sd.mmul("out", input, sd.var("weight", weights));
            sd.setOutputs("out");
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

            INDArray stableInput = Nd4j.create(DataType.HALF, new long[]{2, 4}, 'c');
            double previousSum = Double.NaN;
            for (int execution = 1; execution <= 8; ++execution) {
                stableInput.assign(execution * 0.25);
                INDArray actual = sd.output(Map.of("input", stableInput), "out").get("out");
                double expectedSum = 2.0 * 3.0 * 4.0 * execution * 0.25;
                double actualSum = actual.sumNumber().doubleValue();
                assertTrue(Math.abs(expectedSum - actualSum) < 0.1,
                        "mutable FP16 promotion was stale at execution " + execution
                                + ": expected sum=" + expectedSum + " actual=" + actualSum);
                if (!Double.isNaN(previousSum)) {
                    assertTrue(Math.abs(actualSum - previousSum) > 0.1,
                            "mutable FP16 output did not change at execution " + execution);
                }
                previousSum = actualSum;
            }
            assertTrue(sd.dsp().segmentCompiledBackend(0).contains("OpenVINO"));
        } finally {
            sd.close();
        }
    }

    // ── Test 2: RMSNorm ─────────────────────────────────────────────────────

    /**
     * x -> rms_norm(gamma) -> output
     * Exercises: rms_norm — common in transformer blocks, OpenVINO may decompose it.
     */
    @Test
    void testRmsNorm() {
        int batch = 1, seq = 4, dim = 128;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seq, dim);
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));
        SDVariable y = sd.nn.rmsNorm("y", x, gamma, 1e-6);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, batch, seq, dim));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testRmsNorm", expected, actual);
    }

    // ── Test 3: RMSNorm + Matmul (transformer pre-attention block) ──────────

    /**
     * x -> rms_norm(gamma) -> matmul(W) -> output
     * Exercises: rms_norm -> matmul chain — the Q/K/V projection pattern.
     */
    @Test
    void testRmsNormMatmul() {
        int batch = 1, seq = 4, dim = 128;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seq, dim);
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-6);
        // Reshape [1,4,128] -> [4,128] for matmul, then back
        SDVariable flat = sd.reshape("flat", normed, batch * seq, dim);
        SDVariable proj = sd.mmul("proj", flat, w);
        SDVariable y = sd.reshape("y", proj, batch, seq, dim);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, batch, seq, dim));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testRmsNormMatmul", expected, actual);
    }

    // ── Test 4: Dot product attention V2 (4D flash) ─────────────────────────

    /**
     * Q,K,V -> dot_product_attention_v2 -> output
     * 4D inputs: [batch, seq, heads, headDim]
     * This is the op that's native-deferred within OpenVINO segments.
     */
    @Test
    void testDotProductAttentionV2() {
        int batch = 1, seqQ = 4, seqKV = 4, heads = 4, headDim = 32;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, heads, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqKV, heads, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqKV, heads, headDim);
        SDVariable y = sd.nn.dotProductAttentionV2("y", q, v, k, null, null,
                scale, 0.0, false, false);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("q", Nd4j.randn(DataType.FLOAT, batch, seqQ, heads, headDim).muli(0.1));
        inputs.put("k", Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1));
        inputs.put("v", Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testDotProductAttentionV2", expected, actual);
    }

    // ── Test 5: Attention decode (seqQ=1) ───────────────────────────────────

    /**
     * Same as above but with seqQ=1 — the decode path.
     * This is the critical path for LLM token generation.
     */
    @Test
    void testAttentionDecode() {
        int batch = 1, seqQ = 1, seqKV = 8, heads = 4, headDim = 32;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, batch, seqQ, heads, headDim);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, batch, seqKV, heads, headDim);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, batch, seqKV, heads, headDim);
        SDVariable y = sd.nn.dotProductAttentionV2("y", q, v, k, null, null,
                scale, 0.0, true, false);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("q", Nd4j.randn(DataType.FLOAT, batch, seqQ, heads, headDim).muli(0.1));
        inputs.put("k", Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1));
        inputs.put("v", Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testAttentionDecode", expected, actual);
    }

    // ── Test 6: Mini transformer layer (rmsNorm + QKV proj + attention + residual) ──

    /**
     * Minimal transformer attention layer:
     * x -> rms_norm -> Q,K,V projections -> attention -> output_proj -> add(residual) -> y
     *
     * This combines all the patterns from tests 2-5 into one graph that
     * resembles a single Qwen attention layer. If tests 1-5 pass but this
     * fails, the issue is in how OpenVINO handles the data flow between
     * native-deferred attention and the surrounding fused ops.
     */
    @Test
    void testMiniTransformerLayer() {
        int batch = 1, seq = 4, dim = 128, heads = 4;
        int headDim = dim / heads;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();

        // Input placeholder
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seq, dim);

        // RMSNorm
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));
        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-6);

        // Q, K, V projections: [batch*seq, dim] @ [dim, dim] -> [batch*seq, dim]
        SDVariable wq = sd.var("wq", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wk = sd.var("wk", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable wv = sd.var("wv", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));

        SDVariable flat = sd.reshape("flat", normed, batch * seq, dim);
        SDVariable qFlat = sd.mmul("qFlat", flat, wq);
        SDVariable kFlat = sd.mmul("kFlat", flat, wk);
        SDVariable vFlat = sd.mmul("vFlat", flat, wv);

        // Reshape to 4D: [batch, seq, heads, headDim]
        SDVariable q = sd.reshape("q", qFlat, batch, seq, heads, headDim);
        SDVariable k = sd.reshape("k", kFlat, batch, seq, heads, headDim);
        SDVariable v = sd.reshape("v", vFlat, batch, seq, heads, headDim);

        // Attention
        SDVariable attnOut = sd.nn.dotProductAttentionV2("attn", q, v, k, null, null,
                scale, 0.0, false, false);

        // Output projection: [batch*seq, dim] @ [dim, dim]
        SDVariable wo = sd.var("wo", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable attnFlat = sd.reshape("attnFlat", attnOut, batch * seq, dim);
        SDVariable proj = sd.mmul("proj", attnFlat, wo);
        SDVariable projReshaped = sd.reshape("projReshaped", proj, batch, seq, dim);

        // Residual connection
        SDVariable y = sd.math.add("y", x, projReshaped);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, batch, seq, dim).muli(0.1));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testMiniTransformerLayer", expected, actual);
    }

    // ── Test 7: Mini transformer decode (seqQ=1, frozen steady state) ───────

    /**
     * Same as test 6 but with seqQ=1 for decode and runs multiple iterations
     * to ensure the frozen fast path produces correct results over time.
     * This is the closest reproduction of the Qwen decode path.
     */
    @Test
    void testMiniTransformerDecode() {
        int batch = 1, seqQ = 1, seqKV = 8, dim = 128, heads = 4;
        int headDim = dim / heads;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seqQ, dim);
        SDVariable kIn = sd.placeHolder("kIn", DataType.FLOAT, batch, seqKV, heads, headDim);
        SDVariable vIn = sd.placeHolder("vIn", DataType.FLOAT, batch, seqKV, heads, headDim);

        // RMSNorm
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));
        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-6);

        // Q projection only (K,V come from cache)
        SDVariable wq = sd.var("wq", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable flat = sd.reshape("flat", normed, batch * seqQ, dim);
        SDVariable qFlat = sd.mmul("qFlat", flat, wq);
        SDVariable q = sd.reshape("q", qFlat, batch, seqQ, heads, headDim);

        // Attention with cached K,V
        SDVariable attnOut = sd.nn.dotProductAttentionV2("attn", q, vIn, kIn, null, null,
                scale, 0.0, true, false);

        // Output projection + residual
        SDVariable wo = sd.var("wo", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable attnFlat = sd.reshape("attnFlat", attnOut, batch * seqQ, dim);
        SDVariable proj = sd.mmul("proj", attnFlat, wo);
        SDVariable projReshaped = sd.reshape("projReshaped", proj, batch, seqQ, dim);
        SDVariable y = sd.math.add("y", x, projReshaped);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, batch, seqQ, dim).muli(0.1));
        inputs.put("kIn", Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1));
        inputs.put("vIn", Nd4j.randn(DataType.FLOAT, batch, seqKV, heads, headDim).muli(0.1));

        // Get reference
        Map<String, INDArray> expected = runEager(sd, inputs, "y");

        // Run DSP with more iterations to exercise frozen steady state
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        InferenceSession.setDynamicShapePlanEnabled(true);

        Map<String, INDArray> actual = null;
        for (int i = 0; i < WARMUP_ITERS + 10; i++) {
            actual = sd.output(inputs, "y");
            if (i == WARMUP_ITERS - 1) {
                // Check accuracy at the warmup boundary
                assertOutputsMatch("testMiniTransformerDecode[warmup_end]", expected, actual);
            }
        }
        // Check accuracy after many frozen iterations
        assertOutputsMatch("testMiniTransformerDecode[frozen_steady]", expected, actual);
    }

    // ── Test 8: SiLU + multiply (swish_mul pattern) ─────────────────────────

    /**
     * x -> split -> (silu(a), b) -> multiply -> output
     * This is the MLP gate pattern in Qwen/LLaMA.
     */
    @Test
    void testSwishMulPattern() {
        int batch = 1, seq = 4, dim = 128;
        int halfDim = dim / 2;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, seq, dim);

        // Split into two halves along last dim
        SDVariable a = sd.stridedSlice("a", x,
                new long[]{0, 0, 0}, new long[]{batch, seq, halfDim}, new long[]{1, 1, 1},
                0, 0, 0, 0, 0);
        SDVariable b = sd.stridedSlice("b", x,
                new long[]{0, 0, halfDim}, new long[]{batch, seq, dim}, new long[]{1, 1, 1},
                0, 0, 0, 0, 0);

        // SiLU(a) * b
        SDVariable silu = sd.nn.sigmoid("sig", a).mul("silu", a);
        SDVariable y = silu.mul("y", b);
        sd.setOutputs("y");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", Nd4j.randn(DataType.FLOAT, batch, seq, dim));

        Map<String, INDArray> expected = runEager(sd, inputs, "y");
        Map<String, INDArray> actual = runDsp(sd, inputs, "y");
        assertOutputsMatch("testSwishMulPattern", expected, actual);
    }

    // ── Test 9: Prefill → Decode shape transition ───────────────────────────

    /**
     * Run a graph first with seq=4 (prefill), then with seq=1 (decode).
     * Tests that the DSP plan correctly handles the shape transition and
     * doesn't produce stale/wrong outputs after resegmentation.
     */
    @Test
    void testPrefillToDecodeTransition() {
        int batch = 1, dim = 128;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, batch, -1, dim);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.02));
        SDVariable gamma = sd.var("gamma", Nd4j.ones(DataType.FLOAT, dim));

        SDVariable normed = sd.nn.rmsNorm("normed", x, gamma, 1e-6);
        // Flatten to 2D using -1 for dynamic batch*seq
        SDVariable flat = sd.reshape("flat", normed, -1, dim);
        SDVariable proj = sd.mmul("proj", flat, w);
        SDVariable y = sd.reshape("y", proj, batch, -1, dim);
        sd.setOutputs("y");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Phase 1: Prefill (seq=4)
        Map<String, INDArray> prefillInputs = new LinkedHashMap<>();
        prefillInputs.put("x", Nd4j.randn(DataType.FLOAT, batch, 4, dim));
        for (int i = 0; i < 3; i++) {
            sd.output(prefillInputs, "y");
        }

        // Phase 2: Decode (seq=1) — shape transition triggers resegment
        Map<String, INDArray> decodeInputs = new LinkedHashMap<>();
        decodeInputs.put("x", Nd4j.randn(DataType.FLOAT, batch, 1, dim));

        // Get reference for decode shape
        Map<String, INDArray> expected = runEager(sd, decodeInputs, "y");

        // Run DSP decode iterations
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
        InferenceSession.setDynamicShapePlanEnabled(true);

        Map<String, INDArray> actual = null;
        for (int i = 0; i < 10; i++) {
            actual = sd.output(decodeInputs, "y");
        }
        assertOutputsMatch("testPrefillToDecodeTransition", expected, actual);
    }
}
