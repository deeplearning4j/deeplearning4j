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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Direct comparison of output() vs outputDirect() at every step across 50 steps.
 *
 * Uses two separate SameDiff instances with identical weights:
 * - "dsp" instance: uses DSP + frozen shapes + outputDirect() (the optimized path)
 * - "ref" instance: uses plain output() every step (the reference path)
 *
 * At every step, both get the exact same input data. Results are compared to detect
 * the precise step where DSP/frozen execution diverges from standard execution.
 *
 * This directly pinpoints the progressive degeneration bug.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class OutputVsOutputDirectDivergenceTest extends BaseNd4jTestWithBackends {

    private static final int HIDDEN = 64;
    private static final int NUM_LAYERS = 4;
    private static final int TOTAL_STEPS = 50;
    private static final double TOLERANCE = 1e-3;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void cleanup() {
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
    }

    /**
     * Build a deeper network to exercise more ops in the DSP plan:
     *   input -> (matmul + bias + tanh) x NUM_LAYERS -> matmul(output_proj) -> output
     */
    private SameDiff buildNetwork(long seed) {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, HIDDEN);
        SDVariable current = input;

        Nd4j.getRandom().setSeed(seed);
        for (int i = 0; i < NUM_LAYERS; i++) {
            SDVariable w = sd.var("w_" + i, Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.02));
            SDVariable b = sd.var("b_" + i, Nd4j.zeros(DataType.FLOAT, HIDDEN));
            current = sd.mmul("matmul_" + i, current, w);
            current = current.add("add_" + i, b);
            current = sd.math.tanh("act_" + i, current);
        }

        // Output projection to a different size (simulating vocab projection)
        SDVariable wOut = sd.var("w_out", Nd4j.randn(DataType.FLOAT, HIDDEN, 32).muli(0.02));
        current = sd.mmul("output_proj", current, wOut);
        current = sd.identity("output", current);
        sd.setOutputs("output");
        return sd;
    }

    /**
     * Copy all VARIABLE weights from src to dst (must have matching variable names).
     */
    private void copyWeights(SameDiff src, SameDiff dst) {
        for (String varName : src.variableMap().keySet()) {
            if (src.getVariable(varName).getVariableType() == VariableType.VARIABLE) {
                INDArray srcArr = src.getVariable(varName).getArr();
                if (srcArr != null && dst.hasVariable(varName)) {
                    dst.getVariable(varName).getArr().assign(srcArr);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOutputVsOutputDirectEveryStep(Nd4jBackend backend) {
        // Build two identical networks
        SameDiff dspSd = buildNetwork(42L);
        SameDiff refSd = buildNetwork(42L);
        // Weights are already identical (same seed), but verify by copying
        copyWeights(dspSd, refSd);

        // Configure DSP on the "dsp" instance
        dspSd.setDspAutoCompileEnabled(true);
        dspSd.setDspNativeAutoCompileEnabled(true);
        dspSd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Fixed-address buffer for DSP instance (mimics decode loop pattern)
        INDArray reusableInput = Nd4j.zeros(DataType.FLOAT, 1, HIDDEN);

        // Warm up DSP path (2 output() calls to trigger compilation)
        for (int warmup = 0; warmup < 2; warmup++) {
            Nd4j.getRandom().setSeed(warmup * 999L);
            reusableInput.assign(Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f));
            dspSd.output(Collections.singletonMap("input", reusableInput), "output");
        }

        // Freeze shapes on DSP instance
        InferenceSession session = dspSd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            log.info("DSP shapes frozen");
        }

        // Step-by-step comparison
        int matchCount = 0;
        int mismatchCount = 0;
        int firstDivergenceStep = -1;
        double worstDiff = 0;
        int worstDiffStep = -1;

        for (int step = 0; step < TOTAL_STEPS; step++) {
            // Generate the same input for both
            Nd4j.getRandom().setSeed(step * 12345L + 67890);
            INDArray stepInput = Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f);

            // DSP path: assign to reusable buffer, call outputDirect
            reusableInput.assign(stepInput);
            Map<String, INDArray> dspResult = dspSd.outputDirect(
                    Collections.singletonMap("input", reusableInput), "output");
            float[] dspData = dspResult.get("output").dup().data().asFloat();

            // Reference path: fresh input, call output()
            Nd4j.getRandom().setSeed(step * 12345L + 67890);
            INDArray refInput = Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f);
            Map<String, INDArray> refResult = refSd.output(
                    Collections.singletonMap("input", refInput), "output");
            float[] refData = refResult.get("output").dup().data().asFloat();

            // Compare element-by-element
            double maxDiff = 0;
            int maxDiffIdx = 0;
            for (int i = 0; i < Math.min(dspData.length, refData.length); i++) {
                double diff = Math.abs(dspData[i] - refData[i]);
                if (diff > maxDiff) {
                    maxDiff = diff;
                    maxDiffIdx = i;
                }
            }

            if (maxDiff > worstDiff) {
                worstDiff = maxDiff;
                worstDiffStep = step;
            }

            boolean matches = maxDiff < TOLERANCE;
            if (matches) {
                matchCount++;
            } else {
                mismatchCount++;
                if (firstDivergenceStep < 0) {
                    firstDivergenceStep = step;
                    log.warn("FIRST DIVERGENCE at step {}: maxDiff={} at index {}", step,
                            String.format("%.8f", maxDiff), maxDiffIdx);
                    // Log first few values for debugging
                    StringBuilder sb = new StringBuilder();
                    sb.append("  DSP:  [");
                    for (int i = 0; i < Math.min(8, dspData.length); i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(String.format("%.6f", dspData[i]));
                    }
                    sb.append("]");
                    log.warn(sb.toString());
                    sb = new StringBuilder();
                    sb.append("  Ref:  [");
                    for (int i = 0; i < Math.min(8, refData.length); i++) {
                        if (i > 0) sb.append(", ");
                        sb.append(String.format("%.6f", refData[i]));
                    }
                    sb.append("]");
                    log.warn(sb.toString());
                }
            }

            if (step < 5 || step % 10 == 0 || !matches) {
                log.info("Step {}: maxDiff={} {} (match={}/{})",
                        step, String.format("%.8f", maxDiff),
                        matches ? "OK" : "DIVERGED",
                        matchCount, step + 1);
            }
        }

        // Summary
        log.info("=== OUTPUT VS OUTPUT_DIRECT DIVERGENCE RESULTS ===");
        log.info("  Total steps: {}", TOTAL_STEPS);
        log.info("  Matches: {} / {}", matchCount, TOTAL_STEPS);
        log.info("  Mismatches: {}", mismatchCount);
        log.info("  First divergence step: {}", firstDivergenceStep);
        log.info("  Worst diff: {} at step {}", String.format("%.8f", worstDiff), worstDiffStep);

        // Assertions
        assertTrue(matchCount >= TOTAL_STEPS * 0.8,
                "At least 80% of steps should match between output() and outputDirect(), got " +
                        matchCount + "/" + TOTAL_STEPS);
        if (firstDivergenceStep >= 0) {
            log.warn("output() vs outputDirect() diverged starting at step {} -- " +
                    "this indicates a DSP/frozen execution bug", firstDivergenceStep);
        }
        assertTrue(worstDiff < 1.0,
                "Worst diff between output() and outputDirect() should be <1.0, got " +
                        String.format("%.8f", worstDiff) + " at step " + worstDiffStep);

        dspSd.close();
        refSd.close();
    }

    /**
     * Same as above but also tests the pattern where the graph has multiple inputs
     * and only one changes per step (the decode pattern).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testOutputVsOutputDirectWithStaticAndDynamicInputs(Nd4jBackend backend) {
        SameDiff dspSd = SameDiff.create();
        SameDiff refSd = SameDiff.create();

        // Build a graph: output = tanh(dynamicInput * W1 + staticInput * W2)
        Nd4j.getRandom().setSeed(777L);
        for (SameDiff sd : new SameDiff[]{dspSd, refSd}) {
            Nd4j.getRandom().setSeed(777L);
            SDVariable dynamic = sd.placeHolder("dynamic", DataType.FLOAT, 1, HIDDEN);
            SDVariable fixed = sd.placeHolder("static", DataType.FLOAT, 1, HIDDEN);
            SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.02));
            SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.02));
            SDVariable h1 = sd.mmul("mmul1", dynamic, w1);
            SDVariable h2 = sd.mmul("mmul2", fixed, w2);
            SDVariable sum = h1.add("sum", h2);
            SDVariable output = sd.math.tanh("output", sum);
            sd.setOutputs("output");
        }
        copyWeights(dspSd, refSd);
        Nd4j.getExecutioner().commit();

        // Configure DSP
        dspSd.setDspAutoCompileEnabled(true);
        dspSd.setDspNativeAutoCompileEnabled(true);
        dspSd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Fixed inputs
        INDArray staticInput = Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f);
        INDArray reusableDynamic = Nd4j.zeros(DataType.FLOAT, 1, HIDDEN);
        Nd4j.getExecutioner().commit();

        // Warm up
        for (int warmup = 0; warmup < 2; warmup++) {
            reusableDynamic.assign(Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f));
            Nd4j.getExecutioner().commit();
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("dynamic", reusableDynamic);
            inputs.put("static", staticInput);
            dspSd.output(inputs, "output");
            Nd4j.getExecutioner().commit();
        }

        // Freeze
        InferenceSession sess = dspSd.getOrCreateSession();
        DynamicShapePlanExecutor exec = sess.getDynamicShapePlanExecutor();
        if (exec != null) exec.setShapesFrozen(true);

        // Run 50 steps
        int matches = 0;
        int firstDivergence = -1;

        for (int step = 0; step < TOTAL_STEPS; step++) {
            // Generate input data once, dup for each path
            Nd4j.getRandom().setSeed(step * 54321L);
            INDArray dynamicData = Nd4j.randn(DataType.FLOAT, 1, HIDDEN).muli(0.1f);
            Nd4j.getExecutioner().commit();

            // DSP path
            reusableDynamic.assign(dynamicData);
            Nd4j.getExecutioner().commit();
            Map<String, INDArray> dspInputs = new HashMap<>();
            dspInputs.put("dynamic", reusableDynamic);
            dspInputs.put("static", staticInput);
            Map<String, INDArray> dspResult = dspSd.outputDirect(dspInputs, "output");
            Nd4j.getExecutioner().commit();
            float[] dspData = dspResult.get("output").dup().data().asFloat();

            // Reference path — fresh input dup
            INDArray refDynamic = dynamicData.dup();
            INDArray refStatic = staticInput.dup();
            Nd4j.getExecutioner().commit();
            Map<String, INDArray> refInputs = new HashMap<>();
            refInputs.put("dynamic", refDynamic);
            refInputs.put("static", refStatic);
            Map<String, INDArray> refResult = refSd.output(refInputs, "output");
            Nd4j.getExecutioner().commit();
            float[] refData = refResult.get("output").dup().data().asFloat();

            double maxDiff = 0;
            for (int i = 0; i < Math.min(dspData.length, refData.length); i++) {
                maxDiff = Math.max(maxDiff, Math.abs(dspData[i] - refData[i]));
            }

            if (maxDiff < TOLERANCE) {
                matches++;
            } else if (firstDivergence < 0) {
                firstDivergence = step;
                log.warn("DIVERGENCE at step {} with mixed static/dynamic inputs: maxDiff={}",
                        step, String.format("%.8f", maxDiff));
            }

            if (step < 3 || step % 10 == 0) {
                log.info("Step {} (mixed inputs): maxDiff={}", step, String.format("%.8f", maxDiff));
            }
        }

        log.info("=== MIXED INPUT DIVERGENCE RESULTS ===");
        log.info("  Matches: {} / {}", matches, TOTAL_STEPS);
        log.info("  First divergence: {}", firstDivergence);

        assertTrue(matches >= TOTAL_STEPS * 0.8,
                "At least 80% of steps should match with mixed inputs, got " + matches + "/" + TOTAL_STEPS);

        dspSd.close();
        refSd.close();
    }
}
