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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1dWithPrefix;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Prefix-history capture contract for causal_conv1d_with_prefix.
 *
 * For every live prefix m, checkpoint slot m-1 must equal the final state of an
 * independent legacy execution at actualLen=m from the SAME nonzero initial state,
 * and ordinary outputs must be identical. Together with the GDN prefix fixture
 * this is the oracle that lets the MTP controller select prefix[consumed-1]
 * instead of re-executing the target for the consumed prefix.
 */
public class TestCausalConvPrefixStates {

    private static final int B = 1;
    private static final int D = 4;
    private static final int KC = 4; // convolution width; history length KC-1 = 3

    private static INDArray[] deterministicInputs(int l) {
        INDArray x = Nd4j.linspace(1, B * l * D, B * l * D, DataType.FLOAT)
                .reshape(B, l, D).muli(0.1f);
        INDArray weight = Nd4j.linspace(1, D * KC, D * KC, DataType.FLOAT)
                .reshape(D, KC).muli(0.01f);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, D);
        return new INDArray[]{x, weight, bias};
    }

    private static INDArray nonzeroHistory() {
        INDArray s = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        float value = 0.25f;
        for (int b = 0; b < B; b++) {
            for (int d = 0; d < D; d++) {
                for (int kk = 0; kk < KC - 1; kk++) {
                    s.putScalar(new int[]{b, d, kk}, value);
                    value = value * -1.5f + 0.125f;
                }
            }
        }
        return s;
    }

    private static void assertFinite(INDArray a, String name) {
        assertTrue(!a.isNaN().any(), name + " contains NaN");
        assertTrue(!a.isInfinite().any(), name + " contains Inf");
    }

    /** history_m element [b, d, kk] computed directly from the declared oracle. */
    private static float oracleHistory(INDArray stateIn, INDArray x, int b, int d, int kk, int m) {
        int srcT = m - (KC - 1) + (int) kk;
        if (srcT >= 0) {
            return x.getFloat(b, srcT, d);
        }
        int stateIdx = (KC - 1) + srcT;
        if (stateIdx >= 0) {
            return stateIn.getFloat(b, d, stateIdx);
        }
        return 0.0f;
    }

    @Test
    public void testPrefixSlotsMatchOracleAndLegacy() {
        for (int l : new int[]{2, 5, 8}) {
            INDArray[] in = deterministicInputs(l);
            INDArray x = in[0];
            INDArray weight = in[1];
            INDArray bias = in[2];
            INDArray stateIn = nonzeroHistory();
            INDArray stateInBackup = stateIn.dup();

            INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
            INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
            INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);

            CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                    x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l));
            op.addOutputArgument(output, stateOut, prefix);
            Nd4j.getExecutioner().exec(op);
            Nd4j.getExecutioner().commit();

            assertFinite(output, "output L=" + l);
            assertFinite(stateOut, "stateOut L=" + l);
            assertFinite(prefix, "prefix L=" + l);

            // Every prefix slot against the direct oracle.
            for (int m = 1; m <= l; m++) {
                for (int kk = 0; kk < KC - 1; kk++) {
                    assertEquals(oracleHistory(stateInBackup, x, 0, 0, kk, m),
                            prefix.getFloat(m - 1, 0, 0, kk), 0.0f,
                            "L=" + l + " prefix[" + (m - 1) + "] kk=" + kk);
                    assertEquals(oracleHistory(stateInBackup, x, 0, D - 1, kk, m),
                            prefix.getFloat(m - 1, 0, D - 1, kk), 0.0f,
                            "L=" + l + " last-channel prefix[" + (m - 1) + "] kk=" + kk);
                }
            }

            // Final state equals legacy execution at actualLen = l.
            INDArray legacyState = runLegacyState(x, weight, bias, stateInBackup, l);
            assertEquals(0.0, stateOut.sub(legacyState).amaxNumber().doubleValue(), 0.0f,
                    "L=" + l + " final state differs from legacy");
            assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                    "stateIn was mutated, L=" + l);
        }
    }

    @Test
    public void testShortenedActualLenMatchesLegacy() {
        int l = 8;
        int active = 5;
        INDArray[] in = deterministicInputs(l);
        INDArray x = in[0];
        INDArray weight = in[1];
        INDArray bias = in[2];
        INDArray stateIn = nonzeroHistory();
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);

        CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) active));
        op.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();

        for (int m = 1; m <= active; m++) {
            for (int kk = 0; kk < KC - 1; kk++) {
                assertEquals(oracleHistory(stateInBackup, x, 0, 1, kk, m),
                        prefix.getFloat(m - 1, 0, 1, kk), 0.0f,
                        "active=" + active + " prefix[" + (m - 1) + "] kk=" + kk);
            }
        }
        // Slots beyond active are not written by this invocation; assert final state
        // equals the legacy shortened run.
        INDArray legacyState = runLegacyState(x, weight, bias, stateInBackup, active);
        assertEquals(0.0, stateOut.sub(legacyState).amaxNumber().doubleValue(), 0.0f,
                "active=" + active + " final state differs from legacy");
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                "stateIn was mutated");
    }

    private static INDArray runLegacyState(INDArray x, INDArray weight, INDArray bias,
                                           INDArray stateIn, int actualLen) {
        INDArray output = Nd4j.create(DataType.FLOAT, B, x.size(1), D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        CausalConv1d op = new CausalConv1d(x, weight, bias, stateIn,
                Nd4j.scalar(DataType.INT64, (long) actualLen), 0, 0);
        op.addOutputArgument(output, stateOut);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();
        return stateOut;
    }

    @Test
    public void testFullOutputsMatchLegacyAndReuseKeepsFreshData() {
        int l = 6;
        INDArray[] in = deterministicInputs(l);
        INDArray x = in[0];
        INDArray weight = in[1];
        INDArray bias = in[2];
        INDArray stateIn = nonzeroHistory();
        INDArray stateInBackup = stateIn.dup();

        INDArray legacyOutput = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray legacyState = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        CausalConv1d legacyOp = new CausalConv1d(x, weight, bias, stateIn,
                Nd4j.scalar(DataType.INT64, (long) l), 0, 0);
        legacyOp.addOutputArgument(legacyOutput, legacyState);
        Nd4j.getExecutioner().exec(legacyOp);
        Nd4j.getExecutioner().commit();

        INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);
        CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l));
        op.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();

        assertEquals(0.0, output.sub(legacyOutput).amaxNumber().doubleValue(), 1e-5f,
                "full activation output differs from legacy");
        assertEquals(0.0, stateOut.sub(legacyState).amaxNumber().doubleValue(), 0.0f,
                "final state differs from legacy");

        // Retained-buffer reuse: rerun with DIFFERENT x through the SAME output
        // arrays; stale prior-run data must not survive.
        INDArray x2 = x.mul(2.0f).addi(0.5f);
        INDArray legacyOutput2 = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray legacyState2 = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        CausalConv1d legacyOp2 = new CausalConv1d(x2, weight, bias, stateInBackup,
                Nd4j.scalar(DataType.INT64, (long) l), 0, 0);
        legacyOp2.addOutputArgument(legacyOutput2, legacyState2);
        Nd4j.getExecutioner().exec(legacyOp2);
        Nd4j.getExecutioner().commit();

        CausalConv1dWithPrefix op2 = new CausalConv1dWithPrefix(
                x2, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l));
        op2.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(op2);
        Nd4j.getExecutioner().commit();

        assertEquals(0.0, output.sub(legacyOutput2).amaxNumber().doubleValue(), 1e-5f,
                "reused-buffer output shows stale data");
        assertEquals(0.0, stateOut.sub(legacyState2).amaxNumber().doubleValue(), 0.0f,
                "reused-buffer state shows stale data");
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                "stateIn mutated across reuse");
    }

    @Test
    public void testSiluActivationMatchesLegacy() {
        int l = 5;
        INDArray[] in = deterministicInputs(l);
        INDArray x = in[0];
        INDArray weight = in[1];
        INDArray bias = in[2];
        INDArray stateIn = nonzeroHistory();
        INDArray stateInBackup = stateIn.dup();

        // Production Qwen3.5 uses SiLU (=1); the companion must honor activation.
        INDArray legacyOutput = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray legacyState = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        CausalConv1d legacyOp = new CausalConv1d(x, weight, bias, stateIn,
                Nd4j.scalar(DataType.INT64, (long) l), 1, 0);
        legacyOp.addOutputArgument(legacyOutput, legacyState);
        Nd4j.getExecutioner().exec(legacyOp);
        Nd4j.getExecutioner().commit();

        INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);
        CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l), 1, 0);
        op.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();

        assertEquals(0.0, output.sub(legacyOutput).amaxNumber().doubleValue(), 1e-5f,
                "SiLU activation output differs from legacy");
        assertEquals(0.0, stateOut.sub(legacyState).amaxNumber().doubleValue(), 0.0f,
                "SiLU final state differs from legacy");
        // Prefix slots are raw-input based, unaffected by activation; slot m-1 must
        // still equal the legacy state at actualLen=m (SiLU on both sides).
        for (int m = 1; m <= l; m++) {
            INDArray lo = Nd4j.create(DataType.FLOAT, B, l, D);
            INDArray ls = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
            CausalConv1d legacyM = new CausalConv1d(x, weight, bias, stateInBackup,
                    Nd4j.scalar(DataType.INT64, (long) m), 1, 0);
            legacyM.addOutputArgument(lo, ls);
            Nd4j.getExecutioner().exec(legacyM);
            Nd4j.getExecutioner().commit();
            INDArray slot = prefix.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            assertEquals(0.0, slot.sub(ls).amaxNumber().doubleValue(), 0.0f,
                    "SiLU prefix[" + (m - 1) + "] differs from legacy actualLen=" + m);
        }
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                "stateIn mutated in SiLU arm");
    }

    @Test
    public void testStateChangingInputsReuseUpdatesPrefixSlots() {
        // Conv history is raw-x based, so an x change changes every live prefix
        // slot. Rerun through the SAME buffers with different x and verify all
        // live slots against an independent reference.
        int l = 5;
        INDArray[] in = deterministicInputs(l);
        INDArray x = in[0];
        INDArray weight = in[1];
        INDArray bias = in[2];
        INDArray stateIn = nonzeroHistory();
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, l, D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        INDArray prefix = Nd4j.create(DataType.FLOAT, l, B, D, KC - 1);

        CausalConv1dWithPrefix first = new CausalConv1dWithPrefix(
                x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l));
        first.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(first);
        Nd4j.getExecutioner().commit();

        INDArray x2 = x.mul(2.0f).addi(0.5f);
        CausalConv1dWithPrefix second = new CausalConv1dWithPrefix(
                x2, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) l));
        second.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(second);
        Nd4j.getExecutioner().commit();

        for (int m = 1; m <= l; m++) {
            INDArray lo = Nd4j.create(DataType.FLOAT, B, l, D);
            INDArray ls = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
            CausalConv1d legacyM = new CausalConv1d(x2, weight, bias, stateInBackup,
                    Nd4j.scalar(DataType.INT64, (long) m), 0, 0);
            legacyM.addOutputArgument(lo, ls);
            Nd4j.getExecutioner().exec(legacyM);
            Nd4j.getExecutioner().commit();
            INDArray slot = prefix.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            assertEquals(0.0, slot.sub(ls).amaxNumber().doubleValue(), 0.0f,
                    "state-changing reuse prefix[" + (m - 1) + "] stale or wrong");
        }
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                "stateIn mutated across state-changing reuse");
    }

    @Test
    public void testFixedWidthActiveLengthReuseShrinksThenGrows() {
        // Fixed-W 5 -> 1 -> 3 active-length reuse: live slots [0,m) must track the
        // current active length; slots beyond it may keep older data but must never
        // be selected.
        int w = 5;
        INDArray[] in = deterministicInputs(w);
        INDArray x = in[0];
        INDArray weight = in[1];
        INDArray bias = in[2];
        INDArray stateIn = nonzeroHistory();
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, w, D);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
        INDArray prefix = Nd4j.create(DataType.FLOAT, w, B, D, KC - 1);

        int[] activeLengths = {5, 1, 3};
        for (int active : activeLengths) {
            CausalConv1dWithPrefix op = new CausalConv1dWithPrefix(
                    x, weight, bias, stateIn, Nd4j.scalar(DataType.INT64, (long) active));
            op.addOutputArgument(output, stateOut, prefix);
            Nd4j.getExecutioner().exec(op);
            Nd4j.getExecutioner().commit();

            for (int m = 1; m <= active; m++) {
                INDArray lo = Nd4j.create(DataType.FLOAT, B, w, D);
                INDArray ls = Nd4j.create(DataType.FLOAT, B, D, KC - 1);
                CausalConv1d legacyM = new CausalConv1d(x, weight, bias, stateInBackup,
                        Nd4j.scalar(DataType.INT64, (long) m), 0, 0);
                legacyM.addOutputArgument(lo, ls);
                Nd4j.getExecutioner().exec(legacyM);
                Nd4j.getExecutioner().commit();
                INDArray slot = prefix.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                assertEquals(0.0, slot.sub(ls).amaxNumber().doubleValue(), 0.0f,
                        "active=" + active + " live prefix[" + (m - 1) + "] stale or wrong");
            }
            assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0f,
                    "stateIn mutated at active=" + active);
        }
    }

    @Test
    public void testCompanionShapeDerivationAndOptimizedExecution() {
        // Packet 01 conv counterpart: deriveRecurrentStateShape must resolve the
        // COMPANION conv op by name, and the optimized graph must execute with the
        // derived state, producing both the ordinary history and the checkpoint.
        int l = 3;
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, B, l, D);
        // Production contract: the conv weight is a graph VARIABLE (sd.var) with a
        // real array; deriveConvStateShapeFromOp reads its shape to get [D, K].
        INDArray weightArr = Nd4j.linspace(1, D * KC, D * KC, DataType.FLOAT)
                .reshape(D, KC).muli(0.01f);
        SDVariable weight = sd.var("weight", weightArr);
        SDVariable stateIn = sd.placeHolder("past_conv_state.5", DataType.FLOAT, B, D, KC - 1);
        SDVariable actualLen = sd.placeHolder("actual_sequence_length", DataType.INT64);

        SDVariable[] out = sd.nn().causalConv1dWithPrefix(
                new String[]{"conv_out_5", "conv_state_out_5", "conv_state_prefix_5"},
                x, weight, null, stateIn, actualLen, 1, 0);
        assertEquals(3, out.length, "companion op must expose output, state, prefix");
        sd.setOutputs("conv_out_5", "conv_state_out_5", "conv_state_prefix_5");

        SameDiff optimized = GraphOptimizer.optimize(
                sd, List.copyOf(sd.outputs()), GraphOptimizer.defaultOptimizations());
        assertNotNull(optimized, "GraphOptimizer must return a graph");

        // Discovery: exactly one conv pair binding the state handoff.
        ModelIOConfig ioConfig = ModelIOConfig.discover(optimized);
        List<ModelIOConfig.RecurrentStatePair> pairs =
                ModelIOConfig.findRecurrentStatePairs(optimized, ioConfig);
        assertEquals(1, pairs.size(),
                "expected one conv recurrent pair, got " + pairs);
        ModelIOConfig.RecurrentStatePair pair = pairs.get(0);
        assertTrue(pair.isConv(), "companion op must classify as conv: " + pair);
        assertEquals("past_conv_state.5", pair.inputName);
        assertEquals("conv_state_out_5", pair.outputName);
        assertEquals("conv_state_prefix_5", pair.prefixOutputName());

        // Shape derivation from the companion op.
        long[] derived = GenerationPipeline.deriveRecurrentStateShape(
                optimized, pair.inputName);
        assertNotNull(derived, "deriveRecurrentStateShape must resolve the companion conv op");
        assertArrayEquals(new long[]{B, D, KC - 1}, derived,
                "derived conv state shape must be [B,D,K-1]");

        // Execute the optimized graph with the derived zero state.
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x", deterministicInputs(l)[0]);
        inputs.put(pair.inputName, Nd4j.zeros(DataType.FLOAT, derived));
        inputs.put("actual_sequence_length", Nd4j.scalar(DataType.INT64, (long) l));
        Map<String, INDArray> results = optimized.output(inputs,
                "conv_state_out_5", "conv_state_prefix_5");
        INDArray stateOut = results.get("conv_state_out_5");
        INDArray prefixOut = results.get("conv_state_prefix_5");
        assertNotNull(stateOut, "optimized graph must produce the ordinary history output");
        assertNotNull(prefixOut, "optimized graph must produce the checkpoint output");
        assertArrayEquals(new long[]{B, D, KC - 1}, stateOut.shape(),
                "ordinary history output shape");
        assertArrayEquals(new long[]{l, B, D, KC - 1}, prefixOut.shape(),
                "checkpoint output shape [W,B,D,K-1]");
        INDArray lastSlot = prefixOut.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(l - 1),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        assertEquals(0.0, stateOut.sub(lastSlot).amaxNumber().doubleValue(), 0.0f,
                "prefix[l-1] must equal the ordinary final history");
        for (INDArray result : results.values()) result.close();
    }
}
