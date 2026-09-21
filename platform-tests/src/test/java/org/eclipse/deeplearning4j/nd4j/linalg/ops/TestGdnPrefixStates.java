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
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRuleWithPrefix;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Prefix-state capture contract for gated_delta_rule_with_prefix.
 *
 * For every live prefix m, checkpoint slot m-1 must equal the final state of an
 * independent legacy execution at actualLen=m from the SAME nonzero initial state,
 * and ordinary outputs must be identical. This is the oracle that lets the MTP
 * controller select prefix[consumed-1] instead of re-executing the target for the
 * consumed prefix.
 */
public class TestGdnPrefixStates {

    private static final int B = 1;
    private static final int H = 2;
    private static final int DK = 8;
    private static final int DV = 8;
    private static final float EPS = 1e-5f;

    private static INDArray[] deterministicInputs(int l, DataType dtype) {
        INDArray q = Nd4j.linspace(1, B * l * H * DK, B * l * H * DK, DataType.FLOAT)
                .reshape(B, l, H, DK).castTo(dtype).muli(0.01f);
        INDArray k = Nd4j.linspace(1, B * l * H * DK, B * l * H * DK, DataType.FLOAT)
                .reshape(B, l, H, DK).castTo(dtype).addi(0.25f).muli(0.01f);
        INDArray v = Nd4j.linspace(1, B * l * H * DV, B * l * H * DV, DataType.FLOAT)
                .reshape(B, l, H, DV).castTo(dtype).muli(0.01f);
        INDArray beta = Nd4j.ones(DataType.FLOAT, B, l, H).muli(0.5f).castTo(dtype);
        INDArray gate = Nd4j.zeros(DataType.FLOAT, B, l, H).subi(0.05f).castTo(dtype);
        return new INDArray[]{q, k, v, beta, gate};
    }

    private static INDArray nonzeroState(DataType dtype) {
        INDArray s = Nd4j.create(dtype, B, H, DK, DV);
        float value = 0.125f;
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                for (int dk = 0; dk < DK; dk++) {
                    for (int dv = 0; dv < DV; dv++) {
                        s.putScalar(new int[]{b, h, dk, dv}, value);
                        value = value * -1.0625f + 0.0625f;
                    }
                }
            }
        }
        return s;
    }

    private static void assertFinite(INDArray a, String name) {
        assertTrue(!a.isNaN().any(), name + " contains NaN");
        assertTrue(!a.isInfinite().any(), name + " contains Inf");
    }

    @Test
    public void testPrefixSlotsMatchLegacyExecutionAtEveryPrefix() {
        for (int w : new int[]{1, 2, 5}) {
            INDArray[] in = deterministicInputs(w, DataType.FLOAT);
            INDArray q = in[0];
            INDArray k = in[1];
            INDArray v = in[2];
            INDArray beta = in[3];
            INDArray gate = in[4];
            INDArray stateIn = nonzeroState(DataType.FLOAT);
            INDArray stateInBackup = stateIn.dup();

            INDArray output = Nd4j.create(DataType.FLOAT, B, w, H, DV);
            INDArray stateOut = Nd4j.create(DataType.FLOAT, B, H, DK, DV);
            INDArray prefix = Nd4j.create(DataType.FLOAT, w, B, H, DK, DV);

            GatedDeltaRuleWithPrefix op = new GatedDeltaRuleWithPrefix(
                    q, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, (long) w));
            op.addOutputArgument(output, stateOut, prefix);
            Nd4j.getExecutioner().exec(op);
            Nd4j.getExecutioner().commit();

            assertFinite(output, "output W=" + w);
            assertFinite(stateOut, "stateOut W=" + w);
            assertFinite(prefix, "prefix W=" + w);

            // Final state/output equal legacy execution at actualLen = w.
            INDArray[] legacyFinal = runLegacy(q, k, v, beta, gate, stateInBackup, w);
            assertEquals(0.0, stateOut.sub(legacyFinal[1]).amaxNumber().doubleValue(), EPS,
                    "W=" + w + " final state differs from legacy actualLen=" + w);

            // Every prefix slot m-1 equals a legacy execution at actualLen=m from
            // the same initial state.
            for (int m = 1; m <= w; m++) {
                INDArray[] legacy = runLegacy(q, k, v, beta, gate, stateInBackup, m);
                INDArray slot = prefix.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                assertEquals(0.0, slot.sub(legacy[1]).amaxNumber().doubleValue(), EPS,
                        "W=" + w + " prefix[" + (m - 1) + "] differs from legacy actualLen=" + m);
            }

            // stateIn was not mutated.
            assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0,
                    "stateIn was mutated by prefix capture, W=" + w);
        }
    }

    @Test
    public void testOptimizerRoundTripPreservesNamesAndPairing() {
        // The production verification graph is built once, then cloned by
        // GraphOptimizer through a SDNB round-trip before any plan is compiled. A
        // companion op that registers name-colliding variables makes that round-trip
        // rename unrelated variables (observed as the recurrent placeholder losing its
        // name and the state input being reported missing at plan execution), so the
        // graph must survive the clone with every semantic name intact and exactly one
        // recurrent pair per layer.
        int l = 3;
        SameDiff sd = SameDiff.create();
        // Production contract: Q/K/V are reshape outputs whose shape argument is a
        // stack [batchDim, seqDim, const(H), const(D)] - exactly what
        // resolveReshapeHeadDims/tryExtractStackConstants walk backwards through.
        // Flat placeholders mirror the QKV projection output before the head reshape.
        SDVariable qFlat = sd.placeHolder("q_flat", DataType.FLOAT, B, l, H * DK);
        SDVariable kFlat = sd.placeHolder("k_flat", DataType.FLOAT, B, l, H * DK);
        SDVariable vFlat = sd.placeHolder("v_flat", DataType.FLOAT, B, l, H * DV);
        SDVariable beta = sd.placeHolder("beta", DataType.FLOAT, B, l, H);
        SDVariable gate = sd.placeHolder("gate", DataType.FLOAT, B, l, H);
        SDVariable stateIn = sd.placeHolder("past_gdn_state.7", DataType.FLOAT, B, H, DK, DV);
        SDVariable actualLen = sd.placeHolder("actual_sequence_length", DataType.INT64);

        SDVariable batchDim = sd.sizeAt(qFlat, 0);
        SDVariable seqDim = sd.sizeAt(qFlat, 1);
        SDVariable q = sd.reshape("q", qFlat, sd.stack("q_head_shape", 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) H)), sd.constant(Nd4j.scalar((long) DK))));
        SDVariable k = sd.reshape("k", kFlat, sd.stack("k_head_shape", 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) H)), sd.constant(Nd4j.scalar((long) DK))));
        SDVariable v = sd.reshape("v", vFlat, sd.stack("v_head_shape", 0,
                batchDim, seqDim,
                sd.constant(Nd4j.scalar((long) H)), sd.constant(Nd4j.scalar((long) DV))));

        SDVariable[] out = sd.nn().gatedDeltaRuleWithPrefix(
                new String[]{"gdn_out_7", "gdn_state_out_7", "gdn_state_prefix_7"},
                q, k, v, beta, gate, stateIn, actualLen);
        assertEquals(3, out.length, "companion op must expose output, state, prefix");
        sd.setOutputs("gdn_out_7", "gdn_state_out_7", "gdn_state_prefix_7");

        SameDiff optimized = GraphOptimizer.optimize(
                sd, List.copyOf(sd.outputs()), GraphOptimizer.defaultOptimizations());
        assertNotNull(optimized, "GraphOptimizer must return a graph");

        // Every semantic name survives the round-trip: the recurrent placeholder (the
        // one the state commit writes back to) and all three op outputs.
        for (String name : new String[]{"past_gdn_state.7", "gdn_out_7", "gdn_state_out_7",
                "gdn_state_prefix_7"}) {
            assertNotNull(optimized.getVariable(name),
                    "optimizer round-trip lost variable '" + name + "'");
        }

        // Exactly one recurrent pair, and it is the state handoff - never the
        // per-timestep checkpoint.
        ModelIOConfig ioConfig = ModelIOConfig.discover(optimized);
        List<ModelIOConfig.RecurrentStatePair> pairs =
                ModelIOConfig.findRecurrentStatePairs(optimized, ioConfig);
        assertEquals(1, pairs.size(),
                "expected one recurrent pair for one layer, got " + pairs);
        ModelIOConfig.RecurrentStatePair pair = pairs.get(0);
        assertEquals("past_gdn_state.7", pair.inputName,
                "recurrent pair must bind the state placeholder");
        assertEquals("gdn_state_out_7", pair.outputName,
                "recurrent pair must bind the state handoff, not the checkpoint");
        assertTrue(pair.isGdn(), "companion op must classify as a GDN state: " + pair);
        assertTrue(pair.hasPrefixCapture(), "companion op must report prefix capture: " + pair);
        assertEquals("gdn_state_prefix_7", pair.prefixOutputName(),
                "prefix binding must name the checkpoint output");

        // Packet 01: the production input-map builder must derive the state shape
        // from the COMPANION op (name-based dispatch), and the optimized graph must
        // execute with the derived state plus all required inputs, returning both
        // the ordinary state handoff and the checkpoint.
        long[] derived = GenerationPipeline.deriveRecurrentStateShape(
                optimized, pair.inputName);
        assertNotNull(derived, "deriveRecurrentStateShape must resolve the companion GDN op");
        assertArrayEquals(new long[]{B, H, DK, DV}, derived,
                "derived GDN state shape must be [B,H,Dk,Dv]");

        // Execute the optimized graph with the derived zero state and request both
        // the ordinary state output and the prefix checkpoint.
        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("q_flat", deterministicInputs(l, DataType.FLOAT)[0].reshape(B, l, H * DK));
        inputs.put("k_flat", deterministicInputs(l, DataType.FLOAT)[1].reshape(B, l, H * DK));
        inputs.put("v_flat", deterministicInputs(l, DataType.FLOAT)[2].reshape(B, l, H * DV));
        inputs.put("beta", deterministicInputs(l, DataType.FLOAT)[3]);
        inputs.put("gate", deterministicInputs(l, DataType.FLOAT)[4]);
        inputs.put(pair.inputName, Nd4j.zeros(DataType.FLOAT, derived));
        inputs.put("actual_sequence_length", Nd4j.scalar(DataType.INT64, (long) l));
        Map<String, INDArray> results = optimized.output(inputs,
                "gdn_state_out_7", "gdn_state_prefix_7");
        INDArray stateOut = results.get("gdn_state_out_7");
        INDArray prefixOut = results.get("gdn_state_prefix_7");
        assertNotNull(stateOut, "optimized graph must produce the ordinary state output");
        assertNotNull(prefixOut, "optimized graph must produce the checkpoint output");
        assertArrayEquals(new long[]{B, H, DK, DV}, stateOut.shape(),
                "ordinary state output shape");
        assertArrayEquals(new long[]{l, B, H, DK, DV}, prefixOut.shape(),
                "checkpoint output shape [W,B,H,Dk,Dv]");
        // The final checkpoint slot equals the ordinary final state: both are the
        // state after consuming all l rows.
        INDArray lastSlot = prefixOut.get(
                org.nd4j.linalg.indexing.NDArrayIndex.point(l - 1),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all());
        assertEquals(0.0, stateOut.sub(lastSlot).amaxNumber().doubleValue(), EPS,
                "prefix[l-1] must equal the ordinary final state");
        for (INDArray result : results.values()) result.close();
    }

    @Test
    public void testShortenedActualLenMatchesLegacyAtThatLength() {
        int w = 5;
        int active = 3;
        INDArray[] in = deterministicInputs(w, DataType.FLOAT);
        INDArray q = in[0];
        INDArray k = in[1];
        INDArray v = in[2];
        INDArray beta = in[3];
        INDArray gate = in[4];
        INDArray stateIn = nonzeroState(DataType.FLOAT);
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, w, H, DV);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, H, DK, DV);
        INDArray prefix = Nd4j.create(DataType.FLOAT, w, B, H, DK, DV);

        GatedDeltaRuleWithPrefix op = new GatedDeltaRuleWithPrefix(
                q, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, (long) active));
        op.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();

        INDArray[] legacy = runLegacy(q, k, v, beta, gate, stateInBackup, active);
        for (int m = 1; m <= active; m++) {
            INDArray[] legacyM = runLegacy(q, k, v, beta, gate, stateInBackup, m);
            INDArray slot = prefix.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            assertEquals(0.0, slot.sub(legacyM[1]).amaxNumber().doubleValue(), EPS,
                    "active=" + active + " prefix[" + (m - 1) + "] differs from legacy actualLen=" + m);
        }
        assertEquals(0.0, stateOut.sub(legacy[1]).amaxNumber().doubleValue(), EPS,
                "active=" + active + " final state differs from legacy");
        // Full ordinary output at the active length must equal legacy too.
        assertEquals(0.0, output.sub(legacy[0]).amaxNumber().doubleValue(), EPS,
                "active=" + active + " full activation output differs from legacy");
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0,
                "stateIn was mutated");
    }

    @Test
    public void testRetainedBufferReuseStaysFresh() {
        // Two back-to-back invocations through the SAME output arrays with different
        // inputs; the second result must match an independent legacy execution.
        int w = 4;
        INDArray[] in = deterministicInputs(w, DataType.FLOAT);
        INDArray q = in[0];
        INDArray k = in[1];
        INDArray v = in[2];
        INDArray beta = in[3];
        INDArray gate = in[4];
        INDArray stateIn = nonzeroState(DataType.FLOAT);
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, w, H, DV);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, H, DK, DV);
        INDArray prefix = Nd4j.create(DataType.FLOAT, w, B, H, DK, DV);

        // First run with q/k/v, then rerun with doubled q through the same buffers.
        GatedDeltaRuleWithPrefix first = new GatedDeltaRuleWithPrefix(
                q, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, (long) w));
        first.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(first);
        Nd4j.getExecutioner().commit();

        INDArray q2 = q.mul(2.0f);
        INDArray[] legacy2 = runLegacy(q2, k, v, beta, gate, stateInBackup, w);
        GatedDeltaRuleWithPrefix second = new GatedDeltaRuleWithPrefix(
                q2, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, (long) w));
        second.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(second);
        Nd4j.getExecutioner().commit();

        assertEquals(0.0, output.sub(legacy2[0]).amaxNumber().doubleValue(), EPS,
                "reused-buffer output shows stale data");
        assertEquals(0.0, stateOut.sub(legacy2[1]).amaxNumber().doubleValue(), EPS,
                "reused-buffer state shows stale data");
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0,
                "stateIn mutated across reuse");
    }

    @Test
    public void testStateChangingInputsReuseUpdatesPrefixSlots() {
        // Q alone does not change recurrent state. V/K/gate DO. Rerun through the
        // same buffers with a state-changing input and verify ALL live prefix slots
        // reflect the new inputs (no stale slots from the previous invocation).
        int w = 4;
        INDArray[] in = deterministicInputs(w, DataType.FLOAT);
        INDArray q = in[0];
        INDArray k = in[1];
        INDArray v = in[2];
        INDArray beta = in[3];
        INDArray gate = in[4];
        INDArray stateIn = nonzeroState(DataType.FLOAT);
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, w, H, DV);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, H, DK, DV);
        INDArray prefix = Nd4j.create(DataType.FLOAT, w, B, H, DK, DV);

        GatedDeltaRuleWithPrefix first = new GatedDeltaRuleWithPrefix(
                q, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, (long) w));
        first.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(first);
        Nd4j.getExecutioner().commit();

        // State-changing change: V and K (and gate) shift, Q unchanged.
        INDArray v2 = v.mul(3.0f).addi(0.25f);
        INDArray k2 = k.mul(-0.5f).addi(0.125f);
        INDArray gate2 = gate.addi(-0.02f);

        GatedDeltaRuleWithPrefix second = new GatedDeltaRuleWithPrefix(
                q, k2, v2, beta, gate2, stateIn, Nd4j.scalar(DataType.INT64, (long) w));
        second.addOutputArgument(output, stateOut, prefix);
        Nd4j.getExecutioner().exec(second);
        Nd4j.getExecutioner().commit();

        // Independent reference from the ORIGINAL state with the NEW inputs.
        INDArray[] legacy2 = runLegacy(q, k2, v2, beta, gate2, stateInBackup, w);
        for (int m = 1; m <= w; m++) {
            INDArray[] legacyM = runLegacy(q, k2, v2, beta, gate2, stateInBackup, m);
            INDArray slot = prefix.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            assertEquals(0.0, slot.sub(legacyM[1]).amaxNumber().doubleValue(), EPS,
                    "state-changing reuse prefix[" + (m - 1) + "] stale or wrong");
        }
        assertEquals(0.0, output.sub(legacy2[0]).amaxNumber().doubleValue(), EPS,
                "state-changing reuse output differs from legacy");
        assertEquals(0.0, stateOut.sub(legacy2[1]).amaxNumber().doubleValue(), EPS,
                "state-changing reuse final state differs from legacy");
        assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0,
                "stateIn mutated across state-changing reuse");
    }

    @Test
    public void testFixedWidthActiveLengthReuseShrinksThenGrows() {
        // Fixed-W 5 -> 1 -> 3 active-length reuse: live slots [0,m) must track the
        // CURRENT active length each call; slots beyond the active length may keep
        // older data but must never be selected by the controller.
        int w = 5;
        INDArray[] in = deterministicInputs(w, DataType.FLOAT);
        INDArray q = in[0];
        INDArray k = in[1];
        INDArray v = in[2];
        INDArray beta = in[3];
        INDArray gate = in[4];
        INDArray stateIn = nonzeroState(DataType.FLOAT);
        INDArray stateInBackup = stateIn.dup();

        INDArray output = Nd4j.create(DataType.FLOAT, B, w, H, DV);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, H, DK, DV);
        INDArray prefix = Nd4j.create(DataType.FLOAT, w, B, H, DK, DV);

        int[] activeLengths = {5, 1, 3};
        for (int active : activeLengths) {
            GatedDeltaRuleWithPrefix op = new GatedDeltaRuleWithPrefix(
                    q, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, (long) active));
            op.addOutputArgument(output, stateOut, prefix);
            Nd4j.getExecutioner().exec(op);
            Nd4j.getExecutioner().commit();

            for (int m = 1; m <= active; m++) {
                INDArray[] legacyM = runLegacy(q, k, v, beta, gate, stateInBackup, m);
                INDArray slot = prefix.get(
                        org.nd4j.linalg.indexing.NDArrayIndex.point(m - 1),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all(),
                        org.nd4j.linalg.indexing.NDArrayIndex.all());
                assertEquals(0.0, slot.sub(legacyM[1]).amaxNumber().doubleValue(), EPS,
                        "active=" + active + " live prefix[" + (m - 1) + "] stale or wrong");
            }
            assertEquals(0.0, stateIn.sub(stateInBackup).amaxNumber().doubleValue(), 0.0,
                    "stateIn mutated at active=" + active);
        }
    }

    private static INDArray[] runLegacy(INDArray q, INDArray k, INDArray v,
                                        INDArray beta, INDArray gate,
                                        INDArray stateIn, int actualLen) {
        INDArray output = Nd4j.create(DataType.FLOAT, B, q.size(1), H, DV);
        INDArray stateOut = Nd4j.create(DataType.FLOAT, B, H, DK, DV);
        GatedDeltaRule op = new GatedDeltaRule(q, k, v, beta, gate, stateIn,
                Nd4j.scalar(DataType.INT64, (long) actualLen));
        op.addOutputArgument(output, stateOut);
        Nd4j.getExecutioner().exec(op);
        Nd4j.getExecutioner().commit();
        return new INDArray[]{output, stateOut};
    }
}
