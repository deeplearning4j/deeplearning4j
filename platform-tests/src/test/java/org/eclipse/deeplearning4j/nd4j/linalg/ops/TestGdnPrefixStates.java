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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRuleWithPrefix;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
