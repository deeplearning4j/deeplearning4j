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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Tests for the gated_delta_rule op (arXiv:2412.06464).
 *
 * gated_delta_rule implements:
 *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
 *   output_t = S_t^T * q_t
 */
public class TestGatedDeltaRule {

    @Test
    public void testBasicShapesNoState() {
        int B = 2, L = 4, H = 3, Dk = 8, Dv = 8;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.5);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));

        assertEquals(2, result.length, "Should produce 2 outputs");
        assertArrayEquals(new long[]{B, L, H, Dv}, result[0].shape(), "Output shape");
        assertArrayEquals(new long[]{B, H, Dk, Dv}, result[1].shape(), "State shape");
        assertFalse(result[0].isNaN().any(), "Output contains NaN");
        assertFalse(result[0].isInfinite().any(), "Output contains Inf");
        assertFalse(result[1].isNaN().any(), "State contains NaN");
    }

    @Test
    public void testBasicShapesWithState() {
        int B = 1, L = 3, H = 2, Dk = 4, Dv = 6;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.5);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate, stateIn));

        assertEquals(2, result.length);
        assertArrayEquals(new long[]{B, L, H, Dv}, result[0].shape(), "Output shape with state");
        assertArrayEquals(new long[]{B, H, Dk, Dv}, result[1].shape(), "State out shape");
        assertFalse(result[0].isNaN().any());
    }

    @Test
    public void testZeroGateDecaysState() {
        // gate=0 means exp(0)=1, so state is NOT decayed (full retention)
        int B = 1, L = 1, H = 1, Dk = 4, Dv = 4;
        INDArray q = Nd4j.ones(DataType.FLOAT, B, L, H, Dk);
        INDArray k = Nd4j.ones(DataType.FLOAT, B, L, H, Dk);
        INDArray v = Nd4j.ones(DataType.FLOAT, B, L, H, Dv);
        INDArray beta = Nd4j.ones(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.zeros(DataType.FLOAT, B, L, H);  // exp(0) = 1
        INDArray stateIn = Nd4j.ones(DataType.FLOAT, B, H, Dk, Dv).muli(0.5);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate, stateIn));
        assertFalse(result[0].isNaN().any(), "Zero-gate output NaN");
        assertFalse(result[1].isNaN().any(), "Zero-gate state NaN");

        // State should not be all zeros since gate=0 means full retention
        double stateMax = result[1].amaxNumber().doubleValue();
        assertTrue(stateMax > 0, "State should be non-zero with gate=0 (full retention)");
    }

    @Test
    public void testSequentialStateChaining() {
        // Run two steps: step1 produces stateOut, step2 uses it as stateIn
        int B = 1, L = 2, H = 2, Dk = 4, Dv = 4;
        INDArray q1 = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k1 = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v1 = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta1 = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate1 = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);

        INDArray[] step1 = Nd4j.exec(new GatedDeltaRule(q1, k1, v1, beta1, gate1));
        INDArray stateAfterStep1 = step1[1];

        // Step 2 with different inputs but chained state
        INDArray q2 = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k2 = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v2 = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta2 = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate2 = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);

        INDArray[] step2 = Nd4j.exec(new GatedDeltaRule(q2, k2, v2, beta2, gate2, stateAfterStep1));

        assertFalse(step2[0].isNaN().any(), "Chained step2 output NaN");
        assertFalse(step2[1].isNaN().any(), "Chained step2 state NaN");

        // Step 2 with state should differ from step 2 without state
        INDArray[] step2NoState = Nd4j.exec(new GatedDeltaRule(q2, k2, v2, beta2, gate2));
        double diff = step2[0].sub(step2NoState[0]).amaxNumber().doubleValue();
        assertTrue(diff > 1e-6, "Chained state should change output, diff=" + diff);
    }

    @Test
    public void testTwoTokenWindowMatchesChainedSingleTokenSteps() {
        Nd4j.getRandom().setSeed(12345);
        int B = 1, L = 2, H = 2, Dk = 8, Dv = 8;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01);

        INDArray[] window = Nd4j.exec(new GatedDeltaRule(
                q, k, v, beta, gate, stateIn, Nd4j.scalar(DataType.INT64, 2L)));

        INDArray[] first = Nd4j.exec(new GatedDeltaRule(
                q.get(all(), interval(0, 1), all(), all()).dup(),
                k.get(all(), interval(0, 1), all(), all()).dup(),
                v.get(all(), interval(0, 1), all(), all()).dup(),
                beta.get(all(), interval(0, 1), all()).dup(),
                gate.get(all(), interval(0, 1), all()).dup(),
                stateIn.dup(), Nd4j.scalar(DataType.INT64, 1L)));
        INDArray[] second = Nd4j.exec(new GatedDeltaRule(
                q.get(all(), interval(1, 2), all(), all()).dup(),
                k.get(all(), interval(1, 2), all(), all()).dup(),
                v.get(all(), interval(1, 2), all(), all()).dup(),
                beta.get(all(), interval(1, 2), all()).dup(),
                gate.get(all(), interval(1, 2), all()).dup(),
                first[1], Nd4j.scalar(DataType.INT64, 1L)));

        assertEquals(0.0, window[0].get(all(), interval(0, 1), all(), all())
                .sub(first[0]).amaxNumber().doubleValue(), 1e-5,
                "W=2 row 0 must match the first W=1 call");
        assertEquals(0.0, window[0].get(all(), interval(1, 2), all(), all())
                .sub(second[0]).amaxNumber().doubleValue(), 1e-5,
                "W=2 row 1 must match the chained W=1 call");
        assertEquals(0.0, window[1].sub(second[1]).amaxNumber().doubleValue(), 1e-5,
                "W=2 final state must match two chained W=1 calls");
    }

    @Test
    public void testFiveTokenWindowExactlyMatchesChainedSingleTokenStepsAtQwenDimensions() {
        Nd4j.getRandom().setSeed(34567);
        int B = 1, L = 5, H = 16, Dk = 128, Dv = 128;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01);

        INDArray[] window = Nd4j.exec(new GatedDeltaRule(
                q, k, v, beta, gate, stateIn.dup(), Nd4j.scalar(DataType.INT64, L)));

        INDArray chainedState = stateIn.dup();
        for (int t = 0; t < L; t++) {
            INDArray[] scalar = Nd4j.exec(new GatedDeltaRule(
                    q.get(all(), interval(t, t + 1), all(), all()).dup(),
                    k.get(all(), interval(t, t + 1), all(), all()).dup(),
                    v.get(all(), interval(t, t + 1), all(), all()).dup(),
                    beta.get(all(), interval(t, t + 1), all()).dup(),
                    gate.get(all(), interval(t, t + 1), all()).dup(),
                    chainedState, Nd4j.scalar(DataType.INT64, 1L)));
            INDArray windowRow = window[0].get(
                    all(), interval(t, t + 1), all(), all()).dup();
            double maxDiff = windowRow.sub(scalar[0]).amaxNumber().doubleValue();
            double l1Diff = windowRow.sub(scalar[0]).norm1Number().doubleValue();
            assertEquals(0.0, maxDiff, 0.0,
                    "W=5 GDN row " + t + " differs from chained W=1: maxDiff="
                            + maxDiff + ", l1Diff=" + l1Diff);
            chainedState = scalar[1];
        }

        double stateMaxDiff = window[1].sub(chainedState).amaxNumber().doubleValue();
        double stateL1Diff = window[1].sub(chainedState).norm1Number().doubleValue();
        assertEquals(0.0, stateMaxDiff, 0.0,
                "W=5 GDN final state differs from chained W=1: maxDiff="
                        + stateMaxDiff + ", l1Diff=" + stateL1Diff);
    }

    @Test
    public void testFrozenFiveTokenBufferActivePrefixExactlyMatchesChainedStepsAtQwenDimensions() {
        Nd4j.getRandom().setSeed(56789);
        int B = 1, L = 5, H = 16, Dk = 128, Dv = 128;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01);

        for (int activeLength : new int[]{1, 2}) {
            INDArray[] window = Nd4j.exec(new GatedDeltaRule(
                    q, k, v, beta, gate, stateIn.dup(),
                    Nd4j.scalar(DataType.INT64, activeLength)));

            INDArray chainedState = stateIn.dup();
            for (int t = 0; t < activeLength; t++) {
                INDArray[] scalar = Nd4j.exec(new GatedDeltaRule(
                        q.get(all(), interval(t, t + 1), all(), all()).dup(),
                        k.get(all(), interval(t, t + 1), all(), all()).dup(),
                        v.get(all(), interval(t, t + 1), all(), all()).dup(),
                        beta.get(all(), interval(t, t + 1), all()).dup(),
                        gate.get(all(), interval(t, t + 1), all()).dup(),
                        chainedState, Nd4j.scalar(DataType.INT64, 1L)));
                INDArray windowRow = window[0].get(
                        all(), interval(t, t + 1), all(), all()).dup();
                double maxDiff = windowRow.sub(scalar[0]).amaxNumber().doubleValue();
                double l1Diff = windowRow.sub(scalar[0]).norm1Number().doubleValue();
                assertEquals(0.0, maxDiff, 0.0,
                        "Frozen W=5 GDN activeLength=" + activeLength + " row=" + t
                                + " differs from chained W=1: maxDiff=" + maxDiff
                                + ", l1Diff=" + l1Diff);
                chainedState = scalar[1];
            }

            double stateMaxDiff = window[1].sub(chainedState).amaxNumber().doubleValue();
            double stateL1Diff = window[1].sub(chainedState).norm1Number().doubleValue();
            assertEquals(0.0, stateMaxDiff, 0.0,
                    "Frozen W=5 GDN activeLength=" + activeLength
                            + " final state differs from chained W=1: maxDiff="
                            + stateMaxDiff + ", l1Diff=" + stateL1Diff);
        }
    }

    @Test
    public void testAsymmetricDimensions() {
        // D_k != D_v
        int B = 1, L = 3, H = 2, Dk = 8, Dv = 16;
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.5);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));

        assertArrayEquals(new long[]{B, L, H, Dv}, result[0].shape(), "Output Dv dimension");
        assertArrayEquals(new long[]{B, H, Dk, Dv}, result[1].shape(), "State [Dk, Dv]");
    }

    @Test
    public void testDoubleType() {
        int B = 1, L = 2, H = 1, Dk = 4, Dv = 4;
        INDArray q = Nd4j.randn(DataType.DOUBLE, B, L, H, Dk).muli(0.1);
        INDArray k = Nd4j.randn(DataType.DOUBLE, B, L, H, Dk).muli(0.1);
        INDArray v = Nd4j.randn(DataType.DOUBLE, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.DOUBLE, B, L, H);
        INDArray gate = Nd4j.randn(DataType.DOUBLE, B, L, H).muli(0.5);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));
        assertEquals(DataType.DOUBLE, result[0].dataType());
        assertFalse(result[0].isNaN().any());
    }

    /**
     * Numerical accuracy test with hand-computable values.
     * B=1, L=1, H=1, Dk=2, Dv=2, stateIn=zeros.
     *
     * Algorithm (single timestep, no prior state):
     *   prediction[dv] = sum_dk(S[dk,dv] * k[dk]) = 0 (state is zero)
     *   delta[dv] = v[dv] - exp(g) * prediction[dv] = v[dv]
     *   S_new[dk,dv] = exp(g)*S[dk,dv] + beta*k[dk]*delta[dv] = beta*k[dk]*v[dv]
     *   output[dv] = sum_dk(S_new[dk,dv] * q[dk]) = beta * (q dot k) * v[dv]
     *
     * With q=[1,2], k=[3,4], v=[5,6], beta=0.5, gate=0 (exp(0)=1):
     *   q dot k = 1*3 + 2*4 = 11
     *   output[0] = 0.5 * 11 * 5 = 27.5
     *   output[1] = 0.5 * 11 * 6 = 33.0
     *   state[dk=0,dv=0] = 0.5 * 3 * 5 = 7.5
     *   state[dk=0,dv=1] = 0.5 * 3 * 6 = 9.0
     *   state[dk=1,dv=0] = 0.5 * 4 * 5 = 10.0
     *   state[dk=1,dv=1] = 0.5 * 4 * 6 = 12.0
     */
    @Test
    public void testNumericalAccuracyMinimal() {
        int B = 1, L = 1, H = 1, Dk = 2, Dv = 2;
        INDArray q = Nd4j.createFromArray(new float[]{1, 2}).reshape(B, L, H, Dk);
        INDArray k = Nd4j.createFromArray(new float[]{3, 4}).reshape(B, L, H, Dk);
        INDArray v = Nd4j.createFromArray(new float[]{5, 6}).reshape(B, L, H, Dv);
        INDArray beta = Nd4j.createFromArray(new float[]{0.5f}).reshape(B, L, H);
        INDArray gate = Nd4j.zeros(DataType.FLOAT, B, L, H);  // exp(0)=1

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));
        INDArray output = result[0];
        INDArray stateOut = result[1];

        // Expected output: beta * (q.k) * v = 0.5 * 11 * [5, 6] = [27.5, 33.0]
        float out0 = output.getFloat(0, 0, 0, 0);
        float out1 = output.getFloat(0, 0, 0, 1);
        System.out.println("[GDR accuracy] output[0]=" + out0 + " expected=27.5");
        System.out.println("[GDR accuracy] output[1]=" + out1 + " expected=33.0");
        assertEquals(27.5f, out0, 1e-4f, "output[0] = beta*(q.k)*v[0]");
        assertEquals(33.0f, out1, 1e-4f, "output[1] = beta*(q.k)*v[1]");

        // Expected state: beta * k[dk] * v[dv] = 0.5 * k (x) v
        // stateOut shape = [B, H, Dk, Dv] = [1, 1, 2, 2]
        float s00 = stateOut.getFloat(0, 0, 0, 0);  // 0.5*3*5 = 7.5
        float s01 = stateOut.getFloat(0, 0, 0, 1);  // 0.5*3*6 = 9.0
        float s10 = stateOut.getFloat(0, 0, 1, 0);  // 0.5*4*5 = 10.0
        float s11 = stateOut.getFloat(0, 0, 1, 1);  // 0.5*4*6 = 12.0
        System.out.println("[GDR accuracy] state[0,0]=" + s00 + " expected=7.5");
        System.out.println("[GDR accuracy] state[0,1]=" + s01 + " expected=9.0");
        System.out.println("[GDR accuracy] state[1,0]=" + s10 + " expected=10.0");
        System.out.println("[GDR accuracy] state[1,1]=" + s11 + " expected=12.0");
        assertEquals(7.5f, s00, 1e-4f, "state[0,0]");
        assertEquals(9.0f, s01, 1e-4f, "state[0,1]");
        assertEquals(10.0f, s10, 1e-4f, "state[1,0]");
        assertEquals(12.0f, s11, 1e-4f, "state[1,1]");
    }

    /**
     * Multi-timestep accuracy test with state propagation.
     * B=1, L=2, H=1, Dk=2, Dv=2, stateIn=zeros, gate=0 (exp(0)=1)
     *
     * Timestep 0: same as testNumericalAccuracyMinimal
     *   q=[1,2], k=[3,4], v=[5,6], beta=1.0
     *   prediction = 0 (state zero)
     *   delta = v = [5,6]
     *   S_new[dk,dv] = 1*0 + 1*k[dk]*delta[dv] = k[dk]*v[dv]
     *     S = [[15, 18], [20, 24]]  (k=3,4 x v=5,6)
     *   output[dv] = sum_dk(S[dk,dv]*q[dk]) = [15*1+20*2, 18*1+24*2] = [55, 66]
     *
     * Timestep 1: q=[1,0], k=[1,0], v=[0,0], beta=1.0, gate=0
     *   prediction[dv] = sum_dk(S[dk,dv]*k[dk]) = S[0,dv]*1 + S[1,dv]*0 = [15, 18]
     *   delta[dv] = v[dv] - 1*prediction[dv] = [0-15, 0-18] = [-15, -18]
     *   S_new[dk,dv] = 1*S[dk,dv] + 1*k[dk]*delta[dv]
     *     S_new[0,0] = 15 + 1*(-15) = 0
     *     S_new[0,1] = 18 + 1*(-18) = 0
     *     S_new[1,0] = 20 + 0*(-15) = 20
     *     S_new[1,1] = 24 + 0*(-18) = 24
     *   output[dv] = sum_dk(S_new[dk,dv]*q[dk]) = [0*1+20*1, 0*1+24*0]
     *     Ugh wait, q=[1,0]: output[0] = S[0,0]*1+S[1,0]*0 = 0, output[1] = S[0,1]*1+S[1,1]*0 = 0
     *   Actually: output[dv] = sum_dk(S_new[dk,dv]*q[dk])
     *     output[0] = S_new[0,0]*q[0] + S_new[1,0]*q[1] = 0*1 + 20*0 = 0
     *     output[1] = S_new[0,1]*q[0] + S_new[1,1]*q[1] = 0*1 + 24*0 = 0
     */
    @Test
    public void testNumericalAccuracyMultiStep() {
        int B = 1, L = 2, H = 1, Dk = 2, Dv = 2;
        // Timestep 0: q=[1,2], k=[3,4], v=[5,6]; Timestep 1: q=[1,0], k=[1,0], v=[0,0]
        INDArray q = Nd4j.createFromArray(new float[]{1, 2, 1, 0}).reshape(B, L, H, Dk);
        INDArray k = Nd4j.createFromArray(new float[]{3, 4, 1, 0}).reshape(B, L, H, Dk);
        INDArray v = Nd4j.createFromArray(new float[]{5, 6, 0, 0}).reshape(B, L, H, Dv);
        INDArray beta = Nd4j.ones(DataType.FLOAT, B, L, H);    // beta=1
        INDArray gate = Nd4j.zeros(DataType.FLOAT, B, L, H);   // exp(0)=1

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));
        INDArray output = result[0];
        INDArray stateOut = result[1];

        // Timestep 0 output: [55, 66]
        float out_t0_0 = output.getFloat(0, 0, 0, 0);
        float out_t0_1 = output.getFloat(0, 0, 0, 1);
        System.out.println("[GDR multi] t=0 output=[" + out_t0_0 + ", " + out_t0_1 + "] expected=[55, 66]");
        assertEquals(55.0f, out_t0_0, 1e-3f, "t=0 output[0]");
        assertEquals(66.0f, out_t0_1, 1e-3f, "t=0 output[1]");

        // Timestep 1 output: [0, 0]
        float out_t1_0 = output.getFloat(0, 1, 0, 0);
        float out_t1_1 = output.getFloat(0, 1, 0, 1);
        System.out.println("[GDR multi] t=1 output=[" + out_t1_0 + ", " + out_t1_1 + "] expected=[0, 0]");
        assertEquals(0.0f, out_t1_0, 1e-3f, "t=1 output[0]");
        assertEquals(0.0f, out_t1_1, 1e-3f, "t=1 output[1]");

        // Final state: [[0,0],[20,24]]
        float s00 = stateOut.getFloat(0, 0, 0, 0);
        float s01 = stateOut.getFloat(0, 0, 0, 1);
        float s10 = stateOut.getFloat(0, 0, 1, 0);
        float s11 = stateOut.getFloat(0, 0, 1, 1);
        System.out.println("[GDR multi] stateOut=[[" + s00 + "," + s01 + "],[" + s10 + "," + s11 + "]] expected=[[0,0],[20,24]]");
        assertEquals(0.0f, s00, 1e-3f, "final state[0,0]");
        assertEquals(0.0f, s01, 1e-3f, "final state[0,1]");
        assertEquals(20.0f, s10, 1e-3f, "final state[1,0]");
        assertEquals(24.0f, s11, 1e-3f, "final state[1,1]");
    }

    @Test
    public void testActualLengthStopsStateUpdates() {
        int B = 1, L = 4, H = 1, Dk = 2, Dv = 2;
        INDArray qPadded = Nd4j.createFromArray(new float[]{
                1, 2, 1, 0, 9, 9, 8, 8
        }).reshape(B, L, H, Dk);
        INDArray kPadded = Nd4j.createFromArray(new float[]{
                3, 4, 1, 0, 7, 7, 6, 6
        }).reshape(B, L, H, Dk);
        INDArray vPadded = Nd4j.createFromArray(new float[]{
                5, 6, 0, 0, 5, 5, 4, 4
        }).reshape(B, L, H, Dv);
        INDArray betaPadded = Nd4j.ones(DataType.FLOAT, B, L, H);
        INDArray gatePadded = Nd4j.zeros(DataType.FLOAT, B, L, H);
        INDArray actualLen = Nd4j.scalar(DataType.INT64, 2L);

        INDArray[] padded = Nd4j.exec(new GatedDeltaRule(qPadded, kPadded, vPadded, betaPadded, gatePadded, null, actualLen));

        INDArray qShort = qPadded.get(point(0), interval(0, 2), all(), all()).dup().reshape(B, 2, H, Dk);
        INDArray kShort = kPadded.get(point(0), interval(0, 2), all(), all()).dup().reshape(B, 2, H, Dk);
        INDArray vShort = vPadded.get(point(0), interval(0, 2), all(), all()).dup().reshape(B, 2, H, Dv);
        INDArray betaShort = betaPadded.get(point(0), interval(0, 2), all()).dup().reshape(B, 2, H);
        INDArray gateShort = gatePadded.get(point(0), interval(0, 2), all()).dup().reshape(B, 2, H);
        INDArray[] shortRun = Nd4j.exec(new GatedDeltaRule(qShort, kShort, vShort, betaShort, gateShort));

        assertEquals(0.0, padded[1].sub(shortRun[1]).amaxNumber().doubleValue(), 1e-5, "state should ignore padded timesteps");
        assertEquals(0.0, padded[0].get(point(0), interval(0, 2), all(), all()).sub(shortRun[0]).amaxNumber().doubleValue(), 1e-5,
                "real prefix outputs should match an unpadded run");
    }

    @Test
    public void testActualLengthStopsStateUpdatesWithDsp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        int B = 1, L = 4, H = 1, Dk = 2, Dv = 2;
        INDArray qPadded = Nd4j.createFromArray(new float[]{
                1, 2, 1, 0, 9, 9, 8, 8
        }).reshape(B, L, H, Dk);
        INDArray kPadded = Nd4j.createFromArray(new float[]{
                3, 4, 1, 0, 7, 7, 6, 6
        }).reshape(B, L, H, Dk);
        INDArray vPadded = Nd4j.createFromArray(new float[]{
                5, 6, 0, 0, 5, 5, 4, 4
        }).reshape(B, L, H, Dv);
        INDArray betaPadded = Nd4j.ones(DataType.FLOAT, B, L, H);
        INDArray gatePadded = Nd4j.zeros(DataType.FLOAT, B, L, H);
        INDArray actualLen = Nd4j.scalar(DataType.INT64, 2L);

        INDArray qShort = qPadded.get(point(0), interval(0, 2), all(), all()).dup().reshape(B, 2, H, Dk);
        INDArray kShort = kPadded.get(point(0), interval(0, 2), all(), all()).dup().reshape(B, 2, H, Dk);
        INDArray vShort = vPadded.get(point(0), interval(0, 2), all(), all()).dup().reshape(B, 2, H, Dv);
        INDArray betaShort = betaPadded.get(point(0), interval(0, 2), all()).dup().reshape(B, 2, H);
        INDArray gateShort = gatePadded.get(point(0), interval(0, 2), all()).dup().reshape(B, 2, H);
        INDArray[] shortRun = Nd4j.exec(new GatedDeltaRule(qShort, kShort, vShort, betaShort, gateShort));

        SameDiff sd = SameDiff.create();
        try {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, L, H, Dk);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, B, L, H, Dk);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, B, L, H, Dv);
            SDVariable beta = sd.placeHolder("beta", DataType.FLOAT, B, L, H);
            SDVariable gate = sd.placeHolder("gate", DataType.FLOAT, B, L, H);
            SDVariable len = sd.placeHolder("actual_sequence_length", DataType.INT64);
            SDVariable[] result = new GatedDeltaRule(sd, q, k, v, beta, gate, null, len).outputVariables();
            sd.updateVariableNameAndReference(result[0], "gdr_out");
            sd.updateVariableNameAndReference(result[1], "gdr_state");
            sd.setOutputs("gdr_out", "gdr_state");

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("q", qPadded);
            inputs.put("k", kPadded);
            inputs.put("v", vPadded);
            inputs.put("beta", betaPadded);
            inputs.put("gate", gatePadded);
            inputs.put("actual_sequence_length", actualLen);
            Map<String, INDArray> out = sd.output(inputs, "gdr_out", "gdr_state");

            assertEquals(0.0, out.get("gdr_state").sub(shortRun[1]).amaxNumber().doubleValue(), 1e-5,
                    "DSP state should ignore padded timesteps");
            assertEquals(0.0, out.get("gdr_out").get(point(0), interval(0, 2), all(), all()).sub(shortRun[0]).amaxNumber().doubleValue(), 1e-5,
                    "DSP real prefix outputs should match an unpadded run");
        } finally {
            sd.close();
        }
    }

    @Test
    public void testFrozenDspRerunRefreshesChangedActualLength() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        int B = 1, L = 5, active = 2, H = 16, Dk = 128, Dv = 128;
        Nd4j.getRandom().setSeed(12345L);
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.02);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.02);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H).muli(0.5);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01);

        INDArray qActive = q.get(point(0), interval(0, active), all(), all())
                .dup().reshape(B, active, H, Dk);
        INDArray kActive = k.get(point(0), interval(0, active), all(), all())
                .dup().reshape(B, active, H, Dk);
        INDArray vActive = v.get(point(0), interval(0, active), all(), all())
                .dup().reshape(B, active, H, Dv);
        INDArray betaActive = beta.get(point(0), interval(0, active), all())
                .dup().reshape(B, active, H);
        INDArray gateActive = gate.get(point(0), interval(0, active), all())
                .dup().reshape(B, active, H);
        INDArray expectedState = Nd4j.exec(new GatedDeltaRule(
                qActive, kActive, vActive, betaActive, gateActive, stateIn))[1];

        SameDiff sd = SameDiff.create();
        try {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            SDVariable qVar = sd.placeHolder("q", DataType.FLOAT, B, L, H, Dk);
            SDVariable kVar = sd.placeHolder("k", DataType.FLOAT, B, L, H, Dk);
            SDVariable vVar = sd.placeHolder("v", DataType.FLOAT, B, L, H, Dv);
            SDVariable betaVar = sd.placeHolder("beta", DataType.FLOAT, B, L, H);
            SDVariable gateVar = sd.placeHolder("gate", DataType.FLOAT, B, L, H);
            SDVariable stateVar = sd.placeHolder("state_in", DataType.FLOAT, B, H, Dk, Dv);
            SDVariable lenVar = sd.placeHolder("actual_sequence_length", DataType.INT64);
            SDVariable[] result = new GatedDeltaRule(
                    sd, qVar, kVar, vVar, betaVar, gateVar, stateVar, lenVar).outputVariables();
            sd.updateVariableNameAndReference(result[0], "gdr_rerun_out");
            sd.updateVariableNameAndReference(result[1], "gdr_rerun_state");
            sd.setOutputs("gdr_rerun_out", "gdr_rerun_state");

            INDArray actualLength = Nd4j.scalar(DataType.INT64, (long) L);
            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("q", q);
            inputs.put("k", k);
            inputs.put("v", v);
            inputs.put("beta", beta);
            inputs.put("gate", gate);
            inputs.put("state_in", stateIn);
            inputs.put("actual_sequence_length", actualLength);

            INDArray fullState = sd.output(inputs, "gdr_rerun_state")
                    .get("gdr_rerun_state").dup();
            InferenceSession session = sd.getOrCreateSession();
            assertNotNull(session.getDynamicShapePlanExecutor());
            session.getDynamicShapePlanExecutor().setShapesFrozen(true);

            actualLength.assign(active);
            INDArray rerunState = sd.output(inputs, "gdr_rerun_state")
                    .get("gdr_rerun_state");

            assertTrue(fullState.sub(expectedState).amaxNumber().doubleValue() > 1e-5,
                    "Discriminator must observe a different full-window state");
            assertEquals(0.0,
                    rerunState.sub(expectedState).amaxNumber().doubleValue(), 1e-5,
                    "Frozen DSP rerun must consume the updated active length");
        } finally {
            sd.close();
        }
    }

    @Test
    public void testModelRealisticActualLengthWithDsp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        int B = 1, L = 64, actual = 17, H = 16, Dk = 128, Dv = 128;
        Nd4j.getRandom().setSeed(123);
        INDArray qPadded = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.02);
        INDArray kPadded = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.02);
        INDArray vPadded = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray betaPadded = Nd4j.rand(DataType.FLOAT, B, L, H).muli(0.5);
        INDArray gatePadded = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01);
        INDArray actualLen = Nd4j.scalar(DataType.INT64, (long) actual);

        INDArray qShort = qPadded.get(point(0), interval(0, actual), all(), all()).dup().reshape(B, actual, H, Dk);
        INDArray kShort = kPadded.get(point(0), interval(0, actual), all(), all()).dup().reshape(B, actual, H, Dk);
        INDArray vShort = vPadded.get(point(0), interval(0, actual), all(), all()).dup().reshape(B, actual, H, Dv);
        INDArray betaShort = betaPadded.get(point(0), interval(0, actual), all()).dup().reshape(B, actual, H);
        INDArray gateShort = gatePadded.get(point(0), interval(0, actual), all()).dup().reshape(B, actual, H);
        INDArray[] shortRun = Nd4j.exec(new GatedDeltaRule(qShort, kShort, vShort, betaShort, gateShort, stateIn));

        SameDiff sd = SameDiff.create();
        try {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            SDVariable q = sd.placeHolder("q", DataType.FLOAT, B, L, H, Dk);
            SDVariable k = sd.placeHolder("k", DataType.FLOAT, B, L, H, Dk);
            SDVariable v = sd.placeHolder("v", DataType.FLOAT, B, L, H, Dv);
            SDVariable beta = sd.placeHolder("beta", DataType.FLOAT, B, L, H);
            SDVariable gate = sd.placeHolder("gate", DataType.FLOAT, B, L, H);
            SDVariable state = sd.placeHolder("state_in", DataType.FLOAT, B, H, Dk, Dv);
            SDVariable len = sd.placeHolder("actual_sequence_length", DataType.INT64);
            SDVariable[] result = new GatedDeltaRule(sd, q, k, v, beta, gate, state, len).outputVariables();
            sd.updateVariableNameAndReference(result[0], "gdr_out_realistic");
            sd.updateVariableNameAndReference(result[1], "gdr_state_realistic");
            sd.setOutputs("gdr_out_realistic", "gdr_state_realistic");

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("q", qPadded);
            inputs.put("k", kPadded);
            inputs.put("v", vPadded);
            inputs.put("beta", betaPadded);
            inputs.put("gate", gatePadded);
            inputs.put("state_in", stateIn);
            inputs.put("actual_sequence_length", actualLen);
            Map<String, INDArray> out = sd.output(inputs, "gdr_out_realistic", "gdr_state_realistic");

            assertEquals(0.0, out.get("gdr_state_realistic").sub(shortRun[1]).amaxNumber().doubleValue(), 1e-4,
                    "model-realistic DSP state should ignore padded timesteps");
            assertEquals(0.0, out.get("gdr_out_realistic").get(point(0), interval(0, actual), all(), all()).sub(shortRun[0]).amaxNumber().doubleValue(), 1e-4,
                    "model-realistic DSP real prefix outputs should match an unpadded run");
        } finally {
            sd.close();
        }
    }

    /**
     * Test at model-realistic dimensions (Qwen3: Dk=128, Dv=128, H=16)
     * with known seed for reproducibility. Checks that output values are
     * in a reasonable range (not suppressed or exploded).
     */
    @Test
    public void testModelRealisticDimensions() {
        int B = 1, L = 7, H = 16, Dk = 128, Dv = 128;
        Nd4j.getRandom().setSeed(42);
        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.02);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, Dk).muli(0.02);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, Dv).muli(0.1);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H).muli(0.5);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H).muli(0.3);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));
        INDArray output = result[0];
        INDArray stateOut = result[1];

        assertFalse(output.isNaN().any(), "Output contains NaN");
        assertFalse(output.isInfinite().any(), "Output contains Inf");
        assertFalse(stateOut.isNaN().any(), "State contains NaN");

        double outMax = output.amaxNumber().doubleValue();
        double outMean = output.meanNumber().doubleValue();
        double stateMax = stateOut.amaxNumber().doubleValue();
        System.out.println("[GDR realistic] output max=" + outMax + " mean=" + outMean + " stateMax=" + stateMax);

        // Output should not be all zeros or near-zero (signal suppression bug)
        assertTrue(outMax > 1e-4, "Output max too small (signal suppression): " + outMax);
        // Output should not explode
        assertTrue(outMax < 100.0, "Output max too large (numerical instability): " + outMax);
    }
}
