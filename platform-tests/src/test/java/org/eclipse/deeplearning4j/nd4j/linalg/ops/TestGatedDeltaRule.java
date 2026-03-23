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
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the gated_delta_rule op.
 *
 * The gated delta rule implements:
 *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
 *   output_t = S_t^T * q_t
 *
 * Inputs: Q [B,L,H,D_k], K [B,L,H,D_k], V [B,L,H,D_v], beta [B,L,H], gate [B,L,H]
 * Outputs: output [B,L,H,D_v], state_out [B,H,D_k,D_v]
 */
public class TestGatedDeltaRule {

    @Test
    public void testBasicShapesNoState() {
        int B = 2, L = 4, H = 2, D_k = 8, D_v = 8;

        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, D_v);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H).muli(0.1);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H);

        INDArray[] results = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));

        assertEquals(2, results.length, "Should produce 2 outputs");
        assertArrayEquals(new long[]{B, L, H, D_v}, results[0].shape(), "Output shape");
        assertArrayEquals(new long[]{B, H, D_k, D_v}, results[1].shape(), "State shape");

        // Output should be finite
        assertFalse(results[0].isNaN().any(), "Output contains NaN");
        assertFalse(results[0].isInfinite().any(), "Output contains Inf");
    }

    @Test
    public void testBasicShapesWithState() {
        int B = 1, L = 3, H = 2, D_k = 4, D_v = 4;

        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, D_v);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H).muli(0.1);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, D_k, D_v);

        INDArray[] results = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate, stateIn));

        assertEquals(2, results.length);
        assertArrayEquals(new long[]{B, L, H, D_v}, results[0].shape());
        assertArrayEquals(new long[]{B, H, D_k, D_v}, results[1].shape());
        assertFalse(results[0].isNaN().any());
        assertFalse(results[0].isInfinite().any());
    }

    @Test
    public void testZeroGateDecaysState() {
        // With gate = -inf (exp(g) = 0), the update ignores previous state:
        //   S_t = 0 * S_{t-1} + beta * k * (v - 0 * S_{t-1}^T * k)
        //       = beta * k * v
        int B = 1, L = 1, H = 1, D_k = 4, D_v = 4;

        INDArray q = Nd4j.ones(DataType.FLOAT, B, L, H, D_k);
        INDArray k = Nd4j.ones(DataType.FLOAT, B, L, H, D_k);
        INDArray v = Nd4j.ones(DataType.FLOAT, B, L, H, D_v);
        INDArray beta = Nd4j.ones(DataType.FLOAT, B, L, H);
        INDArray gate = Nd4j.valueArrayOf(new long[]{B, L, H}, -1000.0f); // exp(-1000) ≈ 0

        // With large initial state that should be decayed away
        INDArray stateIn = Nd4j.ones(DataType.FLOAT, B, H, D_k, D_v).muli(100);

        INDArray[] results = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate, stateIn));

        INDArray output = results[0];
        INDArray stateOut = results[1];

        // State should NOT retain the large initial values
        double stateMax = stateOut.amaxNumber().doubleValue();
        assertTrue(stateMax < 50.0,
                "Large gate=-inf should decay state to near zero, but max was " + stateMax);

        assertFalse(output.isNaN().any());
        assertFalse(stateOut.isNaN().any());
    }

    @Test
    public void testSequentialStateChaining() {
        // Run two single-step calls, passing state from first to second.
        // Result should match a single two-step call.
        int B = 1, H = 2, D_k = 4, D_v = 4;

        INDArray q1 = Nd4j.randn(DataType.FLOAT, B, 1, H, D_k);
        INDArray k1 = Nd4j.randn(DataType.FLOAT, B, 1, H, D_k);
        INDArray v1 = Nd4j.randn(DataType.FLOAT, B, 1, H, D_v);
        INDArray beta1 = Nd4j.rand(DataType.FLOAT, B, 1, H).muli(0.1);
        INDArray gate1 = Nd4j.randn(DataType.FLOAT, B, 1, H).muli(0.5);

        INDArray q2 = Nd4j.randn(DataType.FLOAT, B, 1, H, D_k);
        INDArray k2 = Nd4j.randn(DataType.FLOAT, B, 1, H, D_k);
        INDArray v2 = Nd4j.randn(DataType.FLOAT, B, 1, H, D_v);
        INDArray beta2 = Nd4j.rand(DataType.FLOAT, B, 1, H).muli(0.1);
        INDArray gate2 = Nd4j.randn(DataType.FLOAT, B, 1, H).muli(0.5);

        // Sequential: step 1, then step 2 with state from step 1
        INDArray[] step1 = Nd4j.exec(new GatedDeltaRule(q1, k1, v1, beta1, gate1));
        INDArray stateAfter1 = step1[1];
        INDArray[] step2 = Nd4j.exec(new GatedDeltaRule(q2, k2, v2, beta2, gate2, stateAfter1));
        INDArray seqOutput2 = step2[0];
        INDArray seqState2 = step2[1];

        // Batched: both steps at once (L=2)
        INDArray qAll = Nd4j.concat(1, q1, q2);
        INDArray kAll = Nd4j.concat(1, k1, k2);
        INDArray vAll = Nd4j.concat(1, v1, v2);
        INDArray betaAll = Nd4j.concat(1, beta1, beta2);
        INDArray gateAll = Nd4j.concat(1, gate1, gate2);

        INDArray[] batched = Nd4j.exec(new GatedDeltaRule(qAll, kAll, vAll, betaAll, gateAll));
        INDArray batchedOutput = batched[0];
        INDArray batchedState = batched[1];

        // Compare step 2 output and final state
        INDArray batchedOutput2 = batchedOutput.get(
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.point(1),
                org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).reshape(seqOutput2.shape());

        double outputDiff = seqOutput2.sub(batchedOutput2).amaxNumber().doubleValue();
        double stateDiff = seqState2.sub(batchedState).amaxNumber().doubleValue();

        assertTrue(outputDiff < 1e-3,
                "Sequential vs batched output diff too large: " + outputDiff);
        assertTrue(stateDiff < 1e-3,
                "Sequential vs batched state diff too large: " + stateDiff);
    }

    @Test
    public void testAsymmetricDimensions() {
        // Test with D_k != D_v
        int B = 1, L = 2, H = 2, D_k = 8, D_v = 16;

        INDArray q = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray k = Nd4j.randn(DataType.FLOAT, B, L, H, D_k);
        INDArray v = Nd4j.randn(DataType.FLOAT, B, L, H, D_v);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, L, H).muli(0.1);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, L, H);

        INDArray[] results = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));

        assertArrayEquals(new long[]{B, L, H, D_v}, results[0].shape());
        assertArrayEquals(new long[]{B, H, D_k, D_v}, results[1].shape());
        assertFalse(results[0].isNaN().any());
    }

    @Test
    public void testDoubleType() {
        int B = 1, L = 2, H = 1, D_k = 4, D_v = 4;

        INDArray q = Nd4j.randn(DataType.DOUBLE, B, L, H, D_k);
        INDArray k = Nd4j.randn(DataType.DOUBLE, B, L, H, D_k);
        INDArray v = Nd4j.randn(DataType.DOUBLE, B, L, H, D_v);
        INDArray beta = Nd4j.rand(DataType.DOUBLE, B, L, H).muli(0.1);
        INDArray gate = Nd4j.randn(DataType.DOUBLE, B, L, H);

        INDArray[] results = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));

        assertEquals(DataType.DOUBLE, results[0].dataType());
        assertEquals(DataType.DOUBLE, results[1].dataType());
        assertFalse(results[0].isNaN().any());
    }
}
