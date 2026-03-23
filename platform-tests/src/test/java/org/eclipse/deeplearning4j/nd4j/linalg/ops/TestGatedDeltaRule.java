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
}
