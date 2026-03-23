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
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the gated_delta_net_block op.
 *
 * Fuses: linear projection -> causal_conv1d + SiLU -> gated_delta_rule
 *        -> RMSNorm + Swish gate -> output projection
 *
 * Inputs:
 *   0: x [B, L, D]
 *   1: Wqkv [D, qkv_dim]  where qkv_dim = H * (2*D_k + D_v)
 *   2: Wbeta [D, H]
 *   3: Wgate [D, H]
 *   4: Wout [H*D_v, D]
 *   5: conv_weight [D, K]  where D = H * (D_k + D_v) in the conv path
 *   6: conv_bias [D]       where D is the dimension passed through conv
 *
 * Outputs:
 *   0: output [B, L, D]
 *   1: recurrent_state_out [B, H, D_k, D_v]
 *   2: conv_state_out [B, D_conv, K-1]
 */
public class TestGatedDeltaNetBlock {

    @Test
    public void testBasicShapes() {
        int B = 1, L = 4, D = 32;
        int H = 2, D_k = 8, D_v = 8;
        int K = 4; // conv kernel size
        // qkv_dim: H * (2*D_k + D_v) = 2 * (16 + 8) = 48
        int qkvDim = H * (2 * D_k + D_v);
        // conv operates on D_k + D_v per head = 16 per head, H heads = 32 total
        int convDim = H * (D_k + D_v);

        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D);
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, qkvDim).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, H * D_v, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, convDim, K).muli(0.02);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, convDim);

        INDArray[] results = Nd4j.exec(new GatedDeltaNetBlock(
                x, wqkv, wbeta, wgate, wout, convWeight, convBias, H, D_k, D_v));

        assertEquals(3, results.length, "Should produce 3 outputs");
        assertArrayEquals(new long[]{B, L, D}, results[0].shape(), "Output shape");
        assertArrayEquals(new long[]{B, H, D_k, D_v}, results[1].shape(), "Recurrent state shape");
        assertArrayEquals(new long[]{B, convDim, K - 1}, results[2].shape(), "Conv state shape");

        assertFalse(results[0].isNaN().any(), "Output contains NaN");
        assertFalse(results[0].isInfinite().any(), "Output contains Inf");
    }

    @Test
    public void testWithRecurrentState() {
        int B = 1, L = 2, D = 16;
        int H = 2, D_k = 4, D_v = 4;
        int K = 3;
        int qkvDim = H * (2 * D_k + D_v);
        int convDim = H * (D_k + D_v);

        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D);
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, qkvDim).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, H * D_v, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, convDim, K).muli(0.02);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, convDim);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, H, D_k, D_v).muli(0.1);

        INDArray[] results = Nd4j.exec(new GatedDeltaNetBlock(
                x, wqkv, wbeta, wgate, wout, convWeight, convBias,
                stateIn, H, D_k, D_v, 1e-5));

        assertEquals(3, results.length);
        assertArrayEquals(new long[]{B, L, D}, results[0].shape());
        assertArrayEquals(new long[]{B, H, D_k, D_v}, results[1].shape());
        assertArrayEquals(new long[]{B, convDim, K - 1}, results[2].shape());

        assertFalse(results[0].isNaN().any());
    }

    @Test
    public void testStateChaining() {
        // Process in two chunks and verify the second chunk with state produces
        // different results than without state
        int B = 1, L = 2, D = 16;
        int H = 2, D_k = 4, D_v = 4;
        int K = 3;
        int qkvDim = H * (2 * D_k + D_v);
        int convDim = H * (D_k + D_v);

        Nd4j.getRandom().setSeed(42);

        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, qkvDim).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, H * D_v, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, convDim, K).muli(0.02);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, convDim);

        INDArray x1 = Nd4j.randn(DataType.FLOAT, B, L, D);
        INDArray x2 = Nd4j.randn(DataType.FLOAT, B, L, D);

        // Step 1
        INDArray[] step1 = Nd4j.exec(new GatedDeltaNetBlock(
                x1, wqkv, wbeta, wgate, wout, convWeight, convBias, H, D_k, D_v));
        INDArray stateAfter1 = step1[1];

        // Step 2 WITH state from step 1
        INDArray[] step2WithState = Nd4j.exec(new GatedDeltaNetBlock(
                x2, wqkv, wbeta, wgate, wout, convWeight, convBias,
                stateAfter1, H, D_k, D_v, 1e-5));

        // Step 2 WITHOUT state
        INDArray[] step2NoState = Nd4j.exec(new GatedDeltaNetBlock(
                x2, wqkv, wbeta, wgate, wout, convWeight, convBias, H, D_k, D_v));

        // Results should differ when state is provided
        double diff = step2WithState[0].sub(step2NoState[0]).amaxNumber().doubleValue();
        assertTrue(diff > 1e-6,
                "State should influence output, but diff was only " + diff);
    }

    @Test
    public void testBatchDimension() {
        int B = 4, L = 3, D = 16;
        int H = 2, D_k = 4, D_v = 4;
        int K = 3;
        int qkvDim = H * (2 * D_k + D_v);
        int convDim = H * (D_k + D_v);

        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D);
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, qkvDim).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, H).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, H * D_v, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, convDim, K).muli(0.02);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, convDim);

        INDArray[] results = Nd4j.exec(new GatedDeltaNetBlock(
                x, wqkv, wbeta, wgate, wout, convWeight, convBias, H, D_k, D_v));

        assertArrayEquals(new long[]{B, L, D}, results[0].shape());
        assertArrayEquals(new long[]{B, H, D_k, D_v}, results[1].shape());
        assertFalse(results[0].isNaN().any());
    }
}
