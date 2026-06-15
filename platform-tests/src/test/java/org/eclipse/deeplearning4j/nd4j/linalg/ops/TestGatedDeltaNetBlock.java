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
 */
@org.junit.jupiter.api.Disabled("gated_delta_net_block helper not yet implemented - op wrapper exists but helpers::gatedDeltaNetBlock() is missing")
public class TestGatedDeltaNetBlock {

    private static final int NUM_HEADS = 2;
    private static final int HEAD_DIM_K = 8;
    private static final int HEAD_DIM_V = 8;
    private static final int D = NUM_HEADS * HEAD_DIM_V;  // hidden dim = 16
    private static final int CONV_K = 4;  // causal conv kernel size
    private static final double RMS_EPS = 1e-5;

    // qkv_dim = H * (D_k + D_v) + H * D_v (for gate projection within block)
    // Actually: Wqkv projects to Q, K, V: [D, H*(Dk + Dv + Dv)] but the block handles it
    // The block projects x -> [Q, K, V] via Wqkv, so qkv_dim = H * (Dk + Dv)
    // But looking at the C++ code, it's just [D, qkv_dim] where the block splits internally
    private static final int QKV_DIM = NUM_HEADS * (HEAD_DIM_K + HEAD_DIM_V);

    @Test
    public void testBasicShapes() {
        int B = 1, L = 4;
        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, QKV_DIM).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, NUM_HEADS * HEAD_DIM_V, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, D, CONV_K).muli(0.1);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, D);

        INDArray[] result = Nd4j.exec(new GatedDeltaNetBlock(
                x, wqkv, wbeta, wgate, wout, convWeight, convBias,
                NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V, RMS_EPS));

        assertEquals(3, result.length, "Should produce 3 outputs");
        assertArrayEquals(new long[]{B, L, D}, result[0].shape(), "Output shape");
        assertArrayEquals(new long[]{B, NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V}, result[1].shape(), "Recurrent state shape");
        assertArrayEquals(new long[]{B, D, CONV_K - 1}, result[2].shape(), "Conv state shape");
        assertFalse(result[0].isNaN().any(), "Output contains NaN");
        assertFalse(result[0].isInfinite().any(), "Output contains Inf");
    }

    @Test
    public void testWithRecurrentState() {
        int B = 1, L = 3;
        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, QKV_DIM).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, NUM_HEADS * HEAD_DIM_V, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, D, CONV_K).muli(0.1);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, D);
        INDArray stateIn = Nd4j.randn(DataType.FLOAT, B, NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V).muli(0.01);

        INDArray[] result = Nd4j.exec(new GatedDeltaNetBlock(
                x, wqkv, wbeta, wgate, wout, convWeight, convBias, stateIn,
                NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V, RMS_EPS));

        assertEquals(3, result.length);
        assertArrayEquals(new long[]{B, L, D}, result[0].shape());
        assertFalse(result[0].isNaN().any(), "Output with state contains NaN");
    }

    @Test
    public void testStateChaining() {
        // Two sequential chunks, passing state from chunk1 to chunk2
        int B = 1, L = 2;
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, QKV_DIM).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, NUM_HEADS * HEAD_DIM_V, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, D, CONV_K).muli(0.1);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, D);

        // Chunk 1
        INDArray x1 = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray[] chunk1 = Nd4j.exec(new GatedDeltaNetBlock(
                x1, wqkv, wbeta, wgate, wout, convWeight, convBias,
                NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V, RMS_EPS));

        // Chunk 2 with chained state
        INDArray x2 = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray[] chunk2 = Nd4j.exec(new GatedDeltaNetBlock(
                x2, wqkv, wbeta, wgate, wout, convWeight, convBias, chunk1[1],
                NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V, RMS_EPS));

        // Chunk 2 without state (should differ)
        INDArray[] chunk2NoState = Nd4j.exec(new GatedDeltaNetBlock(
                x2, wqkv, wbeta, wgate, wout, convWeight, convBias,
                NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V, RMS_EPS));

        double diff = chunk2[0].sub(chunk2NoState[0]).amaxNumber().doubleValue();
        assertTrue(diff > 1e-6, "Chained state should affect output, diff=" + diff);
    }

    @Test
    public void testBatchDimension() {
        int B = 3, L = 2;
        INDArray x = Nd4j.randn(DataType.FLOAT, B, L, D).muli(0.1);
        INDArray wqkv = Nd4j.randn(DataType.FLOAT, D, QKV_DIM).muli(0.02);
        INDArray wbeta = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wgate = Nd4j.randn(DataType.FLOAT, D, NUM_HEADS).muli(0.02);
        INDArray wout = Nd4j.randn(DataType.FLOAT, NUM_HEADS * HEAD_DIM_V, D).muli(0.02);
        INDArray convWeight = Nd4j.randn(DataType.FLOAT, D, CONV_K).muli(0.1);
        INDArray convBias = Nd4j.zeros(DataType.FLOAT, D);

        INDArray[] result = Nd4j.exec(new GatedDeltaNetBlock(
                x, wqkv, wbeta, wgate, wout, convWeight, convBias,
                NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V, RMS_EPS));

        assertArrayEquals(new long[]{B, L, D}, result[0].shape(), "Batch output shape");
        assertArrayEquals(new long[]{B, NUM_HEADS, HEAD_DIM_K, HEAD_DIM_V}, result[1].shape(), "Batch state shape");
        assertArrayEquals(new long[]{B, D, CONV_K - 1}, result[2].shape(), "Batch conv state shape");
        assertFalse(result[0].isNaN().any());
    }
}
