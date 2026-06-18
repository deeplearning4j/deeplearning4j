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

package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.DecoderMaskedMha;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Structural tests for the {@link DecoderMaskedMha} op.
 * Validates op name, argument wiring, output count, and constructor variants.
 * Does not execute the op (that requires a native backend with the kernel registered).
 *
 * Adam Gibson
 */
class DecoderMaskedMhaTest {

    // ==================== Op Name ====================

    @Test
    void testOpName() {
        DecoderMaskedMha op = new DecoderMaskedMha();
        assertEquals("decoder_masked_mha", op.opName());
    }

    // ==================== Output Count ====================

    @Test
    void testGetNumOutputs() {
        DecoderMaskedMha op = new DecoderMaskedMha();
        assertEquals(3, op.getNumOutputs(), "Should produce 3 outputs: output, presentKey, presentValue");
    }

    // ==================== Default Constructor ====================

    @Test
    void testDefaultConstructorDefaults() {
        DecoderMaskedMha op = new DecoderMaskedMha();
        assertEquals(32, op.getNumHeads());
        assertEquals(0, op.getNumKvHeads());
        assertEquals(0, op.getHeadDim());
        assertEquals(1, op.getUseRoPE());
        assertEquals(10000, op.getRopeBase());
        assertEquals(-3.4028235e+38f, op.getMaskFilterValue(), 1e30f);
    }

    // ==================== Full SameDiff Constructor ====================

    @Test
    void testFullSameDiffConstructor() {
        SameDiff sd = SameDiff.create();
        int B = 1, H = 256, heads = 8, kvHeads = 4, dim = 32, seqLen = 10;

        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, B, 1, H);
        SDVariable qkvW = sd.placeHolder("qkvW", DataType.FLOAT, H, 3 * H);
        SDVariable outW = sd.placeHolder("outW", DataType.FLOAT, H, H);
        SDVariable pastK = sd.placeHolder("pastK", DataType.FLOAT, B, heads, seqLen, dim);
        SDVariable pastV = sd.placeHolder("pastV", DataType.FLOAT, B, heads, seqLen, dim);

        int numHeads = 8, numKvHeads = 4, headDim = 32;
        int useRoPE = 1, ropeBase = 50000;
        float maskFilter = -1e9f;

        DecoderMaskedMha op = new DecoderMaskedMha(sd, hidden, qkvW, outW, pastK, pastV,
                numHeads, numKvHeads, headDim, useRoPE, ropeBase, maskFilter);

        assertEquals("decoder_masked_mha", op.opName());
        assertEquals(numHeads, op.getNumHeads());
        assertEquals(numKvHeads, op.getNumKvHeads());
        assertEquals(headDim, op.getHeadDim());
        assertEquals(useRoPE, op.getUseRoPE());
        assertEquals(ropeBase, op.getRopeBase());
        assertEquals(maskFilter, op.getMaskFilterValue(), 1e-5f);
        assertEquals(3, op.getNumOutputs());
    }

    // ==================== Boolean useRoPE Constructor ====================

    @Test
    void testBooleanRoPEConstructorTrue() {
        SameDiff sd = SameDiff.create();
        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, 1, 1, 128);
        SDVariable qkvW = sd.placeHolder("qkvW", DataType.FLOAT, 128, 384);
        SDVariable outW = sd.placeHolder("outW", DataType.FLOAT, 128, 128);
        SDVariable pastK = sd.placeHolder("pastK", DataType.FLOAT, 1, 4, 5, 32);
        SDVariable pastV = sd.placeHolder("pastV", DataType.FLOAT, 1, 4, 5, 32);

        DecoderMaskedMha op = new DecoderMaskedMha(sd, hidden, qkvW, outW, pastK, pastV,
                4, 2, 32, true);

        assertEquals(4, op.getNumHeads());
        assertEquals(2, op.getNumKvHeads());
        assertEquals(32, op.getHeadDim());
        assertEquals(1, op.getUseRoPE());
        assertEquals(10000, op.getRopeBase(), "Default ropeBase should be 10000");
        assertEquals(-3.4028235e+38f, op.getMaskFilterValue(), 1e30f, "Default maskFilterValue");
    }

    @Test
    void testBooleanRoPEConstructorFalse() {
        SameDiff sd = SameDiff.create();
        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, 1, 1, 128);
        SDVariable qkvW = sd.placeHolder("qkvW", DataType.FLOAT, 128, 384);
        SDVariable outW = sd.placeHolder("outW", DataType.FLOAT, 128, 128);
        SDVariable pastK = sd.placeHolder("pastK", DataType.FLOAT, 1, 4, 5, 32);
        SDVariable pastV = sd.placeHolder("pastV", DataType.FLOAT, 1, 4, 5, 32);

        DecoderMaskedMha op = new DecoderMaskedMha(sd, hidden, qkvW, outW, pastK, pastV,
                4, 2, 32, false);

        assertEquals(0, op.getUseRoPE());
    }

    // ==================== INDArray Constructor ====================

    @Test
    void testINDArrayConstructorWithOutputs() {
        int B = 1, H = 128, heads = 4, dim = 32, seqLen = 5;

        INDArray hidden = Nd4j.zeros(DataType.FLOAT, B, 1, H);
        INDArray qkvW = Nd4j.zeros(DataType.FLOAT, H, 3 * H);
        INDArray outW = Nd4j.zeros(DataType.FLOAT, H, H);
        INDArray pastK = Nd4j.zeros(DataType.FLOAT, B, heads, seqLen, dim);
        INDArray pastV = Nd4j.zeros(DataType.FLOAT, B, heads, seqLen, dim);

        INDArray output = Nd4j.zeros(DataType.FLOAT, B, 1, H);
        INDArray presentK = Nd4j.zeros(DataType.FLOAT, B, heads, seqLen + 1, dim);
        INDArray presentV = Nd4j.zeros(DataType.FLOAT, B, heads, seqLen + 1, dim);

        DecoderMaskedMha op = new DecoderMaskedMha(hidden, qkvW, outW, pastK, pastV,
                output, presentK, presentV,
                heads, 2, dim, 1, 10000, -3.4028235e+38f);

        assertEquals("decoder_masked_mha", op.opName());
        assertEquals(heads, op.getNumHeads());
        assertEquals(2, op.getNumKvHeads());
        assertEquals(dim, op.getHeadDim());
        assertEquals(3, op.getNumOutputs());
    }

    @Test
    void testINDArrayConstructorNullOutputs() {
        int B = 1, H = 64, heads = 2, dim = 32, seqLen = 3;

        INDArray hidden = Nd4j.zeros(DataType.FLOAT, B, 1, H);
        INDArray qkvW = Nd4j.zeros(DataType.FLOAT, H, 3 * H);
        INDArray outW = Nd4j.zeros(DataType.FLOAT, H, H);
        INDArray pastK = Nd4j.zeros(DataType.FLOAT, B, heads, seqLen, dim);
        INDArray pastV = Nd4j.zeros(DataType.FLOAT, B, heads, seqLen, dim);

        DecoderMaskedMha op = new DecoderMaskedMha(hidden, qkvW, outW, pastK, pastV,
                null, null, null,
                heads, heads, dim, 0, 10000, -1e4f);

        assertEquals("decoder_masked_mha", op.opName());
        assertEquals(heads, op.getNumHeads());
        assertEquals(0, op.getUseRoPE());
        assertEquals(-1e4f, op.getMaskFilterValue(), 1e-5f);
    }

    // ==================== configureFromArguments ====================

    @Test
    void testConfigureFromArguments() {
        DecoderMaskedMha op = new DecoderMaskedMha();
        // Simulate what happens when arguments are loaded from serialized form
        op.addIArgument(16L, 4L, 64L, 0L, 50000L);
        op.addTArgument(-1e9);
        op.configureFromArguments();

        assertEquals(16, op.getNumHeads());
        assertEquals(4, op.getNumKvHeads());
        assertEquals(64, op.getHeadDim());
        assertEquals(0, op.getUseRoPE());
        assertEquals(50000, op.getRopeBase());
        assertEquals(-1e9f, op.getMaskFilterValue(), 1.0f);
    }

    // ==================== calculateOutputDataTypes ====================

    @Test
    void testCalculateOutputDataTypesFloat() {
        DecoderMaskedMha op = new DecoderMaskedMha();
        var outputTypes = op.calculateOutputDataTypes(
                java.util.Arrays.asList(DataType.FLOAT, DataType.FLOAT, DataType.FLOAT,
                        DataType.FLOAT, DataType.FLOAT));
        assertEquals(3, outputTypes.size());
        for (DataType dt : outputTypes) {
            assertEquals(DataType.FLOAT, dt, "All outputs should match input dtype");
        }
    }

    @Test
    void testCalculateOutputDataTypesHalf() {
        DecoderMaskedMha op = new DecoderMaskedMha();
        var outputTypes = op.calculateOutputDataTypes(
                java.util.Arrays.asList(DataType.HALF, DataType.HALF, DataType.HALF,
                        DataType.HALF, DataType.HALF));
        assertEquals(3, outputTypes.size());
        for (DataType dt : outputTypes) {
            assertEquals(DataType.HALF, dt);
        }
    }

    // ==================== GQA / MQA Configuration ====================

    @Test
    void testGQAConfig() {
        // Grouped Query Attention: numKvHeads < numHeads
        SameDiff sd = SameDiff.create();
        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, 1, 1, 256);
        SDVariable qkvW = sd.placeHolder("qkvW", DataType.FLOAT, 256, 768);
        SDVariable outW = sd.placeHolder("outW", DataType.FLOAT, 256, 256);
        SDVariable pastK = sd.placeHolder("pastK", DataType.FLOAT, 1, 2, 5, 64);
        SDVariable pastV = sd.placeHolder("pastV", DataType.FLOAT, 1, 2, 5, 64);

        DecoderMaskedMha op = new DecoderMaskedMha(sd, hidden, qkvW, outW, pastK, pastV,
                8, 2, 64, true);

        assertEquals(8, op.getNumHeads());
        assertEquals(2, op.getNumKvHeads());
        assertTrue(op.getNumKvHeads() < op.getNumHeads(),
                "GQA should have fewer KV heads than query heads");
    }

    @Test
    void testMQAConfig() {
        // Multi-Query Attention: numKvHeads == 1
        SameDiff sd = SameDiff.create();
        SDVariable hidden = sd.placeHolder("hidden", DataType.FLOAT, 1, 1, 128);
        SDVariable qkvW = sd.placeHolder("qkvW", DataType.FLOAT, 128, 384);
        SDVariable outW = sd.placeHolder("outW", DataType.FLOAT, 128, 128);
        SDVariable pastK = sd.placeHolder("pastK", DataType.FLOAT, 1, 1, 5, 128);
        SDVariable pastV = sd.placeHolder("pastV", DataType.FLOAT, 1, 1, 5, 128);

        DecoderMaskedMha op = new DecoderMaskedMha(sd, hidden, qkvW, outW, pastK, pastV,
                4, 1, 128, true);

        assertEquals(4, op.getNumHeads());
        assertEquals(1, op.getNumKvHeads());
    }
}
