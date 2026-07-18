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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the P4 quantize kernels (quantize_q4_0 / quantize_q8_0).
 * These are the exact inverse of the existing ggml_dequantize op, so they are
 * validated by round-trip: dequantize(quantize(x)) must recover x within the
 * per-block quantization step, and re-quantizing the round-tripped values must
 * be byte-identical (idempotency).
 *
 * <h2>Running</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-native \
 *   -Dtest=TestGgmlQuantize 2>&1 | tee /tmp/test-ggml-quantize.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestGgmlQuantize {

    private static final int QK = 32;
    private static final int Q4_0 = 0, Q8_0 = 4;

    private static INDArray quantize(String opName, INDArray x) {
        return Nd4j.exec(DynamicCustomOp.builder(opName).addInputs(x).build())[0];
    }

    private static INDArray dequantize(INDArray bytes, int quantType, long n) {
        return Nd4j.exec(DynamicCustomOp.builder("ggml_dequantize")
                .addInputs(bytes)
                .addIntegerArguments(quantType, 0 /*F32*/, n)
                .build())[0];
    }

    /** Largest absolute value in block b (defines the quant step). */
    private static double blockMaxAbs(INDArray x, int b) {
        double m = 0;
        for (int j = 0; j < QK; j++) m = Math.max(m, Math.abs(x.getDouble((long) b * QK + j)));
        return m;
    }

    @Test
    public void testQ8_0OutputShapeAndRoundTrip() {
        Nd4j.getRandom().setSeed(21);
        int nBlocks = 4, n = nBlocks * QK;
        INDArray x = Nd4j.rand(DataType.FLOAT, n).muli(6).subi(3);

        INDArray bytes = quantize("quantize_q8_0", x);
        assertEquals(DataType.UINT8, bytes.dataType());
        assertArrayEquals(new long[]{(long) nBlocks * 34}, bytes.shape());

        INDArray deq = dequantize(bytes, Q8_0, n);
        assertArrayEquals(new long[]{n}, deq.shape());
        for (int b = 0; b < nBlocks; b++) {
            double step = blockMaxAbs(x, b) / 127.0;
            for (int j = 0; j < QK; j++) {
                int idx = b * QK + j;
                assertEquals(x.getDouble(idx), deq.getDouble(idx), step * 1.5 + 1e-6,
                        "Q8_0 round-trip error too large at " + idx);
            }
        }
    }

    @Test
    public void testQ4_0OutputShapeAndRoundTrip() {
        Nd4j.getRandom().setSeed(22);
        int nBlocks = 4, n = nBlocks * QK;
        INDArray x = Nd4j.rand(DataType.FLOAT, n).muli(4).subi(2);

        INDArray bytes = quantize("quantize_q4_0", x);
        assertEquals(DataType.UINT8, bytes.dataType());
        assertArrayEquals(new long[]{(long) nBlocks * 18}, bytes.shape());

        INDArray deq = dequantize(bytes, Q4_0, n);
        assertArrayEquals(new long[]{n}, deq.shape());
        for (int b = 0; b < nBlocks; b++) {
            double step = blockMaxAbs(x, b) / 8.0;  // Q4_0 step = |d| = max/8
            for (int j = 0; j < QK; j++) {
                int idx = b * QK + j;
                assertEquals(x.getDouble(idx), deq.getDouble(idx), step + 1e-6,
                        "Q4_0 round-trip error too large at " + idx);
            }
        }
    }

    @Test
    public void testIdempotencyQ8_0() {
        Nd4j.getRandom().setSeed(23);
        int n = 3 * QK;
        INDArray x = Nd4j.rand(DataType.FLOAT, n).muli(2).subi(1);

        INDArray bytes1 = quantize("quantize_q8_0", x);
        INDArray deq = dequantize(bytes1, Q8_0, n);
        INDArray bytes2 = quantize("quantize_q8_0", deq);  // re-quantizing must reproduce identical bytes
        assertEquals(bytes1, bytes2, "quantize(dequantize(quantize(x))) must be byte-identical");
    }

    @Test
    public void testKnownConstantBlock() {
        // A constant block of 1.0: Q4_0 recovers it exactly (xi = round(1*-8 + 8.5)=0 → (0-8)*(-0.125)=1.0),
        // Q8_0 recovers ~1.0 (all qi=127, d=fp16(1/127)).
        INDArray x = Nd4j.ones(DataType.FLOAT, QK);

        INDArray q4 = dequantize(quantize("quantize_q4_0", x), Q4_0, QK);
        for (int j = 0; j < QK; j++) assertEquals(1.0, q4.getDouble(j), 1e-3, "Q4_0 const block at " + j);

        INDArray q8 = dequantize(quantize("quantize_q8_0", x), Q8_0, QK);
        for (int j = 0; j < QK; j++) assertEquals(1.0, q8.getDouble(j), 1e-2, "Q8_0 const block at " + j);
    }

    @Test
    public void testRequiresMultipleOf32() {
        INDArray x = Nd4j.rand(DataType.FLOAT, 40);  // not a multiple of 32
        boolean threw = false;
        try {
            quantize("quantize_q8_0", x);
        } catch (Exception e) {
            threw = true;
        }
        assertTrue(threw, "quantize must reject element counts that are not a multiple of 32");
    }
}
