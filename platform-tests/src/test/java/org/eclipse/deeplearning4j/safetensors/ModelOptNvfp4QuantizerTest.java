/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.safetensors;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * NVFP4 weight quantization with NVIDIA ModelOpt's recipe: exact reconstruction of on-grid
 * weights, the E2M1/E4M3 round-to-nearest-even boundaries, and the all-zero block convention.
 */
class ModelOptNvfp4QuantizerTest {
    private static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};

    private static float e2m1(int code) {
        float magnitude = E2M1[code & 7];
        return (code & 8) != 0 ? -magnitude : magnitude;
    }

    /**
     * Weights built on the NVFP4 grid with a power-of-two global scale (so every product is
     * exact) and a +/-6 element in every block (so each block recovers its own scale) must
     * quantize back to exactly the nibbles and block-scale codes they were built from.
     */
    @Test
    void onGridWeightsRoundTripExactly() {
        final int rows = 7, columns = 64, blocks = columns / 16;
        final float global = Math.scalb(1.0f, -10);
        Random random = new Random(20260928L);
        int[] codes = new int[rows * columns];
        int[] scaleCodes = new int[rows * blocks];
        float[] values = new float[rows * columns];
        for (int b = 0; b < rows * blocks; b++) {
            // The first block carries the largest scale (448) so amax / 2688 is exactly the global.
            scaleCodes[b] = b == 0 ? 0x7E : 0x20 + random.nextInt(0x50);
            float scale = ModelOptNvfp4Quantizer.decodeE4m3(scaleCodes[b]);
            for (int i = 0; i < 16; i++) {
                int code = i == 0 ? (random.nextBoolean() ? 7 : 15) : random.nextInt(16);
                codes[b * 16 + i] = code;
                values[b * 16 + i] = e2m1(code) * scale * global;
            }
        }
        INDArray weight = Nd4j.create(values, new long[]{rows, columns}, DataType.FLOAT);
        ModelOptNvfp4Quantizer.Quantized quantized = ModelOptNvfp4Quantizer.quantize(weight);

        assertEquals(DataType.UBYTE, quantized.packed.dataType());
        assertEquals(DataType.FLOAT8, quantized.blockScales.dataType());
        assertEquals(global, quantized.globalScale.getFloat(0), 0.0f);
        byte[] packed = bytes(quantized.packed);
        byte[] scales = bytes(quantized.blockScales);
        for (int b = 0; b < rows * blocks; b++) {
            assertEquals(scaleCodes[b], scales[b] & 0xFF, "block scale " + b);
        }
        for (int i = 0; i < rows * columns; i++) {
            int nibble = (packed[i / 2] >> (4 * (i & 1))) & 0x0F;
            // A -0 source element quantizes to +0: ModelOpt's sign bit is weight < 0.
            int expected = codes[i] == 8 ? 0 : codes[i];
            assertEquals(expected, nibble, "element " + i);
        }
    }

    @Test
    void e2m1RoundsToNearestWithTiesToEven() {
        // Boundaries between codes 0|1|2|3|4|5|6|7: ties go to the even code.
        float[] ties = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        int[] expected = {0, 2, 2, 4, 4, 6, 6};
        for (int i = 0; i < ties.length; i++) {
            assertEquals(expected[i], ModelOptNvfp4Quantizer.encodeE2m1(ties[i]), "tie " + ties[i]);
            assertEquals(expected[i] | 8, ModelOptNvfp4Quantizer.encodeE2m1(-ties[i]), "tie " + -ties[i]);
            float up = Math.nextUp(ties[i]), down = Math.nextDown(ties[i]);
            assertEquals(i + 1, ModelOptNvfp4Quantizer.encodeE2m1(up), "above " + ties[i]);
            assertEquals(i, ModelOptNvfp4Quantizer.encodeE2m1(down), "below " + ties[i]);
        }
        assertEquals(7, ModelOptNvfp4Quantizer.encodeE2m1(100f), "saturates at 6");
        assertEquals(8, ModelOptNvfp4Quantizer.encodeE2m1(-1e-9f), "small negatives keep the sign");
        assertEquals(0, ModelOptNvfp4Quantizer.encodeE2m1(-0.0f), "negative zero is not negative");
    }

    @Test
    void e4m3RoundsToNearestWithTiesToEven() {
        for (int code = 0; code < 0x7E; code++) {
            float value = ModelOptNvfp4Quantizer.decodeE4m3(code);
            float next = ModelOptNvfp4Quantizer.decodeE4m3(code + 1);
            assertEquals(code, ModelOptNvfp4Quantizer.encodeE4m3(value), "exact " + value);
            float midpoint = (value + next) / 2;  // exact: adjacent E4M3 values differ in few bits
            int even = (code & 1) == 0 ? code : code + 1;
            assertEquals(even, ModelOptNvfp4Quantizer.encodeE4m3(midpoint), "midpoint " + midpoint);
            assertEquals(code, ModelOptNvfp4Quantizer.encodeE4m3(Math.nextDown(midpoint)), "below " + midpoint);
            assertEquals(code + 1, ModelOptNvfp4Quantizer.encodeE4m3(Math.nextUp(midpoint)), "above " + midpoint);
        }
        // FP32 scale arithmetic can land just above 448; below the next midpoint it rounds to 448.
        assertEquals(0x7E, ModelOptNvfp4Quantizer.encodeE4m3(Math.nextUp(448f)));
        assertEquals(0x7E, ModelOptNvfp4Quantizer.encodeE4m3(Math.nextDown(464f)));
        assertThrows(IllegalArgumentException.class, () -> ModelOptNvfp4Quantizer.encodeE4m3(464f));
    }

    @Test
    void zeroBlocksUseUnitScale() {
        float[] values = new float[32];
        values[16] = 3.0f;  // second block nonzero; first block all zero
        ModelOptNvfp4Quantizer.Quantized quantized =
                ModelOptNvfp4Quantizer.quantize(Nd4j.create(values, new long[]{1, 32}, DataType.FLOAT));
        byte[] scales = bytes(quantized.blockScales);
        assertEquals(ModelOptNvfp4Quantizer.encodeE4m3(1.0f), scales[0] & 0xFF, "all-zero block scale");
        byte[] packed = bytes(quantized.packed);
        for (int i = 0; i < 8; i++) assertEquals(0, packed[i], "all-zero block nibbles");
        assertEquals(3.0f / 2688f, quantized.globalScale.getFloat(0), 0.0f);
    }

    private static byte[] bytes(INDArray array) {
        byte[] result = new byte[(int) array.length()];
        new org.bytedeco.javacpp.BytePointer(array.data().pointer()).capacity(result.length).get(result);
        return result;
    }
}
