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

package org.eclipse.deeplearning4j.ggml.quantization;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.quantization.Dequantizer;
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.ggml.quantization.Quantizer;
import org.nd4j.ggml.quantization.QuantizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Quantizer Test")
class QuantizerTest {

    private static final float TOLERANCE = 0.5f; // Quantization tolerance

    // ========== Factory Tests ==========

    @Test
    @DisplayName("Test QuantizerFactory has quantizer for Q4_0")
    void testFactoryHasQ4_0() {
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q4_0));
    }

    @Test
    @DisplayName("Test QuantizerFactory has quantizer for Q4_1")
    void testFactoryHasQ4_1() {
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q4_1));
    }

    @Test
    @DisplayName("Test QuantizerFactory has quantizer for Q5_0")
    void testFactoryHasQ5_0() {
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q5_0));
    }

    @Test
    @DisplayName("Test QuantizerFactory has quantizer for Q5_1")
    void testFactoryHasQ5_1() {
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q5_1));
    }

    @Test
    @DisplayName("Test QuantizerFactory has quantizer for Q8_0")
    void testFactoryHasQ8_0() {
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q8_0));
    }

    @Test
    @DisplayName("Test QuantizerFactory has quantizer for K-quant types")
    void testFactoryHasKQuantTypes() {
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q4_K));
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q5_K));
        assertTrue(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_Q6_K));
    }

    @Test
    @DisplayName("Test QuantizerFactory returns false for non-quantized types")
    void testFactoryNoQuantForNonQuantized() {
        assertFalse(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_F32));
        assertFalse(QuantizerFactory.hasQuantizer(GGMLDataType.GGML_TYPE_F16));
    }

    @Test
    @DisplayName("Test getQuantizer throws for unsupported type")
    void testGetQuantizerUnsupported() {
        assertThrows(IllegalArgumentException.class,
            () -> QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_F32));
    }

    @Test
    @DisplayName("Test getSupportedTypes returns expected types")
    void testGetSupportedTypes() {
        var supported = QuantizerFactory.getSupportedTypes();
        assertNotNull(supported);
        assertTrue(supported.contains(GGMLDataType.GGML_TYPE_Q4_0));
        assertTrue(supported.contains(GGMLDataType.GGML_TYPE_Q8_0));
        assertTrue(supported.contains(GGMLDataType.GGML_TYPE_Q4_K));
    }

    // ========== Block Size Tests ==========

    @Test
    @DisplayName("Test Q4_0 quantizer block size")
    void testQ4_0BlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        assertNotNull(quantizer);
        assertEquals(32, quantizer.getBlockSize());
        assertEquals(18, quantizer.getBytesPerBlock()); // 2 (scale) + 16 (data)
    }

    @Test
    @DisplayName("Test Q4_1 quantizer block size")
    void testQ4_1BlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_1);
        assertNotNull(quantizer);
        assertEquals(32, quantizer.getBlockSize());
        assertEquals(20, quantizer.getBytesPerBlock()); // 2 (scale) + 2 (min) + 16 (data)
    }

    @Test
    @DisplayName("Test Q5_0 quantizer block size")
    void testQ5_0BlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q5_0);
        assertNotNull(quantizer);
        assertEquals(32, quantizer.getBlockSize());
        assertEquals(22, quantizer.getBytesPerBlock()); // 2 (scale) + 4 (high bits) + 16 (low bits)
    }

    @Test
    @DisplayName("Test Q5_1 quantizer block size")
    void testQ5_1BlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q5_1);
        assertNotNull(quantizer);
        assertEquals(32, quantizer.getBlockSize());
        assertEquals(24, quantizer.getBytesPerBlock()); // 2 (scale) + 2 (min) + 4 (high bits) + 16 (low bits)
    }

    @Test
    @DisplayName("Test Q8_0 quantizer block size")
    void testQ8_0BlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        assertNotNull(quantizer);
        assertEquals(32, quantizer.getBlockSize());
        assertEquals(34, quantizer.getBytesPerBlock()); // 2 (scale) + 32 (data)
    }

    @Test
    @DisplayName("Test Q4_K quantizer block size")
    void testQ4_KBlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_K);
        assertNotNull(quantizer);
        assertEquals(256, quantizer.getBlockSize());
        assertEquals(148, quantizer.getBytesPerBlock());
    }

    @Test
    @DisplayName("Test Q5_K quantizer block size")
    void testQ5_KBlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q5_K);
        assertNotNull(quantizer);
        assertEquals(256, quantizer.getBlockSize());
        assertEquals(180, quantizer.getBytesPerBlock());
    }

    @Test
    @DisplayName("Test Q6_K quantizer block size")
    void testQ6_KBlockSize() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q6_K);
        assertNotNull(quantizer);
        assertEquals(256, quantizer.getBlockSize());
        assertEquals(210, quantizer.getBytesPerBlock());
    }

    // ========== Output Size Tests ==========

    @Test
    @DisplayName("Test Q4_0 calculateOutputBytes")
    void testQ4_0CalculateOutputBytes() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        // 32 elements = 1 block = 18 bytes
        assertEquals(18, quantizer.calculateOutputBytes(32));
        // 64 elements = 2 blocks = 36 bytes
        assertEquals(36, quantizer.calculateOutputBytes(64));
        // 33 elements = 2 blocks (padding) = 36 bytes
        assertEquals(36, quantizer.calculateOutputBytes(33));
    }

    @Test
    @DisplayName("Test Q8_0 calculateOutputBytes")
    void testQ8_0CalculateOutputBytes() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        // 32 elements = 1 block = 34 bytes
        assertEquals(34, quantizer.calculateOutputBytes(32));
        // 64 elements = 2 blocks = 68 bytes
        assertEquals(68, quantizer.calculateOutputBytes(64));
    }

    @Test
    @DisplayName("Test Q4_K calculateOutputBytes")
    void testQ4_KCalculateOutputBytes() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_K);
        // 256 elements = 1 block = 148 bytes
        assertEquals(148, quantizer.calculateOutputBytes(256));
        // 512 elements = 2 blocks = 296 bytes
        assertEquals(296, quantizer.calculateOutputBytes(512));
    }

    // ========== Quantization Tests ==========

    @Test
    @DisplayName("Test Q4_0 quantization produces correct length")
    void testQ4_0QuantizeLength() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        float[] data = new float[32];
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(18, result.length);
    }

    @Test
    @DisplayName("Test Q8_0 quantization produces correct length")
    void testQ8_0QuantizeLength() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        float[] data = new float[32];
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(34, result.length);
    }

    @Test
    @DisplayName("Test Q4_K quantization produces correct length")
    void testQ4_KQuantizeLength() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_K);
        float[] data = new float[256];
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(148, result.length);
    }

    @Test
    @DisplayName("Test Q4_0 quantize from INDArray")
    void testQ4_0QuantizeINDArray() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        INDArray array = Nd4j.randn(32);
        byte[] result = quantizer.quantize(array);
        assertNotNull(result);
        assertEquals(18, result.length);
    }

    @Test
    @DisplayName("Test Q8_0 quantize from INDArray")
    void testQ8_0QuantizeINDArray() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        INDArray array = Nd4j.randn(64);
        byte[] result = quantizer.quantize(array);
        assertNotNull(result);
        assertEquals(68, result.length);
    }

    // ========== Round-trip Tests (Quantize -> Dequantize) ==========

    @Test
    @DisplayName("Test Q4_0 round-trip quantization")
    void testQ4_0RoundTrip() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q4_0)) {
            return; // Skip if dequantizer not available
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_0);

        float[] original = generateRandomData(32);
        byte[] quantized = quantizer.quantize(original);
        float[] dequantized = dequantizer.dequantize(quantized, 32);

        // Check that values are reasonably close
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], dequantized[i], TOLERANCE,
                "Value at index " + i + " differs too much");
        }
    }

    @Test
    @DisplayName("Test Q8_0 round-trip quantization")
    void testQ8_0RoundTrip() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q8_0)) {
            return;
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q8_0);

        float[] original = generateRandomData(32);
        byte[] quantized = quantizer.quantize(original);
        float[] dequantized = dequantizer.dequantize(quantized, 32);

        // Q8_0 should have better precision than Q4_0
        for (int i = 0; i < original.length; i++) {
            assertEquals(original[i], dequantized[i], 0.1f,
                "Value at index " + i + " differs too much");
        }
    }

    @Test
    @DisplayName("Test Q4_K round-trip quantization")
    void testQ4_KRoundTrip() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q4_K)) {
            return;
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_K);
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_K);

        float[] original = generateRandomData(256);
        byte[] quantized = quantizer.quantize(original);
        float[] dequantized = dequantizer.dequantize(quantized, 256);

        // Check correlation rather than exact values
        double correlation = computeCorrelation(original, dequantized);
        assertTrue(correlation > 0.9, "Correlation should be high: " + correlation);
    }

    @Test
    @DisplayName("Test Q5_K round-trip quantization")
    void testQ5_KRoundTrip() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q5_K)) {
            return;
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q5_K);
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q5_K);

        float[] original = generateRandomData(256);
        byte[] quantized = quantizer.quantize(original);
        float[] dequantized = dequantizer.dequantize(quantized, 256);

        double correlation = computeCorrelation(original, dequantized);
        assertTrue(correlation > 0.9, "Correlation should be high: " + correlation);
    }

    @Test
    @DisplayName("Test Q6_K round-trip quantization")
    void testQ6_KRoundTrip() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q6_K)) {
            return;
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q6_K);
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q6_K);

        float[] original = generateRandomData(256);
        byte[] quantized = quantizer.quantize(original);
        float[] dequantized = dequantizer.dequantize(quantized, 256);

        // Q6_K should have better precision than Q4_K
        double correlation = computeCorrelation(original, dequantized);
        assertTrue(correlation > 0.95, "Correlation should be high: " + correlation);
    }

    // ========== Multiple Block Tests ==========

    @Test
    @DisplayName("Test Q4_0 multiple blocks")
    void testQ4_0MultipleBlocks() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        float[] data = new float[64]; // 2 blocks
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(36, result.length); // 2 * 18 bytes
    }

    @Test
    @DisplayName("Test Q8_0 multiple blocks")
    void testQ8_0MultipleBlocks() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        float[] data = new float[96]; // 3 blocks
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(102, result.length); // 3 * 34 bytes
    }

    @Test
    @DisplayName("Test Q4_K multiple blocks")
    void testQ4_KMultipleBlocks() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_K);
        float[] data = new float[512]; // 2 blocks
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(296, result.length); // 2 * 148 bytes
    }

    // ========== Edge Cases ==========

    @Test
    @DisplayName("Test quantization with zeros")
    void testQuantizeZeros() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        float[] zeros = new float[32];
        byte[] result = quantizer.quantize(zeros);
        assertNotNull(result);
        assertEquals(18, result.length);

        if (DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q4_0)) {
            Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_0);
            float[] dequantized = dequantizer.dequantize(result, 32);
            for (float v : dequantized) {
                assertEquals(0.0f, v, 0.01f);
            }
        }
    }

    @Test
    @DisplayName("Test quantization with constant value")
    void testQuantizeConstant() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        float[] constant = new float[32];
        java.util.Arrays.fill(constant, 5.0f);
        byte[] result = quantizer.quantize(constant);
        assertNotNull(result);
        assertEquals(34, result.length);

        if (DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q8_0)) {
            Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q8_0);
            float[] dequantized = dequantizer.dequantize(result, 32);
            for (float v : dequantized) {
                assertEquals(5.0f, v, 0.1f);
            }
        }
    }

    @Test
    @DisplayName("Test quantization with large values")
    void testQuantizeLargeValues() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        float[] data = new float[32];
        for (int i = 0; i < 32; i++) {
            data[i] = (i - 16) * 1000.0f;
        }
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(18, result.length);
    }

    @Test
    @DisplayName("Test quantization with partial block")
    void testQuantizePartialBlock() {
        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        // 40 elements = 1 full block + partial block = 2 blocks
        float[] data = new float[40];
        byte[] result = quantizer.quantize(data);
        assertNotNull(result);
        assertEquals(36, result.length); // 2 * 18 bytes
    }

    // ========== Statistics Tests ==========

    @Test
    @DisplayName("Test getQuantizationStats for Q4_0")
    void testQuantizationStatsQ4_0() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q4_0)) {
            return;
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        float[] data = generateRandomData(64);

        Quantizer.QuantizationStats stats = quantizer.getQuantizationStats(data);
        assertNotNull(stats);
        assertTrue(stats.getMinValue() <= stats.getMaxValue());
        assertTrue(stats.getMeanAbsError() >= 0);
        assertTrue(stats.getMaxAbsError() >= stats.getMeanAbsError());
    }

    @Test
    @DisplayName("Test getQuantizationStats for Q8_0")
    void testQuantizationStatsQ8_0() {
        if (!DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q8_0)) {
            return;
        }

        Quantizer quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q8_0);
        float[] data = generateRandomData(64);

        Quantizer.QuantizationStats stats = quantizer.getQuantizationStats(data);
        assertNotNull(stats);

        // Q8_0 should have lower error than Q4_0
        Quantizer q4Quantizer = QuantizerFactory.getQuantizer(GGMLDataType.GGML_TYPE_Q4_0);
        Quantizer.QuantizationStats q4Stats = q4Quantizer.getQuantizationStats(data);

        assertTrue(stats.getMeanAbsError() <= q4Stats.getMeanAbsError(),
            "Q8_0 should have lower error than Q4_0");
    }

    // ========== Factory Convenience Methods ==========

    @Test
    @DisplayName("Test QuantizerFactory.quantize convenience method")
    void testFactoryQuantize() {
        float[] data = generateRandomData(32);
        byte[] result = QuantizerFactory.quantize(data, GGMLDataType.GGML_TYPE_Q4_0);
        assertNotNull(result);
        assertEquals(18, result.length);
    }

    @Test
    @DisplayName("Test QuantizerFactory.quantize with INDArray")
    void testFactoryQuantizeINDArray() {
        INDArray array = Nd4j.randn(32);
        byte[] result = QuantizerFactory.quantize(array, GGMLDataType.GGML_TYPE_Q8_0);
        assertNotNull(result);
        assertEquals(34, result.length);
    }

    @Test
    @DisplayName("Test QuantizerFactory.calculateOutputBytes")
    void testFactoryCalculateOutputBytes() {
        assertEquals(18, QuantizerFactory.calculateOutputBytes(32, GGMLDataType.GGML_TYPE_Q4_0));
        assertEquals(34, QuantizerFactory.calculateOutputBytes(32, GGMLDataType.GGML_TYPE_Q8_0));
        assertEquals(148, QuantizerFactory.calculateOutputBytes(256, GGMLDataType.GGML_TYPE_Q4_K));
    }

    // ========== Helper Methods ==========

    private float[] generateRandomData(int size) {
        Random random = new Random(42); // Fixed seed for reproducibility
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (random.nextFloat() - 0.5f) * 2.0f; // Range [-1, 1]
        }
        return data;
    }

    private double computeCorrelation(float[] x, float[] y) {
        if (x.length != y.length || x.length == 0) {
            return 0;
        }

        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        int n = x.length;

        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        double numerator = n * sumXY - sumX * sumY;
        double denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return denominator == 0 ? 0 : numerator / denominator;
    }
}
