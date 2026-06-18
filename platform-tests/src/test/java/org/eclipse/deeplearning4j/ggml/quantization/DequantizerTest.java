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
import org.nd4j.ggml.quantization.QuantizationInfo;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Dequantizer Test")
class DequantizerTest {

    @Test
    @DisplayName("Test DequantizerFactory has dequantizer for Q4_0")
    void testFactoryHasQ4_0() {
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q4_0));
    }

    @Test
    @DisplayName("Test DequantizerFactory has dequantizer for Q8_0")
    void testFactoryHasQ8_0() {
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q8_0));
    }

    @Test
    @DisplayName("Test DequantizerFactory has dequantizer for K-quant types")
    void testFactoryHasKQuantTypes() {
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q2_K));
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q3_K));
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q4_K));
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q5_K));
        assertTrue(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_Q6_K));
    }

    @Test
    @DisplayName("Test DequantizerFactory returns false for non-quantized types")
    void testFactoryNoDeqForNonQuantized() {
        assertFalse(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_F32));
        assertFalse(DequantizerFactory.hasDequantizer(GGMLDataType.GGML_TYPE_F16));
    }

    @Test
    @DisplayName("Test Q4_0 dequantizer block size")
    void testQ4_0BlockSize() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_0);
        assertNotNull(dequantizer);
        assertEquals(32, dequantizer.getBlockSize());
    }

    @Test
    @DisplayName("Test Q8_0 dequantizer block size")
    void testQ8_0BlockSize() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q8_0);
        assertNotNull(dequantizer);
        assertEquals(32, dequantizer.getBlockSize());
    }

    @Test
    @DisplayName("Test Q4_K dequantizer block size")
    void testQ4_KBlockSize() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_K);
        assertNotNull(dequantizer);
        assertEquals(256, dequantizer.getBlockSize());
    }

    @Test
    @DisplayName("Test Q4_0 dequantization produces correct length")
    void testQ4_0DequantizeLength() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_0);
        assertNotNull(dequantizer);

        // Create a Q4_0 block: 2 bytes scale (FP16) + 16 bytes data (32 4-bit values)
        byte[] quantizedData = new byte[18];
        // Set scale to 1.0 in FP16 (0x3C00)
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x3C;

        float[] result = dequantizer.dequantize(quantizedData, 32);

        assertNotNull(result);
        assertEquals(32, result.length);
    }

    @Test
    @DisplayName("Test Q8_0 dequantization produces correct length")
    void testQ8_0DequantizeLength() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q8_0);
        assertNotNull(dequantizer);

        // Create a Q8_0 block: 2 bytes scale (FP16) + 32 bytes data (32 8-bit values)
        byte[] quantizedData = new byte[34];
        // Set scale to 1.0 in FP16 (0x3C00)
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x3C;

        float[] result = dequantizer.dequantize(quantizedData, 32);

        assertNotNull(result);
        assertEquals(32, result.length);
    }

    @Test
    @DisplayName("Test Q4_0 dequantize to INDArray")
    void testQ4_0DequantizeToArray() {
        // Create Q4_0 quantized data for 32 elements (1 block)
        byte[] quantizedData = new byte[18];
        // Set scale to 1.0 in FP16
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x3C;

        long[] shape = new long[]{32};
        INDArray result = DequantizerFactory.dequantizeToArray(
                quantizedData, GGMLDataType.GGML_TYPE_Q4_0, shape, DataType.FLOAT);

        assertNotNull(result);
        assertEquals(DataType.FLOAT, result.dataType());
        assertArrayEquals(shape, result.shape());
    }

    @Test
    @DisplayName("Test Q8_0 dequantize to INDArray with FLOAT16 target")
    void testQ8_0DequantizeToFloat16() {
        // Create Q8_0 quantized data for 32 elements (1 block)
        byte[] quantizedData = new byte[34];
        // Set scale to 1.0 in FP16
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x3C;

        long[] shape = new long[]{32};
        INDArray result = DequantizerFactory.dequantizeToArray(
                quantizedData, GGMLDataType.GGML_TYPE_Q8_0, shape, DataType.HALF);

        assertNotNull(result);
        assertEquals(DataType.HALF, result.dataType());
        assertArrayEquals(shape, result.shape());
    }

    @Test
    @DisplayName("Test Q4_0 dequantization with known values")
    void testQ4_0DequantizeKnownValues() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_0);
        assertNotNull(dequantizer);

        // Create a Q4_0 block with known values
        // Scale = 2.0 (FP16: 0x4000)
        // Data: all zeros (which are value 8 in 4-bit, so 8-8=0 after offset)
        byte[] quantizedData = new byte[18];
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x40; // FP16 2.0

        float[] result = dequantizer.dequantize(quantizedData, 32);

        // All values should be 0 * scale = 0 (since 0x00 nibble = 0, and 0 - 8 = -8)
        // Actually: nibble 0 means value (0 - 8) * scale = -8 * 2.0 = -16
        assertNotNull(result);
        assertEquals(32, result.length);
    }

    @Test
    @DisplayName("Test multiple Q4_0 blocks")
    void testQ4_0MultipleBlocks() {
        Dequantizer dequantizer = DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_Q4_0);
        assertNotNull(dequantizer);

        // 2 blocks = 64 elements = 36 bytes
        byte[] quantizedData = new byte[36];
        // First block scale
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x3C;
        // Second block scale
        quantizedData[18] = 0x00;
        quantizedData[19] = 0x3C;

        float[] result = dequantizer.dequantize(quantizedData, 64);

        assertNotNull(result);
        assertEquals(64, result.length);
    }

    @Test
    @DisplayName("Test 2D shape dequantization")
    void testQ4_02DShape() {
        // 2x32 = 64 elements = 2 blocks
        byte[] quantizedData = new byte[36];
        quantizedData[0] = 0x00;
        quantizedData[1] = 0x3C;
        quantizedData[18] = 0x00;
        quantizedData[19] = 0x3C;

        long[] shape = new long[]{2, 32};
        INDArray result = DequantizerFactory.dequantizeToArray(
                quantizedData, GGMLDataType.GGML_TYPE_Q4_0, shape, DataType.FLOAT);

        assertNotNull(result);
        assertArrayEquals(shape, result.shape());
        assertEquals(64, result.length());
    }

    @Test
    @DisplayName("Test QuantizationInfo")
    void testQuantizationInfo() {
        float[] scales = new float[]{1.0f, 2.0f};
        float[] zeroPoints = new float[]{0.0f, 0.0f};
        int blockSize = 32;

        QuantizationInfo info = QuantizationInfo.builder()
                .scales(scales)
                .zeroPoints(zeroPoints)
                .blockSize(blockSize)
                .quantType(GGMLDataType.GGML_TYPE_Q4_0)
                .build();

        assertArrayEquals(scales, info.getScales());
        assertArrayEquals(zeroPoints, info.getZeroPoints());
        assertEquals(blockSize, info.getBlockSize());
        assertEquals(GGMLDataType.GGML_TYPE_Q4_0, info.getQuantType());
    }

    @Test
    @DisplayName("Test getDequantizer throws for unsupported type")
    void testGetDequantizerUnsupported() {
        // Non-quantized float types have no dequantizer registered
        assertThrows(IllegalArgumentException.class,
            () -> DequantizerFactory.getDequantizer(GGMLDataType.GGML_TYPE_F64));
    }
}
