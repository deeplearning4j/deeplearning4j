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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.factory.Nd4j;
import org.eclipse.deeplearning4j.llm.generation.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.PagedKVCache;
import org.eclipse.deeplearning4j.llm.generation.UnifiedKvCacheManager;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link QuantizedKvCacheManager} and the native KVCacheQuantize/KVCacheDequantize ops.
 */
class TestQuantizedKvCacheManager {

    // ==================== Native Op Tests ====================

    @Test
    void testKvCacheQuantizeInt8RoundTrip() {
        // Create a small KV tensor: [4 rows, 8 cols] simulating [heads, headDim]
        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8).muli(10.0);

        // Quantize
        KVCacheQuantize quantOp = new KVCacheQuantize(input, KVCacheQuantize.FORMAT_INT8);
        INDArray[] quantResult = Nd4j.exec(quantOp);
        INDArray quantized = quantResult[0];
        INDArray scales = quantResult[1];

        assertEquals(DataType.INT8, quantized.dataType(), "Quantized output should be INT8");
        assertArrayEquals(new long[]{4, 8}, quantized.shape(), "Quantized shape should match input");
        assertArrayEquals(new long[]{4}, scales.shape(), "Scales should have one per row");

        // Dequantize
        KVCacheDequantize deqOp = new KVCacheDequantize(quantized, scales, KVCacheQuantize.FORMAT_INT8);
        INDArray[] deqResult = Nd4j.exec(deqOp);
        INDArray dequantized = deqResult[0];

        assertEquals(DataType.FLOAT, dequantized.dataType(), "Dequantized should be FLOAT");
        assertArrayEquals(new long[]{4, 8}, dequantized.shape(), "Dequantized shape should match input");

        // Check accuracy: INT8 quantization with scale = absmax/127 has worst-case relative error
        // of scale/(2*|v|) = absmax/(254*|v|). For error < 2%, we need |v| > absmax*0.197.
        // We threshold at 0.2 * per-row absmax to exclude small values dominated by quant noise.
        int numCols = 8;
        double maxRelError = 0.0;
        for (int r = 0; r < 4; r++) {
            // Compute per-row absmax
            double rowAbsMax = 0.0;
            for (int c = 0; c < numCols; c++) {
                rowAbsMax = Math.max(rowAbsMax, Math.abs(input.getFloat(r, c)));
            }
            // Only measure relative error for values >= 20% of row absmax
            double threshold = 0.2 * rowAbsMax;
            for (int c = 0; c < numCols; c++) {
                float orig = input.getFloat(r, c);
                float deq = dequantized.getFloat(r, c);
                if (Math.abs(orig) >= threshold) {
                    double relError = Math.abs((orig - deq) / orig);
                    maxRelError = Math.max(maxRelError, relError);
                }
            }
        }
        assertTrue(maxRelError < 0.02,
                "INT8 round-trip max relative error should be < 2%, got " + maxRelError);
    }

    @Test
    void testKvCacheQuantizeInt4RoundTrip() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8).muli(5.0);

        KVCacheQuantize quantOp = new KVCacheQuantize(input, KVCacheQuantize.FORMAT_INT4);
        INDArray[] quantResult = Nd4j.exec(quantOp);
        INDArray quantized = quantResult[0];
        INDArray scales = quantResult[1];

        assertNotNull(quantized, "INT4 quantized output should not be null");
        assertNotNull(scales, "INT4 scales should not be null");

        KVCacheDequantize deqOp = new KVCacheDequantize(quantized, scales, KVCacheQuantize.FORMAT_INT4);
        INDArray[] deqResult = Nd4j.exec(deqOp);
        INDArray dequantized = deqResult[0];

        // INT4 has much lower precision. scale = absmax/7, step = scale.
        // Worst-case relative error for a value v: scale/(2*|v|) = absmax/(14*|v|).
        // For error < 25%, we need |v| > absmax/(14*0.25) = absmax*0.286.
        // We threshold at 0.3 * per-row absmax to exclude small values dominated by quant noise.
        int numCols = 8;
        double maxRelError = 0.0;
        for (int r = 0; r < 4; r++) {
            // Compute per-row absmax
            double rowAbsMax = 0.0;
            for (int c = 0; c < numCols; c++) {
                rowAbsMax = Math.max(rowAbsMax, Math.abs(input.getFloat(r, c)));
            }
            // Only measure relative error for values >= 30% of row absmax
            double threshold = 0.3 * rowAbsMax;
            for (int c = 0; c < numCols; c++) {
                float orig = input.getFloat(r, c);
                float deq = dequantized.getFloat(r, c);
                if (Math.abs(orig) >= threshold) {
                    double relError = Math.abs((orig - deq) / orig);
                    maxRelError = Math.max(maxRelError, relError);
                }
            }
        }
        assertTrue(maxRelError < 0.25,
                "INT4 round-trip max relative error should be < 25%, got " + maxRelError);
    }

    // ==================== Manager Lifecycle Tests ====================

    @Test
    void testManagerCreationDefaults() {
        UnifiedKvCacheManager manager = new UnifiedKvCacheManager(KvCacheStrategy.QUANTIZED, ModelIOConfig.builder().build());
        assertEquals(KvCacheStrategy.QUANTIZED, manager.getStrategy());
        assertEquals(QuantizedPagedKVCache.QuantFormat.INT8, manager.getQuantFormat());
        assertFalse(manager.isInitialized());
        assertFalse(manager.supportsCudaGraphReplay());
    }

    @ParameterizedTest
    @EnumSource(QuantizedPagedKVCache.QuantFormat.class)
    void testManagerCreationWithFormat(QuantizedPagedKVCache.QuantFormat format) {
        UnifiedKvCacheManager manager = new UnifiedKvCacheManager(KvCacheStrategy.QUANTIZED, ModelIOConfig.builder().build(),
                format, DataType.FLOAT, 3, PagedKVCache.DEFAULT_BLOCK_SIZE);
        assertEquals(format, manager.getQuantFormat());
        assertEquals(KvCacheStrategy.QUANTIZED, manager.getStrategy());
    }

    @Test
    void testManagerCloseWithoutInit() {
        UnifiedKvCacheManager manager = new UnifiedKvCacheManager(KvCacheStrategy.QUANTIZED, ModelIOConfig.builder().build());
        assertDoesNotThrow(manager::close);
    }

    @Test
    void testManagerGetStaticKvBuffersBeforeInit() {
        UnifiedKvCacheManager manager = new UnifiedKvCacheManager(KvCacheStrategy.QUANTIZED, ModelIOConfig.builder().build());
        assertNull(manager.getStaticKvBuffers(),
                "getStaticKvBuffers should return null before initialization");
    }
}
