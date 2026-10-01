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
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.kvcache.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.kvcache.PagedKVCache;
import org.eclipse.deeplearning4j.llm.generation.kvcache.QuantizedPagedKVCache;
import org.eclipse.deeplearning4j.llm.generation.kvcache.UnifiedKvCacheManager;

import java.util.Arrays;

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

    // ==================== Native Op Layout Contract ====================
    // Rows run along the last dimension and every operand is addressed through its own shape and
    // strides, so a view, a permutation or a caller-provided output in any order must produce
    // exactly the values of the contiguous tensor.

    @ParameterizedTest
    @ValueSource(ints = {KVCacheQuantize.FORMAT_INT8, KVCacheQuantize.FORMAT_INT4})
    void testKvCacheQuantizeStridedLayoutsMatchContiguous(int format) {
        INDArray reference = Nd4j.randn(DataType.FLOAT, 3, 5, 16).muli(6.0);
        INDArray[] expected = Nd4j.exec(new KVCacheQuantize(reference, format));
        int[] expectedQ = logicalInts(expected[0]);
        float[] expectedS = logicalFloats(expected[1]);

        // The same logical [3, 5, 16] tensor with last-dimension strides 15, 3 and 2 (the last at an offset).
        INDArray permuted = Nd4j.create(DataType.FLOAT, 16, 5, 3).permute(2, 1, 0).assign(reference);
        INDArray scrambled = Nd4j.create(DataType.FLOAT, 5, 16, 3).permute(2, 0, 1).assign(reference);
        INDArray stepped = Nd4j.create(DataType.FLOAT, 4, 5, 32)
                .get(NDArrayIndex.interval(1, 4), NDArrayIndex.all(), NDArrayIndex.interval(0, 2, 32)).assign(reference);
        assertEquals(15, permuted.stride(2));
        assertEquals(3, scrambled.stride(2));
        assertEquals(2, stepped.stride(2));

        for (INDArray view : new INDArray[]{permuted, scrambled, stepped}) {
            String layout = "strides " + Arrays.toString(view.stride());
            INDArray[] actual = Nd4j.exec(new KVCacheQuantize(view, format));
            assertArrayEquals(expectedQ, logicalInts(actual[0]), "quantized values, " + layout);
            assertArrayEquals(expectedS, logicalFloats(actual[1]), 0.0f, "scales, " + layout);

            // Caller-provided outputs: F-order quantized, transposed-view scales.
            INDArray quantized = Nd4j.create(DataType.INT8, new long[]{3, 5, 16}, 'f');
            INDArray scales = Nd4j.create(DataType.FLOAT, 5, 3).permute(1, 0);
            Nd4j.exec(DynamicCustomOp.builder("kv_cache_quantize").addInputs(view).addOutputs(quantized, scales)
                    .addIntegerArguments(format).build());
            assertArrayEquals(expectedQ, logicalInts(quantized), "provided quantized output, " + layout);
            assertArrayEquals(expectedS, logicalFloats(scales), 0.0f, "provided scales output, " + layout);
        }

        // Dequantize from an F-order quantized tensor and transposed scales into a permuted output.
        float[] expectedZ = logicalFloats(Nd4j.exec(new KVCacheDequantize(expected[0], expected[1], format))[0]);
        INDArray scalesView = Nd4j.create(DataType.FLOAT, 5, 3).permute(1, 0).assign(expected[1]);
        INDArray output = Nd4j.create(DataType.FLOAT, 16, 5, 3).permute(2, 1, 0);
        Nd4j.exec(DynamicCustomOp.builder("kv_cache_dequantize").addInputs(expected[0].dup('f'), scalesView)
                .addOutputs(output).addIntegerArguments(format).build());
        assertArrayEquals(expectedZ, logicalFloats(output), 0.0f, "dequantized values");
    }

    @Test
    void testKvCacheQuantizeRowInlineLayout() {
        INDArray reference = Nd4j.randn(DataType.FLOAT, 2, 3, 8).muli(4.0);
        INDArray[] separate = Nd4j.exec(new KVCacheQuantize(reference, KVCacheQuantize.FORMAT_INT8));
        int[] separateQ = logicalInts(separate[0]);
        float[] separateS = logicalFloats(separate[1]);

        INDArray permuted = Nd4j.create(DataType.FLOAT, 8, 3, 2).permute(2, 1, 0).assign(reference);
        for (INDArray input : new INDArray[]{reference, permuted}) {
            INDArray rowInline = Nd4j.create(DataType.INT8, 2, 3, 12);
            INDArray unusedScales = Nd4j.createFromArray(42.0f);
            Nd4j.exec(DynamicCustomOp.builder("kv_cache_quantize").addInputs(input).addOutputs(rowInline, unusedScales)
                    .addIntegerArguments(KVCacheQuantize.FORMAT_INT8, 1).build());
            assertEquals(0.0f, unusedScales.getFloat(0), 0.0f, "the unused scales output must still be written");

            // Each row holds its 8 quantized values followed by its FLOAT32 scale, little-endian.
            int[] bytes = logicalInts(rowInline);
            for (int row = 0; row < 6; row++) {
                assertArrayEquals(Arrays.copyOfRange(separateQ, row * 8, row * 8 + 8),
                        Arrays.copyOfRange(bytes, row * 12, row * 12 + 8), "row " + row + " values");
                int scaleBits = (bytes[row * 12 + 8] & 0xFF) | (bytes[row * 12 + 9] & 0xFF) << 8
                        | (bytes[row * 12 + 10] & 0xFF) << 16 | (bytes[row * 12 + 11] & 0xFF) << 24;
                assertEquals(Float.floatToIntBits(separateS[row]), scaleBits, "row " + row + " inline scale");
            }
        }

        // The scale slot is written as raw row bytes, so the row-inline output needs a unit last stride.
        INDArray fOrder = Nd4j.create(DataType.INT8, new long[]{2, 3, 12}, 'f');
        assertThrows(RuntimeException.class, () -> Nd4j.exec(DynamicCustomOp.builder("kv_cache_quantize")
                .addInputs(reference).addOutputs(fOrder, Nd4j.createFromArray(0.0f))
                .addIntegerArguments(KVCacheQuantize.FORMAT_INT8, 1).build()));
    }

    @Test
    void testKvCacheQuantizeInt8RoundsHalfToEven() {
        // absmax 127 gives a unit scale, so each value is its own level before rounding.
        INDArray input = Nd4j.createFromArray(new float[][]{{127f, 2.5f, -2.5f, 0.5f, 1.5f, -127f}});
        INDArray[] out = Nd4j.exec(new KVCacheQuantize(input, KVCacheQuantize.FORMAT_INT8));
        assertEquals(1.0f, out[1].getFloat(0), 0.0f);
        assertArrayEquals(new int[]{127, 2, -2, 0, 2, -127}, logicalInts(out[0]));
    }

    @Test
    void testKvCacheQuantizeInt4PacksEachRowFromItsOwnStart() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 7).muli(3.0);
        // absmax 7 gives row 2 a unit scale: half-to-even rounding (2.5 -> 2, 0.5 -> 0), and the odd
        // tail element packs against a zero high nibble.
        input.putRow(2, Nd4j.createFromArray(7f, 3.5f, -3.5f, 2.5f, 0.5f, -7f, 1f));
        INDArray[] whole = Nd4j.exec(new KVCacheQuantize(input, KVCacheQuantize.FORMAT_INT4));
        int[] q = logicalInts(whole[0]);
        float[] s = logicalFloats(whole[1]);

        assertEquals(1.0f, s[2], 0.0f);
        assertArrayEquals(new int[]{(byte) 0xCF, (byte) 0xA4, 0x18, (byte) 0x89, 0, 0, 0},
                Arrays.copyOfRange(q, 14, 21), "row 2 packed nibbles");
        for (int row = 0; row < 4; row++) {
            int[] rowBytes = Arrays.copyOfRange(q, row * 7, row * 7 + 7);
            assertArrayEquals(new int[3], Arrays.copyOfRange(rowBytes, 4, 7), "row " + row + " bytes past the packed half");
            // Any leading-dimension slice is itself a valid INT4 tensor.
            INDArray[] single = Nd4j.exec(new KVCacheQuantize(
                    input.get(NDArrayIndex.interval(row, row + 1), NDArrayIndex.all()), KVCacheQuantize.FORMAT_INT4));
            assertArrayEquals(rowBytes, logicalInts(single[0]), "row " + row + " quantized alone");
            assertEquals(s[row], single[1].getFloat(0), 0.0f, "row " + row + " scale");
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {KVCacheQuantize.FORMAT_INT8, KVCacheQuantize.FORMAT_INT4})
    void testKvCacheQuantizeHalfMatchesFloat(int format) {
        // Half accumulates in float, so a half tensor quantizes exactly like its float image.
        INDArray half = Nd4j.randn(DataType.FLOAT, 4, 16).muli(3.0).castTo(DataType.HALF);
        INDArray[] fromHalf = Nd4j.exec(new KVCacheQuantize(half, format));
        INDArray[] fromFloat = Nd4j.exec(new KVCacheQuantize(half.castTo(DataType.FLOAT), format));
        assertArrayEquals(logicalInts(fromFloat[0]), logicalInts(fromHalf[0]));
        assertArrayEquals(logicalFloats(fromFloat[1]), logicalFloats(fromHalf[1]), 0.0f);

        INDArray dequantizedFloat = Nd4j.exec(new KVCacheDequantize(fromHalf[0], fromHalf[1], format))[0];
        INDArray dequantizedHalf = Nd4j.create(DataType.HALF, 4, 16);
        Nd4j.exec(DynamicCustomOp.builder("kv_cache_dequantize").addInputs(fromHalf[0], fromHalf[1])
                .addOutputs(dequantizedHalf).addIntegerArguments(format).build());
        assertArrayEquals(logicalFloats(dequantizedFloat.castTo(DataType.HALF)), logicalFloats(dequantizedHalf), 0.0f);
    }

    @Test
    void testKvCacheQuantizeEmptyInput() {
        INDArray[] out = Nd4j.exec(new KVCacheQuantize(Nd4j.create(DataType.FLOAT, 0, 8), KVCacheQuantize.FORMAT_INT8));
        assertTrue(out[0].isEmpty());
        assertArrayEquals(new long[]{0, 8}, out[0].shape());
        assertTrue(out[1].isEmpty());
        INDArray dequantized = Nd4j.exec(new KVCacheDequantize(out[0], out[1], KVCacheQuantize.FORMAT_INT8))[0];
        assertTrue(dequantized.isEmpty());
        assertArrayEquals(new long[]{0, 8}, dequantized.shape());
    }

    @Test
    void testKvCacheQuantizeRejectsUnsupportedContracts() {
        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        assertThrows(RuntimeException.class,
                () -> Nd4j.exec(new KVCacheQuantize(input, KVCacheQuantize.FORMAT_INT4, true)),
                "INT4 packs nibbles and has no row-inline scale layout");
        assertThrows(RuntimeException.class, () -> Nd4j.exec(new KVCacheQuantize(input, 7)), "unknown format");
        INDArray quantized = Nd4j.exec(new KVCacheQuantize(input, KVCacheQuantize.FORMAT_INT8))[0];
        assertThrows(RuntimeException.class,
                () -> Nd4j.exec(new KVCacheDequantize(quantized, Nd4j.ones(DataType.FLOAT, 3), KVCacheQuantize.FORMAT_INT8)),
                "one scale per row");
    }

    /** Row-major logical values, whatever the array's order, strides or offset. */
    private static int[] logicalInts(INDArray array) {
        return array.castTo(DataType.INT).dup('c').reshape(array.length()).toIntVector();
    }

    private static float[] logicalFloats(INDArray array) {
        return array.castTo(DataType.FLOAT).dup('c').reshape(array.length()).toFloatVector();
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
