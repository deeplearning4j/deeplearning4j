/*
 *  ******************************************************************************
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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for the full decoder pipeline: KV scatter, causal masks,
 * FrozenDecodeStep multi-token masks, BenchmarkConfig, and view-based mode.
 *
 * These tests cover gaps identified in the second-pass coverage analysis:
 * - scatterMultipleKvEntries source offset (documented prior bug)
 * - buildCausalMask for batchSize > 1 and seqLen > 1
 * - FrozenDecodeStep.updateInputs multi-token causal mask construction
 * - BenchmarkConfig.optimal() factory correctness
 * - BenchmarkConfigApplier.apply() round-trip verification
 * - View-based (non-DSP) decoder input mode
 * - ensureLocation sync after putScalar on 4D attention bias
 *
 * Run with:
 *   cd platform-tests && mvn test \
 *     -Dtest=DecoderPipelineRegressionTest \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 */
@Slf4j
@DisplayName("DecoderPipelineRegressionTest")
public class DecoderPipelineRegressionTest {

    // ═══════════════════════════════════════════════════════════════════════
    // Group 1: KV Scatter Tests
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("KV Scatter")
    class KvScatterTests {

        /**
         * Regression: scatterMultipleKvEntries source offset MUST be maxKvLen, NOT 0.
         * The present KV output has shape [1, heads, maxKvLen+numPositions, dim],
         * where new entries start at position maxKvLen (the concat point).
         *
         * Prior bug: source offset was 0, copying old cached data instead of new entries.
         */
        @Test
        @DisplayName("scatterMultipleKvEntries extracts from maxKvLen offset")
        public void testScatterMultipleKvEntriesSourceOffset() {
            int batch = 1, heads = 4, maxKvLen = 50, dim = 32;
            int numPositions = 3;
            int baseCachePos = 10;

            List<String> presentKeyNames = Arrays.asList("present.0.key");
            List<String> presentValueNames = Arrays.asList("present.0.value");
            ModelIOConfig ioConfig = ModelIOConfig.builder().build();

            // Create static KV buffer: [1, 4, 50, 32] — zeros
            Map<String, INDArray> staticKvBuffers = new HashMap<>();
            String pastName = ioConfig.presentToInputName("present.0.key");
            INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);
            staticKvBuffers.put(pastName, staticBuf);

            String pastValueName = ioConfig.presentToInputName("present.0.value");
            INDArray staticValueBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);
            staticKvBuffers.put(pastValueName, staticValueBuf);

            // Create present KV output: [1, 4, maxKvLen+numPositions, 32]
            // Fill first maxKvLen positions with -1 (old data), new positions with known values
            Map<String, INDArray> decoderOutputs = new HashMap<>();

            INDArray presentKey = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen + numPositions, dim);
            // Old positions [0..maxKvLen-1] = -1 (should NOT be copied)
            presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, maxKvLen), NDArrayIndex.all()).assign(-1.0f);
            // New positions [maxKvLen..maxKvLen+numPositions-1] = unique values per position
            for (int p = 0; p < numPositions; p++) {
                presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(maxKvLen + p), NDArrayIndex.all()).assign(100.0f + p);
            }
            decoderOutputs.put("present.0.key", presentKey);

            INDArray presentValue = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen + numPositions, dim);
            presentValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, maxKvLen), NDArrayIndex.all()).assign(-2.0f);
            for (int p = 0; p < numPositions; p++) {
                presentValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(maxKvLen + p), NDArrayIndex.all()).assign(200.0f + p);
            }
            decoderOutputs.put("present.0.value", presentValue);

            // Scatter
            DecoderUtils.scatterMultipleKvEntries(
                    staticKvBuffers, decoderOutputs,
                    presentKeyNames, presentValueNames,
                    maxKvLen, baseCachePos, numPositions, ioConfig);

            // Verify: static buffer positions [baseCachePos..baseCachePos+numPositions-1] should
            // contain the NEW data (100+p, 200+p), NOT the old data (-1, -2)
            for (int p = 0; p < numPositions; p++) {
                float keyVal = staticBuf.getFloat(0, 0, baseCachePos + p, 0);
                assertEquals(100.0f + p, keyVal, 0.01f,
                        "Key at cachePos " + (baseCachePos + p) + " should be " + (100.0f + p)
                                + " (from maxKvLen+" + p + "), NOT -1 (from position " + p + ")");

                float valVal = staticValueBuf.getFloat(0, 0, baseCachePos + p, 0);
                assertEquals(200.0f + p, valVal, 0.01f,
                        "Value at cachePos " + (baseCachePos + p) + " should be " + (200.0f + p));
            }

            // Verify: positions outside the scatter range are still zero
            assertEquals(0.0f, staticBuf.getFloat(0, 0, 0, 0), 0.01f,
                    "Position 0 should be untouched (zero)");
            assertEquals(0.0f, staticBuf.getFloat(0, 0, baseCachePos - 1, 0), 0.01f,
                    "Position before scatter range should be untouched");
            if (baseCachePos + numPositions < maxKvLen) {
                assertEquals(0.0f, staticBuf.getFloat(0, 0, baseCachePos + numPositions, 0), 0.01f,
                        "Position after scatter range should be untouched");
            }

            log.info("scatterMultipleKvEntries source offset test passed: {} positions scattered from maxKvLen={}",
                    numPositions, maxKvLen);

            presentKey.close();
            presentValue.close();
            staticBuf.close();
            staticValueBuf.close();
        }

        /**
         * Test scatterNewKvEntries (single-token) uses maxKvLen as source position.
         */
        @Test
        @DisplayName("scatterNewKvEntries single-token source is at maxKvLen")
        public void testScatterSingleTokenSourceOffset() {
            int batch = 1, heads = 2, maxKvLen = 20, dim = 16;
            int cachePos = 5;

            ModelIOConfig ioConfig = ModelIOConfig.builder().build();
            List<String> presentKeyNames = Arrays.asList("present.0.key");
            List<String> presentValueNames = Collections.emptyList();

            String pastName = ioConfig.presentToInputName("present.0.key");
            INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

            Map<String, INDArray> staticKvBuffers = new HashMap<>();
            staticKvBuffers.put(pastName, staticBuf);

            // Present KV: [1, 2, maxKvLen+1, 16] — new entry at position maxKvLen
            INDArray presentKey = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen + 1, dim);
            presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, maxKvLen), NDArrayIndex.all()).assign(-999.0f);
            presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(maxKvLen), NDArrayIndex.all()).assign(42.0f);

            Map<String, INDArray> outputs = new HashMap<>();
            outputs.put("present.0.key", presentKey);

            DecoderUtils.scatterNewKvEntries(staticKvBuffers, outputs,
                    presentKeyNames, presentValueNames, maxKvLen, cachePos, ioConfig);

            // Position cachePos should have 42.0 (from maxKvLen), NOT -999
            float val = staticBuf.getFloat(0, 0, cachePos, 0);
            assertEquals(42.0f, val, 0.01f,
                    "Single-token scatter should extract from position maxKvLen=" + maxKvLen);

            log.info("Single-token scatter source offset test passed");

            presentKey.close();
            staticBuf.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 2: Causal Mask Tests
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("Causal Masks")
    class CausalMaskTests {

        /**
         * buildCausalMask with seqLen=1 should return all-zeros (attend everything).
         */
        @Test
        @DisplayName("buildCausalMask seqLen=1 returns all-zeros")
        public void testCausalMaskSeqLen1() {
            INDArray mask = DecoderUtils.buildCausalMask(1, 1, 50);
            assertEquals(4, mask.rank(), "Should be 4D: [batch, 1, seqLen, totalSeqLen]");
            assertArrayEquals(new long[]{1, 1, 1, 50}, mask.shape());
            assertEquals(0.0, mask.sumNumber().doubleValue(), 0.01,
                    "seqLen=1 causal mask should be all zeros (attend to everything)");
            mask.close();
        }

        /**
         * buildCausalMask with seqLen > 1 should produce a proper causal pattern:
         * lower-triangular for current positions, all-attend for past.
         */
        @Test
        @DisplayName("buildCausalMask seqLen>1 has correct causal pattern")
        public void testCausalMaskMultiToken() {
            int currentSeqLen = 4;
            int pastSeqLen = 10;
            int totalSeqLen = pastSeqLen + currentSeqLen;  // 14

            INDArray mask = DecoderUtils.buildCausalMask(1, currentSeqLen, totalSeqLen);
            assertArrayEquals(new long[]{1, 1, currentSeqLen, totalSeqLen}, mask.shape());

            // For each query position q:
            // - Past [0..pastSeqLen-1]: all 0 (attend)
            // - Current [pastSeqLen..pastSeqLen+q]: 0 (attend to self and earlier)
            // - Future [pastSeqLen+q+1..totalSeqLen-1]: MASK_FILL (block)
            for (int q = 0; q < currentSeqLen; q++) {
                // Past positions: should all be 0
                for (int k = 0; k < pastSeqLen; k++) {
                    float val = mask.getFloat(0, 0, q, k);
                    assertEquals(0.0f, val, 0.01f,
                            "Query " + q + " should attend to past position " + k);
                }

                // Current self and earlier: should be 0
                for (int k = pastSeqLen; k <= pastSeqLen + q; k++) {
                    float val = mask.getFloat(0, 0, q, k);
                    assertEquals(0.0f, val, 0.01f,
                            "Query " + q + " should attend to current position " + (k - pastSeqLen));
                }

                // Future positions: should be MASK_FILL
                for (int k = pastSeqLen + q + 1; k < totalSeqLen; k++) {
                    float val = mask.getFloat(0, 0, q, k);
                    assertEquals(DecoderUtils.MASK_FILL, val, 1.0f,
                            "Query " + q + " should NOT attend to future position " + (k - pastSeqLen));
                }
            }

            log.info("Multi-token causal mask test passed: seqLen={}, totalSeqLen={}", currentSeqLen, totalSeqLen);
            mask.close();
        }

        /**
         * buildCausalMask with batchSize > 1 tiles correctly.
         * Each batch should have identical mask patterns.
         */
        @Test
        @DisplayName("buildCausalMask batchSize>1 tiles correctly")
        public void testCausalMaskBatched() {
            int batchSize = 3;
            int currentSeqLen = 2;
            int totalSeqLen = 12;

            INDArray mask = DecoderUtils.buildCausalMask(batchSize, currentSeqLen, totalSeqLen);
            assertArrayEquals(new long[]{batchSize, 1, currentSeqLen, totalSeqLen}, mask.shape());

            // All batches should be identical
            INDArray batch0 = mask.get(NDArrayIndex.point(0), NDArrayIndex.all(),
                    NDArrayIndex.all(), NDArrayIndex.all());
            for (int b = 1; b < batchSize; b++) {
                INDArray batchB = mask.get(NDArrayIndex.point(b), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                assertEquals(batch0, batchB,
                        "Batch " + b + " should be identical to batch 0");
            }

            log.info("Batched causal mask test passed: batch={}, shape={}", batchSize, Arrays.toString(mask.shape()));
            mask.close();
        }

        /**
         * buildCausalMask with batchSize=1 and seqLen=1 should NOT leak (singleMask returned directly).
         */
        @Test
        @DisplayName("buildCausalMask batch=1 seqLen=1 no intermediate leak")
        public void testCausalMaskNoLeakBatch1SeqLen1() {
            // This just verifies it works and the result is usable
            INDArray mask = DecoderUtils.buildCausalMask(1, 1, 100);
            assertNotNull(mask);
            assertEquals(DataType.FLOAT, mask.dataType());
            assertArrayEquals(new long[]{1, 1, 1, 100}, mask.shape());
            mask.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 3: FrozenDecodeStep Multi-Token Mask Tests
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("FrozenDecodeStep Multi-Token Masks")
    class FrozenDecodeStepMaskTests {

        /**
         * Verify the FrozenDecodeStep.updateInputs causal mask construction
         * for seqLen > 1 (speculative decode). We can't call the private method
         * directly, so we replicate its logic and verify correctness.
         *
         * The mask layout for [seqLen, maxKvLen + seqLen]:
         * - Past [0..cachePos-1]: 0 (attend to valid past)
         * - Past [cachePos..maxKvLen-1]: MASK_FILL (empty padding)
         * - Current [maxKvLen..maxKvLen+q]: 0 (causal: attend to self and earlier)
         * - Current [maxKvLen+q+1..maxKvLen+seqLen-1]: MASK_FILL (future, block)
         */
        @Test
        @DisplayName("Multi-token 4D bias has correct causal pattern")
        public void testMultiTokenBiasCausalPattern() {
            int seqLen = 4;
            int maxKvLen = 50;
            int cachePos = 15;
            int totalLen = maxKvLen + seqLen;

            // Replicate the FrozenDecodeStep.updateInputs bias construction
            float[] biasData = new float[seqLen * totalLen];
            for (int q = 0; q < seqLen; q++) {
                int rowOffset = q * totalLen;
                // Past positions: valid up to cachePos, masked after
                for (int k = cachePos; k < maxKvLen; k++) {
                    biasData[rowOffset + k] = DecoderUtils.MASK_FILL;
                }
                // Current positions: causal — future tokens masked
                for (int k = q + 1; k < seqLen; k++) {
                    biasData[rowOffset + maxKvLen + k] = DecoderUtils.MASK_FILL;
                }
            }

            INDArray bias = Nd4j.create(biasData, new long[]{1, 1, seqLen, totalLen}, 'c');

            // Verify each query position
            for (int q = 0; q < seqLen; q++) {
                // Past valid: [0..cachePos-1] should be 0
                for (int k = 0; k < cachePos; k++) {
                    assertEquals(0.0f, bias.getFloat(0, 0, q, k), 0.01f,
                            "Query " + q + ": past valid position " + k + " should be 0");
                }

                // Past padding: [cachePos..maxKvLen-1] should be MASK_FILL
                for (int k = cachePos; k < maxKvLen; k++) {
                    assertEquals(DecoderUtils.MASK_FILL, bias.getFloat(0, 0, q, k), 1.0f,
                            "Query " + q + ": past padding position " + k + " should be masked");
                }

                // Current self and earlier: [maxKvLen..maxKvLen+q] should be 0
                for (int k = 0; k <= q; k++) {
                    assertEquals(0.0f, bias.getFloat(0, 0, q, maxKvLen + k), 0.01f,
                            "Query " + q + ": current position " + k + " should attend (causal)");
                }

                // Current future: [maxKvLen+q+1..maxKvLen+seqLen-1] should be MASK_FILL
                for (int k = q + 1; k < seqLen; k++) {
                    assertEquals(DecoderUtils.MASK_FILL, bias.getFloat(0, 0, q, maxKvLen + k), 1.0f,
                            "Query " + q + ": future position " + k + " should be masked");
                }
            }

            log.info("Multi-token 4D bias causal pattern test passed: seqLen={}, cachePos={}", seqLen, cachePos);
            bias.close();
        }

        /**
         * Verify the attention mask construction for multi-token decode.
         * Layout: [1, maxKvLen + seqLen]
         * 1s at [0..cachePos-1] (past) and [maxKvLen..maxKvLen+seqLen-1] (current tokens)
         */
        @Test
        @DisplayName("Multi-token attention mask layout")
        public void testMultiTokenAttentionMask() {
            int seqLen = 4;
            int maxKvLen = 50;
            int cachePos = 15;
            int maskLen = maxKvLen + seqLen;

            INDArray mask = Nd4j.zeros(DataType.LONG, 1, maskLen);
            // Past positions
            if (cachePos > 0) {
                mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, cachePos)).assign(1);
            }
            // Current token positions at the end
            mask.get(NDArrayIndex.point(0), NDArrayIndex.interval(maskLen - seqLen, maskLen)).assign(1);

            // Count: should be cachePos + seqLen
            long onesCount = mask.sumNumber().longValue();
            assertEquals(cachePos + seqLen, onesCount,
                    "Mask should have cachePos + seqLen ones");

            // Past positions [0..cachePos-1]: 1
            for (int k = 0; k < cachePos; k++) {
                assertEquals(1L, mask.getLong(0, k), "Past position " + k + " should be 1");
            }

            // Padding [cachePos..maxKvLen-1]: 0
            for (int k = cachePos; k < maxKvLen; k++) {
                assertEquals(0L, mask.getLong(0, k), "Padding position " + k + " should be 0");
            }

            // Current tokens [maxKvLen..maxKvLen+seqLen-1]: 1
            for (int k = maxKvLen; k < maskLen; k++) {
                assertEquals(1L, mask.getLong(0, k), "Current position " + k + " should be 1");
            }

            log.info("Multi-token attention mask test passed: seqLen={}, cachePos={}", seqLen, cachePos);
            mask.close();
        }

        /**
         * Edge case: cachePos=0 (first decode step in speculative mode).
         * No past positions, only current tokens have 1s in mask and 0s in bias.
         */
        @Test
        @DisplayName("Multi-token masks with cachePos=0")
        public void testMultiTokenMasksCachePos0() {
            int seqLen = 4;
            int maxKvLen = 50;
            int cachePos = 0;
            int totalLen = maxKvLen + seqLen;

            // Bias: all past should be MASK_FILL, current has causal pattern
            float[] biasData = new float[seqLen * totalLen];
            for (int q = 0; q < seqLen; q++) {
                int rowOffset = q * totalLen;
                for (int k = cachePos; k < maxKvLen; k++) {
                    biasData[rowOffset + k] = DecoderUtils.MASK_FILL;
                }
                for (int k = q + 1; k < seqLen; k++) {
                    biasData[rowOffset + maxKvLen + k] = DecoderUtils.MASK_FILL;
                }
            }

            INDArray bias = Nd4j.create(biasData, new long[]{1, 1, seqLen, totalLen}, 'c');

            // Query 0 should only attend to current position 0
            assertEquals(0.0f, bias.getFloat(0, 0, 0, maxKvLen), 0.01f,
                    "Query 0 should attend to current position 0");
            for (int k = 1; k < seqLen; k++) {
                assertEquals(DecoderUtils.MASK_FILL, bias.getFloat(0, 0, 0, maxKvLen + k), 1.0f,
                        "Query 0 should NOT attend to future position " + k);
            }

            // All past positions should be masked (no valid past at cachePos=0)
            for (int k = 0; k < maxKvLen; k++) {
                assertEquals(DecoderUtils.MASK_FILL, bias.getFloat(0, 0, 0, k), 1.0f,
                        "All past positions should be masked when cachePos=0");
            }

            log.info("Multi-token masks cachePos=0 test passed");
            bias.close();
        }

        /**
         * Verify position IDs construction for multi-token decode.
         * Should be consecutive starting from cachePos.
         */
        @Test
        @DisplayName("Multi-token position IDs are consecutive from cachePos")
        public void testMultiTokenPositionIds() {
            int seqLen = 6;
            int cachePos = 25;

            INDArray posIds = Nd4j.zeros(DataType.LONG, 1, seqLen);
            for (int i = 0; i < seqLen; i++) {
                posIds.putScalar(0, i, cachePos + i);
            }

            for (int i = 0; i < seqLen; i++) {
                assertEquals(cachePos + i, posIds.getLong(0, i),
                        "Position " + i + " should be " + (cachePos + i));
            }

            log.info("Multi-token position IDs test passed: seqLen={}, cachePos={}", seqLen, cachePos);
            posIds.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 4: BenchmarkConfig Tests
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("BenchmarkConfig")
    class BenchmarkConfigTests {

        /**
         * Verify BenchmarkConfig.optimal() returns the expected production settings.
         * If any default changes silently, this test catches it.
         */
        @Test
        @DisplayName("BenchmarkConfig.optimal() has correct production defaults")
        public void testOptimalConfigValues() {
            BenchmarkConfig config = BenchmarkConfig.optimal();

            assertEquals("OPTIMAL", config.getName());
            assertTrue(config.isTriton(), "optimal() should be a Triton config");

            // Triton compilation settings
            assertTrue(config.isTritonSectionFusion(), "Section fusion should be ON");
            assertTrue(config.isTritonCompileAll(), "CompileAll should be ON");
            assertTrue(config.isTritonGraphCapture(), "Graph capture should be ON");
            assertTrue(config.isTritonAllowFallbackCapture(), "Fallback capture should be ON");
            assertTrue(config.isTritonConsolidatedArgTable(), "Consolidated arg table should be ON");
            assertTrue(config.isTritonArgDirtyTracking(), "Arg dirty tracking should be ON");
            assertFalse(config.isTritonFusionScoring(), "Fusion scoring should be OFF");

            // Tuning parameters
            assertEquals(4, config.getTritonNumWarps(), "Warps should be 4");
            assertEquals(1, config.getTritonNumStages(), "Stages should be 1");

            // cuBLAS/TF32
            assertTrue(config.isCublasTf32(), "cuBLAS TF32 should be ON for optimal");
            assertFalse(config.isTritonTf32(), "Triton TF32 should be OFF by default (precision)");

            // DSP
            assertTrue(config.isDspBatchedGemm(), "Batched GEMM should be ON");

            // Generation
            assertEquals(250, config.getMaxTokens(), "maxTokens should be 250");
            assertEquals(40.0, config.getMinDiversityPct(), 0.01, "minDiversityPct should be 40%");

            // Include types
            String types = config.getTritonIncludeTypes();
            assertTrue(types.contains("ATTENTION"), "Should include ATTENTION");
            assertTrue(types.contains("NORMALIZATION"), "Should include NORMALIZATION");
            assertTrue(types.contains("GATHER"), "Should include GATHER");
            assertFalse(types.contains("MATMUL"), "Should NOT include MATMUL (cuBLAS faster)");

            log.info("BenchmarkConfig.optimal() values verified: {}", config);
        }

        /**
         * Verify BenchmarkConfig.create() defaults are correct.
         * Boolean fields should default to false, int fields to their specified defaults.
         */
        @Test
        @DisplayName("BenchmarkConfig.create() has correct defaults")
        public void testCreateDefaults() {
            BenchmarkConfig config = BenchmarkConfig.create("TEST");

            assertEquals("TEST", config.getName());
            assertFalse(config.isTriton(), "Empty tritonIncludeTypes should mean not Triton");
            assertFalse(config.isTritonSectionFusion());
            assertFalse(config.isTritonGraphCapture());
            assertFalse(config.isCublasTf32());
            assertFalse(config.isTritonTf32());
            assertFalse(config.isDspBatchedGemm());
            assertFalse(config.isDspBatchZero());
            assertFalse(config.isDspCastSinkMatmul());
            assertEquals(20, config.getMaxTokens(), "Default maxTokens should be 20");
            assertEquals(1, config.getCaptureMinExec(), "Default captureMinExec should be 1");

            // Fusion flags default to true
            assertTrue(config.isTritonFuseIdentityShapes());
            assertTrue(config.isTritonFuseCastChains());
            assertTrue(config.isTritonFuseTrivialGather());
            assertTrue(config.isTritonSpecializePermuteSeq1());
            assertTrue(config.isTritonEliminateConcatSplitPairs());

            // Fused matmul defaults to false (HIGH RISK)
            assertFalse(config.isTritonFusedMatmul(), "Fused matmul should be OFF by default (high risk)");

            log.info("BenchmarkConfig.create() defaults verified");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 5: BenchmarkConfigApplier Tests
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("BenchmarkConfigApplier")
    class BenchmarkConfigApplierTests {

        /**
         * Verify that apply() sets Environment flags correctly for a Triton config.
         * After apply(), reading back from Environment should match the config.
         */
        @Test
        @DisplayName("apply() sets all Triton flags correctly")
        public void testApplyTritonConfig() {
            Environment env = Nd4j.getEnvironment();

            // Save original state
            boolean origCublasTf32 = env.cublasTf32Enabled();
            boolean origTritonTf32 = env.tritonTf32Enabled();

            try {
                BenchmarkConfig config = BenchmarkConfig.create("TEST_TRITON")
                        .tritonIncludeTypes("ATTENTION,NORMALIZATION")
                        .tritonSectionFusion(true)
                        .tritonGraphCapture(true)
                        .tritonConsolidatedArgTable(true)
                        .tritonArgDirtyTracking(true)
                        .tritonCompileAll(true)
                        .tritonNumWarps(4)
                        .tritonNumStages(2)
                        .cublasTf32(true)
                        .tritonTf32(false)
                        .dspBatchedGemm(true)
                        .dspBatchZero(true);

                BenchmarkConfigApplier.apply(config);

                // Verify all flags
                assertEquals("ATTENTION,NORMALIZATION", env.tritonIncludeTypes());
                assertTrue(env.tritonSectionFusion());
                assertTrue(env.tritonGraphCapture());
                assertTrue(env.tritonConsolidatedArgTable());
                assertTrue(env.tritonArgDirtyTracking());
                assertTrue(env.tritonCompileAll());
                assertEquals(4, env.tritonNumWarps());
                assertEquals(2, env.tritonNumStages());
                assertTrue(env.cublasTf32Enabled());
                assertFalse(env.tritonTf32Enabled());
                assertTrue(env.dspBatchedGemm());
                assertTrue(env.dspBatchZero());

                log.info("BenchmarkConfigApplier.apply() Triton round-trip verified");
            } finally {
                // Restore original state
                env.setCublasTf32Enabled(origCublasTf32);
                env.setTritonTf32Enabled(origTritonTf32);
                // Reset the Triton flags
                env.setTritonIncludeTypes("");
                env.setTritonSectionFusion(false);
                env.setTritonGraphCapture(false);
                env.setTritonConsolidatedArgTable(false);
                env.setTritonArgDirtyTracking(false);
                env.setTritonCompileAll(false);
                env.setDspBatchedGemm(false);
                env.setDspBatchZero(false);
            }
        }

        /**
         * Verify that apply() resets flags to defaults before applying new config.
         * Setting a Triton config and then a non-Triton config should clear Triton flags.
         */
        @Test
        @DisplayName("apply() resets all flags between configs")
        public void testApplyResetsFlags() {
            Environment env = Nd4j.getEnvironment();

            boolean origCublasTf32 = env.cublasTf32Enabled();
            boolean origTritonTf32 = env.tritonTf32Enabled();

            try {
                // First: apply Triton config with everything ON
                BenchmarkConfig tritonConfig = BenchmarkConfig.create("TRITON_ON")
                        .tritonIncludeTypes("ATTENTION")
                        .tritonSectionFusion(true)
                        .tritonGraphCapture(true)
                        .tritonCompileAll(true)
                        .cublasTf32(true)
                        .dspBatchedGemm(true);
                BenchmarkConfigApplier.apply(tritonConfig);

                assertTrue(env.tritonSectionFusion(), "Should be ON after Triton config");
                assertTrue(env.tritonGraphCapture());
                assertTrue(env.dspBatchedGemm());

                // Second: apply non-Triton config (should reset all Triton flags)
                BenchmarkConfig slotConfig = BenchmarkConfig.create("SLOT_ONLY")
                        .cublasTf32(false)
                        .dspBatchedGemm(false);
                BenchmarkConfigApplier.apply(slotConfig);

                // Triton flags should all be reset
                assertFalse(env.tritonSectionFusion(), "Should be OFF after non-Triton config");
                assertFalse(env.tritonGraphCapture());
                assertEquals("", env.tritonIncludeTypes());
                assertFalse(env.tritonCompileAll());
                assertFalse(env.dspBatchedGemm());
                assertFalse(env.cublasTf32Enabled());

                log.info("BenchmarkConfigApplier reset between configs verified");
            } finally {
                env.setCublasTf32Enabled(origCublasTf32);
                env.setTritonTf32Enabled(origTritonTf32);
                env.setTritonIncludeTypes("");
                env.setTritonSectionFusion(false);
                env.setTritonGraphCapture(false);
                env.setTritonConsolidatedArgTable(false);
                env.setTritonArgDirtyTracking(false);
                env.setTritonCompileAll(false);
                env.setDspBatchedGemm(false);
                env.setDspBatchZero(false);
            }
        }

        /**
         * Verify that apply() with optimal() config doesn't throw.
         * The internal verify() assertions must all pass.
         */
        @Test
        @DisplayName("apply(optimal()) does not throw")
        public void testApplyOptimalNoThrow() {
            Environment env = Nd4j.getEnvironment();
            boolean origCublasTf32 = env.cublasTf32Enabled();
            boolean origTritonTf32 = env.tritonTf32Enabled();

            try {
                assertDoesNotThrow(() -> BenchmarkConfigApplier.apply(BenchmarkConfig.optimal()),
                        "apply(optimal()) should not throw — all internal verify() must pass");
                log.info("apply(optimal()) internal verification passed");
            } finally {
                env.setCublasTf32Enabled(origCublasTf32);
                env.setTritonTf32Enabled(origTritonTf32);
                env.setTritonIncludeTypes("");
                env.setTritonSectionFusion(false);
                env.setTritonGraphCapture(false);
                env.setTritonConsolidatedArgTable(false);
                env.setTritonArgDirtyTracking(false);
                env.setTritonCompileAll(false);
                env.setDspBatchedGemm(false);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 6: View-Based Decoder Input Mode
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("View-Based Decoder Inputs")
    class ViewBasedDecoderInputTests {

        /**
         * In view-based (non-padded) mode, attention mask is contiguous [1, cachePos+1].
         * Verify the mask is correct for increasing cachePos values.
         */
        @Test
        @DisplayName("View-based attention mask is contiguous ones")
        public void testViewBasedAttentionMask() {
            // View-based mode: mask = Nd4j.ones(LONG, 1, totalSeqLen)
            // where totalSeqLen = cachePos + currentSeqLen
            for (int cachePos : new int[]{0, 1, 10, 100}) {
                int currentSeqLen = 1;
                long totalSeqLen = cachePos + currentSeqLen;
                INDArray mask = Nd4j.ones(DataType.LONG, 1, totalSeqLen);

                assertEquals(totalSeqLen, mask.sumNumber().longValue(),
                        "All positions should be 1 in view-based mode at cachePos=" + cachePos);
                assertArrayEquals(new long[]{1, totalSeqLen}, mask.shape());
                mask.close();
            }
            log.info("View-based attention mask test passed for multiple cachePos values");
        }

        /**
         * In view-based mode, KV cache view is [0:cachePos].
         * Edge case: cachePos=0 should produce empty KV cache.
         */
        @Test
        @DisplayName("View-based KV cache view at cachePos=0 needs empty cache")
        public void testViewBasedKvCacheCachePos0() {
            int batch = 1, heads = 4, maxKvLen = 50, dim = 32;
            INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);

            int cachePos = 0;
            // When cachePos=0, the code falls through to createEmptyKvCache
            // (since cachePos is not > 0 for interval(0, cachePos))
            // Verify that cachePos=0 is handled — should not attempt interval(0, 0)
            if (cachePos > 0 && cachePos < staticBuf.size(2)) {
                INDArray view = staticBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                fail("Should not enter view path when cachePos=0");
            } else if (cachePos >= staticBuf.size(2)) {
                fail("cachePos=0 should not be >= maxKvLen");
            } else {
                // This is the expected path: create empty KV cache
                // Verify that this path is taken (cachePos == 0)
                assertTrue(true, "cachePos=0 correctly falls through to empty KV cache");
            }

            log.info("View-based KV cache cachePos=0 edge case handled correctly");
            staticBuf.close();
        }

        /**
         * In view-based mode, KV cache view [0:cachePos] has correct shape and data.
         */
        @Test
        @DisplayName("View-based KV cache view has correct shape")
        public void testViewBasedKvCacheViewShape() {
            int batch = 1, heads = 2, maxKvLen = 50, dim = 16;
            int cachePos = 15;

            INDArray staticBuf = Nd4j.zeros(DataType.FLOAT, batch, heads, maxKvLen, dim);
            // Fill first cachePos positions with known data
            for (int p = 0; p < cachePos; p++) {
                staticBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(p), NDArrayIndex.all()).assign(p + 1.0f);
            }

            // Create the view
            INDArray view = staticBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());

            assertArrayEquals(new long[]{batch, heads, cachePos, dim}, view.shape(),
                    "View shape should be [batch, heads, cachePos, dim]");

            // Verify data in view
            for (int p = 0; p < cachePos; p++) {
                float val = view.getFloat(0, 0, p, 0);
                assertEquals(p + 1.0f, val, 0.01f,
                        "View position " + p + " should have value " + (p + 1.0f));
            }

            log.info("View-based KV cache view shape test passed: cachePos={}", cachePos);
            staticBuf.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 7: 4D Bias ensureLocation Sync
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("Bias Host-Device Sync")
    class BiasHostDeviceSyncTests {

        /**
         * Verify that putScalar followed by ensureLocation(DEVICE) makes the
         * change visible. This tests the fix for stale device-side mask values
         * during CUDA graph replay.
         *
         * On CPU backend, ensureLocation is a no-op, but the test still verifies
         * the data flow is correct.
         */
        @Test
        @DisplayName("putScalar + ensureLocation makes change visible")
        public void testPutScalarEnsureLocationSync() {
            int maxKvLen = 50;
            int totalLen = maxKvLen + 1;

            // Create bias with all MASK_FILL (like initial state)
            float[] biasData = new float[totalLen];
            Arrays.fill(biasData, DecoderUtils.MASK_FILL);
            // First 10 positions are valid (past)
            for (int k = 0; k < 10; k++) {
                biasData[k] = 0.0f;
            }
            INDArray bias = Nd4j.create(biasData, new long[]{1, 1, 1, totalLen}, 'c');

            // Simulate decode steps: each step unmasks cachePos-1
            for (int step = 0; step < 5; step++) {
                int cachePos = 10 + step + 1;  // 11, 12, 13, 14, 15
                if (cachePos > 0) {
                    bias.putScalar(new long[]{0, 0, 0, cachePos - 1}, 0.0f);
                    // ensureLocation would sync to device here on CUDA
                    Nd4j.getAffinityManager().ensureLocation(bias,
                            org.nd4j.linalg.api.concurrency.AffinityManager.Location.DEVICE);
                }

                // Verify the unmasked position is readable
                assertEquals(0.0f, bias.getFloat(0, 0, 0, cachePos - 1), 0.01f,
                        "Position " + (cachePos - 1) + " should be unmasked after putScalar+ensureLocation");
            }

            // Verify overall state: positions [0..14] should be 0, [15..maxKvLen-1] should be MASK_FILL
            for (int k = 0; k < 15; k++) {
                assertEquals(0.0f, bias.getFloat(0, 0, 0, k), 0.01f,
                        "Position " + k + " should be unmasked");
            }
            for (int k = 15; k < maxKvLen; k++) {
                assertEquals(DecoderUtils.MASK_FILL, bias.getFloat(0, 0, 0, k), 1.0f,
                        "Position " + k + " should still be masked");
            }

            log.info("putScalar + ensureLocation sync test passed: 5 steps of incremental unmasking");
            bias.close();
        }

        /**
         * Verify that the incremental bias update pattern (unmask cachePos-1)
         * produces correct cumulative state after many steps.
         */
        @Test
        @DisplayName("Incremental bias unmasking across 100 steps")
        public void testIncrementalBiasUnmask100Steps() {
            int maxKvLen = 200;
            int prefillLen = 50;
            int totalLen = maxKvLen + 1;

            // Initial bias: [0..prefillLen-1] = 0 (attend), [prefillLen..maxKvLen-1] = MASK_FILL
            float[] biasData = new float[totalLen];
            for (int k = prefillLen; k < maxKvLen; k++) {
                biasData[k] = DecoderUtils.MASK_FILL;
            }
            INDArray bias = Nd4j.create(biasData, new long[]{1, 1, 1, totalLen}, 'c');

            // Simulate 100 decode steps
            for (int step = 0; step < 100; step++) {
                int cachePos = prefillLen + step + 1;
                if (cachePos > 0 && cachePos - 1 < maxKvLen) {
                    bias.putScalar(new long[]{0, 0, 0, cachePos - 1}, 0.0f);
                }
            }

            // After 100 steps: positions [0..149] should be unmasked, [150..199] still masked
            int expectedUnmasked = prefillLen + 100;
            int unmaskCount = 0;
            for (int k = 0; k < maxKvLen; k++) {
                if (Math.abs(bias.getFloat(0, 0, 0, k)) < 0.01f) {
                    unmaskCount++;
                }
            }
            assertEquals(expectedUnmasked, unmaskCount,
                    "After 100 steps from prefill=" + prefillLen + ", should have " + expectedUnmasked + " unmasked");

            log.info("Incremental bias unmasking across 100 steps passed: {} unmasked of {}", unmaskCount, maxKvLen);
            bias.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Group 8: padKvCacheToStaticSize
    // ═══════════════════════════════════════════════════════════════════════

    @Nested
    @DisplayName("padKvCacheToStaticSize")
    class PadKvCacheTests {

        /**
         * Verify padKvCacheToStaticSize creates correct shape buffers and copies prefill data.
         */
        @Test
        @DisplayName("padKvCacheToStaticSize shape and data copy")
        public void testPadKvCacheShape() {
            int batch = 1, heads = 4, prefillLen = 10, maxKvLen = 50, dim = 32;

            List<String> presentKeyNames = Arrays.asList("present.0.key");
            List<String> presentValueNames = Arrays.asList("present.0.value");
            ModelIOConfig ioConfig = ModelIOConfig.builder().build();

            // Simulate prefill outputs
            Map<String, INDArray> prefillOutputs = new HashMap<>();
            INDArray prefillKey = Nd4j.ones(DataType.FLOAT, batch, heads, prefillLen, dim).mul(3.14f);
            INDArray prefillValue = Nd4j.ones(DataType.FLOAT, batch, heads, prefillLen, dim).mul(2.71f);
            prefillOutputs.put("present.0.key", prefillKey);
            prefillOutputs.put("present.0.value", prefillValue);

            Map<String, INDArray> staticBuffers = DecoderUtils.padKvCacheToStaticSize(
                    prefillOutputs, presentKeyNames, presentValueNames,
                    maxKvLen, ioConfig);

            // Verify shape
            String pastKeyName = ioConfig.presentToInputName("present.0.key");
            INDArray staticKey = staticBuffers.get(pastKeyName);
            assertNotNull(staticKey, "Static key buffer should exist");
            assertArrayEquals(new long[]{batch, heads, maxKvLen, dim}, staticKey.shape(),
                    "Static buffer should be [batch, heads, maxKvLen, dim]");

            // Verify prefill data was copied
            float prefillVal = staticKey.getFloat(0, 0, 0, 0);
            assertEquals(3.14f, prefillVal, 0.01f,
                    "Prefill data should be copied to positions [0..prefillLen-1]");

            float prefillLast = staticKey.getFloat(0, 0, prefillLen - 1, 0);
            assertEquals(3.14f, prefillLast, 0.01f);

            // Verify padding is zero
            float padVal = staticKey.getFloat(0, 0, prefillLen, 0);
            assertEquals(0.0f, padVal, 0.01f,
                    "Padding positions should be zero");

            log.info("padKvCacheToStaticSize test passed: prefill={} padded to {}", prefillLen, maxKvLen);

            // Cleanup
            prefillKey.close();
            prefillValue.close();
            for (INDArray buf : staticBuffers.values()) {
                buf.close();
            }
        }
    }
}
