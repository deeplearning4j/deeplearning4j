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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.generation.DecoderUtils;
import org.eclipse.deeplearning4j.llm.generation.KvCacheStrategy;
import org.eclipse.deeplearning4j.llm.generation.ModelIOConfig;
import org.eclipse.deeplearning4j.llm.generation.TurboQuantKvCacheManager;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@DisplayName("TurboQuantKvCacheManagerTest")
public class TurboQuantKvCacheManagerTest {

    private static final int BATCH = 1;
    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 64;
    private static final int PREFILL_LEN = 8;
    private static final int MAX_NEW_TOKENS = 16;

    private ModelIOConfig createTestIOConfig() {
        return ModelIOConfig.builder()
                .kvCachePrefix("past_key_values.")
                .kvPresentToInputReplace(new String[]{"present", "past_key_values"})
                .build();
    }

    private Map<String, INDArray> createPrefillOutputs(int numLayers) {
        Map<String, INDArray> outputs = new HashMap<>();
        for (int i = 0; i < numLayers; i++) {
            outputs.put("present." + i + ".key",
                    Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
            outputs.put("present." + i + ".value",
                    Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, PREFILL_LEN, HEAD_DIM));
        }
        return outputs;
    }

    private DecoderUtils.KVCacheNames createKvNames(int numLayers) {
        java.util.List<String> keys = new java.util.ArrayList<>();
        java.util.List<String> values = new java.util.ArrayList<>();
        for (int i = 0; i < numLayers; i++) {
            keys.add("past_key_values." + i + ".key");
            values.add("past_key_values." + i + ".value");
        }
        return new DecoderUtils.KVCacheNames(keys, values);
    }

    @Test
    @DisplayName("Strategy returns TURBOQUANT")
    void testStrategy() {
        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            assertEquals(KvCacheStrategy.TURBOQUANT, mgr.getStrategy());
        }
    }

    @Test
    @DisplayName("Does not support CUDA graph replay")
    void testNoCudaGraphReplay() {
        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            assertFalse(mgr.supportsCudaGraphReplay());
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 4})
    @DisplayName("InitializeFromPrefill sets correct cache state")
    void testInitializeFromPrefill(int bits) {
        int numLayers = 2;
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
        DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(bits, createTestIOConfig())) {
            mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

            assertTrue(mgr.isInitialized(), "Should be initialized after prefill");
            assertEquals(PREFILL_LEN, mgr.getCachePosition(), "Cache position should equal prefill length");
            assertEquals(PREFILL_LEN + MAX_NEW_TOKENS, mgr.getMaxKvLen(), "maxKvLen = prefill + maxNew");

            // Static buffers should be returned
            Map<String, INDArray> staticBuffers = mgr.getStaticKvBuffers();
            assertNotNull(staticBuffers);
            assertFalse(staticBuffers.isEmpty());
        }

        // Cleanup prefill outputs
        prefillOutputs.values().forEach(INDArray::close);
    }

    @Test
    @DisplayName("Value compress/dequantize round-trip preserves similarity > 0.95")
    void testValueRoundTripCosineSimilarity() {
        int numLayers = 1;
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
        DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

            // Get the dequantized values from the static buffers
            Map<String, INDArray> staticBuffers = mgr.getStaticKvBuffers();
            String valueName = "past_key_values.0.value";
            INDArray dequantized = staticBuffers.get(valueName);
            assertNotNull(dequantized, "Should have dequantized values for " + valueName);

            // Compare with original values
            INDArray original = prefillOutputs.get("present.0.value");

            // Check cosine similarity for each head and token position
            double totalSim = 0;
            int count = 0;
            for (int h = 0; h < NUM_HEADS; h++) {
                for (int t = 0; t < PREFILL_LEN; t++) {
                    INDArray origVec = original.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(t), NDArrayIndex.all()).dup();
                    INDArray deqVec = dequantized.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(t), NDArrayIndex.all()).castTo(DataType.FLOAT).dup();

                    double sim = cosineSimilarity(origVec, deqVec);
                    totalSim += sim;
                    count++;
                    origVec.close();
                    deqVec.close();
                }
            }

            double avgSim = totalSim / count;
            log.info("Average value cosine similarity (3-bit, d={}): {}", HEAD_DIM, avgSim);
            assertTrue(avgSim > 0.95,
                    String.format("Average value cosine similarity %.4f should be > 0.95", avgSim));
        }

        prefillOutputs.values().forEach(INDArray::close);
    }

    @Test
    @DisplayName("Key MSE reconstruction preserves similarity > 0.90")
    void testKeyMseReconstructionSimilarity() {
        int numLayers = 1;
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
        DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

            Map<String, INDArray> staticBuffers = mgr.getStaticKvBuffers();
            String keyName = "past_key_values.0.key";
            INDArray kMse = staticBuffers.get(keyName);
            assertNotNull(kMse, "Should have k_mse reconstruction for " + keyName);

            INDArray original = prefillOutputs.get("present.0.key");

            double totalSim = 0;
            int count = 0;
            for (int h = 0; h < NUM_HEADS; h++) {
                for (int t = 0; t < PREFILL_LEN; t++) {
                    INDArray origVec = original.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(t), NDArrayIndex.all()).dup();
                    INDArray mseVec = kMse.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                            NDArrayIndex.point(t), NDArrayIndex.all()).castTo(DataType.FLOAT).dup();

                    double sim = cosineSimilarity(origVec, mseVec);
                    totalSim += sim;
                    count++;
                    origVec.close();
                    mseVec.close();
                }
            }

            double avgSim = totalSim / count;
            log.info("Average key MSE cosine similarity (3-bit, d={}): {}", HEAD_DIM, avgSim);
            // Key MSE-only is Stage 1, somewhat lower fidelity than full two-stage
            assertTrue(avgSim > 0.90,
                    String.format("Average key MSE cosine similarity %.4f should be > 0.90", avgSim));
        }

        prefillOutputs.values().forEach(INDArray::close);
    }

    @Test
    @DisplayName("Scatter new entries advances cache position")
    void testScatterNewEntries() {
        int numLayers = 1;
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
        DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);
            assertEquals(PREFILL_LEN, mgr.getCachePosition());

            // Simulate a decode step: present KV has prefillLen + 1 entries
            long newTotalLen = PREFILL_LEN + 1;
            Map<String, INDArray> decodeOutputs = new HashMap<>();
            decodeOutputs.put("present.0.key",
                    Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, newTotalLen, HEAD_DIM));
            decodeOutputs.put("present.0.value",
                    Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, newTotalLen, HEAD_DIM));

            mgr.scatterNewEntries(decodeOutputs, kvNames);
            assertEquals(PREFILL_LEN + 1, mgr.getCachePosition(),
                    "Cache position should advance by 1 after scatter");

            // Scatter again
            long newTotalLen2 = PREFILL_LEN + 2;
            Map<String, INDArray> decodeOutputs2 = new HashMap<>();
            decodeOutputs2.put("present.0.key",
                    Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, newTotalLen2, HEAD_DIM));
            decodeOutputs2.put("present.0.value",
                    Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, newTotalLen2, HEAD_DIM));

            mgr.scatterNewEntries(decodeOutputs2, kvNames);
            assertEquals(PREFILL_LEN + 2, mgr.getCachePosition(),
                    "Cache position should advance by 1 again");

            decodeOutputs.values().forEach(INDArray::close);
            decodeOutputs2.values().forEach(INDArray::close);
        }

        prefillOutputs.values().forEach(INDArray::close);
    }

    @Test
    @DisplayName("Set/get cache position")
    void testSetCachePosition() {
        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            mgr.setCachePosition(42);
            assertEquals(42, mgr.getCachePosition());
        }
    }

    @Test
    @DisplayName("Close releases resources without error")
    void testCloseReleases() {
        int numLayers = 1;
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
        DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

        TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig());
        mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);
        assertTrue(mgr.isInitialized());

        mgr.close();
        assertFalse(mgr.isInitialized(), "Should not be initialized after close");

        // Double close should be safe
        assertDoesNotThrow(mgr::close, "Double close should not throw");

        prefillOutputs.values().forEach(INDArray::close);
    }

    @Test
    @DisplayName("Multi-layer initialization with 4 layers")
    void testMultiLayerInit() {
        int numLayers = 4;
        Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
        DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

        try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(3, createTestIOConfig())) {
            mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

            Map<String, INDArray> staticBuffers = mgr.getStaticKvBuffers();
            // Should have entries for all layers (keys + values)
            assertEquals(numLayers * 2, staticBuffers.size(),
                    "Should have static buffers for all key and value layers");
        }

        prefillOutputs.values().forEach(INDArray::close);
    }

    @Test
    @DisplayName("Higher bit width gives better reconstruction fidelity")
    void testHigherBitsGiveBetterFidelity() {
        int numLayers = 1;
        double[] avgSims = new double[3];
        int[] bitWidths = {2, 3, 4};

        for (int b = 0; b < bitWidths.length; b++) {
            Map<String, INDArray> prefillOutputs = createPrefillOutputs(numLayers);
            DecoderUtils.KVCacheNames kvNames = createKvNames(numLayers);

            try (TurboQuantKvCacheManager mgr = new TurboQuantKvCacheManager(bitWidths[b], createTestIOConfig())) {
                mgr.initializeFromPrefill(prefillOutputs, kvNames, MAX_NEW_TOKENS, PREFILL_LEN);

                Map<String, INDArray> staticBuffers = mgr.getStaticKvBuffers();
                INDArray dequantized = staticBuffers.get("past_key_values.0.value");
                INDArray original = prefillOutputs.get("present.0.value");

                double totalSim = 0;
                int count = 0;
                for (int h = 0; h < NUM_HEADS; h++) {
                    for (int t = 0; t < PREFILL_LEN; t++) {
                        INDArray origVec = original.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                                NDArrayIndex.point(t), NDArrayIndex.all()).dup();
                        INDArray deqVec = dequantized.get(NDArrayIndex.point(0), NDArrayIndex.point(h),
                                NDArrayIndex.point(t), NDArrayIndex.all()).castTo(DataType.FLOAT).dup();
                        totalSim += cosineSimilarity(origVec, deqVec);
                        count++;
                        origVec.close();
                        deqVec.close();
                    }
                }
                avgSims[b] = totalSim / count;
            }

            prefillOutputs.values().forEach(INDArray::close);
        }

        log.info("Value cosine sim: 2-bit={}, 3-bit={}, 4-bit={}", avgSims[0], avgSims[1], avgSims[2]);
        assertTrue(avgSims[1] > avgSims[0], "3-bit should have better fidelity than 2-bit");
        assertTrue(avgSims[2] > avgSims[1], "4-bit should have better fidelity than 3-bit");
    }

    private static double cosineSimilarity(INDArray a, INDArray b) {
        double dot = Nd4j.getBlasWrapper().dot(a, b);
        double normA = a.norm2Number().doubleValue();
        double normB = b.norm2Number().doubleValue();
        if (normA < 1e-10 || normB < 1e-10) return 0;
        return dot / (normA * normB);
    }
}
