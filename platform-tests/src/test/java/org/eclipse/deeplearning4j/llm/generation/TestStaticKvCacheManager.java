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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for StaticKvCacheManager — verifies initialization, scatter, and
 * multi-step correctness of the static KV cache management.
 */
public class TestStaticKvCacheManager {

    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 16;
    private static final int NUM_LAYERS = 2;

    private DecoderUtils.KVCacheNames createKvNames() {
        List<String> keyNames = new ArrayList<>();
        List<String> valueNames = new ArrayList<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            keyNames.add("present." + i + ".key");
            valueNames.add("present." + i + ".value");
        }
        return new DecoderUtils.KVCacheNames(keyNames, valueNames);
    }

    private Map<String, INDArray> createPrefillOutputs(int prefillLen) {
        Map<String, INDArray> outputs = new HashMap<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            outputs.put("present." + i + ".key",
                    Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, prefillLen, HEAD_DIM));
            outputs.put("present." + i + ".value",
                    Nd4j.randn(DataType.FLOAT, 1, NUM_HEADS, prefillLen, HEAD_DIM));
        }
        return outputs;
    }

    @Test
    public void testInitializeFromPrefill() {
        StaticKvCacheManager manager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        int prefillLen = 10;
        int maxNewTokens = 50;

        Map<String, INDArray> prefillOutputs = createPrefillOutputs(prefillLen);

        manager.initializeFromPrefill(prefillOutputs, kvNames, maxNewTokens, prefillLen);

        assertTrue(manager.isInitialized());
        assertEquals(prefillLen + maxNewTokens, manager.getMaxKvLen());
        assertEquals(prefillLen, manager.getCachePosition());
        assertTrue(manager.supportsCudaGraphReplay());
        assertEquals(KvCacheStrategy.STATIC, manager.getStrategy());

        // Verify buffers exist for all KV inputs
        Map<String, INDArray> buffers = manager.getStaticKvBuffers();
        assertNotNull(buffers);
        assertEquals(NUM_LAYERS * 2, buffers.size());

        // Verify buffer shapes
        for (INDArray buf : buffers.values()) {
            assertArrayEquals(new long[]{1, NUM_HEADS, prefillLen + maxNewTokens, HEAD_DIM},
                    buf.shape());
        }

        // Verify prefill data was copied into buffers
        INDArray firstKey = buffers.get("past_key_values.0.key");
        assertNotNull(firstKey);
        INDArray prefillSlice = firstKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, prefillLen), NDArrayIndex.all());
        // Prefill data should be non-zero
        assertTrue(prefillSlice.amaxNumber().doubleValue() > 0.01);

        // Padding should be zeros
        INDArray padSlice = firstKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(prefillLen, prefillLen + maxNewTokens), NDArrayIndex.all());
        assertEquals(0.0, padSlice.amaxNumber().doubleValue(), 1e-6,
                "Padding region should be zeros");

        manager.close();
    }

    @Test
    public void testScatterNewEntries() {
        StaticKvCacheManager manager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        int prefillLen = 5;
        int maxNewTokens = 10;

        Map<String, INDArray> prefillOutputs = createPrefillOutputs(prefillLen);
        manager.initializeFromPrefill(prefillOutputs, kvNames, maxNewTokens, prefillLen);

        long initialPos = manager.getCachePosition();
        assertEquals(prefillLen, initialPos);

        // Create decode outputs (present KV from the MHA op)
        // Present shape is [1, numHeads, cachePos+1, headDim] (concat of past + new)
        long presentLen = initialPos + 1;
        Map<String, INDArray> decodeOutputs = new HashMap<>();
        for (int i = 0; i < NUM_LAYERS; i++) {
            INDArray presentKey = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, presentLen, HEAD_DIM);
            INDArray presentValue = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, presentLen, HEAD_DIM);
            // Set last position to known value
            presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(presentLen - 1), NDArrayIndex.all()).assign(42.0);
            presentValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(presentLen - 1), NDArrayIndex.all()).assign(-42.0);
            decodeOutputs.put("present." + i + ".key", presentKey);
            decodeOutputs.put("present." + i + ".value", presentValue);
        }

        manager.scatterNewEntries(decodeOutputs, kvNames);

        // Position should advance
        assertEquals(initialPos + 1, manager.getCachePosition());

        // Verify the value was written at the correct position
        Map<String, INDArray> buffers = manager.getStaticKvBuffers();
        INDArray firstKey = buffers.get("past_key_values.0.key");
        INDArray scatteredPos = firstKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(initialPos), NDArrayIndex.all());
        assertEquals(42.0, scatteredPos.getDouble(0), 0.01,
                "Scattered value should be 42.0 at position " + initialPos);

        manager.close();
    }

    @Test
    public void testMultiStepScatter() {
        StaticKvCacheManager manager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();
        int prefillLen = 3;
        int maxNewTokens = 20;

        Map<String, INDArray> prefillOutputs = createPrefillOutputs(prefillLen);
        manager.initializeFromPrefill(prefillOutputs, kvNames, maxNewTokens, prefillLen);

        // Simulate 10 decode steps
        for (int step = 0; step < 10; step++) {
            long pos = manager.getCachePosition();
            long presentLen = pos + 1;

            Map<String, INDArray> decodeOutputs = new HashMap<>();
            for (int i = 0; i < NUM_LAYERS; i++) {
                INDArray presentKey = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, presentLen, HEAD_DIM);
                INDArray presentValue = Nd4j.zeros(DataType.FLOAT, 1, NUM_HEADS, presentLen, HEAD_DIM);
                // Fill last position with step-specific value
                float stepVal = (step + 1) * 10.0f;
                presentKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(presentLen - 1), NDArrayIndex.all()).assign(stepVal);
                presentValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.point(presentLen - 1), NDArrayIndex.all()).assign(-stepVal);
                decodeOutputs.put("present." + i + ".key", presentKey);
                decodeOutputs.put("present." + i + ".value", presentValue);
            }

            manager.scatterNewEntries(decodeOutputs, kvNames);
        }

        assertEquals(prefillLen + 10, manager.getCachePosition());

        // Verify each step wrote correctly
        Map<String, INDArray> buffers = manager.getStaticKvBuffers();
        INDArray firstKey = buffers.get("past_key_values.0.key");

        for (int step = 0; step < 10; step++) {
            long pos = prefillLen + step;
            double val = firstKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all()).getDouble(0);
            float expected = (step + 1) * 10.0f;
            assertEquals(expected, val, 0.01,
                    "Step " + step + " value at position " + pos);
        }

        manager.close();
    }

    @Test
    public void testSetCachePosition() {
        StaticKvCacheManager manager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();

        Map<String, INDArray> prefillOutputs = createPrefillOutputs(5);
        manager.initializeFromPrefill(prefillOutputs, kvNames, 10, 5);

        assertEquals(5, manager.getCachePosition());

        // Set to arbitrary position (e.g., for speculative decode rollback)
        manager.setCachePosition(3);
        assertEquals(3, manager.getCachePosition());

        manager.close();
    }

    @Test
    public void testClose() {
        StaticKvCacheManager manager = new StaticKvCacheManager();
        DecoderUtils.KVCacheNames kvNames = createKvNames();

        Map<String, INDArray> prefillOutputs = createPrefillOutputs(5);
        manager.initializeFromPrefill(prefillOutputs, kvNames, 10, 5);

        assertTrue(manager.isInitialized());
        assertNotNull(manager.getStaticKvBuffers());

        manager.close();

        assertFalse(manager.isInitialized());
        assertNull(manager.getStaticKvBuffers());
    }
}
