/*
 *  ******************************************************************************
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingDecodeStateTest {
    @Test
    void closeUnpinsAndReleasesEveryOwnedDecodeInputExactlyOnce() {
        INDArray kv = Nd4j.zeros(1, 1, 8, 2);
        kv.setCloseable(false);
        Map<String, INDArray> kvBuffers = new LinkedHashMap<>();
        kvBuffers.put("past_key_values.0.key", kv);
        INDArray embeddings = Nd4j.zeros(1, 1, 4);
        INDArray mask = Nd4j.ones(1, 9);

        EmbeddingDecodeState state = new EmbeddingDecodeState(kvBuffers, embeddings, mask);
        assertTrue(state.kvBuffer("past_key_values.0.key") == kv);
        assertTrue(state.decodeInput(0) == embeddings);
        assertTrue(state.decodeInput(1) == mask);
        assertTrue(state.matches(kvBuffers, embeddings, mask));
        assertNull(state.mismatch(kvBuffers, embeddings, mask));
        assertFalse(state.matches(kvBuffers, mask, embeddings));
        assertTrue(state.mismatch(kvBuffers, mask, embeddings).startsWith("decode input[0]"));

        state.close();
        state.close();

        assertTrue(kv.wasClosed(), "pinned static KV must be unpinned and closed by its owner");
        assertTrue(embeddings.wasClosed());
        assertTrue(mask.wasClosed());
    }

    @Test
    void fullPoolTrimCompletesStreamOrderedRetainedStateFrees() {
        INDArray input = Nd4j.zeros(1, 1, 32);
        EmbeddingDecodeState state = new EmbeddingDecodeState(Map.of(), input);

        state.close();
        SameDiffMemoryUtils.trimAllDevicePools();

        assertTrue(input.wasClosed());
    }
}
