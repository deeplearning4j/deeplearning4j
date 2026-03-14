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
import org.nd4j.autodiff.samediff.SameDiff;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ChunkedPrefillEngine construction and validation.
 *
 * <p>Since ChunkedPrefillEngine.prefill() requires a real SameDiff model with specific
 * variable names, these tests focus on constructor validation, default values, and
 * chunk count arithmetic.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
class TestChunkedPrefill {

    // ==================== Constructor Validation ====================

    @Test
    void testConstruction_withDefaultChunkSize() {
        SameDiff model = SameDiff.create();
        ChunkedPrefillEngine engine = new ChunkedPrefillEngine(model);
        assertEquals(ChunkedPrefillEngine.DEFAULT_CHUNK_SIZE, engine.getChunkSize());
        assertSame(model, engine.getModel());
    }

    @Test
    void testConstruction_withCustomChunkSize() {
        SameDiff model = SameDiff.create();
        ChunkedPrefillEngine engine = new ChunkedPrefillEngine(model, 256);
        assertEquals(256, engine.getChunkSize());
        assertSame(model, engine.getModel());
    }

    @Test
    void testConstruction_chunkSizeOne() {
        SameDiff model = SameDiff.create();
        ChunkedPrefillEngine engine = new ChunkedPrefillEngine(model, 1);
        assertEquals(1, engine.getChunkSize());
    }

    @Test
    void testConstruction_nullModelThrows() {
        assertThrows(IllegalArgumentException.class, () -> new ChunkedPrefillEngine(null));
    }

    @Test
    void testConstruction_nullModelWithChunkSizeThrows() {
        assertThrows(IllegalArgumentException.class, () -> new ChunkedPrefillEngine(null, 512));
    }

    @Test
    void testConstruction_zeroChunkSizeThrows() {
        SameDiff model = SameDiff.create();
        assertThrows(IllegalArgumentException.class, () -> new ChunkedPrefillEngine(model, 0));
    }

    @Test
    void testConstruction_negativeChunkSizeThrows() {
        SameDiff model = SameDiff.create();
        assertThrows(IllegalArgumentException.class, () -> new ChunkedPrefillEngine(model, -1));
    }

    // ==================== Default Chunk Size Constant ====================

    @Test
    void testDefaultChunkSize() {
        assertEquals(512, ChunkedPrefillEngine.DEFAULT_CHUNK_SIZE,
                "Default chunk size should be 512 as documented");
    }

    // ==================== Chunk Count Arithmetic ====================

    @Test
    void testChunkCount_exactMultiple() {
        // 1024 tokens / 512 chunk size = exactly 2 chunks
        int totalLen = 1024;
        int chunkSize = 512;
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        assertEquals(2, numChunks);
    }

    @Test
    void testChunkCount_withRemainder() {
        // 1025 tokens / 512 chunk size = 3 chunks (last chunk has 1 token)
        int totalLen = 1025;
        int chunkSize = 512;
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        assertEquals(3, numChunks);
    }

    @Test
    void testChunkCount_singleToken() {
        // 1 token / 512 chunk size = 1 chunk
        int totalLen = 1;
        int chunkSize = 512;
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        assertEquals(1, numChunks);
    }

    @Test
    void testChunkCount_chunkSizeOne() {
        // 10 tokens / 1 chunk size = 10 chunks
        int totalLen = 10;
        int chunkSize = 1;
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        assertEquals(10, numChunks);
    }

    @Test
    void testChunkCount_equalToChunkSize() {
        // 512 tokens / 512 chunk size = 1 chunk
        int totalLen = 512;
        int chunkSize = 512;
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        assertEquals(1, numChunks);
    }

    @Test
    void testChunkCount_lastChunkSize() {
        int totalLen = 1030;
        int chunkSize = 512;
        int numChunks = (int) Math.ceil((double) totalLen / chunkSize);
        assertEquals(3, numChunks);

        long lastChunkTokens = totalLen - (long)(numChunks - 1) * chunkSize;
        assertEquals(6, lastChunkTokens, "Last chunk should contain the remainder tokens");
    }

    // ==================== toString ====================

    @Test
    void testToString() {
        SameDiff model = SameDiff.create();
        ChunkedPrefillEngine engine = new ChunkedPrefillEngine(model, 128);
        String str = engine.toString();
        assertTrue(str.contains("ChunkedPrefillEngine"));
        assertTrue(str.contains("128"));
    }
}
