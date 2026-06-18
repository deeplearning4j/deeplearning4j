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

import org.eclipse.deeplearning4j.llm.generation.batch.BatchCompactor;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class BatchCompactorTest extends BaseNd4jTestWithBackends {

    // ==================== shouldCompact Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShouldCompact_NoneFinished(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(4);
        boolean[] finished = {false, false, false, false};
        assertFalse(compactor.shouldCompact(finished));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShouldCompact_AllFinished(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(4);
        boolean[] finished = {true, true, true, true};
        assertFalse(compactor.shouldCompact(finished));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShouldCompact_BelowThreshold(Nd4jBackend backend) {
        // 1/4 = 25%, threshold is 25% — should trigger (>=)
        BatchCompactor compactor = new BatchCompactor(4);
        boolean[] finished = {true, false, false, false};
        assertTrue(compactor.shouldCompact(finished));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShouldCompact_HighThreshold(Nd4jBackend backend) {
        // 1/4 = 25% < 50% threshold
        BatchCompactor compactor = new BatchCompactor(4, 0.50);
        boolean[] finished = {true, false, false, false};
        assertFalse(compactor.shouldCompact(finished));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testShouldCompact_AboveThreshold(Nd4jBackend backend) {
        // 2/4 = 50% >= 50% threshold
        BatchCompactor compactor = new BatchCompactor(4, 0.50);
        boolean[] finished = {true, true, false, false};
        assertTrue(compactor.shouldCompact(finished));
    }

    // ==================== compactBatch Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompactBatch_2D(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(4);
        INDArray tensor = Nd4j.createFromArray(new float[][]{
                {1, 2, 3},   // seq 0
                {4, 5, 6},   // seq 1 - finished
                {7, 8, 9},   // seq 2
                {10, 11, 12} // seq 3 - finished
        });

        boolean[] finished = {false, true, false, true};
        INDArray result = compactor.compactBatch(tensor, finished);

        assertArrayEquals(new long[]{2, 3}, result.shape());
        assertEquals(1.0f, result.getFloat(0, 0), 1e-6);
        assertEquals(7.0f, result.getFloat(1, 0), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompactBatch_3D(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(3);
        // Shape [3, 2, 2]: 3 sequences, seqLen=2, hidden=2
        INDArray tensor = Nd4j.createFromArray(new float[][][]{
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}},   // finished
                {{9, 10}, {11, 12}}
        });

        boolean[] finished = {false, true, false};
        INDArray result = compactor.compactBatch(tensor, finished);

        assertArrayEquals(new long[]{2, 2, 2}, result.shape());
        assertEquals(1.0f, result.getFloat(0, 0, 0), 1e-6);
        assertEquals(9.0f, result.getFloat(1, 0, 0), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompactBatch_NothingToCompact(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(3);
        INDArray tensor = Nd4j.ones(3, 4);
        boolean[] finished = {false, false, false};
        INDArray result = compactor.compactBatch(tensor, finished);
        assertSame(tensor, result); // No compaction needed, same reference
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompactBatch_NullTensor(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(3);
        boolean[] finished = {false, true, false};
        assertNull(compactor.compactBatch(null, finished));
    }

    // ==================== compactKvCache Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompactKvCache(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(3);

        // KV cache entry: [batch=3, numHeads=2, seqLen=4, headDim=2]
        INDArray kvEntry = Nd4j.linspace(1, 48, 48).reshape(3, 2, 4, 2);
        Map<String, INDArray> kvCache = new HashMap<>();
        kvCache.put("present.0.key", kvEntry);

        boolean[] finished = {false, true, false};
        Map<String, INDArray> compacted = compactor.compactKvCache(kvCache, finished);

        INDArray result = compacted.get("present.0.key");
        assertNotNull(result);
        assertArrayEquals(new long[]{2, 2, 4, 2}, result.shape());
    }

    // ==================== compact + Index Mapping Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCompactUpdatesMapping(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(4);

        boolean[] finished = {false, true, false, true};
        int newSize = compactor.compact(finished);

        assertEquals(2, newSize);
        assertEquals(2, compactor.getCurrentBatchSize());
        assertTrue(compactor.isCompacted());

        // Compact idx 0 → original idx 0
        assertEquals(0, compactor.getOriginalIndex(0));
        // Compact idx 1 → original idx 2
        assertEquals(2, compactor.getOriginalIndex(1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testDoubleCompaction(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(5);

        // First compaction: remove indices 1 and 3
        boolean[] finished1 = {false, true, false, true, false};
        compactor.compact(finished1);
        assertEquals(3, compactor.getCurrentBatchSize());
        // Mapping: [0→0, 1→2, 2→4]

        // Second compaction: remove current index 1 (original 2)
        boolean[] finished2 = {false, true, false};
        compactor.compact(finished2);
        assertEquals(2, compactor.getCurrentBatchSize());
        // Mapping: [0→0, 1→4]
        assertEquals(0, compactor.getOriginalIndex(0));
        assertEquals(4, compactor.getOriginalIndex(1));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNewFinishedArray(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(4);
        boolean[] finished = {true, false, false, true};
        compactor.compact(finished);

        boolean[] newFinished = compactor.newFinishedArray();
        assertEquals(2, newFinished.length);
        assertFalse(newFinished[0]);
        assertFalse(newFinished[1]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testInitialState(Nd4jBackend backend) {
        BatchCompactor compactor = new BatchCompactor(5);
        assertEquals(5, compactor.getOriginalBatchSize());
        assertEquals(5, compactor.getCurrentBatchSize());
        assertFalse(compactor.isCompacted());

        // Identity mapping initially
        for (int i = 0; i < 5; i++) {
            assertEquals(i, compactor.getOriginalIndex(i));
        }
    }
}
