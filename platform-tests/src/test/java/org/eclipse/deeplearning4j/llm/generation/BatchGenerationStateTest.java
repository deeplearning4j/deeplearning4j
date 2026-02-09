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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@NativeTag
@Tag(TagNames.SAMEDIFF)
public class BatchGenerationStateTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testRecordTokensMarksEOS(Nd4jBackend backend) {
        int eosTokenId = 2;
        BatchGenerationState state = new BatchGenerationState(2, eosTokenId);

        assertFalse(state.allFinished());
        assertEquals(2, state.activeCount());

        // Step 1: normal tokens
        state.recordTokens(new int[]{10, 20}, 1_000_000);
        assertFalse(state.isFinished(0));
        assertFalse(state.isFinished(1));
        assertEquals(1, state.getTokenCount(0));
        assertEquals(1, state.getTokenCount(1));

        // Step 2: sequence 0 hits EOS
        state.recordTokens(new int[]{eosTokenId, 30}, 2_000_000);
        assertTrue(state.isFinished(0));
        assertFalse(state.isFinished(1));
        assertEquals(1, state.activeCount());

        // Step 3: sequence 1 hits EOS
        state.recordTokens(new int[]{99, eosTokenId}, 3_000_000);
        assertTrue(state.isFinished(1));
        assertTrue(state.allFinished());
        assertEquals(0, state.activeCount());

        // Finished sequence 0 should not record more tokens
        assertEquals(2, state.getTokenCount(0)); // Only 2 tokens (10, EOS)
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultipleStopTokens(Nd4jBackend backend) {
        int eosTokenId = 2;
        int endOfUtterance = 49155;
        BatchGenerationState state = new BatchGenerationState(2, eosTokenId, endOfUtterance);

        // Sequence 0 hits the custom stop token
        state.recordTokens(new int[]{endOfUtterance, 10}, 1_000_000);
        assertTrue(state.isFinished(0));
        assertEquals(GenerationResult.FinishReason.EOS, state.getFinishReasons()[0]);

        // Sequence 1 hits the normal EOS
        state.recordTokens(new int[]{99, eosTokenId}, 2_000_000);
        assertTrue(state.isFinished(1));
        assertTrue(state.allFinished());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAllFinished(Nd4jBackend backend) {
        BatchGenerationState state = new BatchGenerationState(3, 0);

        assertFalse(state.allFinished());

        state.markMaxTokens(0);
        assertFalse(state.allFinished());

        state.markStopSequence(1);
        assertFalse(state.allFinished());

        state.recordTokens(new int[]{99, 99, 0}, 1_000_000); // seq 2 hits EOS
        assertTrue(state.allFinished());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testBuildResults(Nd4jBackend backend) {
        int eosTokenId = 2;
        BatchGenerationState state = new BatchGenerationState(2, eosTokenId);

        state.recordTokens(new int[]{10, 20}, 100_000_000); // 100ms
        state.recordTokens(new int[]{eosTokenId, 30}, 200_000_000);
        state.recordTokens(new int[]{99, eosTokenId}, 300_000_000);

        String[] texts = {"Hello", "World"};
        int[] promptCounts = {5, 5};
        long totalNanos = 300_000_000L;

        GenerationResult[] results = state.buildResults(texts, promptCounts, totalNanos);

        assertEquals(2, results.length);

        // Sequence 0: 2 tokens (10, EOS)
        assertEquals("Hello", results[0].getText());
        assertEquals(2, results[0].getGeneratedTokenCount());
        assertEquals(5, results[0].getPromptTokenCount());
        assertEquals(7, results[0].getTotalTokenCount());
        assertEquals(GenerationResult.FinishReason.EOS, results[0].getFinishReason());
        assertTrue(results[0].getFirstTokenLatencyMs() > 0);

        // Sequence 1: 3 tokens (20, 30, EOS)
        assertEquals("World", results[1].getText());
        assertEquals(3, results[1].getGeneratedTokenCount());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testFinishReasons(Nd4jBackend backend) {
        BatchGenerationState state = new BatchGenerationState(3, 999);

        state.markMaxTokens(0);
        assertEquals(GenerationResult.FinishReason.MAX_TOKENS, state.getFinishReasons()[0]);

        state.markStopSequence(1);
        assertEquals(GenerationResult.FinishReason.STOP_SEQUENCE, state.getFinishReasons()[1]);

        state.recordTokens(new int[]{99, 99, 999}, 1_000_000);
        assertEquals(GenerationResult.FinishReason.EOS, state.getFinishReasons()[2]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGetTokenArrayForSequence(Nd4jBackend backend) {
        BatchGenerationState state = new BatchGenerationState(1, 99);

        state.recordTokens(new int[]{10}, 1_000_000);
        state.recordTokens(new int[]{20}, 2_000_000);
        state.recordTokens(new int[]{30}, 3_000_000);

        int[] tokens = state.getTokenArrayForSequence(0);
        assertArrayEquals(new int[]{10, 20, 30}, tokens);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMaxTokenCount(Nd4jBackend backend) {
        BatchGenerationState state = new BatchGenerationState(2, 0);

        state.recordTokens(new int[]{10, 20}, 1_000_000);
        state.recordTokens(new int[]{0, 30}, 2_000_000); // seq 0 finishes
        state.recordTokens(new int[]{99, 40}, 3_000_000); // seq 0 doesn't record

        assertEquals(2, state.getTokenCount(0)); // 10, EOS
        assertEquals(3, state.getTokenCount(1)); // 20, 30, 40
        assertEquals(3, state.getMaxTokenCount());
    }
}
