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

import org.eclipse.deeplearning4j.llm.generation.speculative.NgramSpeculator;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NgramSpeculatorTest {

    // ==================== Constructor Tests ====================

    @Test
    void testDefaultConstructor() {
        NgramSpeculator spec = new NgramSpeculator();
        assertEquals(3, spec.getNgramSize());
        assertEquals(5, spec.getMaxSpeculativeTokens());
    }

    @Test
    void testCustomConstructor() {
        NgramSpeculator spec = new NgramSpeculator(2, 8);
        assertEquals(2, spec.getNgramSize());
        assertEquals(8, spec.getMaxSpeculativeTokens());
    }

    @Test
    void testInvalidNgramSize() {
        assertThrows(IllegalArgumentException.class, () -> new NgramSpeculator(0, 5));
    }

    @Test
    void testInvalidMaxSpeculative() {
        assertThrows(IllegalArgumentException.class, () -> new NgramSpeculator(3, 0));
    }

    // ==================== Speculate Tests ====================

    @Test
    void testSpeculate_TooShortSequence() {
        NgramSpeculator spec = new NgramSpeculator(3, 5);
        List<Integer> tokens = Arrays.asList(10, 20); // Only 2, need 3
        int[] result = spec.speculate(tokens);
        assertEquals(0, result.length);
    }

    @Test
    void testSpeculate_NoMatch() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);
        // All unique bigrams: (1,2), (2,3), (3,4), (4,5) - last bigram (4,5) not seen before
        List<Integer> tokens = Arrays.asList(1, 2, 3, 4, 5);
        int[] result = spec.speculate(tokens);
        assertEquals(0, result.length);
    }

    @Test
    void testSpeculate_SimpleRepetition() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);
        // Pattern: [10, 20, 30, 40, 10, 20]
        // Last bigram: (10, 20)
        // Earlier match at position 0: (10, 20) followed by [30, 40]
        List<Integer> tokens = Arrays.asList(10, 20, 30, 40, 10, 20);
        int[] result = spec.speculate(tokens);
        assertArrayEquals(new int[]{30, 40}, result);
    }

    @Test
    void testSpeculate_TrigramMatch() {
        NgramSpeculator spec = new NgramSpeculator(3, 5);
        // Pattern: [1, 2, 3, 4, 5, 6, 1, 2, 3]
        // Last trigram: (1, 2, 3) at position 6
        // Earlier match at position 0: followed by [4, 5, 6]
        List<Integer> tokens = Arrays.asList(1, 2, 3, 4, 5, 6, 1, 2, 3);
        int[] result = spec.speculate(tokens);
        assertArrayEquals(new int[]{4, 5, 6}, result);
    }

    @Test
    void testSpeculate_MaxSpeculativeLimits() {
        NgramSpeculator spec = new NgramSpeculator(2, 2); // max 2 speculative tokens
        // Pattern: [10, 20, 30, 40, 50, 10, 20]
        // Match at pos 0: followed by [30, 40, 50] but limited to 2
        List<Integer> tokens = Arrays.asList(10, 20, 30, 40, 50, 10, 20);
        int[] result = spec.speculate(tokens);
        assertEquals(2, result.length);
        assertArrayEquals(new int[]{30, 40}, result);
    }

    @Test
    void testSpeculate_MostRecentMatch() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);
        // Pattern: [1, 2, 100, 1, 2, 200, 1, 2]
        // Last bigram: (1, 2)
        // Most recent match at pos 3: (1, 2) followed by [200]
        // Earlier match at pos 0: (1, 2) followed by [100]
        // Should prefer most recent (pos 3)
        List<Integer> tokens = Arrays.asList(1, 2, 100, 1, 2, 200, 1, 2);
        int[] result = spec.speculate(tokens);
        assertArrayEquals(new int[]{200}, result);
    }

    @Test
    void testSpeculate_StructuredOutput() {
        // Simulates DocTag-like structured output where patterns repeat
        NgramSpeculator spec = new NgramSpeculator(3, 5);
        // <text> Hello </text> <text>
        // Token IDs: 50, 60, 70, 50, 60
        // Trigram (70, 50, 60) won't match — but bigram (50, 60) repeats
        NgramSpeculator spec2 = new NgramSpeculator(2, 5);
        List<Integer> tokens = Arrays.asList(50, 60, 70, 50, 60);
        int[] result = spec2.speculate(tokens);
        assertArrayEquals(new int[]{70}, result);
    }

    @Test
    void testSpeculate_SingleTokenNgram() {
        NgramSpeculator spec = new NgramSpeculator(1, 3);
        // With n=1, matches on any single token seen before
        // [5, 10, 15, 5] → last token 5 seen at pos 0, followed by [10, 15]
        List<Integer> tokens = Arrays.asList(5, 10, 15, 5);
        int[] result = spec.speculate(tokens);
        assertArrayEquals(new int[]{10, 15}, result);
    }

    // ==================== Verification Tests ====================

    @Test
    void testVerify_AllAccepted() {
        int[] specTokens = {3, 7, 11};
        float[][] logits = {
                {0, 0, 0, 10, 0},   // argmax=3 matches spec[0]
                {0, 0, 0, 0, 0, 0, 0, 10}, // argmax=7 matches spec[1]
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10}, // argmax=11 matches spec[2]
                {0, 5, 0} // correction position
        };
        int accepted = NgramSpeculator.verifySpeculation(specTokens, logits);
        assertEquals(3, accepted);
    }

    @Test
    void testVerify_NoneAccepted() {
        int[] specTokens = {3, 7};
        float[][] logits = {
                {0, 10, 0, 0, 0}, // argmax=1, spec[0]=3 → mismatch
                {0, 0, 0, 0, 0, 0, 0, 10},
                {0, 5, 0}
        };
        int accepted = NgramSpeculator.verifySpeculation(specTokens, logits);
        assertEquals(0, accepted);
    }

    @Test
    void testVerify_PartialAccepted() {
        int[] specTokens = {3, 7, 11};
        float[][] logits = {
                {0, 0, 0, 10, 0},   // argmax=3 matches spec[0]
                {0, 0, 0, 0, 0, 0, 0, 10, 0}, // argmax=7 matches spec[1]
                {10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // argmax=0, spec[2]=11 → mismatch
                {0, 5, 0}
        };
        int accepted = NgramSpeculator.verifySpeculation(specTokens, logits);
        assertEquals(2, accepted);
    }

    @Test
    void testGetCorrectionToken() {
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};
        assertEquals(1, NgramSpeculator.getCorrectionToken(logits));
    }

    @Test
    void testGetCorrectionToken_FirstElement() {
        float[] logits = {10.0f, 5.0f, 3.0f};
        assertEquals(0, NgramSpeculator.getCorrectionToken(logits));
    }

    // ==================== Batch Speculation Tests ====================

    @Test
    void testBatchSpeculate_AllCanSpeculate() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);

        List<List<Integer>> batchTokens = new ArrayList<>();
        // Sequence 0: [1, 2, 3, 1, 2] → speculates [3]
        batchTokens.add(Arrays.asList(1, 2, 3, 1, 2));
        // Sequence 1: [4, 5, 6, 4, 5] → speculates [6]
        batchTokens.add(Arrays.asList(4, 5, 6, 4, 5));

        boolean[] finished = {false, false};
        NgramSpeculator.SpeculationResult result = spec.speculateBatch(batchTokens, finished);

        assertTrue(result.hasSpeculation());
        assertEquals(1, result.getCommonLength());
        assertArrayEquals(new int[]{3}, result.getPerSequenceTokens()[0]);
        assertArrayEquals(new int[]{6}, result.getPerSequenceTokens()[1]);
    }

    @Test
    void testBatchSpeculate_OneCannotSpeculate() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);

        List<List<Integer>> batchTokens = new ArrayList<>();
        // Sequence 0: [1, 2, 3, 1, 2] → can speculate
        batchTokens.add(Arrays.asList(1, 2, 3, 1, 2));
        // Sequence 1: [4, 5, 6, 7, 8] → all unique bigrams, no speculation
        batchTokens.add(Arrays.asList(4, 5, 6, 7, 8));

        boolean[] finished = {false, false};
        NgramSpeculator.SpeculationResult result = spec.speculateBatch(batchTokens, finished);

        // Can't speculate uniformly
        assertFalse(result.hasSpeculation());
    }

    @Test
    void testBatchSpeculate_FinishedSkipped() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);

        List<List<Integer>> batchTokens = new ArrayList<>();
        // Sequence 0: can speculate
        batchTokens.add(Arrays.asList(1, 2, 3, 1, 2));
        // Sequence 1: finished, doesn't matter
        batchTokens.add(Arrays.asList(4, 5, 6, 7, 8));

        boolean[] finished = {false, true};
        NgramSpeculator.SpeculationResult result = spec.speculateBatch(batchTokens, finished);

        assertTrue(result.hasSpeculation());
        assertEquals(1, result.getCommonLength());
    }

    @Test
    void testBatchSpeculate_TruncatesToCommonLength() {
        NgramSpeculator spec = new NgramSpeculator(2, 10);

        List<List<Integer>> batchTokens = new ArrayList<>();
        // Sequence 0: [1, 2, 3, 4, 5, 1, 2] → speculates [3, 4, 5] (3 tokens)
        batchTokens.add(Arrays.asList(1, 2, 3, 4, 5, 1, 2));
        // Sequence 1: [6, 7, 8, 6, 7] → speculates [8] (1 token)
        batchTokens.add(Arrays.asList(6, 7, 8, 6, 7));

        boolean[] finished = {false, false};
        NgramSpeculator.SpeculationResult result = spec.speculateBatch(batchTokens, finished);

        assertTrue(result.hasSpeculation());
        assertEquals(1, result.getCommonLength()); // min(3, 1) = 1
        assertEquals(1, result.getPerSequenceTokens()[0].length);
        assertEquals(1, result.getPerSequenceTokens()[1].length);
    }
}
