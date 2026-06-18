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
import org.eclipse.deeplearning4j.llm.generation.speculative.SpeculativeDecodeLoop;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for SpeculativeDecodeLoop.
 *
 * <p>Tests the speculative decode loop statistics and result handling.
 * The full step() method requires SameDiff models, so we test the
 * components and stats tracking here. Integration testing with models
 * is covered in the VLM pipeline test.</p>
 */
class SpeculativeDecodeLoopTest {

    @Test
    void testDefaultConstructor() {
        SpeculativeDecodeLoop loop = new SpeculativeDecodeLoop();
        assertEquals(0, loop.getTotalAccepted());
        assertEquals(0, loop.getTotalAttempted());
        assertEquals(0, loop.getSpeculativeSteps());
        assertEquals(0, loop.getNormalSteps());
    }

    @Test
    void testCustomSpeculator() {
        NgramSpeculator spec = new NgramSpeculator(2, 8);
        SpeculativeDecodeLoop loop = new SpeculativeDecodeLoop(spec);
        assertNotNull(loop);
    }

    @Test
    void testStepReturnsNull_WhenNoSpeculation() {
        // With a short sequence that won't match any n-grams,
        // step() should return null (caller should do normal decode)
        NgramSpeculator spec = new NgramSpeculator(3, 5);
        SpeculativeDecodeLoop loop = new SpeculativeDecodeLoop(spec);

        // Only 2 tokens — too short for trigram speculation
        List<Integer> tokens = Arrays.asList(1, 2);
        SpeculativeDecodeLoop.SpeculativeStepResult result = loop.step(
                tokens, 2, null, null, null, null, null, null, null, null, 0, 1, 0, null);

        assertNull(result);
        assertEquals(1, loop.getNormalSteps());
        assertEquals(0, loop.getSpeculativeSteps());
    }

    @Test
    void testStats_ZeroState() {
        SpeculativeDecodeLoop loop = new SpeculativeDecodeLoop();
        String stats = loop.getStats();
        assertNotNull(stats);
        assertTrue(stats.contains("speculative=0"));
        assertTrue(stats.contains("normal=0"));
    }

    // ==================== SpeculativeStepResult Tests ====================

    @Test
    void testStepResult_Basic() {
        int[] tokens = {10, 20, 30};
        java.util.Map<String, org.nd4j.linalg.api.ndarray.INDArray> kvCache = new java.util.HashMap<>();
        SpeculativeDecodeLoop.SpeculativeStepResult result =
                new SpeculativeDecodeLoop.SpeculativeStepResult(tokens, kvCache, 3, false);

        assertArrayEquals(new int[]{10, 20, 30}, result.getAcceptedTokens());
        assertSame(kvCache, result.getUpdatedKvCache());
        assertEquals(3, result.getNewPositions());
        assertFalse(result.hitEos());
    }

    @Test
    void testStepResult_WithEos() {
        int[] tokens = {10, 2}; // 2 = EOS
        SpeculativeDecodeLoop.SpeculativeStepResult result =
                new SpeculativeDecodeLoop.SpeculativeStepResult(tokens, null, 2, true);

        assertTrue(result.hitEos());
        assertEquals(2, result.getAcceptedTokens().length);
    }

    // ==================== End-to-End Verification Logic ====================

    @Test
    void testVerificationAndCorrection_AllAccepted() {
        // Simulates what step() does internally: speculate, verify, correct
        NgramSpeculator spec = new NgramSpeculator(2, 5);
        List<Integer> tokens = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 1, 2));
        int[] specTokens = spec.speculate(tokens);

        // spec should propose [3, 4, 5]
        assertArrayEquals(new int[]{3, 4, 5}, specTokens);

        // Simulate model logits that agree with all speculation
        float[][] logits = new float[4][10]; // 3 positions + 1 correction
        logits[0][3] = 10.0f;  // position 0 predicts token 3 ✓
        logits[1][4] = 10.0f;  // position 1 predicts token 4 ✓
        logits[2][5] = 10.0f;  // position 2 predicts token 5 ✓
        logits[3][7] = 10.0f;  // correction: next token is 7

        int accepted = NgramSpeculator.verifySpeculation(specTokens, logits);
        assertEquals(3, accepted);

        int correction = NgramSpeculator.getCorrectionToken(logits[accepted]);
        assertEquals(7, correction);

        // Build accepted tokens: [3, 4, 5, 7]
        int[] acceptedTokens = new int[accepted + 1];
        System.arraycopy(specTokens, 0, acceptedTokens, 0, accepted);
        acceptedTokens[accepted] = correction;
        assertArrayEquals(new int[]{3, 4, 5, 7}, acceptedTokens);
    }

    @Test
    void testVerificationAndCorrection_PartialAccepted() {
        NgramSpeculator spec = new NgramSpeculator(2, 5);
        List<Integer> tokens = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5, 1, 2));
        int[] specTokens = spec.speculate(tokens);
        // specTokens = [3, 4, 5]

        // Model agrees with first 2 but not the third
        float[][] logits = new float[4][10];
        logits[0][3] = 10.0f;  // position 0 predicts token 3 ✓
        logits[1][4] = 10.0f;  // position 1 predicts token 4 ✓
        logits[2][9] = 10.0f;  // position 2 predicts token 9, not 5 ✗
        logits[3][0] = 10.0f;  // not reached

        int accepted = NgramSpeculator.verifySpeculation(specTokens, logits);
        assertEquals(2, accepted);

        // Correction at position 2 (the first mismatch)
        int correction = NgramSpeculator.getCorrectionToken(logits[accepted]);
        assertEquals(9, correction);

        // Build: [3, 4, 9]
        int[] acceptedTokens = new int[accepted + 1];
        System.arraycopy(specTokens, 0, acceptedTokens, 0, accepted);
        acceptedTokens[accepted] = correction;
        assertArrayEquals(new int[]{3, 4, 9}, acceptedTokens);
    }

    @Test
    void testVerificationAndCorrection_NoneAccepted() {
        int[] specTokens = {3, 4, 5};

        // Model disagrees from the start
        float[][] logits = new float[4][10];
        logits[0][8] = 10.0f;  // predicts 8, not 3

        int accepted = NgramSpeculator.verifySpeculation(specTokens, logits);
        assertEquals(0, accepted);

        int correction = NgramSpeculator.getCorrectionToken(logits[0]);
        assertEquals(8, correction);

        // Result is just [8] — the correction token
        int[] acceptedTokens = new int[]{correction};
        assertArrayEquals(new int[]{8}, acceptedTokens);
    }
}
