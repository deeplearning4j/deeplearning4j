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

package org.eclipse.deeplearning4j.nd4j.linalg.dataset.curation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.curation.TokenizedTextDataIterator;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link TokenizedTextDataIterator}.
 *
 * <p>Uses a simple identity tokenizer (one token per character, ASCII ordinal modulo
 * vocabSize) so that tests do not depend on any external tokenizer library.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@Slf4j
@DisplayName("TokenizedTextDataIterator Tests")
public class TestTokenizedTextDataIterator extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /** Vocabulary size for the test tokenizer. */
    private static final int VOCAB = 256;

    /**
     * Simple tokenizer: one token per character, value = ASCII ordinal mod VOCAB.
     * Deterministic and independent of any external library.
     */
    private static final TokenizedTextDataIterator.TokenizerFunction TOKENIZER =
            text -> {
                int[] ids = new int[text.length()];
                for (int i = 0; i < text.length(); i++) {
                    ids[i] = text.charAt(i) % VOCAB;
                }
                return ids;
            };

    // =========================================================================
    // 1. Pretraining iterator — basic batch assembly
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testPretrainingIterator")
    public void testPretrainingIterator(Nd4jBackend backend) {
        List<String> texts = Arrays.asList("abcdef", "ghijkl", "mnopqr");
        int maxSeq = 4;
        int batchSize = 2;

        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, maxSeq, batchSize);

        assertTrue(iter.hasNext(), "iterator should have at least one batch");

        MultiDataSet batch = iter.next();
        assertNotNull(batch, "batch must not be null");

        INDArray inputIds = batch.getFeatures(0);
        INDArray labels   = batch.getLabels(0);

        // Shape: [batchSize, maxSeqLength]
        assertNotNull(inputIds);
        assertNotNull(labels);
        assertEquals(batchSize, inputIds.size(0), "batch dimension mismatch");
        assertEquals(maxSeq,    inputIds.size(1), "sequence length mismatch");
        assertEquals(batchSize, labels.size(0));
        assertEquals(maxSeq,    labels.size(1));

        // Labels should be input shifted right (next-token prediction)
        // For first example "abcd" (4 chars, maxSeq=4):
        //   input:  [a, b, c, d]
        //   labels: [b, c, d, 0]
        long inputA = inputIds.getLong(0, 0);
        long labelA = labels.getLong(0, 0);
        // label[t] == input[t+1] for t < maxSeq-1
        assertEquals(inputIds.getLong(0, 1), labelA,
                "label[0] should equal input[1] (causal LM shift)");
    }

    // =========================================================================
    // 2. SFT iterator — loss mask present and correctly populated
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testSFTIterator")
    public void testSFTIterator(Nd4jBackend backend) {
        // Create FormattedExamples: first half of each text is masked (0), second half is (1)
        List<FormattedExample> examples = new ArrayList<>();
        String text1 = "aaaabbbb";  // 8 chars — first 4 prompt (0), last 4 response (1)
        int[] mask1 = {0, 0, 0, 0, 1, 1, 1, 1};
        examples.add(new FormattedExample(text1, mask1));

        String text2 = "ccccdddd";
        int[] mask2 = {0, 0, 0, 0, 1, 1, 1, 1};
        examples.add(new FormattedExample(text2, mask2));

        int maxSeq = 8;
        int batchSize = 2;

        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forSFT(
                TOKENIZER, examples, maxSeq, batchSize);

        assertTrue(iter.hasNext());
        MultiDataSet batch = iter.next();

        assertNotNull(batch.getFeatures(0), "input_ids should be present");
        assertNotNull(batch.getLabels(0),   "labels should be present");

        // In SFT mode the feature mask (loss_mask) should be non-null
        assertNotNull(batch.getFeaturesMaskArray(0),
                "SFT mode should produce a non-null loss_mask");

        INDArray lossMask = batch.getFeaturesMaskArray(0);
        assertEquals(2, lossMask.size(0), "batch size should be 2");
        assertEquals(maxSeq, lossMask.size(1), "loss mask length should match maxSeqLength");

        // Verify that the mask has some 0s (prompt) and some 1s (response)
        float minVal = lossMask.minNumber().floatValue();
        float maxVal = lossMask.maxNumber().floatValue();
        assertTrue(minVal <= 0.0f, "loss mask should contain zeros for prompt tokens");
        assertTrue(maxVal >= 1.0f, "loss mask should contain ones for response tokens");
    }

    // =========================================================================
    // 3. Batch dimensions correctness
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testBatchSizes")
    public void testBatchSizes(Nd4jBackend backend) {
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            texts.add("text" + i + "_some_content_here");
        }

        int maxSeq = 8;
        int batchSize = 3;

        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, maxSeq, batchSize);

        int batchCount = 0;
        while (iter.hasNext()) {
            MultiDataSet batch = iter.next();
            INDArray inputIds = batch.getFeatures(0);
            // All batches except possibly the last should have exactly batchSize rows
            assertTrue(inputIds.size(0) >= 1 && inputIds.size(0) <= batchSize,
                    "batch size out of range: " + inputIds.size(0));
            assertEquals(maxSeq, inputIds.size(1), "sequence length should always be maxSeqLength");
            batchCount++;
        }

        // 10 examples with batchSize=3 → ceil(10/3) = 4 batches
        assertEquals(4, batchCount, "expected 4 batches for 10 examples with batchSize=3");
    }

    // =========================================================================
    // 4. Padding — short sequences padded to maxSeqLength with token ID 0
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testPadding")
    public void testPadding(Nd4jBackend backend) {
        // Text with only 2 chars; maxSeqLength = 8 → 6 positions should be padded with 0
        List<String> texts = Arrays.asList("ab");
        int maxSeq = 8;

        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, maxSeq, 1);

        assertTrue(iter.hasNext());
        MultiDataSet batch = iter.next();
        INDArray inputIds = batch.getFeatures(0);

        assertEquals(1, inputIds.size(0));
        assertEquals(maxSeq, inputIds.size(1));

        // Positions 2..7 should be 0 (padding)
        for (int t = 2; t < maxSeq; t++) {
            assertEquals(0L, inputIds.getLong(0, t),
                    "padding position " + t + " should be token ID 0");
        }

        // Positions 0 and 1 should be non-zero (ASCII 'a'=97, 'b'=98)
        assertEquals((long) 'a' % VOCAB, inputIds.getLong(0, 0));
        assertEquals((long) 'b' % VOCAB, inputIds.getLong(0, 1));
    }

    // =========================================================================
    // 5. Reset — iterator restarts from beginning
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testReset")
    public void testReset(Nd4jBackend backend) {
        List<String> texts = Arrays.asList("hello", "world", "test");
        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, 8, 2);

        // Consume all batches
        int firstPass = 0;
        while (iter.hasNext()) {
            iter.next();
            firstPass++;
        }
        assertFalse(iter.hasNext(), "iterator should be exhausted after first pass");

        // Reset and consume again
        iter.reset();
        assertTrue(iter.hasNext(), "iterator should be available again after reset()");

        int secondPass = 0;
        while (iter.hasNext()) {
            iter.next();
            secondPass++;
        }

        assertEquals(firstPass, secondPass,
                "second pass should produce same number of batches as first pass");
    }

    // =========================================================================
    // 6. Empty input — produces no batches
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testEmptyInput")
    public void testEmptyInput(Nd4jBackend backend) {
        List<String> texts = new ArrayList<>();  // empty
        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, 128, 4);

        assertFalse(iter.hasNext(), "empty text list should produce zero batches");
    }

    // =========================================================================
    // 7. Custom tokenizer lambda works end-to-end
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testTokenizerFunction")
    public void testTokenizerFunction(Nd4jBackend backend) {
        // Tokenizer that produces a fixed sequence [1, 2, 3, 4] regardless of input
        TokenizedTextDataIterator.TokenizerFunction fixedTokenizer = text -> new int[]{1, 2, 3, 4};

        List<String> texts = Arrays.asList("anything", "at all");
        int maxSeq = 4;
        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                fixedTokenizer, texts, maxSeq, 2);

        assertTrue(iter.hasNext());
        MultiDataSet batch = iter.next();
        INDArray inputIds = batch.getFeatures(0);

        // Every token in every row should match [1, 2, 3, 4]
        for (int row = 0; row < inputIds.size(0); row++) {
            assertEquals(1L, inputIds.getLong(row, 0));
            assertEquals(2L, inputIds.getLong(row, 1));
            assertEquals(3L, inputIds.getLong(row, 2));
            assertEquals(4L, inputIds.getLong(row, 3));
        }

        // Labels should be shift of [1,2,3,4] → [2,3,4,0]
        INDArray labels = batch.getLabels(0);
        for (int row = 0; row < labels.size(0); row++) {
            assertEquals(2L, labels.getLong(row, 0), "label[0] should be 2");
            assertEquals(3L, labels.getLong(row, 1), "label[1] should be 3");
            assertEquals(4L, labels.getLong(row, 2), "label[2] should be 4");
            assertEquals(0L, labels.getLong(row, 3), "last label should be 0 (pad)");
        }
    }

    // =========================================================================
    // 8. charMaskToTokenMask utility — correct majority vote
    // =========================================================================

    @Test
    @DisplayName("testCharMaskToTokenMask")
    public void testCharMaskToTokenMask() {
        // 8 characters, 4 chars per token → 2 tokens
        // Token 0: chars 0-3 = [1, 1, 0, 0] → 2/4 ones → majority=NO (2*2 == 4, not >4) → 0
        // Token 1: chars 4-7 = [1, 1, 1, 0] → 3/4 ones → majority=YES (3*2=6 > 4) → 1
        int[] charMask = {1, 1, 0, 0,   1, 1, 1, 0};
        int[] tokenMask = TokenizedTextDataIterator.charMaskToTokenMask(charMask, 2, 4);

        assertEquals(2, tokenMask.length);
        assertEquals(0, tokenMask[0], "token 0: 2/4 ones → not majority → 0");
        assertEquals(1, tokenMask[1], "token 1: 3/4 ones → majority → 1");
    }

    @Test
    @DisplayName("testCharMaskToTokenMaskBeyondEnd")
    public void testCharMaskToTokenMaskBeyondEnd() {
        // seqLen > charMask.length / charsPerToken
        int[] charMask = {1, 1};  // only 2 chars
        int[] tokenMask = TokenizedTextDataIterator.charMaskToTokenMask(charMask, 4, 2);
        // Tokens 0: chars[0,1] = [1,1] → 2/2 → 1
        // Tokens 1,2,3: beyond end → 0
        assertEquals(4, tokenMask.length);
        assertEquals(1, tokenMask[0]);
        assertEquals(0, tokenMask[1]);
        assertEquals(0, tokenMask[2]);
        assertEquals(0, tokenMask[3]);
    }

    // =========================================================================
    // 9. Pretraining mode has no loss mask (featureMask null)
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testPretrainingHasNoLossMask")
    public void testPretrainingHasNoLossMask(Nd4jBackend backend) {
        List<String> texts = Arrays.asList("hello world");
        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, 8, 1);

        assertTrue(iter.hasNext());
        MultiDataSet batch = iter.next();

        // Pretraining: no loss mask
        assertNull(batch.getFeaturesMaskArray(0),
                "pretraining mode should NOT have a feature mask (loss_mask)");
    }

    // =========================================================================
    // 10. DataType of output arrays is INT64 for token IDs
    // =========================================================================

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("testOutputDataTypes")
    public void testOutputDataTypes(Nd4jBackend backend) {
        List<String> texts = Arrays.asList("data type check");
        TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
                TOKENIZER, texts, 8, 1);

        MultiDataSet batch = iter.next();

        assertEquals(DataType.INT64, batch.getFeatures(0).dataType(),
                "input_ids should be INT64 (long)");
        assertEquals(DataType.INT64, batch.getLabels(0).dataType(),
                "labels should be INT64 (long)");
    }
}
