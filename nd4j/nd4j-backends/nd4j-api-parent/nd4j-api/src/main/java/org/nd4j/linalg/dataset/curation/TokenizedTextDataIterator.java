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

package org.nd4j.linalg.dataset.curation;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Bridges raw text (or formatted instruction examples) to the
 * {@link MultiDataSetIterator} API expected by
 * {@link org.nd4j.autodiff.samediff.training.ContinuedPretrainingWorkflow#train} and
 * {@link org.nd4j.autodiff.samediff.training.SFTTrainingPipeline#train}.
 *
 * <h2>Pretraining mode</h2>
 * <p>Created via {@link #forPretraining}.  All tokens are trained on (no loss mask).
 * The label sequence is the input shifted right by one position (next-token prediction):
 * <pre>
 *   input:  [t0, t1, t2, ..., tN-1, &lt;pad&gt;]
 *   labels: [t1, t2, t3, ..., tN-1, 0    ]
 * </pre>
 * Sequences shorter than {@code maxSeqLength} are padded with token ID 0 on the right.
 *
 * <h2>SFT mode</h2>
 * <p>Created via {@link #forSFT}.  Accepts {@link FormattedExample} objects which carry
 * a character-level {@code lossMask} indicating which characters are assistant-generated
 * and should contribute to the loss.  The char-level mask is converted to a token-level
 * mask with a majority-vote heuristic (4 chars per token by default).  The resulting
 * {@link MultiDataSet} has a non-null {@code featureMasks[0]} ({@code loss_mask})
 * with 1 = train, 0 = ignore.
 *
 * <h2>Sequence packing</h2>
 * <p>When {@code packSequences=true} (pretraining mode only), multiple short sequences
 * are concatenated into a single {@code maxSeqLength} window before batching.  This
 * avoids padding waste for datasets with variable-length short documents.
 *
 * <h2>MultiDataSet layout</h2>
 * <ul>
 *   <li>{@code features[0]} — {@code input_ids}: LONG, shape {@code [batch, seqLen]}</li>
 *   <li>{@code labels[0]}   — {@code labels}: LONG, shape {@code [batch, seqLen]}</li>
 *   <li>{@code featureMasks[0]} — {@code loss_mask}: FLOAT, shape {@code [batch, seqLen]}
 *       (SFT mode only; null in pretraining mode)</li>
 * </ul>
 *
 * <p>Example — pretraining:
 * <pre>
 * TokenizedTextDataIterator iter = TokenizedTextDataIterator.forPretraining(
 *     text -> myTokenizer.encode(text), texts, 2048, 8);
 * workflow.train(iter, totalSteps);
 * </pre>
 *
 * <p>Example — SFT:
 * <pre>
 * List&lt;FormattedExample&gt; examples = formatter.format(conversations);
 * TokenizedTextDataIterator iter = TokenizedTextDataIterator.forSFT(
 *     text -> myTokenizer.encode(text), examples, 2048, 4);
 * pipeline.train(iter, totalSteps);
 * </pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see org.nd4j.autodiff.samediff.training.ContinuedPretrainingWorkflow
 * @see org.nd4j.autodiff.samediff.training.SFTTrainingPipeline
 * @see FormattedExample
 */
@Slf4j
public class TokenizedTextDataIterator implements MultiDataSetIterator {

    /**
     * Functional interface for pluggable tokenizers.
     *
     * <p>Implementations should return token IDs as a plain {@code int[]} so that the
     * iterator remains independent of any specific tokenizer library.
     *
     * <pre>
     * // Lambda using a HuggingFace tokenizer wrapper
     * TokenizerFunction fn = text -> hfTokenizer.encode(text).getIds();
     *
     * // Lambda using a custom BPE tokenizer
     * TokenizerFunction fn = myBpe::tokenize;
     * </pre>
     */
    @FunctionalInterface
    public interface TokenizerFunction {
        /**
         * Encode a text string into token IDs.
         *
         * @param text input text
         * @return array of integer token IDs
         */
        int[] encode(String text);
    }

    /** Estimated characters per token used when converting char-level masks. */
    private static final int CHARS_PER_TOKEN = 4;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    @Getter
    private final int maxSeqLength;

    @Getter
    private final int batchSize;

    /** Whether to pack multiple sequences into a single window (pretraining only). */
    @Getter
    private final boolean packSequences;

    /** Whether to mask prompt tokens so only response tokens contribute to loss (SFT). */
    @Getter
    private final boolean maskPrompts;

    // -------------------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------------------

    /** Tokenized token-ID sequences, one per input text or example. */
    private final List<int[]> tokenSequences;

    /**
     * Per-token loss masks (one int[] per sequence, values 0 or 1).
     * Null for pretraining mode; non-null for SFT mode.
     */
    private final List<int[]> tokenMasks;

    @Getter
    private MultiDataSetPreProcessor preProcessor;

    /** Cursor into the token sequences list (post-pack in packing mode). */
    private int cursor = 0;

    /** Packed windows of token IDs assembled from raw sequences. */
    private final List<int[]> packedWindows;

    /** Packed windows of token masks (parallel to packedWindows; null in pretraining). */
    private final List<int[]> packedMasks;

    // -------------------------------------------------------------------------
    // Private constructor
    // -------------------------------------------------------------------------

    private TokenizedTextDataIterator(TokenizerFunction tokenizer,
                                       List<String> texts,
                                       List<int[]> prebuiltMasks,
                                       int maxSeqLength,
                                       int batchSize,
                                       boolean packSequences,
                                       boolean maskPrompts) {
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.packSequences = packSequences;
        this.maskPrompts = maskPrompts;

        // Tokenize all texts
        this.tokenSequences = new ArrayList<>(texts.size());
        for (String text : texts) {
            int[] ids = tokenizer.encode(text);
            this.tokenSequences.add(ids);
        }

        this.tokenMasks = prebuiltMasks;

        // Build packed windows (or treat each sequence as one window in non-pack mode)
        if (packSequences && prebuiltMasks == null) {
            this.packedWindows = buildPackedWindows(tokenSequences, null);
            this.packedMasks = null;
        } else if (packSequences) {
            List<int[]>[] packed = buildPackedWindowsWithMasks(tokenSequences, prebuiltMasks);
            this.packedWindows = packed[0];
            this.packedMasks = packed[1];
        } else {
            // Non-packing: treat each sequence as a single fixed-length window
            this.packedWindows = buildTruncatedWindows(tokenSequences);
            this.packedMasks = (prebuiltMasks != null)
                    ? buildTruncatedMaskWindows(prebuiltMasks) : null;
        }

        log.debug("TokenizedTextDataIterator: {} sequences → {} windows, batchSize={}, " +
                        "pack={}, maskPrompts={}",
                texts.size(), packedWindows.size(), batchSize, packSequences, maskPrompts);
    }

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Create an iterator for causal-LM pretraining (all tokens trained, no loss mask).
     *
     * @param tokenizer    Tokenizer function that maps text to token IDs.
     * @param texts        Raw text documents.
     * @param maxSeqLength Maximum sequence length; longer sequences are truncated.
     * @param batchSize    Number of sequences per batch.
     * @return A ready-to-use iterator.
     */
    public static TokenizedTextDataIterator forPretraining(TokenizerFunction tokenizer,
                                                            List<String> texts,
                                                            int maxSeqLength,
                                                            int batchSize) {
        Preconditions.checkNotNull(tokenizer, "tokenizer must not be null");
        Preconditions.checkNotNull(texts, "texts must not be null");
        Preconditions.checkState(maxSeqLength > 0, "maxSeqLength must be positive");
        Preconditions.checkState(batchSize > 0, "batchSize must be positive");
        return new TokenizedTextDataIterator(
                tokenizer, texts, null, maxSeqLength, batchSize, false, false);
    }

    /**
     * Create an iterator for pretraining with optional sequence packing.
     *
     * @param tokenizer      Tokenizer function.
     * @param texts          Raw text documents.
     * @param maxSeqLength   Maximum sequence length.
     * @param batchSize      Batch size.
     * @param packSequences  If true, concatenate short sequences to fill windows.
     * @return A ready-to-use iterator.
     */
    public static TokenizedTextDataIterator forPretraining(TokenizerFunction tokenizer,
                                                            List<String> texts,
                                                            int maxSeqLength,
                                                            int batchSize,
                                                            boolean packSequences) {
        Preconditions.checkNotNull(tokenizer, "tokenizer must not be null");
        Preconditions.checkNotNull(texts, "texts must not be null");
        Preconditions.checkState(maxSeqLength > 0, "maxSeqLength must be positive");
        Preconditions.checkState(batchSize > 0, "batchSize must be positive");
        return new TokenizedTextDataIterator(
                tokenizer, texts, null, maxSeqLength, batchSize, packSequences, false);
    }

    /**
     * Create an iterator for SFT (Supervised Fine-Tuning) mode.
     *
     * <p>The character-level loss mask in each {@link FormattedExample} is converted
     * to a token-level mask using a 4-chars-per-token heuristic.  Only tokens where
     * the mask is 1 contribute to the loss.
     *
     * @param tokenizer    Tokenizer function.
     * @param examples     Pre-formatted instruction examples with character-level loss masks.
     * @param maxSeqLength Maximum sequence length.
     * @param batchSize    Batch size.
     * @return A ready-to-use iterator.
     */
    public static TokenizedTextDataIterator forSFT(TokenizerFunction tokenizer,
                                                    List<FormattedExample> examples,
                                                    int maxSeqLength,
                                                    int batchSize) {
        Preconditions.checkNotNull(tokenizer, "tokenizer must not be null");
        Preconditions.checkNotNull(examples, "examples must not be null");
        Preconditions.checkState(maxSeqLength > 0, "maxSeqLength must be positive");
        Preconditions.checkState(batchSize > 0, "batchSize must be positive");

        List<String> texts = new ArrayList<>(examples.size());
        List<int[]> charMasks = new ArrayList<>(examples.size());
        for (FormattedExample ex : examples) {
            texts.add(ex.getText());
            charMasks.add(ex.getLossMask());
        }

        // Build token-level masks lazily after tokenization lengths are known
        // We pass charMasks and convert them inside the constructor after tokenization
        TokenizedTextDataIterator iter = new TokenizedTextDataIterator(
                tokenizer, texts, null, maxSeqLength, batchSize, false, true);

        // Convert char masks to token masks aligned with the actual token sequences
        List<int[]> tokenMaskList = new ArrayList<>(iter.tokenSequences.size());
        for (int i = 0; i < iter.tokenSequences.size(); i++) {
            int seqLen = Math.min(iter.tokenSequences.get(i).length, maxSeqLength);
            int[] charMask = charMasks.get(i);
            tokenMaskList.add(charMaskToTokenMask(charMask, seqLen));
        }

        // Rebuild with the correct masks applied
        return new TokenizedTextDataIterator(
                tokenizer, texts, tokenMaskList, maxSeqLength, batchSize, false, true);
    }

    // -------------------------------------------------------------------------
    // MultiDataSetIterator
    // -------------------------------------------------------------------------

    @Override
    public boolean hasNext() {
        return cursor * batchSize < packedWindows.size();
    }

    @Override
    public org.nd4j.linalg.dataset.api.MultiDataSet next() {
        int start = cursor * batchSize;
        int end = Math.min(start + batchSize, packedWindows.size());
        cursor++;

        List<int[]> tokenBatch = packedWindows.subList(start, end);
        List<int[]> maskBatch = (packedMasks != null)
                ? packedMasks.subList(start, end) : null;

        org.nd4j.linalg.dataset.api.MultiDataSet ds = createBatch(tokenBatch, maskBatch);
        if (preProcessor != null) {
            preProcessor.preProcess(ds);
        }
        return ds;
    }

    @Override
    public org.nd4j.linalg.dataset.api.MultiDataSet next(int num) {
        return next();
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    // -------------------------------------------------------------------------
    // Batch assembly
    // -------------------------------------------------------------------------

    /**
     * Assemble a list of token-ID windows and optional mask windows into a
     * padded {@link MultiDataSet}.
     *
     * @param tokenBatch Parallel list of token-ID arrays (may differ in length).
     * @param maskBatch  Parallel list of token-mask arrays, or null for pretraining.
     * @return A padded MultiDataSet with input_ids, labels, and optional loss_mask.
     */
    private org.nd4j.linalg.dataset.api.MultiDataSet createBatch(List<int[]> tokenBatch,
                                                                   List<int[]> maskBatch) {
        int b = tokenBatch.size();

        long[] inputData = new long[b * maxSeqLength];
        long[] labelData = new long[b * maxSeqLength];
        float[] maskData = (maskBatch != null) ? new float[b * maxSeqLength] : null;

        for (int i = 0; i < b; i++) {
            int[] ids = tokenBatch.get(i);
            int len = Math.min(ids.length, maxSeqLength);
            int[] mask = (maskBatch != null) ? maskBatch.get(i) : null;

            for (int t = 0; t < len; t++) {
                inputData[i * maxSeqLength + t] = ids[t];
            }
            // Causal LM labels: shift left by one
            for (int t = 0; t < len - 1; t++) {
                labelData[i * maxSeqLength + t] = ids[t + 1];
            }
            // labelData[i * maxSeqLength + len - 1] stays 0 (padding label)

            if (mask != null) {
                int maskLen = Math.min(mask.length, maxSeqLength);
                for (int t = 0; t < maskLen; t++) {
                    maskData[i * maxSeqLength + t] = mask[t];
                }
            }
        }

        INDArray inputIds = Nd4j.createFromArray(inputData).castTo(DataType.INT64)
                .reshape(b, maxSeqLength);
        INDArray labels = Nd4j.createFromArray(labelData).castTo(DataType.INT64)
                .reshape(b, maxSeqLength);

        if (maskData != null) {
            INDArray lossMask = Nd4j.createFromArray(maskData).reshape(b, maxSeqLength);
            return new MultiDataSet(
                    new INDArray[]{inputIds},
                    new INDArray[]{labels},
                    new INDArray[]{lossMask},
                    null);
        }

        return new MultiDataSet(
                new INDArray[]{inputIds},
                new INDArray[]{labels});
    }

    // -------------------------------------------------------------------------
    // Window building helpers
    // -------------------------------------------------------------------------

    /**
     * Build packed windows by concatenating token sequences until a window of
     * {@link #maxSeqLength} is full, then starting a new window.
     */
    private List<int[]> buildPackedWindows(List<int[]> sequences, List<int[]> masks) {
        List<int[]> windows = new ArrayList<>();
        int[] current = new int[maxSeqLength];
        int pos = 0;

        for (int[] seq : sequences) {
            int idx = 0;
            while (idx < seq.length) {
                int space = maxSeqLength - pos;
                int toCopy = Math.min(space, seq.length - idx);
                System.arraycopy(seq, idx, current, pos, toCopy);
                pos += toCopy;
                idx += toCopy;

                if (pos == maxSeqLength) {
                    windows.add(current);
                    current = new int[maxSeqLength];
                    pos = 0;
                }
            }
        }
        // Emit partial window if non-empty
        if (pos > 0) {
            windows.add(current);
        }
        return windows;
    }

    /**
     * Build packed windows for both token IDs and masks in parallel.
     *
     * @return Two-element array: [packedTokenWindows, packedMaskWindows].
     */
    @SuppressWarnings("unchecked")
    private List<int[]>[] buildPackedWindowsWithMasks(List<int[]> sequences,
                                                        List<int[]> masks) {
        List<int[]> tokenWindows = new ArrayList<>();
        List<int[]> maskWindows = new ArrayList<>();

        int[] currentTokens = new int[maxSeqLength];
        int[] currentMasks = new int[maxSeqLength];
        int pos = 0;

        for (int s = 0; s < sequences.size(); s++) {
            int[] seq = sequences.get(s);
            int[] mask = masks.get(s);
            int idx = 0;
            while (idx < seq.length) {
                int space = maxSeqLength - pos;
                int toCopy = Math.min(space, seq.length - idx);
                System.arraycopy(seq, idx, currentTokens, pos, toCopy);
                if (mask != null && idx < mask.length) {
                    int maskCopy = Math.min(toCopy, mask.length - idx);
                    System.arraycopy(mask, idx, currentMasks, pos, maskCopy);
                }
                pos += toCopy;
                idx += toCopy;

                if (pos == maxSeqLength) {
                    tokenWindows.add(currentTokens);
                    maskWindows.add(currentMasks);
                    currentTokens = new int[maxSeqLength];
                    currentMasks = new int[maxSeqLength];
                    pos = 0;
                }
            }
        }
        if (pos > 0) {
            tokenWindows.add(currentTokens);
            maskWindows.add(currentMasks);
        }

        return new List[]{tokenWindows, maskWindows};
    }

    /**
     * Build one fixed-length window per sequence (truncate if needed; pad if shorter).
     * This is used in non-packing mode.
     */
    private List<int[]> buildTruncatedWindows(List<int[]> sequences) {
        List<int[]> windows = new ArrayList<>(sequences.size());
        for (int[] seq : sequences) {
            int[] window = new int[maxSeqLength];
            int len = Math.min(seq.length, maxSeqLength);
            System.arraycopy(seq, 0, window, 0, len);
            windows.add(window);
        }
        return windows;
    }

    /**
     * Build one fixed-length mask window per mask sequence (truncate / zero-pad).
     */
    private List<int[]> buildTruncatedMaskWindows(List<int[]> masks) {
        List<int[]> windows = new ArrayList<>(masks.size());
        for (int[] mask : masks) {
            int[] window = new int[maxSeqLength];
            int len = Math.min(mask.length, maxSeqLength);
            System.arraycopy(mask, 0, window, 0, len);
            windows.add(window);
        }
        return windows;
    }

    // -------------------------------------------------------------------------
    // Mask conversion
    // -------------------------------------------------------------------------

    /**
     * Convert a character-level loss mask to a token-level mask using a
     * majority-vote heuristic (default 4 chars per token).
     *
     * <p>A token is trainable (mask=1) if strictly more than half of its
     * corresponding characters are marked 1 in the character mask.
     *
     * @param charMask  Character-level mask (1 = train, 0 = ignore).
     * @param seqLen    Number of tokens in the tokenized sequence.
     * @return Token-level mask of length {@code seqLen}.
     */
    public static int[] charMaskToTokenMask(int[] charMask, int seqLen) {
        return charMaskToTokenMask(charMask, seqLen, CHARS_PER_TOKEN);
    }

    /**
     * Convert a character-level loss mask to a token-level mask.
     *
     * @param charMask      Character-level mask (1 = train, 0 = ignore).
     * @param seqLen        Number of tokens in the tokenized sequence.
     * @param charsPerToken Estimated characters per token.
     * @return Token-level mask of length {@code seqLen}.
     */
    public static int[] charMaskToTokenMask(int[] charMask, int seqLen, int charsPerToken) {
        int[] tokenMask = new int[seqLen];
        for (int t = 0; t < seqLen; t++) {
            int charStart = t * charsPerToken;
            if (charStart >= charMask.length) {
                tokenMask[t] = 0;
                continue;
            }
            int charEnd = Math.min(charStart + charsPerToken, charMask.length);
            int ones = 0;
            int total = charEnd - charStart;
            for (int c = charStart; c < charEnd; c++) {
                ones += charMask[c];
            }
            // Majority vote: train if more than half the chars are masked
            tokenMask[t] = (ones * 2 > total) ? 1 : 0;
        }
        return tokenMask;
    }
}
