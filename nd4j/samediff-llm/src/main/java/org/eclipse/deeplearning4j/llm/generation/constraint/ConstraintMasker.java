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

package org.eclipse.deeplearning4j.llm.generation.constraint;

import java.util.function.IntFunction;

/**
 * The main integration point for constrained decoding.
 *
 * <p>{@code ConstraintMasker} wraps a {@link TextConstraint} and a
 * {@link ConstraintVocabCache}, tracks the accumulated text emitted so far, and
 * provides the {@link #maskLogits} method that the sampling loop calls after every
 * forward pass.</p>
 *
 * <h2>Masking strategy</h2>
 * <ol>
 *   <li>Find the top-{@link #evalTopK} token indices by raw logit value.</li>
 *   <li>Check each against the constraint using the vocab cache.</li>
 *   <li>If at least one top-K token is allowed: mask <em>all other</em> tokens to
 *       {@link Float#NEGATIVE_INFINITY}.</li>
 *   <li>If <em>none</em> of the top-K tokens are allowed: widen to the full vocabulary
 *       and mask only the disallowed tokens. This prevents generation from getting
 *       permanently stuck on a low-probability constraint-satisfying token.</li>
 *   <li>EOS ({@code eosTokenId}) is allowed only when the constraint is in an
 *       {@link TextConstraint#isAccepting(String) accepting} state.</li>
 * </ol>
 *
 * <h2>Thread safety</h2>
 * <p>Not thread-safe. Each concurrent generation stream should own its own
 * {@code ConstraintMasker} instance.</p>
 *
 * <h2>Typical usage</h2>
 * <pre>{@code
 * ConstraintConfig cfg = ConstraintConfig.jsonObject();
 * ConstraintMasker masker = new ConstraintMasker(cfg.buildConstraint(), cfg.getEvalTopK());
 *
 * while (!masker.isComplete()) {
 *     float[] logits = model.nextLogits();                       // [vocabSize]
 *     float[] masked = masker.maskLogits(logits, eosId, idToPiece);
 *     int tokenId = sampler.sample(masked);
 *     masker.tokenEmitted(tokenId, idToPiece);
 *     if (tokenId == eosId) break;
 * }
 * String output = masker.getEmittedText();
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see ConstraintConfig
 * @see TextConstraint
 * @see ConstraintVocabCache
 */
public class ConstraintMasker {

    private final TextConstraint constraint;
    private final ConstraintVocabCache cache;
    /** Number of top-logit candidates evaluated before falling back to full vocab. */
    private final int evalTopK;
    /** Accumulated text emitted by the model so far. */
    private String emittedText;

    /**
     * Constructs a masker with a shared {@link ConstraintVocabCache}.
     *
     * @param constraint the constraint automaton to enforce
     * @param evalTopK   how many top-logit candidates to evaluate first (must be &gt; 0)
     */
    public ConstraintMasker(TextConstraint constraint, int evalTopK) {
        this(constraint, evalTopK, new ConstraintVocabCache());
    }

    /**
     * Constructs a masker with a caller-supplied vocab cache.
     * Useful when sharing one cache across multiple maskers over the same vocabulary.
     *
     * @param constraint the constraint automaton to enforce
     * @param evalTopK   how many top-logit candidates to evaluate first (must be &gt; 0)
     * @param cache      the vocab cache to use
     */
    public ConstraintMasker(TextConstraint constraint, int evalTopK, ConstraintVocabCache cache) {
        if (constraint == null) throw new IllegalArgumentException("constraint must not be null");
        if (evalTopK <= 0) throw new IllegalArgumentException("evalTopK must be > 0; got: " + evalTopK);
        if (cache == null) throw new IllegalArgumentException("cache must not be null");
        this.constraint = constraint;
        this.evalTopK = evalTopK;
        this.cache = cache;
        this.emittedText = "";
    }

    // -------------------------------------------------------------------------
    // Core masking logic
    // -------------------------------------------------------------------------

    /**
     * Applies the constraint mask to the raw logits array.
     *
     * <p>The input {@code logits} array is <em>not</em> mutated; a new array is
     * returned. If no masking is necessary (all top-K tokens are allowed and EOS
     * handling requires no change) the returned array may share storage with the
     * input — callers should not assume independence.</p>
     *
     * <p>EOS is allowed only when the constraint is in an accepting state.
     * When in an accepting state EOS is always allowed regardless of constraint
     * evaluation (the generation loop may choose to stop).</p>
     *
     * @param logits      1-D float array of shape [vocabSize], raw model output
     * @param eosTokenId  the end-of-sequence token ID; pass {@code -1} to disable
     *                    EOS gating
     * @param idToPiece   maps a token ID to its decoded string piece; may return
     *                    {@code null} for special / out-of-vocab tokens
     * @return a new {@code float[vocabSize]} with disallowed tokens set to
     *         {@link Float#NEGATIVE_INFINITY}
     */
    public float[] maskLogits(
            float[] logits,
            int eosTokenId,
            IntFunction<String> idToPiece) {

        int vocabSize = logits.length;
        boolean accepting = constraint.isAccepting(emittedText);

        // Get the allowed-token mask from the cache for the current emitted text.
        boolean[] allowed = cache.getAllowedTokens(constraint, emittedText, vocabSize, idToPiece);

        // Find top-evalTopK indices by raw logit value.
        int k = Math.min(evalTopK, vocabSize);
        int[] topKIndices = topKIndices(logits, k);

        // Check how many top-K candidates are allowed (ignoring EOS for now).
        boolean anyTopKAllowed = false;
        for (int idx : topKIndices) {
            if (idx == eosTokenId) continue; // handle EOS separately
            if (allowed[idx]) {
                anyTopKAllowed = true;
                break;
            }
        }

        float[] masked = new float[vocabSize];

        if (anyTopKAllowed) {
            // Mask strategy A: keep only the allowed top-K tokens.
            // Initialise everything to -inf, then restore allowed top-K logits.
            for (int i = 0; i < vocabSize; i++) {
                masked[i] = Float.NEGATIVE_INFINITY;
            }
            for (int idx : topKIndices) {
                if (idx == eosTokenId) continue;
                if (allowed[idx]) {
                    masked[idx] = logits[idx];
                }
            }
        } else {
            // Mask strategy B: widen to full vocab — mask only disallowed tokens.
            for (int i = 0; i < vocabSize; i++) {
                masked[i] = allowed[i] ? logits[i] : Float.NEGATIVE_INFINITY;
            }
        }

        // EOS gating: only permit EOS when the constraint is in an accepting state.
        if (eosTokenId >= 0 && eosTokenId < vocabSize) {
            if (accepting) {
                // Restore EOS logit so the sampler can stop.
                masked[eosTokenId] = logits[eosTokenId];
            } else {
                // Suppress EOS — we are not done yet.
                masked[eosTokenId] = Float.NEGATIVE_INFINITY;
            }
        }

        return masked;
    }

    // -------------------------------------------------------------------------
    // State updates
    // -------------------------------------------------------------------------

    /**
     * Notifies the masker that a token was selected and emitted by the sampler.
     * Updates the internal accumulated-text state.
     *
     * <p>Must be called once per token, in order, before the next {@link #maskLogits}
     * call.</p>
     *
     * @param tokenId   the emitted token ID
     * @param idToPiece maps token ID to its decoded string piece
     */
    public void tokenEmitted(int tokenId, IntFunction<String> idToPiece) {
        String piece = idToPiece.apply(tokenId);
        if (piece != null && !piece.isEmpty()) {
            emittedText += piece;
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /**
     * Returns the accumulated text emitted so far.
     *
     * @return the text generated under this constraint
     */
    public String getEmittedText() {
        return emittedText;
    }

    /**
     * Returns {@code true} if the constraint is in an accepting (complete) state for
     * the current emitted text — i.e., a structurally valid output has been produced.
     *
     * @return {@code true} if the constraint is satisfied
     */
    public boolean isComplete() {
        return constraint.isAccepting(emittedText);
    }

    /**
     * Returns the underlying constraint.
     *
     * @return the active {@link TextConstraint}
     */
    public TextConstraint getConstraint() {
        return constraint;
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Returns the indices of the {@code k} largest values in {@code values} using a
     * simple selection sort over the top-k positions.  O(vocabSize * k) but k is
     * typically small (default 256) so this is acceptable.
     *
     * @param values the full logit array
     * @param k      how many top indices to return
     * @return array of length {@code k} containing the top-k indices (order may vary)
     */
    static int[] topKIndices(float[] values, int k) {
        int n = values.length;
        // Use a min-heap simulation via a simple partial sort.
        // For small k (e.g., 256) vs large n (e.g., 32000) this is O(n log k) equivalent.
        int[] indices = new int[k];
        float[] topValues = new float[k];
        // Initialise with -inf sentinels.
        for (int i = 0; i < k; i++) {
            topValues[i] = Float.NEGATIVE_INFINITY;
            indices[i] = -1;
        }
        // Find the position of the minimum in the top-k window.
        int minPos = 0;
        float minVal = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < n; i++) {
            float v = values[i];
            if (v > minVal) {
                topValues[minPos] = v;
                indices[minPos] = i;
                // Find new minimum.
                minPos = 0;
                minVal = topValues[0];
                for (int j = 1; j < k; j++) {
                    if (topValues[j] < minVal) {
                        minVal = topValues[j];
                        minPos = j;
                    }
                }
            }
        }
        return indices;
    }
}
