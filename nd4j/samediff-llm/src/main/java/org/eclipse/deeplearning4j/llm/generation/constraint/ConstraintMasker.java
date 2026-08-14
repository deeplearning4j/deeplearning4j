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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
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
        Set<Integer> stopTokenIds = eosTokenId >= 0
                ? Collections.singleton(eosTokenId)
                : Collections.emptySet();
        return maskLogits(logits, stopTokenIds, idToPiece);
    }

    /**
     * Applies the constraint mask while gating every configured terminal token.
     *
     * <p>Models commonly expose multiple stop sentinels (for example both
     * {@code <|tool_call_end|>} and {@code <|im_end|>}). Treating only the primary EOS
     * specially lets another stop sentinel leak through while a quoted argument is still
     * open, because it is otherwise just another text piece to the automaton. All terminal
     * tokens are therefore suppressed until the constraint reaches an accepting state.</p>
     *
     * @param logits       raw vocabulary logits
     * @param stopTokenIds all tokens that terminate generation
     * @param idToPiece    token-ID to decoded-piece mapping
     * @return masked logits
     */
    public float[] maskLogits(
            float[] logits,
            Set<Integer> stopTokenIds,
            IntFunction<String> idToPiece) {
        return maskLogits(logits, stopTokenIds, Collections.emptySet(), idToPiece);
    }

    /**
     * Applies terminal gating and blocks tokenizer-declared special tokens unless
     * the active constraint explicitly owns that token at the current state.
     */
    public float[] maskLogits(
            float[] logits,
            Set<Integer> stopTokenIds,
            Set<Integer> specialTokenIds,
            IntFunction<String> idToPiece) {

        int vocabSize = logits.length;
        boolean accepting = constraint.isAccepting(emittedText);
        Set<Integer> terminals = stopTokenIds == null
                ? Collections.emptySet()
                : stopTokenIds;
        Set<Integer> specials = specialTokenIds == null
                ? Collections.emptySet()
                : specialTokenIds;
        List<String> specialPieces = specialTokenPieces(specials, idToPiece);

        // Get the allowed-token mask from the cache for the current emitted text.
        boolean[] allowed = cache.getAllowedTokens(constraint, emittedText, vocabSize, idToPiece);

        // Find top-evalTopK indices by raw logit value.
        int k = Math.min(evalTopK, vocabSize);
        int[] topKIndices = topKIndices(logits, k);

        // Check how many top-K candidates are allowed (ignoring terminals for now).
        boolean anyTopKAllowed = false;
        for (int idx : topKIndices) {
            if (idx < 0 || terminals.contains(idx)) continue;
            if (isAllowedToken(idx, allowed, specials, specialPieces, idToPiece)) {
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
                if (idx < 0 || terminals.contains(idx)) continue;
                if (isAllowedToken(idx, allowed, specials, specialPieces, idToPiece)) {
                    masked[idx] = logits[idx];
                }
            }
        } else {
            // Mask strategy B: widen to full vocab — mask only disallowed tokens.
            for (int i = 0; i < vocabSize; i++) {
                masked[i] = isAllowedToken(i, allowed, specials, specialPieces, idToPiece)
                        ? logits[i] : Float.NEGATIVE_INFINITY;
            }
        }

        // Terminal gating: a constraint-owned terminal may be the transition that completes its
        // envelope (for example <|tool_call_end|>). Other stop tokens remain blocked until the
        // constraint is already accepting.
        for (Integer terminal : terminals) {
            if (terminal == null || terminal < 0 || terminal >= vocabSize) {
                continue;
            }
            String piece = idToPiece.apply(terminal);
            boolean ownedTransition = constraint.allowsSpecialToken(emittedText, piece)
                    && constraint.canExtend(emittedText, piece);
            masked[terminal] = accepting || ownedTransition
                    ? logits[terminal]
                    : Float.NEGATIVE_INFINITY;
        }

        return masked;
    }

    /**
     * Builds a constraint mask from the tokenizer's exact decode of the complete candidate
     * sequence. Token-piece lookup is deliberately approximate: for tokenizers with whitespace
     * cleanup or byte-level composition, {@code decode(prefix + token)} is not necessarily equal
     * to {@code decode(prefix) + decode(token)}. This method therefore checks the top-K candidates
     * using their exact sequence decodes and widens to the full vocabulary only when none is legal.
     *
     * @param logits raw vocabulary logits
     * @param stopTokenIds tokens that terminate generation
     * @param idToDecodedCandidate maps a token ID to the exact decode of prefix plus that token
     * @param specialPieces tokenizer control lexemes that must be owned by the constraint
     * @return a fresh exact mask, limited to legal top-K candidates when possible
     */
    public float[] maskLogitsByDecodedCandidate(
            float[] logits,
            Set<Integer> stopTokenIds,
            IntFunction<String> idToDecodedCandidate,
            List<String> specialPieces) {
        return maskLogitsByDecodedCandidate(
                logits,
                stopTokenIds,
                Collections.emptySet(),
                ignored -> null,
                idToDecodedCandidate,
                specialPieces);
    }

    /**
     * Exact candidate masking that retains tokenizer control-token identity.
     *
     * <p>A tokenizer may omit a control token from a full-sequence decode. Text-only validation
     * cannot distinguish that disappearance from a legitimate byte-boundary rewrite and can
     * accidentally reset the constraint state. Registered special-token IDs are therefore checked
     * against the constraint-owned protocol transition before their decoded text is considered.</p>
     */
    public float[] maskLogitsByDecodedCandidate(
            float[] logits,
            Set<Integer> stopTokenIds,
            Set<Integer> specialTokenIds,
            IntFunction<String> idToPiece,
            IntFunction<String> idToDecodedCandidate,
            List<String> specialPieces) {
        if (logits == null) {
            throw new IllegalArgumentException("logits must not be null");
        }
        if (idToDecodedCandidate == null) {
            throw new IllegalArgumentException("idToDecodedCandidate must not be null");
        }
        Set<Integer> terminals = stopTokenIds == null
                ? Collections.emptySet() : stopTokenIds;
        Set<Integer> specialsById = specialTokenIds == null
                ? Collections.emptySet() : specialTokenIds;
        IntFunction<String> pieceLookup = idToPiece == null ? ignored -> null : idToPiece;
        List<String> specials = specialPieces == null
                ? Collections.emptyList() : specialPieces;
        boolean accepting = isComplete();
        int[] topKIndices = topKIndices(logits, Math.min(evalTopK, logits.length));
        boolean anyTopKAllowed = false;
        for (int tokenId : topKIndices) {
            if (tokenId >= 0 && !terminals.contains(tokenId)
                    && allowsDecodedCandidate(
                    tokenId, specialsById, pieceLookup, idToDecodedCandidate, specials)) {
                anyTopKAllowed = true;
                break;
            }
        }

        float[] masked = new float[logits.length];
        java.util.Arrays.fill(masked, Float.NEGATIVE_INFINITY);
        if (anyTopKAllowed) {
            for (int tokenId : topKIndices) {
                if (tokenId >= 0 && !terminals.contains(tokenId)
                        && allowsDecodedCandidate(
                        tokenId, specialsById, pieceLookup, idToDecodedCandidate, specials)) {
                    masked[tokenId] = logits[tokenId];
                }
            }
        } else {
            // Narrow the exact widening pass with the inexpensive piece-level automaton first.
            // Exact full-sequence decode remains the authority, but decoding every vocabulary item
            // through a native tokenizer at every structural boundary is prohibitively expensive.
            boolean[] approximatelyAllowed = cache.getAllowedTokens(
                    constraint, emittedText, logits.length, pieceLookup);
            boolean foundExactCandidate = false;
            for (int tokenId = 0; tokenId < logits.length; tokenId++) {
                if (!terminals.contains(tokenId)
                        && isAllowedToken(
                        tokenId, approximatelyAllowed, specialsById, specials, pieceLookup)
                        && allowsDecodedCandidate(
                        tokenId, specialsById, pieceLookup, idToDecodedCandidate, specials)) {
                    masked[tokenId] = logits[tokenId];
                    foundExactCandidate = true;
                }
            }
            // Piece lookup can be a false negative for byte-level/token-boundary rewrites. Preserve
            // the exact decoder's widening guarantee when the fast candidate set finds nothing.
            if (!foundExactCandidate) {
                for (int tokenId = 0; tokenId < logits.length; tokenId++) {
                    if (!terminals.contains(tokenId)
                            && allowsDecodedCandidate(
                            tokenId, specialsById, pieceLookup, idToDecodedCandidate, specials)) {
                        masked[tokenId] = logits[tokenId];
                    }
                }
            }
        }

        // Preserve the existing terminal semantics: an already complete constraint may stop, and
        // a constraint-owned terminal may itself be the exact transition that closes the envelope.
        for (Integer terminal : terminals) {
            if (terminal == null || terminal < 0 || terminal >= logits.length) {
                continue;
            }
            boolean allowed = accepting
                    || allowsDecodedCandidate(
                    terminal, specialsById, pieceLookup, idToDecodedCandidate, specials);
            masked[terminal] = allowed ? logits[terminal] : Float.NEGATIVE_INFINITY;
        }
        return masked;
    }

    private boolean allowsDecodedCandidate(
            int tokenId,
            Set<Integer> specialTokenIds,
            IntFunction<String> idToPiece,
            IntFunction<String> idToDecodedCandidate,
            List<String> specialPieces) {
        if (specialTokenIds.contains(tokenId)) {
            return allowsSpecialToken(idToPiece.apply(tokenId));
        }
        return allowsDecodedText(idToDecodedCandidate.apply(tokenId), specialPieces);
    }

    private boolean isAllowedToken(
            int tokenId,
            boolean[] allowed,
            Set<Integer> specialTokenIds,
            List<String> specialPieces,
            IntFunction<String> idToPiece) {
        if (!allowed[tokenId]) {
            return false;
        }
        String piece = idToPiece.apply(tokenId);
        if (specialTokenIds.contains(tokenId)) {
            return constraint.allowsSpecialToken(emittedText, piece);
        }
        return !completesUnownedSpecialToken(emittedText, piece, specialPieces);
    }

    private static List<String> specialTokenPieces(
            Set<Integer> specialTokenIds,
            IntFunction<String> idToPiece) {
        if (specialTokenIds.isEmpty()) {
            return Collections.emptyList();
        }
        List<String> pieces = new ArrayList<>(specialTokenIds.size());
        for (Integer tokenId : specialTokenIds) {
            if (tokenId == null || tokenId < 0) {
                continue;
            }
            String piece = idToPiece.apply(tokenId);
            if (piece != null && !piece.isEmpty() && !pieces.contains(piece)) {
                pieces.add(piece);
            }
        }
        return pieces;
    }

    /**
     * Token-ID gating is insufficient when an ordinary-token sequence can spell the same control
     * lexeme. Reject the token that would complete such a lexeme unless the constraint owns that
     * exact protocol transition. This remains tokenizer-driven and does not hard-code model tokens.
     */
    private boolean completesUnownedSpecialToken(
            String currentText,
            String piece,
            List<String> specialPieces) {
        if (piece == null || piece.isEmpty() || specialPieces.isEmpty()) {
            return false;
        }
        String current = currentText == null ? "" : currentText;
        String candidate = current + piece;
        int boundary = current.length();
        for (String special : specialPieces) {
            int searchFrom = Math.max(0, boundary - special.length() + 1);
            int occurrence = candidate.indexOf(special, searchFrom);
            while (occurrence >= 0) {
                int end = occurrence + special.length();
                if (end > boundary
                        && !constraint.allowsSpecialToken(
                                candidate.substring(0, occurrence), special)) {
                    return true;
                }
                occurrence = candidate.indexOf(special, occurrence + 1);
            }
        }
        return false;
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

    /**
     * Validates the exact text produced by decoding the complete candidate token sequence.
     * Singleton token decoding is only an efficient vocabulary approximation and is not
     * compositionally equivalent for every tokenizer.
     */
    public boolean allowsDecodedText(String decodedText, List<String> specialPieces) {
        if (decodedText == null || decodedText.isEmpty()) {
            return false;
        }
        boolean validContinuation;
        if (decodedText.startsWith(emittedText)) {
            String extension = decodedText.substring(emittedText.length());
            validContinuation = !extension.isEmpty()
                    && (constraint.canExtend(emittedText, extension)
                    || constraint.isAccepting(decodedText));
        } else {
            // Some tokenizers rewrite an earlier byte-level boundary when the next token is
            // decoded. In that case there is no stable suffix to validate incrementally. A
            // state-derived constraint cannot validate the whole decode as one giant extension
            // from the empty state: it must observe each protocol transition in order.
            validContinuation = isValidDecodedPrefixFromInitialState(decodedText);
        }
        if (!validContinuation) {
            return false;
        }
        List<String> specials = specialPieces == null
                ? Collections.emptyList() : specialPieces;
        for (String special : specials) {
            if (special == null || special.isEmpty()) {
                continue;
            }
            int occurrence = decodedText.indexOf(special);
            while (occurrence >= 0) {
                if (!constraint.allowsSpecialToken(
                        decodedText.substring(0, occurrence), special)) {
                    return false;
                }
                occurrence = decodedText.indexOf(special, occurrence + 1);
            }
        }
        return true;
    }

    private boolean isValidDecodedPrefixFromInitialState(String decodedText) {
        StringBuilder prefix = new StringBuilder(decodedText.length());
        for (int offset = 0; offset < decodedText.length();) {
            int codePoint = decodedText.codePointAt(offset);
            String piece = new String(Character.toChars(codePoint));
            if (!constraint.canExtend(prefix.toString(), piece)) {
                return false;
            }
            prefix.append(piece);
            offset += Character.charCount(codePoint);
        }
        return true;
    }

    /** Replaces approximate per-token state with the tokenizer's exact sequence decode. */
    public void decodedTextEmitted(String decodedText) {
        if (decodedText == null) {
            throw new IllegalArgumentException("decodedText must not be null");
        }
        emittedText = decodedText;
    }

    /** Returns whether a tokenizer-declared control token is owned at the current protocol state. */
    public boolean allowsSpecialToken(String piece) {
        return piece != null
                && !piece.isEmpty()
                && constraint.allowsSpecialToken(emittedText, piece)
                && constraint.canExtend(emittedText, piece);
    }

    /** Advances the exact state through a constraint-owned tokenizer control token. */
    public void specialTokenEmitted(String piece) {
        if (!allowsSpecialToken(piece)) {
            throw new IllegalArgumentException(
                    "Special token is not allowed at the current constraint state: " + piece);
        }
        emittedText += piece;
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
