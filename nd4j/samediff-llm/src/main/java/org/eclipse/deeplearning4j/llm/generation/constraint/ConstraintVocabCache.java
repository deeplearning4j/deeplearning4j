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

import java.util.HashMap;
import java.util.function.IntFunction;

/**
 * A per-constraint-state cache that pre-computes, for each (emittedText, tokenId) pair,
 * whether that token is an allowed next step.
 *
 * <h2>Cache key</h2>
 * <p>The cache is keyed on the emitted text string (v1 text-based automaton). For each
 * unique emitted-text key, a {@code boolean[vocabSize]} mask is computed and stored.
 * Tokens whose decoded piece is {@code null} or empty are marked as {@code false} by
 * default (they cannot advance any text-based constraint).</p>
 *
 * <h2>Cache sizing and eviction</h2>
 * <p>The cache is capped at {@value #MAX_ENTRIES} entries. When the cap is reached the
 * entire map is cleared (simple "clear on full" policy — adequate for v1 because text
 * grows monotonically during a single generation pass, so re-hits after a clear are
 * unlikely). This avoids the overhead of a true LRU structure while bounding memory
 * use.</p>
 *
 * <h2>Thread safety</h2>
 * <p>This class is <em>not</em> thread-safe. Each generation thread should use its own
 * {@code ConstraintVocabCache} instance.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class ConstraintVocabCache {

    /** Maximum number of emitted-text keys stored before the cache is cleared. */
    static final int MAX_ENTRIES = 512;

    /**
     * Maps emitted-text snapshot → boolean[vocabSize] allowed-token mask.
     * Package-private for testing.
     */
    final HashMap<String, boolean[]> cache = new HashMap<>();

    /**
     * Returns a boolean array of length {@code vocabSize} where {@code result[id] == true}
     * means token {@code id} is an allowed continuation given {@code emittedText} under
     * {@code constraint}.
     *
     * <p>The result is cached: subsequent calls with the same constraint state key
     * ({@code emittedText}) return the cached array directly without re-evaluation.
     * Callers must <em>not</em> mutate the returned array.</p>
     *
     * <p>Tokens whose piece (from {@code idToPiece}) is {@code null} or the empty string
     * are always marked {@code false}; they cannot meaningfully extend any text-based
     * constraint.</p>
     *
     * @param constraint  the active constraint; must be stateless / pure
     * @param emittedText the text generated so far (used as the cache key)
     * @param vocabSize   total vocabulary size
     * @param idToPiece   function mapping token ID to its decoded piece; may return
     *                    {@code null} for out-of-vocabulary or special tokens
     * @return an immutable view of the allowed-token mask (do not mutate)
     */
    public boolean[] getAllowedTokens(
            TextConstraint constraint,
            String emittedText,
            int vocabSize,
            IntFunction<String> idToPiece) {

        boolean[] cached = cache.get(emittedText);
        if (cached != null) {
            return cached;
        }

        // Evict if at capacity (simple clear-on-full strategy).
        if (cache.size() >= MAX_ENTRIES) {
            cache.clear();
        }

        boolean[] allowed = computeAllowedTokens(constraint, emittedText, vocabSize, idToPiece);
        cache.put(emittedText, allowed);
        return allowed;
    }

    /**
     * Computes the allowed-token mask from scratch for the given state.
     *
     * @param constraint  the active constraint
     * @param emittedText the text generated so far
     * @param vocabSize   total vocabulary size
     * @param idToPiece   token-ID to piece decoder
     * @return a freshly allocated {@code boolean[vocabSize]}
     */
    private static boolean[] computeAllowedTokens(
            TextConstraint constraint,
            String emittedText,
            int vocabSize,
            IntFunction<String> idToPiece) {

        boolean[] allowed = new boolean[vocabSize];
        for (int id = 0; id < vocabSize; id++) {
            String piece = idToPiece.apply(id);
            if (piece == null || piece.isEmpty()) {
                // Null/empty pieces cannot advance a text-based constraint.
                allowed[id] = false;
            } else {
                allowed[id] = constraint.canExtend(emittedText, piece);
            }
        }
        return allowed;
    }

    /**
     * Clears all cached entries.  Useful for resetting between generation calls.
     */
    public void clear() {
        cache.clear();
    }

    /**
     * Returns the current number of entries held in the cache.
     *
     * @return cache size
     */
    public int size() {
        return cache.size();
    }
}
