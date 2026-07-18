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

/**
 * A constraint on the text produced by a language model.
 *
 * <p>TextConstraint implementations define a formal language (or structural shape) that
 * generated text must belong to. During constrained decoding, the sampler uses
 * {@link #canExtend(String, String)} at each step to decide which token pieces are
 * legal continuations given the text emitted so far, and {@link #isAccepting(String)}
 * to decide when generation may stop.</p>
 *
 * <p>The constraint contract is prefix-monotone: if {@code canExtend(t, piece)} returns
 * {@code false} for every non-empty piece, then {@code t} is a dead end (generation
 * should stop or backtrack). Implementations must be pure — the same inputs always
 * produce the same outputs — so they can be safely used inside the vocabulary cache
 * ({@link ConstraintVocabCache}) without external locking.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * TextConstraint c = new JsonObjectConstraint();
 * String emitted = "";
 * // For each candidate token piece:
 * if (c.canExtend(emitted, piece)) {
 *     emitted += piece;
 * }
 * boolean done = c.isAccepting(emitted);
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see JsonObjectConstraint
 * @see ToolCallConstraint
 */
public interface TextConstraint {

    /**
     * Returns {@code true} if appending {@code piece} to {@code currentText} keeps the
     * accumulated string as a valid prefix of some accepting string.
     *
     * <p>Implementations must not modify any state; this method is called speculatively
     * for every candidate token in the vocabulary.</p>
     *
     * @param currentText the text emitted so far (may be empty)
     * @param piece       the candidate token piece to test (never null, may be empty)
     * @return {@code true} if {@code currentText + piece} is still a valid prefix
     */
    boolean canExtend(String currentText, String piece);

    /**
     * Returns {@code true} if {@code currentText} is in a complete, accepting state —
     * i.e., generation may legally stop here.
     *
     * @param currentText the text emitted so far
     * @return {@code true} if the text satisfies the constraint fully
     */
    boolean isAccepting(String currentText);

    /**
     * Returns a new instance of the same constraint type in its initial state.
     *
     * <p>Used for reuse across multiple generation calls without allocating a new
     * implementation class each time. Implementations that are stateless may return
     * {@code this}.</p>
     *
     * @return a reset instance of this constraint
     */
    TextConstraint reset();

    /**
     * A short, stable type identifier for this constraint — used in logging and
     * to satisfy the ADR options contract for constraint configuration.
     *
     * <p>Conventional values: {@code "json_object"}, {@code "tool_call"}.</p>
     *
     * @return the constraint type identifier
     */
    String type();
}
