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
 * A {@link TextConstraint} that accepts exactly one syntactically valid JSON object.
 *
 * <p>The constraint uses an incremental, single-pass state machine over the accumulated
 * character stream to decide whether a candidate extension is a legal JSON-object prefix.
 * It does <em>not</em> use a real JSON parser; instead, it tracks the minimal structural
 * invariants needed to enforce correct nesting for tool-calling use cases:</p>
 *
 * <ul>
 *   <li>String boundaries ({@code "}) — including backslash escape handling inside strings</li>
 *   <li>Brace depth ({@code { }}) — must never go negative; object is complete when it
 *       returns to 0 after the opening {@code {}</li>
 *   <li>Bracket depth ({@code [ ]}) — must never go negative</li>
 * </ul>
 *
 * <p>The state machine deliberately ignores JSON value-level validity (e.g., numeric
 * format, keyword spelling) to stay O(n) and allocation-free. This is intentional:
 * the LLM's probability distribution already encodes syntactic preferences; we only
 * enforce the structural skeleton.</p>
 *
 * <h2>Acceptance rule</h2>
 * <p>A string is <em>accepting</em> (complete) when:</p>
 * <ul>
 *   <li>It starts with {@code {}</li>
 *   <li>brace depth == 0</li>
 *   <li>bracket depth == 0</li>
 *   <li>not inside a string literal</li>
 *   <li>total trimmed length &gt; 0</li>
 * </ul>
 *
 * <h2>Extension rule</h2>
 * <p>{@code canExtend(currentText, piece)} returns {@code true} when
 * {@code currentText + piece} is still a valid JSON-object prefix — i.e., the state
 * machine reports brace_depth &gt;= 0 and bracket_depth &gt;= 0 and the combined string
 * starts with (or could start with) {@code {}.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see ToolCallConstraint
 */
public class JsonObjectConstraint implements TextConstraint {

    /** Type identifier returned by {@link #type()}. */
    public static final String TYPE = "json_object";

    /**
     * Compact holder for the result of running the state machine over a string.
     */
    private static final class ParseState {
        boolean inString;
        boolean escapeNext;
        int braceDepth;
        int bracketDepth;
        /** Set to true the moment an invariant is violated (depth goes negative). */
        boolean invalid;

        ParseState() {
            this.inString = false;
            this.escapeNext = false;
            this.braceDepth = 0;
            this.bracketDepth = 0;
            this.invalid = false;
        }
    }

    /**
     * Run the JSON-structure state machine over {@code text} and return the resulting
     * parse state.
     *
     * <p>The machine is deliberately not a full JSON parser — it only tracks nesting
     * depth and string boundaries so it can remain O(n) and allocation-free.</p>
     *
     * @param text the string to analyse
     * @return the final state after processing every character
     */
    static ParseState runStateMachine(String text) {
        ParseState s = new ParseState();
        for (int i = 0, len = text.length(); i < len; i++) {
            char c = text.charAt(i);

            if (s.inString) {
                if (s.escapeNext) {
                    // Any character following a backslash is consumed as an escape sequence.
                    s.escapeNext = false;
                } else if (c == '\\') {
                    s.escapeNext = true;
                } else if (c == '"') {
                    s.inString = false;
                } else if (c <= 0x1f) {
                    // RFC 8259 requires control characters in strings to be escaped.
                    s.invalid = true;
                    return s;
                }
                // All other characters inside a string are just content — skip depth tracking.
            } else {
                switch (c) {
                    case '{':
                        s.braceDepth++;
                        break;
                    case '}':
                        s.braceDepth--;
                        if (s.braceDepth < 0) {
                            s.invalid = true;
                            return s;
                        }
                        break;
                    case '[':
                        s.bracketDepth++;
                        break;
                    case ']':
                        s.bracketDepth--;
                        if (s.bracketDepth < 0) {
                            s.invalid = true;
                            return s;
                        }
                        break;
                    case '"':
                        s.inString = true;
                        break;
                    default:
                        // JSON admits only SP, HTAB, LF, and CR as structural whitespace.
                        // Reject other control characters instead of giving generation a
                        // non-advancing form-feed/vertical-tab loop.
                        if (c <= 0x1f && !isJsonWhitespace(c)) {
                            s.invalid = true;
                            return s;
                        }
                        // Numbers, colons, commas, and legal JSON whitespace have no
                        // structural impact in this deliberately lightweight automaton.
                        break;
                }
            }
        }
        return s;
    }

    static boolean isJsonWhitespace(char value) {
        return value == ' ' || value == '\t' || value == '\r' || value == '\n';
    }

    static String stripLeadingJsonWhitespace(String value) {
        if (value == null || value.isEmpty()) {
            return "";
        }
        int start = 0;
        while (start < value.length() && isJsonWhitespace(value.charAt(start))) {
            start++;
        }
        return value.substring(start);
    }

    static String stripJsonWhitespace(String value) {
        String leadingStripped = stripLeadingJsonWhitespace(value);
        int end = leadingStripped.length();
        while (end > 0 && isJsonWhitespace(leadingStripped.charAt(end - 1))) {
            end--;
        }
        return leadingStripped.substring(0, end);
    }

    static boolean isOnlyJsonWhitespace(String value) {
        if (value == null || value.isEmpty()) {
            return false;
        }
        for (int index = 0; index < value.length(); index++) {
            if (!isJsonWhitespace(value.charAt(index))) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns {@code true} if {@code text} is a valid JSON-object prefix, meaning:
     * <ul>
     *   <li>It is empty (the opening {@code {} has not been emitted yet, which is a
     *       valid prefix of any JSON object); OR</li>
     *   <li>It starts with {@code {}, brace_depth &gt;= 0, bracket_depth &gt;= 0, and
     *       the state machine did not encounter a structural violation</li>
     * </ul>
     *
     * @param text the accumulated text to test
     * @return {@code true} if {@code text} is a valid prefix of some JSON object
     */
    static boolean isValidJsonPrefix(String text) {
        if (text.isEmpty()) {
            // Empty string is a valid prefix of any JSON object.
            return true;
        }
        // The first non-whitespace character must be '{'.
        for (int i = 0, len = text.length(); i < len; i++) {
            char c = text.charAt(i);
            if (c == '{') {
                break;
            } else if (isJsonWhitespace(c)) {
                continue;
            } else {
                // First non-whitespace is not '{' — not a JSON object prefix.
                return false;
            }
        }
        ParseState s = runStateMachine(text);
        return !s.invalid && s.braceDepth >= 0 && s.bracketDepth >= 0;
    }

    // -------------------------------------------------------------------------
    // TextConstraint implementation
    // -------------------------------------------------------------------------

    @Override
    public boolean canExtend(String currentText, String piece) {
        if (piece == null || piece.isEmpty()) {
            // An empty piece does not advance the state — allowed unless already invalid.
            return isValidJsonPrefix(currentText);
        }
        String combined = currentText + piece;
        return isValidJsonPrefix(combined);
    }

    @Override
    public boolean isAccepting(String currentText) {
        if (currentText == null || stripJsonWhitespace(currentText).isEmpty()) {
            return false;
        }
        // Must start with '{' (ignoring leading whitespace).
        boolean foundOpen = false;
        for (int i = 0, len = currentText.length(); i < len; i++) {
            char c = currentText.charAt(i);
            if (c == '{') {
                foundOpen = true;
                break;
            } else if (!isJsonWhitespace(c)) {
                return false;
            }
        }
        if (!foundOpen) {
            return false;
        }
        ParseState s = runStateMachine(currentText);
        return !s.invalid
                && s.braceDepth == 0
                && s.bracketDepth == 0
                && !s.inString;
    }

    @Override
    public TextConstraint reset() {
        // This implementation is stateless — the same instance can be reused.
        return new JsonObjectConstraint();
    }

    @Override
    public String type() {
        return TYPE;
    }
}
