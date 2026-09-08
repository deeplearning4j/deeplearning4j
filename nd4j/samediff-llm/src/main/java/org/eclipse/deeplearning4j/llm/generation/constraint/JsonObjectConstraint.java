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

import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonParser;
import org.nd4j.shade.jackson.core.JsonToken;
import org.nd4j.shade.jackson.core.async.ByteArrayFeeder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * A {@link TextConstraint} that accepts exactly one syntactically valid JSON object.
 *
 * <p>Jackson's non-blocking parser validates syntax without treating the end of a
 * candidate token piece as end-of-input. Unfinished strings, escapes, numbers and
 * literals remain valid prefixes; invalid keys, values and nesting are rejected.
 * Exactly one root object is permitted, with only JSON whitespace after it.</p>
 *
 * @author Eclipse Deeplearning4j Contributors
 * @see ToolCallConstraint
 */
public class JsonObjectConstraint implements TextConstraint {

    /** Type identifier returned by {@link #type()}. */
    public static final String TYPE = "json_object";

    private static final JsonFactory JSON_FACTORY = new JsonFactory()
            .disable(JsonFactory.Feature.CANONICALIZE_FIELD_NAMES)
            .disable(JsonFactory.Feature.INTERN_FIELD_NAMES);

    private static final class ParseState {
        boolean invalid;
        boolean complete;
    }

    static ParseState runStateMachine(String text) {
        ParseState s = new ParseState();
        if (!hasValidLiteralPrefixes(text)) {
            s.invalid = true;
            return s;
        }
        byte[] input = text.getBytes(StandardCharsets.UTF_8);
        try (JsonParser parser = JSON_FACTORY.createNonBlockingByteArrayParser()) {
            ((ByteArrayFeeder) parser.getNonBlockingInputFeeder()).feedInput(input, 0, input.length);
            int depth = 0;
            boolean rootStarted = false;
            JsonToken token;
            while ((token = parser.nextToken()) != JsonToken.NOT_AVAILABLE && token != null) {
                if (!rootStarted) {
                    if (token != JsonToken.START_OBJECT) {
                        s.invalid = true;
                        return s;
                    }
                    rootStarted = true;
                }
                if (token == JsonToken.START_OBJECT || token == JsonToken.START_ARRAY) {
                    depth++;
                } else if (token == JsonToken.END_OBJECT || token == JsonToken.END_ARRAY) {
                    depth--;
                    if (depth == 0) {
                        // Do not allow a second root or even an incomplete trailing token.
                        int consumed = Math.toIntExact(parser.getCurrentLocation().getByteOffset());
                        for (int i = consumed; i < input.length; i++) {
                            if (!isJsonWhitespace((char) input[i])) {
                                s.invalid = true;
                                return s;
                            }
                        }
                        s.complete = true;
                        return s;
                    }
                }
            }
        } catch (IOException e) {
            s.invalid = true;
        }
        return s;
    }

    // The async parser accumulates malformed keywords until a delimiter arrives.
    // A token candidate must already be a possible prefix, not merely need more bytes.
    private static boolean hasValidLiteralPrefixes(String text) {
        boolean inString = false;
        boolean escaped = false;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == '"') inString = false;
                continue;
            }
            if (c == '"') {
                inString = true;
                continue;
            }
            String literal = c == 't' ? "true" : c == 'f' ? "false" : c == 'n' ? "null" : null;
            if (literal == null) continue;
            for (int j = 0; j < literal.length(); j++) {
                if (i + j == text.length()) return true;
                if (text.charAt(i + j) != literal.charAt(j)) return false;
            }
            i += literal.length() - 1;
            if (i + 1 < text.length()) {
                char next = text.charAt(i + 1);
                if (!isJsonWhitespace(next) && next != ',' && next != ']' && next != '}') return false;
            }
        }
        return true;
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
     * Returns true for a syntactically valid prefix of a single JSON object.
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
        return !s.invalid;
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
        return !s.invalid && s.complete;
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
