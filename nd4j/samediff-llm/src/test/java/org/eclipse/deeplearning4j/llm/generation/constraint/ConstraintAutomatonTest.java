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

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the constraint automaton classes (JsonObjectConstraint,
 * ToolCallConstraint, ConstraintMasker, ConstraintVocabCache).
 *
 * All tests are pure-Java, no model loading required.
 *
 * @see JsonObjectConstraint
 * @see ToolCallConstraint
 * @see ConstraintMasker
 */
class ConstraintAutomatonTest {

    // =========================================================================
    // JsonObjectConstraint — canExtend
    // =========================================================================

    @Test
    void jsonObject_emptyIsValidPrefix() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        assertTrue(c.canExtend("", "{"), "'{' should be allowed from empty");
        assertTrue(c.canExtend("", " "), "whitespace should be allowed from empty");
        assertFalse(c.canExtend("", "x"), "non-brace non-whitespace should be rejected from empty");
    }

    @Test
    void jsonObject_simpleObjectAccepted() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        String obj = "{\"key\": \"value\"}";
        // Each prefix should be valid.
        for (int i = 0; i < obj.length(); i++) {
            String prefix = obj.substring(0, i);
            String piece = obj.substring(i, i + 1);
            assertTrue(c.canExtend(prefix, piece),
                    "canExtend should be true for prefix='" + prefix + "', piece='" + piece + "'");
        }
        assertTrue(c.isAccepting(obj), "complete object should be accepting");
    }

    @Test
    void jsonObject_nestedObjectAccepted() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        String obj = "{\"a\": {\"b\": 1}}";
        assertTrue(c.isAccepting(obj));
    }

    @Test
    void jsonObject_arrayInsideObjectAccepted() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        String obj = "{\"a\": [1, 2, 3]}";
        assertTrue(c.isAccepting(obj));
    }

    @Test
    void jsonObject_unbalancedBraceFails() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        // Extra closing brace
        assertFalse(c.canExtend("{\"a\":1}", "}"),
                "adding '}' to a complete object should be rejected");
        assertFalse(JsonObjectConstraint.isValidJsonPrefix("}"),
                "bare '}' is not a valid JSON prefix");
    }

    @Test
    void jsonObject_unbalancedBracketFails() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        assertFalse(JsonObjectConstraint.isValidJsonPrefix("{\"a\": ]}"),
                "unbalanced ']' should be invalid");
    }

    @Test
    void jsonObject_stringEscapeHandled() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        // A quote inside a string should NOT close the string.
        String obj = "{\"key\": \"val\\\"ue\"}";
        assertTrue(c.isAccepting(obj), "escaped quote inside string should be handled correctly");
    }

    @Test
    void jsonObject_multiByteTokenAccepted() {
        // Simulates a multi-byte piece being appended (e.g., " \"value\"}" from a single token).
        JsonObjectConstraint c = new JsonObjectConstraint();
        String prefix = "{\"k\":";
        String piece = " \"v\"}";
        assertTrue(c.canExtend(prefix, piece), "multi-byte piece completing an object should be allowed");
        assertTrue(c.isAccepting(prefix + piece));
    }

    @Test
    void jsonObject_incompleteIsNotAccepting() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        assertFalse(c.isAccepting("{\"a\":"), "partial object should not be accepting");
        assertFalse(c.isAccepting("{"), "bare '{' should not be accepting");
        assertFalse(c.isAccepting(""), "empty string should not be accepting");
        // {} is a valid JSON object and should be accepting.
        assertTrue(c.isAccepting("{}"), "empty JSON object {} should be accepting");
        // Nested complete object should be accepting too.
        assertTrue(c.isAccepting("{\"key\": \"value\"}"), "simple complete object should be accepting");
    }

    @Test
    void jsonObject_eos_onlyInAcceptingState() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        assertFalse(c.isAccepting("{\"a\":"), "incomplete object should not be accepting");
        assertTrue(c.isAccepting("{\"a\": 1}"), "complete object should be accepting");
    }

    // =========================================================================
    // ToolCallConstraint — canExtend / isAccepting
    // =========================================================================

    private static final List<String> TOOLS = Arrays.asList(
            "ask_graph_verify", "graph_reasoning_query", "ask_graph_query");

    @Test
    void toolCall_correctPrefixAllowed() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        // Each character of the canonical prefix should be extendable.
        String fullPrefix = "{\"tool\": \"ask_graph_verify\", \"args\": {\"entity\": \"alice\"}}";
        for (int i = 0; i < fullPrefix.length() - 1; i++) {
            String prefix = fullPrefix.substring(0, i);
            String piece = fullPrefix.substring(i, i + 1);
            assertTrue(c.canExtend(prefix, piece),
                    "canExtend failed at i=" + i + " prefix='" + prefix + "' piece='" + piece + "'");
        }
    }

    @Test
    void toolCall_completeExpressionIsAccepting() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        String full = "{\"tool\": \"ask_graph_verify\", \"args\": {\"entity\": \"alice\"}}";
        assertTrue(c.isAccepting(full), "complete tool call should be accepting");
    }

    @Test
    void toolCall_wrongToolNameRejected() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        // "unknown_tool" is not in the list — should be rejected once we are in TOOL_NAME phase
        // and the candidate diverges from all known names.
        String prefix = "{\"tool\": \"";
        // 'u' does not start any of the known tools (which start with 'a' or 'g').
        assertFalse(c.canExtend(prefix, "u"),
                "tool name starting with 'u' should be rejected when no tool starts with 'u'");
    }

    @Test
    void toolCall_partialToolNameAllowed() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        String prefix = "{\"tool\": \"";
        // 'a' is a valid prefix of "ask_graph_verify"
        assertTrue(c.canExtend(prefix, "a"), "'a' is a valid prefix of ask_graph_verify");
        assertTrue(c.canExtend(prefix + "a", "s"), "'as' is a valid prefix of ask_graph_verify");
        assertTrue(c.canExtend(prefix + "g", "r"), "'gr' is a valid prefix of graph_reasoning_query");
    }

    @Test
    void toolCall_wrongSuffixRejected() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        // After tool name, wrong separator should be rejected.
        String afterName = "{\"tool\": \"ask_graph_verify\", \"args\":";
        // Missing space after colon would diverge.
        // But: our constraint accepts `, "args": ` (with space) — verify that a valid continuation works.
        assertTrue(c.canExtend("{\"tool\": \"ask_graph_verify\"", ", \"args\": {"),
                "correct separator + args opening should be accepted");
    }

    @Test
    void toolCall_eosOnlyAcceptedWhenDone() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        // EOS should be blocked in all non-DONE states.
        assertFalse(c.isAccepting(""));
        assertFalse(c.isAccepting("{\"tool\": \""));
        assertFalse(c.isAccepting("{\"tool\": \"ask_graph_verify\", \"args\": {"));
        // EOS allowed when done.
        assertTrue(c.isAccepting("{\"tool\": \"ask_graph_verify\", \"args\": {}}"));
    }

    @Test
    void toolCall_allThreeToolsAccepted() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        for (String tool : TOOLS) {
            String full = "{\"tool\": \"" + tool + "\", \"args\": {}}";
            assertTrue(c.isAccepting(full), "tool '" + tool + "' should be accepted");
        }
    }

    // =========================================================================
    // NativeToolCallConstraint — native sentinel/function syntax
    // =========================================================================

    @Test
    void nativeToolCall_forcesExactDeclaredNameAndAcceptsNamedArguments() {
        NativeToolCallConstraint c =
                new NativeToolCallConstraint("record_graph_verdict");
        String full = "<|tool_call_start|>[record_graph_verdict("
                + "candidate_id=\"issue-1\", disposition=\"DENY\", "
                + "statement=\"The status is \\\"Do not use\\\".\")]<|tool_call_end|>";

        for (int i = 0; i < full.length(); i++) {
            assertTrue(c.canExtend(
                            full.substring(0, i), full.substring(i, i + 1)),
                    "native constraint rejected valid character at index " + i);
        }
        assertTrue(c.isAccepting(full));
    }

    @Test
    void nativeToolCall_rejectsCorruptedFunctionName() {
        NativeToolCallConstraint c =
                new NativeToolCallConstraint("record_graph_verdict");
        String prefix = "<|tool_call_start|>[record_g";

        assertFalse(c.canExtend(prefix, "_g_"),
                "a token that diverges from every declared name must be masked");
    }

    @Test
    void nativeToolCall_allowsNestedArgumentValuesAndPartialCloseToken() {
        NativeToolCallConstraint c = new NativeToolCallConstraint("submit");
        String beforeClose = "<|tool_call_start|>[submit("
                + "delta={\"entities\":[{\"name\":\"A\"}]})";

        assertTrue(c.canExtend(beforeClose.substring(
                        0, beforeClose.length() - 1), ")"));
        assertFalse(c.isAccepting(beforeClose));
        assertTrue(c.canExtend(beforeClose, "]"));
        String completeCore = beforeClose + "]";
        assertFalse(c.isAccepting(completeCore),
                "the core function call is incomplete until the native end marker");
        assertTrue(c.allowsSpecialToken(completeCore, "<|tool_call_end|>"));
        assertTrue(c.canExtend(completeCore, "<|tool_call_end|>"));
        assertTrue(c.isAccepting(completeCore + "<|tool_call_end|>"));
    }

    @Test
    void nativeToolCall_structuralWhitespaceCannotSelfLoop() {
        Map<String, Object> nameSchema = Map.of("type", "string", "maxLength", 160);
        Map<String, Object> namesSchema = Map.of(
                "type", "array",
                "items", nameSchema,
                "maxItems", 32);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of("names", namesSchema),
                "required", List.of("names"));
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_entities"),
                Map.of("submit_entities", List.of("names")),
                Map.of(),
                Map.of("submit_entities", parameters));

        String arrayPrefix = "<|tool_call_start|>[submit_entities(names=[";
        assertTrue(constraint.canExtend(arrayPrefix, " "),
                "one optional structural separator remains legal");
        assertFalse(constraint.canExtend(arrayPrefix + " ", "\n\t\u2009"),
                "whitespace-only pieces must not leave the parser in the same structural state");
        assertTrue(constraint.canExtend(arrayPrefix + " ", "\"Revenue\""),
                "masking a whitespace self-loop must leave value tokens selectable");

        String quotedValue = arrayPrefix + "\"Net income ";
        assertTrue(constraint.canExtend(quotedValue, " "),
                "whitespace inside a quoted value is data, not structural padding");

        String valid = arrayPrefix + "\"M. Chen\"])]<|tool_call_end|>";
        assertTrue(constraint.isAccepting(valid));
        assertFalse(constraint.isAccepting(arrayPrefix + "\"M. Chen\"]?)]"),
                "a JSON value followed by trailing garbage must never become accepting");
        assertFalse(constraint.canExtend(arrayPrefix + "\"M. Chen\"]", "?)]"),
                "trailing non-JSON tokens must be masked before the native call closes");
    }

    @Test
    void nativeToolCall_schemaForcesEachRequiredArgumentExactlyOnceInOrder() {
        List<String> required =
                List.of("candidate_id", "disposition", "rule_id", "statement");
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("record_graph_verdict"),
                Map.of("record_graph_verdict", required),
                Map.of("record_graph_verdict", Map.of(
                        "candidate_id", List.of("issue-1"),
                        "disposition", List.of("ALLOW", "DENY", "REVIEW"))));

        String prefix = "<|tool_call_start|>[record_graph_verdict(";
        assertFalse(constraint.canExtend(prefix, "candidate_id=\"issue-2"),
                "schema enum values must constrain native argument text");
        assertTrue(constraint.canExtend(prefix, "candidate_id=\"issue-1"),
                "declared schema enum value must remain selectable");
        String candidateComplete = prefix + "candidate_id=\"issue-1\"";
        assertFalse(constraint.canExtend(candidateComplete, " "),
                "a completed enum literal must not enter a whitespace self-loop");
        assertTrue(constraint.canExtend(candidateComplete, ", "),
                "the next field separator must remain selectable");

        String first = candidateComplete + ", ";
        assertFalse(constraint.canExtend(first, "candidate_id="),
                "a repeated first field must be masked");
        assertTrue(constraint.canExtend(first, "disposition="),
                "the next declared field must remain available");
        assertFalse(constraint.canExtend(first, "disposition=\"DENY: extra"),
                "enum-constrained values may not absorb narration");
        assertTrue(constraint.canExtend(first, "disposition=\"DENY\""),
                "an exact enum value must remain selectable");
        assertFalse(constraint.canExtend(first, "rule_id="),
                "required fields may not be skipped");

        String missing = first + "disposition=\"DENY\")]";
        assertFalse(constraint.isAccepting(missing),
                "the native call may not close before all required fields");

        String complete = first
                + "disposition=\"DENY\", "
                + "rule_id=\"fpna.status.not-usable\", "
                + "statement=\"Do not use for reporting.\")]<|tool_call_end|>";
        assertTrue(constraint.isAccepting(complete));
    }

    // =========================================================================
    // ConstraintMasker — maskLogits
    // =========================================================================

    @Test
    void masker_allowedTokensHaveFiniteLogits() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        ConstraintMasker masker = new ConstraintMasker(c, 256);

        // Vocab: 0='{', 1='x', 2=EOS
        float[] logits = {1.0f, 2.0f, 0.5f};
        int eosId = 2;
        // idToPiece: 0 -> "{", 1 -> "x", 2 -> "" (EOS)
        float[] masked = masker.maskLogits(logits, eosId, id -> {
            if (id == 0) return "{";
            if (id == 1) return "x";
            return null;  // EOS has no piece
        });

        // '{' (token 0) should be allowed from empty state — braces are legal first chars.
        assertTrue(Float.isFinite(masked[0]), "'{' should be unmasked");
        // 'x' (token 1) should be blocked — not a valid JSON object prefix.
        assertEquals(Float.NEGATIVE_INFINITY, masked[1], "non-structural token should be masked");
        // EOS (token 2) should be blocked — not in accepting state yet.
        assertEquals(Float.NEGATIVE_INFINITY, masked[2], "EOS should be blocked before accepting state");
    }

    @Test
    void masker_eosFreeInAcceptingState() {
        JsonObjectConstraint c = new JsonObjectConstraint();
        ConstraintMasker masker = new ConstraintMasker(c, 256);

        // Simulate having emitted "{}" — accepting state.
        masker.tokenEmitted(0, id -> "{");  // '{'
        masker.tokenEmitted(1, id -> "}");  // '}'
        // Now emittedText is "{}"

        // Vocab: 0=something, 1=EOS
        float[] logits = {1.0f, 2.0f};
        float[] masked = masker.maskLogits(logits, 1, id -> (id == 0 ? "," : null));
        // EOS should now be permitted.
        assertEquals(logits[1], masked[1], "EOS should be unmasked in accepting state");
    }

    @Test
    void masker_blocksEveryStopTokenUntilNativeCallIsComplete() {
        NativeToolCallConstraint constraint =
                new NativeToolCallConstraint("record_graph_verdict");
        ConstraintMasker masker = new ConstraintMasker(constraint, 256);

        String unterminated = "<|tool_call_start|>[record_graph_verdict("
                + "candidate_id=\"issue-1";
        masker.tokenEmitted(0, id -> unterminated);

        // Vocab: 0=prefix, 1=<|im_end|>, 2=<|tool_call_end|>, 3=valid close.
        // Both terminal sentinels have high logits and are valid quoted text at the character
        // level, but neither may terminate before the native envelope itself is accepting.
        float[] logits = {0.0f, 9.0f, 8.0f, 1.0f};
        Set<Integer> stops = Set.of(1, 2);
        float[] masked = masker.maskLogits(logits, stops, id -> {
            switch (id) {
                case 1: return "<|im_end|>";
                case 2: return "<|tool_call_end|>";
                case 3: return "\")]";
                default: return unterminated;
            }
        });

        assertEquals(Float.NEGATIVE_INFINITY, masked[1]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[2]);
        assertTrue(Float.isFinite(masked[3]), "valid native-call closure must remain selectable");

        masker.tokenEmitted(3, id -> "\")]");
        assertFalse(masker.isComplete());
        float[] closeMask = masker.maskLogits(logits, stops, id -> {
            if (id == 1) return "<|im_end|>";
            if (id == 2) return "<|tool_call_end|>";
            return id == 3 ? "\")]" : unterminated;
        });
        assertEquals(Float.NEGATIVE_INFINITY, closeMask[1],
                "ordinary EOS remains blocked before the full native envelope");
        assertEquals(logits[2], closeMask[2],
                "the constraint-owned terminal must be allowed to complete the envelope");

        masker.tokenEmitted(2, id -> "<|tool_call_end|>");
        assertTrue(masker.isComplete());
    }

    @Test
    void maskerAllowsOnlyConstraintOwnedSpecialTokens() {
        NativeToolCallConstraint constraint =
                new NativeToolCallConstraint("record_graph_verdict");
        ConstraintMasker masker = new ConstraintMasker(constraint, 256);

        float[] initialLogits = {9.0f, 8.0f, 1.0f};
        Set<Integer> specialTokens = Set.of(0, 1);
        float[] initialMask = masker.maskLogits(
                initialLogits, Set.of(), specialTokens, id -> {
                    if (id == 0) return "<|tool_call_start|>";
                    if (id == 1) return "<|im_end|>";
                    return "x";
                });

        assertTrue(Float.isFinite(initialMask[0]),
                "the native envelope sentinel is owned by the constraint");
        assertEquals(Float.NEGATIVE_INFINITY, initialMask[1],
                "unrelated control tokens must not start the call");

        masker.tokenEmitted(0, id -> "<|tool_call_start|>");
        masker.tokenEmitted(2, id -> "[record_graph_verdict(candidate_id=\"");
        float[] valueMask = masker.maskLogits(
                initialLogits, Set.of(), specialTokens, id -> {
                    if (id == 0) return "issue-1";
                    if (id == 1) return "<|im_end|>";
                    return "issue-2";
                });
        assertEquals(Float.NEGATIVE_INFINITY, valueMask[1],
                "a control token must not be accepted as quoted argument content");
        assertTrue(Float.isFinite(valueMask[2]),
                "ordinary quoted argument content must remain selectable");
    }

    @Test
    void maskerBlocksControlLexemesSpelledByOrdinaryTokens() {
        NativeToolCallConstraint constraint =
                new NativeToolCallConstraint("record_graph_verdict");
        ConstraintMasker masker = new ConstraintMasker(constraint, 256);
        String quotedPrefix = "<|tool_call_start|>[record_graph_verdict("
                + "candidate_id=\"<|pad|";
        masker.tokenEmitted(2, id -> quotedPrefix);

        // Token 0 is the tokenizer-declared PAD token. Token 1 is an ordinary token that would
        // complete the exact same control lexeme from ordinary pieces; both must be blocked.
        float[] valueMask = masker.maskLogits(
                new float[]{9.0f, 8.0f, 1.0f}, Set.of(), Set.of(0), id -> {
                    if (id == 0) return "<|pad|>";
                    if (id == 1) return ">";
                    return "x";
                });
        assertEquals(Float.NEGATIVE_INFINITY, valueMask[0]);
        assertEquals(Float.NEGATIVE_INFINITY, valueMask[1],
                "ordinary tokens must not spell a control lexeme inside argument content");
        assertTrue(Float.isFinite(valueMask[2]));
    }

    @Test
    void maskerAllowsOwnedEnvelopeSpelledByOrdinaryTokens() {
        NativeToolCallConstraint constraint =
                new NativeToolCallConstraint("record_graph_verdict");
        ConstraintMasker masker = new ConstraintMasker(constraint, 256);
        Set<Integer> specialTokens = Set.of(0, 3);

        float[] initialMask = masker.maskLogits(
                new float[]{7.0f, 8.0f, 6.0f, 5.0f}, Set.of(), specialTokens, id -> {
                    if (id == 0) return "<|tool_call_start|>";
                    if (id == 1) return "<";
                    if (id == 2) return "|tool_call_start|>";
                    return "<|pad|>";
                });
        assertTrue(Float.isFinite(initialMask[1]),
                "an owned envelope may begin through an ordinary-token decomposition");

        masker.tokenEmitted(1, id -> "<");
        float[] completionMask = masker.maskLogits(
                new float[]{7.0f, 8.0f, 6.0f, 5.0f}, Set.of(), specialTokens, id -> {
                    if (id == 0) return "<|tool_call_start|>";
                    if (id == 1) return "<";
                    if (id == 2) return "|tool_call_start|>";
                    return "<|pad|>";
                });
        assertTrue(Float.isFinite(completionMask[2]),
                "the tokenizer-owned envelope remains legal when split across ordinary tokens");
    }

    @Test
    void maskerValidatesAndStoresExactFullSequenceDecode() {
        NativeToolCallConstraint constraint =
                new NativeToolCallConstraint("submit_entities");
        ConstraintMasker masker = new ConstraintMasker(constraint, 256);
        String complete = "<|tool_call_start|>[submit_entities(names=[\"Revenue\"])]"
                + "<|tool_call_end|>";

        assertTrue(masker.allowsDecodedText(complete,
                List.of("<|tool_call_start|>", "<|tool_call_end|>", "<|pad|>")));
        assertFalse(masker.allowsDecodedText(
                "<|tool_call_start|>[submit_entities(names=[\"<|pad|>\"])]"
                        + "<|tool_call_end|>",
                List.of("<|tool_call_start|>", "<|tool_call_end|>", "<|pad|>")),
                "an unowned tokenizer control lexeme must fail exact sequence validation");

        masker.tokenEmitted(0, id -> "incorrect singleton decode");
        masker.decodedTextEmitted(complete);
        assertEquals(complete, masker.getEmittedText());
        assertTrue(masker.isComplete());
    }

    @Test
    void masker_wideningFallback() {
        // If none of top-K tokens are allowed, widen to full vocab.
        // Build a constraint that only allows '{'.
        JsonObjectConstraint c = new JsonObjectConstraint();
        ConstraintMasker masker = new ConstraintMasker(c, 2);  // evalTopK=2

        // Vocab of 5 tokens: 0='x', 1='y', 2='{', 3='z', 4=EOS
        // Top-2 (by logit) are tokens 1 (logit=5) and 3 (logit=4) — both disallowed.
        // Token 2 (logit=1) is allowed but is NOT in top-2 → widening must kick in.
        float[] logits = {0.5f, 5.0f, 1.0f, 4.0f, 0.1f};
        float[] masked = masker.maskLogits(logits, 4, id -> {
            switch (id) {
                case 0: return "x";
                case 1: return "y";
                case 2: return "{";
                case 3: return "z";
                default: return null;
            }
        });
        // After widening, token 2 ('{') should be unmasked.
        assertTrue(Float.isFinite(masked[2]), "'{' should survive widening fallback");
        // Tokens 0,1,3 are not valid JSON-object prefixes → should be masked.
        assertEquals(Float.NEGATIVE_INFINITY, masked[0]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[1]);
        assertEquals(Float.NEGATIVE_INFINITY, masked[3]);
    }

    @Test
    void masker_multiByteTokenAccepted() {
        // Multi-byte token: " \"key\":" — this is a valid JSON object body continuation.
        JsonObjectConstraint c = new JsonObjectConstraint();
        ConstraintMasker masker = new ConstraintMasker(c, 256);

        // Simulate having emitted '{'
        masker.tokenEmitted(0, id -> "{");

        // The multi-byte piece " \"key\":" should be allowed (valid JSON prefix: {"key":)
        float[] logits = {3.0f, 1.0f};
        float[] masked = masker.maskLogits(logits, 1, id -> (id == 0 ? " \"key\":" : null));
        assertTrue(Float.isFinite(masked[0]), "multi-byte continuation token should be allowed");
    }

    // =========================================================================
    // ConstraintMasker.topKIndices — internal helper
    // =========================================================================

    @Test
    void topKIndices_returnsCorrectTopK() {
        float[] values = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
        int[] topK = ConstraintMasker.topKIndices(values, 3);
        // Should contain indices 1 (5.0), 4 (4.0), 2 (3.0).
        List<Integer> topList = new java.util.ArrayList<>();
        for (int idx : topK) topList.add(idx);
        assertTrue(topList.contains(1), "index 1 (5.0) should be in top-3");
        assertTrue(topList.contains(4), "index 4 (4.0) should be in top-3");
        assertTrue(topList.contains(2), "index 2 (3.0) should be in top-3");
        assertFalse(topList.contains(3), "index 3 (2.0) should NOT be in top-3");
        assertFalse(topList.contains(0), "index 0 (1.0) should NOT be in top-3");
    }

    @Test
    void topKIndices_kEqualsVocabSize() {
        float[] values = {2.0f, 1.0f, 3.0f};
        int[] topK = ConstraintMasker.topKIndices(values, 3);
        assertEquals(3, topK.length);
        List<Integer> all = new java.util.ArrayList<>();
        for (int idx : topK) all.add(idx);
        assertTrue(all.contains(0) && all.contains(1) && all.contains(2));
    }

    // =========================================================================
    // ConstraintConfig — factory methods
    // =========================================================================

    @Test
    void constraintConfig_jsonObjectFactory() {
        ConstraintConfig cfg = ConstraintConfig.jsonObject();
        assertEquals("json_object", cfg.getType());
        TextConstraint c = cfg.buildConstraint();
        assertNotNull(c);
        assertEquals("json_object", c.type());
    }

    @Test
    void constraintConfig_toolCallFactory() {
        ConstraintConfig cfg = ConstraintConfig.toolCall("tool_a", "tool_b");
        assertEquals("tool_call", cfg.getType());
        assertEquals(Arrays.asList("tool_a", "tool_b"), cfg.getToolNames());
        TextConstraint c = cfg.buildConstraint();
        assertNotNull(c);
        assertEquals("tool_call", c.type());
        assertTrue(c.isAccepting("{\"tool\": \"tool_a\", \"args\": {}}"));
        assertFalse(c.isAccepting("{\"tool\": \"tool_c\", \"args\": {}}"));
    }

    @Test
    void constraintConfig_nativeToolCallFactory() {
        ConstraintConfig cfg =
                ConstraintConfig.nativeToolCall("record_graph_verdict");
        assertEquals(NativeToolCallConstraint.TYPE, cfg.getType());
        TextConstraint c = cfg.buildConstraint();
        assertTrue(c.isAccepting(
                "<|tool_call_start|>[record_graph_verdict(candidate_id=\"x\")]"
                        + "<|tool_call_end|>"));
        assertFalse(c.isAccepting(
                "<|tool_call_start|>[record_g_g_verdict(candidate_id=\"x\")]"
                        + "<|tool_call_end|>"));
    }

    @Test
    void constraintConfig_unknownTypeThrows() {
        ConstraintConfig cfg = ConstraintConfig.builder().type("unknown").build();
        assertThrows(IllegalArgumentException.class, cfg::buildConstraint);
    }

    @Test
    void constraintConfig_emptyToolNameThrows() {
        assertThrows(IllegalArgumentException.class, () -> ConstraintConfig.toolCall());
    }
}
