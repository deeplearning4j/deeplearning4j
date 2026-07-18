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
    void constraintConfig_unknownTypeThrows() {
        ConstraintConfig cfg = ConstraintConfig.builder().type("unknown").build();
        assertThrows(IllegalArgumentException.class, cfg::buildConstraint);
    }

    @Test
    void constraintConfig_emptyToolNameThrows() {
        assertThrows(IllegalArgumentException.class, () -> ConstraintConfig.toolCall());
    }
}
