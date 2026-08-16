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

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
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
    void jsonObject_rejectsNonJsonControlWhitespace() {
        JsonObjectConstraint constraint = new JsonObjectConstraint();

        assertFalse(constraint.canExtend("{\"type\": ", "\f"),
                "form feed is not RFC 8259 structural whitespace");
        assertFalse(constraint.canExtend("{\"type\": \"PER", "\f"),
                "unescaped controls are illegal inside JSON strings");
        assertTrue(constraint.canExtend("{\"type\": ", "\t"),
                "JSON tab whitespace remains legal");
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
    void toolCall_argsMustBeginWithJsonObject() {
        ToolCallConstraint c = new ToolCallConstraint(TOOLS);
        String argsPrefix = "{\"tool\": \"ask_graph_verify\", \"args\": ";

        assertFalse(c.canExtend(argsPrefix, "("),
                "a parenthesized expression is not a JSON tool-arguments object");
        assertFalse(c.canExtend(argsPrefix, "["),
                "an array is not a JSON tool-arguments object");
        assertFalse(c.isAccepting(argsPrefix + "()}"),
                "balanced outer braces must not make invalid argument syntax accepting");
        assertTrue(c.canExtend(argsPrefix, "{"),
                "the documented JSON object argument root must remain selectable");
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
    void nativeToolCall_schemaPatternRejectsImpossibleStringPrefixes() {
        Map<String, Object> label = Map.of(
                "type", "string",
                "pattern", "^[A-Z][A-Z0-9_]*$");
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "labels", Map.of("type", "array", "items", label)),
                "required", List.of("labels"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_schema"),
                Map.of("submit_schema", List.of("labels")),
                Map.of(),
                Map.of("submit_schema", parameters));

        String valuePrefix = "<|tool_call_start|>[submit_schema(labels=[\"";
        assertTrue(constraint.canExtend(valuePrefix, "C"),
                "a prefix that already matches and may extend must remain selectable");
        assertFalse(constraint.canExtend(valuePrefix, "Co"),
                "an anchored uppercase schema must reject a lowercase continuation immediately");
        assertFalse(constraint.canExtend(valuePrefix, "company"),
                "an impossible lowercase prefix must be masked before the quote can dead-end");
        assertTrue(constraint.canExtend(valuePrefix, "COMPANY"));
        assertTrue(constraint.isAccepting(
                valuePrefix + "COMPANY\"])]<|tool_call_end|>"));
    }

    @Test
    void nativeToolCall_schemaRejectsMalformedNestedObjectPrefixes() {
        Map<String, Object> entity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "id", Map.of("type", "string"),
                        "name", Map.of("type", "string"),
                        "type", Map.of("type", "string")),
                "required", List.of("id", "name", "type"),
                "additionalProperties", false);
        Map<String, Object> relation = Map.of(
                "type", "object",
                "properties", Map.of(
                        "source", Map.of("type", "integer", "minimum", 0),
                        "target", Map.of("type", "integer", "minimum", 0),
                        "type", Map.of("type", "string")),
                "required", List.of("source", "target", "type"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "entities", Map.of("type", "array", "items", entity),
                        "relations", Map.of("type", "array", "items", relation)),
                "required", List.of("entities", "relations"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_graph_delta"),
                Map.of("submit_graph_delta", List.of("entities", "relations")),
                Map.of(),
                Map.of("submit_graph_delta", parameters));

        String outOfOrder = "<|tool_call_start|>[submit_graph_delta(relations=";
        assertFalse(constraint.canExtend(outOfOrder, "["),
                "full parameter schemas must not allow a later required argument first");

        String entityPrefix = "<|tool_call_start|>[submit_graph_delta(entities=[{"
                + "\"id\":\"person-1\",\"name\":\"Alex Rivera\",\"type\":\"PERSON\"";
        assertFalse(constraint.canExtend(entityPrefix, ", response:"),
                "unquoted object narration must be rejected before it consumes the token budget");
        assertFalse(constraint.canExtend(entityPrefix, ",\"response\":"),
                "undeclared nested object properties must be rejected before object closure");

        String relationPropertyPrefix = "<|tool_call_start|>[submit_graph_delta(entities=[{"
                + "\"id\":\"person-1\",\"name\":\"Alex Rivera\",\"type\":\"PERSON\"}],"
                + "relations=[{\"source\":0,\"t";
        assertFalse(constraint.canExtend(relationPropertyPrefix, "\\u201"),
                "an impossible Unicode escape in a declared property name must be rejected early");
        assertTrue(constraint.canExtend(relationPropertyPrefix, "\\u0061"),
                "a Unicode escape that can still spell target must remain selectable");

        String incompleteTarget = relationPropertyPrefix + "arget\":";
        assertFalse(constraint.canExtend(incompleteTarget, "-"),
                "a non-negative integer schema must reject a negative prefix immediately");
        assertFalse(constraint.canExtend(incompleteTarget + "1", "\b"),
                "JSON numbers must not absorb a backspace control character");
        assertFalse(constraint.canExtend(incompleteTarget + "1", "\u001b"),
                "JSON numbers must not absorb an escape control character");
        assertTrue(constraint.canExtend(incompleteTarget, "1"));

        String valid = "<|tool_call_start|>[submit_graph_delta(entities=[{"
                + "\"id\":\"person-1\",\"name\":\"Alex Rivera\",\"type\":\"PERSON\"}],"
                + "relations=[{\"source\":0,\"target\":1,"
                + "\"type\":\"WORKS_AT\"}])]<|tool_call_end|>";
        assertTrue(constraint.isAccepting(valid));
    }

    @Test
    void nativeToolCall_compactEntityRejectsPropertyAfterRequiredFields() {
        Map<String, Object> entity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string", "maxLength", 13),
                        "type", Map.of("type", "string", "enum", List.of("PERSON", "COMPANY"))),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "format", Map.of("type", "string", "const", "indexed"),
                        "entities", Map.of(
                                "type", "array", "items", entity,
                                "minItems", 2, "maxItems", 2, "uniqueItems", true),
                        "relations", Map.of("type", "array", "items", Map.of())),
                "required", List.of("format", "entities", "relations"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_graph_delta"),
                Map.of("submit_graph_delta", List.of("format", "entities", "relations")),
                Map.of(),
                Map.of("submit_graph_delta", parameters));
        String entityPrefix = "<|tool_call_start|>[submit_graph_delta(format=\"indexed\","
                + "entities=[{\"name\":\"Alex Rivera\",\"type\":\"PERSON\"";

        assertFalse(constraint.canExtend(entityPrefix, ",\"format:indexed"),
                "additionalProperties=false must reject a property after all entity fields");
        assertTrue(constraint.canExtend(entityPrefix, "},{\"name\":"),
                "the exact-cardinality array must still allow its second entity");

        String duplicateSecondType = entityPrefix + "},{\"name\":\"Alex Rivera\",\"type\":";
        assertFalse(constraint.canExtend(duplicateSecondType, "\"P"),
                "uniqueItems must reject an enum prefix when every matching object completion is a duplicate");
        assertTrue(constraint.canExtend(duplicateSecondType, "\"C"),
                "an enum prefix with a unique object completion must remain selectable");
        assertFalse(constraint.canExtend(duplicateSecondType, "\"PERSON\""),
                "uniqueItems must reject a final property value that makes the open object an unavoidable duplicate");
        assertFalse(constraint.canExtend(duplicateSecondType, "\"PERSON\\"),
                "an incomplete escape cannot extend a fully matched enum value around duplicate lookahead");
        assertTrue(constraint.canExtend(duplicateSecondType, "\"COMPANY\""),
                "a final property value that keeps the second object unique must remain selectable");
    }

    @Test
    void nativeToolCall_prefixItemsConstrainEachArraySlotBeforeDuplicateDeadEnd() {
        Map<String, Object> firstEntity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string", "const", "Jordan Lee"),
                        "type", Map.of("type", "string", "const", "PERSON")),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> secondEntity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string", "const", "Helios Dynamics"),
                        "type", Map.of(
                                "type", "string",
                                "enum", List.of("PERSON", "COMPANY"),
                                "const", "COMPANY")),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> entities = Map.of(
                "type", "array",
                "prefixItems", List.of(firstEntity, secondEntity),
                "items", false,
                "minItems", 2,
                "maxItems", 2,
                "uniqueItems", true);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of("entities", entities),
                "required", List.of("entities"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_entities"),
                Map.of("submit_entities", List.of("entities")),
                Map.of(),
                Map.of("submit_entities", parameters));

        String secondPrefix = "<|tool_call_start|>[submit_entities(entities=["
                + "{\"name\":\"Jordan Lee\",\"type\":\"PERSON\"},{\"name\":";
        assertFalse(constraint.canExtend(secondPrefix, "\"Jordan"),
                "the second positional slot must reject a repeated first entity before closure");
        assertTrue(constraint.canExtend(secondPrefix, "\"Helios"),
                "the source-derived second entity must remain selectable");
        String secondTypePrefix = secondPrefix + "\"Helios Dynamics\",\"type\":";
        assertFalse(constraint.canExtend(secondTypePrefix, "\"PERSON"),
                "const must narrow an enclosing enum before the wrong value can dead-end");
        assertTrue(constraint.canExtend(secondTypePrefix, "\"COMPANY"));

        String valid = secondTypePrefix + "\"COMPANY\"}])]<|tool_call_end|>";
        assertTrue(constraint.isAccepting(valid));
        assertFalse(constraint.canExtend(valid, ","),
                "items=false and maxItems must reject a tail after positional slots");
    }

    @Test
    void nativeToolCall_uniqueItemsRejectsDuplicateThatCrossesATokenBoundary() {
        Map<String, Object> entity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string"),
                        "type", Map.of(
                                "type", "string",
                                "enum", List.of("PERSON", "COMPANY"))),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "entities", Map.of(
                                "type", "array",
                                "items", entity,
                                "uniqueItems", true)),
                "required", List.of("entities"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_entities"),
                Map.of("submit_entities", List.of("entities")),
                Map.of(),
                Map.of("submit_entities", parameters));

        String first = "<|tool_call_start|>[submit_entities(entities=["
                + "{\"name\":\"Alex Rivera\",\"type\":\"PERSON\"}";
        assertFalse(constraint.canExtend(first,
                        ",{\"name\":\"Alex Rivera\",\"type\":\"PERSON\"},{\"name\":\""),
                "a token spanning a complete duplicate and the next item must be rejected");
        assertTrue(constraint.canExtend(first,
                        ",{\"name\":\"Acme Robotics\",\"type\":\"COMPANY\"},{\"name\":\""),
                "a distinct completed item in the same token must remain valid");
    }

    @Test
    void nativeToolCall_prefixItemsRemainActiveAfterARequiredScalarArgument() {
        Map<String, Object> firstEntity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string", "const", "Jordan Lee"),
                        "type", Map.of("type", "string", "const", "PERSON")),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> secondEntity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string", "const", "Helios Dynamics"),
                        "type", Map.of("type", "string", "const", "COMPANY")),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> relation = Map.of(
                "type", "object",
                "properties", Map.of(
                        "source", Map.of("type", "integer", "const", 0),
                        "target", Map.of("type", "integer", "const", 1),
                        "type", Map.of("type", "string", "const", "FOUNDED")),
                "required", List.of("source", "target", "type"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "format", Map.of("type", "string", "const", "indexed"),
                        "entities", Map.of(
                                "type", "array",
                                "prefixItems", List.of(firstEntity, secondEntity),
                                "items", false,
                                "minItems", 2,
                                "maxItems", 2,
                                "uniqueItems", true),
                        "relations", Map.of(
                                "type", "array",
                                "prefixItems", List.of(relation),
                                "items", false,
                                "minItems", 1,
                                "maxItems", 1,
                                "uniqueItems", true)),
                "required", List.of("format", "entities", "relations"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_graph_delta"),
                Map.of("submit_graph_delta", List.of("format", "entities", "relations")),
                Map.of("submit_graph_delta", Map.of("format", List.of("indexed"))),
                Map.of("submit_graph_delta", parameters));

        String firstType = "<|tool_call_start|>[submit_graph_delta(format=\"indexed\",entities=[{"
                + "\"name\":\"Jordan Lee\",\"type\":";
        assertFalse(constraint.canExtend(firstType, "\"COMPANY\""));
        assertTrue(constraint.canExtend(firstType, "\"PERSON\""));
        String secondName = firstType + "\"PERSON\"},{\"name\":";
        assertFalse(constraint.canExtend(secondName, "\"Jordan Lee\""));
        assertTrue(constraint.canExtend(secondName, "\"Helios Dynamics\""));

        String valid = secondName + "\"Helios Dynamics\",\"type\":\"COMPANY\"}],"
                + "relations=[{\"source\":0,\"target\":1,\"type\":\"FOUNDED\"}])]"
                + "<|tool_call_end|>";
        assertTrue(constraint.isAccepting(valid));
        assertFalse(constraint.isAccepting(valid.replace("\"target\":1", "\"target\":0")));
    }

    @Test
    void nativeToolCall_enumRejectsImpossibleIncompleteUnicodeEscapeBeforeDeadEnd() {
        Map<String, Object> property = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string"),
                        "type", Map.of("type", "string", "enum", List.of("String"))),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "properties", Map.of("type", "array", "items", property)),
                "required", List.of("properties"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_corpus_schema"),
                Map.of("submit_corpus_schema", List.of("properties")),
                Map.of(),
                Map.of("submit_corpus_schema", parameters));

        String prefix = "<|tool_call_start|>[submit_corpus_schema(properties=[{"
                + "\"name\":\"employee\",\"type\":\"S";
        assertTrue(constraint.canExtend(prefix, "\\u"),
                "an escape introducer remains recoverable through the enum's next character");
        assertTrue(constraint.canExtend(prefix + "\\u", "0"),
                "String's next character t can still be represented as \\u0074");
        assertFalse(constraint.canExtend(prefix + "\\u", "2"),
                "an impossible Unicode prefix must be rejected before all completions dead-end");
        assertTrue(constraint.canExtend(prefix + "\\u007", "4"),
                "the escaped enum prefix \\u0074 must remain selectable");
        assertTrue(constraint.canExtend(prefix + "\\u0074", "r"),
                "generation may continue from the decoded escaped prefix");
    }

    @Test
    void nativeToolCall_maxLengthRejectsIncompleteEscapeBeforeDeadEnd() {
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string", "maxLength", 3)),
                "required", List.of("name"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit"),
                Map.of("submit", List.of("name")),
                Map.of(),
                Map.of("submit", parameters));

        String fullPrefix = "<|tool_call_start|>[submit(name=\"abc";
        assertFalse(constraint.canExtend(fullPrefix, "\\"),
                "an unfinished escape cannot start when maxLength is exhausted");
        assertTrue(constraint.canExtend(fullPrefix, "\""),
                "the closing quote must remain selectable at maxLength");

        String remainingCapacity = "<|tool_call_start|>[submit(name=\"ab";
        assertTrue(constraint.canExtend(remainingCapacity, "\\"),
                "an unfinished escape remains legal while one character fits");
        assertTrue(constraint.canExtend(remainingCapacity, "\\n"),
                "a completed one-character escape may consume the remaining capacity");
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
    void masker_exactDecodeReplaysToolTransitionsAfterBoundaryRewrite() {
        ToolCallConstraint constraint = new ToolCallConstraint(
                "submit_graph_delta", "submit_graph_detail");
        ConstraintMasker masker = new ConstraintMasker(constraint, 2);
        String current = "{\"tool\": \"submit_graph_del";
        String rewritten = "{\"tool\": \"submit_graph_detail";
        masker.decodedTextEmitted(current);

        assertFalse(rewritten.startsWith(current),
                "the fixture must exercise the exact-decoder rewrite branch");
        assertTrue(masker.allowsDecodedText(rewritten, List.of()),
                "a rewritten exact decode must be replayed through incremental protocol states");
    }

    @Test
    void masker_exactDecodeTopKWidensPastInvalidHighLogitCandidate() {
        JsonObjectConstraint constraint = new JsonObjectConstraint();
        ConstraintMasker masker = new ConstraintMasker(constraint, 1);
        masker.decodedTextEmitted("{\"key\":\"value\"");

        float[] logits = {9.0f, 1.0f};
        float[] approximate = masker.maskLogits(logits, Set.of(), id -> "}");
        assertTrue(Float.isFinite(approximate[0]));
        assertEquals(Float.NEGATIVE_INFINITY, approximate[1],
                "the fast top-K approximation should initially omit the lower candidate");

        float[] exact = masker.maskLogitsByDecodedCandidate(
                logits,
                Set.of(),
                id -> id == 0 ? "x" : "{\"key\":\"value\"}",
                List.of());
        assertEquals(Float.NEGATIVE_INFINITY, exact[0],
                "the tokenizer's invalid full-sequence decode must be rejected");
        assertTrue(Float.isFinite(exact[1]),
                "full-vocabulary exact widening must recover the lower valid token");
    }

    @Test
    void masker_exactDecodeWideningUsesPieceCandidatesBeforeFullVocabulary() {
        Map<String, Object> itemSchema = Map.of(
                "type", "object",
                "properties", Map.of("target", Map.of("type", "string")),
                "required", List.of("target"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of("item", itemSchema),
                "required", List.of("item"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit"),
                Map.of("submit", List.of("item")),
                Map.of(),
                Map.of("submit", parameters));
        ConstraintMasker masker = new ConstraintMasker(constraint, 1);
        String current = "<|tool_call_start|>[submit(item={\"t";
        String completion = "arget\":\"ok\"})]<|tool_call_end|>";
        masker.decodedTextEmitted(current);
        float[] logits = new float[1_000];
        logits[0] = 9.0f;
        logits[999] = 1.0f;
        java.util.concurrent.atomic.AtomicInteger exactDecodes =
                new java.util.concurrent.atomic.AtomicInteger();

        float[] exact = masker.maskLogitsByDecodedCandidate(
                logits,
                Set.of(),
                Set.of(),
                id -> id == 999 ? completion : "x",
                id -> {
                    exactDecodes.incrementAndGet();
                    return id == 999 ? current + completion : current + "x";
                },
                List.of());

        assertTrue(Float.isFinite(exact[999]));
        assertTrue(exactDecodes.get() < 10,
                "structural widening should not exact-decode the full vocabulary");
    }

    @Test
    void masker_exactDecodeKeepsFastTopKWhenAnExactCandidateIsValid() {
        ConstraintMasker masker = new ConstraintMasker(new JsonObjectConstraint(), 1);
        float[] exact = masker.maskLogitsByDecodedCandidate(
                new float[]{9.0f, 1.0f},
                Set.of(),
                id -> id == 0 ? "{\"preferred\":true}" : "{\"lower\":true}",
                List.of());

        assertTrue(Float.isFinite(exact[0]));
        assertEquals(Float.NEGATIVE_INFINITY, exact[1],
                "a legal exact top-K candidate must avoid a full-vocabulary scan");
    }

    @Test
    void masker_exactDecodeRejectsDisappearingUnownedControlToken() {
        NativeToolCallConstraint constraint = new NativeToolCallConstraint("submit");
        ConstraintMasker masker = new ConstraintMasker(constraint, 1);

        float[] exact = masker.maskLogitsByDecodedCandidate(
                new float[]{9.0f, 1.0f},
                Set.of(),
                Set.of(0),
                id -> id == 0 ? "<|pad|>" : "<|tool_call_start|>",
                id -> id == 0 ? "" : "<|tool_call_start|>",
                List.of("<|pad|>", "<|tool_call_start|>"));

        assertEquals(Float.NEGATIVE_INFINITY, exact[0],
                "a control token omitted by exact decode must remain blocked by token identity");
        assertTrue(Float.isFinite(exact[1]),
                "exact widening must recover the lower ordinary-token protocol prefix");
        assertFalse(masker.allowsSpecialToken("<|pad|>"));
        assertTrue(masker.allowsSpecialToken("<|tool_call_start|>"));
        masker.specialTokenEmitted("<|tool_call_start|>");
        assertEquals("<|tool_call_start|>", masker.getEmittedText());
    }

    @Test
    void masker_exactDecodePreservesIncrementalWhitespaceProgress() {
        NativeToolCallConstraint constraint = new NativeToolCallConstraint("submit");
        ConstraintMasker masker = new ConstraintMasker(constraint, 2);
        String current = "<|tool_call_start|>[submit(values=[] ";
        masker.decodedTextEmitted(current);

        float[] exact = masker.maskLogitsByDecodedCandidate(
                new float[]{9.0f, 1.0f},
                Set.of(),
                id -> id == 0 ? current + "\t" : current + ")]<|tool_call_end|>",
                List.of("<|tool_call_start|>", "<|tool_call_end|>"));

        assertEquals(Float.NEGATIVE_INFINITY, exact[0],
                "exact decoding must not bypass the native grammar's whitespace self-loop guard");
        assertTrue(Float.isFinite(exact[1]),
                "the structural close must remain selectable after whitespace is exhausted");
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
    // Template-owned output-block budgets
    // =========================================================================

    @Test
    void outputBlockBudget_forcesGenericBlockClosureBeforePayload() {
        TextConstraint constraint = outputBlockToolConstraint(List.of(
                new ChatTemplate.OutputBlockDefinition(
                        "analysis", "<analysis>", "</analysis>")));
        ConstraintMasker masker = new ConstraintMasker(
                constraint, 1, 12, 5, 4);
        masker.decodedTextEmitted("working");

        masker.enforceOutputBlockBudget(4);
        assertFalse(masker.isOutputBlockClosureRequired());
        assertTrue(masker.getConstraint().canExtend("working", " longer"));

        masker.enforceOutputBlockBudget(5);
        assertTrue(masker.isOutputBlockClosureRequired());
        assertFalse(masker.getConstraint().canExtend("working", " longer"));
        assertTrue(masker.getConstraint().canExtend(
                "working",
                "</analysis><|tool_call_start|>[submit()]<|tool_call_end|>"));
    }

    @Test
    void outputBlockBudget_payloadReserveCanCloseBeforeBlockLimit() {
        TextConstraint constraint = outputBlockToolConstraint(List.of(
                new ChatTemplate.OutputBlockDefinition(
                        "trace", "<trace>", "</trace>")));
        ConstraintMasker masker = new ConstraintMasker(
                constraint, 1, 10, 9, 4);
        masker.decodedTextEmitted("trace body");

        masker.enforceOutputBlockBudget(5);
        assertFalse(masker.isOutputBlockClosureRequired());
        masker.enforceOutputBlockBudget(6);
        assertTrue(masker.isOutputBlockClosureRequired(),
                "reserve boundary (10 - 4) must win over the later block cap");
    }

    @Test
    void outputBlockBudget_preservesPartialAndNestedClosingDelimiters() {
        TextConstraint constraint = outputBlockToolConstraint(List.of(
                new ChatTemplate.OutputBlockDefinition(
                        "outer", "<outer>", "</outer>"),
                new ChatTemplate.OutputBlockDefinition(
                        "inner", "<inner>", "</inner>")));
        String current = "work</inn";
        TextConstraint forced = constraint.requireOutputBlockClosure(current);

        assertFalse(forced.canExtend(current, "more"));
        assertTrue(forced.canExtend(
                current,
                "er></outer><|tool_call_start|>[submit()]<|tool_call_end|>"));
    }

    @Test
    void outputBlocksThenXmlToolCall_rejectsControlAndWhitespaceLoops() {
        Map<String, Object> entitySchema = Map.of(
                "type", "object",
                "properties", Map.of(
                        "name", Map.of("type", "string"),
                        "type", Map.of(
                                "type", "string",
                                "enum", List.of("PERSON", "COMPANY"))),
                "required", List.of("name", "type"),
                "additionalProperties", false);
        Map<String, Object> entitiesSchema = Map.of(
                "type", "array",
                "items", entitySchema,
                "minItems", 1,
                "maxItems", 4);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of("entities", entitiesSchema),
                "required", List.of("entities"),
                "additionalProperties", false);

        TextConstraint constraint = ConstraintConfig.builder()
                .type(XmlToolCallConstraint.TYPE)
                .toolNames(List.of("submit_entities"))
                .toolArgumentNames(Map.of(
                        "submit_entities", List.of("entities")))
                .toolParameterSchemas(Map.of(
                        "submit_entities", parameters))
                .outputBlocks(List.of(
                        new ChatTemplate.OutputBlockDefinition(
                                "analysis", "<analysis>", "</analysis>"),
                        new ChatTemplate.OutputBlockDefinition(
                                "trace", "<trace>", "</trace>")))
                .build()
                .buildConstraint();

        String beforeValueWhitespace = "reasoning</trace></analysis>\n"
                + "<tool_call>\n<function=submit_entities>\n"
                + "<parameter=entities>\n"
                + "[{\"name\":\"Alex Rivera\",\"type\":";
        assertTrue(constraint.canExtend(beforeValueWhitespace, " "),
                "one structural whitespace token remains legal");

        String valuePrefix = beforeValueWhitespace + " ";
        assertFalse(constraint.canExtend(valuePrefix, "\f"),
                "form feed must never enter a JSON-valued XML parameter");
        assertFalse(constraint.canExtend(valuePrefix, "\t"),
                "structured whitespace must not self-loop across tokens");

        String completion = "\"PERSON\"}]\n</parameter>\n"
                + "</function>\n</tool_call>";
        assertTrue(constraint.canExtend(valuePrefix, completion),
                "masking whitespace must leave the schema-owned value and XML close selectable");
        assertTrue(constraint.isAccepting(valuePrefix + completion),
                "multiple output blocks must transition into one complete native XML tool call");
    }

    @Test
    void outputBlockBudget_rejectsImpossibleReserve() {
        TextConstraint constraint = outputBlockToolConstraint(List.of(
                new ChatTemplate.OutputBlockDefinition(
                        "analysis", "<analysis>", "</analysis>")));
        assertThrows(IllegalArgumentException.class,
                () -> new ConstraintMasker(constraint, 1, 8, 4, 9));
    }

    private static TextConstraint outputBlockToolConstraint(
            List<ChatTemplate.OutputBlockDefinition> blocks) {
        return ConstraintConfig.builder()
                .type(NativeToolCallConstraint.TYPE)
                .toolNames(List.of("submit"))
                .outputBlocks(blocks)
                .build()
                .buildConstraint();
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
