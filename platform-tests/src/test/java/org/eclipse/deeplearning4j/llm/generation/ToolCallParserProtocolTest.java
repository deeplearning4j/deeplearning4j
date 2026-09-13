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
package org.eclipse.deeplearning4j.llm.generation;

import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintConfig;
import org.eclipse.deeplearning4j.llm.generation.constraint.ConstraintMasker;
import org.eclipse.deeplearning4j.llm.generation.constraint.NativeToolCallConstraint;
import org.eclipse.deeplearning4j.llm.generation.constraint.TextConstraint;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ToolCallParserProtocolTest {
    private static final List<ChatTemplate.Tool> ENTITY_TOOLS = List.of(
            ChatTemplate.Tool.function(
                    "submit_entities",
                    "Submit extracted entities",
                    Map.of(
                            "type", "object",
                            "properties", Map.of("names", Map.of(
                                    "type", "array",
                                    "items", Map.of("type", "string"))),
                            "required", List.of("names"))));

    private static String xmlCall(String name) {
        return "<tool_call>\n<function=submit_entities>\n"
                + "<parameter=names>\n[\"" + name + "\"]\n</parameter>\n"
                + "</function>\n</tool_call>";
    }

    private static TextConstraint xmlConstraint() {
        return ConstraintConfig.xmlToolCall(
                Map.of("submit_entities", List.of("names")), Map.of(),
                Map.of("submit_entities", ENTITY_TOOLS.get(0).getParameters()))
                .buildConstraint();
    }

    @Test
    void xmlMultipleEnvelopesParseInOrderIncludingRepeatedCalls() {
        String first = xmlCall("Alice");
        String second = xmlCall("Bob");
        for (String separator : List.of("", "\n", " \t\r\n")) {
            ToolCallParser.ParseResult result = ToolCallParser.parse(
                    "Found entities.\n" + first + separator + second + separator + first,
                    ENTITY_TOOLS, ChatTemplate.ToolCallFormat.XML);
            assertTrue(result.isClean(), result.getErrors().toString());
            assertEquals("Found entities.", result.getContent());
            assertEquals(List.of(List.of("Alice"), List.of("Bob"), List.of("Alice")),
                    result.getToolCalls().stream().map(c -> c.getArguments().get("names"))
                            .collect(Collectors.toList()));
        }
    }

    @Test
    void xmlAcceptingCallAllowsContinuationAndSingleCallEos() {
        TextConstraint constraint = xmlConstraint();
        String first = xmlCall("Alice");
        assertTrue(constraint.isAccepting(first));
        ConstraintMasker masker = new ConstraintMasker(constraint, 2);
        masker.decodedTextEmitted(first);
        assertEquals(2f, masker.maskLogits(new float[]{2f, 1f}, 0,
                id -> id == 0 ? "</s>" : "<tool_call>")[0]);
        assertTrue(constraint.allowsSpecialToken(first, "<tool_call>"));
        for (String separator : List.of("", "\n", " \t\r\n")) {
            String extension = separator + xmlCall("Bob") + separator + xmlCall("Carol");
            for (int i = 0; i < extension.length(); i++) {
                assertTrue(constraint.canExtend(first + extension.substring(0, i),
                        extension.substring(i, i + 1)), "continuation offset " + i);
            }
            assertTrue(constraint.canExtend(first, extension));
            assertTrue(constraint.isAccepting(first + extension));
        }
        ToolCallParser.ParseResult single = ToolCallParser.parse(
                first, ENTITY_TOOLS, ChatTemplate.ToolCallFormat.XML);
        assertTrue(single.isClean());
        assertEquals(1, single.getToolCalls().size());
        assertEquals(List.of("Alice"), single.getToolCalls().get(0).getArguments().get("names"));
    }

    @Test
    void xmlIncompleteSecondCallDeniesEosAndExecutesNothing() {
        TextConstraint constraint = xmlConstraint();
        String first = xmlCall("Alice");
        String second = xmlCall("Bob");
        for (int i = 1; i < second.length(); i++) {
            String prefix = first + "\n" + second.substring(0, i);
            assertFalse(constraint.isAccepting(prefix), "second call offset " + i);
        }
        String prefix = first + "\n<tool_call>\n";
        ConstraintMasker masker = new ConstraintMasker(constraint, 2);
        masker.decodedTextEmitted(prefix);
        assertEquals(Float.NEGATIVE_INFINITY, masker.maskLogits(new float[]{2f, 1f}, 0,
                id -> id == 0 ? "</s>" : "<function=submit_entities>\n")[0]);
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                prefix, ENTITY_TOOLS, ChatTemplate.ToolCallFormat.XML);
        assertFalse(result.isClean());
        assertTrue(result.getToolCalls().isEmpty());
    }

    @Test
    void xmlLaterCallUsesItsOwnToolSchema() {
        Map<String, Object> textSchema = Map.of("type", "object",
                "properties", Map.of("names", Map.of("type", "string")),
                "required", List.of("names"));
        TextConstraint constraint = ConstraintConfig.xmlToolCall(
                Map.of("submit_text", List.of("names"), "submit_entities", List.of("names")),
                Map.of(), Map.of("submit_text", textSchema,
                        "submit_entities", ENTITY_TOOLS.get(0).getParameters())).buildConstraint();
        String first = xmlCall("Alice").replace("submit_entities", "submit_text")
                .replace("[\"Alice\"]", "Alice");
        String second = xmlCall("Bob");
        assertTrue(constraint.isAccepting(first + "\n" + second));
        String openArray = first + "\n<tool_call>\n<function=submit_entities>\n"
                + "<parameter=names>\n[ ";
        assertTrue(constraint.canExtend(openArray, "\"Bob\"]\n</parameter>\n</function>\n</tool_call>"));
        assertFalse(constraint.canExtend(openArray, "\t"),
                "later structured parameters retain the existing whitespace-loop protection");
        ToolCallParser.ParseResult result = ToolCallParser.parse(first + "\n" + second,
                List.of(ChatTemplate.Tool.function("submit_text", "", textSchema), ENTITY_TOOLS.get(0)),
                ChatTemplate.ToolCallFormat.XML);
        assertTrue(result.isClean(), result.getErrors().toString());
        assertEquals(List.of("submit_text", "submit_entities"), result.getToolCalls().stream()
                .map(ChatTemplate.ToolCall::getName).collect(Collectors.toList()));
    }

    @Test
    void xmlMalformedTrailingDataRejectsWholeCallSequence() {
        String first = xmlCall("Alice");
        for (String suffix : List.of("garbage", "\n<wrong>", "\n</tool_call>",
                "\n" + xmlCall("Bob") + "garbage",
                "\n" + xmlCall("Bob").replace("submit_entities", "undeclared"),
                "\n" + xmlCall("Bob").replace("[\"Bob\"]", "42"))) {
            assertFalse(xmlConstraint().canExtend(first, suffix), suffix);
            assertFalse(xmlConstraint().isAccepting(first + suffix), suffix);
            ToolCallParser.ParseResult result = ToolCallParser.parse(
                    first + suffix, ENTITY_TOOLS, ChatTemplate.ToolCallFormat.XML);
            assertFalse(result.isClean(), suffix);
            assertTrue(result.getToolCalls().isEmpty(), suffix);
        }
    }

    @Test
    void nativeConstraintRejectsMalformedGraphObjectBeforeParsing() {
        Map<String, Object> entity = Map.of(
                "type", "object",
                "properties", Map.of(
                        "id", Map.of("type", "string"),
                        "name", Map.of("type", "string"),
                        "type", Map.of("type", "string")),
                "required", List.of("id", "name", "type"),
                "additionalProperties", false);
        Map<String, Object> parameters = Map.of(
                "type", "object",
                "properties", Map.of(
                        "entities", Map.of("type", "array", "items", entity),
                        "relations", Map.of("type", "array", "items", Map.of(
                                "type", "object",
                                "properties", Map.of(),
                                "additionalProperties", false))),
                "required", List.of("entities", "relations"),
                "additionalProperties", false);
        NativeToolCallConstraint constraint = new NativeToolCallConstraint(
                List.of("submit_graph_delta"),
                Map.of("submit_graph_delta", List.of("entities", "relations")),
                Map.of(),
                Map.of("submit_graph_delta", parameters));
        String prefix = "<|tool_call_start|>[submit_graph_delta(entities=[{"
                + "\"id\":\"person-1\",\"name\":\"Alex Rivera\",\"type\":\"PERSON\"";

        assertFalse(constraint.canExtend(prefix, ", response:"));
        assertFalse(constraint.canExtend(prefix, ",\"response\":"));
    }

    @Test
    void truncatedNativeEnvelopesCannotBecomeExecutableCalls() {
        List<String> malformed = List.of(
                "<|tool_call_start|>[submit_entities(names=\"submit_entities()"
                        + "<|endoftext|><|startoftext|>",
                "<|tool_call_start|>[submit_entities( names=\"submit_entities. "
                        + "I will now submit the distinct entities...");

        for (String raw : malformed) {
            ToolCallParser.ParseResult result = ToolCallParser.parse(
                    raw, ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
            assertTrue(result.getToolCalls().isEmpty(), raw);
            assertEquals(List.of("incomplete native tool-call envelope"),
                    result.getErrors(), raw);
        }
    }

    @Test
    void completeNativeEnvelopeStillParses() {
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                "<|tool_call_start|>[submit_entities(names=[\"M. Chen\",\"J. Park\"])]"
                        + "<|tool_call_end|>",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);

        assertTrue(result.getErrors().isEmpty());
        assertEquals(1, result.getToolCalls().size());
        assertEquals(List.of("M. Chen", "J. Park"),
                result.getToolCalls().get(0).getArguments().get("names"));
    }

    @Test
    void nativeModeDoesNotAcceptJsonOrBarePythonFallbacks() {
        List<String> wrongProtocols = List.of(
                "{\"tool\":\"submit_entities\",\"args\":{\"names\":[\"M. Chen\"]}}",
                "submit_entities(names=[\"M. Chen\"])",
                "<|python_tag|>submit_entities(names=[\"M. Chen\"])");

        for (String raw : wrongProtocols) {
            ToolCallParser.ParseResult result = ToolCallParser.parse(
                    raw, ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
            assertTrue(result.getToolCalls().isEmpty(), raw);
        }
    }

    @Test
    void importedTemplateOwnsToolAndReasoningProtocols() {
        ChatTemplate lfm = new ChatTemplate(
                "{{ messages }}<|tool_call_start|><|tool_call_end|>",
                "", "<|endoftext|>");
        ChatTemplate qwenThinking = new ChatTemplate(
                "{% if enable_thinking %}<think>{{ content }}</think>{% endif %}",
                "", "<|im_end|>");

        assertEquals(ChatTemplate.ToolCallFormat.NATIVE, lfm.toolCallFormat());
        assertEquals(ChatTemplate.ToolCallFormat.JSON, qwenThinking.toolCallFormat());

        ChatTemplate.AssistantOutput parsed = qwenThinking.parseAssistantOutput(
                "<think>inspect graph evidence</think>Use Alice.<|im_end|>");
        assertEquals("inspect graph evidence", parsed.getReasoningContent());
        assertEquals("Use Alice.", parsed.getContent());
        assertTrue(parsed.getErrors().isEmpty());

        ChatTemplate plain = new ChatTemplate("{{ messages }}", "", "</s>");
        ChatTemplate.AssistantOutput untouched = plain.parseAssistantOutput(
                "<think>literal text</think>answer</s>");
        assertEquals("<think>literal text</think>answer", untouched.getContent());
        assertEquals("", untouched.getReasoningContent());
    }

    @Test
    void templateCanDeclareMultipleNestedOutputBlockKinds() {
        ChatTemplate template = new ChatTemplate(
                "{{ messages }}{% if add_generation_prompt %}<analysis><think>{% endif %}"
                        + "</think></analysis><tool_call></tool_call>",
                "", "<|im_end|>");
        List<ChatTemplate.OutputBlockDefinition> definitions =
                template.prefilledOutputBlocks("messages", "messages<analysis><think>\n");

        assertEquals(List.of("analysis", "think"), definitions.stream()
                .map(ChatTemplate.OutputBlockDefinition::getType)
                .collect(Collectors.toList()));

        String raw = "inspect graph evidence</think>cite source</analysis>"
                + "<tool_call>\n<function=submit_entities>\n"
                + "<parameter=names>\n[\"Alice\"]\n</parameter>\n"
                + "</function>\n</tool_call><|im_end|>";
        ChatTemplate.AssistantOutput parsed =
                template.parseAssistantOutput(raw, definitions);

        assertEquals(List.of("think", "analysis"), parsed.getOutputBlocks().stream()
                .map(ChatTemplate.OutputBlock::getType)
                .collect(Collectors.toList()));
        assertEquals(List.of("inspect graph evidence", "cite source"),
                parsed.getOutputBlocks().stream()
                        .map(ChatTemplate.OutputBlock::getContent)
                        .collect(Collectors.toList()));
        assertEquals("inspect graph evidence", parsed.getReasoningContent());
        assertTrue(parsed.getContent().startsWith("<tool_call>"));
        assertTrue(parsed.getErrors().isEmpty());

        ChatGenerationResult result = new ChatGenerationResult(
                raw, parsed, ENTITY_TOOLS, ChatTemplate.ToolCallFormat.XML,
                ChatTemplate.ToolChoice.REQUIRED);
        assertTrue(result.isParsedCleanly());
        assertEquals(2, result.getOutputBlocks().size());
        assertEquals(List.of("Alice"),
                result.getToolCalls().get(0).getArguments().get("names"));
    }

    @Test
    void requiredXmlConstraintAllowsMultipleOutputBlocksBeforePayload() {
        List<ChatTemplate.OutputBlockDefinition> definitions = List.of(
                new ChatTemplate.OutputBlockDefinition(
                        "analysis", "<analysis>", "</analysis>"),
                new ChatTemplate.OutputBlockDefinition(
                        "think", "<think>", "</think>"));
        TextConstraint constraint = ConstraintConfig.xmlToolCall(
                        Map.of("submit_entities", List.of("names")),
                        Map.of(),
                        Map.of("submit_entities", ENTITY_TOOLS.get(0).getParameters()))
                .toBuilder()
                .outputBlocks(definitions)
                .build()
                .buildConstraint();

        assertTrue(constraint.canExtend("", "inspect graph evidence"));
        assertTrue(constraint.canExtend("inspect graph evidence", "</think>"));
        String blockPrefix = "inspect graph evidence</think>cite source</analysis>";
        assertTrue(constraint.canExtend("", blockPrefix));
        assertFalse(constraint.isAccepting(blockPrefix));
        assertFalse(constraint.canExtend(blockPrefix, "<not_a_tool>"));

        String payload = "<tool_call>\n<function=submit_entities>\n"
                + "<parameter=names>\n[\"Alice\"]\n</parameter>\n"
                + "</function>\n</tool_call>";
        String complete = blockPrefix + "\n" + payload;
        assertTrue(constraint.canExtend("", complete));
        assertTrue(constraint.isAccepting(complete));
        assertEquals("output_blocks_then_xml_tool_call", constraint.type());
    }

    @Test
    void requiredChoiceFailsAnyMissingOrInvalidConfiguredProtocol() {
        ToolCallParser.ParseResult json = ToolCallParser.parse(
                "I might call submit_entities later", ENTITY_TOOLS,
                ChatTemplate.ToolCallFormat.JSON, ChatTemplate.ToolChoice.REQUIRED);
        ToolCallParser.ParseResult nativeCall = ToolCallParser.parse(
                "{\"tool\":\"submit_entities\",\"args\":{\"names\":[\"M. Chen\"]}}",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE,
                ChatTemplate.ToolChoice.REQUIRED);

        assertFalse(json.getErrors().isEmpty());
        assertFalse(nativeCall.getErrors().isEmpty());
        assertTrue(json.getToolCalls().isEmpty());
        assertTrue(nativeCall.getToolCalls().isEmpty());
    }
}
