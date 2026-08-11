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

import org.eclipse.deeplearning4j.llm.generation.constraint.NativeToolCallConstraint;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolCallParserTest {
    private static final List<ChatTemplate.Tool> TOOLS = List.of(
            ChatTemplate.Tool.function(
                    "search_graph",
                    "Search the graph",
                    Map.of(
                            "type", "object",
                            "properties", Map.of("query", Map.of(
                                    "type", "string",
                                    "enum", List.of("M. Chen"))),
                            "required", List.of("query"))));
    private static final List<ChatTemplate.Tool> ENTITY_TOOLS = List.of(
            ChatTemplate.Tool.function(
                    "submit_entities",
                    "Submit extracted entities",
                    Map.of(
                            "type", "object",
                            "properties", Map.of("names", Map.of(
                                    "type", "array",
                                    "minItems", 1,
                                    "maxItems", 4,
                                    "uniqueItems", true,
                                    "items", Map.of(
                                            "type", "string",
                                            "maxLength", 32))),
                            "required", List.of("names"))));
    private static final List<ChatTemplate.Tool> UNBOUNDED_QUERY_TOOLS = List.of(
            ChatTemplate.Tool.function(
                    "search_graph",
                    "Search the graph",
                    Map.of(
                            "type", "object",
                            "properties", Map.of("query", Map.of("type", "string")),
                            "required", List.of("query"))));

    @Test
    void parsesJsonEnvelopeAndRetainsContent() {
        String raw = "{\"tool_calls\":[{\"id\":\"call_1\",\"function\":"
                + "{\"name\":\"search_graph\",\"arguments\":{\"query\":\"M. Chen\"}}}]}";
        ToolCallParser.ParseResult result = ToolCallParser.parse(raw, TOOLS);

        assertEquals(1, result.getToolCalls().size());
        assertEquals("call_1", result.getToolCalls().get(0).getId());
        assertEquals("search_graph", result.getToolCalls().get(0).getName());
        assertEquals("M. Chen", result.getToolCalls().get(0).getArguments().get("query"));
        assertEquals("", result.getContent());
        assertTrue(result.getErrors().isEmpty());
    }

    @Test
    void requiredJsonModeRejectsMissingCanonicalEnvelopeWithoutTokenSniffing() {
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                "search_graph(query='M. Chen')", TOOLS,
                ChatTemplate.ToolCallFormat.JSON, ChatTemplate.ToolChoice.REQUIRED);

        assertTrue(result.getToolCalls().isEmpty());
        assertFalse(result.getErrors().isEmpty());
    }

    @Test
    void decodesEscapedStringsInLfmNativeArguments() {
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                "<|tool_call_start|>[search_graph("
                        + "query=\"The status is \\\"Do not use\\\".\")]<|tool_call_end|>",
                UNBOUNDED_QUERY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);

        assertEquals(1, result.getToolCalls().size());
        assertEquals("The status is \"Do not use\".",
                result.getToolCalls().get(0).getArguments().get("query"));
        assertTrue(result.getErrors().isEmpty());
    }

    @Test
    void ignoresSchemaDeclarationsAndReportsUndeclaredCalls() {
        ToolCallParser.ParseResult declaration = ToolCallParser.parse(
                "function search_graph(query: string) -> object", TOOLS);
        assertTrue(declaration.getToolCalls().isEmpty());

        ToolCallParser.ParseResult undeclared = ToolCallParser.parse(
                "{\"tool\":\"delete_everything\",\"args\":{}}", TOOLS);
        assertTrue(undeclared.getToolCalls().isEmpty());
        assertFalse(undeclared.getErrors().isEmpty());
    }

    @Test
    void rejectsMalformedJsonEnvelopeWithoutInventingCall() {
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                "{\"tool\":\"search_graph\",\"args\":", TOOLS);
        assertTrue(result.getToolCalls().isEmpty());
        assertFalse(result.getErrors().isEmpty());
    }

    @Test
    void reportsMalformedNativeEnvelopeInsteadOfSilentlyIgnoringIt() {
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                "<|tool_call_start|>[search_g_g_graph(query=\"M. Chen\")]"
                        + "<|tool_call_end|>",
                TOOLS, ChatTemplate.ToolCallFormat.NATIVE);

        assertTrue(result.getToolCalls().isEmpty());
        assertEquals(List.of("undeclared tool search_g_g_graph"),
                result.getErrors());
    }

    @Test
    void rejectsTruncatedNativeCallsWithoutRecoveringInnerPythonFragments() {
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
    void rejectsClosedNativeEnvelopeWithUnterminatedArgumentString() {
        ToolCallParser.ParseResult result = ToolCallParser.parse(
                "<|tool_call_start|>[submit_entities(names=\"truncated)]"
                        + "<|tool_call_end|>",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);

        assertTrue(result.getToolCalls().isEmpty());
        assertEquals(List.of("invalid arguments for tool submit_entities"),
                result.getErrors());
    }

    @Test
    void configuredFormatsNeverFallThroughToAnotherProtocol() {
        String json = "{\"tool\":\"search_graph\",\"args\":{\"query\":\"M. Chen\"}}";
        String nativeCall = "<|tool_call_start|>[search_graph(query=\"M. Chen\")]"
                + "<|tool_call_end|>";
        String barePython = "search_graph(query='M. Chen')";

        ToolCallParser.ParseResult nativeRejectsJson = ToolCallParser.parse(
                json, TOOLS, ChatTemplate.ToolCallFormat.NATIVE,
                ChatTemplate.ToolChoice.REQUIRED);
        ToolCallParser.ParseResult nativeRejectsBarePython = ToolCallParser.parse(
                barePython, TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
        ToolCallParser.ParseResult jsonRejectsNative = ToolCallParser.parse(
                nativeCall, TOOLS, ChatTemplate.ToolCallFormat.JSON,
                ChatTemplate.ToolChoice.REQUIRED);
        ToolCallParser.ParseResult jsonRejectsBarePython = ToolCallParser.parse(
                barePython, TOOLS, ChatTemplate.ToolCallFormat.JSON);

        assertTrue(nativeRejectsJson.getToolCalls().isEmpty());
        assertFalse(nativeRejectsJson.getErrors().isEmpty());
        assertTrue(nativeRejectsBarePython.getToolCalls().isEmpty());
        assertTrue(jsonRejectsNative.getToolCalls().isEmpty());
        assertFalse(jsonRejectsNative.getErrors().isEmpty());
        assertTrue(jsonRejectsBarePython.getToolCalls().isEmpty());
    }

    @Test
    void callsCannotExecuteWithoutADeclaration() {
        ToolCallParser.ParseResult json = ToolCallParser.parse(
                "{\"tool\":\"search_graph\",\"args\":{\"query\":\"M. Chen\"}}",
                List.of(), ChatTemplate.ToolCallFormat.JSON);
        ToolCallParser.ParseResult nativeCall = ToolCallParser.parse(
                "<|tool_call_start|>[search_graph(query=\"M. Chen\")]<|tool_call_end|>",
                List.of(), ChatTemplate.ToolCallFormat.NATIVE);

        assertTrue(json.getToolCalls().isEmpty());
        assertEquals(List.of("undeclared tool search_graph"), json.getErrors());
        assertTrue(nativeCall.getToolCalls().isEmpty());
        assertEquals(List.of("undeclared tool search_graph"), nativeCall.getErrors());
    }

    @Test
    void requiredNativeChatDerivesExactToolConstraint() {
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("Search")))
                .tools(TOOLS)
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .toolChoice(ChatTemplate.ToolChoice.REQUIRED)
                .build();
        SamplingConfig base = SamplingConfig.greedy();

        SamplingConfig resolved = GenerationPipeline.samplingForChat(request, base);

        assertFalse(base.hasConstraint());
        assertTrue(resolved.hasConstraint());
        assertEquals(NativeToolCallConstraint.TYPE,
                resolved.getConstraintConfig().getType());
        assertEquals(List.of("search_graph"),
                resolved.getConstraintConfig().getToolNames());
        assertEquals(Map.of("search_graph", List.of("query")),
                resolved.getConstraintConfig().getToolArgumentNames());
        assertEquals(Map.of("search_graph", Map.of("query", List.of("M. Chen"))),
                resolved.getConstraintConfig().getToolArgumentValues());
        assertTrue(resolved.getConstraintConfig().buildConstraint().isAccepting(
                "<|tool_call_start|>[search_graph(query=\"M. Chen\")]"
                        + "<|tool_call_end|>"));
        assertFalse(resolved.getConstraintConfig().buildConstraint().isAccepting(
                "<|tool_call_start|>[search_graph(other=\"M. Chen\")]"
                        + "<|tool_call_end|>"));
        assertFalse(resolved.getConstraintConfig().buildConstraint().isAccepting(
                "<|tool_call_start|>[search_graph(query=\"Someone else\")]"
                        + "<|tool_call_end|>"));
    }

    @Test
    void requiredNativeChatPreservesArraySchemaDuringTokenMasking() {
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("Extract")))
                .tools(ENTITY_TOOLS)
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .toolChoice(ChatTemplate.ToolChoice.REQUIRED)
                .build();

        SamplingConfig resolved =
                GenerationPipeline.samplingForChat(request, SamplingConfig.greedy());
        assertEquals(ENTITY_TOOLS.get(0).getParameters(),
                resolved.getConstraintConfig().getToolParameterSchemas()
                        .get("submit_entities"));

        var constraint = resolved.getConstraintConfig().buildConstraint();
        String prefix = "<|tool_call_start|>[submit_entities(names=[";
        assertFalse(constraint.canExtend(prefix, "M"),
                "array string items must begin with a JSON string quote");
        assertTrue(constraint.canExtend(prefix, "\"M. Chen\""));
        assertTrue(constraint.isAccepting(
                prefix + "\"M. Chen\",\"J. Park\"])]<|tool_call_end|>"));
        assertFalse(constraint.isAccepting(
                prefix + "\"M. Chen\",\"M. Chen\"])]<|tool_call_end|>"),
                "uniqueItems must reject duplicate entities");
        assertFalse(constraint.canExtend(
                prefix + "\"M. Chen\",", "\"M. Chen\""),
                "the duplicate must be masked before a complete call is emitted");
        assertFalse(constraint.isAccepting(
                prefix + "\"A\",\"B\",\"C\",\"D\",\"E\"])]<|tool_call_end|>"),
                "maxItems must cap runaway arrays");
        assertFalse(constraint.canExtend(
                prefix + "\"A\",\"B\",\"C\",\"D\"", ","),
                "a fifth item separator must be masked at maxItems");
        assertFalse(constraint.isAccepting(
                prefix + "\"" + "x".repeat(33) + "\"])]<|tool_call_end|>"),
                "item maxLength must cap a runaway quoted value");
        assertFalse(constraint.canExtend(
                prefix + "\"" + "x".repeat(32), "x"),
                "the next character must be masked at item maxLength");
    }

    @Test
    void nativeSchemaAllowsDeclaredOptionalArgumentsWithoutRequiringThem() {
        ChatTemplate.Tool graphQuery = ChatTemplate.Tool.function(
                "query_graph", "Query graph",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "operation", Map.of(
                                        "type", "string",
                                        "enum", List.of("LOOKUP")),
                                "query", Map.of("type", "string")),
                        "required", List.of("operation")));
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("Query")))
                .tools(List.of(graphQuery))
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .toolChoice(ChatTemplate.ToolChoice.REQUIRED)
                .build();
        var constraint = GenerationPipeline.samplingForChat(
                request, SamplingConfig.greedy())
                .getConstraintConfig().buildConstraint();
        String prefix = "<|tool_call_start|>[query_graph(";

        assertTrue(constraint.isAccepting(
                prefix + "operation=\"LOOKUP\")]<|tool_call_end|>"));
        assertTrue(constraint.isAccepting(
                prefix + "query=\"M. Chen\",operation=\"LOOKUP\")]<|tool_call_end|>"));
        assertFalse(constraint.isAccepting(
                prefix + "operation=\"LOOKUP\",unknown=\"x\")]<|tool_call_end|>"));
        assertFalse(constraint.isAccepting(
                prefix + "operation=\"LOOKUP\",operation=\"LOOKUP\")]<|tool_call_end|>"));
    }

    @Test
    void parserRejectsCompleteCallsWhoseArgumentsViolateTheDeclaredSchema() {
        ToolCallParser.ParseResult scalar = ToolCallParser.parse(
                "<|tool_call_start|>[submit_entities(names=\"M. Chen\")]"
                        + "<|tool_call_end|>",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
        ToolCallParser.ParseResult duplicates = ToolCallParser.parse(
                "<|tool_call_start|>[submit_entities("
                        + "names=[\"M. Chen\",\"M. Chen\"])]<|tool_call_end|>",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
        ToolCallParser.ParseResult valid = ToolCallParser.parse(
                "<|tool_call_start|>[submit_entities("
                        + "names=[\"M. Chen\",\"J. Park\"])]<|tool_call_end|>",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
        ToolCallParser.ParseResult trailingNative = ToolCallParser.parse(
                "<|tool_call_start|>[submit_entities("
                        + "names=[\"M. Chen\"]?)]<|tool_call_end|>",
                ENTITY_TOOLS, ChatTemplate.ToolCallFormat.NATIVE);
        ToolCallParser.ParseResult trailingJson = ToolCallParser.parse(
                "{\"tool\":\"search_graph\",\"args\":{\"query\":\"M. Chen\"}} ignored",
                TOOLS, ChatTemplate.ToolCallFormat.JSON,
                ChatTemplate.ToolChoice.REQUIRED);

        assertTrue(scalar.getToolCalls().isEmpty());
        assertTrue(scalar.getErrors().get(0).contains("$.names must be an array"));
        assertTrue(duplicates.getToolCalls().isEmpty());
        assertTrue(duplicates.getErrors().get(0).contains("unique items"));
        assertEquals(List.of("M. Chen", "J. Park"),
                valid.getToolCalls().get(0).getArguments().get("names"));
        assertTrue(valid.getErrors().isEmpty());
        assertTrue(trailingNative.getToolCalls().isEmpty());
        assertFalse(trailingNative.getErrors().isEmpty());
        assertTrue(trailingNative.getErrors().get(0).contains("$.names must be an array"));
        assertTrue(trailingJson.getToolCalls().isEmpty());
        assertFalse(trailingJson.getErrors().isEmpty());
    }

    @Test
    void automaticToolChoiceDoesNotForceAConstraint() {
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("Maybe search")))
                .tools(TOOLS)
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .toolChoice(ChatTemplate.ToolChoice.AUTO)
                .build();
        SamplingConfig base = SamplingConfig.greedy();

        assertSame(base, GenerationPipeline.samplingForChat(request, base));
    }

    @Test
    void requiredToolChoiceRejectsAnEmptyToolSet() {
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("Search")))
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .toolChoice(ChatTemplate.ToolChoice.REQUIRED)
                .build();

        assertThrows(IllegalArgumentException.class,
                () -> GenerationPipeline.samplingForChat(
                        request, SamplingConfig.greedy()));
    }
}
