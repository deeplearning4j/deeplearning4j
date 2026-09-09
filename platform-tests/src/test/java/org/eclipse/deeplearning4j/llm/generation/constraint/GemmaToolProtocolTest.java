/*
 * SPDX-License-Identifier: Apache-2.0
 * This program is made available under the Apache License, Version 2.0.
 * See https://www.apache.org/licenses/LICENSE-2.0 and the NOTICE file.
 */
package org.eclipse.deeplearning4j.llm.generation.constraint;

import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.generation.ToolCallParser;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Test;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Pure Java protocol regressions: no tokenizer/model artifacts or inference required. */
class GemmaToolProtocolTest {
    private static final String Q = ChatTemplate.GEMMA_STRING;
    private static final String OPEN = ChatTemplate.GEMMA_TOOL_CALL_START;
    private static final String CLOSE = ChatTemplate.GEMMA_TOOL_CALL_END;
    private static final ChatTemplate.Tool ORGANIZATION = new ChatTemplate.Tool(
            "record_organization", "Record organization", Map.of(
            "type", "object", "properties", Map.of("name", Map.of("type", "string")),
            "required", List.of("name"), "additionalProperties", false));

    private static ChatTemplate template() {
        return new ChatTemplate(OPEN + CLOSE + "<|tool>declaration:<tool|>"
                + ChatTemplate.GEMMA_THOUGHT_START + ChatTemplate.GEMMA_CHANNEL_END, "", "");
    }

    private static String call(String name, String arguments) {
        return OPEN + "call:" + name + arguments + CLOSE;
    }

    private static TextConstraint constraint(ChatTemplate.Tool tool) {
        return ConstraintConfig.gemmaToolCall(Map.of(tool.getName(), List.of()),
                Map.of(tool.getName(), tool.getParameters())).buildConstraint();
    }

    private static ToolCallParser.ParseResult parse(String text, ChatTemplate.Tool tool) {
        return ToolCallParser.parse(text, List.of(tool), ChatTemplate.ToolCallFormat.GEMMA,
                ChatTemplate.ToolChoice.REQUIRED);
    }

    @Test
    void detectsTemplateMarkersAndRendersCanonicalDeclaration() {
        assertEquals(ChatTemplate.ToolCallFormat.GEMMA, template().toolCallFormat());
        ChatTemplate.Tool tool = new ChatTemplate.Tool("record_organization", "Record organization",
                Map.of("type", "object", "properties", Map.of("name", Map.of("type", "string")),
                        "required", List.of("name")));
        String prompt = template().apply(ChatTemplate.Request.builder().tools(List.of(tool))
                .messages(List.of(ChatTemplate.Message.user("Remember Acme Robotics"))).build());
        assertTrue(prompt.contains("<|tool>declaration:record_organization{description:" + Q
                + "Record organization" + Q + ",parameters:{properties:{name:{type:" + Q
                + "STRING" + Q + "}},required:[" + Q + "name" + Q + "],type:" + Q
                + "OBJECT" + Q + "}}<tool|>"), prompt);
        assertFalse(prompt.contains("Available tools:"));
        assertFalse(prompt.contains("List of tools:"));
        assertEquals(ChatTemplate.ToolCallFormat.JSON, new ChatTemplate("Gemma model", "", "").toolCallFormat());
    }

    @Test
    void literalQuotesBackslashesAndNullSurviveWithoutJsonReplacement() throws Exception {
        String literal = "Acme \"Robotics\" \\ path\\n\nnext line";
        String raw = call(ORGANIZATION.getName(), "{name:" + Q + literal + Q + "}");
        ToolCallParser.ParseResult result = parse(raw, ORGANIZATION);
        assertTrue(result.isClean(), result.getErrors().toString());
        assertEquals(literal, result.getToolCalls().get(0).getArguments().get("name"));
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(result.getToolCalls().get(0).getArguments());
        assertEquals(literal, mapper.readTree(json).get("name").asText());
        assertEquals(Q + literal + Q, GemmaToolCallCodec.encode(literal));

        ChatTemplate.Tool nullable = new ChatTemplate.Tool("nullable", "", Map.of("type", "object",
                "properties", Map.of("value", Map.of("type", "null")), "required", List.of("value")));
        ToolCallParser.ParseResult nullResult = parse(call("nullable", "{value:null}"), nullable);
        assertTrue(nullResult.isClean());
        assertTrue(nullResult.getToolCalls().get(0).getArguments().containsKey("value"));
        assertNull(nullResult.getToolCalls().get(0).getArguments().get("value"));
    }

    @Test
    void everyTokenBoundaryAndWholeCandidateUseTheSameGrammar() {
        String raw = call(ORGANIZATION.getName(), "{name:" + Q + "Acme Robotics" + Q + "}");
        TextConstraint constraint = constraint(ORGANIZATION);
        assertTrue(constraint.canExtend("", raw));
        for (int boundary = 0; boundary < raw.length(); boundary++) {
            String prefix = raw.substring(0, boundary);
            assertFalse(constraint.isAccepting(prefix), prefix);
            assertTrue(constraint.canExtend(prefix, raw.substring(boundary)), "suffix at " + boundary);
            assertTrue(constraint.canExtend(prefix, raw.substring(boundary, boundary + 1)), "char at " + boundary);
        }
        assertTrue(constraint.isAccepting(raw));
        assertTrue(constraint.reset().isAccepting(raw));
        assertTrue(constraint.allowsSpecialToken("", OPEN));
        assertFalse(constraint.allowsSpecialToken("", Q));
        assertFalse(constraint.allowsSpecialToken("", CLOSE));
        String valueStart = OPEN + "call:record_organization{name:";
        assertTrue(constraint.allowsSpecialToken(valueStart, Q));
        assertTrue(constraint.allowsSpecialToken(valueStart + Q + "Acme", Q));
        assertFalse(constraint.allowsSpecialToken(valueStart + Q + "Acme", CLOSE));
        assertFalse(constraint.allowsSpecialToken(valueStart + Q + "Acme", OPEN));
        assertFalse(constraint.allowsSpecialToken(valueStart + Q, "<|channel>"));
        assertTrue(constraint.canExtend(valueStart + Q + "Acme<|", "\"|>}" + CLOSE));
        assertFalse(constraint.canExtend("", raw + "garbage"));
    }

    @Test
    void maskerValidatesSpecialsAcrossFullCandidatePieces() {
        TextConstraint c = constraint(ORGANIZATION);
        ConstraintMasker masker = new ConstraintMasker(c, 16);
        List<String> specials = List.of(OPEN, CLOSE, Q, "<|channel>", "<eos>");
        String prefix = OPEN + "call:record_organization{name:" + Q + "Acme<|";
        masker.tokenEmitted(0, ignored -> prefix);
        String complete = prefix + "\"|>}" + CLOSE;
        assertTrue(masker.allowsDecodedText(complete, specials));
        assertFalse(masker.allowsDecodedText(prefix + "channel>bad" + Q + "}" + CLOSE, specials));
        assertFalse(masker.allowsDecodedText(complete + "<eos>", specials));
        assertTrue(masker.allowsDecodedText(complete + call(ORGANIZATION.getName(),
                "{name:" + Q + "Second" + Q + "}"), specials));
    }

    @Test
    void preservesAdjacentCallsAndRejectsPartialOrInvalidListsAtomically() {
        String raw = call(ORGANIZATION.getName(), "{name:" + Q + "Acme Robotics" + Q + "}");
        ToolCallParser.ParseResult two = parse(raw + raw, ORGANIZATION);
        assertTrue(two.isClean());
        assertEquals(2, two.getCalls().size(), "Identical adjacent invocations must not be deduplicated");
        assertTrue(constraint(ORGANIZATION).isAccepting(raw + raw));
        assertTrue(constraint(ORGANIZATION).canExtend(raw, raw));
        for (String invalid : List.of(raw + OPEN, raw + call("unknown", "{}"),
                raw + call(ORGANIZATION.getName(), "{}"))) {
            assertFalse(parse(invalid, ORGANIZATION).isClean(), invalid);
            assertTrue(parse(invalid, ORGANIZATION).getCalls().isEmpty(), invalid);
        }
    }

    @Test
    void rejectsMalformedUndeclaredMissingRequiredAndWrongTypes() {
        List<String> arguments = List.of("{}", "{name:12}", "{name:true}", "{name:null}",
                "{name:\"Acme\"}", "{name:" + Q + "Acme}",
                "{name:" + Q + "Acme" + Q + ",}",
                "{name:" + Q + "A" + Q + ",name:" + Q + "B" + Q + "}",
                "{unknown:" + Q + "Acme" + Q + "}",
                "{name:" + Q + "<|channel>thought" + Q + "}");
        for (String args : arguments) {
            String raw = call(ORGANIZATION.getName(), args);
            assertFalse(parse(raw, ORGANIZATION).isClean(), raw);
            assertTrue(parse(raw, ORGANIZATION).getCalls().isEmpty(), raw);
            assertFalse(constraint(ORGANIZATION).isAccepting(raw), raw);
            assertFalse(constraint(ORGANIZATION).canExtend("", raw), raw);
        }
        assertFalse(parse(call("unknown", "{}"), ORGANIZATION).isClean());
        assertFalse(constraint(ORGANIZATION).canExtend(OPEN + "call:", "unknown"));
        assertFalse(parse("", ORGANIZATION).isClean());
    }

    @Test
    void nestedSchemaAndScalarPrefixesAreEnforced() {
        Map<String, Object> item = Map.of("type", "object", "properties", Map.of(
                "label", Map.of("type", "string", "enum", List.of("ok")),
                "weight", Map.of("type", "number", "minimum", 10)),
                "required", List.of("label", "weight"), "additionalProperties", false);
        ChatTemplate.Tool nested = new ChatTemplate.Tool("submit", "", Map.of("type", "object",
                "properties", Map.of("items", Map.of("type", "array", "items", item, "minItems", 1,
                                "maxItems", 2), "active", Map.of("type", "boolean")),
                "required", List.of("items", "active"), "additionalProperties", false));
        String args = "{items:[{label:" + Q + "ok" + Q + ",weight:10}],active:true}";
        String raw = call("submit", args);
        assertTrue(parse(raw, nested).isClean());
        TextConstraint c = constraint(nested);
        for (int i = 0; i < raw.length(); i++) {
            assertTrue(c.canExtend(raw.substring(0, i), raw.substring(i, i + 1)), "nested boundary " + i);
        }
        assertTrue(c.isAccepting(raw));
        for (String bad : List.of(args.replace("weight:10", "weight:false"),
                args.replace("weight:10", "weight:1"), args.replace("weight:10", "weight:01"),
                args.replace(",weight:10", ""), args.replace("active:true", "active:truth"),
                args.replace("ok", "bad"), args.replace("items:[{", "items:[{extra:null,"))) {
            assertFalse(c.canExtend("", call("submit", bad)), bad);
            assertFalse(parse(call("submit", bad), nested).isClean(), bad);
        }
        assertFalse(c.canExtend(OPEN + "call:submit{items:[{label:" + Q, "bad"));
        assertFalse(c.canExtend(OPEN + "call:submit{items:[{label:" + Q + "ok" + Q + ",weight:", Q));
    }

    @Test
    void nestedObjectsArraysNumbersBooleansAndNullAreLiteralData() {
        ChatTemplate.Tool any = new ChatTemplate.Tool("any", "", Map.of("type", "object"));
        String raw = call("any", "{obj:{x:[null,true,false,-1.25e+2,{},[]," + Q + "a,}:b" + Q + "]}}");
        assertTrue(parse(raw, any).isClean());
        assertTrue(constraint(any).isAccepting(raw));
        Map<?, ?> obj = (Map<?, ?>) parse(raw, any).getCalls().get(0).getArgs().get("obj");
        assertEquals(new BigDecimal("-1.25e+2"), ((List<?>) obj.get("x")).get(3));
    }

    @Test
    void thoughtChannelCannotBecomeExecutableArguments() {
        String raw = call(ORGANIZATION.getName(), "{name:" + Q + "Acme Robotics" + Q + "}");
        String thought = ChatTemplate.GEMMA_THOUGHT_START + call("unknown", "{}")
                + ChatTemplate.GEMMA_CHANNEL_END;
        assertTrue(parse(thought + raw, ORGANIZATION).isClean());
        assertEquals(1, parse(thought + raw, ORGANIZATION).getCalls().size());
        assertTrue(parse(thought, ORGANIZATION).getCalls().isEmpty());
        assertFalse(parse(ChatTemplate.GEMMA_THOUGHT_START + raw, ORGANIZATION).isClean());
        ChatTemplate.AssistantOutput output = template().parseAssistantOutput(thought + raw);
        assertEquals(raw, output.getContent());
        assertEquals(call("unknown", "{}"), output.getReasoningContent());
        List<ChatTemplate.OutputBlockDefinition> blocks = template().prefilledOutputBlocks(
                "base", "base" + ChatTemplate.GEMMA_THOUGHT_START);
        assertEquals(1, blocks.size());
        TextConstraint c = ConstraintConfig.gemmaToolCall(Map.of(ORGANIZATION.getName(), List.of("name")),
                        Map.of(ORGANIZATION.getName(), ORGANIZATION.getParameters()))
                .toBuilder().outputBlocks(blocks).build().buildConstraint();
        String prefix = "reasoning";
        assertFalse(c.isAccepting(prefix));
        assertTrue(c.allowsSpecialToken(prefix, ChatTemplate.GEMMA_CHANNEL_END));
        assertTrue(c.canExtend(prefix, ChatTemplate.GEMMA_CHANNEL_END + raw));
        assertTrue(c.isAccepting(prefix + ChatTemplate.GEMMA_CHANNEL_END + raw));
    }

    @Test
    void otherProtocolsRemainExplicitAndBoundsFailClosed() {
        assertTrue(ToolCallParser.parse("{\"tool\":\"record_organization\",\"args\":{\"name\":\"Acme\"}}",
                List.of(ORGANIZATION), ChatTemplate.ToolCallFormat.JSON).hasToolCalls());
        assertTrue(ToolCallParser.parse("<|tool_call_start|>[record_organization(name=\"Acme\")]<|tool_call_end|>",
                List.of(ORGANIZATION), ChatTemplate.ToolCallFormat.NATIVE).hasToolCalls());
        assertTrue(ToolCallParser.parse("<tool_call><function=record_organization>"
                        + "<parameter=name>Acme</parameter></function></tool_call>",
                List.of(ORGANIZATION), ChatTemplate.ToolCallFormat.XML).hasToolCalls());
        String raw = call(ORGANIZATION.getName(), "{name:" + Q + "Acme" + Q + "}");
        assertFalse(ToolCallParser.parse(raw, List.of(ORGANIZATION), ChatTemplate.ToolCallFormat.JSON).hasToolCalls());
        assertFalse(constraint(ORGANIZATION).isAccepting(raw.repeat(GemmaToolCallCodec.MAX_CALLS + 1)));
        assertFalse(constraint(ORGANIZATION).canExtend("", " ".repeat(GemmaToolCallCodec.MAX_CHARS + 1)));
        ChatTemplate.Tool any = new ChatTemplate.Tool("any", "", Map.of("type", "object"));
        String deep = call("any", "{x:" + "[".repeat(65) + "0" + "]".repeat(65) + "}");
        assertFalse(parse(deep, any).isClean());
    }
}
