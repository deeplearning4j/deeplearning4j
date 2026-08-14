/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TokenizerChatTemplateOverrideTest {

    private static final String TEST_TOKENIZER_JSON = "{"
            + "\"version\":\"1.0\","
            + "\"truncation\":null,"
            + "\"padding\":null,"
            + "\"added_tokens\":[],"
            + "\"normalizer\":null,"
            + "\"pre_tokenizer\":{\"type\":\"Whitespace\"},"
            + "\"post_processor\":null,"
            + "\"decoder\":null,"
            + "\"model\":{"
            + "\"type\":\"WordLevel\","
            + "\"vocab\":{\"[UNK]\":0,\"hello\":1,\"world\":2},"
            + "\"unk_token\":\"[UNK]\""
            + "}"
            + "}";

    @Test
    void addedSpecialTokensComeFromTokenizerMetadata() {
        String tokenizerJson = "{\"added_tokens\":["
                + "{\"id\":6,\"content\":\"<|im_start|>\",\"special\":true},"
                + "{\"id\":7,\"content\":\"<|im_end|>\",\"special\":true},"
                + "{\"id\":42,\"content\":\"ordinary\",\"special\":false}]}";

        assertEquals(java.util.Set.of(6, 7),
                HuggingFaceTokenizer.parseSpecialTokenIds(tokenizerJson));
    }

    @Test
    void addedProtocolTokensRemainAddressableWhenNotMarkedSpecial() {
        String tokenizerJson = "{\"added_tokens\":["
                + "{\"id\":10,\"content\":\"<|tool_call_start|>\",\"special\":false},"
                + "{\"id\":11,\"content\":\"<|tool_call_end|>\",\"special\":false}]}";

        assertEquals(Map.of(
                        ChatTemplate.NATIVE_TOOL_CALL_START, 10,
                        ChatTemplate.NATIVE_TOOL_CALL_END, 11),
                HuggingFaceTokenizer.parseAddedTokenIds(tokenizerJson));
        assertTrue(HuggingFaceTokenizer.parseSpecialTokenIds(tokenizerJson).isEmpty());
    }

    @Test
    void structuredToolSchemaIsRenderedAsDeterministicJson() {
        Tokenizer tokenizer = new TemplateLessTokenizer();
        ChatTemplate.Tool tool = ChatTemplate.Tool.function(
                "record_graph_verdict",
                "Record a verdict.",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "disposition", Map.of(
                                        "type", "string",
                                        "enum", List.of("ALLOW", "DENY", "REVIEW"))),
                        "required", List.of("disposition"),
                        "additionalProperties", false));
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("route this verdict")))
                .tools(List.of(tool))
                .toolDefinitionFormat(ChatTemplate.ToolDefinitionFormat.STANDARD)
                .addGenerationPrompt(true)
                .build();

        String rendered = tokenizer.applyChatTemplate(request,
                "### Instruction:\n{{ prompt }}\n### Response:\n");

        assertTrue(rendered.contains("\"parameters\":{\"additionalProperties\":false"));
        assertTrue(rendered.contains("\"enum\":[\"ALLOW\",\"DENY\",\"REVIEW\"]"));
        assertTrue(rendered.contains("\"type\":\"function\""));
        assertTrue(!rendered.contains("type=object"));
    }

    @Test
    void lfmNativeTemplateReceivesFlatToolsInItsModelOwnedSystemShape() {
        Tokenizer tokenizer = new TemplateLessTokenizer();
        ChatTemplate.Tool tool = ChatTemplate.Tool.function(
                "record_graph_verdict",
                "Record a verdict.",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "candidate_id", Map.of("type", "string"),
                                "disposition", Map.of("type", "string")),
                        "required", List.of("candidate_id", "disposition")));
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(
                        ChatTemplate.Message.system("Use the declared function."),
                        ChatTemplate.Message.user("Route the graph verdict.")))
                .tools(List.of(tool))
                .toolDefinitionFormat(ChatTemplate.ToolDefinitionFormat.FLAT)
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .addGenerationPrompt(true)
                .build();

        String rendered = tokenizer.applyChatTemplate(request,
                "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>"
                        + "<|tool_call_start|><|tool_call_end|>");

        assertTrue(rendered.startsWith(
                "<|im_start|>system\nUse the declared function.\nList of tools: ["));
        assertTrue(rendered.contains("\"name\":\"record_graph_verdict\""));
        assertTrue(rendered.contains(
                "\"parameters\":{\"properties\":{\"candidate_id\":{\"type\":\"string\"}"));
        assertTrue(!rendered.contains("\"type\":\"function\""));
        assertTrue(!rendered.contains("Available tools:"));
        assertTrue(rendered.endsWith("<|im_start|>assistant\n"));
    }

    @Test
    void resolvedNativeProtocolUsesLfmToolShapeWithPreFixTemplate() {
        Tokenizer tokenizer = new TemplateLessTokenizer();
        ChatTemplate.Tool tool = ChatTemplate.Tool.function(
                "submit_graph_delta",
                "Submit the extracted graph.",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "entities", Map.of("type", "array"),
                                "relations", Map.of("type", "array")),
                        "required", List.of("entities", "relations")));
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(
                        ChatTemplate.Message.system("Extract the graph."),
                        ChatTemplate.Message.user("Text: Alex works at Acme.")))
                .tools(List.of(tool))
                .toolDefinitionFormat(ChatTemplate.ToolDefinitionFormat.FLAT)
                .toolCallFormat(ChatTemplate.ToolCallFormat.NATIVE)
                .addGenerationPrompt(true)
                .build();

        // LFM2.5's pre-fix template did not contain the native call sentinels even
        // though tokenizer metadata declared them and the pipeline resolved NATIVE.
        String rendered = tokenizer.applyChatTemplate(request,
                "{{- bos_token -}}<|im_start|>{{ message.role }}\n"
                        + "{{ message.content }}<|im_end|>");

        assertTrue(rendered.contains("Extract the graph.\nList of tools: ["));
        assertTrue(rendered.contains("\"name\":\"submit_graph_delta\""));
        assertTrue(!rendered.contains("Available tools:"));
        assertTrue(rendered.endsWith("<|im_start|>assistant\n"));
    }

    @Test
    void structuredRequestUsesPipelineOverrideWhenTokenizerHasNoTemplate() {
        Tokenizer tokenizer = new TemplateLessTokenizer();
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("route this verdict")))
                .addGenerationPrompt(true)
                .build();

        assertThrows(IllegalStateException.class, () -> tokenizer.applyChatTemplate(request));

        String rendered = tokenizer.applyChatTemplate(request,
                "### Instruction:\n{{ prompt }}\n### Response:\n");

        assertTrue(rendered.contains("### Instruction:\nroute this verdict"));
        assertTrue(rendered.endsWith("### Response:\n"));
    }

    @Test
    void activeTemplateDerivesAssistantTurnStopWithoutModelNameChecks() {
        String tokenizerJson = "{"
                + "\"version\":\"1.0\","
                + "\"truncation\":null,\"padding\":null,"
                + "\"added_tokens\":["
                + "{\"id\":3,\"content\":\"<|im_start|>\",\"single_word\":false,"
                + "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true},"
                + "{\"id\":4,\"content\":\"<|im_end|>\",\"single_word\":false,"
                + "\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}],"
                + "\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"Whitespace\"},"
                + "\"post_processor\":null,\"decoder\":null,"
                + "\"model\":{\"type\":\"WordLevel\","
                + "\"vocab\":{\"[UNK]\":0,\"hello\":1,\"world\":2},"
                + "\"unk_token\":\"[UNK]\"}}";
        String template = "{% for message in messages %}<|im_start|>{{ message.role }}\\n"
                + "{{ message.content }}<|im_end|>\\n{% endfor %}"
                + "{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}";

        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromJson(tokenizerJson)) {
            assertEquals(java.util.Set.of(4), tokenizer.getChatTemplateStopTokenIds(template));
        }
    }

    @Test
    void nativeTokenizerRendersImportedOverrideWithCompleteToolContext() {
        ChatTemplate.Tool tool = ChatTemplate.Tool.function(
                "submit_graph_delta",
                "Submit the extracted graph.",
                Map.of(
                        "type", "object",
                        "properties", Map.of(
                                "entities", Map.of("type", "array"),
                                "relations", Map.of("type", "array")),
                        "required", List.of("entities", "relations")));
        ChatTemplate.Request request = ChatTemplate.Request.builder()
                .messages(List.of(ChatTemplate.Message.user("Alex works at Acme.")))
                .tools(List.of(tool))
                .toolDefinitionFormat(ChatTemplate.ToolDefinitionFormat.STANDARD)
                .toolCallFormat(ChatTemplate.ToolCallFormat.JSON)
                .addGenerationPrompt(true)
                .build();
        String override = "{% if tools %}<tools>{% for tool in tools %}"
                + "{{ tool.function.name }}="
                + "{{ tool.function.parameters.properties | length }};"
                + "{% endfor %}</tools>{% endif %}"
                + "{% for message in messages %}<{{ message.role }}>"
                + "{{ message.content }}</{{ message.role }}>{% endfor %}"
                + "{% if add_generation_prompt %}<assistant>{% endif %}";

        try (HuggingFaceTokenizer tokenizer =
                     HuggingFaceTokenizer.fromJson(TEST_TOKENIZER_JSON)) {
            assertEquals(
                    "<tools>submit_graph_delta=2;</tools>"
                            + "<user>Alex works at Acme.</user><assistant>",
                    tokenizer.applyChatTemplate(request, override));
        }
    }

    private static final class TemplateLessTokenizer implements Tokenizer {
        @Override
        public Encoding encode(String text, boolean addSpecialTokens) {
            return null;
        }

        @Override
        public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
            return List.of();
        }

        @Override
        public String decode(int[] ids, boolean skipSpecialTokens) {
            return "";
        }

        @Override
        public List<String> decodeBatch(List<int[]> idsBatch, boolean skipSpecialTokens) {
            return List.of();
        }

        @Override
        public int getVocabSize() {
            return 0;
        }

        @Override
        public Integer getTokenId(String token) {
            return null;
        }

        @Override
        public String getToken(int id) {
            return "";
        }

        @Override
        public Map<String, Integer> getVocab() {
            return Map.of();
        }

        @Override
        public int getPadTokenId() {
            return -1;
        }

        @Override
        public int getBosTokenId() {
            return -1;
        }

        @Override
        public int getEosTokenId() {
            return -1;
        }

        @Override
        public int getUnkTokenId() {
            return -1;
        }

        @Override
        public boolean isValid() {
            return true;
        }

        @Override
        public void close() {
        }
    }
}
