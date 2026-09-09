/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm.generation;

import org.eclipse.deeplearning4j.llm.generation.constraint.GemmaToolCallConstraint;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.nd4j.ggml.format.GGUFReader;
import org.nd4j.ggml.format.GGMLMetadata;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** Header/tokenizer only: never imports weights, initializes ND4J or runs inference. */
class GemmaChatInputContractTest {
    private static final ChatTemplate.Tool TOOL = ChatTemplate.Tool.function("record_organization",
            "Record one organization mentioned in the text.", Map.of("type", "object",
                    "properties", Map.of("name", Map.of("type", "string")), "required", List.of("name")));

    @Test
    void requiredChatUsesDeclaredGemmaSchemaWithoutReflection() {
        var required = ChatTemplate.Request.builder().tools(List.of(TOOL))
                .toolCallFormat(ChatTemplate.ToolCallFormat.GEMMA).toolChoice(ChatTemplate.ToolChoice.REQUIRED).build();
        var config = GenerationPipeline.samplingForChat(required, SamplingConfig.greedy());
        assertEquals(GemmaToolCallConstraint.TYPE, config.getConstraintConfig().getType());
        var constraint = config.getConstraintConfig().buildConstraint();
        assertFalse(constraint.isAccepting("<|tool_call>call:record_organization{}<tool_call|>"));
        assertTrue(constraint.isAccepting("<|tool_call>call:record_organization{name:<|\"|>Acme<|\"|>}<tool_call|>"));
        var optional = ChatTemplate.Request.builder().tools(List.of(TOOL))
                .toolCallFormat(ChatTemplate.ToolCallFormat.GEMMA).build();
        var base = SamplingConfig.greedy();
        assertSame(base, GenerationPipeline.samplingForChat(optional, base));
    }

    @Test
    void localGgufVocabularyAndRenderedPromptAgree() throws Exception {
        String directory = System.getProperty("gemma.assets");
        assertNotNull(directory, "Supply -Dgemma.assets for the real weights-free contract gate");
        File root = new File(directory);
        File gguf = new File(root, "gemma-4-E2B-it-Q4_K_M.gguf");
        GGMLMetadata.TokenizerInfo info;
        try (GGUFReader reader = new GGUFReader(gguf)) {
            info = GGMLMetadata.TokenizerInfo.fromGGUFHeader(reader.getHeader());
            reader.getHeader().getMetadata().forEach((key, value) -> {
                if (key.startsWith("gemma") || key.equals("general.architecture")) {
                    System.out.println("GEMMA_METADATA " + key + "=" + java.util.Arrays.deepToString(new Object[]{value}));
                }
            });
            for (var tensor : reader.getTensorInfos()) {
                if (!tensor.getName().startsWith("blk.") || tensor.getName().startsWith("blk.0.")
                        || tensor.getName().startsWith("blk.5.") || tensor.getName().startsWith("blk.34.")) {
                    System.out.println("GEMMA_TENSOR " + tensor.getName() + "="
                            + java.util.Arrays.toString(tensor.getShape()) + " type=" + tensor.getDataType());
                }
            }
        }
        assertNotNull(info.getChatTemplate());
        assertTrue(info.getChatTemplate().contains("<|turn>"), "wrong turn template");
        assertTrue(info.getChatTemplate().contains("<|tool_call>"), "GGUF must declare tool-call protocol");
        var metadata = GenerationPipeline.ModelMetadata.of(info.getBosTokenId(), info.getEosTokenId(),
                info.getPadTokenId(), info.getChatTemplate(), Set.of(), Set.of());
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromFile(new File(root, "tokenizer.json"))) {
            assertEquals(info.getTokens().size(), tokenizer.getVocabSize(), "GGUF/HF vocabulary size mismatch");
            for (int id = 0; id < info.getTokens().size(); id++) {
                assertEquals(Integer.valueOf(id), tokenizer.getTokenId(info.getTokens().get(id)),
                        "GGUF/HF vocabulary mismatch at id=" + id);
            }
            var arguments = GenerationPipeline.chatTemplateArguments(Map.of(), metadata, tokenizer);
            assertEquals("<bos>", arguments.get("bos_token"));
            assertEquals("<bos>", tokenizer.getToken(info.getBosTokenId()));
            String text = "Classify organization types in: The quick brown fox";
            var request = ChatTemplate.Request.builder().messages(List.of(ChatTemplate.Message.user(text)))
                    .tools(List.of(TOOL)).toolChoice(ChatTemplate.ToolChoice.REQUIRED)
                    .toolCallFormat(ChatTemplate.ToolCallFormat.GEMMA).templateArguments(arguments).build();
            String rendered = tokenizer.applyChatTemplate(request, info.getChatTemplate());
            System.out.println("GEMMA_CHAT_TEMPLATE=" + info.getChatTemplate());
            System.out.println("GEMMA_RENDERED_PROMPT=" + rendered);
            assertTrue(rendered.startsWith("<bos><|turn>system\n"), rendered);
            assertTrue(rendered.contains("<|tool>declaration:record_organization"), rendered);
            assertTrue(rendered.contains("name:{type:<|\"|>STRING<|\"|>}"), rendered);
            assertTrue(rendered.contains("required:[<|\"|>name<|\"|>]"), rendered);
            assertTrue(rendered.contains("<|turn>user\n" + text + "<turn|>"), rendered);
            assertTrue(rendered.endsWith("<|turn>model\n"), rendered);
            assertFalse(rendered.contains("Available tools:"));
            int[] ids = tokenizer.ensureLeadingBos(tokenizer.encode(rendered, false)).getIds();
            System.out.println("GEMMA_PROMPT_IDS=" + java.util.Arrays.toString(ids));
            assertEquals(info.getBosTokenId(), ids[0]);
            assertNotEquals(info.getBosTokenId(), ids[1], "duplicate BOS");
            assertEquals(rendered, tokenizer.decode(ids, false));
            assertEquals(ChatTemplate.ToolCallFormat.GEMMA,
                    GenerationPipeline.selectModelToolCallFormat(new ChatTemplate(info.getChatTemplate(),
                            "<bos>", "<eos>"), tokenizer));
            assertTrue(tokenizer.getChatTemplateStopTokenIds(info.getChatTemplate()).contains(106));
            var explicit = GenerationPipeline.chatTemplateArguments(Map.of("bos_token", "custom"), metadata, tokenizer);
            assertEquals("custom", explicit.get("bos_token"));
        }
    }
}
