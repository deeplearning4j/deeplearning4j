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
    void generationConfigPreservesScalarAndListTerminals(@org.junit.jupiter.api.io.TempDir java.nio.file.Path root)
            throws Exception {
        java.nio.file.Path tokenizerFile = root.resolve("tokenizer.json");
        java.nio.file.Files.writeString(tokenizerFile,
                "{\"version\":\"1.0\",\"model\":{\"type\":\"WordLevel\",\"vocab\":{\"[UNK]\":0,\"end\":1,\"turn\":2},\"unk_token\":\"[UNK]\"}}");
        try (var tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.toFile())) {
            assertTrue(tokenizer.getGenerationStopTokenIds().isEmpty());
        }
        for (String eos : List.of("1", "[1,2,1]")) {
            java.nio.file.Files.writeString(root.resolve("generation_config.json"), "{\"eos_token_id\":" + eos + "}");
            try (var tokenizer = HuggingFaceTokenizer.fromFile(tokenizerFile.toFile())) {
                assertEquals(eos.equals("1") ? Set.of(1) : Set.of(1, 2), tokenizer.getGenerationStopTokenIds());
            }
        }
        for (String eos : List.of("-1", "1.5", "\"1\"", "2147483648")) {
            java.nio.file.Files.writeString(root.resolve("generation_config.json"), "{\"eos_token_id\":" + eos + "}");
            assertThrows(org.eclipse.deeplearning4j.llm.tokenizer.TokenizerException.class,
                    () -> HuggingFaceTokenizer.fromFile(tokenizerFile.toFile()), eos);
        }
    }

    @Test
    void generationRolesFillMissingConfigWithoutInventingBosPolicy(
            @org.junit.jupiter.api.io.TempDir java.nio.file.Path root) throws Exception {
        var file = root.resolve("tokenizer.json");
        java.nio.file.Files.writeString(file, "{\"version\":\"1.0\",\"model\":{\"type\":\"WordLevel\","
                + "\"vocab\":{\"[UNK]\":0,\"start\":1,\"explicit\":2,\"pad\":3},\"unk_token\":\"[UNK]\"}}");
        var generation = root.resolve("generation_config.json");
        String template = "{{ bos_token }}";
        try (var tokenizer = HuggingFaceTokenizer.fromFile(file.toFile())) {
            assertEquals(-1, tokenizer.getBosTokenId());
            var encoding = tokenizer.encode("start", false);
            assertSame(encoding, tokenizer.ensureLeadingBos(encoding), "No model-owned BOS, no insertion");
        }
        java.nio.file.Files.writeString(generation, "{\"bos_token_id\":1,\"pad_token_id\":3}");
        for (boolean explicit : List.of(false, true)) {
            if (explicit) java.nio.file.Files.writeString(root.resolve("tokenizer_config.json"),
                    "{\"bos_token\":\"explicit\",\"pad_token\":\"explicit\"}");
            try (var tokenizer = HuggingFaceTokenizer.fromFile(file.toFile())) {
                int expected = explicit ? 2 : 1;
                assertEquals(expected, tokenizer.getBosTokenId());
                assertEquals(explicit ? 2 : 3, tokenizer.getPadTokenId());
                var args = GenerationPipeline.chatTemplateArguments(Map.of(),
                        GenerationPipeline.ModelMetadata.empty(), tokenizer);
                var request = ChatTemplate.Request.builder().messages(List.of(ChatTemplate.Message.user("hello")))
                        .templateArguments(args).build();
                String rendered = tokenizer.applyChatTemplate(request, template);
                int[] ids = tokenizer.ensureLeadingBos(tokenizer.encode(rendered, false)).getIds();
                assertArrayEquals(new int[]{expected}, ids, "Exactly one explicitly rendered BOS");
                assertFalse(tokenizer.addsLeadingBos(), "Metadata role must not imply post-processor insertion policy");
                var plain = tokenizer.encode("pad", false);
                assertSame(plain, tokenizer.ensureLeadingBos(plain));
            }
        }
        java.nio.file.Files.delete(root.resolve("tokenizer_config.json"));
        for (String invalid : List.of("-1", "1.5", "\"1\"", "2147483648", "99", "[1]")) {
            java.nio.file.Files.writeString(generation, "{\"bos_token_id\":" + invalid + "}");
            assertThrows(org.eclipse.deeplearning4j.llm.tokenizer.TokenizerException.class,
                    () -> HuggingFaceTokenizer.fromFile(file.toFile()), invalid);
        }
        java.nio.file.Files.writeString(generation, "{\"bos_token_id\":null}");
        try (var tokenizer = HuggingFaceTokenizer.fromFile(file.toFile())) {
            assertEquals(-1, tokenizer.getBosTokenId());
        }
    }

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
    void deployedExternalTemplateBosMetadataDiscrimination() throws Exception {
        File root = new File(java.util.Objects.requireNonNull(System.getProperty("gemma.assets")));
        String template = java.nio.file.Files.readString(java.nio.file.Path.of(
                java.util.Objects.requireNonNull(System.getProperty("gemma.externalTemplate"))));
        GGMLMetadata.TokenizerInfo info;
        try (GGUFReader reader = new GGUFReader(new File(root, "gemma-4-E2B-it-Q4_K_M.gguf"))) {
            info = GGMLMetadata.TokenizerInfo.fromGGUFHeader(reader.getHeader());
        }
        assertTrue(info.getBosTokenId() >= 0, "Source container must declare BOS for this diagnostic");
        try (var tokenizer = HuggingFaceTokenizer.fromFile(new File(root, "tokenizer.json"))) {
            assertFalse(new File(root, "tokenizer_config.json").exists(), "Diagnostic expects exact missing-config deployment");
            assertEquals(info.getBosTokenId(), tokenizer.getBosTokenId());
            String bos = info.getTokens().get(info.getBosTokenId());
            assertEquals(Integer.valueOf(info.getBosTokenId()), tokenizer.getTokenId(bos));
            var metadata = GenerationPipeline.ModelMetadata.of(info.getBosTokenId(), info.getEosTokenId(),
                    info.getPadTokenId(), template, Set.of(), Set.of());
            for (boolean withTools : List.of(false, true)) {
                for (boolean withMetadata : List.of(false, true)) {
                    var request = ChatTemplate.Request.builder()
                            .messages(List.of(ChatTemplate.Message.system("Extract the organizations."),
                                    ChatTemplate.Message.user("Acme is an organization.")))
                            .tools(withTools ? List.of(TOOL) : List.of())
                            .toolCallFormat(ChatTemplate.ToolCallFormat.GEMMA)
                            .templateArguments(GenerationPipeline.chatTemplateArguments(Map.of(),
                                    withMetadata ? metadata : GenerationPipeline.ModelMetadata.empty(), tokenizer)).build();
                    String rendered = tokenizer.applyChatTemplate(request, template);
                    int[] ids = tokenizer.ensureLeadingBos(tokenizer.encode(rendered, false)).getIds();
                    assertEquals(rendered, tokenizer.decode(ids, false));
                    assertTrue(rendered.startsWith(bos));
                    assertEquals(info.getBosTokenId(), ids[0]);
                    assertEquals(1L,
                            java.util.Arrays.stream(ids).filter(id -> id == info.getBosTokenId()).count());
                    System.out.println("DEPLOYED_BOS tools=" + withTools + " metadata=" + withMetadata
                            + " tokenizerBos=" + tokenizer.getBosTokenId() + " ggufBos=" + info.getBosTokenId()
                            + " firstId=" + ids[0] + " decoded=" + tokenizer.decode(ids, false));
                }
            }
        }
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
                if (key.startsWith("gemma") || key.equals("general.architecture")
                        || (key.startsWith("tokenizer.") && (value instanceof Number || value instanceof Boolean))) {
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
            assertEquals(Set.of(1, 106, 50), tokenizer.getGenerationStopTokenIds(),
                    "published generation config must preserve all model-owned terminals");
            var inherited = GenerationPipelineConfig.builder().build();
            assertEquals(Set.of(1, 106, 50), GenerationPipeline.buildStopTokenIds(
                    106, inherited, metadata, tokenizer, Set.of(106)));
            var optedOut = GenerationPipelineConfig.builder()
                    .inheritModelStopTokenIds(false).inheritChatTemplateStopTokenIds(false).build();
            assertEquals(Set.of(106), GenerationPipeline.buildStopTokenIds(
                    106, optedOut, metadata, tokenizer, Set.of(106)),
                    "explicit primary EOS survives while model/template inheritance is disabled");
            assertTrue(GenerationPipeline.buildStopTokenIds(
                    -1, optedOut, metadata, tokenizer, Set.of(106)).isEmpty());
            var explicit = GenerationPipeline.chatTemplateArguments(Map.of("bos_token", "custom"), metadata, tokenizer);
            assertEquals("custom", explicit.get("bos_token"));
        }
    }
}
