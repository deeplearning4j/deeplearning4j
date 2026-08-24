/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class VlmOutputProtocolRegistryTest {
    @Test
    void doctagsProtocolOwnsPromptRenderingAndStructuralCompletion(@TempDir Path dir) throws Exception {
        installExample(dir, "smoldocling-doctags.json");
        VlmOutputProtocolRegistry registry = VlmOutputProtocolRegistry.load(dir.toFile());
        StubTokenizer tokenizer = new StubTokenizer();
        VlmProtocolRequest request = VlmProtocolRequest.builder()
                .renderFormat(VlmRenderFormat.MARKDOWN).build();

        VlmProtocolPlan plan = registry.prepare(request, tokenizer, null);
        assertEquals("Convert this page to docling.", plan.getPrompt());
        assertTrue(plan.getStops().isEmpty(),
                "DocTags close is structural content, not a hard-coded stop");

        GenerationResult result = GenerationResult.eos(
                "<doctag><paragraph>Hello</paragraph></doctag>", new int[]{1}, 3, 1);
        VlmProtocolOutput output = registry.process(request, plan, result, tokenizer);
        assertTrue(output.getCompletion().isComplete());
        assertTrue(output.getRenderedText().contains("Hello"));
    }

    @Test
    void qwenAndGotResolveAtomicAndMultiTokenTurnStops(@TempDir Path dir) throws Exception {
        StubTokenizer tokenizer = new StubTokenizer();
        tokenizer.tokens.put("<|im_end|>", 151645);
        installExample(dir, "qwen-chat.json");
        VlmProtocolPlan qwen = VlmOutputProtocolRegistry.load(dir.toFile())
                .prepare(VlmProtocolRequest.builder().build(), tokenizer, null);
        assertArrayEquals(new int[]{151645}, qwen.getStops().get(0).getTokenIds());
        assertTrue(qwen.isInheritModelEos());
        assertTrue(qwen.isInheritChatTemplateStops());

        tokenizer.tokens.clear();
        installExample(dir, "got-ocr.json");
        VlmProtocolPlan got = VlmOutputProtocolRegistry.load(dir.toFile())
                .prepare(VlmProtocolRequest.builder().build(), tokenizer, null);
        assertArrayEquals(new int[]{7, 8}, got.getStops().get(0).getTokenIds(),
                "non-atomic turn markers must remain ordered stop sequences");
        GenerationResult stopped = GenerationResult.builder().text("body<|im_end|>")
                .tokenIds(new int[]{42, 7, 8}).generatedTokenCount(3)
                .promptTokenCount(1).totalTokenCount(4)
                .finishReason(GenerationResult.FinishReason.STOP_SEQUENCE).build();
        VlmProtocolRequest request = VlmProtocolRequest.builder().build();
        VlmOutputProtocolRegistry gotRegistry = VlmOutputProtocolRegistry.load(dir.toFile());
        VlmProtocolOutput stripped = gotRegistry.process(request, got, stopped, tokenizer);
        assertEquals("decoded", stripped.getRenderedText());

        GenerationResult merged = gotRegistry.mergeRegions(
                request, got, List.of(stopped, stopped), tokenizer);
        assertArrayEquals(new int[]{42, 42}, merged.getTokenIds(),
                "provider-consumed DROP_MATCH suffixes must not survive region merging");
        assertEquals("decoded\ndecoded",
                gotRegistry.process(request, got, merged, tokenizer).getRenderedText(),
                "processing the merged page must not re-decode stripped region terminators");
    }

    @Test
    void regionMergeBelongsToTheSelectedGrammar(@TempDir Path dir) throws Exception {
        GenerationResult left = GenerationResult.eos("left", new int[]{1}, 2, 1);
        GenerationResult right = GenerationResult.eos("right", new int[]{2}, 2, 1);
        StubTokenizer tokenizer = new StubTokenizer();

        VlmOutputProtocolRegistry fallback = VlmOutputProtocolRegistry.load(dir.toFile());
        VlmProtocolRequest plainRequest = VlmProtocolRequest.builder().build();
        VlmProtocolPlan plainPlan = fallback.prepare(plainRequest, tokenizer, null);
        assertEquals("left\nright", fallback.mergeRegions(
                plainRequest, plainPlan, List.of(left, right), tokenizer).getText());
        GenerationResult truncated = right.toBuilder()
                .finishReason(GenerationResult.FinishReason.MAX_TOKENS).build();
        assertEquals(GenerationResult.FinishReason.MAX_TOKENS, fallback.mergeRegions(
                plainRequest, plainPlan, List.of(left, truncated), tokenizer).getFinishReason());

        installExample(dir, "smoldocling-doctags.json");
        VlmOutputProtocolRegistry doctags = VlmOutputProtocolRegistry.load(dir.toFile());
        VlmProtocolRequest doctagsRequest = VlmProtocolRequest.builder().build();
        VlmProtocolPlan doctagsPlan = doctags.prepare(doctagsRequest, tokenizer, null);
        GenerationResult merged = doctags.mergeRegions(doctagsRequest, doctagsPlan,
                List.of(left.toBuilder().text("<doctag>left</doctag>").build(),
                        right.toBuilder().text("<doctag>right</doctag>").build()), tokenizer);
        assertEquals("<doctag>leftright</doctag>", merged.getText());
    }

    @Test
    void taggedProtocolRequiresAtLeastOneBalancedTag(@TempDir Path dir) throws Exception {
        installExample(dir, "donut-tags.json");
        VlmOutputProtocolRegistry registry = VlmOutputProtocolRegistry.load(dir.toFile());
        StubTokenizer tokenizer = new StubTokenizer();
        VlmProtocolRequest request = VlmProtocolRequest.builder()
                .renderFormat(VlmRenderFormat.RAW).build();
        VlmProtocolPlan plan = registry.prepare(request, tokenizer, null);

        VlmProtocolOutput plain = registry.process(request, plan,
                GenerationResult.eos("ordinary text", new int[]{1}, 1, 1), tokenizer);
        assertFalse(plain.getCompletion().isComplete());
        assertFalse(plain.getCompletion().isUsable());

        VlmProtocolOutput tagged = registry.process(request, plan,
                GenerationResult.eos("<s_name>Ada</s_name>", new int[]{1}, 1, 1), tokenizer);
        assertTrue(tagged.getCompletion().isComplete());
    }

    @Test
    void genericProviderRejectsUnsupportedRenderingInsteadOfReturningNativeSyntax(@TempDir Path dir)
            throws Exception {
        installExample(dir, "donut-tags.json");
        VlmOutputProtocolRegistry registry = VlmOutputProtocolRegistry.load(dir.toFile());
        StubTokenizer tokenizer = new StubTokenizer();
        VlmProtocolRequest json = VlmProtocolRequest.builder()
                .renderFormat(VlmRenderFormat.JSON).build();
        VlmProtocolPlan plan = registry.prepare(json, tokenizer, null);
        assertThrows(IllegalArgumentException.class, () -> registry.process(json, plan,
                GenerationResult.eos("<s_name>Ada</s_name>", new int[]{1}, 1, 1), tokenizer));
    }

    @Test
    void manifestValidationRejectsUnknownDefaultsAndExplicitTasks(@TempDir Path dir) throws Exception {
        Files.writeString(dir.resolve(VlmOutputProtocolRegistry.MANIFEST_NAME),
                "{\"schemaVersion\":1,\"defaultProtocol\":\"missing\",\"protocols\":{" +
                        "\"known\":{\"provider\":\"builtin.plain\",\"tasks\":{}," +
                        "\"termination\":{},\"completion\":{},\"output\":{}}}}" );
        assertThrows(java.io.IOException.class,
                () -> VlmOutputProtocolRegistry.load(dir.toFile()));

        installExample(dir, "donut-tags.json");
        VlmOutputProtocolRegistry registry = VlmOutputProtocolRegistry.load(dir.toFile());
        assertThrows(IllegalArgumentException.class, () -> registry.prepare(
                VlmProtocolRequest.builder().task("missing-task").build(), new StubTokenizer(), null));
    }

    @Test
    void manifestCanDisableInheritedStops(@TempDir Path dir) throws Exception {
        Files.writeString(dir.resolve(VlmOutputProtocolRegistry.MANIFEST_NAME),
                "{\"schemaVersion\":1,\"defaultProtocol\":\"strict\",\"protocols\":{" +
                        "\"strict\":{\"provider\":\"builtin.plain\",\"defaultTask\":\"default\"," +
                        "\"tasks\":{\"default\":{\"prompt\":\"\",\"framing\":\"RAW\"}}," +
                        "\"termination\":{\"inheritModelEos\":false," +
                        "\"inheritChatTemplateStops\":false,\"sequences\":[]}," +
                        "\"completion\":{\"required\":false},\"output\":{\"nativeFormat\":\"PLAIN_TEXT\"}}}}" );
        VlmProtocolPlan plan = VlmOutputProtocolRegistry.load(dir.toFile()).prepare(
                VlmProtocolRequest.builder().build(), new StubTokenizer(), null);
        assertFalse(plan.isInheritModelEos());
        assertFalse(plan.isInheritChatTemplateStops());
    }

    @Test
    void everyDocumentedFamilyManifestLoads(@TempDir Path dir) throws Exception {
        for (String example : List.of("smoldocling-doctags.json", "qwen-chat.json", "got-ocr.json",
                "florence2-tasks.json", "donut-tags.json", "nougat-markup.json", "pix2struct.json")) {
            installExample(dir, example);
            VlmOutputProtocolRegistry registry = VlmOutputProtocolRegistry.load(dir.toFile());
            assertNotNull(registry.resolve(null), example);
        }
    }

    @Test
    void missingManifestFallsBackButExplicitUnknownProtocolFails(@TempDir Path dir) throws Exception {
        VlmOutputProtocolRegistry registry = VlmOutputProtocolRegistry.load(dir.toFile());
        assertEquals("fallback", registry.resolve(null).id());
        assertThrows(IllegalArgumentException.class, () -> registry.resolve("missing"));
    }

    private static void installExample(Path dir, String name) throws Exception {
        try (InputStream in = VlmOutputProtocolRegistryTest.class.getResourceAsStream(
                "/vlm-protocol-examples/" + name)) {
            assertNotNull(in, name);
            Files.copy(in, dir.resolve(VlmOutputProtocolRegistry.MANIFEST_NAME),
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static final class StubTokenizer implements Tokenizer {
        private final Map<String, Integer> tokens = new LinkedHashMap<>();

        @Override public Encoding encode(String text, boolean addSpecialTokens) {
            int[] ids = "<|im_end|>".equals(text) ? new int[]{7, 8}
                    : text.chars().limit(4).toArray();
            return Encoding.builder().ids(ids).attentionMask(new int[ids.length]).build();
        }
        @Override public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
            List<Encoding> result = new ArrayList<>();
            for (String text : texts) result.add(encode(text, addSpecialTokens));
            return result;
        }
        @Override public String decode(int[] ids, boolean skipSpecialTokens) { return "decoded"; }
        @Override public List<String> decodeBatch(List<int[]> ids, boolean skipSpecialTokens) {
            return Collections.nCopies(ids.size(), "decoded");
        }
        @Override public int getVocabSize() { return 200000; }
        @Override public Integer getTokenId(String token) { return tokens.get(token); }
        @Override public String getToken(int id) { return null; }
        @Override public Map<String, Integer> getVocab() { return tokens; }
        @Override public int getPadTokenId() { return -1; }
        @Override public int getBosTokenId() { return -1; }
        @Override public int getEosTokenId() { return 2; }
        @Override public int getUnkTokenId() { return -1; }
        @Override public boolean isValid() { return true; }
        @Override public void close() { }
    }
}
