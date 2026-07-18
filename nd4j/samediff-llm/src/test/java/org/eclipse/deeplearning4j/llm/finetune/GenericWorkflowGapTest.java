package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class GenericWorkflowGapTest {
    @Test
    void splitIsOrderIndependentGroupedAndRejectsDuplicates() {
        List<GeneratedTrainingExample> examples = Arrays.asList(example("b", "g1"), example("a", "g1"),
                example("c", "g2"), example("d", "g3"));
        DatasetSplitConfig config = new DatasetSplitConfig(0.5, 0.25, 0.25, "seed");
        DatasetSplit first = DeterministicDatasetSplitter.split(examples, config,
                e -> String.valueOf(e.getMetadata().get("group")));
        List<GeneratedTrainingExample> reversed = new ArrayList<>(examples);
        Collections.reverse(reversed);
        DatasetSplit second = DeterministicDatasetSplitter.split(reversed, config,
                e -> String.valueOf(e.getMetadata().get("group")));
        assertEquals(ids(first.getTrain()), ids(second.getTrain()));
        assertEquals(location(first, "a"), location(first, "b"));
        assertThrows(IllegalArgumentException.class, () ->
                DeterministicDatasetSplitter.split(Arrays.asList(examples.get(0), examples.get(0)),
                        config, null));
    }

    @Test
    void manifestFingerprintAndJsonAreStable() {
        List<GeneratedTrainingExample> examples = Arrays.asList(example("b", "g"), example("a", "g"));
        List<GeneratedTrainingExample> reverse = new ArrayList<>(examples);
        Collections.reverse(reverse);
        assertEquals(FineTuneManifest.fingerprint(examples), FineTuneManifest.fingerprint(reverse));
        FineTuneManifest manifest = new FineTuneManifest();
        manifest.setCorpusFingerprint(FineTuneManifest.fingerprint(examples));
        manifest.setTokenizerIdentity("tokenizer@1");
        manifest.setSequenceLength(128);
        manifest.setStudentConfig(Collections.singletonMap("layers", 2));
        assertEquals(manifest.toJson(), manifest.toJson());
    }

    @Test
    void rejectionAuditFallbackAndResume(@TempDir Path temp) throws Exception {
        OfflineTeacherDataGenerator generator = new OfflineTeacherDataGenerator(
                (prompt, max) -> "bad", null,
                Collections.singletonList((request, output) -> TeacherValidationResult.reject("invalid")), 8);
        TeacherExampleRequest request = new TeacherExampleRequest();
        request.setId("one");
        request.setContext(Collections.singletonMap("prompt", "value"));
        File output = temp.resolve("accepted.jsonl").toFile();
        File rejected = temp.resolve("rejected.jsonl").toFile();

        TeacherGenerationReport report = new TeacherGenerationJob(generator).run(
                Collections.singletonList(request), output, rejected, (source, reason) -> {
                    GeneratedTrainingExample fallback = new GeneratedTrainingExample();
                    fallback.setId(source.getId());
                    fallback.setPrompt("prompt");
                    fallback.setResponse("fallback");
                    return fallback;
                }, "run-1");
        assertEquals(1, report.getFallbackCount());
        assertEquals("fallback", FineTuneJsonl.read(output, GeneratedTrainingExample.class).get(0).getResponse());
        assertFalse(rejected.exists());
        assertEquals(1, new TeacherGenerationJob(generator)
                .run(Collections.singletonList(request), output).getSkipped());

        TeacherExampleRequest noFallback = new TeacherExampleRequest();
        noFallback.setId("two");
        noFallback.setContext(Collections.emptyMap());
        new TeacherGenerationJob(generator).run(Collections.singletonList(noFallback),
                output, rejected, null, "run-2");
        assertEquals("two", FineTuneJsonl.read(rejected, TeacherGenerationRejection.class).get(0).getRequestId());
    }

    @Test
    void truncationPoliciesReportAndPreserveLatestTurn() {
        GeneratedTrainingExample longExample = example("long", "g");
        longExample.setMessages(Arrays.asList(
                new FineTuneMessage("system", "system"),
                new FineTuneMessage("user", "old old old old"),
                new FineTuneMessage("assistant", "old answer answer"),
                new FineTuneMessage("user", "latest"),
                new FineTuneMessage("assistant", "latest answer")));
        assertThrows(IllegalArgumentException.class, () ->
                new ResponseMaskedDatasetBuilder(new FakeTokenizer(), 12,
                        TruncationPolicy.REJECT, false).tokenize(longExample));
        ResponseMaskedTokens tokens = new ResponseMaskedDatasetBuilder(new FakeTokenizer(), 32,
                TruncationPolicy.DROP_OLDEST_TURNS, false).tokenize(longExample);
        assertTrue(tokens.isTruncated());
        assertTrue(tokens.getDroppedMessageCount() > 0);
        assertFalse(tokens.isPartialAssistant());
    }

    private static GeneratedTrainingExample example(String id, String group) {
        GeneratedTrainingExample e = new GeneratedTrainingExample();
        e.setId(id); e.setPrompt("prompt"); e.setResponse("response");
        e.setMetadata(Collections.singletonMap("group", group));
        return e;
    }

    private static Set<String> ids(List<GeneratedTrainingExample> values) {
        Set<String> result = new LinkedHashSet<>();
        for (GeneratedTrainingExample value : values) result.add(value.getId());
        return result;
    }

    private static int location(DatasetSplit split, String id) {
        if (ids(split.getTrain()).contains(id)) return 0;
        if (ids(split.getValidation()).contains(id)) return 1;
        return 2;
    }

    private static final class FakeTokenizer implements Tokenizer {
        private final Map<String, Integer> vocab = new LinkedHashMap<>();
        @Override public Encoding encode(String text, boolean special) {
            String[] parts = text.split("(?<=\\W)|(?=\\W)");
            int[] ids = new int[parts.length];
            for (int i = 0; i < parts.length; i++) ids[i] = vocab.computeIfAbsent(parts[i], k -> vocab.size() + 1);
            return Encoding.builder().ids(ids).tokens(parts).attentionMask(new int[ids.length])
                    .typeIds(new int[ids.length]).build();
        }
        @Override public String applyChatTemplate(List<ChatTemplate.Message> messages, boolean generation) {
            StringBuilder b = new StringBuilder();
            for (ChatTemplate.Message m : messages) b.append('<').append(m.getRole()).append('>')
                    .append(m.getContent()).append("</").append(m.getRole()).append('>');
            if (generation) b.append("<assistant>");
            return b.toString();
        }
        @Override public List<Encoding> encodeBatch(List<String> texts, boolean special) { List<Encoding> r=new ArrayList<>(); for(String t:texts)r.add(encode(t,special)); return r; }
        @Override public String decode(int[] ids, boolean skip) { return ""; }
        @Override public List<String> decodeBatch(List<int[]> ids, boolean skip) { return Collections.emptyList(); }
        @Override public int getVocabSize() { return 1000; }
        @Override public Integer getTokenId(String token) { return vocab.get(token); }
        @Override public String getToken(int id) { return ""; }
        @Override public Map<String,Integer> getVocab() { return vocab; }
        @Override public int getPadTokenId() { return 0; }
        @Override public int getBosTokenId() { return -1; }
        @Override public int getEosTokenId() { return -1; }
        @Override public int getUnkTokenId() { return -1; }
        @Override public boolean isValid() { return true; }
        @Override public void close() {}
    }
}
