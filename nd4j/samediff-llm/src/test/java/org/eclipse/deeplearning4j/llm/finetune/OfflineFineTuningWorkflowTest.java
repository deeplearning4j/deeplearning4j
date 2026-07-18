package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class OfflineFineTuningWorkflowTest {

    @Test
    void canonicalSerializationIsStableAcrossMapInsertionOrder() {
        Map<String, Object> first = new LinkedHashMap<>();
        first.put("z", 1);
        first.put("a", Collections.singletonMap("b", 2));
        Map<String, Object> second = new LinkedHashMap<>();
        second.put("a", Collections.singletonMap("b", 2));
        second.put("z", 1);

        CanonicalContextSerializer serializer = new CanonicalContextSerializer();
        assertEquals(serializer.serialize(first), serializer.serialize(second));
    }

    @Test
    void teacherGenerationCanonicalizesCleansAndValidates() {
        TeacherExampleRequest request = request();
        OfflineTeacherDataGenerator generator = new OfflineTeacherDataGenerator(
                (prompt, max) -> "Response: grounded answer",
                null,
                Collections.singletonList((r, output) -> output.contains("grounded")
                        ? TeacherValidationResult.accept() : TeacherValidationResult.reject("not grounded")),
                16);

        GeneratedTrainingExample example = generator.generate(request);
        assertEquals("grounded answer", example.getResponse());
        assertTrue(example.getPrompt().indexOf("\"a\"") < example.getPrompt().indexOf("\"z\""));
        assertEquals(request.getContext(), example.getContext());
    }

    @Test
    void rejectedTeacherOutputFailsWithReasons() {
        OfflineTeacherDataGenerator generator = new OfflineTeacherDataGenerator(
                (prompt, max) -> "bad",
                null,
                Collections.singletonList((request, output) -> TeacherValidationResult.reject("unsafe")),
                8);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> generator.generate(request()));
        assertTrue(error.getMessage().contains("unsafe"));
    }

    @Test
    void jsonlRoundTripIsStrict() throws Exception {
        GeneratedTrainingExample example = new GeneratedTrainingExample();
        example.setId("one");
        example.setPrompt("prompt");
        example.setResponse("response");

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        FineTuneJsonl.write(output, Collections.singletonList(example));
        List<GeneratedTrainingExample> restored = FineTuneJsonl.read(
                new ByteArrayInputStream(output.toByteArray()), GeneratedTrainingExample.class);
        assertEquals(1, restored.size());
        assertEquals("response", restored.get(0).getResponse());

        assertThrows(Exception.class, () -> FineTuneJsonl.read(
                new ByteArrayInputStream("{bad}\n".getBytes(StandardCharsets.UTF_8)),
                GeneratedTrainingExample.class));
    }

    @Test
    void responseMaskOnlyCoversAssistantTokens() {
        GeneratedTrainingExample example = new GeneratedTrainingExample();
        example.setId("mask");
        example.setSystemPrompt("system");
        example.setPrompt("user prompt");
        example.setResponse("assistant answer");

        ResponseMaskedTokens tokens =
                new ResponseMaskedDatasetBuilder(new FakeTokenizer(), 32).tokenize(example);
        float[] mask = tokens.getLossMask();
        int first = -1;
        int count = 0;
        for (int i = 0; i < mask.length; i++) {
            if (mask[i] > 0) {
                if (first < 0) first = i;
                count++;
            }
        }
        assertTrue(first > 0);
        assertTrue(count > 0);
        for (int i = 0; i < first; i++) assertEquals(0.0f, mask[i]);
    }

    @Test
    void multiTurnMaskCoversEachAssistantTurn() {
        GeneratedTrainingExample example = new GeneratedTrainingExample();
        example.setId("multi");
        example.setMessages(Arrays.asList(
                new FineTuneMessage("system", "system"),
                new FineTuneMessage("user", "first question"),
                new FineTuneMessage("assistant", "first answer"),
                new FineTuneMessage("user", "second question"),
                new FineTuneMessage("assistant", "second answer")));

        float[] mask = new ResponseMaskedDatasetBuilder(new FakeTokenizer(), 64)
                .tokenize(example).getLossMask();
        int runs = 0;
        boolean active = false;
        for (float value : mask) {
            if (value > 0 && !active) runs++;
            active = value > 0;
        }
        assertEquals(2, runs);
    }

    @Test
    void generationJobResumesAndReportsRejections(@TempDir Path tempDir) throws Exception {
        TeacherExampleRequest accepted = request();
        TeacherExampleRequest rejected = request();
        rejected.setId("request-2");

        OfflineTeacherDataGenerator generator = new OfflineTeacherDataGenerator(
                (prompt, max) -> prompt.contains("\"reject\"") ? "bad" : "grounded",
                null,
                Collections.singletonList((request, output) -> "bad".equals(output)
                        ? TeacherValidationResult.reject("rejected")
                        : TeacherValidationResult.accept()),
                8);
        rejected.getContext().put("reject", true);
        File output = tempDir.resolve("examples.jsonl").toFile();

        TeacherGenerationReport first =
                new TeacherGenerationJob(generator).run(Arrays.asList(accepted, rejected), output);
        assertEquals(1, first.getAccepted().size());
        assertEquals(1, first.getRejected().size());
        assertEquals(0, first.getSkipped());

        TeacherGenerationReport resumed =
                new TeacherGenerationJob(generator).run(Collections.singletonList(accepted), output);
        assertEquals(0, resumed.getAccepted().size());
        assertEquals(1, resumed.getSkipped());
        assertEquals(1, FineTuneJsonl.read(output, GeneratedTrainingExample.class).size());
    }

    private static TeacherExampleRequest request() {
        TeacherExampleRequest request = new TeacherExampleRequest();
        request.setId("request-1");
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("z", 1);
        context.put("a", "value");
        request.setContext(context);
        return request;
    }

    private static final class FakeTokenizer implements Tokenizer {
        private final Map<String, Integer> vocabulary = new LinkedHashMap<>();

        @Override public Encoding encode(String text, boolean addSpecialTokens) {
            List<String> pieces = new ArrayList<>();
            for (String part : text.split("(?<=\\W)|(?=\\W)")) {
                if (!part.isEmpty()) pieces.add(part);
            }
            int[] ids = new int[pieces.size()];
            int[] mask = new int[pieces.size()];
            for (int i = 0; i < pieces.size(); i++) {
                ids[i] = vocabulary.computeIfAbsent(pieces.get(i), key -> vocabulary.size() + 1);
                mask[i] = 1;
            }
            return Encoding.builder().ids(ids).tokens(pieces.toArray(new String[0]))
                    .attentionMask(mask).typeIds(new int[ids.length]).build();
        }

        @Override public String applyChatTemplate(List<ChatTemplate.Message> messages, boolean generationPrompt) {
            StringBuilder result = new StringBuilder();
            for (ChatTemplate.Message message : messages) {
                result.append('<').append(message.getRole()).append('>')
                        .append(message.getContent()).append("</").append(message.getRole()).append('>');
            }
            if (generationPrompt) result.append("<assistant>");
            return result.toString();
        }

        @Override public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
            List<Encoding> result = new ArrayList<>();
            for (String text : texts) result.add(encode(text, addSpecialTokens));
            return result;
        }
        @Override public String decode(int[] ids, boolean skipSpecialTokens) { return ""; }
        @Override public List<String> decodeBatch(List<int[]> idsBatch, boolean skipSpecialTokens) { return Collections.emptyList(); }
        @Override public int getVocabSize() { return 1000; }
        @Override public Integer getTokenId(String token) { return vocabulary.get(token); }
        @Override public String getToken(int id) { return ""; }
        @Override public Map<String, Integer> getVocab() { return vocabulary; }
        @Override public int getPadTokenId() { return 0; }
        @Override public int getBosTokenId() { return -1; }
        @Override public int getEosTokenId() { return -1; }
        @Override public int getUnkTokenId() { return -1; }
        @Override public boolean isValid() { return true; }
        @Override public void close() { }
    }
}
