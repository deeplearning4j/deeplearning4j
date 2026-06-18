package org.eclipse.deeplearning4j.nd4j.linalg.dataset.curation;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.curation.batching.LengthBucketingIterator;
import org.nd4j.linalg.dataset.curation.curriculum.CurriculumIterator;
import org.nd4j.linalg.dataset.curation.decontamination.DecontaminationResult;
import org.nd4j.linalg.dataset.curation.decontamination.NGramDecontaminator;
import org.nd4j.linalg.dataset.curation.dedup.MinHasher;
import org.nd4j.linalg.dataset.curation.dedup.TextDeduplicator;
import org.nd4j.linalg.dataset.curation.filtering.FilterResult;
import org.nd4j.linalg.dataset.curation.filtering.TextQualityFilter;
import org.nd4j.linalg.dataset.curation.format.ChatTemplate;
import org.nd4j.linalg.dataset.curation.format.ConversationTurn;
import org.nd4j.linalg.dataset.curation.format.FormattedExample;
import org.nd4j.linalg.dataset.curation.format.InstructionDataFormatter;
import org.nd4j.linalg.dataset.curation.mixing.WeightedDataMixer;
import org.nd4j.linalg.dataset.curation.packing.PackedSequence;
import org.nd4j.linalg.dataset.curation.packing.SequencePacker;
import org.nd4j.linalg.dataset.curation.splitting.SplitResult;
import org.nd4j.linalg.dataset.curation.splitting.StratifiedSplitter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestDatasetCuration {

    @Test
    public void testTextQualityFilter() {
        TextQualityFilter filter = TextQualityFilter.builder()
                .minChars(10)
                .maxChars(1000)
                .minWords(3)
                .minAlphaRatio(0.5)
                .maxSpecialCharRatio(0.3)
                .maxRepetitionRatio(0.5)
                .repetitionNgramSize(2)
                .maxAllCapsRatio(0.8)
                .build();

        // Accept good text
        assertTrue(filter.accept("This is a perfectly normal English sentence with reasonable length."));

        // Reject too short
        assertFalse(filter.accept("Hi"));

        // Reject low alpha ratio
        assertFalse(filter.accept("123 456 789 012 345 678 901 234 567 890"));

        // Detailed evaluation
        FilterResult result = filter.evaluate("Hi");
        assertFalse(result.isAccepted());
        assertFalse(result.getRejectionReasons().isEmpty());

        // Repetition detection
        String repetitive = "spam spam spam spam spam spam spam spam spam spam spam spam";
        FilterResult repResult = filter.evaluate(repetitive);
        assertFalse(repResult.isAccepted());
        assertTrue(repResult.getRejectionReasons().stream().anyMatch(r -> r.contains("repetition")));
    }

    @Test
    public void testExactDeduplication() {
        TextDeduplicator dedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(false)
                .build();

        List<String> texts = Arrays.asList(
                "Hello world",
                "Goodbye world",
                "Hello world",  // exact duplicate
                "Something unique",
                "hello world"   // different case = exact dup after normalization
        );

        List<String> unique = dedup.deduplicate(texts);
        assertEquals(3, unique.size());
        assertTrue(unique.contains("Hello world"));
        assertTrue(unique.contains("Goodbye world"));
        assertTrue(unique.contains("Something unique"));

        Set<Integer> dupIndices = dedup.findDuplicateIndices(texts);
        assertTrue(dupIndices.contains(2));
        assertTrue(dupIndices.contains(4));
    }

    @Test
    public void testMinHashDeduplication() {
        TextDeduplicator dedup = TextDeduplicator.builder()
                .useExact(false)
                .useMinHash(true)
                .shingleSize(3)
                .numBands(32)
                .rowsPerBand(4)
                .jaccardThreshold(0.5)
                .build();

        // Use longer texts with more shared content for reliable MinHash detection
        List<String> texts = Arrays.asList(
                "The quick brown fox jumps over the lazy dog in the park on a sunny afternoon while birds sing",
                "Something completely different about cats and mice in the house during a rainy evening with thunder",
                "The quick brown fox jumps over the lazy dog in the park on a sunny afternoon while birds chirp" // near-duplicate
        );

        Set<Integer> dupIndices = dedup.findDuplicateIndices(texts);
        // The third text should be flagged as near-duplicate of the first
        assertTrue(dupIndices.contains(2), "Near-duplicate should be detected");
        assertFalse(dupIndices.contains(1), "Completely different text should not be flagged");

        // Test streaming mode
        TextDeduplicator streamDedup = TextDeduplicator.builder()
                .useExact(true)
                .useMinHash(false)
                .build();
        assertTrue(streamDedup.addIfUnique("First text"));
        assertTrue(streamDedup.addIfUnique("Second text"));
        assertFalse(streamDedup.addIfUnique("first text")); // duplicate after normalization
    }

    @Test
    public void testInstructionFormattingChatML() {
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        FormattedExample result = formatter.format("What is 2+2?", "4");

        String text = result.getText();
        assertTrue(text.contains("<|im_start|>user"));
        assertTrue(text.contains("What is 2+2?"));
        assertTrue(text.contains("<|im_start|>assistant"));
        assertTrue(text.contains("4"));

        // Verify loss mask: only assistant content should be 1
        int[] mask = result.getLossMask();
        assertEquals(text.length(), mask.length);

        // Find where "4" is in the assistant section
        int assistantContentStart = text.indexOf("<|im_start|>assistant\n") + "<|im_start|>assistant\n".length();
        int assistantContentEnd = text.indexOf("<|im_end|>", assistantContentStart);

        // Content within assistant response should be masked as 1
        for (int i = assistantContentStart; i < assistantContentEnd; i++) {
            assertEquals(1, mask[i], "Position " + i + " should be trainable (assistant content)");
        }

        // User prefix should be masked as 0
        for (int i = 0; i < text.indexOf("What is 2+2?"); i++) {
            assertEquals(0, mask[i], "Position " + i + " should not be trainable (user prefix)");
        }
    }

    @Test
    public void testInstructionFormattingAlpaca() {
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.ALPACA)
                .build();

        FormattedExample result = formatter.format("Translate to French", "Bonjour");

        String text = result.getText();
        assertTrue(text.contains("### Input:"));
        assertTrue(text.contains("Translate to French"));
        assertTrue(text.contains("### Response:"));
        assertTrue(text.contains("Bonjour"));
    }

    @Test
    public void testMultiTurnConversation() {
        InstructionDataFormatter formatter = InstructionDataFormatter.builder()
                .template(ChatTemplate.CHATML)
                .build();

        List<ConversationTurn> conversation = Arrays.asList(
                ConversationTurn.system("You are a helpful assistant."),
                ConversationTurn.user("Hi there!"),
                ConversationTurn.assistant("Hello! How can I help?"),
                ConversationTurn.user("What is the capital of France?"),
                ConversationTurn.assistant("Paris is the capital of France.")
        );

        FormattedExample result = formatter.format(conversation);
        String text = result.getText();
        int[] mask = result.getLossMask();

        // Both assistant responses should have trainable content
        assertTrue(text.contains("Hello! How can I help?"));
        assertTrue(text.contains("Paris is the capital of France."));

        // Count trainable characters
        int trainable = 0;
        for (int m : mask) trainable += m;
        assertTrue(trainable > 0, "Should have trainable characters");

        // Trainable chars should equal the length of assistant content
        int expectedTrainable = "Hello! How can I help?".length() + "Paris is the capital of France.".length();
        assertEquals(expectedTrainable, trainable);
    }

    @Test
    public void testSequencePacking() {
        SequencePacker packer = new SequencePacker(99, 0); // separator=99, padding=0

        List<int[]> sequences = Arrays.asList(
                new int[]{1, 2, 3},       // len 3
                new int[]{4, 5},           // len 2
                new int[]{6, 7, 8, 9},    // len 4
                new int[]{10, 11}          // len 2
        );

        List<PackedSequence> packed = packer.pack(sequences, 10);

        // All sequences should be packed
        int totalTokens = 0;
        for (PackedSequence ps : packed) {
            for (int i = 0; i < ps.length(); i++) {
                if (ps.getTokens()[i] != 0 && ps.getTokens()[i] != 99) {
                    totalTokens++;
                }
            }
        }
        assertEquals(11, totalTokens); // 3+2+4+2 = 11 original tokens

        // Verify segment IDs are set
        for (PackedSequence ps : packed) {
            assertNotNull(ps.getSegmentIds());
            assertTrue(ps.numSegments() >= 1);
        }

        // Verify attention masks are block-diagonal
        for (PackedSequence ps : packed) {
            INDArray attn = ps.getAttentionMask();
            assertNotNull(attn);
            assertEquals(2, attn.rank());
            assertEquals(10, attn.size(0));
            assertEquals(10, attn.size(1));

            // Cross-segment attention should be 0
            int[] segIds = ps.getSegmentIds();
            for (int i = 0; i < ps.length(); i++) {
                for (int j = 0; j < ps.length(); j++) {
                    if (segIds[i] >= 0 && segIds[j] >= 0 && segIds[i] != segIds[j]) {
                        assertEquals(0.0, attn.getDouble(i, j), 1e-6,
                                "Cross-segment attention should be 0");
                    }
                }
            }
        }
    }

    @Test
    public void testLengthBucketing() {
        List<String> data = Arrays.asList(
                "short",                              // 5
                "a medium length string here",        // 28
                "tiny",                                // 4
                "another medium one for testing",     // 30
                "this is a somewhat longer string for testing purposes in our system", // 67
                "hi"                                   // 2
        );

        LengthBucketingIterator<String> iter = LengthBucketingIterator.<String>builder()
                .bucketBoundaries(10, 40, 100)
                .fixedBatchSize(2)
                .lengthFunction(String::length)
                .shuffle(42L)
                .build(data);

        List<List<String>> batches = new ArrayList<>();
        while (iter.hasNext()) {
            batches.add(iter.next());
        }

        // All data should be returned
        int totalItems = 0;
        for (List<String> batch : batches) {
            totalItems += batch.size();
            // Each batch should have items of similar length (same bucket)
            if (batch.size() > 1) {
                int minLen = batch.stream().mapToInt(String::length).min().orElse(0);
                int maxLen = batch.stream().mapToInt(String::length).max().orElse(0);
                // Both should fall in the same bucket
                assertTrue(maxLen - minLen < 40, "Bucket items should have similar lengths");
            }
        }
        assertEquals(6, totalItems);
    }

    @Test
    public void testWeightedDataMixer() {
        // Use large sources so neither exhausts before we can observe weight effects
        List<String> source1 = new ArrayList<>();
        List<String> source2 = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            source1.add("a" + i);
            source2.add("b" + i);
        }

        Map<String, WeightedDataMixer.WeightedSource<String>> sources = new LinkedHashMap<>();
        sources.put("sourceA", new WeightedDataMixer.WeightedSource<>(source1.iterator(), 0.8));
        sources.put("sourceB", new WeightedDataMixer.WeightedSource<>(source2.iterator(), 0.2));

        WeightedDataMixer<String> mixer = new WeightedDataMixer<>(sources, 1.0, 42L);

        // Draw only 200 samples (both sources have 500 each, so neither exhausts)
        int drawCount = 200;
        int countA = 0, countB = 0;
        for (int i = 0; i < drawCount && mixer.hasNext(); i++) {
            String item = mixer.next();
            if (item.startsWith("a")) countA++;
            else countB++;
        }

        assertEquals(drawCount, countA + countB);

        // With 0.8/0.2 weights and 200 draws, sourceA should clearly dominate
        // Expected: ~160 A, ~40 B. Allow wide margin but A must be > B.
        assertTrue(countA > countB, "Source A (weight 0.8) should get more samples than source B (weight 0.2), got A=" + countA + " B=" + countB);
        // Also verify stats tracking works
        Map<String, Long> stats = mixer.getStats();
        assertTrue(stats.containsKey("sourceA"));
        assertTrue(stats.containsKey("sourceB"));
    }

    @Test
    public void testStratifiedSplitter() {
        // Create labeled data: 60 items with label "A", 40 with label "B"
        List<String> data = new ArrayList<>();
        for (int i = 0; i < 60; i++) data.add("A_" + i);
        for (int i = 0; i < 40; i++) data.add("B_" + i);

        StratifiedSplitter<String> splitter = new StratifiedSplitter<>(42L);

        // Stratified split
        SplitResult<String> result = splitter.splitStratified(data, 0.8, 0.1, s -> s.substring(0, 1));

        // Check sizes are reasonable
        assertEquals(100, result.getTrain().size() + result.getValidation().size() + result.getTest().size());

        // Count class proportions in each split
        long trainA = result.getTrain().stream().filter(s -> s.startsWith("A")).count();
        long trainB = result.getTrain().stream().filter(s -> s.startsWith("B")).count();
        long valA = result.getValidation().stream().filter(s -> s.startsWith("A")).count();
        long valB = result.getValidation().stream().filter(s -> s.startsWith("B")).count();

        // Proportions should be roughly 60/40 in each split
        if (result.getTrain().size() > 0) {
            double trainRatioA = (double) trainA / result.getTrain().size();
            assertTrue(trainRatioA > 0.4 && trainRatioA < 0.8,
                    "Train split should roughly preserve class proportions, got " + trainRatioA);
        }

        // Simple random split
        SplitResult<String> randomResult = splitter.split(data, 0.8, 0.1);
        assertEquals(80, randomResult.getTrain().size());
        assertEquals(10, randomResult.getValidation().size());
        assertEquals(10, randomResult.getTest().size());
    }

    @Test
    public void testNGramDecontamination() {
        // Use small n-gram size for testing
        NGramDecontaminator decontaminator = new NGramDecontaminator(3); // 3-gram for testing

        // Benchmark data
        List<String> benchmark = Arrays.asList(
                "The quick brown fox jumps over the lazy dog",
                "To be or not to be that is the question"
        );
        decontaminator.indexBenchmark(benchmark);

        // Training data with some contaminated samples
        List<String> training = Arrays.asList(
                "A completely original sentence about machine learning",
                "The quick brown fox jumps over the lazy dog in the park", // contains benchmark n-grams
                "Another unique training example for deep learning",
                "To be or not to be that is the question indeed" // contains benchmark n-grams
        );

        DecontaminationResult result = decontaminator.check(training);

        assertTrue(result.getContaminatedIndices().contains(1), "Should detect contamination in index 1");
        assertTrue(result.getContaminatedIndices().contains(3), "Should detect contamination in index 3");
        assertFalse(result.getContaminatedIndices().contains(0), "Index 0 should be clean");
        assertFalse(result.getContaminatedIndices().contains(2), "Index 2 should be clean");
        assertEquals(4, result.getTotalChecked());

        // Decontaminate
        List<String> clean = decontaminator.decontaminate(training);
        assertEquals(2, clean.size());
        assertTrue(clean.get(0).contains("machine learning"));
        assertTrue(clean.get(1).contains("deep learning"));
    }

    @Test
    public void testCurriculumOrdering() {
        List<String> data = Arrays.asList(
                "short",                                // len 5
                "medium length text",                   // len 18
                "a",                                    // len 1
                "this is a much longer sentence here",  // len 35
                "hi"                                    // len 2
        );

        // Easy-to-hard (length-based)
        CurriculumIterator<String> easyToHard = new CurriculumIterator<>(
                data, CurriculumIterator.lengthScorer(),
                CurriculumIterator.Strategy.EASY_TO_HARD
        );

        List<String> ordered = easyToHard.getOrderedData();
        assertEquals(5, ordered.size());

        // Verify easy-to-hard ordering by length
        for (int i = 1; i < ordered.size(); i++) {
            assertTrue(ordered.get(i).length() >= ordered.get(i - 1).length(),
                    "Items should be ordered easy (short) to hard (long)");
        }

        // Hard-to-easy
        CurriculumIterator<String> hardToEasy = new CurriculumIterator<>(
                data, CurriculumIterator.lengthScorer(),
                CurriculumIterator.Strategy.HARD_TO_EASY
        );

        List<String> reverseOrdered = hardToEasy.getOrderedData();
        for (int i = 1; i < reverseOrdered.size(); i++) {
            assertTrue(reverseOrdered.get(i).length() <= reverseOrdered.get(i - 1).length(),
                    "Items should be ordered hard (long) to easy (short)");
        }

        // Mixed: should interleave easy and hard
        CurriculumIterator<String> mixed = new CurriculumIterator<>(
                data, CurriculumIterator.lengthScorer(),
                CurriculumIterator.Strategy.MIXED
        );

        List<String> mixedOrdered = mixed.getOrderedData();
        assertEquals(5, mixedOrdered.size());
        // First should be easiest, second should be hardest
        assertEquals("a", mixedOrdered.get(0));
        assertEquals("this is a much longer sentence here", mixedOrdered.get(1));
    }
}
