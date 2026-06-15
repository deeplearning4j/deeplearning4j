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

package org.eclipse.deeplearning4j.llm.eval;

import org.eclipse.deeplearning4j.llm.eval.dataset.CustomDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.dataset.JsonlDataset;
import org.eclipse.deeplearning4j.llm.eval.metrics.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the evaluation framework: metrics, datasets, answer extraction, config, and results.
 */
public class EvalFrameworkTest {

    // ==================== ExactMatchMetric ====================

    @Test
    public void testExactMatchIdentical() {
        ExactMatchMetric metric = new ExactMatchMetric(true);
        assertEquals(1.0, metric.score("Paris", List.of("Paris")), 1e-9);
    }

    @Test
    public void testExactMatchNormalized() {
        ExactMatchMetric metric = new ExactMatchMetric(true);
        // normalize: lowercase, strip punct, remove articles, collapse whitespace
        assertEquals(1.0, metric.score("The answer is: Paris!", List.of("answer is paris")), 1e-9);
    }

    @Test
    public void testExactMatchNoNormalize() {
        ExactMatchMetric metric = new ExactMatchMetric(false);
        assertEquals(0.0, metric.score("paris", List.of("Paris")), 1e-9);
    }

    @Test
    public void testExactMatchMultipleReferences() {
        ExactMatchMetric metric = new ExactMatchMetric(true);
        assertEquals(1.0, metric.score("NYC", List.of("New York", "NYC", "New York City")), 1e-9);
    }

    @Test
    public void testExactMatchMismatch() {
        ExactMatchMetric metric = new ExactMatchMetric(true);
        assertEquals(0.0, metric.score("London", List.of("Paris")), 1e-9);
    }

    // ==================== F1Metric ====================

    @Test
    public void testF1PerfectMatch() {
        F1Metric metric = new F1Metric();
        assertEquals(1.0, metric.score("the quick brown fox", List.of("the quick brown fox")), 1e-9);
    }

    @Test
    public void testF1PartialOverlap() {
        F1Metric metric = new F1Metric();
        // After SQuAD normalization: "big red cat" vs "big blue dog"
        // Tokens: {big, red, cat} vs {big, blue, dog}
        // Common: {big} = 1
        // Precision = 1/3, Recall = 1/3, F1 = 2*(1/3)*(1/3) / (1/3+1/3) = 1/3
        double f1 = metric.score("big red cat", List.of("big blue dog"));
        assertEquals(1.0 / 3.0, f1, 1e-9);
    }

    @Test
    public void testF1NoOverlap() {
        F1Metric metric = new F1Metric();
        assertEquals(0.0, metric.score("cat", List.of("dog")), 1e-9);
    }

    // ==================== AnlsMetric ====================

    @Test
    public void testAnlsExactMatch() {
        AnlsMetric metric = new AnlsMetric();
        assertEquals(1.0, metric.score("hello", List.of("hello")), 1e-9);
    }

    @Test
    public void testAnlsSimilar() {
        AnlsMetric metric = new AnlsMetric();
        // "hello" vs "helo" => distance=1, maxLen=5, NLS=0.8, above threshold 0.5
        double score = metric.score("hello", List.of("helo"));
        assertEquals(0.8, score, 1e-9);
    }

    @Test
    public void testAnlsBelowThreshold() {
        AnlsMetric metric = new AnlsMetric();
        // Very different strings -> NLS < 0.5 -> score = 0
        assertEquals(0.0, metric.score("abcdef", List.of("xyz")), 1e-9);
    }

    @Test
    public void testLevenshteinDistance() {
        assertEquals(0, AnlsMetric.levenshteinDistance("test", "test"));
        assertEquals(1, AnlsMetric.levenshteinDistance("test", "tests"));
        assertEquals(3, AnlsMetric.levenshteinDistance("kitten", "sitting"));
    }

    // ==================== BleuMetric ====================

    @Test
    public void testBleuPerfectMatch() {
        BleuMetric metric = new BleuMetric(4);
        double score = metric.score("the cat sat on the mat", List.of("the cat sat on the mat"));
        assertEquals(1.0, score, 1e-9);
    }

    @Test
    public void testBleuPartialMatch() {
        BleuMetric metric = new BleuMetric(1);
        // BLEU-1 only checks unigram precision
        double score = metric.score("the cat sat", List.of("the cat sat on the mat"));
        assertTrue(score > 0.0, "BLEU-1 should be positive for partial match");
        assertTrue(score <= 1.0, "BLEU should not exceed 1.0");
    }

    @Test
    public void testBleuNoMatch() {
        BleuMetric metric = new BleuMetric(4);
        assertEquals(0.0, metric.score("abc def", List.of("xyz uvw")), 1e-9);
    }

    // ==================== RougeMetric ====================

    @Test
    public void testRougeLPerfectMatch() {
        RougeMetric metric = new RougeMetric(RougeMetric.RougeType.ROUGE_L, RougeMetric.ScoreType.F1);
        assertEquals(1.0, metric.score("the cat sat", List.of("the cat sat")), 1e-9);
    }

    @Test
    public void testRouge1PartialOverlap() {
        RougeMetric metric = new RougeMetric(RougeMetric.RougeType.ROUGE_1, RougeMetric.ScoreType.RECALL);
        // pred: "the cat", ref: "the cat sat on"
        // unigram overlap: {the, cat} = 2, ref total = 4
        // recall = 2/4 = 0.5
        double score = metric.score("the cat", List.of("the cat sat on"));
        assertEquals(0.5, score, 1e-9);
    }

    // ==================== VqaAccuracyMetric ====================

    @Test
    public void testVqaAccuracyAllMatch() {
        VqaAccuracyMetric metric = new VqaAccuracyMetric();
        // 4 matching refs -> min(4/3, 1.0) = 1.0
        assertEquals(1.0, metric.score("yes", List.of("yes", "yes", "yes", "yes")), 1e-9);
    }

    @Test
    public void testVqaAccuracyPartialMatch() {
        VqaAccuracyMetric metric = new VqaAccuracyMetric();
        // 2 matching out of 5 refs -> min(2/3, 1.0) = 0.6667
        double score = metric.score("cat", List.of("cat", "dog", "cat", "bird", "fish"));
        assertEquals(2.0 / 3.0, score, 1e-9);
    }

    // ==================== RelaxedAccuracyMetric ====================

    @Test
    public void testRelaxedAccuracyExactMatch() {
        RelaxedAccuracyMetric metric = new RelaxedAccuracyMetric();
        assertEquals(1.0, metric.score("42", List.of("42")), 1e-9);
    }

    @Test
    public void testRelaxedAccuracyWithinTolerance() {
        RelaxedAccuracyMetric metric = new RelaxedAccuracyMetric(0.05);
        // 104 vs 100: 4% error, within 5% tolerance
        assertEquals(1.0, metric.score("104", List.of("100")), 1e-9);
    }

    @Test
    public void testRelaxedAccuracyOutsideTolerance() {
        RelaxedAccuracyMetric metric = new RelaxedAccuracyMetric(0.05);
        // 110 vs 100: 10% error, outside 5% tolerance
        assertEquals(0.0, metric.score("110", List.of("100")), 1e-9);
    }

    // ==================== MultipleChoiceAccuracyMetric ====================

    @Test
    public void testMCAccuracySingleLetter() {
        MultipleChoiceAccuracyMetric metric = new MultipleChoiceAccuracyMetric();
        assertEquals(1.0, metric.score("A", List.of("A")), 1e-9);
    }

    @Test
    public void testMCAccuracyAnswerIsPattern() {
        MultipleChoiceAccuracyMetric metric = new MultipleChoiceAccuracyMetric();
        assertEquals(1.0, metric.score("The answer is B", List.of("B")), 1e-9);
    }

    @Test
    public void testMCAccuracyWrong() {
        MultipleChoiceAccuracyMetric metric = new MultipleChoiceAccuracyMetric();
        assertEquals(0.0, metric.score("A", List.of("C")), 1e-9);
    }

    // ==================== AnswerExtractor ====================

    @Test
    public void testExtractMultipleChoice() {
        assertEquals("A", AnswerExtractor.extractMultipleChoice("A"));
        assertEquals("B", AnswerExtractor.extractMultipleChoice("The answer is B"));
        assertEquals("C", AnswerExtractor.extractMultipleChoice("C. Some explanation"));
        assertNull(AnswerExtractor.extractMultipleChoice(""));
    }

    @Test
    public void testExtractNumber() {
        assertEquals("42", AnswerExtractor.extractNumber("Let me compute... #### 42"));
        assertEquals("3.14", AnswerExtractor.extractNumber("The answer is 3.14"));
        assertNull(AnswerExtractor.extractNumber("no numbers here"));
    }

    @Test
    public void testExtractYesNo() {
        assertEquals("yes", AnswerExtractor.extractYesNo("Yes, that is correct"));
        assertEquals("no", AnswerExtractor.extractYesNo("No, I disagree"));
        assertNull(AnswerExtractor.extractYesNo("maybe"));
    }

    @Test
    public void testNormalizeAnswer() {
        assertEquals("answer is paris", AnswerExtractor.normalizeAnswer("The answer is: Paris!"));
    }

    // ==================== CustomDataset ====================

    @Test
    public void testCustomDatasetBuilder() {
        CustomDataset dataset = CustomDataset.builder("test-ds")
                .addSample("1", "What is 1+1?", "2")
                .addSample("2", "Capital of France?", "Paris")
                .addMultipleChoice("3", "Color of sky?", List.of("Red", "Blue", "Green"), "B")
                .build();

        assertEquals("test-ds", dataset.name());
        assertEquals(3, dataset.size());
        assertEquals("What is 1+1?", dataset.get(0).getInput());
        assertEquals(List.of("2"), dataset.get(0).getReferences());
        assertEquals(List.of("Red", "Blue", "Green"), dataset.get(2).getChoices());
    }

    @Test
    public void testCustomDatasetIteration() {
        CustomDataset dataset = CustomDataset.builder("iter-test")
                .addSample("1", "q1", "a1")
                .addSample("2", "q2", "a2")
                .build();

        int count = 0;
        for (EvalSample sample : dataset) {
            assertNotNull(sample.getInput());
            count++;
        }
        assertEquals(2, count);
    }

    // ==================== JsonlDataset ====================

    @Test
    public void testJsonlDatasetLoading(@TempDir Path tempDir) throws IOException {
        File jsonlFile = tempDir.resolve("test.jsonl").toFile();
        try (FileWriter writer = new FileWriter(jsonlFile)) {
            writer.write("{\"question\": \"What is 2+2?\", \"answer\": \"4\"}\n");
            writer.write("{\"question\": \"Capital of Japan?\", \"answer\": \"Tokyo\"}\n");
            writer.write("{\"question\": \"Largest planet?\", \"answer\": \"Jupiter\"}\n");
        }

        JsonlDataset.FieldMapping mapping = JsonlDataset.FieldMapping.builder()
                .inputField("question")
                .referenceField("answer")
                .build();

        JsonlDataset dataset = new JsonlDataset("test-jsonl", jsonlFile, mapping);

        assertEquals("test-jsonl", dataset.name());
        assertEquals(3, dataset.size());
        assertEquals("What is 2+2?", dataset.get(0).getInput());
        assertEquals(List.of("4"), dataset.get(0).getReferences());
        assertEquals("Tokyo", dataset.get(1).getReferences().get(0));
    }

    // ==================== EvalConfig ====================

    @Test
    public void testEvalConfigDefaults() {
        EvalConfig config = EvalConfig.builder().build();
        assertEquals(-1, config.getNumFewShot());
        assertEquals(0, config.getMaxSamples());
        assertEquals(256, config.getMaxNewTokens());
        assertEquals(1, config.getBatchSize());
        assertFalse(config.isLogSamples());
        assertNull(config.getOutputFile());
    }

    @Test
    public void testEvalConfigCustom() {
        EvalConfig config = EvalConfig.builder()
                .numFewShot(5)
                .maxSamples(100)
                .maxNewTokens(512)
                .batchSize(8)
                .logSamples(true)
                .build();

        assertEquals(5, config.getNumFewShot());
        assertEquals(100, config.getMaxSamples());
        assertEquals(512, config.getMaxNewTokens());
        assertEquals(8, config.getBatchSize());
        assertTrue(config.isLogSamples());
    }

    // ==================== EvalResult ====================

    @Test
    public void testEvalResultAccuracy() {
        EvalResult result = EvalResult.builder()
                .benchmarkName("test")
                .primaryScore(0.85)
                .totalSamples(100)
                .correctSamples(85)
                .evaluationTimeMs(5000)
                .metricScores(Map.of("exact_match", 0.85))
                .build();

        assertEquals(0.85, result.accuracy(), 1e-9);
        assertTrue(result.summary().contains("test"));
        assertTrue(result.summary().contains("85"));
    }

    @Test
    public void testEvalResultWriteJson(@TempDir Path tempDir) throws IOException {
        EvalResult result = EvalResult.builder()
                .benchmarkName("json-test")
                .primaryScore(0.75)
                .totalSamples(50)
                .correctSamples(37)
                .evaluationTimeMs(3000)
                .metricScores(Map.of("exact_match", 0.74))
                .build();

        File outputFile = tempDir.resolve("result.json").toFile();
        result.writeJson(outputFile);
        assertTrue(outputFile.exists());
        assertTrue(outputFile.length() > 0);
    }

    // ==================== Metric Aggregation ====================

    @Test
    public void testMetricAggregation() {
        ExactMatchMetric metric = new ExactMatchMetric(true);
        double aggregated = metric.aggregate(List.of(1.0, 0.0, 1.0, 1.0));
        assertEquals(0.75, aggregated, 1e-9);
    }

    // ==================== Metric Properties ====================

    @Test
    public void testMetricNames() {
        assertEquals("exact_match", new ExactMatchMetric().name());
        assertEquals("f1", new F1Metric().name());
        assertEquals("anls", new AnlsMetric().name());
        assertEquals("bleu-4", new BleuMetric().name());
        assertEquals("bleu-1", new BleuMetric(1).name());
        assertEquals("rouge-l-f1", new RougeMetric().name());
        assertEquals("rouge-1-recall", new RougeMetric(RougeMetric.RougeType.ROUGE_1, RougeMetric.ScoreType.RECALL).name());
        assertEquals("vqa_accuracy", new VqaAccuracyMetric().name());
        assertEquals("relaxed_accuracy", new RelaxedAccuracyMetric().name());
        assertEquals("mc_accuracy", new MultipleChoiceAccuracyMetric().name());
    }

    @Test
    public void testAllMetricsHigherIsBetter() {
        List<EvalMetric> metrics = List.of(
                new ExactMatchMetric(),
                new F1Metric(),
                new AnlsMetric(),
                new BleuMetric(),
                new RougeMetric(),
                new VqaAccuracyMetric(),
                new RelaxedAccuracyMetric(),
                new MultipleChoiceAccuracyMetric()
        );
        for (EvalMetric metric : metrics) {
            assertTrue(metric.higherIsBetter(), metric.name() + " should have higherIsBetter=true");
        }
    }
}
