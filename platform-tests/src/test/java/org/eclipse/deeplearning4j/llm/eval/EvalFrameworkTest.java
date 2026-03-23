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

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the evaluation framework: metrics, datasets, and answer extraction.
 */
public class EvalFrameworkTest {

    // ======================== Exact Match ========================

    @Test
    public void testExactMatchNormalized() {
        ExactMatchMetric metric = new ExactMatchMetric(true);
        assertEquals("exact_match", metric.name());
        assertTrue(metric.higherIsBetter());

        // Exact match after normalization
        assertEquals(1.0, metric.score("The Paris", List.of("paris")));
        assertEquals(1.0, metric.score("  PARIS  ", List.of("paris")));

        // No match
        assertEquals(0.0, metric.score("London", List.of("Paris")));

        // Multiple references - match any
        assertEquals(1.0, metric.score("Paris", List.of("London", "Paris")));
    }

    @Test
    public void testExactMatchNoNormalization() {
        ExactMatchMetric metric = new ExactMatchMetric(false);
        assertEquals(0.0, metric.score("paris", List.of("Paris")));
        assertEquals(1.0, metric.score("Paris", List.of("Paris")));
    }

    @Test
    public void testExactMatchEdgeCases() {
        ExactMatchMetric metric = new ExactMatchMetric();
        assertEquals(0.0, metric.score(null, List.of("x")));
        assertEquals(0.0, metric.score("x", null));
        assertEquals(0.0, metric.score("x", Collections.emptyList()));
    }

    // ======================== F1 Metric ========================

    @Test
    public void testF1PerfectMatch() {
        F1Metric metric = new F1Metric();
        assertEquals("f1", metric.name());

        // Perfect match
        assertEquals(1.0, metric.score("the cat sat on the mat",
                List.of("the cat sat on the mat")));
    }

    @Test
    public void testF1PartialOverlap() {
        F1Metric metric = new F1Metric();

        // "big red cat" vs "big blue dog" → after normalize: "big red cat" vs "big blue dog"
        // 1 common token ("big"), P=1/3, R=1/3, F1=1/3
        double score = metric.score("big red cat", List.of("big blue dog"));
        assertEquals(1.0 / 3.0, score, 0.01);
    }

    @Test
    public void testF1NoOverlap() {
        F1Metric metric = new F1Metric();
        assertEquals(0.0, metric.score("cat", List.of("dog")));
    }

    @Test
    public void testF1MultipleReferences() {
        F1Metric metric = new F1Metric();
        // Should return max F1 across references
        double score = metric.score("Paris is the capital",
                List.of("Paris is the capital of France", "Paris"));
        assertTrue(score > 0.5);
    }

    // ======================== ANLS Metric ========================

    @Test
    public void testAnlsPerfectMatch() {
        AnlsMetric metric = new AnlsMetric();
        assertEquals("anls", metric.name());
        assertEquals(1.0, metric.score("hello", List.of("hello")));
    }

    @Test
    public void testAnlsCloseMatch() {
        AnlsMetric metric = new AnlsMetric();
        // "helo" vs "hello" → edit distance 1, NLS = 1 - 1/5 = 0.8 > 0.5 threshold
        double score = metric.score("helo", List.of("hello"));
        assertTrue(score > 0.7, "Expected ANLS > 0.7, got " + score);
    }

    @Test
    public void testAnlsBelowThreshold() {
        AnlsMetric metric = new AnlsMetric();
        // Very different strings should score 0 (below 0.5 threshold)
        assertEquals(0.0, metric.score("xyz", List.of("abcdefgh")));
    }

    @Test
    public void testLevenshteinDistance() {
        assertEquals(0, AnlsMetric.levenshteinDistance("", ""));
        assertEquals(3, AnlsMetric.levenshteinDistance("", "abc"));
        assertEquals(3, AnlsMetric.levenshteinDistance("abc", ""));
        assertEquals(1, AnlsMetric.levenshteinDistance("cat", "bat"));
        assertEquals(3, AnlsMetric.levenshteinDistance("kitten", "sitting"));
    }

    // ======================== BLEU Metric ========================

    @Test
    public void testBleuPerfectMatch() {
        BleuMetric metric = new BleuMetric(4);
        assertEquals("bleu-4", metric.name());

        double score = metric.score(
                "the cat is on the mat",
                List.of("the cat is on the mat"));
        assertEquals(1.0, score, 0.01);
    }

    @Test
    public void testBleuPartialMatch() {
        BleuMetric metric = new BleuMetric(1); // BLEU-1 for simplicity
        double score = metric.score(
                "the the the the",
                List.of("the cat is on the mat"));
        // Clipped: "the" appears 2x in ref, 4x in pred → clipped=2, total=4 → precision=0.5
        assertTrue(score > 0.0);
        assertTrue(score <= 1.0);
    }

    @Test
    public void testBleuNoOverlap() {
        BleuMetric metric = new BleuMetric(1);
        assertEquals(0.0, metric.score("xyz abc", List.of("foo bar baz")));
    }

    // ======================== ROUGE Metric ========================

    @Test
    public void testRouge1F1() {
        RougeMetric metric = new RougeMetric(RougeMetric.RougeType.ROUGE_1);
        assertEquals("rouge-1-f1", metric.name());

        double score = metric.score(
                "the cat sat on the mat",
                List.of("the cat sat on the mat"));
        assertEquals(1.0, score, 0.01);
    }

    @Test
    public void testRougeLF1() {
        RougeMetric metric = new RougeMetric(RougeMetric.RougeType.ROUGE_L);
        assertEquals("rouge-l-f1", metric.name());

        // Partial overlap via LCS
        double score = metric.score(
                "the cat sat",
                List.of("the cat sat on the mat"));
        assertTrue(score > 0.5, "Expected ROUGE-L > 0.5, got " + score);
    }

    @Test
    public void testRougeRecall() {
        RougeMetric metric = new RougeMetric(RougeMetric.RougeType.ROUGE_1, RougeMetric.ScoreType.RECALL);
        assertEquals("rouge-1-recall", metric.name());
    }

    // ======================== VQA Accuracy ========================

    @Test
    public void testVqaAccuracy() {
        VqaAccuracyMetric metric = new VqaAccuracyMetric();
        assertEquals("vqa_accuracy", metric.name());

        // 3+ matching annotators → score = 1.0
        assertEquals(1.0, metric.score("yes",
                List.of("yes", "yes", "yes", "no", "no")));

        // 1 matching annotator → score = 1/3
        assertEquals(1.0 / 3.0, metric.score("maybe",
                List.of("maybe", "yes", "no")), 0.01);

        // 0 matching → score = 0
        assertEquals(0.0, metric.score("xyz", List.of("yes", "no")));
    }

    // ======================== Relaxed Accuracy ========================

    @Test
    public void testRelaxedAccuracyExactMatch() {
        RelaxedAccuracyMetric metric = new RelaxedAccuracyMetric();
        assertEquals("relaxed_accuracy", metric.name());
        assertEquals(1.0, metric.score("42", List.of("42")));
    }

    @Test
    public void testRelaxedAccuracyNumericTolerance() {
        RelaxedAccuracyMetric metric = new RelaxedAccuracyMetric(0.05);
        // 42 vs 43: |42-43|/43 = 2.3% < 5%
        assertEquals(1.0, metric.score("42", List.of("43")));
        // 42 vs 50: |42-50|/50 = 16% > 5%
        assertEquals(0.0, metric.score("42", List.of("50")));
    }

    @Test
    public void testRelaxedAccuracyFormattedNumbers() {
        RelaxedAccuracyMetric metric = new RelaxedAccuracyMetric(0.05);
        // Should handle $, commas, %
        assertEquals(1.0, metric.score("$1,000", List.of("1000")));
        assertEquals(1.0, metric.score("50%", List.of("50")));
    }

    // ======================== MC Accuracy ========================

    @Test
    public void testMultipleChoiceAccuracy() {
        MultipleChoiceAccuracyMetric metric = new MultipleChoiceAccuracyMetric();
        assertEquals("mc_accuracy", metric.name());

        assertEquals(1.0, metric.score("A", List.of("A")));
        assertEquals(1.0, metric.score("The answer is B", List.of("B")));
        assertEquals(1.0, metric.score("(C)", List.of("C")));
        assertEquals(0.0, metric.score("A", List.of("B")));
    }

    // ======================== Metric Aggregation ========================

    @Test
    public void testMetricAggregation() {
        EvalMetric metric = new ExactMatchMetric();
        assertEquals(0.5, metric.aggregate(List.of(1.0, 0.0, 1.0, 0.0)), 0.01);
        assertEquals(1.0, metric.aggregate(List.of(1.0, 1.0)), 0.01);
        assertEquals(0.0, metric.aggregate(List.of(0.0, 0.0)), 0.01);
        assertEquals(0.0, metric.aggregate(Collections.emptyList()));
    }

    // ======================== Custom Dataset ========================

    @Test
    public void testCustomDataset() {
        CustomDataset dataset = CustomDataset.builder("test-qa")
                .addSample("1", "What is 2+2?", "4")
                .addSample("2", "Capital of France?", "Paris")
                .addMultipleChoice("3", "Which is a fruit?",
                        List.of("Apple", "Chair", "Book"), "A")
                .build();

        assertEquals("test-qa", dataset.name());
        assertEquals(3, dataset.size());

        EvalSample sample = dataset.get(0);
        assertEquals("1", sample.getId());
        assertEquals("What is 2+2?", sample.getInput());
        assertEquals(List.of("4"), sample.getReferences());

        EvalSample mcSample = dataset.get(2);
        assertEquals(List.of("Apple", "Chair", "Book"), mcSample.getChoices());
        assertEquals(List.of("A"), mcSample.getReferences());

        // Test iteration
        int count = 0;
        for (EvalSample s : dataset) {
            count++;
        }
        assertEquals(3, count);

        // Test splits
        assertEquals(3, dataset.getSplit("test").size());
        assertEquals(0, dataset.getSplit("train").size());
    }

    // ======================== JSONL Dataset ========================

    @Test
    public void testJsonlDatasetLoading() throws Exception {
        // Create a temp JSONL file
        String jsonl = String.join("\n",
                "{\"id\": \"q1\", \"question\": \"What color is the sky?\", \"answer\": \"blue\"}",
                "{\"id\": \"q2\", \"question\": \"2+2=\", \"answer\": \"4\", \"subject\": \"math\"}",
                "{\"id\": \"q3\", \"question\": \"Best color?\", \"answers\": [\"red\", \"blue\"]}"
        );

        File tempFile = File.createTempFile("test-dataset", ".jsonl");
        tempFile.deleteOnExit();
        Files.writeString(tempFile.toPath(), jsonl, StandardCharsets.UTF_8);

        JsonlDataset dataset = new JsonlDataset("test", tempFile);
        assertEquals("test", dataset.name());
        assertEquals(3, dataset.size());

        EvalSample s1 = dataset.get(0);
        assertEquals("q1", s1.getId());
        assertEquals("What color is the sky?", s1.getInput());
        assertEquals(List.of("blue"), s1.getReferences());

        EvalSample s2 = dataset.get(1);
        assertEquals("math", s2.getSubject());

        EvalSample s3 = dataset.get(2);
        assertEquals(List.of("red", "blue"), s3.getReferences());
    }

    @Test
    public void testJsonlDatasetCustomMapping() throws Exception {
        String jsonl = "{\"q\": \"What is 1+1?\", \"a\": \"2\", \"cat\": \"arithmetic\"}";

        File tempFile = File.createTempFile("test-custom", ".jsonl");
        tempFile.deleteOnExit();
        Files.writeString(tempFile.toPath(), jsonl, StandardCharsets.UTF_8);

        JsonlDataset.FieldMapping mapping = JsonlDataset.FieldMapping.builder()
                .inputField("q")
                .referenceField("a")
                .subjectField("cat")
                .build();

        JsonlDataset dataset = new JsonlDataset("custom", tempFile, mapping);
        assertEquals(1, dataset.size());
        assertEquals("What is 1+1?", dataset.get(0).getInput());
        assertEquals(List.of("2"), dataset.get(0).getReferences());
        assertEquals("arithmetic", dataset.get(0).getSubject());
    }

    // ======================== Answer Extraction ========================

    @Test
    public void testExtractMultipleChoice() {
        assertEquals("A", AnswerExtractor.extractMultipleChoice("A"));
        assertEquals("B", AnswerExtractor.extractMultipleChoice("The answer is B"));
        assertEquals("C", AnswerExtractor.extractMultipleChoice("The answer is (C)"));
        assertEquals("D", AnswerExtractor.extractMultipleChoice("D. Something"));
        assertNull(AnswerExtractor.extractMultipleChoice(""));
        assertNull(AnswerExtractor.extractMultipleChoice(null));
    }

    @Test
    public void testExtractNumber() {
        assertEquals("42", AnswerExtractor.extractNumber("#### 42"));
        assertEquals("1000", AnswerExtractor.extractNumber("#### 1,000"));
        assertEquals("3.14", AnswerExtractor.extractNumber("The answer is 3.14"));
        assertNull(AnswerExtractor.extractNumber("no numbers here"));
        assertNull(AnswerExtractor.extractNumber(null));
    }

    @Test
    public void testExtractYesNo() {
        assertEquals("yes", AnswerExtractor.extractYesNo("Yes, it is."));
        assertEquals("no", AnswerExtractor.extractYesNo("No, it isn't."));
        assertNull(AnswerExtractor.extractYesNo("maybe"));
        assertNull(AnswerExtractor.extractYesNo(null));
    }

    @Test
    public void testNormalizeAnswer() {
        assertEquals("paris", AnswerExtractor.normalizeAnswer("The Paris!"));
        assertEquals("42", AnswerExtractor.normalizeAnswer("  42  "));
    }

    // ======================== EvalConfig ========================

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

    // ======================== EvalResult ========================

    @Test
    public void testEvalResultSummary() {
        EvalResult result = EvalResult.builder()
                .benchmarkName("test")
                .primaryScore(0.75)
                .totalSamples(100)
                .correctSamples(75)
                .evaluationTimeMs(1000)
                .build();

        assertEquals(0.75, result.accuracy(), 0.01);
        String summary = result.summary();
        assertTrue(summary.contains("test"));
        assertTrue(summary.contains("0.7500"));
        assertTrue(summary.contains("75/100"));
    }

    // ======================== SampleResult ========================

    @Test
    public void testSampleResult() {
        SampleResult result = SampleResult.builder()
                .sampleId("s1")
                .prediction("Paris")
                .references(List.of("Paris", "paris"))
                .score(1.0)
                .correct(true)
                .build();

        assertEquals("s1", result.getSampleId());
        assertEquals("Paris", result.getPrediction());
        assertTrue(result.isCorrect());
    }
}
