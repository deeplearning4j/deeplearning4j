package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.F1Metric;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link HeldOutGenerationEvaluator} — held-out generation evaluation
 * with pluggable validators, text metrics, and per-example audit trails.
 *
 * <p>No ND4J native backend is required; all tests use purely in-memory generators
 * and Java-only validators.</p>
 */
class HeldOutGenerationEvaluatorTest {

    private static GeneratedTrainingExample example(String id, String prompt, String response) {
        GeneratedTrainingExample ex = new GeneratedTrainingExample();
        ex.setId(id);
        ex.setPrompt(prompt);
        ex.setResponse(response);
        return ex;
    }

    private static GeneratedTrainingExample multiTurnExample(String id,
                                                              String userContent,
                                                              String assistantContent) {
        GeneratedTrainingExample ex = new GeneratedTrainingExample();
        ex.setId(id);
        ex.setMessages(Arrays.asList(
                new FineTuneMessage("user", userContent),
                new FineTuneMessage("assistant", assistantContent)));
        return ex;
    }

    @Test
    void allPassWhenGeneratorMatchesExact() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "Paris",
                Collections.singletonList(new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "What is the capital of France?", "Paris")));

        assertEquals(1, result.getTotal());
        assertEquals(1, result.getPassed());
        assertEquals(0, result.getFailed());
        assertEquals(1.0, result.passRate(), 1e-9);
        assertTrue(result.failures().isEmpty());
    }

    @Test
    void exactMatchFailsWhenOutputDiffers() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "London",
                Collections.singletonList(new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "What is the capital of France?", "Paris")));

        assertEquals(0, result.getPassed());
        assertEquals(1, result.getFailed());
        assertEquals(0.0, result.passRate(), 1e-9);
        assertEquals(1, result.failures().size());
        assertTrue(result.failures().get(0).getFailureReasons().get(0).contains("exact match failed"));
    }

    @Test
    void normalizedMatchPassesWithCaseAndWhitespaceDifferences() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "  THE   answer is Paris!  ",
                Collections.singletonList(new HeldOutGenerationEvaluator.NormalizedMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "The answer is Paris")));

        assertEquals(1, result.getPassed());
        assertEquals(0, result.getFailed());
    }

    @Test
    void normalizedMatchFailsWhenContentDiffers() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "Tokyo",
                Collections.singletonList(new HeldOutGenerationEvaluator.NormalizedMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "Paris")));

        assertEquals(0, result.getPassed());
        assertEquals(1, result.getFailed());
        assertTrue(result.failures().get(0).getFailureReasons().get(0).contains("normalized match failed"));
    }

    @Test
    void multipleValidatorsAllMustPass() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "Paris is the capital of France",
                Arrays.asList(
                        new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator(),
                        new HeldOutGenerationEvaluator.ContainsSubstringHeldOutValidator(
                                Arrays.asList("Paris", "capital", "France"))),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "Paris is the capital of France")));

        assertEquals(1, result.getPassed());
    }

    @Test
    void containsSubstringValidatorAccumulatesMissing() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "Some other text",
                Arrays.asList(
                        new HeldOutGenerationEvaluator.ContainsSubstringHeldOutValidator(
                                Arrays.asList("Paris", "France"))),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "Paris capital of France")));

        assertEquals(0, result.getPassed());
        HeldOutEvaluationResult.ExampleResult failure = result.failures().get(0);
        assertEquals(2, failure.getFailureReasons().size());
        assertTrue(failure.getFailureReasons().get(0).contains("missing expected substring"));
        assertTrue(failure.getFailureReasons().get(1).contains("missing expected substring"));
    }

    @Test
    void aggregatePassRateAcrossMultipleExamples() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> prompt.contains("France") ? "Paris" : "wrong answer",
                Collections.singletonList(new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        List<GeneratedTrainingExample> heldOut = Arrays.asList(
                example("1", "What is the capital of France?", "Paris"),
                example("2", "What is the capital of Germany?", "Berlin"),
                example("3", "What is the capital of Japan?", "Tokyo"));

        HeldOutEvaluationResult result = evaluator.evaluate(heldOut);

        assertEquals(3, result.getTotal());
        assertEquals(1, result.getPassed());
        assertEquals(2, result.getFailed());
        assertEquals(1.0 / 3.0, result.passRate(), 1e-9);
    }

    @Test
    void multiTurnExamplesUseAssistantMessages() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "The capital of France is Paris",
                Collections.singletonList(new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        List<GeneratedTrainingExample> heldOut = Collections.singletonList(
                multiTurnExample("mt1", "What is the capital of France?",
                        "The capital of France is Paris"));

        HeldOutEvaluationResult result = evaluator.evaluate(heldOut);

        assertEquals(1, result.getPassed());
    }

    @Test
    void textMetricsAreComputedWhenProvided() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "Paris",
                Collections.singletonList(new HeldOutGenerationEvaluator.NormalizedMatchHeldOutValidator()),
                Arrays.asList(new ExactMatchMetric(true), new F1Metric()), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "Paris")));

        Map<String, Double> metrics = result.getTextMetrics();
        assertTrue(metrics.containsKey("exact_match"));
        assertTrue(metrics.containsKey("f1"));
        assertEquals(1.0, metrics.get("exact_match"), 1e-9);
    }

    @Test
    void f1MetricReflectsPartialMatch() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "quick brown fox",
                Collections.emptyList(),
                Collections.singletonList(new F1Metric()), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "quick red fox")));

        Map<String, Double> metrics = result.getTextMetrics();
        assertTrue(metrics.containsKey("f1"));
        assertTrue(metrics.get("f1") > 0.0);
        assertTrue(metrics.get("f1") < 1.0);
    }

    @Test
    void perExampleAuditPreservesGeneratedAndReferenceText() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "generated text",
                Collections.singletonList(new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator()),
                Collections.emptyList(), 64);

        GeneratedTrainingExample example = example("audit1", "What is 2+2?", "reference text");
        HeldOutEvaluationResult result = evaluator.evaluate(Collections.singletonList(example));

        HeldOutEvaluationResult.ExampleResult er = result.getExamples().get(0);
        assertEquals("audit1", er.getExampleId());
        assertEquals("generated text", er.getGeneratedText());
        assertEquals("reference text", er.getReferenceText());
        assertEquals("What is 2+2?", er.getPrompt());
    }

    @Test
    void generatorReceivesFullPromptFromMultiTurnMessages() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> {
                    assertTrue(prompt.contains("user question"));
                    assertTrue(prompt.contains("system context"));
                    return "assistant reply";
                },
                Collections.emptyList(), Collections.emptyList(), 64);

        GeneratedTrainingExample ex = new GeneratedTrainingExample();
        ex.setId("fullprompt");
        ex.setMessages(Arrays.asList(
                new FineTuneMessage("system", "system context"),
                new FineTuneMessage("user", "user question"),
                new FineTuneMessage("assistant", "assistant reply")));

        HeldOutEvaluationResult result = evaluator.evaluate(Collections.singletonList(ex));
        assertEquals(1, result.getPassed());
    }

    @Test
    void emptyValidatorsAlwaysPass() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "anything",
                Collections.emptyList(), Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "reference")));

        assertEquals(1, result.getPassed());
        assertEquals(0, result.getFailed());
        assertTrue(result.getTextMetrics().isEmpty());
    }

    @Test
    void nullValidatorsAndMetricsConstructorAccepts() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "output", null, null, 32);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "output")));

        assertEquals(1, result.getPassed());
        assertTrue(result.getTextMetrics().isEmpty());
    }

    @Test
    void evaluationTimeIsRecorded() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "fast", Collections.emptyList(), Collections.emptyList(), 16);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "fast")));

        assertTrue(result.getEvaluationTimeMs() >= 0);
    }

    @Test
    void nullHeldOutSetRejected() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "out", Collections.emptyList(), Collections.emptyList(), 16);

        assertThrows(IllegalArgumentException.class, () -> evaluator.evaluate(null));
        assertThrows(IllegalArgumentException.class, () -> evaluator.evaluate(Collections.emptyList()));
    }

    @Test
    void summaryIncludesAllSections() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "Paris",
                Collections.singletonList(new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator()),
                Collections.singletonList(new ExactMatchMetric(true)), 64);

        List<GeneratedTrainingExample> heldOut = Arrays.asList(
                example("1", "Capital of France?", "Paris"),
                example("2", "Capital of Germany?", "Berlin"));

        HeldOutEvaluationResult result = evaluator.evaluate(heldOut);
        String summary = result.summary();

        assertTrue(summary.contains("Pass rate"));
        assertTrue(summary.contains("Text metrics"));
        assertTrue(summary.contains("exact_match"));
        assertTrue(summary.contains("Failures"));
    }

    @Test
    void maxNewTokensZeroRejectedInConstructor() {
        assertThrows(IllegalArgumentException.class, () -> new HeldOutGenerationEvaluator(
                (prompt, max) -> "out", Collections.emptyList(), Collections.emptyList(), 0));
    }

    @Test
    void nullGeneratorRejectedInConstructor() {
        assertThrows(IllegalArgumentException.class, () -> new HeldOutGenerationEvaluator(
                null, Collections.emptyList(), Collections.emptyList(), 16));
    }

    @Test
    void multiValidatorFailureReasonsAreAccumulated() {
        HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
                (prompt, max) -> "wrong",
                Arrays.asList(
                        new HeldOutGenerationEvaluator.ExactMatchHeldOutValidator(),
                        new HeldOutGenerationEvaluator.ContainsSubstringHeldOutValidator(
                                Arrays.asList("Paris"))),
                Collections.emptyList(), 64);

        HeldOutEvaluationResult result = evaluator.evaluate(
                Collections.singletonList(example("1", "Q", "Paris is the capital")));

        assertEquals(0, result.getPassed());
        assertEquals(2, result.failures().get(0).getFailureReasons().size());
    }
}
