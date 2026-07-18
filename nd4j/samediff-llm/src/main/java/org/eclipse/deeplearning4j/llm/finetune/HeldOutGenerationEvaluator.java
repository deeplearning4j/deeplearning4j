package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;

import java.util.*;

/**
 * Generic held-out generation evaluation engine.
 *
 * <p>Evaluates a held-out set of {@link GeneratedTrainingExample} records by running
 * a pluggable {@link SampleGenerator} on each prompt, applying a list of
 * {@link HeldOutValidator}s (analogous to {@link TeacherOutputValidator}), and
 * collecting aggregate pass rates, per-example failure reasons, and optional
 * text-level metrics via llm.eval {@link EvalMetric}s.</p>
 *
 * <p>The evaluator is deliberately decoupled from domain schemas — it works with
 * {@code GeneratedTrainingExample} and generic text validators without any
 * dependency on teacher requests or training pipelines.</p>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * HeldOutGenerationEvaluator evaluator = new HeldOutGenerationEvaluator(
 *     (prompt, maxTokens) -> pipeline.generate(prompt, maxTokens).getText(),
 *     Arrays.asList(new ExactMatchHeldOutValidator(), new NormalizedMatchHeldOutValidator()),
 *     Arrays.asList(new ExactMatchMetric(), new F1Metric()),
 *     128
 * );
 * HeldOutEvaluationResult result = evaluator.evaluate(heldOutExamples);
 * System.out.println(result.summary());
 * }</pre>
 */
public class HeldOutGenerationEvaluator {

    @FunctionalInterface
    public interface SampleGenerator {
        String generate(String prompt, int maxNewTokens);
    }

    private final SampleGenerator generator;
    private final List<HeldOutValidator> validators;
    private final List<EvalMetric> textMetrics;
    private final int maxNewTokens;

    public HeldOutGenerationEvaluator(SampleGenerator generator,
                                      List<HeldOutValidator> validators,
                                      List<EvalMetric> textMetrics,
                                      int maxNewTokens) {
        if (generator == null) throw new IllegalArgumentException("generator is required");
        if (maxNewTokens < 1) throw new IllegalArgumentException("maxNewTokens must be positive");
        this.generator = generator;
        this.validators = validators == null ? Collections.emptyList() : new ArrayList<>(validators);
        this.textMetrics = textMetrics == null ? Collections.emptyList() : new ArrayList<>(textMetrics);
        this.maxNewTokens = maxNewTokens;
    }

    /**
     * Evaluate a held-out set of training examples.
     *
     * @param heldOutSet the held-out examples to evaluate (not null, not empty)
     * @return aggregated evaluation result
     */
    public HeldOutEvaluationResult evaluate(List<GeneratedTrainingExample> heldOutSet) {
        if (heldOutSet == null || heldOutSet.isEmpty()) {
            throw new IllegalArgumentException("heldOutSet must not be null or empty");
        }

        long start = System.currentTimeMillis();
        List<HeldOutEvaluationResult.ExampleResult> exampleResults = new ArrayList<>();
        int passed = 0;
        int failed = 0;

        for (GeneratedTrainingExample example : heldOutSet) {
            example.validate();
            String prompt = resolvePrompt(example);
            String generated = generator.generate(prompt, maxNewTokens);
            if (generated == null) generated = "";

            List<String> failureReasons = validateAll(example, generated);
            boolean examplePassed = failureReasons.isEmpty();
            if (examplePassed) passed++; else failed++;

            String referenceText = resolveReference(example);
            exampleResults.add(new HeldOutEvaluationResult.ExampleResult(
                    example.getId(), prompt, referenceText, generated,
                    examplePassed, failureReasons));
        }

        Map<String, Double> metricScores = computeTextMetrics(exampleResults);

        return HeldOutEvaluationResult.builder()
                .total(heldOutSet.size())
                .passed(passed)
                .failed(failed)
                .examples(exampleResults)
                .textMetrics(metricScores)
                .evaluationTimeMs(System.currentTimeMillis() - start)
                .build();
    }

    private String resolvePrompt(GeneratedTrainingExample example) {
        List<FineTuneMessage> messages = example.effectiveMessages();
        int target = lastAssistant(messages);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < target; i++) {
            FineTuneMessage msg = messages.get(i);
            if (msg.getContent() != null) sb.append(msg.getContent()).append("\n");
        }
        return sb.toString().trim();
    }

    private String resolveReference(GeneratedTrainingExample example) {
        List<FineTuneMessage> messages = example.effectiveMessages();
        int target = lastAssistant(messages);
        return target >= 0 ? messages.get(target).getContent()
                : (example.getResponse() == null ? "" : example.getResponse());
    }

    private int lastAssistant(List<FineTuneMessage> messages) {
        for (int i = messages.size() - 1; i >= 0; i--) {
            if ("assistant".equals(messages.get(i).getRole())) return i;
        }
        return messages.size();
    }

    private List<String> validateAll(GeneratedTrainingExample example, String generatedText) {
        List<String> reasons = new ArrayList<>();
        for (HeldOutValidator validator : validators) {
            TeacherValidationResult result = validator.validate(example, generatedText);
            if (result != null && !result.isAccepted()) {
                reasons.addAll(result.getReasons());
            }
        }
        return reasons;
    }

    private Map<String, Double> computeTextMetrics(
            List<HeldOutEvaluationResult.ExampleResult> results) {
        if (textMetrics.isEmpty()) return Collections.emptyMap();
        Map<String, Double> scores = new LinkedHashMap<>();
        for (EvalMetric metric : textMetrics) {
            List<Double> perSample = new ArrayList<>();
            for (HeldOutEvaluationResult.ExampleResult r : results) {
                double s = metric.score(r.getGeneratedText(),
                        Collections.singletonList(r.getReferenceText()));
                perSample.add(s);
            }
            scores.put(metric.name(), metric.aggregate(perSample));
        }
        return scores;
    }

    /**
     * Held-out validator that requires exact string match between generated
     * and reference text.
     */
    public static final class ExactMatchHeldOutValidator implements HeldOutValidator {
        @Override
        public TeacherValidationResult validate(GeneratedTrainingExample example, String generatedText) {
            String reference = example.getResponse();
            if (reference == null && !example.getMessages().isEmpty()) {
                for (FineTuneMessage msg : example.getMessages()) {
                    if ("assistant".equals(msg.getRole())) {
                        reference = msg.getContent();
                        break;
                    }
                }
            }
            if (reference == null) reference = "";
            if (reference.equals(generatedText)) return TeacherValidationResult.accept();
            return TeacherValidationResult.reject("exact match failed: expected='" + reference + "' got='" + generatedText + "'");
        }
    }

    /**
     * Held-out validator that requires normalized (lowercased, punctuation-stripped,
     * whitespace-collapsed) match between generated and reference text.
     */
    public static final class NormalizedMatchHeldOutValidator implements HeldOutValidator {
        @Override
        public TeacherValidationResult validate(GeneratedTrainingExample example, String generatedText) {
            String reference = example.getResponse();
            if (reference == null && !example.getMessages().isEmpty()) {
                for (FineTuneMessage msg : example.getMessages()) {
                    if ("assistant".equals(msg.getRole())) {
                        reference = msg.getContent();
                        break;
                    }
                }
            }
            if (reference == null) reference = "";
            String normalizedRef = org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric.normalize(reference);
            String normalizedGen = org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric.normalize(generatedText);
            if (normalizedRef.equals(normalizedGen)) return TeacherValidationResult.accept();
            return TeacherValidationResult.reject(
                    "normalized match failed: expected='" + normalizedRef + "' got='" + normalizedGen + "'");
        }
    }

    /**
     * Held-out validator that requires the generated text to contain expected substrings.
     */
    public static final class ContainsSubstringHeldOutValidator implements HeldOutValidator {
        private final List<String> requiredSubstrings;

        public ContainsSubstringHeldOutValidator(List<String> requiredSubstrings) {
            this.requiredSubstrings = new ArrayList<>(requiredSubstrings);
        }

        @Override
        public TeacherValidationResult validate(GeneratedTrainingExample example, String generatedText) {
            List<String> missing = new ArrayList<>();
            for (String sub : requiredSubstrings) {
                if (!generatedText.toLowerCase().contains(sub.toLowerCase())) {
                    missing.add("missing expected substring: '" + sub + "'");
                }
            }
            if (missing.isEmpty()) return TeacherValidationResult.accept();
            return TeacherValidationResult.reject(missing);
        }
    }
}
