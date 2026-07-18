package org.eclipse.deeplearning4j.llm.finetune;

import java.util.*;

/**
 * Aggregate result of a held-out generation evaluation run.
 *
 * <p>Reports overall pass rate, per-example pass/fail details with reasons,
 * and optional text-metric scores. Generated and reference text are preserved
 * in the per-example records for audit.</p>
 */
public final class HeldOutEvaluationResult {

    private final String evaluationName;
    private final int total;
    private final int passed;
    private final int failed;
    private final List<ExampleResult> examples;
    private final Map<String, Double> textMetrics;
    private final long evaluationTimeMs;

    private HeldOutEvaluationResult(Builder builder) {
        this.evaluationName = builder.evaluationName;
        this.total = builder.total;
        this.passed = builder.passed;
        this.failed = builder.failed;
        this.examples = Collections.unmodifiableList(new ArrayList<>(builder.examples));
        this.textMetrics = Collections.unmodifiableMap(new LinkedHashMap<>(builder.textMetrics));
        this.evaluationTimeMs = builder.evaluationTimeMs;
    }

    public String getEvaluationName() { return evaluationName; }
    public int getTotal() { return total; }
    public int getPassed() { return passed; }
    public int getFailed() { return failed; }
    public List<ExampleResult> getExamples() { return examples; }
    public Map<String, Double> getTextMetrics() { return textMetrics; }
    public long getEvaluationTimeMs() { return evaluationTimeMs; }
    public double passRate() { return total > 0 ? (double) passed / total : 0.0; }

    public List<ExampleResult> failures() {
        List<ExampleResult> result = new ArrayList<>();
        for (ExampleResult e : examples) {
            if (!e.passed) result.add(e);
        }
        return result;
    }

    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Evaluation: %s%n", evaluationName != null ? evaluationName : "(unnamed)"));
        sb.append(String.format("Pass rate: %d/%d (%.1f%%, %d failed)%n",
                passed, total, passRate() * 100, failed));
        sb.append(String.format("Time: %dms%n", evaluationTimeMs));
        if (!textMetrics.isEmpty()) {
            sb.append("Text metrics:\n");
            for (Map.Entry<String, Double> e : textMetrics.entrySet()) {
                sb.append(String.format("  %s: %.4f%n", e.getKey(), e.getValue()));
            }
        }
        if (failed > 0) {
            sb.append("Failures:\n");
            for (ExampleResult e : failures()) {
                sb.append(String.format("  [%s] %s%n", e.exampleId, e.failureReasons));
            }
        }
        return sb.toString();
    }

    /** Per-example result with generated and reference text preserved for audit. */
    public static final class ExampleResult {
        private final String exampleId;
        private final String prompt;
        private final String referenceText;
        private final String generatedText;
        private final boolean passed;
        private final List<String> failureReasons;

        ExampleResult(String exampleId, String prompt, String referenceText,
                      String generatedText, boolean passed, List<String> failureReasons) {
            this.exampleId = exampleId;
            this.prompt = prompt;
            this.referenceText = referenceText;
            this.generatedText = generatedText;
            this.passed = passed;
            this.failureReasons = Collections.unmodifiableList(new ArrayList<>(failureReasons));
        }

        public String getExampleId() { return exampleId; }
        public String getPrompt() { return prompt; }
        public String getReferenceText() { return referenceText; }
        public String getGeneratedText() { return generatedText; }
        public boolean isPassed() { return passed; }
        public List<String> getFailureReasons() { return failureReasons; }
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String evaluationName;
        private int total;
        private int passed;
        private int failed;
        private final List<ExampleResult> examples = new ArrayList<>();
        private final Map<String, Double> textMetrics = new LinkedHashMap<>();
        private long evaluationTimeMs;

        public Builder evaluationName(String name) { this.evaluationName = name; return this; }
        public Builder total(int total) { this.total = total; return this; }
        public Builder passed(int passed) { this.passed = passed; return this; }
        public Builder failed(int failed) { this.failed = failed; return this; }
        public Builder addExample(ExampleResult example) { this.examples.add(example); return this; }
        public Builder examples(List<ExampleResult> values) { this.examples.addAll(values); return this; }
        public Builder textMetric(String name, double value) { this.textMetrics.put(name, value); return this; }
        public Builder textMetrics(Map<String, Double> metrics) { this.textMetrics.putAll(metrics); return this; }
        public Builder evaluationTimeMs(long ms) { this.evaluationTimeMs = ms; return this; }

        public HeldOutEvaluationResult build() {
            return new HeldOutEvaluationResult(this);
        }
    }
}
