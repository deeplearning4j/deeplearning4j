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

package org.eclipse.deeplearning4j.llm.eval.benchmark;

import org.eclipse.deeplearning4j.llm.eval.AnswerExtractor;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.dataset.HuggingFaceDataset;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.MultipleChoiceAccuracyMetric;

import java.io.IOException;
import java.util.List;

/**
 * MMLU (Massive Multitask Language Understanding) benchmark.
 * 57-subject multiple-choice evaluation with 5-shot prompting.
 */
public class MMLUBenchmark implements BenchmarkTask {

    private static final String[] CHOICE_LABELS = {"A", "B", "C", "D"};

    @Override
    public String name() {
        return "mmlu";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.create("cais/mmlu", "all", "test",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .inputField("question")
                        .choicesField("choices")
                        .referenceField("answer")
                        .subjectField("subject")
                        .build());
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        StringBuilder sb = new StringBuilder();

        if (sample.getSubject() != null) {
            sb.append("The following are multiple choice questions about ")
                    .append(sample.getSubject().replace("_", " ")).append(".\n\n");
        }

        // Few-shot examples
        for (EvalSample ex : fewShotExamples) {
            sb.append(formatQuestion(ex));
            if (!ex.getReferences().isEmpty()) {
                String answer = ex.getReferences().get(0);
                // Reference might be index (0-3) or letter (A-D)
                sb.append("Answer: ").append(resolveAnswer(answer, ex.getChoices())).append("\n\n");
            }
        }

        // Target question
        sb.append(formatQuestion(sample));
        sb.append("Answer:");
        return sb.toString();
    }

    private String formatQuestion(EvalSample sample) {
        StringBuilder sb = new StringBuilder();
        sb.append(sample.getInput()).append("\n");
        if (sample.getChoices() != null) {
            for (int i = 0; i < sample.getChoices().size(); i++) {
                String label = i < CHOICE_LABELS.length ? CHOICE_LABELS[i] : String.valueOf(i);
                sb.append(label).append(". ").append(sample.getChoices().get(i)).append("\n");
            }
        }
        return sb.toString();
    }

    private String resolveAnswer(String answer, List<String> choices) {
        // If answer is a number (index), convert to letter
        try {
            int idx = Integer.parseInt(answer.trim());
            if (idx >= 0 && idx < CHOICE_LABELS.length) return CHOICE_LABELS[idx];
        } catch (NumberFormatException ignored) {}
        return answer.trim();
    }

    @Override
    public String extractAnswer(String modelOutput) {
        String choice = AnswerExtractor.extractMultipleChoice(modelOutput);
        return choice != null ? choice : modelOutput.trim();
    }

    @Override
    public EvalMetric primaryMetric() {
        return new MultipleChoiceAccuracyMetric();
    }

    @Override
    public List<EvalMetric> allMetrics() {
        return List.of(new MultipleChoiceAccuracyMetric());
    }

    @Override
    public OutputType outputType() {
        return OutputType.MULTIPLE_CHOICE;
    }

    @Override
    public int defaultFewShot() {
        return 5;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 5;
    }
}
