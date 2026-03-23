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
 * HellaSwag benchmark for commonsense NLI.
 * 4-choice completion task.
 *
 * HF schema: {ind, activity_label, ctx_a, ctx_b, ctx, endings: [...], label: 0-3 (int)}
 */
public class HellaSwagBenchmark implements BenchmarkTask {

    private static final String[] CHOICE_LABELS = {"A", "B", "C", "D"};

    @Override
    public String name() {
        return "hellaswag";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.create("Rowan/hellaswag", null, "validation",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .idField("ind")
                        .inputField("ctx")
                        .choicesField("endings")
                        .referenceField("label")
                        .build());
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        StringBuilder sb = new StringBuilder();
        sb.append("Complete the following sentence with the most appropriate ending.\n\n");

        for (EvalSample ex : fewShotExamples) {
            sb.append(formatQuestion(ex));
            if (!ex.getReferences().isEmpty()) {
                sb.append("Answer: ").append(resolveLabel(ex.getReferences().get(0))).append("\n\n");
            }
        }

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

    private String resolveLabel(String label) {
        try {
            int idx = Integer.parseInt(label.trim());
            if (idx >= 0 && idx < CHOICE_LABELS.length) return CHOICE_LABELS[idx];
        } catch (NumberFormatException ignored) {}
        return label.trim();
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
        return 10;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 5;
    }
}
