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

import com.fasterxml.jackson.databind.JsonNode;
import org.eclipse.deeplearning4j.llm.eval.AnswerExtractor;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.dataset.HuggingFaceDataset;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.MultipleChoiceAccuracyMetric;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * TruthfulQA benchmark for measuring model truthfulness.
 * Uses MC1 format (single correct answer from mc1_targets).
 *
 * HF schema: {question, mc1_targets: {choices: [...], labels: [0,1,0,0]},
 *              mc2_targets: {choices: [...], labels: [1,0,0,0]}}
 */
public class TruthfulQABenchmark implements BenchmarkTask {

    private static final String[] CHOICE_LABELS = {"A", "B", "C", "D", "E", "F", "G", "H"};

    @Override
    public String name() {
        return "truthfulqa";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.createWithTransformer(
                "truthful_qa", "multiple_choice", "validation", 0,
                (row, index) -> {
                    String question = row.has("question") ? row.get("question").asText() : "";

                    List<String> choices = new ArrayList<>();
                    String correctLabel = null;
                    JsonNode mc1 = row.get("mc1_targets");
                    if (mc1 != null) {
                        JsonNode choicesNode = mc1.get("choices");
                        JsonNode labelsNode = mc1.get("labels");
                        if (choicesNode != null && choicesNode.isArray()) {
                            for (int i = 0; i < choicesNode.size(); i++) {
                                choices.add(choicesNode.get(i).asText());
                                if (labelsNode != null && i < labelsNode.size()
                                        && labelsNode.get(i).asInt() == 1) {
                                    correctLabel = i < CHOICE_LABELS.length ?
                                            CHOICE_LABELS[i] : String.valueOf(i);
                                }
                            }
                        }
                    }

                    return EvalSample.builder()
                            .id(String.valueOf(index))
                            .input(question)
                            .choices(choices)
                            .references(correctLabel != null ? List.of(correctLabel) : List.of())
                            .build();
                });
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        StringBuilder sb = new StringBuilder();
        sb.append("Q: ").append(sample.getInput()).append("\n");
        if (sample.getChoices() != null) {
            for (int i = 0; i < sample.getChoices().size(); i++) {
                String label = i < CHOICE_LABELS.length ? CHOICE_LABELS[i] : String.valueOf(i + 1);
                sb.append(label).append(". ").append(sample.getChoices().get(i)).append("\n");
            }
        }
        sb.append("A:");
        return sb.toString();
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
        return 0;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 5;
    }
}
