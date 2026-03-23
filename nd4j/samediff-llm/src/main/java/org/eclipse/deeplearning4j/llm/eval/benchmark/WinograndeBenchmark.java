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
 * Winogrande benchmark for pronoun resolution / commonsense reasoning.
 * Binary choice task: fill in "_" with option1 or option2.
 *
 * HF schema: {sentence: "... _ ...", option1: "Sarah", option2: "Maria", answer: "1" or "2"}
 * answer is 1-indexed: "1" = option1, "2" = option2
 */
public class WinograndeBenchmark implements BenchmarkTask {

    @Override
    public String name() {
        return "winogrande";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.createWithTransformer(
                "allenai/winogrande", "winogrande_xl", "validation", 0,
                (row, index) -> {
                    String sentence = row.has("sentence") ? row.get("sentence").asText() : "";
                    String option1 = row.has("option1") ? row.get("option1").asText() : "";
                    String option2 = row.has("option2") ? row.get("option2").asText() : "";
                    String answer = row.has("answer") ? row.get("answer").asText() : "";

                    return EvalSample.builder()
                            .id(String.valueOf(index))
                            .input(sentence)
                            .choices(List.of(option1, option2))
                            .references(List.of(answer)) // "1" or "2"
                            .build();
                });
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        StringBuilder sb = new StringBuilder();
        sb.append("Fill in the blank with the correct option.\n\n");

        for (EvalSample ex : fewShotExamples) {
            sb.append(ex.getInput()).append("\n");
            if (ex.getChoices() != null && ex.getChoices().size() >= 2) {
                sb.append("A. ").append(ex.getChoices().get(0)).append("\n");
                sb.append("B. ").append(ex.getChoices().get(1)).append("\n");
            }
            if (!ex.getReferences().isEmpty()) {
                String ref = ex.getReferences().get(0);
                String label = "1".equals(ref) ? "A" : "B";
                sb.append("Answer: ").append(label).append("\n\n");
            }
        }

        sb.append(sample.getInput()).append("\n");
        if (sample.getChoices() != null && sample.getChoices().size() >= 2) {
            sb.append("A. ").append(sample.getChoices().get(0)).append("\n");
            sb.append("B. ").append(sample.getChoices().get(1)).append("\n");
        }
        sb.append("Answer:");
        return sb.toString();
    }

    @Override
    public String extractAnswer(String modelOutput) {
        String choice = AnswerExtractor.extractMultipleChoice(modelOutput);
        if (choice != null) {
            // Convert A/B to 1/2 to match Winogrande reference format
            if ("A".equals(choice)) return "1";
            if ("B".equals(choice)) return "2";
            return choice;
        }
        return modelOutput.trim();
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
