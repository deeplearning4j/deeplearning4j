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
import java.util.List;

/**
 * ARC (AI2 Reasoning Challenge) benchmark.
 * Multiple-choice science questions in Easy and Challenge subsets.
 *
 * HF schema: {id, question, choices: {text: [...], label: [...]}, answerKey: "A"/"B"/"C"/"D"}
 */
public class ArcBenchmark implements BenchmarkTask {

    public enum Subset { EASY, CHALLENGE }

    private final Subset subset;

    public ArcBenchmark() {
        this(Subset.CHALLENGE);
    }

    public ArcBenchmark(Subset subset) {
        this.subset = subset;
    }

    @Override
    public String name() {
        return "arc-" + subset.name().toLowerCase();
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        String config = subset == Subset.EASY ? "ARC-Easy" : "ARC-Challenge";
        return HuggingFaceDataset.createWithTransformer(
                "allenai/ai2_arc", config, "test", 0,
                (row, index) -> {
                    String id = row.has("id") ? row.get("id").asText() : String.valueOf(index);
                    String question = row.has("question") ? row.get("question").asText() : "";

                    // choices is nested: {text: ["...", "..."], label: ["A", "B", "C", "D"]}
                    List<String> choiceTexts = null;
                    JsonNode choicesNode = row.get("choices");
                    if (choicesNode != null && choicesNode.has("text")) {
                        choiceTexts = HuggingFaceDataset.jsonArrayToStringList(choicesNode.get("text"));
                    }

                    String answerKey = row.has("answerKey") ? row.get("answerKey").asText() : "";

                    return EvalSample.builder()
                            .id(id)
                            .input(question)
                            .choices(choiceTexts)
                            .references(List.of(answerKey))
                            .build();
                });
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        StringBuilder sb = new StringBuilder();
        sb.append("Answer the following science question.\n\n");

        for (EvalSample ex : fewShotExamples) {
            sb.append("Question: ").append(ex.getInput()).append("\n");
            appendChoices(sb, ex);
            if (!ex.getReferences().isEmpty()) {
                sb.append("Answer: ").append(ex.getReferences().get(0)).append("\n\n");
            }
        }

        sb.append("Question: ").append(sample.getInput()).append("\n");
        appendChoices(sb, sample);
        sb.append("Answer:");
        return sb.toString();
    }

    private void appendChoices(StringBuilder sb, EvalSample sample) {
        if (sample.getChoices() != null) {
            String[] labels = {"A", "B", "C", "D", "E"};
            for (int i = 0; i < sample.getChoices().size(); i++) {
                String label = i < labels.length ? labels[i] : String.valueOf(i + 1);
                sb.append(label).append(". ").append(sample.getChoices().get(i)).append("\n");
            }
        }
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
        return 25;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 5;
    }
}
