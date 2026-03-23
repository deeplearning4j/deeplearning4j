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
import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;

import java.io.IOException;
import java.util.List;

/**
 * GSM8K (Grade School Math 8K) benchmark.
 * Math word problems evaluated by exact match on the extracted final number.
 */
public class Gsm8kBenchmark implements BenchmarkTask {

    @Override
    public String name() {
        return "gsm8k";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.create("openai/gsm8k", "main", "test",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .inputField("question")
                        .referenceField("answer")
                        .build());
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        StringBuilder sb = new StringBuilder();
        sb.append("Solve the following math problem step by step. ")
                .append("Put your final answer after ####.\n\n");

        for (EvalSample ex : fewShotExamples) {
            sb.append("Question: ").append(ex.getInput()).append("\n");
            if (!ex.getReferences().isEmpty()) {
                sb.append("Answer: ").append(ex.getReferences().get(0)).append("\n\n");
            }
        }

        sb.append("Question: ").append(sample.getInput()).append("\nAnswer:");
        return sb.toString();
    }

    @Override
    public String extractAnswer(String modelOutput) {
        String number = AnswerExtractor.extractNumber(modelOutput);
        return number != null ? number : modelOutput.trim();
    }

    @Override
    public EvalMetric primaryMetric() {
        return new ExactMatchMetric(false); // No normalization for numbers
    }

    @Override
    public List<EvalMetric> allMetrics() {
        return List.of(new ExactMatchMetric(false));
    }

    @Override
    public OutputType outputType() {
        return OutputType.GENERATE_UNTIL;
    }

    @Override
    public int defaultFewShot() {
        return 8;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 512;
    }
}
