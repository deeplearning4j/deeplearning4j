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

package org.eclipse.deeplearning4j.vlm.eval.benchmark;

import org.eclipse.deeplearning4j.llm.eval.AnswerExtractor;
import org.eclipse.deeplearning4j.llm.eval.benchmark.BenchmarkTask;
import org.eclipse.deeplearning4j.llm.eval.benchmark.OutputType;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.dataset.HuggingFaceDataset;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.F1Metric;

import java.io.IOException;
import java.util.List;

/**
 * POPE (Polling-based Object Probing Evaluation) benchmark.
 * Tests VLM hallucination by asking yes/no questions about objects in images.
 * Primary metric: F1 on yes/no classification.
 */
public class PopeBenchmark implements BenchmarkTask {

    @Override
    public String name() {
        return "pope";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.create("lmms-lab/POPE", null, "test",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .inputField("question")
                        .referenceField("answer")
                        .imagePathField("image")
                        .build());
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        return sample.getInput() + " Answer with yes or no.";
    }

    @Override
    public String extractAnswer(String modelOutput) {
        String yesNo = AnswerExtractor.extractYesNo(modelOutput);
        return yesNo != null ? yesNo : modelOutput.trim().toLowerCase();
    }

    @Override
    public EvalMetric primaryMetric() {
        return new F1Metric();
    }

    @Override
    public List<EvalMetric> allMetrics() {
        return List.of(new F1Metric(), new ExactMatchMetric());
    }

    @Override
    public OutputType outputType() {
        return OutputType.GENERATE_UNTIL;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 10;
    }
}
