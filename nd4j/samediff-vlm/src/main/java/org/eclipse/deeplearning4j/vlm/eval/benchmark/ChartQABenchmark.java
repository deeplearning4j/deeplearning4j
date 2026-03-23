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

import org.eclipse.deeplearning4j.llm.eval.benchmark.BenchmarkTask;
import org.eclipse.deeplearning4j.llm.eval.benchmark.OutputType;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.dataset.HuggingFaceDataset;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.RelaxedAccuracyMetric;

import java.io.IOException;
import java.util.List;

/**
 * ChartQA benchmark for chart understanding and QA.
 * Primary metric: relaxed accuracy (5% numeric tolerance OR exact string match).
 *
 * HF schema: {image, query, label: ["14"], human_or_machine: 0/1}
 * label is a List[string] (use referencesField).
 */
public class ChartQABenchmark implements BenchmarkTask {

    @Override
    public String name() {
        return "chartqa";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.create("HuggingFaceM4/ChartQA", null, "test",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .inputField("query")
                        .referencesField("label")
                        .imagePathField("image")
                        .build());
    }

    @Override
    public String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples) {
        return sample.getInput();
    }

    @Override
    public String extractAnswer(String modelOutput) {
        return modelOutput != null ? modelOutput.trim() : "";
    }

    @Override
    public EvalMetric primaryMetric() {
        return new RelaxedAccuracyMetric();
    }

    @Override
    public List<EvalMetric> allMetrics() {
        return List.of(new RelaxedAccuracyMetric(), new ExactMatchMetric());
    }

    @Override
    public OutputType outputType() {
        return OutputType.GENERATE_UNTIL;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 100;
    }
}
