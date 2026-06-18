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
import org.eclipse.deeplearning4j.vlm.eval.metrics.VqaAccuracyMetric;

import java.io.IOException;
import java.util.List;

/**
 * TextVQA benchmark for scene text reading and QA.
 * Primary metric: VQA accuracy (min(count/3, 1.0) over annotator set).
 *
 * HF schema: {image_id, question_id, question, question_tokens, image,
 *              image_width, image_height, flickr_original_url, flickr_300k_url,
 *              answers: ["ans1", "ans2", ...], image_classes, set_name, ocr_tokens}
 */
public class TextVQABenchmark implements BenchmarkTask {

    @Override
    public String name() {
        return "textvqa";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        return HuggingFaceDataset.create("lmms-lab/textvqa", null, "validation",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .idField("question_id")
                        .inputField("question")
                        .referencesField("answers")
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
        return new VqaAccuracyMetric();
    }

    @Override
    public List<EvalMetric> allMetrics() {
        return List.of(new VqaAccuracyMetric());
    }

    @Override
    public OutputType outputType() {
        return OutputType.GENERATE_UNTIL;
    }

    @Override
    public int defaultMaxNewTokens() {
        return 50;
    }
}
