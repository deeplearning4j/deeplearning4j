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
import org.eclipse.deeplearning4j.llm.eval.metrics.AnlsMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;

import java.io.IOException;
import java.util.List;

/**
 * DocVQA (Document Visual Question Answering) benchmark.
 * Evaluates document understanding via question answering on document images.
 * Primary metric: ANLS (Average Normalized Levenshtein Similarity).
 *
 * HF schema: {questionId, question, question_types, image, docId,
 *              ucsf_document_id, ucsf_document_page_no, answers, data_split}
 * Note: test split has null answers; uses validation split for evaluation.
 */
public class DocVQABenchmark implements BenchmarkTask {

    @Override
    public String name() {
        return "docvqa";
    }

    @Override
    public EvalDataset loadDataset() throws IOException {
        // Use validation split — test split has null answers (no ground truth)
        return HuggingFaceDataset.create("lmms-lab/DocVQA", "DocVQA", "validation",
                HuggingFaceDataset.HfFieldMapping.builder()
                        .idField("questionId")
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
        return new AnlsMetric();
    }

    @Override
    public List<EvalMetric> allMetrics() {
        return List.of(new AnlsMetric(), new ExactMatchMetric());
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
