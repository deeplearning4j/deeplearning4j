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

import org.eclipse.deeplearning4j.llm.eval.dataset.EvalDataset;
import org.eclipse.deeplearning4j.llm.eval.dataset.EvalSample;
import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;

import java.io.IOException;
import java.util.List;

/**
 * Interface for benchmark task definitions.
 */
public interface BenchmarkTask {

    String name();

    EvalDataset loadDataset() throws IOException;

    String formatPrompt(EvalSample sample, List<EvalSample> fewShotExamples);

    String extractAnswer(String modelOutput);

    EvalMetric primaryMetric();

    List<EvalMetric> allMetrics();

    OutputType outputType();

    default int defaultFewShot() {
        return 0;
    }

    default int defaultMaxNewTokens() {
        return 256;
    }
}
