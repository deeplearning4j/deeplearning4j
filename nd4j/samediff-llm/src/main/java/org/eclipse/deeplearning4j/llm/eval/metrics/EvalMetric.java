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

package org.eclipse.deeplearning4j.llm.eval.metrics;

import java.util.List;

/**
 * Interface for evaluation metrics used in LLM/VLM benchmark evaluation.
 */
public interface EvalMetric {

    /**
     * @return the name of this metric (e.g., "exact_match", "f1", "bleu-4")
     */
    String name();

    /**
     * Compute the metric score for a single prediction against reference answers.
     *
     * @param prediction the model's predicted answer
     * @param references the ground truth reference answer(s)
     * @return score in [0, 1] range
     */
    double score(String prediction, List<String> references);

    /**
     * Aggregate individual scores into a single summary score.
     * Default implementation returns the arithmetic mean.
     *
     * @param scores list of per-sample scores
     * @return aggregated score
     */
    default double aggregate(List<Double> scores) {
        if (scores == null || scores.isEmpty()) return 0.0;
        return scores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    /**
     * @return true if higher scores indicate better performance
     */
    boolean higherIsBetter();
}
