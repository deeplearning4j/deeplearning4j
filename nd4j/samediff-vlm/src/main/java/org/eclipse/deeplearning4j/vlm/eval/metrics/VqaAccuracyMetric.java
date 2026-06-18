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

package org.eclipse.deeplearning4j.vlm.eval.metrics;

import org.eclipse.deeplearning4j.llm.eval.metrics.EvalMetric;
import org.eclipse.deeplearning4j.llm.eval.metrics.ExactMatchMetric;

import java.util.List;

/**
 * VQA accuracy metric following the VQA 2.0 evaluation protocol.
 * Score = min(count_matching_annotators / 3, 1.0).
 */
public class VqaAccuracyMetric implements EvalMetric {

    @Override
    public String name() {
        return "vqa_accuracy";
    }

    @Override
    public double score(String prediction, List<String> references) {
        if (prediction == null || references == null || references.isEmpty()) return 0.0;
        String normalizedPred = ExactMatchMetric.normalize(prediction);
        int matchCount = 0;
        for (String ref : references) {
            if (ExactMatchMetric.normalize(ref).equals(normalizedPred)) {
                matchCount++;
            }
        }
        return Math.min((double) matchCount / 3.0, 1.0);
    }

    @Override
    public boolean higherIsBetter() {
        return true;
    }
}
