/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.scoring.impl;

import lombok.*;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function that calculates an evaluation {@link Evaluation.Metric} on the test set for a
 * {@link MultiLayerNetwork} or {@link ComputationGraph}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED)  //JSON
public class EvaluationScoreFunction extends BaseNetScoreFunction {

    protected Evaluation.Metric metric;

    /**
     * @param metric Evaluation metric to calculate
     */
    public EvaluationScoreFunction(@NonNull Evaluation.Metric metric) {
        this.metric = metric;
    }

    @Override
    public String toString() {
        return "EvaluationScoreFunction(metric=" + metric + ")";
    }

    @Override
    public double score(MultiLayerNetwork net, DataSetIterator iterator) {
        Evaluation e = net.evaluate(iterator);
        return e.scoreForMetric(metric);
    }

    @Override
    public double score(MultiLayerNetwork net, MultiDataSetIterator iterator) {
        return score(net, new MultiDataSetWrapperIterator(iterator));
    }

    @Override
    public double score(ComputationGraph graph, DataSetIterator iterator) {
        Evaluation e = graph.evaluate(iterator);
        return e.scoreForMetric(metric);
    }

    @Override
    public double score(ComputationGraph graph, MultiDataSetIterator iterator) {
        Evaluation e = graph.evaluate(iterator);
        return e.scoreForMetric(metric);
    }

    @Override
    public boolean minimize() {
        return false;   //Want to maximize all evaluation metrics: Accuracy, F1, precision, recall, g-measure, mcc
    }
}
