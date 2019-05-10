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
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCBinary;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function that calculates AUC (area under ROC curve) or AUPRC (area under precision/recall curve) on a test set
 * for a {@link MultiLayerNetwork} or {@link ComputationGraph}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED)  //JSON
public class ROCScoreFunction extends BaseNetScoreFunction {

    /**
     * Type of ROC evaluation to perform:<br>
     * ROC: use {@link ROC} to perform evaluation (single output binary classification)<br>
     * BINARY: use {@link ROCBinary} to perform evaluation (multi-output/multi-task binary classification)<br>
     * MULTICLASS: use {@link ROCMultiClass} to perform evaluation (1 vs. all multi-class classification)
     *
     */
    public enum ROCType {ROC, BINARY, MULTICLASS}

    /**
     * Metric to calculate.<br>
     * AUC: Area under ROC curve<br>
     * AUPRC: Area under precision/recall curve
     */
    public enum Metric {AUC, AUPRC};

    protected ROCType type;
    protected Metric metric;

    /**
     * @param type ROC type to use for evaluation
     * @param metric Evaluation metric to calculate
     */
    public ROCScoreFunction(@NonNull ROCType type, @NonNull Metric metric) {
        this.type = type;
        this.metric = metric;
    }

    @Override
    public String toString() {
        return "ROCScoreFunction(type=" + type + ",metric=" + metric + ")";
    }

    @Override
    public double score(MultiLayerNetwork net, DataSetIterator iterator) {
        switch (type){
            case ROC:
                ROC r = net.evaluateROC(iterator);
                return metric == Metric.AUC ? r.calculateAUC() : r.calculateAUCPR();
            case BINARY:
                ROCBinary r2 = net.doEvaluation(iterator, new ROCBinary())[0];
                return metric == Metric.AUC ? r2.calculateAverageAuc() : r2.calculateAverageAUCPR();
            case MULTICLASS:
                ROCMultiClass r3 = net.evaluateROCMultiClass(iterator);
                return metric == Metric.AUC ? r3.calculateAverageAUC() : r3.calculateAverageAUCPR();
            default:
                throw new RuntimeException("Unknown type: " + type);
        }
    }

    @Override
    public double score(MultiLayerNetwork net, MultiDataSetIterator iterator) {
        return score(net, new MultiDataSetWrapperIterator(iterator));
    }

    @Override
    public double score(ComputationGraph graph, DataSetIterator iterator) {
        return score(graph, new MultiDataSetIteratorAdapter(iterator));
    }

    @Override
    public double score(ComputationGraph net, MultiDataSetIterator iterator) {
        switch (type){
            case ROC:
                ROC r = net.evaluateROC(iterator);
                return metric == Metric.AUC ? r.calculateAUC() : r.calculateAUCPR();
            case BINARY:
                ROCBinary r2 = net.doEvaluation(iterator, new ROCBinary())[0];
                return metric == Metric.AUC ? r2.calculateAverageAuc() : r2.calculateAverageAUCPR();
            case MULTICLASS:
                ROCMultiClass r3 = net.evaluateROCMultiClass(iterator, 0);
                return metric == Metric.AUC ? r3.calculateAverageAUC() : r3.calculateAverageAUCPR();
            default:
                throw new RuntimeException("Unknown type: " + type);
        }
    }

    @Override
    public boolean minimize() {
        return false;   //Want to maximize all evaluation metrics: Accuracy, F1, precision, recall, g-measure, mcc
    }
}
