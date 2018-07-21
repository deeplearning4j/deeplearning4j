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

package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.earlystopping.scorecalc.base.BaseIEvaluationScoreCalculator;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCBinary;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Calculate ROC AUC (area under ROC curve) or AUCPR (area under precision recall curve) for a MultiLayerNetwork or
 * ComputationGraph
 *
 * @author Alex Black
 */
public class ROCScoreCalculator extends BaseIEvaluationScoreCalculator<Model, IEvaluation> {

    public enum ROCType {ROC, BINARY, MULTICLASS}
    public enum Metric {AUC, AUPRC};

    protected final ROCType type;
    protected final Metric metric;

    public ROCScoreCalculator(ROCType type, DataSetIterator iterator) {
        this(type, Metric.AUC, iterator);
    }

    public ROCScoreCalculator(ROCType type, MultiDataSetIterator iterator){
        this(type, Metric.AUC, iterator);
    }

    public ROCScoreCalculator(ROCType type, Metric metric, DataSetIterator iterator){
        super(iterator);
        this.type = type;
        this.metric = metric;
    }

    public ROCScoreCalculator(ROCType type, Metric metric, MultiDataSetIterator iterator){
        super(iterator);
        this.type = type;
        this.metric = metric;
    }


    @Override
    protected IEvaluation newEval() {
        switch (type){
            case ROC:
                return new ROC();
            case BINARY:
                return new ROCBinary();
            case MULTICLASS:
                return new ROCMultiClass();
            default:
                throw new IllegalStateException("Unknown type: " + type);
        }
    }

    @Override
    protected double finalScore(IEvaluation eval) {
        switch (type){
            case ROC:
                ROC r = (ROC)eval;
                return metric == Metric.AUC ? r.calculateAUC() : r.calculateAUCPR();
            case BINARY:
                ROCBinary r2 = (ROCBinary) eval;
                return metric == Metric.AUC ? r2.calculateAverageAuc() : r2.calculateAverageAuc();
            case MULTICLASS:
                ROCMultiClass r3 = (ROCMultiClass)eval;
                return metric == Metric.AUC ? r3.calculateAverageAUC() : r3.calculateAverageAUCPR();
            default:
                throw new IllegalStateException("Unknown type: " + type);
        }
    }

    @Override
    public boolean minimizeScore() {
        return false;   //Maximize AUC, AUPRC
    }
}
