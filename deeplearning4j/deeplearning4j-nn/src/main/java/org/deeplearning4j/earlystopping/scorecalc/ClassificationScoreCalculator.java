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
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function for evaluating a MultiLayerNetwork according to an evaluation metric ({@link Evaluation.Metric} such
 * as accuracy, F1 score, etc.
 * Used for both MultiLayerNetwork and ComputationGraph
 *
 * @author Alex Black
 */
public class ClassificationScoreCalculator extends BaseIEvaluationScoreCalculator<Model, Evaluation> {

    protected final Evaluation.Metric metric;

    public ClassificationScoreCalculator(Evaluation.Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    public ClassificationScoreCalculator(Evaluation.Metric metric, MultiDataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    @Override
    protected Evaluation newEval() {
        return new Evaluation();
    }

    @Override
    protected double finalScore(Evaluation e) {
        return e.scoreForMetric(metric);
    }

    @Override
    public boolean minimizeScore() {
        //All classification metrics should be maximized: ACCURACY, F1, PRECISION, RECALL, GMEASURE, MCC
        return false;
    }
}
