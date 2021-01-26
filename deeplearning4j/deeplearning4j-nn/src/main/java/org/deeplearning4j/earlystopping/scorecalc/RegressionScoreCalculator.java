/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.earlystopping.scorecalc.base.BaseIEvaluationScoreCalculator;
import org.deeplearning4j.nn.api.Model;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation.Metric;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Calculate the regression score of the network (MultiLayerNetwork or ComputationGraph) on a test set, using the
 * specified regression metric - {@link Metric}
 *
 * @author Alex Black
 */
public class RegressionScoreCalculator extends BaseIEvaluationScoreCalculator<Model, RegressionEvaluation> {

    protected final Metric metric;

    public RegressionScoreCalculator(Metric metric, DataSetIterator iterator){
        super(iterator);
        this.metric = metric;
    }

    @Override
    protected RegressionEvaluation newEval() {
        return new RegressionEvaluation();
    }

    @Override
    protected double finalScore(RegressionEvaluation eval) {
        return eval.scoreForMetric(metric);
    }

    @Override
    public boolean minimizeScore() {
        return metric.minimize();
    }
}
