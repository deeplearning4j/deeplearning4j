/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.scoring.graph;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;

/**
 * Calculate the test set accuracy on a MultiDataSetIterator, for a ComputationGraph with one output
 *
 * @author Alex Black
 */
public class GraphTestSetAccuracyScoreFunction extends BaseGraphTestSetEvaluationScoreFunction {

    @Override
    public String toString() {
        return "GraphTestSetAccuracyScoreFunction";
    }

    @Override
    public double score(ComputationGraph model, DataProvider<Object> dataProvider, Map<String, Object> dataParameters) {
        return getEvaluation(model, dataProvider, dataParameters).accuracy();
    }
}
