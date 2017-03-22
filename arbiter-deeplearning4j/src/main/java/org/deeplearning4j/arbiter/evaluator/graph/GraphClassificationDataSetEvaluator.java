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
package org.deeplearning4j.arbiter.evaluator.graph;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 * A model evaluator for doing additional evaluation (classification evaluation)
 * for a ComputationGraph given a DataSetIterator
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
public class GraphClassificationDataSetEvaluator implements ModelEvaluator<ComputationGraph, Object, Evaluation> {
    private Map<String,Object> evalParams = null;


    @Override
    public Evaluation evaluateModel(ComputationGraph model, DataProvider<Object> dataProvider) {
        DataSetIterator iterator = ScoreUtil.getIterator(dataProvider.testData(evalParams));
        return ScoreUtil.getEvaluation(model,iterator);
    }
}
