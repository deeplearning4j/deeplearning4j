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
package org.deeplearning4j.arbiter.evaluator.multilayer;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.evaluation.ModelEvaluator;
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * A model evaluator for doing additional
 * evaluation (classification evaluation)
 * for a {@link MultiLayerNetwork} given a {@link DataSetIterator}
 *
 * @author Alex Black
 */
@NoArgsConstructor
@AllArgsConstructor
public class ClassificationEvaluator implements ModelEvaluator {
    private Map<String, Object> params = null;


    @Override
    public Evaluation evaluateModel(Object model, DataProvider dataProvider) {

        if (model instanceof MultiLayerNetwork) {
            DataSetIterator iterator = ScoreUtil.getIterator(dataProvider.testData(params));
            return ScoreUtil.getEvaluation((MultiLayerNetwork) model, iterator);
        } else {
            DataSetIterator iterator = ScoreUtil.getIterator(dataProvider.testData(params));
            return ScoreUtil.getEvaluation((ComputationGraph) model, iterator);
        }
    }

    @Override
    public List<Class<?>> getSupportedModelTypes() {
        return Arrays.<Class<?>>asList(MultiLayerNetwork.class, ComputationGraph.class);
    }

    @Override
    public List<Class<?>> getSupportedDataTypes() {
        return Arrays.<Class<?>>asList(DataSetIterator.class, MultiDataSetIterator.class);
    }
}
