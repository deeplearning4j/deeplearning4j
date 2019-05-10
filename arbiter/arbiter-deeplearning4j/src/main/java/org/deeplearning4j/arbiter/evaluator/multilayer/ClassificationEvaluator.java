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
