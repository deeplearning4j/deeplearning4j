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

package org.deeplearning4j.earlystopping.scorecalc.base;

import lombok.AllArgsConstructor;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Base score function based on an IEvaluation instance. Used for both MultiLayerNetwork and ComputationGraph
 *
 * @param <T> Type of model
 * @param <U> Type of evaluation
 */
public abstract class BaseIEvaluationScoreCalculator<T extends Model, U extends IEvaluation> implements ScoreCalculator<T> {

    protected MultiDataSetIterator iterator;
    protected DataSetIterator iter;

    protected BaseIEvaluationScoreCalculator(MultiDataSetIterator iterator){
        this.iterator = iterator;
    }

    protected BaseIEvaluationScoreCalculator(DataSetIterator iterator){
        this.iter = iterator;
    }

    @Override
    public double calculateScore(T network) {
        U eval = newEval();

        if(network instanceof MultiLayerNetwork){
            DataSetIterator i = (iter != null ? iter : new MultiDataSetWrapperIterator(iterator));
            eval = ((MultiLayerNetwork) network).doEvaluation(i, eval)[0];
        } else if(network instanceof ComputationGraph){
            MultiDataSetIterator i = (iterator != null ? iterator : new MultiDataSetIteratorAdapter(iter));
            eval = ((ComputationGraph) network).doEvaluation(i, eval)[0];
        } else {
            throw new RuntimeException("Unknown model type: " + network.getClass());
        }
        return finalScore(eval);
    }

    protected abstract U newEval();

    protected abstract double finalScore(U eval);


}
