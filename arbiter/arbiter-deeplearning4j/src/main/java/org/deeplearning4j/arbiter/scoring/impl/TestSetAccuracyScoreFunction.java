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

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Score function that calculates the accuracy on a
 * test set for a {@link MultiLayerNetwork} or {@link ComputationGraph}
 *
 * @author Alex Black
 * @deprecated Use {@link EvaluationScoreFunction}
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Deprecated
public class TestSetAccuracyScoreFunction extends BaseNetScoreFunction {


    @Override
    public String toString() {
        return "TestSetAccuracyScoreFunction()";
    }

    @Override
    public double score(MultiLayerNetwork net, DataSetIterator iterator) {
        Evaluation e = net.evaluate(iterator);
        return e.accuracy();
    }

    @Override
    public double score(MultiLayerNetwork net, MultiDataSetIterator iterator) {
        throw new UnsupportedOperationException("Cannot evaluate MultiLayerNetwork on MultiDataSetIterator");
    }

    @Override
    public double score(ComputationGraph graph, DataSetIterator iterator) {
        Evaluation e = graph.evaluate(iterator);
        return e.accuracy();
    }

    @Override
    public double score(ComputationGraph graph, MultiDataSetIterator iterator) {
        Evaluation e = graph.evaluate(iterator);
        return e.accuracy();
    }

    @Override
    public boolean minimize() {
        return false;
    }
}
