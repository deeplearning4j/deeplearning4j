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
import org.deeplearning4j.arbiter.scoring.util.ScoreUtil;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Score function that calculates the test set loss
 * on a test set for a {@link MultiLayerNetwork} or {@link ComputationGraph}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class TestSetLossScoreFunction extends BaseNetScoreFunction {
    @JsonProperty
    private final boolean average;

    public TestSetLossScoreFunction() {
        this(true);
    }

    public TestSetLossScoreFunction(boolean average) {
        this.average = average;
    }


    @Override
    public boolean minimize() {
        return true;
    }

    @Override
    public String toString() {
        return "TestSetLossScoreFunction()";
    }

    @Override
    public double score(MultiLayerNetwork net, DataSetIterator iterator) {
        return ScoreUtil.score(net, iterator, average);
    }

    @Override
    public double score(MultiLayerNetwork net, MultiDataSetIterator iterator) {
        throw new UnsupportedOperationException("Cannot evaluate MultiLayerNetwork on MultiDataSetIterator");
    }

    @Override
    public double score(ComputationGraph graph, DataSetIterator iterator) {
        return ScoreUtil.score(graph, iterator, average);
    }

    @Override
    public double score(ComputationGraph graph, MultiDataSetIterator iterator) {
        return ScoreUtil.score(graph, iterator, average);
    }
}
