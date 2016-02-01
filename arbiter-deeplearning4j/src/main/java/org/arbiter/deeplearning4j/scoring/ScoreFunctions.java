/*
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
 */

package org.arbiter.deeplearning4j.scoring;


import org.arbiter.deeplearning4j.scoring.graph.GraphTestSetLossScoreFunction;
import org.arbiter.deeplearning4j.scoring.graph.GraphTestSetLossScoreFunctionDataSet;
import org.arbiter.deeplearning4j.scoring.multilayer.TestSetLossScoreFunction;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public class ScoreFunctions {

    private ScoreFunctions(){
    }

    public static ScoreFunction<MultiLayerNetwork,DataSetIterator> testSetLoss(boolean average){
        return new TestSetLossScoreFunction(average);
    }

    public static ScoreFunction<ComputationGraph,DataSetIterator> testSetLossGraphDataSet(boolean average){
        return new GraphTestSetLossScoreFunctionDataSet(average);
    }

    public static ScoreFunction<ComputationGraph,MultiDataSetIterator> testSetLossGraph(boolean average){
        return new GraphTestSetLossScoreFunction(average);
    }

}
