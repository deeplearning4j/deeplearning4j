/*
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
package org.arbiter.deeplearning4j.scoring.graph;

import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Map;

public class GraphTestSetLossScoreFunctionDataSet implements ScoreFunction<ComputationGraph,DataSetIterator> {

    private final boolean average;

    public GraphTestSetLossScoreFunctionDataSet(){
        this(false);
    }

    public GraphTestSetLossScoreFunctionDataSet(boolean average){
        this.average = average;
    }

    @Override
    public double score(ComputationGraph model, DataProvider<DataSetIterator> dataProvider, Map<String,Object> dataParameters) {

        DataSetIterator testData = dataProvider.testData(dataParameters);

        //TODO: do this properly taking into account division by N, L1/L2 etc
        double sumScore = 0.0;
        int totalExamples = 0;
        while(testData.hasNext()){
            DataSet ds = testData.next();
            int numExamples = testData.numExamples();

            sumScore += numExamples*model.score(ds);
            totalExamples += numExamples;
        }

        if(!average) return sumScore;
        return sumScore / totalExamples;
    }

    @Override
    public String toString(){
        return "GraphTestSetLossScoreFunctionDataSet()";
    }
}
