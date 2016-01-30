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

package org.deeplearning4j.spark.impl.computationgraph;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/** Function used to score a DataSet using a ComputationGraph */
public class ScoreFlatMapFunctionCGDataSet implements FlatMapFunction<Iterator<DataSet>, Double> {

    private String json;
    private Broadcast<INDArray> params;
    private static Logger log = LoggerFactory.getLogger(IterativeReduceFlatMapCG.class);

    public ScoreFlatMapFunctionCGDataSet(String json, Broadcast<INDArray> params){
        this.json = json;
        this.params = params;
    }

    @Override
    public Iterable<Double> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if(!dataSetIterator.hasNext()) {
            return Collections.singletonList(0.0);
        }
        List<DataSet> collect = new ArrayList<>();
        while(dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }

        DataSet data = DataSet.merge(collect,false);
        if(log.isDebugEnabled()) {
            log.debug("Scoring {} examples with data {}",data.numExamples(), data.labelCounts());
        }

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        network.init();
        INDArray val = params.value();
        if(val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcast set parameters");
        network.setParams(val);

        double score = network.score(data,false);
        if(network.conf().isMiniBatch()) score *= data.getFeatureMatrix().size(0);
        return Collections.singletonList(score);
    }
}
