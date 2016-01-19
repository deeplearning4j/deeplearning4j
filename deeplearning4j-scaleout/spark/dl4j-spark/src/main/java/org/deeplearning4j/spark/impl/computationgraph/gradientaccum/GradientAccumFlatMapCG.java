/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.spark.impl.computationgraph.gradientaccum;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Iterative reduce with
 * flat map using map partitions
 *
 * @author Adam Gibson
 */
public class GradientAccumFlatMapCG implements FlatMapFunction<Iterator<MultiDataSet>, Tuple3<Gradient,ComputationGraphUpdater,Double>> {

    private String json;
    private Broadcast<INDArray> params;
    private Broadcast<ComputationGraphUpdater> updater;
    private static Logger log = LoggerFactory.getLogger(GradientAccumFlatMapCG.class);

    /**
     * Pass in json configuration and baseline parameters
     * @param json json configuration for the network
     * @param params the parameters to use for the network
     */
    public GradientAccumFlatMapCG(String json, Broadcast<INDArray> params, Broadcast<ComputationGraphUpdater> updater) {
        this.json = json;
        this.params = params;
        this.updater = updater;
    }



    @Override
    public Iterable<Tuple3<Gradient,ComputationGraphUpdater,Double>> call(Iterator<MultiDataSet> dataSetIterator) throws Exception {
        if(!dataSetIterator.hasNext()) {
            return Collections.singletonList(new Tuple3<Gradient,ComputationGraphUpdater,Double>(new DefaultGradient(),null,0.0));
        }

        List<MultiDataSet> collect = new ArrayList<>();
        while(dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }

        MultiDataSet data = org.nd4j.linalg.dataset.MultiDataSet.merge(collect);

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        network.init();
        INDArray val = params.value();
        if(val.length() != network.numParams())
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParams(val);
        network.setUpdater(updater.getValue());
        network.fit(data);

        return Collections.singletonList(new Tuple3<>(network.gradient(),network.getUpdater(),network.score()));
    }
}
