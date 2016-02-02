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

package org.deeplearning4j.spark.impl.multilayer.gradientaccum;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

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
public class GradientAccumFlatMap implements FlatMapFunction<Iterator<DataSet>, Tuple2<Gradient,Updater>> {

    private String json;
    private Broadcast<INDArray> params;
    private Broadcast<Updater> updater;
    private static Logger log = LoggerFactory.getLogger(GradientAccumFlatMap.class);

    /**
     * Pass in json configuration and baseline parameters
     * @param json json configuration for the network
     * @param params the parameters to use for the network
     */
    public GradientAccumFlatMap(String json, Broadcast<INDArray> params, Broadcast<Updater> updater) {
        this.json = json;
        this.params = params;
        this.updater = updater;
    }



    @Override
    public Iterable<Tuple2<Gradient,Updater>> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if(!dataSetIterator.hasNext()) {
            return Collections.emptyList();
        }

        List<DataSet> collect = new ArrayList<>();
        while(dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }

        DataSet data = DataSet.merge(collect,false);
        if(log.isDebugEnabled()) {
            log.debug("Training on {} examples with data {}",data.numExamples(), data.labelCounts());
        }
        MultiLayerNetwork network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(json));
        network.init();
        //Clone/dup as params and updater are mutable (but: getValue() object from broadcast will be shared by all executors on same machine)
        INDArray val = params.value().dup();
        if(val.length() != network.numParams())
            throw new IllegalStateException("Network did not have same number of parameters as the broadcasted set parameters");
        network.setParameters(val);
        network.setUpdater(updater.getValue().clone());
        network.fit(data);

        return Collections.singletonList(new Tuple2<>(network.gradient(),network.getUpdater()));

    }
}
