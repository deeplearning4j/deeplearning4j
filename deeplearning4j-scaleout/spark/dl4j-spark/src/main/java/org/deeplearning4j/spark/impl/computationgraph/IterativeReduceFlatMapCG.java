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
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple3;

import java.util.*;

/**
 * Iterative reduce for ComputationGraph with flat map using map partitions
 */
public class IterativeReduceFlatMapCG implements FlatMapFunction<Iterator<MultiDataSet>,Tuple3<INDArray,ComputationGraphUpdater,Double>> {
    protected static Logger log = LoggerFactory.getLogger(IterativeReduceFlatMapCG.class);

    private String json;
    private Broadcast<INDArray> params;
    private Broadcast<ComputationGraphUpdater> updater;

    /**
     * Pass in json configuration and baseline parameters
     *
     * @param json         json configuration for the network
     * @param params       the parameters to use for the network
     */
    public IterativeReduceFlatMapCG(String json, Broadcast<INDArray> params, Broadcast<ComputationGraphUpdater> updater) {
        this.json = json;
        this.params = params;
        this.updater = updater;
        if (updater.getValue() == null)
            throw new IllegalArgumentException("Updater shouldn't be null");
    }


    @Override
    public Iterable<Tuple3<INDArray, ComputationGraphUpdater, Double>> call(Iterator<MultiDataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.emptyList();
        }
        List<MultiDataSet> collect = new ArrayList<>();
        while (dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }

        MultiDataSet data = org.nd4j.linalg.dataset.MultiDataSet.merge(collect);

        ComputationGraph network = new ComputationGraph(ComputationGraphConfiguration.fromJson(json));
        network.setInitDone(true);
        network.init();
        network.setListeners(new ScoreIterationListener(1));

        //Need to clone: parameters and updaters are mutable values -> .getValue() object will be shared by ALL executors on the same machine!
        INDArray val = params.getValue().dup();
        ComputationGraphUpdater upd = updater.getValue().clone();
        if (val.length() != network.numParams(false))
            throw new IllegalStateException("Network did not have same number of parameters as the broadcast parameters");
        network.setParams(val);
        network.setUpdater(upd);
        network.fit(data);

        return Collections.singletonList(new Tuple3<>(network.params(false), network.getUpdater(), network.score()));
    }
}
