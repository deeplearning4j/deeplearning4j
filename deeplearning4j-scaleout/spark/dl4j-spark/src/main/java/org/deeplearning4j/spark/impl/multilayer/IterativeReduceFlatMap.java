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

package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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


public class IterativeReduceFlatMap implements FlatMapFunction<Iterator<DataSet>,Tuple3<INDArray,Updater,Double>> {
    protected static Logger log = LoggerFactory.getLogger(IterativeReduceFlatMap.class);

    protected MultiLayerNetwork network;

    private final Accumulator<Double> best_score_acc;

    /**
     * Pass in network and the bestScore
     * @param network      the network
     * @param bestScoreAcc accumulator which tracks best score seen
     */
    public IterativeReduceFlatMap(MultiLayerNetwork network,
                                  Accumulator<Double> bestScoreAcc) {
        this.best_score_acc = bestScoreAcc;
        this.network = network;
    }

    @Override
    public Iterable<Tuple3<INDArray, Updater, Double>> call(Iterator<DataSet> dataSetIterator) throws Exception {
        if (!dataSetIterator.hasNext()) {
            return Collections.singletonList(new Tuple3<INDArray, Updater, Double>(Nd4j.zeros(network.params().shape()), null, 0.0));
        }
        List<DataSet> collect = new ArrayList<>();
        while (dataSetIterator.hasNext()) {
            collect.add(dataSetIterator.next());
        }
        DataSet data = DataSet.merge(collect, false);
        if (log.isDebugEnabled()) {
            log.debug("Training on {} examples with data {}", data.numExamples(), data.labelCounts());
        }
        network.fit(data);
        return Collections.singletonList(new Tuple3<>(network.params(false), network.getUpdater(), network.score()));

    }
}
