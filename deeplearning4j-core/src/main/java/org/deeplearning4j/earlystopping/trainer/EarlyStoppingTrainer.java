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

package org.deeplearning4j.earlystopping.trainer;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * Class for conducting early stopping training locally (single machine), for training a
 * {@link MultiLayerNetwork}. To train a {@link ComputationGraph}, use {@link EarlyStoppingGraphTrainer}
 */
public class EarlyStoppingTrainer extends BaseEarlyStoppingTrainer<MultiLayerNetwork> {

    private MultiLayerNetwork net;
    private boolean isMultiEpoch = false;


    public EarlyStoppingTrainer(EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration, MultiLayerConfiguration configuration,
                                DataSetIterator train) {
        this(earlyStoppingConfiguration, new MultiLayerNetwork(configuration), train);
        net.init();
    }

    public EarlyStoppingTrainer(EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net,
                                DataSetIterator train) {
        this(esConfig, net, train, null);
    }

    public EarlyStoppingTrainer(EarlyStoppingConfiguration<MultiLayerNetwork> esConfig, MultiLayerNetwork net,
                                DataSetIterator train, EarlyStoppingListener<MultiLayerNetwork> listener) {
        super(esConfig, net, train, null, listener);
        this.net = net;
    }

    @Override
    protected void fit(DataSet ds) {
        net.fit(ds);
    }

    @Override
    protected void fit(MultiDataSet mds) {
        throw new UnsupportedOperationException();
    }
}
