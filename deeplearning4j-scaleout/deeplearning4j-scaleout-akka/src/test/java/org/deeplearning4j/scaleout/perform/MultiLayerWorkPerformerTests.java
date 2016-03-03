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

package org.deeplearning4j.scaleout.perform;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.scaleout.job.Job;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerWorkPerformerTests extends NeuralNetWorkPerformerTest {

    @Test
    public void testDbn() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(9e-1f)
                .iterations(10)
                .learningRate(1e-1f)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(4).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(1e-1, 1))
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3).nOut(3)
                        .activation("softmax")
                        .weightInit(WeightInit.ZERO).build())
                .build();

        String json = conf.toJson();

        Configuration conf2 = new Configuration();
        conf2.set(DeepLearningConfigurable.MULTI_LAYER_CONF,json);
        WorkerPerformer performer = new BaseMultiLayerNetworkWorkPerformer();
        performer.setup(conf2);
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(10);
        DataSet d = fetcher.next();
        Job j = new Job(d,"1");
        assumeJobResultNotNull(performer,j);
        performer.update(j.getResult());

    }


}
