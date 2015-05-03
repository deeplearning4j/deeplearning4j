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

package org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive;

import java.util.Arrays;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by agibsonccc on 1/8/15.
 */
public class RecursiveAutoEncoderTest {

    @Test
    public void testRecursiveAutoEncoder() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        LayerFactory layerFactory = LayerFactories.getFactory(RecursiveAutoEncoder.class);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .corruptionLevel(0.3).weightInit(WeightInit.VI)
                .iterations(100)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .learningRate(1e-1f).nIn(784).nOut(600).layerFactory(layerFactory).build();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();

        RecursiveAutoEncoder da = layerFactory.create(conf, 
                Arrays.<IterationListener>asList(new ScoreIterationListener(10)));
        da.setParams(da.params());
        da.fit(input);
    }

}
