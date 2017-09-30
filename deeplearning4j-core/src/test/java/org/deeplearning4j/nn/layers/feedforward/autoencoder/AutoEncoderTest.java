/*-
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

package org.deeplearning4j.nn.layers.feedforward.autoencoder;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class AutoEncoderTest {

    private static final ActivationsFactory af = ActivationsFactory.getInstance();

    @Test
    public void testAutoEncoderBiasInit() {
        org.deeplearning4j.nn.conf.layers.AutoEncoder conf =
                        new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder().nIn(1).nOut(3).biasInit(1).build();
        

        //        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        int numParams = conf.initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.instantiate(null, null, 0, 1, params, true);

        assertEquals(1, layer.getParam("b").size(0));
    }


    @Test
    public void testAutoEncoder() throws Exception {

        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        org.deeplearning4j.nn.conf.layers.Layer conf = new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder().nIn(784).nOut(600)
                .corruptionLevel(0.6)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build();


        fetcher.fetch(100);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        int numParams = conf.initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        AutoEncoder da = (AutoEncoder) conf.instantiate(
                Collections.<IterationListener>singletonList(new ScoreIterationListener(1)), null, 0, 1, params, true);
        assertEquals(da.params(), da.params());
        assertEquals(471784, da.params().length());
        da.setParams(da.params());
        da.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));
        da.fit(af.create(input));
    }



    @Test
    public void testBackProp() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        //        LayerFactory layerFactory = LayerFactories.getFactory(new org.deeplearning4j.nn.conf.layers.AutoEncoder());
        org.deeplearning4j.nn.conf.layers.Layer conf = new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder().nIn(784).nOut(600)
                .corruptionLevel(0.6)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build();

        fetcher.fetch(100);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        int numParams = conf.initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        AutoEncoder da = (AutoEncoder) conf.instantiate(null, null, 0, 1, params, true);
        Gradient g = new DefaultGradient();
        g.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, da.decode(da.activate(ActivationsFactory.getInstance().create(input)).get(0)).sub(input));
    }

}
