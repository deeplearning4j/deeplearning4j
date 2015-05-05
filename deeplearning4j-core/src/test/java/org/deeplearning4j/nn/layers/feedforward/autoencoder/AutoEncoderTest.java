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

package org.deeplearning4j.nn.layers.feedforward.autoencoder;

import java.util.Arrays;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.optimize.GradientAdjustment;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

public class AutoEncoderTest {
        @Test
        public void testAutoEncoder() throws Exception {

                MnistDataFetcher fetcher = new MnistDataFetcher(true);
                LayerFactory layerFactory = LayerFactories.getFactory(AutoEncoder.class);
                NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                        .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                        .corruptionLevel(0.6)
                        .iterations(100)
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .learningRate(1e-1f).nIn(784).nOut(600).layerFactory(layerFactory).build();

                IterationListener listener = new IterationListener() {
                    @Override
                    public void iterationDone(Model model, int iteration) {
                        if (iteration > 0 && iteration % 20 == 0) {
                            NeuralNetPlotter plotter = new NeuralNetPlotter();
                            Layer l = (Layer) model;
                            plotter.renderFilter(l.getParam(PretrainParamInitializer.WEIGHT_KEY));

                            INDArray gradient = l.gradient().gradient();
                            GradientAdjustment.updateGradientAccordingToParams(l.conf(),
                                0,l.getOptimizer().getAdaGrad(),gradient,l.params(),l.batchSize());

                        }
                    }
                };
                
                fetcher.fetch(100);
                DataSet d2 = fetcher.next();

                INDArray input = d2.getFeatureMatrix();
                AutoEncoder da = layerFactory.create(conf, Arrays.<IterationListener>asList(listener));
                assertEquals(da.params(),da.params());
                assertEquals(471784,da.params().length());
                da.setParams(da.params());
                da.fit(input);
        }



    @Test
    public void testBackProp() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        LayerFactory layerFactory = LayerFactories.getFactory(AutoEncoder.class);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .corruptionLevel(0.6)
                .iterations(100)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .learningRate(1e-1f).nIn(784).nOut(600).layerFactory(layerFactory).build();

        IterationListener listener = new IterationListener() {
            @Override
            public void iterationDone(Model model, int iteration) {
                if (iteration > 0 && iteration % 20 == 0) {
                    NeuralNetPlotter plotter = new NeuralNetPlotter();
                    Layer l = (Layer) model;
                    plotter.renderFilter(l.getParam(PretrainParamInitializer.WEIGHT_KEY));

                    INDArray gradient = l.gradient().gradient();
                    GradientAdjustment.updateGradientAccordingToParams(l.conf(),
                            0,l.getOptimizer().getAdaGrad(),gradient,l.params(),l.batchSize());

                }
            }
        };
        fetcher.fetch(100);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        AutoEncoder da = layerFactory.create(conf, Arrays.<IterationListener>asList(listener));
        Gradient g = new DefaultGradient();
        g.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, da.decode(da.activate(input)).sub(input));
        Gradient g2 = da.backwardGradient(da.decode(da.activate(input)),g);

    }



}
