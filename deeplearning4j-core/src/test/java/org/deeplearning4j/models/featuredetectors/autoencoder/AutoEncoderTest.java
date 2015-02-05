package org.deeplearning4j.models.featuredetectors.autoencoder;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
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
                NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                        .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                        .corruptionLevel(0.6)
                        .iterations(100).iterationListener(new IterationListener() {
                            @Override
                            public void iterationDone(Model model, int iteration) {
                                if (iteration > 0 && iteration % 20 == 0) {
                                    NeuralNetPlotter plotter = new NeuralNetPlotter();
                                    Layer l = (Layer) model;
                                    plotter.renderFilter(l.getParam(PretrainParamInitializer.WEIGHT_KEY));

                                    INDArray gradient = l.getGradient().gradient();
                                    GradientAdjustment.updateGradientAccordingToParams(l.conf(),0,l.getOptimizer().getAdaGrad(),gradient,l.params(),l.batchSize());

                                }
                            }
                        })
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(new MersenneTwister(123))
                        .learningRate(1e-1f).nIn(784).nOut(600).build();

                fetcher.fetch(100);
                DataSet d2 = fetcher.next();

                INDArray input = d2.getFeatureMatrix();
                LayerFactory layerFactory = LayerFactories.getFactory(AutoEncoder.class);
                AutoEncoder da = layerFactory.create(conf);
                assertEquals(da.params(),da.params());
                assertEquals(471784,da.params().length());
                da.setParams(da.params());
                da.fit(input);
        }



}
