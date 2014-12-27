package org.deeplearning4j.models.featuredetectors.autoencoder;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.models.featuredetectors.da.DenoisingAutoEncoder;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.DefaultLayerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DenoisingAutoEncoderTest {

        @Test
        public void testDenoisingAutoEncoder() throws Exception {

                MnistDataFetcher fetcher = new MnistDataFetcher(true);
                NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                        .iterations(100)
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(new MersenneTwister(123))
                        .learningRate(1e-1f).nIn(784).nOut(600).build();

                fetcher.fetch(10);
                DataSet d2 = fetcher.next();

                INDArray input = d2.getFeatureMatrix();
                LayerFactory layerFactory = new DefaultLayerFactory(DenoisingAutoEncoder.class);
                DenoisingAutoEncoder da = layerFactory.create(conf);

                assertEquals(471000,da.params().length());

                da.fit(input);


        }



}
