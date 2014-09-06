package org.deeplearning4j.models.featuredetectors.rbm;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/5/14.
 */
public class ConvRBMTest {

    private static Logger log = LoggerFactory.getLogger(ConvRBMTest.class);

    @Test
    public void testMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.5f)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT).rng(new MersenneTwister(123))
                .learningRate(1e-1f).nIn(784).nOut(600).build();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();

        ConvolutionalRBM c = new ConvolutionalRBM.Builder().configure(conf).withFilterSize(new int[]{2,2})
                .withVisibleSize(new int[]{4,4})
                .withFmSize(new int[]{2,2}).build();
        c.fit(input);



    }

}
