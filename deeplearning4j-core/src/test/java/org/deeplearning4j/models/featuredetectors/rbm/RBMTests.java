package org.deeplearning4j.models.featuredetectors.rbm;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.distributions.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 8/27/14.
 */
public class RBMTests {
    private static Logger log = LoggerFactory.getLogger(RBMTests.class);


    @Test
    public void testBasic() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        Model rbm = new RBM.Builder().configure(conf).withInput(input).build();
        rbm.fit(input);



    }

    @Test
    public void testMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.5f)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(new MersenneTwister(123))
                .learningRate(1e-1f).nIn(784).nOut(600).build();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();

        RBM rbm = new RBM.Builder()
                .configure(conf).withInput(input).build();

        rbm.fit(input);

    }


    @Test
    public void testSetGetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        Model rbm = new RBM.Builder().configure(conf).build();
        INDArray rand2 = Nd4j.rand(new int[]{1, rbm.numParams()});
        rbm.setParams(rand2);
        INDArray getParams = rbm.params();
        assertEquals(rand2,getParams);
    }





    @Test
    public void testCg() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        RBM rbm = new RBM.Builder().configure(conf).withInput(input).build();
        rbm.setInput(input);
        float value = rbm.score();
        rbm.contrastiveDivergence(1e-1,1,input);
        value = rbm.score();



    }

    @Test
    public void testGradient() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        RBM rbm = new RBM.Builder().configure(conf).withInput(input).build();
        rbm.setInput(input);
        float value = rbm.score();


        NeuralNetworkGradient grad2 = rbm.getGradient(new Object[]{1});
        rbm.fit(input);

    }



}
