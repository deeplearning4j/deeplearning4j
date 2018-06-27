package org.deeplearning4j.nn.layers.samediff;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.layers.samediff.testlayers.SameDiffMSELossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;

public class TestSameDiffOutput extends BaseDL4JTest {

    @Test
    public void testOutputMSELossLayer(){
        Nd4j.getRandom().setSeed(12345);

        MultiLayerConfiguration confSD = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                .layer(new SameDiffMSELossLayer())
                .build();

        MultiLayerConfiguration confStd = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(5).nOut(5).activation(Activation.TANH).build())
                .layer(new LossLayer.Builder().activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        MultiLayerNetwork netSD = new MultiLayerNetwork(confSD);
        netSD.init();

        MultiLayerNetwork netStd = new MultiLayerNetwork(confStd);
        netStd.init();

        INDArray in = Nd4j.rand(3, 5);
        INDArray label = Nd4j.rand(3,5);

        INDArray outSD = netSD.output(in);
        INDArray outStd = netStd.output(in);
        assertEquals(outStd, outSD);

        DataSet ds = new DataSet(in, label);
        double scoreSD = netSD.score(ds);
        double scoreStd = netStd.score(ds);
        assertEquals(scoreStd, scoreSD, 1e-6);

        for( int i=0; i<3; i++ ){
            netSD.fit(ds);
            netStd.fit(ds);

            assertEquals(netStd.params(), netSD.params());
            assertEquals(netStd.getFlattenedGradients(), netSD.getFlattenedGradients());
        }
    }

}
