package org.deeplearning4j.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.samediff.testlayers.SameDiffLoss;
import org.deeplearning4j.samediff.testlayers.SameDiffOutput;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.Assert.*;

@Slf4j
public class TestSameDiffLoss {

    @Test
    public void testSameDiffLossBasic() {

        int minibatch = 3;
        int nIn = 3;
        int nOut = 4;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new DenseLayer.Builder().nIn(3).nOut(4).activation(Activation.TANH).build())
                .layer(new SameDiffLoss.Builder().lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray in = Nd4j.rand(minibatch, nIn);
        INDArray out = net.output(in);
        assertArrayEquals(new int[]{minibatch, nOut}, out.shape());

        INDArray label = Nd4j.rand(minibatch, nOut);
        net.setLabels(label);
        net.computeGradientAndScore();
        double score = net.score();
        assertTrue(score > 0);
    }
}
