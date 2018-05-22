package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 08/09/2016.
 */
public class LRNGradientCheckTests extends BaseDL4JTest {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-5;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-9;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }


    @Test
    public void testGradientLRNSimple() {
        Nd4j.getRandom().setSeed(12345);
        int minibatch = 10;
        int depth = 6;
        int hw = 5;
        int nOut = 4;
        INDArray input = Nd4j.rand(new int[] {minibatch, depth, hw, hw});
        INDArray labels = Nd4j.zeros(minibatch, nOut);
        Random r = new Random(12345);
        for (int i = 0; i < minibatch; i++) {
            labels.putScalar(i, r.nextInt(nOut), 1.0);
        }

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().updater(new NoOp())
                        .seed(12345L).weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 2)).list()
                        .layer(0, new ConvolutionLayer.Builder().nOut(6).kernelSize(2, 2).stride(1, 1)
                                        .activation(Activation.TANH).build())
                        .layer(1, new LocalResponseNormalization.Builder().build())
                        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nOut(nOut).build())
                        .setInputType(InputType.convolutional(hw, hw, depth)).pretrain(false).backprop(true);

        MultiLayerNetwork mln = new MultiLayerNetwork(builder.build());
        mln.init();

        if (PRINT_RESULTS) {
            for (int j = 0; j < mln.getnLayers(); j++)
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
        }

        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

        assertTrue(gradOK);
        TestUtils.testModelSerialization(mln);
    }

}
