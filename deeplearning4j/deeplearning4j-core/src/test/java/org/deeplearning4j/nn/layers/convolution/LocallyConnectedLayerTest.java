package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * @author Max Pumperla
 */
public class LocallyConnectedLayerTest extends BaseDL4JTest {

    @Before
    public void before() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.EPS_THRESHOLD = 1e-4;
    }

    @Test
    public void test2dForward(){
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4)
                        .updater(new Nesterovs(0.9)).dropOut(0.5)
                        .list()
                        .layer(new LocallyConnected2D.Builder().kernelSize(8, 8).nIn(3)
                                                        .stride(4, 4).nOut(16).dropOut(0.5)
                                                        .convolutionMode(ConvolutionMode.Strict)
                                                        .setInputSize(28, 28)
                                                        .activation(Activation.RELU).weightInit(
                                                                        WeightInit.XAVIER)
                                                        .build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS) //output layer
                                        .nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutionalFlat(28, 28, 3)).backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.ones(10, 3, 28, 28);
        INDArray output = network.output(input, false);

        assert Arrays.equals(output.shape(), new long[] {10, 10});
    }

    @Test
    public void test1dForward(){
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4)
                .updater(new Nesterovs(0.9)).dropOut(0.5)
                .list()
                .layer(new LocallyConnected1D.Builder().kernelSize(8).nIn(3)
                        .stride(4).nOut(16).dropOut(0.5)
                        .convolutionMode(ConvolutionMode.Strict)
                        .setInputSize(28)
                        .activation(Activation.RELU).weightInit(
                                WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS) //output layer
                        .nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.recurrent(3,  28)).backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        INDArray input = Nd4j.ones(10, 3, 28);
        INDArray output = network.output(input, false);

        System.out.println(Arrays.toString(output.shape()));
        //assert Arrays.equals(output.shape(), new long[] {10, 10});
    }

}
