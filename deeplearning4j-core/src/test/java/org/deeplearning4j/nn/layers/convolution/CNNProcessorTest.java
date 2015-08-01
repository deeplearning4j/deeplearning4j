package org.deeplearning4j.nn.layers.convolution;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionInputPreProcessor;
import org.deeplearning4j.nn.layers.convolution.preprocessor.ConvolutionOutputPostProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by merlin on 7/31/15.
 */
public class CNNProcessorTest {
    private static int rows = 28;
    private static int cols = 28;

    @Test
    public void testCNNInputPreProcessor() {
        INDArray in2D = Nd4j.create(1, 784);
        INDArray in4D = Nd4j.create(20, 1, 28, 28);

        ConvolutionInputPreProcessor convProcessor = new ConvolutionInputPreProcessor(rows, cols, 1);

        INDArray check2to4 = convProcessor.preProcess(in2D);
        int val2to4 = check2to4.shape().length;
        assertTrue(val2to4 == 4);

        INDArray result2 = convProcessor.preProcess(in4D);
        int val4to4 = result2.shape().length;
        assertTrue(val4to4 == 4);

    }


    @Test
    public void testCNNInputPreProcessorBackprop() {
        INDArray in2D = Nd4j.create(1, 784);
        INDArray in3D = Nd4j.create(1, 784, 7);
        INDArray in4D = Nd4j.create(20, 1, 28, 28);

        ConvolutionInputPreProcessor convProcessor = new ConvolutionInputPreProcessor(rows, cols, 1);

        INDArray check2to2 = convProcessor.backprop(new Pair<>((Gradient)null,in2D)).getSecond();
        int val2to2 = check2to2.shape().length;
        assertTrue(val2to2 == 2);

        INDArray check3to2 = convProcessor.backprop(new Pair<>((Gradient)null,in3D)).getSecond();
        int val3to2 = check3to2.shape().length;
        assertTrue(val3to2 == 2);


        INDArray check4to2 = convProcessor.backprop(new Pair<>((Gradient)null,in4D)).getSecond();
        int val4to2 = check4to2.shape().length;
        assertTrue(val4to2 == 2);

    }

    @Test
    public void testCNNOutputPostProcessorBackprop() {
        INDArray in2D = Nd4j.create(1, 784);
        INDArray in3D = Nd4j.create(1, 784, 7);
        INDArray in4D = Nd4j.create(20, 1, 28, 28);

        ConvolutionOutputPostProcessor convProcessor = new ConvolutionOutputPostProcessor(rows, cols, 1);

        INDArray check2to2 = convProcessor.preProcess(in2D);
        int val2to2 = check2to2.shape().length;
        assertTrue(val2to2 == 2);

        INDArray check3to2 = convProcessor.preProcess(in3D);
        int val3to2 = check3to2.shape().length;
        assertTrue(val3to2 == 2);

        INDArray check4to2 = convProcessor.preProcess(in4D);
        int val4to2 = check4to2.shape().length;
        assertTrue(val4to2 == 2);

    }

    @Test
    public void testCNNOutputPostProcessor() {
        INDArray in2D = Nd4j.create(1, 784);
        INDArray in4D = Nd4j.create(20, 1, 28, 28);

        ConvolutionOutputPostProcessor convProcessor = new ConvolutionOutputPostProcessor(rows, cols, 1);

        INDArray check2to4 = convProcessor.backprop(new Pair<>((Gradient)null,in2D)).getSecond();
        int val2to4 = check2to4.shape().length;
        assertTrue(val2to4 == 4);

        INDArray result2 = convProcessor.backprop(new Pair<>((Gradient)null,in4D)).getSecond();
        int val4to4 = result2.shape().length;
        assertTrue(val4to4 == 4);

    }

    @Test
    public void testCNNInputPreProcessorMnist() throws Exception {
        int numSamples = 1;
        int batchSize = 1;

        DataSet mnistIter = new MnistDataSetIterator(batchSize, numSamples, true).next();
        MultiLayerNetwork model = getCNNMnistConfig();
        model.init();
        model.fit(mnistIter);

        int val2to4 = model.getLayer(0).input().shape().length;
        assertTrue(val2to4 == 4);

        int val4to4 = model.getLayer(1).input().shape().length;
        assertTrue(val4to4 == 4);

    }


    public static MultiLayerNetwork getCNNMnistConfig()  {

        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(5)
                .weightInit(WeightInit.XAVIER)
                .activationFunction("relu")
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{9, 9}, Convolution.Type.VALID)
                        .nIn(rows * cols)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(20)
                        .nOut(10)
                        .activation("softmax")
                        .build())
                .hiddenLayerSizes(50)
                .inputPreProcessor(0, new ConvolutionInputPreProcessor(rows, cols))
                .outputPostProcessor(1, new ConvolutionOutputPostProcessor())
                .build();
        return new MultiLayerNetwork(conf);

    }
}
