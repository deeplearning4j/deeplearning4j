package org.deeplearning4j.nn.conf.preprocessor;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 **/

public class CNNProcessorTest {
    private static int rows = 28;
    private static int cols = 28;
    private static INDArray in2D = Nd4j.create(1, 784);
    private static INDArray in3D = Nd4j.create(20, 784, 7);
    private static INDArray in4D = Nd4j.create(20, 1, 28, 28);


    @Test
    public void testFeeForwardToCnnPreProcessor() {

        FeedForwardToCnnPreProcessor convProcessor = new FeedForwardToCnnPreProcessor(rows, cols, 1);

        INDArray check2to4 = convProcessor.preProcess(in2D,null);
        int val2to4 = check2to4.shape().length;
        assertTrue(val2to4 == 4);
        assertEquals(Nd4j.create(1, 1, 28, 28), check2to4);

        INDArray check4to4 = convProcessor.preProcess(in4D,null);
        int val4to4 = check4to4.shape().length;
        assertTrue(val4to4 == 4);
        assertEquals(Nd4j.create(20, 1, 28, 28), check4to4);

    }


    @Test
    public void testFeeForwardToCnnPreProcessorBackprop() {
        FeedForwardToCnnPreProcessor convProcessor = new FeedForwardToCnnPreProcessor(rows, cols, 1);
        convProcessor.preProcess(in2D,null);

        INDArray check2to2 = convProcessor.backprop(in2D,null);
        int val2to2 = check2to2.shape().length;
        assertTrue(val2to2 == 2);
        assertEquals(Nd4j.create(1, 784), check2to2);

        INDArray check3to2 = convProcessor.backprop(in3D,null);
        int val3to2 = check3to2.shape().length;
        assertTrue(val3to2 == 2);
        assertEquals(Nd4j.create(20, 5488), check3to2);


        INDArray check4to2 = convProcessor.backprop(in4D,null);
        int val4to2 = check4to2.shape().length;
        assertTrue(val4to2 == 2);
        assertEquals(Nd4j.create(20, 784), check4to2);

    }

    @Test
    public void testCnnToFeeForwardProcessor() {
        CnnToFeedForwardPreProcessor convProcessor = new CnnToFeedForwardPreProcessor(rows, cols, 1);

        INDArray check2to4 = convProcessor.backprop(in2D,null);
        int val2to4 = check2to4.shape().length;
        assertTrue(val2to4 == 4);
        assertEquals(Nd4j.create(1, 1, 28, 28), check2to4);

        INDArray check4to4 = convProcessor.backprop(in4D,null);
        int val4to4 = check4to4.shape().length;
        assertTrue(val4to4 == 4);
        assertEquals(Nd4j.create(20, 1, 28, 28), check4to4);

    }

    @Test
    public void testCnnToFeeForwardPreProcessorBackprop() {
        CnnToFeedForwardPreProcessor convProcessor = new CnnToFeedForwardPreProcessor(rows, cols, 1);
        convProcessor.preProcess(in4D,null);

        INDArray check2to2 = convProcessor.preProcess(in2D,null);
        int val2to2 = check2to2.shape().length;
        assertTrue(val2to2 == 2);
        assertEquals(Nd4j.create(1, 784), check2to2);

        INDArray check3to2 = convProcessor.preProcess(in3D,null);
        int val3to2 = check3to2.shape().length;
        assertTrue(val3to2 == 2);
        assertEquals(Nd4j.create(20, 5488), check3to2);

        INDArray check4to2 = convProcessor.preProcess(in4D,null);
        int val4to2 = check4to2.shape().length;
        assertTrue(val4to2 == 2);
        assertEquals(Nd4j.create(20, 784), check4to2);

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

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(5)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{9, 9},new int[]{1,1})
                        .nIn(1)
                        .nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(20)
                        .nOut(10)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(rows, cols, 1))
                .inputPreProcessor(2, new CnnToFeedForwardPreProcessor(rows, cols, 1))
        .build();
        return new MultiLayerNetwork(conf);

    }
}
