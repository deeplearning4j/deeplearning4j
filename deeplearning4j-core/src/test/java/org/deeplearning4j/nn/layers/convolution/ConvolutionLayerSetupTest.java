package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ConvolutionLayerSetupTest {

    @Test
    public void testConvolutionLayerSetup() {
        MultiLayerConfiguration.Builder builder = inComplete();
        new ConvolutionLayerSetup(builder,28,28,1);
        MultiLayerConfiguration completed = complete().build();
        MultiLayerConfiguration test = builder.build();
        assertEquals(completed,test);

    }

    @Test
    public void testMnistLenet() {
        MultiLayerConfiguration mnistAssertion = mnistLenet();
        ConvolutionLayer firstLayer = (ConvolutionLayer) mnistAssertion.getConf(0).getLayer();
        assertEquals(1, firstLayer.getNIn());
        assertEquals(6, firstLayer.getNOut());
        assertArrayEquals(new int[]{5, 5}, firstLayer.getKernelSize());
        assertArrayEquals(new int[]{0, 0}, firstLayer.getPadding());
        assertArrayEquals(new int[]{1,1}, firstLayer.getStride());
        OutputLayer o = (OutputLayer) mnistAssertion.getConf(4).getLayer();
        assertEquals(500,o.getNIn());

        MultiLayerConfiguration.Builder incomplete = incompleteMnistLenet();
        new ConvolutionLayerSetup(incomplete,28,28,1);
        MultiLayerConfiguration testConf = incomplete.build();

        ConvolutionLayer firstLayerConv = (ConvolutionLayer) testConf.getConf(0).getLayer();
        assertEquals(1,firstLayerConv.getNIn());
        assertEquals(6, firstLayerConv.getNOut());
        assertArrayEquals(new int[]{1,1}, firstLayerConv.getStride());
        assertArrayEquals(new int[]{5, 5}, firstLayerConv.getKernelSize());

        SubsamplingLayer firstSubSampling = (SubsamplingLayer) testConf.getConf(1).getLayer();
        assertArrayEquals(new int[]{2,2},firstSubSampling.getStride());

        ConvolutionLayer secondLayerConv = (ConvolutionLayer) testConf.getConf(2).getLayer();
        assertEquals(6,secondLayerConv.getNIn());


        OutputLayer finalLayer = (OutputLayer) testConf.getConf(4).getLayer();
        assertEquals(150,finalLayer.getNIn());

    }

    public MultiLayerConfiguration.Builder incompleteMnistLenet() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(5)
                .layer(0,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(6)
                        .build())
                .layer(1,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{5,5},new int[]{2,2}).build())
                .layer(2,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(6)
                        .build())
                .layer(3,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{5,5},new int[]{2,2}).build())
                .layer(4,new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).build());
        return builder;
    }

    public MultiLayerConfiguration mnistLenet() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(5)
                .layer(0,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(6)
                        .build())
                .layer(1,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{5,5},new int[]{2,2}).build())
                .layer(2,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(6)
                        .build())
                .layer(3,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{5,5},new int[]{2,2}).build())
                .layer(4,new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(500).nOut(10).build()).build();
        return builder;
    }

    public MultiLayerConfiguration.Builder inComplete() {
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 10;
        int seed = 123;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{10, 10})
                        .nIn(nChannels)
                        .nOut(6)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(216)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        return builder;
    }


    public MultiLayerConfiguration.Builder complete() {
        final int numRows = 28;
        final int numColumns = 28;
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 10;
        int seed = 123;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{10, 10})
                        .nIn(nChannels)
                        .nOut(6)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                        .weightInit(WeightInit.XAVIER).kernelSize(10, 10)
                        .activation("relu")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(216)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(numRows, numColumns, 1))
                .inputPreProcessor(2, new CnnToFeedForwardPreProcessor(10,10,6))
                .backprop(true).pretrain(false);

        return builder;
    }




}
