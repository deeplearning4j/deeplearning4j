package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.rbm.RBM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class TestConvolutionLayer {

    @Test
    public void testFeedForward() throws Exception  {
        DataSetIterator mnist = new MnistDataSetIterator(100,100);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().featureMapSize(20,20)
                .activationFunction("relu").constrainGradientToUnitNorm(true)
                .convolutionType(org.deeplearning4j.nn.conf.layers.ConvolutionDownSampleLayer.ConvolutionType.MAX).filterSize(6,1,9,9)
                .layer(new ConvolutionLayer())
                .nIn(784).nOut(1).build();
        Layer convolutionLayer =  LayerFactories.getFactory(new ConvolutionLayer()).create(conf);
        DataSet next = mnist.next();
        INDArray input = next.getFeatureMatrix().reshape(next.numExamples(),1,28,28);
        INDArray conv = convolutionLayer.activate(input);
        assertEquals(input.slices(),conv.slices());


    }

}
