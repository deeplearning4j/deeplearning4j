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
import org.nd4j.linalg.convolution.Convolution;
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
    public void testFeedForwardNumExamplesMatch() throws Exception  {
        DataSetIterator mnist = new MnistDataSetIterator(10,10);
        DataSet next = mnist.next();


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("relu")
                .layer(new ConvolutionLayer.Builder(new int[]{9, 9}, Convolution.Type.SAME).nIn(1).nOut(2).build())
                .build();
        Layer convolutionLayer =  LayerFactories.getFactory(new ConvolutionLayer()).create(conf);

        INDArray input = next.getFeatureMatrix().reshape(next.numExamples(),1,28,28);
        INDArray conv = convolutionLayer.activate(input);

        int v = input.slices();
        assertEquals(input.slices(), conv.slices());


    }

}
