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
    public void testNumExamplesMatch() throws Exception  {
        DataSetIterator mnist = new MnistDataSetIterator(10,10);
        DataSet next = mnist.next();

        Layer layer = getCNNConfig(1, 2, 9, 9, new int[] {3,3}, 1);

        INDArray input = next.getFeatureMatrix().reshape(next.numExamples(),1,28,28);
        INDArray conv = layer.activate(input);

        assertEquals(input.slices(), conv.slices());

    }


    @Test
    public void testFeatureMapSpaceSize() throws Exception  {
        int inputWidth = 28;
        int inputHeight = 28;
        int kernelWidth = 9;
        int kernelHeight = 9;
        int[] stride = new int[] {3,3};
        int padding = 1;
        int nIn = 1;
        int nOut = 2;

        DataSetIterator mnist = new MnistDataSetIterator(10, 10);
        DataSet next = mnist.next();

        Layer layer = getCNNConfig(nIn, nOut, kernelWidth, kernelHeight, stride, padding);

        INDArray input = next.getFeatureMatrix().reshape(next.numExamples(),1, inputWidth, inputHeight);
        INDArray conv = layer.activate(input);

        int featureMapWidth = (inputWidth - kernelWidth) + (2*padding) / stride[0] + 1;
        assertEquals(featureMapWidth, conv.shape()[0]);
        assertEquals(nOut, conv.shape()[2]);

    }



    private static Layer getCNNConfig(int nIn, int nOut, int kernelWidth, int kernelHeight, int[] stride, int padding){
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("relu")
                .iterations(1)
                .stride(stride)
                .padding(padding)
                .layer(new ConvolutionLayer.Builder(new int[]{kernelWidth, kernelHeight}, Convolution.Type.SAME)
                        .nIn(nIn)
                        .nOut(nOut)
                        .build())
                .build();
        return LayerFactories.getFactory(new ConvolutionLayer()).create(conf);

    }

}
