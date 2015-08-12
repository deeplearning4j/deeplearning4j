package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.rbm.RBM;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Before;
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
    private int inputWidth = 28;
    private int inputHeight = 28;

    private int[] kernelSize = new int[] {9, 9};
    private int[] stride = new int[] {2,2};
    private int[] padding = new int[] {0,0};
    private int nChannelsIn = 1;
    private int depth = 10;
    private int nExamples;
    private INDArray epsilon;



    @Test
    public void testCNNInputSetup() throws Exception{
        INDArray input = getData();
        int[] stride = new int[] {3,3};
        int[] padding = new int[] {1,1};

        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize,  stride, padding);
        layer.activate(input);

        assertEquals(input, layer.input());
        assertEquals(input.shape(), layer.input().shape());
    }

    @Test
    public void testFeatureMapSpaceSize() throws Exception  {
        INDArray input = getData();

        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);
        INDArray convActivations = layer.activate(input);

        int featureMapWidth = (inputWidth - kernelSize[0] + 2 * padding[0]) / stride[0] + 1;

        assertEquals(featureMapWidth, convActivations.size(2));
        assertEquals(depth, convActivations.size(0));
    }

    @Test
    public void testSubSampleLayerNoneBackpropShape() throws Exception{
        INDArray input = getData();

        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);
//        Gradient gradient = createPrevGradient();
//
//        Pair<Gradient, INDArray> out= layer.backpropGradient(epsilon, gradient, null);
//        assertEquals(nExamples, out.getSecond().size(1)); // depth retained
    }



    private static Layer getCNNConfig(int nIn, int nOut, int[] kernelSize, int[] stride, int[] padding){
       ConvolutionLayer layer = new ConvolutionLayer.Builder(kernelSize, stride, padding)
               .nIn(nIn)
               .nOut(nOut)
               .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("relu")
                .iterations(1)
                .layer(layer)
                .build();
        return LayerFactories.getFactory(conf).create(conf);

    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5,5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputWidth, inputHeight);
    }


//    private Gradient createPrevGradient() {
//        Gradient gradient = new DefaultGradient();
//        int outH; // TODO get this calculated
//        int outW; // TODO get this calculated
//        INDArray pseudoGradients = Nd4j.ones(nExamples, nChannelsIn, outH, outW);
//        epsilon = Nd4j.ones(nExamples, nChannelsIn, outH, outW);
//
//        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, pseudoGradients);
//        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoGradients);
//        return gradient;
//    }


}
