package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ConvolutionLayerTest {

    // Mnist data example parameters
    private int inputWidth = 28;
    private int inputHeight = 28;

    private int[] kernelSize = new int[] {9, 9};
    private int[] stride = new int[] {2, 2};
    private int[] padding = new int[] {0,0};
    private int nChannelsIn = 1;
    private int depth = 10;
    private int nExamples = 10;
    int featureMapHeight = (inputHeight + padding[0] * 2 - kernelSize[0]) / stride[0] + 1;
    int  featureMapWidth = (inputWidth + padding[1] * 2 - kernelSize[1]) / stride[1] + 1;
//    private INDArray weights = Nd4j.rand() ;
//    private INDArray bias = Nd4j.rand();

    private INDArray epsilon = Nd4j.ones(nExamples, depth, featureMapHeight, featureMapWidth);

    @Test
    public void testCNNInputSetup() throws Exception{
        INDArray input = getMnistData();
        int[] stride = new int[] {3,3};
        int[] padding = new int[] {1,1};
        int[] kernelSize = new int[] {2,2};

        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize,  stride, padding);
        layer.activate(input);

        assertEquals(input, layer.input());
        assertArrayEquals(input.shape(), layer.input().shape());
    }

    @Test
    public void testFeatureMapShape() throws Exception  {
        INDArray input = getMnistData();

        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);
        INDArray convActivations = layer.activate(input);

        assertEquals(featureMapWidth, convActivations.size(2));
        assertEquals(depth, convActivations.size(1));
    }

    @Test
    public void testActivateResults()  {
        Layer layer = getContainedConfig();
        INDArray input = getContainedData();
        INDArray expectedOutput = Nd4j.create(new double[] {
                0.98201379,  0.98201379,  0.98201379,  0.98201379,  0.99966465,
                0.99966465,  0.99966465,  0.99966465,  0.98201379,  0.98201379,
                0.98201379,  0.98201379,  0.99966465,  0.99966465,  0.99966465,
                0.99966465,  0.98201379,  0.98201379,  0.98201379,  0.98201379,
                0.99966465,  0.99966465,  0.99966465,  0.99966465,  0.98201379,
                0.98201379,  0.98201379,  0.98201379,  0.99966465,  0.99966465,
                0.99966465,  0.99966465
        },new int[]{1,2,4,4});

        INDArray convActivations = layer.activate(input);

        assertArrayEquals(expectedOutput.shape(), convActivations.shape());
        assertEquals(expectedOutput, convActivations);

    }


    @Test
    public void testPreOutputMethod()  {
        Layer layer = getContainedConfig();
        INDArray col = getContainedCol();

        INDArray expectedOutput = Nd4j.create(new double[] {
                4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  8.,  8.,  8.,  8.,  8.,
                8.,  8.,  8.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  8.,  8.,
                8., 8., 8., 8., 8.,  8.
        },new int[]{1, 2, 4, 4});

        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer layer2 = (org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) layer;
        layer2.setCol(col);
        INDArray activation = layer2.preOutput(true);

        assertArrayEquals(expectedOutput.shape(), activation.shape());
        assertEquals(expectedOutput, activation);

    }




    @Test
    public void testBackpropResults()  {
        Layer layer = getContainedConfig();
        INDArray input = getContainedData();
        INDArray col = getContainedCol();
        INDArray epsilon = Nd4j.ones(1,2,4,4);

        INDArray expectedWeightGradient = Nd4j.create(new double[] {
                -1440., -1440., -1984., -1984., -1440., -1440., -1984., -1984.
        }, new int[]{2,1,2,2});
        INDArray expectedBiasGradient = Nd4j.create(new double[] {-544., -544.}, new int[]{1,2});
        INDArray expectedEpsilon = Nd4j.create(new double[] {
                -12., -12., -12., -12., -12., -12., -12., -12., -12., -12., -12.,
                -12., -12., -12., -12., -12., -56., -56., -56., -56., -56., -56.,
                -56., -56., -56., -56., -56., -56., -56., -56., -56., -56., -12.,
                -12., -12., -12., -12., -12., -12., -12., -12., -12., -12., -12.,
                -12., -12., -12., -12., -56., -56., -56., -56., -56., -56., -56.,
                -56., -56., -56., -56., -56., -56., -56., -56., -56.
        },new int[]{1,1,8,8});

        layer.setInput(input);
        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer layer2 = (org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) layer;
        layer2.setCol(col);
        Pair<Gradient, INDArray> pair = layer2.backpropGradient(epsilon);

        assertArrayEquals(expectedEpsilon.shape(), pair.getSecond().shape());
        assertArrayEquals(expectedWeightGradient.shape(), pair.getFirst().getGradientFor("W").shape());
        assertArrayEquals(expectedBiasGradient.shape(), pair.getFirst().getGradientFor("b").shape());
        assertEquals(expectedEpsilon, pair.getSecond());
        assertEquals(expectedWeightGradient, pair.getFirst().getGradientFor("W"));
        assertEquals(expectedBiasGradient, pair.getFirst().getGradientFor("b"));

    }

    @Test
    public void testCalculateDelta() {
        Layer layer = getContainedConfig();
        INDArray input = getContainedData();
        INDArray col = getContainedCol();
        INDArray epsilon = Nd4j.ones(1,2,4,4);

        INDArray expectedOutput = Nd4j.create(new double[] {
                -12., -12., -12., -12., -56., -56., -56., -56., -12., -12., -12.,
                -12., -56., -56., -56., -56., -12., -12., -12., -12., -56., -56.,
                -56., -56., -12., -12., -12., -12., -56., -56., -56., -56.
        },new int[]{1, 2, 4, 4});

        layer.setInput(input);
        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer layer2 = (org.deeplearning4j.nn.layers.convolution.ConvolutionLayer) layer;
        layer2.setCol(col);
        INDArray delta = layer2.calculateDelta(epsilon);

        assertArrayEquals(expectedOutput.shape(), delta.shape());
        assertEquals(expectedOutput, delta);

    }

    //////////////////////////////////////////////////////////////////////////////////

    private static Layer getCNNConfig(int nIn, int nOut, int[] kernelSize, int[] stride, int[] padding){
        ConvolutionLayer layer = new ConvolutionLayer.Builder(kernelSize, stride, padding)
                .nIn(nIn)
                .nOut(nOut)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("sigmoid")
                .iterations(1)
                .layer(layer)
                .build();
        return LayerFactories.getFactory(conf).create(conf);

    }

    public Layer getContainedConfig(){
//        int inputWidth = input.shape()[0];
//        int inputHeight = input.shape()[1];

        int[] kernelSize = new int[] {2, 2};
        int[] stride = new int[] {2,2};
        int[] padding = new int[] {0,0};
        int nChannelsIn = 1;
        int depth = 2;
//        int featureMapWidth = (inputWidth + padding[0] * 2 - kernelSize[0]) / stride[0] + 1;
//        int featureMapHeight = (inputHeight + padding[1] * 2 - kernelSize[1]) / stride[0] + 1;
        INDArray W = Nd4j.create(new double[] {0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5}, new int[]{2,1,2,2});
        INDArray b = Nd4j.create(new double[] {1,1});
        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);
        layer.setParam("W", W);
        layer.setParam("b", b);

        return layer;

    }

    public INDArray getMnistData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5,5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputHeight, inputWidth);
    }

    public INDArray getContainedData() {
        INDArray ret = Nd4j.create(new double[]{
                1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                3, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4
        }, new int[]{1,1,8,8});
        return ret;
    }

    public INDArray getContainedCol() {
        return Nd4j.create(new double[] {
                1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3,
                3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4
        },new int[]{1,1,2,2,4,4});
    }

    private Gradient createPrevGradient() {
        Gradient gradient = new DefaultGradient();
        INDArray pseudoGradients = Nd4j.ones(nExamples, nChannelsIn, featureMapHeight, featureMapWidth);

        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, pseudoGradients);
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoGradients);
        return gradient;
    }



}
