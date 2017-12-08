package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Max Pumperla
 */
public class Convolution3DTest {

    private int nExamples = 1;
    private int nChannelsOut = 20;
    private int nChannelsIn = 1;
    private int inputWidth = 28;
    private int inputHeight = 28;
    private int inputDepth = 28;

    private int[] kernelSize = new int[] {2, 1, 2};
    private int outputHeight = inputHeight / kernelSize[0];
    private int outputWidth = inputWidth / kernelSize[1];
    private int outputDepth = inputDepth / kernelSize[2];

    private INDArray epsilon = Nd4j.ones(nExamples, nChannelsOut, outputHeight, outputWidth, outputDepth);


    @Test
    public void testConvolution3dForward() throws Exception {

        double[] outArray = new double[] {1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.};
        INDArray containedExpectedOut = Nd4j.create(outArray, new int[] {1, 1, 4, 4});
        INDArray containedInput = getContainedData();
        Layer layer = getConvolution3DLayer();

        INDArray containedOutput = layer.activate(containedInput);
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);

        INDArray input = getData();
        INDArray output = layer.activate(input);
        assertTrue(Arrays.equals(new int[] {nExamples, nChannelsIn, outputWidth, outputHeight},
                        output.shape()));
        assertEquals(nChannelsIn, output.size(1), 1e-4);
    }


    @Test
    public void testConvolution3DBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput =
                        Nd4j.create(new double[] {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.},
                                new int[] {1, 1, 4, 4});

        INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] {4., 4., 4., 4.},
                        new int[] {1, 1, 2, 2});

        INDArray input = getContainedData();

        Layer layer = getConvolution3DLayer();
        layer.activate(input);

        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput);

        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.getSecond().shape().length);

        INDArray input2 = getData();
        layer.activate(input2);
        int depth = input2.size(1);

        epsilon = Nd4j.ones(5, depth, outputHeight, outputWidth);

        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon);
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1));
    }


    private Layer getConvolution3DLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
                        .layer(new Convolution3D.Builder().kernelSize(kernelSize).nIn(nChannelsIn).nOut(nChannelsOut)
                                .build())
                .build();
        return conf.getLayer().instantiate(conf, null, 0, null, true);
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5, 5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputWidth, inputHeight);
    }

    private INDArray getContainedData() {
        INDArray ret = Nd4j.create
                (new double[] {1., 2., 3., 4., 5., 6., 7., 8.},
                        new int[] {1, 1, 2, 2, 2});
        return ret;
    }

}
