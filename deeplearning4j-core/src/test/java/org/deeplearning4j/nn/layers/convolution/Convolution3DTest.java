package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
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
    private int nChannelsOut = 1;
    private int nChannelsIn = 1;
    private int inputDepth = 2 * 2;
    private int inputWidth = 28 / 2;
    private int inputHeight = 28 / 2;

    private int[] kernelSize = new int[]{2, 2, 2};
    private int outputDepth = inputDepth - kernelSize[0] + 1;
    private int outputHeight = inputHeight - kernelSize[1] + 1;
    private int outputWidth = inputWidth - kernelSize[2] + 1;

    private INDArray epsilon = Nd4j.ones(nExamples, nChannelsOut, outputDepth, outputHeight, outputWidth);


    @Test
    public void testConvolution3dForwardSameMode() throws Exception {

        INDArray containedInput = getContainedData();
        Convolution3DLayer layer = (Convolution3DLayer) getConvolution3DLayer(ConvolutionMode.Same);

        assertTrue(layer.convolutionMode == ConvolutionMode.Same);

        INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());

        assertTrue(Arrays.equals(containedInput.shape(), containedOutput.shape()));

    }

    @Test
    public void testConvolution3dForwardValidMode() throws Exception {

        Convolution3DLayer layer = (Convolution3DLayer) getConvolution3DLayer(ConvolutionMode.Strict);

        assertTrue(layer.convolutionMode == ConvolutionMode.Strict);

        INDArray input = getData();
        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        System.out.println(Arrays.toString(input.shape()));

        System.out.println(Arrays.toString(output.shape()));
        assertTrue(Arrays.equals(new int[]{nExamples, nChannelsOut, outputDepth, outputWidth, outputHeight},
                output.shape()));
    }


    @Test
    public void testConvolution3DBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput =
                Nd4j.create(
                        new double[]{1.},
                        new int[]{1, 1, 1, 1, 1});

        INDArray expectedContainedEpsilonResult = Nd4j.create(
                new double[]{1., 1., 1., 1., 1., 1., 1., 1.},
                new int[]{1, 1, 2, 2, 2});

        INDArray input = getContainedData();

        Layer layer = getConvolution3DLayer(ConvolutionMode.Strict);
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput,
                LayerWorkspaceMgr.noWorkspaces());

        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.getSecond().shape().length);

        INDArray input2 = getData();
        layer.activate(input2, false, LayerWorkspaceMgr.noWorkspaces());
        int depth = input2.size(1);

        epsilon = Nd4j.ones(5, depth, outputHeight, outputWidth);

        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1));
    }


    private Layer getConvolution3DLayer(ConvolutionMode mode) {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
                .layer(new Convolution3D.Builder().kernelSize(kernelSize).nIn(nChannelsIn).nOut(nChannelsOut)
                        .stride(1, 1, 1).dilation(1, 1, 1).padding(0, 0, 0)
                        .dataFormat(Convolution3D.DataFormat.NCDHW).convolutionMode(mode).hasBias(false)
                        .build())
                .build();
        int numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.ones(1, numParams);
        return conf.getLayer().instantiate(conf, null, 0, params, true);
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5, 5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputDepth, inputHeight, inputWidth);
    }

    private INDArray getContainedData() {
        return Nd4j.create
                (new double[]{1., 2., 3., 4., 5., 6., 7., 8},
                        new int[]{1, 1, 2, 2, 2});
    }

}
