package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Upsampling1D;
import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Max Pumperla
 */
public class Upsampling1DTest extends BaseDL4JTest {

    private int nExamples = 1;
    private int depth = 20;
    private int nChannelsIn = 1;
    private int inputLength = 28;
    private int size = 2;
    private int outputLength = inputLength * size;
    private INDArray epsilon = Nd4j.ones(nExamples, depth, outputLength);


    @Test
    public void testUpsampling1D() throws Exception {

        double[] outArray = new double[] {1., 1., 2., 2., 3., 3., 4., 4.};
        INDArray containedExpectedOut = Nd4j.create(outArray, new int[] {1, 1, 8});
        INDArray containedInput = getContainedData();
        INDArray input = getData();
        Layer layer =  getUpsampling1DLayer();

        INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);

        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(new long[] {nExamples, nChannelsIn, outputLength},
                        output.shape()));
        assertEquals(nChannelsIn, output.size(1), 1e-4);
    }


    @Test
    public void testUpsampling1DBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput =
                        Nd4j.create(new double[] {1., 3., 2., 6., 7., 2., 5., 5.},
                                new int[] {1, 1, 8});

        INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] {4., 8., 9., 10.},
                        new int[] {1, 1, 4});

        INDArray input = getContainedData();

        Layer layer = getUpsampling1DLayer();
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.getSecond().shape().length);

        INDArray input2 = getData();
        layer.activate(input2, false, LayerWorkspaceMgr.noWorkspaces());
        val depth = input2.size(1);

        epsilon = Nd4j.ones(5, depth, outputLength);

        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1));
    }


    private Layer getUpsampling1DLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
                        .layer(new Upsampling1D.Builder(size).build()).build();
        return conf.getLayer().instantiate(conf, null, 0,
                null, true);
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5, 5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        INDArray features = mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputLength, inputLength);
        return features.slice(0, 3);
    }

    private INDArray getContainedData() {
        INDArray ret = Nd4j.create
                (new double[] {1., 2., 3., 4.},
                        new int[] {1, 1, 4});
        return ret;
    }

}
