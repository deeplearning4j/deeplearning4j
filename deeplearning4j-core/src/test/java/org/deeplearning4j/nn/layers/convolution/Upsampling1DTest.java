package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
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

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Max Pumperla
 */
public class Upsampling1DTest {

    private static final ActivationsFactory af = ActivationsFactory.getInstance();
    private static final GradientsFactory gf = GradientsFactory.getInstance();

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

        INDArray containedOutput = layer.activate(af.create(containedInput)).get(0);
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);

        INDArray output = layer.activate(af.create(input)).get(0);
        assertTrue(Arrays.equals(new int[] {nExamples, nChannelsIn, outputLength},
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
        layer.activate(af.create(input));

        Gradients containedOutput = layer.backpropGradient(gf.create(expectedContainedEpsilonInput));

        assertEquals(expectedContainedEpsilonResult, containedOutput.get(0));
        assertEquals(null, containedOutput.getParameterGradients().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.get(0).shape().length);

        INDArray input2 = getData();
        layer.activate(af.create(input2));
        int depth = input2.size(1);

        epsilon = Nd4j.ones(5, depth, outputLength);

        Gradients out = layer.backpropGradient(gf.create(epsilon));
        assertEquals(input.shape().length, out.get(0).shape().length);
        assertEquals(depth, out.get(0).size(1));
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
