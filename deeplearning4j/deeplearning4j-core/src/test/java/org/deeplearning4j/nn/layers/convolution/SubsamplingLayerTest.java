package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class SubsamplingLayerTest extends BaseDL4JTest {

    private int nExamples = 1;
    private int depth = 20; //channels & nOut
    private int nChannelsIn = 1;
    private int inputWidth = 28;
    private int inputHeight = 28;
    private int[] kernelSize = new int[] {2, 2};
    private int[] stride = new int[] {2, 2};

    int featureMapWidth = (inputWidth - kernelSize[0]) / stride[0] + 1;
    int featureMapHeight = (inputHeight - kernelSize[1]) / stride[0] + 1;
    private INDArray epsilon = Nd4j.ones(nExamples, depth, featureMapHeight, featureMapWidth);


    @Test
    public void testSubSampleMaxActivate() throws Exception {
        INDArray containedExpectedOut =
                        Nd4j.create(new double[] {5., 7., 6., 8., 4., 7., 5., 9.}, new int[] {1, 2, 2, 2});
        INDArray containedInput = getContainedData();
        INDArray input = getData();
        Layer layer = getSubsamplingLayer(SubsamplingLayer.PoolingType.MAX);

        INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);

        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(new long[] {nExamples, nChannelsIn, featureMapWidth, featureMapHeight},
                        output.shape()));
        assertEquals(nChannelsIn, output.size(1), 1e-4); // channels retained
    }

    @Test
    public void testSubSampleMeanActivate() throws Exception {
        INDArray containedExpectedOut =
                        Nd4j.create(new double[] {2., 4., 3., 5., 3.5, 6.5, 4.5, 8.5}, new int[] {1, 2, 2, 2});
        INDArray containedInput = getContainedData();
        INDArray input = getData();
        Layer layer = getSubsamplingLayer(SubsamplingLayer.PoolingType.AVG);

        INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);

        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(new long[] {nExamples, nChannelsIn, featureMapWidth, featureMapHeight},
                        output.shape()));
        assertEquals(nChannelsIn, output.size(1), 1e-4); // channels retained
    }

    //////////////////////////////////////////////////////////////////////////////////

    @Test
    public void testSubSampleLayerMaxBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput =
                        Nd4j.create(new double[] {1., 1., 1., 1., 1., 1., 1., 1.}, new int[] {1, 2, 2, 2});

        INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] {0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.,
                        0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.},
                        new int[] {1, 2, 4, 4});

        INDArray input = getContainedData();

        Layer layer = getSubsamplingLayer(SubsamplingLayer.PoolingType.MAX);
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.getSecond().shape().length);

        INDArray input2 = getData();
        layer.activate(input2, false, LayerWorkspaceMgr.noWorkspaces());
        long depth = input2.size(1);

        epsilon = Nd4j.ones(5, depth, featureMapHeight, featureMapWidth);

        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1)); // channels retained
    }

    @Test
    public void testSubSampleLayerAvgBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput =
                        Nd4j.create(new double[] {1., 2., 3., 4., 5., 6., 7., 8.}, new int[] {1, 2, 2, 2});

        INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] {0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5,
                        0.75, 0.75, 1., 1., 0.75, 0.75, 1., 1., 1.25, 1.25, 1.5, 1.5, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75,
                        2., 2., 1.75, 1.75, 2., 2.}, new int[] {1, 2, 4, 4});
        INDArray input = getContainedData();

        Layer layer = getSubsamplingLayer(SubsamplingLayer.PoolingType.AVG);
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertArrayEquals(expectedContainedEpsilonResult.shape(), containedOutput.getSecond().shape());

    }


    @Test(expected = IllegalStateException.class)
    public void testSubSampleLayerSumBackprop() throws Exception {
        Layer layer = getSubsamplingLayer(SubsamplingLayer.PoolingType.SUM);
        INDArray input = getData();
        layer.setInput(input, LayerWorkspaceMgr.noWorkspaces());
        layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
    }

    //////////////////////////////////////////////////////////////////////////////////

    private Layer getSubsamplingLayer(SubsamplingLayer.PoolingType pooling) {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
                        .layer(new SubsamplingLayer.Builder(pooling, new int[] {2, 2}).build()).build();

        return conf.getLayer().instantiate(conf, null, 0, null, true);
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5, 5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputWidth, inputHeight);
    }

    public INDArray getContainedData() {
        INDArray ret = Nd4j.create(new double[] {1., 1., 3., 7., 5., 1., 3., 3., 2., 2., 8., 4., 2., 6., 4., 4., 3., 3.,
                        6., 7., 4., 4., 6., 7., 5., 5., 9., 8., 4., 4., 9., 8.}, new int[] {1, 2, 4, 4});
        return ret;
    }

    private Gradient createPrevGradient() {
        Gradient gradient = new DefaultGradient();
        INDArray pseudoGradients = Nd4j.ones(nExamples, nChannelsIn, inputHeight, inputWidth);

        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, pseudoGradients);
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoGradients);
        return gradient;
    }

    //////////////////////////////////////////////////////////////////////////////////

    @Test(expected = Exception.class)
    public void testSubTooLargeKernel() {
        int imageHeight = 20;
        int imageWidth = 23;
        int nChannels = 1;
        int classes = 2;
        int numSamples = 200;

        int kernelHeight = 3;
        int kernelWidth = 3;

        DataSet trainInput;
        MultiLayerConfiguration.Builder builder =
                        new NeuralNetConfiguration.Builder().seed(123).list()
                                        .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(
                                                        kernelHeight, kernelWidth).stride(1, 1).nOut(2)
                                                                        .activation(Activation.RELU).weightInit(
                                                                                        WeightInit.XAVIER)
                                                                        .build())
                                        .layer(1, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                                                        .kernelSize(imageHeight - kernelHeight + 2, 1) //imageHeight-kernelHeight+1 is ok: full height
                                                        .stride(1, 1).build())
                                        .layer(2, new OutputLayer.Builder().nOut(classes).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).build())
                                        .backprop(true).pretrain(false)
                                        .setInputType(InputType.convolutional(imageHeight, imageWidth, nChannels));

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        INDArray emptyFeatures = Nd4j.zeros(numSamples, imageWidth * imageHeight * nChannels);
        INDArray emptyLables = Nd4j.zeros(numSamples, classes);

        trainInput = new DataSet(emptyFeatures, emptyLables);
        model.fit(trainInput);
    }



}
