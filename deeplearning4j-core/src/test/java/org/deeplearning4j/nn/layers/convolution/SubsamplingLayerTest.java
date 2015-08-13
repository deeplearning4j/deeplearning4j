package org.deeplearning4j.nn.layers.convolution;

import static org.junit.Assert.*;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class SubsamplingLayerTest {

    private int nExamples;
    private int depth = 20; //depth & nOut
    private int nChannelsIn = 1;
    private int inputWidth = 28;
    private int inputHeight = 28;
    private int[] kernelSize = new int[] {2, 2};
    private int[] stride = new int[] {2,2};

    int featureMapWidth = (inputWidth - kernelSize[0]) / stride[0] + 1;
    int featureMapHeight = (inputHeight - kernelSize[1]) / stride[0] + 1;
    private INDArray epsilon = Nd4j.ones(nExamples, depth, featureMapHeight, featureMapWidth);


    @Test
    public void testSubSamplingLayerActivateShape() throws Exception {
        INDArray input = getData();
        SubsamplingLayer.PoolingType[] toTest = {
                SubsamplingLayer.PoolingType.MAX,
                SubsamplingLayer.PoolingType.AVG,
//                SubsamplingLayer.PoolingType.SUM, not implemented yet
        };

        for(SubsamplingLayer.PoolingType pool : toTest) {
            Layer model = getSubsamplingLayer(pool);

            INDArray output = model.activate(input);
            assertTrue(Arrays.equals(new int[]{nExamples, nChannelsIn, featureMapWidth, featureMapHeight}, output.shape()));
            assertEquals(nExamples, output.shape()[0], 1e-4); // depth retained
        }
    }

    @Test
    public void testSubSampleNoneActivate() throws Exception  {
        INDArray input = getData();
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.NONE);

        INDArray output = model.activate(input);
        assertTrue(Arrays.equals(new int[]{nExamples, nChannelsIn, inputWidth, inputHeight}, output.shape()));
        assertEquals(nExamples, output.shape()[0], 1e-4); // depth retained
    }

    @Test (expected=IllegalStateException.class)
    public void testSubSampleSumActivate() throws Exception  {
        INDArray input = getData();
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.SUM);

        model.activate(input);
    }

    //////////////////////////////////////////////////////////////////////////////////

    @Test
    public void testSubSampleLayerMaxBackprop() throws Exception {
        INDArray input = getData();
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.MAX);

        Gradient gradient = createPrevGradient();
        model.activate(input);

        Pair<Gradient, INDArray> out = model.backpropGradient(epsilon, gradient, null);
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1)); // depth retained
    }

    @Test
    public void testSubSampleLayerAvgBackprop() throws Exception{
        INDArray input = getData();
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.AVG);
        Gradient gradient = createPrevGradient();

        model.activate(input);

        Pair<Gradient, INDArray> out = model.backpropGradient(epsilon, gradient, null);
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1)); // depth retained
    }

    @Test
    public void testSubSampleLayerNoneBackprop() {
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.NONE);
        Gradient gradient = createPrevGradient();

        Pair<Gradient, INDArray> out= model.backpropGradient(epsilon, gradient, null);
        assertEquals(depth, out.getSecond().size(1)); // depth retained
    }


    @Test (expected=IllegalStateException.class)
    public void testSubSampleLayerSumBackprop() {
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.SUM);
        Gradient gradient = createPrevGradient();

        Pair<Gradient, INDArray> out = model.backpropGradient(epsilon, gradient, null);
    }

    //////////////////////////////////////////////////////////////////////////////////

    private Layer getSubsamplingLayer(SubsamplingLayer.PoolingType pooling){
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("relu")
                .constrainGradientToUnitNorm(true)
                .seed(123)
                .nIn(1)
                .nOut(20)
                .layer(new SubsamplingLayer.Builder(pooling)
                        .build())
                .build();

        return LayerFactories.getFactory(new SubsamplingLayer()).create(conf);
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5,5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputWidth, inputHeight);
    }


    private Gradient createPrevGradient() {
        Gradient gradient = new DefaultGradient();
        INDArray pseudoGradients = Nd4j.ones(nExamples, nChannelsIn, inputHeight, inputWidth);

        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, pseudoGradients);
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoGradients);
        return gradient;
    }


}
