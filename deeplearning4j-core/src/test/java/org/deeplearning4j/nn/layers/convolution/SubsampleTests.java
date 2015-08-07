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
public class SubsampleTests {

    private int nExamples = 1;
    private int nChannels = 20; //depth & nOut
    private int inH = 28;
    private int inW = 28;
    private int outH = 10;
    private int outW = 10;
    private INDArray epsilon = Nd4j.ones(nExamples, nChannels, outH, outW);


    @Test
    public void testSubSampleMaxActivateShape() throws Exception  {
        DataSetIterator mnistIter = new MnistDataSetIterator(nExamples,nExamples);
        DataSet mnist = mnistIter.next();

        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.MAX);
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, inH, inW);

        INDArray output = model.activate(input);
        assertTrue(Arrays.equals(new int[]{nExamples, 1, 14, 14}, output.shape()));
        assertEquals(nExamples, output.shape()[0], 1e-4); // depth retained
        //TODO test max results...
    }

    @Test
    public void testSubSampleAvgActivateShape() throws Exception  {
        DataSetIterator mnistIter = new MnistDataSetIterator(nExamples,nExamples);
        DataSet mnist = mnistIter.next();

        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.AVG);
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, 28, 28);

        INDArray output = model.activate(input);
        assertTrue(Arrays.equals(new int[]{nExamples, 1, 14, 14}, output.shape()));
        assertEquals(nExamples, output.shape()[0], 1e-4); // depth retained
    }

    @Test
    public void testSubSampleNoneActivateShape() throws Exception  {
        DataSetIterator mnistIter = new MnistDataSetIterator(nExamples, nExamples);
        DataSet mnist = mnistIter.next();

        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.NONE);
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, inH, inW);

        INDArray output = model.activate(input);
        assertTrue(Arrays.equals(new int[]{nExamples, 1, inH, inW}, output.shape()));
        assertEquals(nExamples, output.shape()[0], 1e-4); // depth retained
    }

    @Test (expected=IllegalStateException.class)
    public void testSubSampleSumActivateShape() throws Exception  {
        DataSetIterator mnistIter = new MnistDataSetIterator(nExamples, nExamples);
        DataSet mnist = mnistIter.next();

        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.SUM);
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, 28, 28);

        INDArray output = model.activate(input);
    }

    @Test
    public void testSubSampleLayerMaxBackpropShape() throws Exception {
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.MAX);
        Gradient gradient = createPrevGradient();
        DataSetIterator mnistIter = new MnistDataSetIterator(nExamples,nExamples);
        DataSet mnist = mnistIter.next();
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, 28, 28);

        INDArray activations = model.activate(input);

        Pair<Gradient, INDArray> out= model.backpropGradient(epsilon, gradient, null);
        assertEquals(nChannels, out.getSecond().shape()[1]); // depth retained
    }

    @Test
    public void testSubSampleLayerAvgBackpropShape() throws Exception{
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.AVG);
        Gradient gradient = createPrevGradient();
        DataSetIterator mnistIter = new MnistDataSetIterator(nExamples,nExamples);
        DataSet mnist = mnistIter.next();
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, 28, 28);

        INDArray activations = model.activate(input);

        Pair<Gradient, INDArray> out= model.backpropGradient(epsilon, gradient, null);
        assertEquals(nChannels, out.getSecond().shape()[1]); // depth retained
    }

    @Test
    public void testSubSampleLayerNoneBackpropShape() {
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.NONE);
        Gradient gradient = createPrevGradient();

        Pair<Gradient, INDArray> out= model.backpropGradient(epsilon, gradient, null);
        assertEquals(nChannels, out.getSecond().shape()[1]); // depth retained
    }


    @Test (expected=IllegalStateException.class)
    public void testSubSampleLayerSumBackpropShape() {
        Layer model = getSubsamplingLayer(SubsamplingLayer.PoolingType.SUM);
        Gradient gradient = createPrevGradient();

        Pair<Gradient, INDArray> out= model.backpropGradient(epsilon, gradient, null);
    }

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

    private Gradient createPrevGradient() {
        Gradient gradient = new DefaultGradient();
        INDArray pseudoGradients = Nd4j.ones(nExamples, nChannels, outH, outW);

        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, pseudoGradients);
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoGradients);
        return gradient;
        }


}
