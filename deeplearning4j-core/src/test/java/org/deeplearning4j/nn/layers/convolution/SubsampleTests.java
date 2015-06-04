package org.deeplearning4j.nn.layers.convolution;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class SubsampleTests {

    @Test
    public void testSubSampleLayer() throws Exception  {
        DataSetIterator mnistIter = new MnistDataSetIterator(100,100);
        DataSet mnist = mnistIter.next();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("relu").constrainGradientToUnitNorm(true)
                .convolutionType(ConvolutionLayer.ConvolutionType.MAX).filterSize(5,1,28,28)
                .layer(new SubsamplingLayer())
                .nIn(784).nOut(1).build();

        Layer model = LayerFactories.getFactory(new SubsamplingLayer()).create(conf);

        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, 28, 28);
        input = input.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 20), NDArrayIndex.interval(0,20));

        INDArray output = model.activate(input);
        assertTrue(Arrays.equals(new int[]{mnist.numExamples(),1,10,21},output.shape()));
    }

}
