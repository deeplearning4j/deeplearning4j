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

    @Test
    public void testSubSampleLayerActivateShape() throws Exception  {
        DataSetIterator mnistIter = new MnistDataSetIterator(1,1);
        DataSet mnist = mnistIter.next();

        Layer model = getSubsamplingLayer();
        INDArray input = mnist.getFeatureMatrix().reshape(mnist.numExamples(), 1, 28, 28);

        INDArray output = model.activate(input);
        assertTrue(Arrays.equals(new int[]{mnist.numExamples(), 1, 14, 14}, output.shape()));
        assertEquals(mnist.numExamples(), output.shape()[0], 1e-4); // depth retained
    }

    @Test
    public void testSubSampleLayerBackpropShape() throws Exception  {

        Layer model = getSubsamplingLayer();
        Gradient gradient = createPrevGradient(1, 3, 20);
        // TODO generate and pass epsilon
        Pair<Gradient, INDArray> out= model.backpropGradient(null, gradient, null);

        // TODO check epsilon full shape will match expectations
        assertEquals(out.getSecond().shape()[0], 3); // depth retained
    }


    private static Layer getSubsamplingLayer(){
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .activationFunction("relu")
                .constrainGradientToUnitNorm(true)
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .seed(123)
                .nIn(1)
                .nOut(20)
                .layer(new SubsamplingLayer.Builder()
                        .build())
                .build();

        return LayerFactories.getFactory(new SubsamplingLayer()).create(conf);

    }

    private static Gradient createPrevGradient(int miniBatchSize, int nOut, int nHiddenUnits) {
        Gradient gradient = new DefaultGradient();
        INDArray pseudoBiasGradients = Nd4j.ones(miniBatchSize, nOut);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, pseudoBiasGradients);
        INDArray pseudoWeightGradients = Nd4j.ones(miniBatchSize, nHiddenUnits,nOut);
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, pseudoWeightGradients);
        return gradient;
        }


}
