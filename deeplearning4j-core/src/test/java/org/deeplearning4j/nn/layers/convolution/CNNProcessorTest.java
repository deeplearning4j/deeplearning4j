package org.deeplearning4j.nn.layers.convolution;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.ReshapeProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by merlin on 7/31/15.
 */
public class CNNProcessorTest {
    private static int rows = 28;
    private static int cols = 28;

    @Test
    public void testCNNInputPreProcessorMnist() throws Exception {
        int numSamples = 1;
        int batchSize = 1;

        DataSet mnistIter = new MnistDataSetIterator(batchSize, numSamples, true).next();
        MultiLayerNetwork model = getCNNMnistConfig();
        model.init();
        model.fit(mnistIter);

        int val2to4 = model.getLayer(0).input().shape().length;
        assertTrue(val2to4 == 4);

    }


    public static MultiLayerNetwork getCNNMnistConfig()  {

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(5)
                .weightInit(WeightInit.XAVIER)
                .activationFunction("relu")
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{9, 9}, Convolution.Type.VALID)
                        .nIn(1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(20)
                        .nOut(10)
                        .activation("softmax")
                        .build())
                .inputPreProcessor(0, new ReshapeProcessor(new int[]{1, rows*cols}, new int[]{1, 1, rows, cols}))
        .build();
        return new MultiLayerNetwork(conf);

    }
}
