package org.deeplearning4j.nn.layers.feedforward.dense;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

/**
 * Created by nyghtowl on 8/31/15.
 */
public class DenseTest {

@Test
public void testMLPMultiLayerBackprop(){

    final int numInputs = 4;
    int outputNum = 3;
    int numSamples = 150;
    int batchSize = 150;
    int iterations = 100;
    long seed = 6;
    int listenerFreq = iterations/5;

    DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)

            .learningRate(1e-3)
            .l1(0.3).regularization(true).l2(1e-3)
            .constrainGradientToUnitNorm(true)
            .list(3)
            .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(numInputs).nOut(3)
                    .activation("tanh")
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(1, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nIn(3).nOut(2)
                    .activation("tanh")
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .weightInit(WeightInit.XAVIER)
                    .activation("softmax")
                    .nIn(2).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    }

}
