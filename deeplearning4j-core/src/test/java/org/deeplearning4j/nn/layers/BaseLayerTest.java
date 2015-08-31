package org.deeplearning4j.nn.layers;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 */

public class BaseLayerTest {

    private DataSetIterator irisIter = new IrisDataSetIterator(50,50);

    @Test
    public void testRBMParamsAndScores() {
        DataSet data = irisIter.next();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM.Builder()
                        .nIn(4)
                        .nOut(3)
                        .activation("tanh").build())
                .iterations(1)
                .seed(123)
                .build();

        Layer layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
        layer.fit(data.getFeatureMatrix());

        double score = layer.score();
        INDArray parameters = layer.params();
        layer.setParams(parameters);
        layer.computeGradientAndScore();
        double score2 = layer.score();
        assertEquals(parameters, layer.params());
        assertEquals(score, score2, 1e-3);
    }

    // TODO fix - seed in AutoEncoder so the score is consistent
    @Test
    public void testRecursiveAutoEncoderScores() {
        DataSet data = irisIter.next();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RecursiveAutoEncoder.Builder()
                        .nIn(4)
                        .nOut(3)
                        .activation("sigmoid").build())
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .iterations(1)
                .seed(123)
                .build();

        Layer layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
        layer.fit(data.getFeatureMatrix());

        double score = layer.score();
        double score2 = 101.61147665977478;
        assertEquals(score, score2, 1e-3);
    }


    // TODO fix - input isn't set for LSTM and fails on gradient update
    @Test
    public void testLSTMParamsAndScores() {
        DataSet data = irisIter.next();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new ImageLSTM.Builder()
                        .nIn(4)
                        .nOut(3)
                        .activation("sigmoid")
                        .build())
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .iterations(1)
                .seed(123)
                .build();

        Layer layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
        layer.fit(data.getFeatureMatrix());

        double score = layer.score();
        INDArray parameters = layer.params();
        layer.setParams(parameters);
        layer.computeGradientAndScore();
        double score2 = layer.score();
        assertEquals(parameters, layer.params());
        assertEquals(score, score2, 1e-3);
    }

    @Test
    public void testOutputParamsAndScores() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(1)
                .seed(123)
                .list(2)
                .layer(0, new RBM.Builder()
                        .nIn(4)
                        .nOut(10)
                        .activation("sigmoid")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(10)
                        .nOut(3)
                        .activation("sigmoid")
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.fit(irisIter);

        double score = network.getLayer(1).score();
        INDArray parameters = network.getLayer(1).params();
        network.getLayer(1).setParams(parameters);
        network.getLayer(1).computeGradientAndScore();

        double score2 = network.getLayer(1).score();
        assertEquals(parameters, network.getLayer(1).params());
        assertEquals(score, score2, 1e-3);
    }



}
