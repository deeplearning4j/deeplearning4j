package org.deeplearning4j.plot;


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.*;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import org.nd4j.linalg.lossfunctions.LossFunctions;
import static org.junit.Assert.*;


public class ListenerTest {

    /** Very simple back-prop config set up for Iris.
     * Learning Rate = 0.1
     * No regularization, no Adagrad, no momentum etc. One iteration.
     */

    private DataSetIterator irisIter = new IrisDataSetIterator(50,50);

    // TODO fix activation and rendor for MLP...
    @Test
    @Ignore
    public void testNeuralNetGraphsCapturedMLPNetwork() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig("sigmoid", 1));
        network.init();
        DataSet data = irisIter.next();
        IterationListener listener = new NeuralNetPlotterIterationListener(1,true);

        network.setListeners(listener);
        network.fit(data.getFeatureMatrix(), data.getLabels());
        assertNotNull(network.getListeners());
        assertEquals(listener.invoked(), true);
    }

    @Test
    public void testScoreIterationListenerMLP() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig("sigmoid", 5));
        network.init();
        IterationListener listener = new ScoreIterationListener(1);
        network.setListeners(listener);
        while( irisIter.hasNext() ) network.fit(irisIter.next());
        assertEquals(listener.invoked(), true);
    }

    @Test
    public void testScoreIterationListenerBackTrack() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisSimpleConfig("sigmoid", 5));
        network.init();
        IterationListener listener = new ScoreIterationListener(1);
        network.setListeners(listener);
        while( irisIter.hasNext() ) network.fit(irisIter.next());
        assertEquals(listener.invoked(), true);
    }

    @Test
    @Ignore
    public void testNeuralNetGraphsCaptured() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisSimpleConfig("sigmoid", 1));
        network.init();
        DataSet data = irisIter.next();
        IterationListener listener = new NeuralNetPlotterIterationListener(1,true);

        network.setListeners(listener);
        network.fit(data.getFeatureMatrix(), data.getLabels());
        assertNotNull(network.getListeners());
        assertEquals(listener.invoked(), true);
    }

    // TODO fix so it tracks epochs...
    @Test
    @Ignore
    public void testAccuracyGraphCaptured() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisSimpleConfig("sigmoid", 10));
        network.init();
        DataSet data = irisIter.next();
        IterationListener listener = new AccuracyPlotterIterationListener(1, network, data);

        network.setListeners(listener);
        network.fit(data.getFeatureMatrix(), data.getLabels());
        assertNotNull(network.getListeners());
        assertEquals(listener.invoked(), true);
    }

    @Test
    @Ignore
    public void testMultipleGraphsCapturedForMultipleLayers() {
        // Tests Gradient Plotter and Loss Plotter
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisSimpleConfig("sigmoid", 5));
        network.init();
        IterationListener listener = new GradientPlotterIterationListener(2);
        IterationListener listener2 = new LossPlotterIterationListener(2);
        network.setListeners(listener, listener2);
        while( irisIter.hasNext() ) network.fit(irisIter.next());
        assertNotNull(network.getListeners().remove(1));
        assertEquals(listener2.invoked(), true);
    }

    private static MultiLayerConfiguration getIrisSimpleConfig(String activationFunction, int iterations ) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(iterations)
                .learningRate(0.1)
                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .momentum(0.0)
                .seed(12345L)
                .list(3)
                .layer(0, new RBM.Builder()
                        .nIn(4).nOut(10)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .activation(activationFunction)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new RBM.Builder()
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .nIn(10).nOut(5)
                        .activation(activationFunction)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(2, new OutputLayer.Builder()
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .nIn(5).nOut(3)
                        .build())
                .build();
        return c;
    }

    private static MultiLayerConfiguration getIrisMLPSimpleConfig(String activationFunction, int iterations ) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(iterations)
                .constrainGradientToUnitNorm(false)
                .learningRate(0.1)
                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .momentum(0.0)
                .seed(12345L)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .activation(activationFunction)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(5).nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .build())
                .backprop(true).pretrain(false)
                .build();


        return c;
    }



}
