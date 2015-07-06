package org.deeplearning4j.plot;

import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.plot.iterationlistener.*;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;


public class ListenerTest {

    /** Very simple back-prop config set up for Iris.
     * Learning Rate = 0.1
     * No regularization, no Adagrad, no momentum etc. One iteration.
     */

    private DataSetIterator irisIter = new IrisDataSetIterator(50,50);


    @Test
    public void testNeuralNetGraphsCapturedMLPNetwork() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMLPSimpleConfig(new int[]{5}, "sigmoid"));
        network.init();
        DataSet data = irisIter.next();
        IterationListener listener = new NeuralNetPlotterIterationListener(1,true);

        network.setListeners(Collections.singletonList(listener));
        network.fit(data.getFeatureMatrix(), data.getLabels());
        assertNotNull(network.getListeners().get(0));
        assertEquals(listener.invoked(), true);
    }


    @Test
    public void testAccuracyGraphCaptured() {
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisSimpleConfig(new int[]{10}, "sigmoid", 10));
        network.init();
        DataSet data = irisIter.next();
        IterationListener listener = new AccuracyPlotterIterationListener(1, network, data);

        network.setListeners(Collections.singletonList(listener));
        network.fit(data.getFeatureMatrix(), data.getLabels());
        assertNotNull(network.getListeners().get(0).toString());
        assertEquals(listener.invoked(), true);
    }

    @Test
    public void testMultipleGraphsCapturedForMultipleLayers() {
        // Tests Gradient Plotter and Loss Plotter
        MultiLayerNetwork network = new MultiLayerNetwork(getIrisSimpleConfig(new int[]{10, 5}, "sigmoid", 5));
        network.init();
        IterationListener listener = new GradientPlotterIterationListener(4);
        IterationListener listener2 = new LossPlotterIterationListener(4);
        network.setListeners(Arrays.asList(listener, listener2));
        while( irisIter.hasNext() ) network.fit(irisIter.next());
        assertNotNull(network.getListeners().get(1).toString());
        assertEquals(listener2.invoked(), true);
    }

    private static MultiLayerConfiguration getIrisSimpleConfig( int[] hiddenLayerSizes, String activationFunction, int iterations ) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
                .nIn(4).nOut(3)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 0.1))

                .activationFunction(activationFunction)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)

                .iterations(iterations)
                .batchSize(1)
                .constrainGradientToUnitNorm(false)
                .corruptionLevel(0.0)

                .layer(new RBM())
                .learningRate(0.1).useAdaGrad(false)

                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .dropOut(0.0)
                .momentum(0.0)
                .applySparsity(false).sparsity(0.0)
                .seed(12345L)

                .list(hiddenLayerSizes.length + 1).hiddenLayerSizes(hiddenLayerSizes)
                .useDropConnect(false)

                .override(hiddenLayerSizes.length, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.weightInit(WeightInit.DISTRIBUTION);
                        builder.dist(new NormalDistribution(0, 0.1));
                    }
                }).build();


        return c;
    }

    private static MultiLayerConfiguration getIrisMLPSimpleConfig( int[] hiddenLayerSizes, String activationFunction ) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
                .nIn(4).nOut(3)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 0.1))

                .activationFunction(activationFunction)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)

                .iterations(1)
                .batchSize(1)
                .constrainGradientToUnitNorm(false)
                .corruptionLevel(0.0)

                .layer(new RBM())
                .learningRate(0.1).useAdaGrad(false)

                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .dropOut(0.0)
                .momentum(0.0)
                .applySparsity(false).sparsity(0.0)
                .seed(12345L)

                .list(hiddenLayerSizes.length + 1)
                .hiddenLayerSizes(hiddenLayerSizes)
                .backward(true).pretrain(false)
                .useDropConnect(false)

                .override(hiddenLayerSizes.length, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.weightInit(WeightInit.DISTRIBUTION);
                        builder.dist(new NormalDistribution(0, 0.1));
                    }
                }).build();


        return c;
    }


    @Test
    public void testBackTrackLine() {
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        DataSetIterator irisIter = new IrisDataSetIterator(1,1);
        DataSet data = irisIter.next();

        MultiLayerNetwork network = new MultiLayerNetwork(getIris1ItConfig(new int[]{5}, "sigmoid", 1));
        network.init();

        network.fit(data.getFeatureMatrix(), data.getLabels());
    }


    private static MultiLayerConfiguration getIris1ItConfig( int[] hiddenLayerSizes, String activationFunction, int iterations ) {
        MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
                .nIn(4).nOut(3)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0, 0.1))

                .activationFunction(activationFunction)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)

                .iterations(iterations)
                .batchSize(1)
                .constrainGradientToUnitNorm(false)
                .corruptionLevel(0.0)

                .layer(new RBM())
                .learningRate(0.1).useAdaGrad(false)
                .numLineSearchIterations(1)
                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .dropOut(0.0)
                .momentum(0.0)
                .applySparsity(false).sparsity(0.0)
                .seed(12345L)

                .list(hiddenLayerSizes.length + 1).hiddenLayerSizes(hiddenLayerSizes)
                .useDropConnect(false)

                .override(hiddenLayerSizes.length, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
                        builder.activationFunction("softmax");
                        builder.layer(new OutputLayer());
                        builder.weightInit(WeightInit.DISTRIBUTION);
                        builder.dist(new NormalDistribution(0, 0.1));
                    }
                }).build();


        return c;
    }


}
