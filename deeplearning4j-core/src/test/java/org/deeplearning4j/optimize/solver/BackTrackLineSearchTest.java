package org.deeplearning4j.optimize.solver;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.solvers.BackTrackLineSearch;
import org.deeplearning4j.optimize.stepfunctions.DefaultStepFunction;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class BackTrackLineSearchTest {

    @Before
    public void before(){
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
    }

    DataSetIterator irisIter = new IrisDataSetIterator(5,5);
    DataSet irisData = irisIter.next();
    double score1, score2;


    @Test
    public void testSingleMinLineSearch() throws Exception {
        OutputLayer layer = getIrisLogisticLayerConfig("softmax", 1, LossFunctions.LossFunction.RMSE_XENT);
        layer.setInput(irisData.getFeatureMatrix());
        layer.setLabels(irisData.getLabels());
        layer.computeGradientAndScore();

        BackTrackLineSearch lineSearch = new BackTrackLineSearch(layer, layer.getOptimizer());
        double step = lineSearch.optimize(layer.params(), layer.gradient().gradient(), layer.gradient().gradient());

        assertEquals(0.0, step, 1e-3);
    }

    @Test
    public void testSingleMaxLineSearch() throws Exception {
        OutputLayer layer = getIrisLogisticLayerConfig("softmax", 1, LossFunctions.LossFunction.RMSE_XENT);
        layer.setInput(irisData.getFeatureMatrix());
        layer.setLabels(irisData.getLabels());
        layer.computeGradientAndScore();
        score1 = layer.score();

        BackTrackLineSearch lineSearch = new BackTrackLineSearch(layer, new DefaultStepFunction(), layer.getOptimizer());
        double step = lineSearch.optimize(layer.params(), layer.gradient().gradient(), layer.gradient().gradient());

// TODO why is max returning a 1 while min isn't?
        assertEquals(1.0, step,1e-3);
    }


    @Test
    public void testMultMinLineSearch() throws Exception {
        OutputLayer layer = getIrisLogisticLayerConfig("softmax", 5, LossFunctions.LossFunction.RMSE_XENT);
        layer.setInput(irisData.getFeatureMatrix());
        layer.setLabels(irisData.getLabels());
        layer.computeGradientAndScore();
        score1 = layer.score();

        BackTrackLineSearch lineSearch = new BackTrackLineSearch(layer, layer.getOptimizer());
        lineSearch.optimize(layer.params(), layer.gradient().gradient(), layer.gradient().gradient());
        score2 = layer.score();

        assertTrue(score1 > score2);

    }

    @Test
    public void testMultMaxLineSearch() throws Exception {
        irisData.normalizeZeroMeanZeroUnitVariance();
        OutputLayer layer = getIrisLogisticLayerConfig("softmax", 5, LossFunctions.LossFunction.MCXENT);
        layer.setInput(irisData.getFeatureMatrix());
        layer.setLabels(irisData.getLabels());
        layer.computeGradientAndScore();
        score1 = layer.score();

        BackTrackLineSearch lineSearch = new BackTrackLineSearch(layer, new DefaultStepFunction(), layer.getOptimizer());
        lineSearch.optimize(layer.params(), layer.gradient().gradient(), layer.gradient().gradient());
        score2 = layer.score();

        assertTrue(score1 < score2);
    }

    private static OutputLayer getIrisLogisticLayerConfig(String activationFunction, int maxIterations, LossFunctions.LossFunction lossFunction){
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345L)
                .iterations(1)
                .miniBatch(false)
                .maxNumLineSearchIterations(maxIterations)
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(lossFunction)
                        .nIn(4)
                        .nOut(3)
                        .activation(activationFunction)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        return LayerFactories.getFactory(conf.getLayer()).create(conf);
    }

///////////////////////////////////////////////////////////////////////////

    @Test
    public void testBackTrackLineGradientDescent() {
        OptimizationAlgorithm optimizer = OptimizationAlgorithm.LINE_GRADIENT_DESCENT;

        DataSetIterator irisIter = new IrisDataSetIterator(1,1);
        DataSet data = irisIter.next();

        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMultiLayerConfig("sigmoid", 10, optimizer));
        network.init();
        IterationListener listener = new ScoreIterationListener(1);
        network.setListeners(Collections.singletonList(listener));

        network.fit(data.getFeatureMatrix(), data.getLabels());
        double score = network.getLayer(1).score();
        assertTrue(score < 1);
    }

    @Test
    public void testBackTrackLineCG() {
        OptimizationAlgorithm optimizer = OptimizationAlgorithm.CONJUGATE_GRADIENT;

        DataSet data = irisIter.next();

        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMultiLayerConfig("sigmoid", 5, optimizer));
        network.init();
        IterationListener listener = new ScoreIterationListener(1);
        network.setListeners(Collections.singletonList(listener));

        network.fit(data.getFeatureMatrix(), data.getLabels());
        double score = network.getLayer(1).score();
        assertTrue(score < 1);

    }

    @Test
    public void testBackTrackLineLBFGS() {
        OptimizationAlgorithm optimizer = OptimizationAlgorithm.LBFGS;
        DataSet data = irisIter.next();

        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMultiLayerConfig("sigmoid", 5, optimizer));
        network.init();
        IterationListener listener = new ScoreIterationListener(1);
        network.setListeners(Collections.singletonList(listener));

        network.fit(data.getFeatureMatrix(), data.getLabels());
        double score = network.getLayer(1).score();
        assertTrue(score < 1);

    }

    @Test(expected=UnsupportedOperationException.class)
    public void testBackTrackLineHessian() {
        OptimizationAlgorithm optimizer = OptimizationAlgorithm.HESSIAN_FREE;
        DataSet data = irisIter.next();

        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMultiLayerConfig("sigmoid", 1, optimizer));
        network.init();
        IterationListener listener = new ScoreIterationListener(1);
        network.setListeners(Collections.singletonList(listener));

        network.fit(data.getFeatureMatrix(), data.getLabels());
    }



    private static MultiLayerConfiguration getIrisMultiLayerConfig( String activationFunction, int iterations,  OptimizationAlgorithm optimizer) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(optimizer)
                .iterations(iterations)
                .batchSize(1)
                .constrainGradientToUnitNorm(false)
                .learningRate(0.1)
                .maxNumLineSearchIterations(2)
                .regularization(false)
                .l1(0.0)
                .l2(0.0)
                .momentum(0.0)
                .seed(12345L)
                .list(2)
                .layer(0, new RBM.Builder()
                        .nIn(4)
                        .nOut(5)
                        .weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 0.1))
                        .activation(activationFunction)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT)
                        .nIn(5)
                        .nOut(3)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 0.1))
                        .activation("softmax")
                        .build())
                .build();


        return conf;
    }

}
