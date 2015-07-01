package org.deeplearning4j.optimize.solver;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.solvers.BackTrackLineSearch;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class BackTrackLineSearchTest {
    @Test
    public void testLineSearch() throws Exception {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT).optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .activationFunction("softmax")
                .iterations(10).weightInit(WeightInit.XAVIER)
                .learningRate(1e-1).nIn(4).nOut(3).layer(new org.deeplearning4j.nn.conf.layers.OutputLayer()).build();

        OutputLayer l = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)),0);
        DataSet d = new IrisDataSetIterator(1,1).next();
        d.normalizeZeroMeanZeroUnitVariance();
        l.setInput(d.getFeatureMatrix());
        l.setLabels(d.getLabels());
        BackTrackLineSearch lineSearch = new BackTrackLineSearch(l,l.getOptimizer());
        double step = lineSearch.optimize(1.0,l.params(),l.gradient().gradient());
        assertEquals(0.0,step,1e-1);
    }

}
