package org.deeplearning4j.optimize.solver;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class BackTrackLineSearchTest {

    @Test
    public void testLineSearch() throws Exception {
        DataSet data = new IrisDataSetIterator(1,1).next();
        data.normalizeZeroMeanZeroUnitVariance();

        OutputLayer layer = getIrisLogisticLayerConfig("softmax", 10);
        layer.setInput(data.getFeatureMatrix());
        layer.setLabels(data.getLabels());
        BackTrackLineSearch lineSearch = new BackTrackLineSearch(layer, layer.getOptimizer());

        double step = lineSearch.optimize(1.0, layer.params(), layer.gradient().gradient());
        assertEquals(0.0,step,1e-1);
    }

    @Test
    public void testBackTrackLine() {
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;

        DataSetIterator irisIter = new IrisDataSetIterator(1,1);
        DataSet data = irisIter.next();

        MultiLayerNetwork network = new MultiLayerNetwork(getIrisMultiLayerConfig(new int[]{5}, "sigmoid", 1));
        network.init();

        network.fit(data.getFeatureMatrix(), data.getLabels());
    }

    private static OutputLayer getIrisLogisticLayerConfig(String activationFunction, int iterations){
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer())
                .nIn(4)
                .nOut(3)
                .activationFunction(activationFunction)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .learningRate(1e-1)
                .seed(12345L)
                .build();

        return LayerFactories.getFactory(conf.getLayer()).create(conf);


    }


    private static MultiLayerConfiguration getIrisMultiLayerConfig( int[] hiddenLayerSizes, String activationFunction, int iterations ) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(4).nOut(3)
                .weightInit(WeightInit.XAVIER)
                .dist(new NormalDistribution(0, 0.1))

                .activationFunction(activationFunction)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .activationFunction("softmax")

                .iterations(iterations)
                .batchSize(1)
                .constrainGradientToUnitNorm(false)
                .corruptionLevel(0.0)

                .layer(new RBM())
                .learningRate(0.1)
                .useAdaGrad(false)
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
                        builder.layer(new org.deeplearning4j.nn.conf.layers.OutputLayer());
                        builder.weightInit(WeightInit.DISTRIBUTION);
                        builder.dist(new NormalDistribution(0, 0.1));
                    }
                }).build();


        return conf;
    }

}
