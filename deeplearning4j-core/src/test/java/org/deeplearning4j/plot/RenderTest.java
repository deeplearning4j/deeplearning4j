package org.deeplearning4j.plot;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.PlotFiltersIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
public class RenderTest {
    @Test
    public void testRender() {
        INDArray test = Nd4j.rand(new int[]{784,600}).transpose();
        PlotFilters render = new PlotFilters(test,new int[]{10,10},new int[]{0,0},new int[]{28,28});
        render.plot();
        INDArray rendered = render.getPlot();


    }


    @Test
    public void testPlotter() throws Exception {
        INDArray test = Nd4j.rand(new int[]{784,600}).transpose();
        PlotFilters render = new PlotFilters(test,new int[]{10,10},new int[]{0,0},new int[]{28,28});
        PlotFiltersIterationListener listener = new PlotFiltersIterationListener(render,Arrays.asList(DefaultParamInitializer.WEIGHT_KEY),0);
        listener.setOutputFile(new File("render.png"));
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .iterations(10)
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                        .nIn(784).nOut(600)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(1e-3, 1e-1))
                        .dropOut(0.5)
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .build();


        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        Collection<IterationListener> listeners = Arrays.asList(new ScoreIterationListener(1),listener);
        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer da = LayerFactories.getFactory(conf.getLayer()).create(conf, listeners,0, params);
        da.fit(input);


    }

}
