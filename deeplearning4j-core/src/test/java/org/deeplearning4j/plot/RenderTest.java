package org.deeplearning4j.plot;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.iterationlistener.NeuralNetPlotterIterationListener;
import org.deeplearning4j.plot.iterationlistener.PlotFiltersIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
public class RenderTest {
    @Test
    public void testRender() {
        INDArray test = Nd4j.rand(new int[]{328,400,4});
        PlotFilters render = new PlotFilters();
        INDArray rendered = render.render(test,1,1);
        assertArrayEquals(new int[]{7619, 95}, rendered.shape());
    }


    @Test
    public void testPlotter() throws Exception {
        PlotFiltersIterationListener listener = new PlotFiltersIterationListener(Arrays.asList(DefaultParamInitializer.WEIGHT_KEY));
        listener.setOutputFile(new File("/home/agibsonccc/Desktop/render.png"));
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .corruptionLevel(0.3).weightInit(WeightInit.DISTRIBUTION).dropOut(0.5)
                .iterations(10).constrainGradientToUnitNorm(true).dist(new NormalDistribution(1e-3,1e-1))
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(784).nOut(600)
                .layer(new org.deeplearning4j.nn.conf.layers.RBM())
                .build();


        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();
        Collection<IterationListener> listeners = Arrays.asList(new ScoreIterationListener(1), new NeuralNetPlotterIterationListener(1));
        Layer da = LayerFactories.getFactory(conf.getLayer()).create(conf, listeners,0);
        da.fit(input);


    }

}
