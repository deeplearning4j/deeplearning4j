package org.deeplearning4j.models.featuredetectors.rbm;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.DefaultLayerFactory;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.plot.NeuralNetworkReconstructionRender;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 8/27/14.
 */
public class RBMTests {
    private static Logger log = LoggerFactory.getLogger(RBMTests.class);

    @Test
    public void testLfw() {
        LFWDataSetIterator iter = new LFWDataSetIterator(10,10,28,28);
        DataSet d = iter.next();

        d.normalizeZeroMeanZeroUnitVariance();

        int nOut = 600;
        RandomGenerator rng = new MersenneTwister(123);
        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED).weightInit(WeightInit.VI).constrainGradientToUnitNorm(true)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).layerFactory(layerFactory).optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.XENT).rng(rng)
                .iterationListener(new IterationListener() {
                    @Override
                    public void iterationDone(Model model, int iteration) {
                        if(iteration > 0 && iteration % 100 == 0) {
                            NeuralNetPlotter plotter = new NeuralNetPlotter();
                            Layer l = (Layer) model;
                            plotter.plotNetworkGradient(l, l.getGradient(), 10);

                        }



                    }
                })
                .learningRate(1e-1f)
                .nIn(d.numInputs()).nOut(nOut).build();

        RBM rbm = layerFactory.create(conf);

        rbm.fit(d.getFeatureMatrix());




    }

    @Test
    public void testIris() {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(150);
        DataSet d = fetcher.next();
        d.normalizeZeroMeanZeroUnitVariance();
        RandomGenerator g = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED).learningRate(1e-1f)
                .nIn(d.numInputs()).rng(g).
                        nOut(3).build();


        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);
        RBM r = layerFactory.create(conf);
        r.fit(d.getFeatureMatrix());

    }



    @Test
    public void testBasic() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);
        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        RBM rbm = layerFactory.create(conf);
        rbm.fit(input);



    }

    @Test
    public void testMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(100)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT).iterationListener(new IterationListener() {
                    @Override
                    public void iterationDone(Model model, int i) {
                        if (i > 0 && i % 1000 == 0) {
                            NeuralNetPlotter plotter = new NeuralNetPlotter();
                            Layer l = (Layer) model;
                            plotter.plotNetworkGradient(l, l.getGradient(), 10);
                        }
                    }
                })
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(new MersenneTwister(123))
                .learningRate(1e-1f).nIn(784).nOut(600).build();

        fetcher.fetch(10);
        DataSet d2 = fetcher.next();

        INDArray input = d2.getFeatureMatrix();

        LayerFactory layerFactory = LayerFactories.getFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);

        rbm.fit(input);






    }


    @Test
    public void testSetGetParams() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        LayerFactory layerFactory = new DefaultLayerFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);
        INDArray rand2 = Nd4j.rand(new int[]{1, rbm.numParams()});
        rbm.setParams(rand2);
        INDArray getParams = rbm.params();
        assertEquals(rand2,getParams);
    }





    @Test
    public void testCg() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        LayerFactory layerFactory = new DefaultLayerFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);
        rbm.setInput(input);
        double value = rbm.score();
        rbm.contrastiveDivergence();
        value = rbm.score();



    }

    @Test
    public void testGradient() {
        float[][] data = new float[][]
                {
                        {1,1,1,0,0,0},
                        {1,0,1,0,0,0},
                        {1,1,1,0,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,0,0},
                        {0,0,1,1,1,0},
                        {0,0,1,1,1,0}
                };


        INDArray input = Nd4j.create(data);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(6).nOut(4).build();
        LayerFactory layerFactory = new DefaultLayerFactory(RBM.class);
        RBM rbm = layerFactory.create(conf);
        rbm.setInput(input);
        double value = rbm.score();


        Gradient grad2 = rbm.getGradient();
        rbm.fit(input);

    }



}
