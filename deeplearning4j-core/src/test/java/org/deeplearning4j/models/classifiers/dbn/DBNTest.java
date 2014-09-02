package org.deeplearning4j.models.classifiers.dbn;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 8/28/14.
 */
public class DBNTest {

    private static Logger log = LoggerFactory.getLogger(DBNTest.class);



    @Test
    public void testIris() {
        RandomGenerator gen = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED).regularization(true).useHiddenActivationsForwardProp(false)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN).momentum(5e-1f).l2(1e-2f)
                .activationFunction(Activations.tanh()).iterations(100)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-2f).nIn(4).nOut(3).build();


        DBN d = new DBN.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{3,2})
                .build();

        d.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
        d.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

        DataSetIterator iter = new IrisDataSetIterator(150, 150);

        DataSet next = iter.next(150);
        next.normalizeZeroMeanZeroUnitVariance();
        d.fit(next);

        Evaluation eval = new Evaluation();
        INDArray output = d.output(next.getFeatureMatrix());
        eval.eval(next.getLabels(),output);
        log.info("Score " +eval.stats());


    }

    @Test
    public void testDbn() throws IOException {
        RandomGenerator gen = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(5e-1f)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT).rng(gen)
                .learningRate(1e-1f).nIn(784).nOut(2).build();


        DBN d = new DBN.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{500,250,100})
                .build();

        d.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
        d.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(10);
        DataSet d2 = fetcher.next();
        d2.filterAndStrip(new int[]{0,1});

        d.fit(d2);
        int[] predict = d.predict(d2.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));

    }

}
