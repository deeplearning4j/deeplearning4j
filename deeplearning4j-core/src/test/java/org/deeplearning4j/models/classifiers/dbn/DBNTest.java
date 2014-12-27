package org.deeplearning4j.models.classifiers.dbn;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

/**
 * Created by agibsonccc on 8/28/14.
 */
public class DBNTest {

    private static Logger log = LoggerFactory.getLogger(DBNTest.class);



    @Test
    public void testIris() {
        RandomGenerator gen = new MersenneTwister(123);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(10)
                .weightInit(WeightInit.SIZE).optimizationAlgo(OptimizationAlgorithm.HESSIAN_FREE)
                .activationFunction(Activations.tanh()).visibleUnit(RBM.VisibleUnit.GAUSSIAN).hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .rng(gen).constrainGradientToUnitNorm(true)
                .learningRate(1e-1f)
                .nIn(4).nOut(3).list(3).hiddenLayerSizes(new int[]{3,2})
                .override(new NeuralNetConfiguration.ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {

                        if (i == 2) {
                            builder.weightInit(WeightInit.ZERO);
                            builder.activationFunction(Activations.softMaxRows());
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                        }
                    }
                }).build();




        DBN d = new DBN.Builder().layerWiseConfiguration(conf)
                .build();



        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        DataSet next = iter.next();
        next.normalizeZeroMeanZeroUnitVariance();
        next.shuffle();
        SplitTestAndTrain split = next.splitTestAndTrain(140);
        DataSet train = split.getTrain();
        d.fit(train);

        DataSet test = split.getTest();

        Evaluation eval = new Evaluation();
        INDArray output = d.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(),output);
        log.info("Score " + eval.stats());


    }

    @Test
    public void testDbn() throws IOException {
        RandomGenerator gen = new MersenneTwister(123);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .withActivationType(NeuralNetConfiguration.ActivationType.HIDDEN_LAYER_ACTIVATION)
                .weightInit(WeightInit.NORMALIZED).momentum(0.5)
                .optimizationAlgo(OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen).iterations(100)
                .learningRate(1e-1f).nIn(784).nOut(10).list(4).hiddenLayerSizes(new int[]{500, 400, 300}).override(new NeuralNetConfiguration.ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {
                        if(i == 3) {
                            builder.weightInit(WeightInit.ZERO);
                            builder.activationFunction(Activations.softMaxRows());
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                        }
                    }
                })
                .build();



        DBN d = new DBN.Builder().layerWiseConfiguration(conf)
                .build();

        MnistDataFetcher fetcher = new MnistDataFetcher(false);
        fetcher.fetch(100);
        DataSet d2 = fetcher.next();
        d.fit(d2);


        INDArray predict2 = d.output(d2.getFeatureMatrix());

        Evaluation eval = new Evaluation();
        eval.eval(d2.getLabels(),predict2);
        log.info(eval.stats());
        int[] predict = d.predict(d2.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));


    }

}
