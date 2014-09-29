package org.deeplearning4j.models.classifiers.sda;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.WeightInit;
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
public class StackedDenoisingAutoEncoderTest {


    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoderTest.class);



    @Test
    public void testDbn() throws IOException {
        RandomGenerator gen = new MersenneTwister(123);
        MnistDataFetcher fetcher = new MnistDataFetcher(true);

        fetcher.fetch(100);
        DataSet d2 = fetcher.next();


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(5e-1f).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.uniform(gen,784,10))
                .withActivationType(NeuralNetConfiguration.ActivationType.SAMPLE).iterations(1)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
                .learningRate(1e-1f).nIn(d2.numInputs()).nOut(d2.numOutcomes()).build();


        StackedDenoisingAutoEncoder d = new StackedDenoisingAutoEncoder.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{500, 250, 200})
                .build();

        d.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
        d.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);

        d.fit(d2);


        INDArray predict2 = d.output(d2.getFeatureMatrix());

        Evaluation eval = new Evaluation();
        eval.eval(d2.getLabels(),predict2);
        log.info(eval.stats());
        int[] predict = d.predict(d2.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));


    }

}
