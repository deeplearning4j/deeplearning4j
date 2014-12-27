package org.deeplearning4j.models.classifiers.sda;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
public class StackedDenoisingAutoEncoderTest {


    private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoderTest.class);



    @Test
    public void testDbn() throws IOException {
        final RandomGenerator gen = new MersenneTwister(123);
        DataSetIterator iter = new MnistDataSetIterator(10,10);

        DataSet d2 = iter.next();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(5e-1f).weightInit(WeightInit.SIZE).constrainGradientToUnitNorm(false)
                .withActivationType(NeuralNetConfiguration.ActivationType.SAMPLE).iterations(10)
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen).optimizationAlgo(OptimizationAlgorithm.HESSIAN_FREE)
                .learningRate(1e-1f).nIn(d2.numInputs()).nOut(d2.numOutcomes()).list(4).hiddenLayerSizes(new int[]{600, 300, 200}).override(new NeuralNetConfiguration.ConfOverride() {
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





        StackedDenoisingAutoEncoder d = new StackedDenoisingAutoEncoder.Builder()
                .layerWiseConfiguration(conf)
                .build();




        d.fit(d2);


        while(iter.hasNext()) {
            d2 = iter.next();
            d.fit(d2);
        }

        INDArray predict2 = d.output(d2.getFeatureMatrix());

        Evaluation eval = new Evaluation();
        eval.eval(d2.getLabels(),predict2);
        log.info(eval.stats());
        int[] predict = d.predict(d2.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));


    }

}
