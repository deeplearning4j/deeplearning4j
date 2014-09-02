package org.deeplearning4j.models.layers;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
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

/**
 * Created by agibsonccc on 9/1/14.
 */
public class OutputLayerTest {
    private static Logger log = LoggerFactory.getLogger(OutputLayerTest.class);


    @Test
    public void testIris() {
        RandomGenerator gen = new MersenneTwister(123);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activationFunction(Activations.softMaxRows()).iterations(10)
                .rng(gen)
                .learningRate(1e-1f).nIn(4).nOut(3).build();

        OutputLayer l = new OutputLayer.Builder()
                .configure(conf).build();
        DataSetIterator iter = new IrisDataSetIterator(150, 150);


        DataSet next = iter.next(150);
        next.normalizeZeroMeanZeroUnitVariance();
        l.fit(next);


        int[] predictions = l.predict(next.getFeatureMatrix());

        Evaluation eval = new Evaluation();
        INDArray output = l.output(next.getFeatureMatrix());
        eval.eval(next.getLabels(),output);
        log.info("Score " +eval.stats());


    }


}
