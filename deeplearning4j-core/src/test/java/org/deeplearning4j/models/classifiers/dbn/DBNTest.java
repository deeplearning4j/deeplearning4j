package org.deeplearning4j.models.classifiers.dbn;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
    public void testDbn() throws IOException {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
               .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .learningRate(1e-1f).nIn(784).nOut(2).build();



        DBN d = new DBN.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{500,250})
                .build();
        d.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
        d.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.MCXENT);
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(100);
        DataSet d2 = fetcher.next();
        d2.filterAndStrip(new int[]{0, 1});
        d.fit(d2);
        int[] predict = d.predict(d2.getFeatureMatrix());
        log.info("Predict " + Arrays.toString(predict));

    }

}
