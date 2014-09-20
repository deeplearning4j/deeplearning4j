package org.deeplearning4j.models.featuredetectors.autoencoder;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;

import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/19/14.
 */
public class SemanticHashingTest {

    private static Logger log = LoggerFactory.getLogger(SemanticHashingTest.class);

    @Test
    public void testSemanticHashingMnist() throws Exception {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(784).nOut(10).momentum(0.5f)
                .build();
        DBN dbn = new DBN.Builder().configure(conf)
                .hiddenLayerSizes(new int[]{500,250,100})
                .build();


        dbn.getOutputLayer().conf().setActivationFunction(Activations.sigmoid());
        dbn.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.RMSE_XENT);
        MnistDataFetcher fetch = new MnistDataFetcher(true);
        fetch.fetch(20);

        DataSet next = fetch.next();

        dbn.fit(next);


        SemanticHashing hashing = new SemanticHashing.Builder()
                .withEncoder(dbn).configure(conf).build();
        next.setLabels(next.getFeatureMatrix());
        hashing.fit(next);


    }


}
