package org.deeplearning4j.autoencoder;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.activation.Activations;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 5/11/14.
 */
public class DeepAutoEncoderTest {
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderTest.class);

    @Test
    public void testWithMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(100);
        DataSet d = fetcher.next();
        d.filterAndStrip(new int[]{0,1});
        log.info("Training on " + d.numExamples());
        StopWatch watch = new StopWatch();


        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{1000,500,250,30})
                .withMomentum(0.5).renderWeights(100)
                .numberOfInputs(784)
                .numberOfOutPuts(2)
                .build();

        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn,new Object[]{1,1e-2,10000});
        encoder.train(d.getFirst(),1e-2,1);





    }

}
