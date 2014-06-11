package org.deeplearning4j.autoencoder;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.dbn.DBN;

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
        fetcher.fetch(20);
        DataSet data = fetcher.next();
        data.filterAndStrip(new int[]{0, 1});
        log.info("Training on " + data.numExamples());

        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{1000, 500, 250, 10})
                .numberOfInputs(784)
                .numberOfOutPuts(2)
                .build();

        dbn.pretrain(data.getFirst(),new Object[]{1,1e-1,1000});


        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);
        assertEquals(encoder.getLayers()[0].getW().length,encoder.getOutputLayer().getW().length);
        assertEquals(encoder.getLayers()[1].getW().length,encoder.getLayers()[encoder.getLayers().length - 1].getW().length);
        assertEquals(encoder.getLayers()[2].getW().length,encoder.getLayers()[encoder.getLayers().length - 2].getW().length);


        encoder.finetune(data.getFirst(),1e-1,1000);




    }

}
