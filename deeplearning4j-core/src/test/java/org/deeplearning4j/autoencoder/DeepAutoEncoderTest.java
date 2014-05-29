package org.deeplearning4j.autoencoder;

import static org.junit.Assume.*;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.dbn.DBN;

import org.jblas.DoubleMatrix;
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
        encoder.finetune(data.getFirst(),1e-3,1000);

        assumeNotNull(encoder.getEncoder());
        assumeNotNull(encoder.getDecoder());
        assumeNotNull(encoder.getDecoder(),encoder.getEncoder(),encoder.getDecoder().getOutputLayer());



        DoubleMatrix reconstruct = encoder.reconstruct(data.getFirst());
        for(int j = 0; j < data.numExamples(); j++) {

            DoubleMatrix draw1 = data.get(j).getFirst().mul(255);
            DoubleMatrix reconstructed2 = reconstruct.getRow(j);
            DoubleMatrix draw2 = reconstructed2.mul(255);

            DrawReconstruction d = new DrawReconstruction(draw1);
            d.title = "REAL";
            d.draw();
            DrawReconstruction d2 = new DrawReconstruction(draw2);
            d2.title = "TEST";
            d2.draw();
            Thread.sleep(10000);
            d.frame.dispose();
            d2.frame.dispose();
        }



    }

}
