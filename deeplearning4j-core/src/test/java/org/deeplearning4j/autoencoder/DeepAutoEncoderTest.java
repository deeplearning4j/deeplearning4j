package org.deeplearning4j.autoencoder;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

/**
 * Created by agibsonccc on 5/11/14.
 */
public class DeepAutoEncoderTest {
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderTest.class);

    @Test
    public void testWithMnist() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(100);
        DataSet data = fetcher.next();
        data.filterAndStrip(new int[]{0, 1});
        log.info("Training on " + data.numExamples());
        StopWatch watch = new StopWatch();

        data = DataSet.merge(data.asList().subList(0,10));
        DBN dbn = new DBN.Builder()
                .withHiddenUnitsByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN))
               .hiddenLayerSizes(new int[]{1000, 500, 250, 30})
                .numberOfInputs(784)
                .numberOfOutPuts(2)
                .build();

        dbn.pretrain(data.getFirst(),new Object[]{1,1e-2,10000});

        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn,new Object[]{1,1e-2,10000});
        encoder.finetune(data.getFirst(),1e-2,1000);

        DoubleMatrix reconstruct = encoder.reconstruct(data.getFirst());
        for(int j = 0; j < data.numExamples(); j++) {

            DoubleMatrix draw1 = data.get(j).getFirst().mul(255);
            DoubleMatrix reconstructed2 = reconstruct.getRow(j);
            DoubleMatrix draw2 = reconstructed2.mul(255);

            DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
            d.title = "REAL";
            d.draw();
            DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2);
            d2.title = "TEST";
            d2.draw();
            Thread.sleep(10000);
            d.frame.dispose();
            d2.frame.dispose();
        }



    }

}
