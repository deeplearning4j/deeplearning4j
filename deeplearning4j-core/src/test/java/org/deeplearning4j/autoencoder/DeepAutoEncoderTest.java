package org.deeplearning4j.autoencoder;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.dbn.DBN;

import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.rmi.server.Activation;

import java.util.Collections;

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
                .withHiddenUnitsByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN))
                .numberOfOutPuts(2).sampleFromHiddenActivations(true)
                .activateForLayer(Collections.singletonMap(3,Activations.linear()))
                .build();

        dbn.pretrain(data.getFirst(),new Object[]{1,1e-1,1000});


        DeepAutoEncoder encoder = new DeepAutoEncoder(dbn);

        assertEquals(encoder.getLayers()[0].getW().length,encoder.getOutputLayer().getW().length);
        assertEquals(encoder.getLayers()[1].getW().length,encoder.getLayers()[encoder.getLayers().length - 1].getW().length);
        assertEquals(encoder.getLayers()[2].getW().length,encoder.getLayers()[encoder.getLayers().length - 2].getW().length);
        assertEquals(encoder.getSigmoidLayers()[0].getActivationFunction(), Activations.sigmoid());
        assertEquals(encoder.getSigmoidLayers()[1].getActivationFunction(),Activations.sigmoid());
        assertEquals(encoder.getSigmoidLayers()[2].getActivationFunction(),Activations.sigmoid());
        assertEquals(encoder.getSigmoidLayers()[3].getActivationFunction().type(),Activations.linear().type());
        assertEquals(encoder.getSigmoidLayers()[4].getActivationFunction(),Activations.sigmoid());
        assertEquals(encoder.getSigmoidLayers()[5].getActivationFunction(),Activations.sigmoid());
        assertEquals(encoder.getSigmoidLayers()[6].getActivationFunction(),Activations.sigmoid());
        //7 activations + 1 output
        assertEquals(7,encoder.getSigmoidLayers().length);
        encoder.finetune(data.getFirst(),1e-1,1000);




    }

}
