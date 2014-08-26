package org.deeplearning4j.autoencoder;

import static org.junit.Assert.*;
import static org.junit.Assume.*;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.dbn.DBN;

import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.rbm.RBM;
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
        fetcher.fetch(20);
        DataSet data = fetcher.next();
        data.filterAndStrip(new int[]{0, 1});
        log.info("Training on " + data.numExamples());

        DBN dbn = new DBN.Builder()
                .hiddenLayerSizes(new int[]{1000, 500, 250, 10})
                .numberOfInputs(784).useRBMPropUpAsActivation(true)
                .withHiddenUnitsByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN))
                .numberOfOutPuts(2).sampleFromHiddenActivations(true)
                .activateForLayer(Collections.singletonMap(3, Activations.linear()))
                .build();

        dbn.pretrain(data.getFeatureMatrix(),new Object[]{1,1e-1,1000});


        DeepAutoEncoder encoder = new DeepAutoEncoder.Builder()
                .withEncoder(dbn).build();
        assertEquals(7,encoder.getLayers().length);
        assertEquals(7,encoder.getSigmoidLayers().length);
        assertEquals(encoder.getLayers()[0].getW().length(),encoder.getOutputLayer().getW().length());
        for(int i = 0; i < encoder.getLayers().length;i++)
            assumeNotNull("Layer " + i + " was null",encoder.getLayers()[i]);
        assertEquals(encoder.getLayers()[1].getW().length(),encoder.getLayers()[encoder.getLayers().length - 1].getW().length());
        assertEquals(encoder.getLayers()[2].getW().length(),encoder.getLayers()[encoder.getLayers().length - 2].getW().length());
        //7 activations + 1 output
        assertEquals(7,encoder.getSigmoidLayers().length);
        encoder.finetune(data.getFeatureMatrix(),1e-1f,1000);

        //output layer is transpose of first should be same length
        assertEquals(encoder.getLayers()[0].getW().length(),encoder.getOutputLayer().getW().length());



    }

}
