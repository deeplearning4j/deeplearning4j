package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.conf.Updater;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 27/03/2017.
 */
public class TestTransferLearningJson {

    @Test
    public void testJsonYaml() {

        FineTuneConfiguration c = new FineTuneConfiguration.Builder().activation(Activation.ELU).backprop(true)
                        .updater(Updater.ADAGRAD).biasLearningRate(10.0).build();

        String asJson = c.toJson();
        String asYaml = c.toYaml();

        FineTuneConfiguration fromJson = FineTuneConfiguration.fromJson(asJson);
        FineTuneConfiguration fromYaml = FineTuneConfiguration.fromYaml(asYaml);

        //        System.out.println(asJson);

        assertEquals(c, fromJson);
        assertEquals(c, fromYaml);
        assertEquals(asJson, fromJson.toJson());
        assertEquals(asYaml, fromYaml.toYaml());
    }

}
