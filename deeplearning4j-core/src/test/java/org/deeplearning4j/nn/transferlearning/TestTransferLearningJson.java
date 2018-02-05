package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaGrad;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 27/03/2017.
 */
public class TestTransferLearningJson extends BaseDL4JTest {

    @Test
    public void testJsonYaml() {

        FineTuneConfiguration c = new FineTuneConfiguration.Builder().activation(Activation.ELU).backprop(true)
                        .updater(new AdaGrad(1.0)).biasUpdater(new AdaGrad(10.0)).build();

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
