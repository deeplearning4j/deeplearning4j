package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetConfigurationTest {
    @Test
    public void testJson() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layerFactory(LayerFactories.getFactory(RBM.class))
                .build();
        String json = conf.toJson();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromJson(json);
        assertEquals(conf,read);
    }


}
