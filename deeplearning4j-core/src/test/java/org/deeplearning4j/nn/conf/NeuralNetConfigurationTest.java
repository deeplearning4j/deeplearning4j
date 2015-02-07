package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetConfigurationTest {
    @Test
    public void testJson() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().dist(Distributions.normal(new MersenneTwister(123),1e-1))
                .layerFactory(LayerFactories.getFactory(RBM.class))
                .build();
        String json = conf.toJson();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromJson(json);
        assertEquals(conf,read);
    }


}
