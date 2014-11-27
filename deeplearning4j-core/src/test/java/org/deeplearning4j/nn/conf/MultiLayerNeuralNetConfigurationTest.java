package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerNeuralNetConfigurationTest {

    @Test
    public void testJson() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(3).hiddenLayerSizes(new int[]{3,2,2}).build();
        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf,from);
    }


}
