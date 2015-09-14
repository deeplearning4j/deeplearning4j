package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LayerConfigTest {

    @Test
    public void testLrL1L2LayerwiseOverride(){
        //Idea: Set some common values for all layers. Then selectively override
        // the global config, and check they actually work.

        //Learning rate without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLr(), 0.3, 0.0);
        assertEquals(conf.getConf(1).getLr(), 0.3, 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).learningRate(0.2).build() )
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLr(), 0.3, 0.0);
        assertEquals(conf.getConf(1).getLr(), 0.2, 0.0);

        //L1 and L2 without layerwise override:
        conf = new NeuralNetConfiguration.Builder()
                .l1(0.1).l2(0.2)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getL1(), 0.1, 0.0);
        assertEquals(conf.getConf(1).getL1(), 0.1, 0.0);
        assertEquals(conf.getConf(0).getL2(), 0.2, 0.0);
        assertEquals(conf.getConf(1).getL2(), 0.2, 0.0);

        //L1 and L2 with layerwise override:
        conf = new NeuralNetConfiguration.Builder()
                .l1(0.1).l2(0.2)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l1(0.9).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.8).build() )
                .build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getL1(), 0.9, 0.0);
        assertEquals(conf.getConf(1).getL1(), 0.1, 0.0);
        assertEquals(conf.getConf(0).getL2(), 0.2, 0.0);
        assertEquals(conf.getConf(1).getL2(), 0.8, 0.0);
    }
}
