package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class LayerConfigTest {

    @Test
    public void testActivationLayerwiseOverride(){
        //Without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation("relu")
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getActivationFunction().toString(), "relu");
        assertEquals(conf.getConf(1).getLayer().getActivationFunction().toString(), "relu");

        //With
        conf = new NeuralNetConfiguration.Builder()
                .activation("relu")
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).activation("tanh").build())
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getActivationFunction().toString(), "relu");
        assertEquals(conf.getConf(1).getLayer().getActivationFunction().toString(), "tanh");
        }


    @Test
    public void testWeightBiasInitLayerwiseOverride(){
        //Without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.SIZE).dist(new NormalDistribution(0, 1.0))
                .biasInit(1)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getWeightInit(), WeightInit.SIZE);
        assertEquals(conf.getConf(1).getLayer().getWeightInit(), WeightInit.SIZE);
        assertEquals(conf.getConf(0).getLayer().getDist().toString(), "NormalDistribution{mean=0.0, std=1.0}");
        assertEquals(conf.getConf(1).getLayer().getDist().toString(), "NormalDistribution{mean=0.0, std=1.0}");
        assertEquals(conf.getConf(0).getLayer().getBiasInit(), 1, 0.0);
        assertEquals(conf.getConf(1).getLayer().getBiasInit(), 1, 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.SIZE).dist(new NormalDistribution(0, 1.0))
                .biasInit(1)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                .weightInit(WeightInit.NORMALIZED).dist(new UniformDistribution(0,1)).biasInit(0).build())
        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getWeightInit(), WeightInit.SIZE);
        assertEquals(conf.getConf(1).getLayer().getWeightInit(), WeightInit.NORMALIZED);
        assertEquals(conf.getConf(0).getLayer().getDist().toString(), "NormalDistribution{mean=0.0, std=1.0}");
        assertEquals(conf.getConf(1).getLayer().getDist().toString(), "UniformDistribution{lower=0.0, upper=1.0}");
        assertEquals(conf.getConf(0).getLayer().getBiasInit(), 1, 0.0);
        assertEquals(conf.getConf(1).getLayer().getBiasInit(), 0, 0.0);

    }

    @Test
    public void testLrL1L2LayerwiseOverride(){
        //Idea: Set some common values for all layers. Then selectively override
        // the global config, and check they actually work.

        //Learning rate without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .learningRateScoreBasedDecayRate(10)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getLearningRate(), 0.3, 0.0);
        assertEquals(conf.getConf(1).getLayer().getLearningRate(), 0.3, 0.0);
        assertEquals(conf.getConf(0).getLayer().getLrScoreBasedDecay(), 10, 0.0);
        assertEquals(conf.getConf(1).getLayer().getLrScoreBasedDecay(), 10, 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .learningRateScoreBasedDecayRate(10)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).learningRate(0.2)
                        .learningRateScoreBasedDecayRate(8).build() )
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getLearningRate(), 0.3, 0.0);
        assertEquals(conf.getConf(1).getLayer().getLearningRate(), 0.2, 0.0);
        assertEquals(conf.getConf(0).getLayer().getLrScoreBasedDecay(), 10, 0.0);
        assertEquals(conf.getConf(1).getLayer().getLrScoreBasedDecay(), 8, 0.0);

        //L1 and L2 without layerwise override:
        conf = new NeuralNetConfiguration.Builder()
                .l1(0.1).l2(0.2)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getL1(), 0.1, 0.0);
        assertEquals(conf.getConf(1).getLayer().getL1(), 0.1, 0.0);
        assertEquals(conf.getConf(0).getLayer().getL2(), 0.2, 0.0);
        assertEquals(conf.getConf(1).getLayer().getL2(), 0.2, 0.0);

        //L1 and L2 with layerwise override:
        conf = new NeuralNetConfiguration.Builder()
                .l1(0.1).l2(0.2)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l1(0.9).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.8).build() )
                .build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getL1(), 0.9, 0.0);
        assertEquals(conf.getConf(1).getLayer().getL1(), 0.1, 0.0);
        assertEquals(conf.getConf(0).getLayer().getL2(), 0.2, 0.0);
        assertEquals(conf.getConf(1).getLayer().getL2(), 0.8, 0.0);
    }



    @Test
    public void testDropoutLayerwiseOverride(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dropOut(1.0)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getDropOut(), 1.0, 0.0);
        assertEquals(conf.getConf(1).getLayer().getDropOut(), 1.0, 0.0);

        conf = new NeuralNetConfiguration.Builder()
                .dropOut(1.0)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).dropOut(2.0).build())
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getDropOut(), 1.0, 0.0);
        assertEquals(conf.getConf(1).getLayer().getDropOut(), 2.0, 0.0);
    }

    @Test
    public void testMomentumLayerwiseOverride(){
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(1.0)
                .momentumAfter(testMomentumAfter)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getMomentum(), 1.0, 0.0);
        assertEquals(conf.getConf(1).getLayer().getMomentum(), 1.0, 0.0);
        assertEquals(conf.getConf(0).getLayer().getMomentumAfter().get(0), 0.1, 0.0);
        assertEquals(conf.getConf(1).getLayer().getMomentumAfter().get(0), 0.1, 0.0);

        Map<Integer, Double> testMomentumAfter2 = new HashMap<>();
        testMomentumAfter2.put(0, 0.2);

        conf = new NeuralNetConfiguration.Builder()
                .momentum(1.0)
                .momentumAfter(testMomentumAfter)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).momentum(2.0).momentumAfter(testMomentumAfter2).build())
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getMomentum(), 1.0, 0.0);
        assertEquals(conf.getConf(1).getLayer().getMomentum(), 2.0, 0.0);
        assertEquals(conf.getConf(0).getLayer().getMomentumAfter().get(0), 0.1, 0.0);
        assertEquals(conf.getConf(1).getLayer().getMomentumAfter().get(0), 0.2, 0.0);

    }

    @Test
    public void testUpdaterRhoRmsDecayLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADADELTA)
                .rho(0.5)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).rho(0.01).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getUpdater().toString(), "ADADELTA");
        assertEquals(conf.getConf(1).getLayer().getUpdater().toString(), "ADADELTA");
        assertEquals(conf.getConf(0).getLayer().getRho(), 0.5, 0.0);
        assertEquals(conf.getConf(1).getLayer().getRho(), 0.01, 0.0);

        conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.RMSPROP)
                .rmsDecay(2.0)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).rmsDecay(1.0).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.ADADELTA).rho(0.5).build())
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getUpdater().toString(), "RMSPROP");
        assertEquals(conf.getConf(1).getLayer().getUpdater().toString(), "ADADELTA");
        assertEquals(conf.getConf(1).getLayer().getRho(), 0.5, 0.0);
        assertEquals(conf.getConf(0).getLayer().getRmsDecay(), 1.0, 0.0);
        assertEquals(conf.getConf(1).getLayer().getRmsDecay(), 2.0, 0.0);
    }


    @Test
    public void testUpdaterAdamParamsLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                        .adamMeanDecay(0.6).adamVarDecay(0.7).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getUpdater().toString(), "ADAM");
        assertEquals(conf.getConf(1).getLayer().getUpdater().toString(), "ADAM");
        assertEquals(conf.getConf(0).getLayer().getAdamMeanDecay(), 0.5, 0.0);
        assertEquals(conf.getConf(1).getLayer().getAdamMeanDecay(), 0.6, 0.0);
        assertEquals(conf.getConf(0).getLayer().getAdamVarDecay(), 0.5, 0.0);
        assertEquals(conf.getConf(1).getLayer().getAdamVarDecay(), 0.7, 0.0);

        conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).adamMeanDecay(1.0).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.ADADELTA).rho(0.5).build())
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(conf.getConf(0).getLayer().getUpdater().toString(), "ADAM");
        assertEquals(conf.getConf(1).getLayer().getUpdater().toString(), "ADADELTA");
        assertEquals(conf.getConf(0).getLayer().getAdamMeanDecay(), 1.0, 0.0);
        assertEquals(conf.getConf(0).getLayer().getAdamVarDecay(), 0.5, 0.0);
    }

}
