package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class LayerConfigTest {

    @Test
    public void testLayerName() {

        String name1 = "genisys";
        String name2 = "bill";

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).name(name1).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).name(name2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(name1, conf.getConf(0).getLayerName());
        assertEquals(name2, conf.getConf(1).getLayerName());

    }

    @Test
    public void testActivationLayerwiseOverride() {
        //Without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().activation(Activation.RELU).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals("relu", ((BaseLayer) conf.getConf(0)).getActivationFn().toString());
        assertEquals("relu", ((BaseLayer) conf.getConf(1)).getActivationFn().toString());

        //With
        conf = new NeuralNetConfiguration.Builder().activation(Activation.RELU).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).activation(Activation.TANH).build()).build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals("relu", ((BaseLayer) conf.getConf(0)).getActivationFn().toString());
        assertEquals("tanh", ((BaseLayer) conf.getConf(1)).getActivationFn().toString());
    }


    @Test
    public void testWeightBiasInitLayerwiseOverride() {
        //Without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 1.0)).biasInit(1).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(0)).getWeightInit());
        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(1)).getWeightInit());
        assertEquals("NormalDistribution{mean=0.0, std=1.0}",
                        ((BaseLayer) conf.getConf(0)).getDist().toString());
        assertEquals("NormalDistribution{mean=0.0, std=1.0}",
                        ((BaseLayer) conf.getConf(1)).getDist().toString());
        assertEquals(1, ((BaseLayer) conf.getConf(0)).getBiasInit(), 0.0);
        assertEquals(1, ((BaseLayer) conf.getConf(1)).getBiasInit(), 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.DISTRIBUTION)
                        .dist(new NormalDistribution(0, 1.0)).biasInit(1).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1,
                                        new DenseLayer.Builder().nIn(2).nOut(2).weightInit(WeightInit.DISTRIBUTION)
                                                        .dist(new UniformDistribution(0, 1)).biasInit(0).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(0)).getWeightInit());
        assertEquals(WeightInit.DISTRIBUTION, ((BaseLayer) conf.getConf(1)).getWeightInit());
        assertEquals("NormalDistribution{mean=0.0, std=1.0}",
                        ((BaseLayer) conf.getConf(0)).getDist().toString());
        assertEquals("UniformDistribution{lower=0.0, upper=1.0}",
                        ((BaseLayer) conf.getConf(1)).getDist().toString());
        assertEquals(1, ((BaseLayer) conf.getConf(0)).getBiasInit(), 0.0);
        assertEquals(0, ((BaseLayer) conf.getConf(1)).getBiasInit(), 0.0);
    }

    /*
    @Test
    public void testLrL1L2LayerwiseOverride() {
        //Idea: Set some common values for all layers. Then selectively override
        // the global config, and check they actually work.

        //Learning rate without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().learningRate(0.3).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.3, ((BaseLayer) conf.getConf(0)).getLearningRate(), 0.0);
        assertEquals(0.3, ((BaseLayer) conf.getConf(1)).getLearningRate(), 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder().learningRate(0.3).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).learningRate(0.2).build()).build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.3, ((BaseLayer) conf.getConf(0)).getLearningRate(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(1)).getLearningRate(), 0.0);

        //L1 and L2 without layerwise override:
        conf = new NeuralNetConfiguration.Builder().l1(0.1).l2(0.2).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.1, ((BaseLayer) conf.getConf(0)).getL1(), 0.0);
        assertEquals(0.1, ((BaseLayer) conf.getConf(1)).getL1(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(0)).getL2(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(1)).getL2(), 0.0);

        //L1 and L2 with layerwise override:
        conf = new NeuralNetConfiguration.Builder().l1(0.1).l2(0.2).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l1(0.9).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.8).build()).build();
        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.9, ((BaseLayer) conf.getConf(0)).getL1(), 0.0);
        assertEquals(0.1, ((BaseLayer) conf.getConf(1)).getL1(), 0.0);
        assertEquals(0.2, ((BaseLayer) conf.getConf(0)).getL2(), 0.0);
        assertEquals(0.8, ((BaseLayer) conf.getConf(1)).getL2(), 0.0);
    }*/



    @Test
    public void testDropoutLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().dropOut(1.0).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(new Dropout(1.0), conf.getConf(0).getIDropout());
        assertEquals(new Dropout(1.0), conf.getConf(1).getIDropout());

        conf = new NeuralNetConfiguration.Builder().dropOut(1.0).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).dropOut(2.0).build()).build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(new Dropout(1.0), conf.getConf(0).getIDropout());
        assertEquals(new Dropout(2.0), conf.getConf(1).getIDropout());
    }

    @Test
    public void testMomentumLayerwiseOverride() {
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter)))
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.1, ((Nesterovs)((BaseLayer) conf.getConf(0)).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);
        assertEquals(0.1, ((Nesterovs)((BaseLayer) conf.getConf(1)).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);

        Map<Integer, Double> testMomentumAfter2 = new HashMap<>();
        testMomentumAfter2.put(0, 0.2);

        conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter) ))
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build()).layer(1, new DenseLayer.Builder()
                                        .nIn(2).nOut(2).updater(new Nesterovs(1.0, new MapSchedule(ScheduleType.ITERATION, testMomentumAfter2))).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();
        assertEquals(0.1, ((Nesterovs)((BaseLayer) conf.getConf(0)).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);
        assertEquals(0.2, ((Nesterovs)((BaseLayer) conf.getConf(1)).getIUpdater()).getMomentumISchedule().valueAt(0,0), 0.0);
    }

    @Test
    public void testUpdaterRhoRmsDecayLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new AdaDelta(0.5, 0.9)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new AdaDelta(0.01,0.9)).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertTrue(((BaseLayer) conf.getConf(0)).getIUpdater() instanceof AdaDelta);
        assertTrue(((BaseLayer) conf.getConf(1)).getIUpdater() instanceof AdaDelta);
        assertEquals(0.5, ((AdaDelta)((BaseLayer) conf.getConf(0)).getIUpdater()).getRho(), 0.0);
        assertEquals(0.01, ((AdaDelta)((BaseLayer) conf.getConf(1)).getIUpdater()).getRho(), 0.0);

        conf = new NeuralNetConfiguration.Builder().updater(new RmsProp(1.0, 2.0, RmsProp.DEFAULT_RMSPROP_EPSILON)).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).updater(new RmsProp(1.0, 1.0, RmsProp.DEFAULT_RMSPROP_EPSILON)).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new AdaDelta(0.5,AdaDelta.DEFAULT_ADADELTA_EPSILON)).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertTrue(((BaseLayer) conf.getConf(0)).getIUpdater() instanceof RmsProp);
        assertTrue(((BaseLayer) conf.getConf(1)).getIUpdater() instanceof AdaDelta);
        assertEquals(1.0, ((RmsProp) ((BaseLayer) conf.getConf(0)).getIUpdater()).getRmsDecay(), 0.0);
        assertEquals(0.5, ((AdaDelta) ((BaseLayer) conf.getConf(1)).getIUpdater()).getRho(), 0.0);
    }


    @Test
    public void testUpdaterAdamParamsLayerwiseOverride() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(1.0, 0.5, 0.5, 1e-8))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(new Adam(1.0, 0.6, 0.7, 1e-8)).build())
                        .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(0.5, ((Adam) ((BaseLayer) conf.getConf(0)).getIUpdater()).getBeta1(), 0.0);
        assertEquals(0.6, ((Adam) ((BaseLayer) conf.getConf(1)).getIUpdater()).getBeta1(), 0.0);
        assertEquals(0.5, ((Adam) ((BaseLayer) conf.getConf(0)).getIUpdater()).getBeta2(), 0.0);
        assertEquals(0.7, ((Adam) ((BaseLayer) conf.getConf(1)).getIUpdater()).getBeta2(), 0.0);
    }

    @Test
    public void testGradientNormalizationLayerwiseOverride() {

        //Learning rate without layerwise override:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build()).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(GradientNormalization.ClipElementWiseAbsoluteValue,
                        ((BaseLayer) conf.getConf(0)).getGradientNormalization());
        assertEquals(GradientNormalization.ClipElementWiseAbsoluteValue,
                        ((BaseLayer) conf.getConf(1)).getGradientNormalization());
        assertEquals(10, ((BaseLayer) conf.getConf(0)).getGradientNormalizationThreshold(), 0.0);
        assertEquals(10, ((BaseLayer) conf.getConf(1)).getGradientNormalizationThreshold(), 0.0);

        //With:
        conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10).list()
                        .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                        .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                                        .gradientNormalization(GradientNormalization.None)
                                        .gradientNormalizationThreshold(2.5).build())
                        .build();

        net = new MultiLayerNetwork(conf);
        net.init();

        assertEquals(GradientNormalization.ClipElementWiseAbsoluteValue,
                        ((BaseLayer) conf.getConf(0)).getGradientNormalization());
        assertEquals(GradientNormalization.None, ((BaseLayer) conf.getConf(1)).getGradientNormalization());
        assertEquals(10, ((BaseLayer) conf.getConf(0)).getGradientNormalizationThreshold(), 0.0);
        assertEquals(2.5, ((BaseLayer) conf.getConf(1)).getGradientNormalizationThreshold(), 0.0);
    }
}
