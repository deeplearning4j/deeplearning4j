package org.deeplearning4j.nn.conf.layers;

import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class LayerConfigValidationTest {


    @Test(expected = IllegalStateException.class)
    public void testDropConnect() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .useDropConnect(true)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testL1L2NotSet() {
        // Warning thrown only since some layers may not have l1 or l2
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .regularization(true)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testRegNotSetL1Global() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .l1(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testRegNotSetL2Local() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).l2(0.5).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testWeightInitDistNotSet() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.3)
                .dist(new GaussianDistribution(1e-3, 2))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetGlobal(){
        // Warnings only thrown
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .momentum(1.0)
                .momentumAfter(testMomentumAfter)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetLocalMomentum(){
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).momentum(0.3).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test
    public void testNesterovsNotSetLocalMuAfter(){
        // Warnings only thrown
        Map<Integer, Double> testMomentumAfter = new HashMap<>();
        testMomentumAfter.put(0, 0.1);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).momentumAfter(testMomentumAfter).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testAdaDeltaValidation() {
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .rho(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).rho(0.01).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

    }

    @Test
    public void testRmsPropValidation() {
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .rmsDecay(2.0)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).rmsDecay(1.0).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).updater(Updater.ADADELTA).rho(0.5).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test
    public void testAdamValidation() {
        // Warnings only thrown
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .adamMeanDecay(0.5)
                .adamVarDecay(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2)
                        .adamMeanDecay(0.6).adamVarDecay(0.7).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingDecayRate(){
        double lr = 2;
        double power = 3;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Poly)
                .lrPolicyPower(power)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }

    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingPower(){
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Inverse)
                .lrPolicyDecayRate(lrDecayRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

    }

    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingSteps(){
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Step)
                .lrPolicyDecayRate(lrDecayRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

    }

    @Test(expected = IllegalStateException.class)
    public void testLRPolicyMissingSchedule(){
        double lr = 2;
        double lrDecayRate = 5;
        int iterations = 1;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .learningRate(lr)
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .lrPolicyDecayRate(lrDecayRate)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build() )
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
    }


}
