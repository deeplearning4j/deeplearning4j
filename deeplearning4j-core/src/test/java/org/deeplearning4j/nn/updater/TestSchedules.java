package org.deeplearning4j.nn.updater;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by nyghtowl on 10/27/15.
 */
public class TestSchedules {
    int nIn = 3;
    int nOut = 2;
    double epsilon = 1e-8;
    INDArray weightGradient = Nd4j.ones(nIn,nOut);
    INDArray biasGradient = Nd4j.ones(1,nOut);
    Gradient gradientSingle = new DefaultGradient();
    Gradient gradientMLN = new DefaultGradient();
    INDArray val, gradExpected, vPrev;
    String key;
    Map<String, INDArray> tmpStorage, tmpStorage2, tmpStorage3, tmpStorage4 = new HashMap<>();
    org.deeplearning4j.nn.conf.Updater[] updaters = {
            org.deeplearning4j.nn.conf.Updater.SGD,
            org.deeplearning4j.nn.conf.Updater.ADAGRAD,
            org.deeplearning4j.nn.conf.Updater.ADAM,
            org.deeplearning4j.nn.conf.Updater.RMSPROP,
    };

    @Before
    public void beforeDo(){
        int nLayers = 2;
        String wKey, bKey;

        gradientSingle.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientSingle.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int j=0; j < nLayers; j++){
            wKey = String.valueOf(j) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientMLN.setGradientFor(wKey, weightGradient);
            bKey = String.valueOf(j) + "_" + DefaultParamInitializer.BIAS_KEY ;
            gradientMLN.setGradientFor(bKey, biasGradient);
        }

        val = null;
        gradExpected = null;
        vPrev = null;
        tmpStorage = new HashMap<>();
        tmpStorage2 = new HashMap<>();
        tmpStorage3 = new HashMap<>();
        tmpStorage4 = new HashMap<>();

    }

    @Test
    public void testLearningRateAfterSingleLayer() {
        Map<Integer, Double> learningRateAfter = new HashMap<>();
        learningRateAfter.put(1, 0.2);
        int iterations = 2;

        for (org.deeplearning4j.nn.conf.Updater updaterFunc : updaters) {
            double lr = 1e-2;
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .learningRate(lr).learningRateAfter(learningRateAfter).schedules(true).iterations(iterations)
                    .layer(new DenseLayer.Builder()
                            .nIn(nIn).nOut(nOut).updater(updaterFunc).build())
                    .build();

            Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
            Updater updater = UpdaterCreator.getUpdater(layer);

            Gradient gradientActual = new DefaultGradient();
            gradientActual.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
            gradientActual.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

            Gradient gradientExpected = new DefaultGradient();
            gradientExpected.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
            gradientExpected.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

            for (int i = 0; i < 2; i++) {
                updater.update(layer, gradientActual, i, 1);

                if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.SGD))
                    lr = testSGDComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAGRAD))
                    lr = testAdaGradComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAM))
                    lr = testAdamComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.RMSPROP))
                    lr = testRMSPropComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                assertEquals(lr, layer.conf().getLayer().getLearningRate(), 1e-4);
            }
        }
    }


    @Test
    public void testLearningRateAfterMLN(){
        Map<Integer,Double> learningRateAfter = new HashMap<>();
        learningRateAfter.put(1, 0.2);
        int iterations = 2;
        int[] nIns = {4,2};
        int[] nOuts = {2,3};

        for (org.deeplearning4j.nn.conf.Updater updaterFunc : updaters) {
            double lr = 1e-2;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .learningRate(lr).learningRateAfter(learningRateAfter).schedules(true).iterations(iterations)
                    .updater(updaterFunc)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).build())
                    .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).build())
                    .backprop(true).pretrain(false)
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            Updater updater = UpdaterCreator.getUpdater(net);
            String wKey, bKey;
            Gradient gradientActual = new DefaultGradient();
            Gradient gradientExpected = new DefaultGradient();
            for (int k = 0; k < net.getnLayers(); k++) {
                wKey = String.valueOf(k) + "_" + DefaultParamInitializer.WEIGHT_KEY;
                gradientActual.setGradientFor(wKey, weightGradient);
                gradientExpected.setGradientFor(wKey, weightGradient);
                bKey = String.valueOf(k) + "_" + DefaultParamInitializer.BIAS_KEY;
                gradientActual.setGradientFor(bKey, biasGradient);
                gradientExpected.setGradientFor(bKey, biasGradient);
            }

            for (int i = 0; i < 2; i++) {
                updater.update(net, gradientActual, i, 1);
                if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.SGD))
                    lr = testSGDComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAGRAD))
                    lr = testAdaGradComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.ADAM))
                    lr = testAdamComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);
                else if(updaterFunc.equals(org.deeplearning4j.nn.conf.Updater.RMSPROP))
                    lr = testRMSPropComputation(gradientActual, gradientExpected, lr, learningRateAfter, i);

                assertEquals(lr, net.getLayer(1).conf().getLayer().getLearningRate(), 1e-4);
            }
        }
    }


    @Test
    public void testmomentumAfterUpdaterSingleLayer(){
        double lr = 1e-2;
        double mu = 0.6;
        Map<Integer,Double> momentumAfter = new HashMap<>();
        momentumAfter.put(1, 0.2);
        int iterations = 2;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).momentum(mu).momentumAfter(momentumAfter).schedules(true).iterations(iterations)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .build();

        Layer layer = LayerFactories.getFactory(conf).create(conf, null, 0);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientExpected = new DefaultGradient();
        gradientExpected.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientExpected.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradientSingle, i, 1);
            mu = testNesterovsComputation(gradientSingle, gradientExpected, lr, mu, momentumAfter, i);
            assertEquals(mu, layer.conf().getLayer().getMomentum(), 1e-4);
        }
    }

    @Test
    public void testMomentumAfterMLN(){
        double lr = 1e-2;
        double mu = 0.6;
        Map<Integer,Double> momentumAfter = new HashMap<>();
        momentumAfter.put(1, 0.2);
        int iterations = 2;
        int nLayers = 2;
        int[] nIns = {4,2};
        int[] nOuts = {2,3};

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).momentum(mu).momentumAfter(momentumAfter).schedules(true).iterations(iterations)
                .list(nLayers)
                .layer(0, new DenseLayer.Builder().nIn(nIns[0]).nOut(nOuts[0]).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .layer(1, new OutputLayer.Builder().nIn(nIns[1]).nOut(nOuts[1]).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater updater = UpdaterCreator.getUpdater(net);

        String wKey, bKey;

        Gradient gradientExpected = new DefaultGradient();
        for (int k=0; k < nLayers; k++){
            wKey = String.valueOf(k) + "_" + DefaultParamInitializer.WEIGHT_KEY;
            gradientExpected.setGradientFor(wKey, weightGradient);
            bKey = String.valueOf(k) + "_" + DefaultParamInitializer.BIAS_KEY ;
            gradientExpected.setGradientFor(bKey, biasGradient);
        }

        for (int i = 0; i < 2; i++) {
            updater.update(net, gradientMLN, i, 1);
            mu = testNesterovsComputation(gradientMLN, gradientExpected, lr, mu, momentumAfter, i);
            assertEquals(mu, net.getLayer(1).conf().getLayer().getMomentum(), 1e-4);
        }
    }

///// Updater Calculations

    public double testSGDComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i){
        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            gradExpected = val.mul(lr);
            gradientExpected.setGradientFor(key, gradExpected);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));
        }
        return lr;
    }

    public double testNesterovsComputation(Gradient gradientActual, Gradient gradientExpected, double lr, double mu, Map<Integer, Double> momentumAfter, int i) {

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if(momentumAfter !=null)
                mu = (momentumAfter.containsKey(i)) ? momentumAfter.get(i) : mu;
            key = entry.getKey();
            val = entry.getValue();
            INDArray vTmp = tmpStorage.get(key);

            if(vTmp == null)
                vTmp = Nd4j.zeros(val.shape());
            vPrev = vTmp;
            vTmp = vPrev.mul(mu).subi(val.mul(lr));
            gradExpected = vPrev.muli(mu).addi(vTmp.mul(-mu - 1));
            gradientExpected.setGradientFor(key, gradExpected);

            assertEquals(gradExpected, gradientActual.getGradientFor(entry.getKey()));
            tmpStorage.put(key, vTmp);
        }
        return mu;
    }


    public double testAdaGradComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i) {

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            INDArray historicalGradient = tmpStorage.get(key);

            if(historicalGradient == null) historicalGradient = val.mul(val);
            else historicalGradient.addi(val.mul(val));

            gradExpected = Transforms.sqrt(historicalGradient.add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));
            gradientExpected.setGradientFor(key, gradExpected);
            tmpStorage.put(key, historicalGradient);
        }

        return lr;
    }

    public double testAdamComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i) {
        double beta1 = 0.9;
        double beta2 = 0.999;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();

            INDArray mTmp = tmpStorage2.get(key);
            INDArray vTmp = tmpStorage3.get(key);

            if(mTmp == null) mTmp = Nd4j.zeros(val.shape());
            if(vTmp == null) vTmp = Nd4j.zeros(val.shape());

            mTmp.muli(beta1).addi(val.mul(1.0-beta1));
            vTmp.muli(beta2).addi(val.mul(val).mul(1.0-beta2));

            double beta1t = FastMath.pow(beta1, i);
            double beta2t = FastMath.pow(beta2, i);
            double alphat = lr * FastMath.sqrt(1-beta2t)/(1-beta1t);

            gradExpected = mTmp.mul(alphat).divi(Transforms.sqrt(vTmp).addi(epsilon));
            gradientExpected.setGradientFor(key, gradExpected);
            assertEquals(gradExpected, gradientActual.getGradientFor(key));

            tmpStorage2.put(key, mTmp);
            tmpStorage3.put(key, vTmp);
        }
        return lr;
    }

    public double testRMSPropComputation(Gradient gradientActual, Gradient gradientExpected, double lr, Map<Integer, Double> learningRateAfter, int i) {
        double rmsDecay = 0.95;

        for (Map.Entry<String, INDArray> entry : gradientExpected.gradientForVariable().entrySet()) {
            if (learningRateAfter != null)
                lr = (learningRateAfter.containsKey(i)) ? learningRateAfter.get(i) : lr;
            key = entry.getKey();
            val = entry.getValue();
            INDArray lastGTmp = tmpStorage4.get(key);

            if(lastGTmp==null)
                lastGTmp = Nd4j.zeros(val.shape());

            lastGTmp.muli(rmsDecay).addi(val.mul(val).muli(1 - rmsDecay));
            gradExpected = val.mul(lr).div(Transforms.sqrt(lastGTmp.add(Nd4j.EPS_THRESHOLD)));
            gradientExpected.setGradientFor(key, gradExpected);

            assertEquals(gradExpected, gradientActual.getGradientFor(key));
            tmpStorage4.put(key, lastGTmp);
        }

        return lr;
    }
}

