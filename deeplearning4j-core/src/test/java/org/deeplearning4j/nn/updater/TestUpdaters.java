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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Field;
import java.util.*;

import static org.junit.Assert.*;

public class TestUpdaters {

    protected int nIn = 3;
    protected int nOut = 2;
    protected double epsilon = 1e-8;
    protected INDArray weightGradient = Nd4j.ones(nIn, nOut);
    protected INDArray biasGradient = Nd4j.ones(1, nOut);
    protected Gradient gradient = new DefaultGradient();
    protected INDArray val, gradExpected;
    protected String key;


    @Before
    public void beforeDo() {
        weightGradient = Nd4j.ones(nIn, nOut);
        biasGradient = Nd4j.ones(1, nOut);
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());
    }

    @Test
    public void testAdaDeltaUpdate() {
        INDArray dxSquared;
        Map<String, INDArray> msg = new HashMap<>();
        Map<String, INDArray> msdx = new HashMap<>();

        double rho = 0.85;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .rho(rho)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADADELTA).epsilon(Nd4j.EPS_THRESHOLD).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = updater.stateSizeForLayer(layer);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradient, i, 1);

            // calculations for one iteration / update

            for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
                key = entry.getKey();
                val = entry.getValue();
                INDArray msgTmp = msg.get(key);
                INDArray msdxTmp = msdx.get(key);

                if (msgTmp == null) {
                    msgTmp = Nd4j.zeros(val.shape());
                    msdxTmp = Nd4j.zeros(val.shape());
                }

                msgTmp.muli(rho);
                msgTmp.addi(val.mul(val).muli(1 - rho));

                gradExpected = Transforms.sqrt(msdxTmp.add(Nd4j.EPS_THRESHOLD))
                        .divi(Transforms.sqrt(msgTmp.add(Nd4j.EPS_THRESHOLD))).muli(val);
                gradientDup.setGradientFor(key, gradExpected);
                assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));

                msdxTmp.muli(rho);
                dxSquared = gradExpected.mul(gradExpected);
                msdxTmp.addi(dxSquared.muli(1 - rho));

                msg.put(key, msgTmp);
                msdx.put(key, msdxTmp);
            }
            assertEquals(rho, layer.conf().getLayer().getRho(), 1e-4);
        }

    }

    @Test
    public void testAdaGradUpdater() {
        double lr = 1e-2;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = updater.stateSizeForLayer(layer);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, -1, 1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = Transforms.sqrt(val.mul(val).add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, layer.conf().getLayer().getLearningRate(), 1e-4);
    }


    @Test
    public void testAdamUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).iterations(iteration).adamMeanDecay(beta1).adamVarDecay(beta2)
                .layer(new DenseLayer.Builder().nIn(nIn)
                        .nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.ADAM).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = updater.stateSizeForLayer(layer);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, iteration, 1);

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);
        double alphat = lr * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0) alphat = epsilon;

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            m.muli(beta1).addi(val.mul(1.0 - beta1));
            v.muli(beta2).addi(val.mul(val).mul(1.0 - beta2));
            gradExpected = m.mul(alphat).divi(Transforms.sqrt(v).addi(epsilon));
            if(!gradExpected.equals(gradient.getGradientFor(entry.getKey()))){
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }

        assertEquals(beta1, layer.conf().getLayer().getAdamMeanDecay(), 1e-4);
        assertEquals(beta2, layer.conf().getLayer().getAdamVarDecay(), 1e-4);

    }

    @Test
    public void testNestorovsUpdater() {
        double lr = 1e-2;
        double mu = 0.6;
        INDArray v, vPrev;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr).momentum(mu)
                .layer(new DenseLayer.Builder()
                        .nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = updater.stateSizeForLayer(layer);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, -1, 1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            v = Nd4j.zeros(val.shape());
            vPrev = v;
            v = vPrev.mul(mu).subi(val.mul(lr));
            gradExpected = vPrev.muli(mu).addi(v.mul(-mu - 1));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }

        assertEquals(mu, layer.conf().getLayer().getMomentum(), 1e-4);
    }


    @Test
    public void testRMSPropUpdater() {
        double lr = 0.01;
        double rmsDecay = 0.25;
        Map<String, INDArray> lastG = new HashMap<>();


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .rmsDecay(rmsDecay)
                .layer(new DenseLayer.Builder().nIn(nIn)
                        .nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.RMSPROP).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = updater.stateSizeForLayer(layer);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, -1, 1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

        double epsilon = 1e-8;

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            key = entry.getKey();
            val = entry.getValue();
            INDArray lastGTmp = lastG.get(key);

            if (lastGTmp == null)
                lastGTmp = Nd4j.zeros(val.shape());

            lastGTmp.muli(rmsDecay).addi(val.mul(val).muli(1 - rmsDecay));
            gradExpected = val.mul(lr).div(Transforms.sqrt(lastGTmp.add(epsilon)));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            lastG.put(key, lastGTmp);
        }
        assertEquals(rmsDecay, layer.conf().getLayer().getRmsDecay(), 1e-4);
    }

    @Test
    public void testSGDUpdater() {
        double lr = 0.05;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .layer(new DenseLayer.Builder().nIn(nIn)
                        .nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);

        updater.update(layer, gradient, -1, 1);

        Gradient gradientDup = new DefaultGradient();
        gradientDup.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient.dup());
        gradientDup.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient.dup());

        for (Map.Entry<String, INDArray> entry : gradientDup.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = val.mul(lr);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, layer.conf().getLayer().getLearningRate(), 1e-4);
    }


    @Test
    public void testNoOpUpdater() {
        Random r = new Random(12345L);
        double lr = 0.5;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                .build();

        int numParams = conf.getLayer().initializer().numParams(conf, true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        Updater updater = UpdaterCreator.getUpdater(layer);

        for (int i = 0; i < weightGradient.length(); i++) weightGradient.putScalar(i, r.nextDouble());
        for (int i = 0; i < biasGradient.length(); i++) biasGradient.putScalar(i, r.nextDouble());

        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGradient);

        updater.update(layer, gradient, -1, 1);

        INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
        INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

        assertEquals(weightGradient, weightGradActual);
        assertEquals(biasGradient, biasGradActual);

    }

    @Test
    public void testMultiLayerUpdater() throws Exception {
        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .momentum(0.6)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .layer(1, new DenseLayer.Builder().nIn(5).nOut(6).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                .layer(2, new DenseLayer.Builder().nIn(6).nOut(7).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                .layer(3, new DenseLayer.Builder().nIn(7).nOut(8).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater updater = UpdaterCreator.getUpdater(net);
        assertNotNull(updater);
        assertTrue(updater.getClass() == MultiLayerUpdater.class);

        Field f = MultiLayerUpdater.class.getDeclaredField("layerUpdaters");
        f.setAccessible(true);
        Updater[] updaters = (Updater[]) f.get(updater);
        assertNotNull(updaters);
        assertTrue(updaters.length == net.getnLayers());
        assertTrue(updaters[0] instanceof LayerUpdater);
        assertTrue(updaters[1] instanceof LayerUpdater);
        assertTrue(updaters[2] instanceof LayerUpdater);
        assertTrue(updaters[3] instanceof LayerUpdater);

        int count = 0;
        for(Updater u : updaters){
            LayerUpdater lu = (LayerUpdater)u;
            for(GradientUpdater gu : lu.updaterForVariable.values()){
                switch(count){
                    case 0:
                        assertTrue(gu instanceof Sgd);
                        break;
                    case 1:
                        assertTrue(gu instanceof org.nd4j.linalg.learning.NoOpUpdater);
                        break;
                    case 2:
                        assertTrue(gu instanceof AdaGrad);
                        break;
                    case 3:
                        assertTrue(gu instanceof Nesterovs);
                        break;
                    default:
                        throw new RuntimeException();
                }
            }
            count++;
        }

        LayerUpdater u = (LayerUpdater)updaters[0];


        Updater[] uArr = new Updater[4];
        uArr[0] = new LayerUpdater();
        uArr[1] = new LayerUpdater();
        uArr[2] = new LayerUpdater();
        int updaterStateSize = uArr[2].stateSizeForLayer(net.getLayer(2));
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        uArr[2].setStateViewArray(net.getLayer(2), updaterState, true);

        uArr[3] = new LayerUpdater();
        updaterStateSize = uArr[3].stateSizeForLayer(net.getLayer(3));
        updaterState = Nd4j.create(1, updaterStateSize);
        uArr[3].setStateViewArray(net.getLayer(3), updaterState, true);

        int[] nIns = {4, 5, 6, 7};
        int[] nOuts = {5, 6, 7, 8};

        for (int i = 0; i < 5; i++) {
            Gradient gradient = new DefaultGradient();
            Map<String, INDArray> expectedGradient = new LinkedHashMap<>();

            for (int j = 0; j < net.getnLayers(); j++) {
                //Generate test gradient:
                INDArray wGrad = Nd4j.rand(1, nIns[j]*nOuts[j]);
                INDArray bGrad = Nd4j.rand(1, nOuts[j]);

                String wKey = j + "_" + DefaultParamInitializer.WEIGHT_KEY;
                String bKey = j + "_" + DefaultParamInitializer.BIAS_KEY;

                gradient.setGradientFor(wKey, wGrad);
                gradient.setGradientFor(bKey, bGrad);

                //Also put copy of gradient through separate layer updaters to compare
                Gradient layerGradient = new DefaultGradient();
                layerGradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wGrad.dup());
                layerGradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, bGrad.dup());

                uArr[j].update(net.getLayer(j), layerGradient, i, 1);
                for (String s : layerGradient.gradientForVariable().keySet()) {
                    expectedGradient.put(j + "_" + s, layerGradient.getGradientFor(s));
                }
            }

            updater.update(net, gradient, i, 1);
            assertEquals(gradient.gradientForVariable(), expectedGradient);
        }
    }


    @Test
    public void testSetGetUpdater() {

        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;

        int nIn = 4;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .momentum(0.6)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .layer(1, new DenseLayer.Builder().nIn(5).nOut(6).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                .layer(2, new DenseLayer.Builder().nIn(6).nOut(7).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                .layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.fit(Nd4j.rand(5, nIn), Nd4j.rand(5, nOut));    //Fit, to initialize optimizer/updater

        Updater updater = net.getUpdater();
        assertTrue(updater instanceof MultiLayerUpdater);

        Updater newUpdater = UpdaterCreator.getUpdater(net);
        net.setUpdater(newUpdater);
        assertTrue(newUpdater == net.getUpdater());    //Should be identical object
    }

    @Test
    public void testSetGetUpdater2() {
        //Same as above test, except that we are doing setUpdater on a new network
        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;
        int nIn = 4;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(lr)
                .momentum(0.6)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5).updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                .layer(1, new DenseLayer.Builder().nIn(5).nOut(6).updater(org.deeplearning4j.nn.conf.Updater.NONE).build())
                .layer(2, new DenseLayer.Builder().nIn(6).nOut(7).updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                .layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut).updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater newUpdater = UpdaterCreator.getUpdater(net);
        net.setUpdater(newUpdater);
        assertTrue(newUpdater == net.getUpdater());    //Should be identical object
    }


    @Test
    public void testEpsilon(){
        //Test epsilon setting - adagrad
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).epsilon(0.123).build())
                .layer(2, new OutputLayer.Builder().nIn(2).nOut(2).epsilon(0.456).build())
                .build();

        assertEquals(1e-6, conf.getConf(0).getLayer().getEpsilon(), 0.0);
        assertEquals(0.123, conf.getConf(1).getLayer().getEpsilon(), 0.0);
        assertEquals(0.456, conf.getConf(2).getLayer().getEpsilon(), 0.0);

        MultiLayerNetwork net =  new MultiLayerNetwork(conf);
        net.init();
        MultiLayerUpdater updater = (MultiLayerUpdater)net.getUpdater();
        Updater[] updaters = updater.getLayerUpdaters();

        LayerUpdater u0 = (LayerUpdater)updaters[0];
        AdaGrad adaGrad = (AdaGrad)u0.updaterForVariable.get("W");
        assertEquals(1e-6, adaGrad.getEpsilon(), 0.0);

        LayerUpdater u1 = (LayerUpdater)updaters[1];
        AdaGrad adaGrad1 = (AdaGrad)u1.updaterForVariable.get("W");
        assertEquals(0.123, adaGrad1.getEpsilon(), 0.0);

        LayerUpdater u2 = (LayerUpdater)updaters[2];
        AdaGrad adaGrad2 = (AdaGrad)u2.updaterForVariable.get("W");
        assertEquals(0.456, adaGrad2.getEpsilon(), 0.0);



        //Test epsilon setting - adadelta
        conf = new NeuralNetConfiguration.Builder()
                .updater(org.deeplearning4j.nn.conf.Updater.ADADELTA)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).epsilon(0.123).build())
                .layer(2, new OutputLayer.Builder().nIn(2).nOut(2).epsilon(0.456).build())
                .build();

        assertEquals(1e-6, conf.getConf(0).getLayer().getEpsilon(), 0.0);
        assertEquals(0.123, conf.getConf(1).getLayer().getEpsilon(), 0.0);
        assertEquals(0.456, conf.getConf(2).getLayer().getEpsilon(), 0.0);

        net =  new MultiLayerNetwork(conf);
        net.init();
        updater = (MultiLayerUpdater)net.getUpdater();
        updaters = updater.getLayerUpdaters();

        LayerUpdater u0_2 = (LayerUpdater) updaters[0];
        AdaDelta adaDelta = (AdaDelta) u0_2.updaterForVariable.get("W");
        assertEquals(1e-6, adaDelta.getEpsilon(), 0.0);

        LayerUpdater u1_2 = (LayerUpdater) updaters[1];
        AdaDelta adaDelta1 = (AdaDelta)u1_2.updaterForVariable.get("W");
        assertEquals(0.123, adaDelta1.getEpsilon(), 0.0);

        LayerUpdater u2_2 = (LayerUpdater) updaters[2];
        AdaDelta adaDelta2 = (AdaDelta) u2_2.updaterForVariable.get("W");
        assertEquals(0.456, adaDelta2.getEpsilon(), 0.0);

    }
}
