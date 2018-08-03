/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.updater;


import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.*;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.*;

import static org.junit.Assert.*;

public class TestUpdaters extends BaseDL4JTest {

    protected int nIn = 3;
    protected int nOut = 2;
    //    protected double epsilon = 1e-8;
    protected INDArray gradients;
    protected INDArray weightGradient;
    protected INDArray biasGradient;
    protected DefaultGradient gradient = new DefaultGradient();
    protected INDArray val, gradExpected;
    protected String key;


    @Before
    public void beforeDo() {
        gradients = Nd4j.ones(nIn * nOut + nOut);
        weightGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradient.setFlattenedGradient(gradients);
    }

    @Test
    public void testAdaDeltaUpdate() {
        //Here: test updaters manually vs. using updater
        INDArray dxSquared;
        Map<String, INDArray> msg = new HashMap<>();
        Map<String, INDArray> msdx = new HashMap<>();

        double rho = 0.85;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                        .updater(new AdaDelta(rho, Nd4j.EPS_THRESHOLD))
                                        .build())
                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (int i = 0; i < 2; i++) {
            updater.update(layer, gradient, i, 0, 1, LayerWorkspaceMgr.noWorkspaces());

            // calculations for one iteration / update

            for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
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
                gradientCopyPreUpdate.setGradientFor(key, gradExpected);

                assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));

                msdxTmp.muli(rho);
                dxSquared = gradExpected.mul(gradExpected);
                msdxTmp.addi(dxSquared.muli(1 - rho));

                msg.put(key, msgTmp);
                msdx.put(key, msdxTmp);
                count++;
            }
            assertEquals(rho, ((AdaDelta)layer.layerConf().getIUpdater()).getRho(), 1e-4);
        }

        assertEquals(4, count);
    }

    @Test
    public void testAdaGradUpdater() {
        double lr = 1e-2;
        double epsilon = AdaGrad.DEFAULT_ADAGRAD_EPSILON;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new AdaGrad(lr))
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = Transforms.sqrt(val.mul(val).add(epsilon)).rdiv(lr).mul(val);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }
        assertEquals(lr, ((AdaGrad)layer.layerConf().getIUpdater()).getLearningRate(), 1e-4);
        assertEquals(2, count);
    }


    @Test
    public void testAdamUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = Adam.DEFAULT_ADAM_EPSILON;


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Adam(lr, beta1, beta2, Adam.DEFAULT_ADAM_EPSILON))
                .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, iteration, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);
        double alphat = lr * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            m.muli(beta1).addi(val.mul(1.0 - beta1));
            v.muli(beta2).addi(val.mul(val).mul(1.0 - beta2));
            gradExpected = m.mul(alphat).divi(Transforms.sqrt(v).addi(epsilon));
            if (!gradExpected.equals(gradient.getGradientFor(entry.getKey()))) {
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals(beta1, ((Adam)layer.layerConf().getIUpdater()).getBeta1(), 1e-4);
        assertEquals(beta2, ((Adam)layer.layerConf().getIUpdater()).getBeta2(), 1e-4);
        assertEquals(2, count);
    }

    @Test
    public void testNadamUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = Nadam.DEFAULT_NADAM_EPSILON;

        NeuralNetConfiguration conf =
                new NeuralNetConfiguration.Builder()
                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut)
                                .updater(Nadam.builder().learningRate(lr).beta1(beta1)
                                        .beta2(beta2).epsilon(epsilon).build())
                                .build())
                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);

        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        /*
        * Making update for layer
        * */
        updater.update(layer, gradient, iteration, 0,1, LayerWorkspaceMgr.noWorkspaces());

        double beta1t = FastMath.pow(beta1, iteration + 1);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            INDArray oneMinusBeta1Grad = val.mul(1.0 - beta1);
            m.muli(beta1).addi(oneMinusBeta1Grad);

            INDArray oneMinusBeta2GradSquared = val.mul(val).muli(1.0 - beta2);
            v.muli(beta2).addi(oneMinusBeta2GradSquared);

            INDArray biasCorrectedEstimateOfMomentum = m.mul(beta1).divi(1.0 - beta1t);
            INDArray secondTerm = oneMinusBeta1Grad.divi(1.0 - beta1t);

            INDArray alphat = biasCorrectedEstimateOfMomentum.add(secondTerm).muli(lr);

            INDArray sqrtV = Transforms.sqrt(v, false).addi(epsilon);

            gradExpected = val.assign(alphat).divi(sqrtV);
            if (!gradExpected.equals(gradient.getGradientFor(entry.getKey()))) {
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals("Count should be equal to 2, one for weight gradient and one for bias gradient", 2, count);

        /*
        * Check that we are not erroneously mutating moving avg gradient while calculating
        * `biasCorrectedEstimateOfMomentum = m * beta1 /(1.0 - beta1t);`
        * */
        BaseMultiLayerUpdater baseUpdater = (BaseMultiLayerUpdater) updater;
        UpdaterBlock ub = (UpdaterBlock) baseUpdater.getUpdaterBlocks().get(0);
        NadamUpdater nadamUpdater = (NadamUpdater) ub.getGradientUpdater();


        //Calculated for following setup: initialWeights are all equal to 1, beta1 = 0.8, beta2 = 0.888, learning rate = 0.01
        double calculatedByHandMScalar = 0.2;
        double[] expectedM = Nd4j.ones(1, numParams).mul(calculatedByHandMScalar).data().asDouble();

        // FIXME: int cast
        double[] actualM = Arrays.copyOfRange(nadamUpdater.getM().data().asDouble(), 0, (int) numParams);
        for (int i = 0; i < actualM.length; i++) {
            actualM[i] = Math.round(actualM[i] * 1e2) / 1e2;
        }

        assertEquals("Wrong weight gradient after first iteration's update", Arrays.equals(actualM, expectedM), true);

    }

    @Test
    public void testAdaMaxUpdater() {
        INDArray m, v;
        double lr = 0.01;
        int iteration = 0;
        double beta1 = 0.8;
        double beta2 = 0.888;
        double epsilon = AdaMax.DEFAULT_ADAMAX_EPSILON;

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new AdaMax(lr, beta1, beta2, AdaMax.DEFAULT_ADAMAX_EPSILON))
                .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        updater.update(layer, gradient, iteration, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        double beta1t = FastMath.pow(beta1, iteration + 1);
        double beta2t = FastMath.pow(beta2, iteration + 1);
        double alphat = lr * FastMath.sqrt(1 - beta2t) / (1 - beta1t);
        if (Double.isNaN(alphat) || alphat == 0.0)
            alphat = epsilon;

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            m = Nd4j.zeros(val.shape());
            v = Nd4j.zeros(val.shape());

            m.muli(beta1).addi(val.mul(1.0 - beta1));
            v.muli(beta2).addi(val.mul(val).mul(1.0 - beta2));
            gradExpected = m.mul(alphat).divi(Transforms.sqrt(v).addi(epsilon));
            if (!gradExpected.equals(gradient.getGradientFor(entry.getKey()))) {
                System.out.println(Arrays.toString(gradExpected.dup().data().asFloat()));
                System.out.println(Arrays.toString(gradient.getGradientFor(entry.getKey()).dup().data().asFloat()));
            }
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals(beta1, ((AdaMax)layer.layerConf().getIUpdater()).getBeta1(), 1e-4);
        assertEquals(beta2, ((AdaMax)layer.layerConf().getIUpdater()).getBeta2(), 1e-4);
        assertEquals(2, count);
    }

    @Test
    public void testNestorovsUpdater() {
        double lr = 1e-2;
        double mu = 0.6;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Nesterovs(lr, mu))
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        int count = 0;
        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            INDArray val = entry.getValue();
            INDArray v = Nd4j.create(val.shape());
            INDArray vPrev = v.dup();
            v = v.mul(mu).subi(val.mul(lr));
            gradExpected = vPrev.muli(mu).addi(v.mul(-mu - 1));

            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
            count++;
        }

        assertEquals(mu, ((Nesterovs)layer.layerConf().getIUpdater()).getMomentum(), 1e-4);
        assertEquals(2, count);
    }


    @Test
    public void testRMSPropUpdater() {
        double lr = 0.01;
        double rmsDecay = 0.25;
        Map<String, INDArray> lastG = new HashMap<>();


        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new RmsProp(lr,rmsDecay, RmsProp.DEFAULT_RMSPROP_EPSILON))
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);
        int updaterStateSize = (int) layer.layerConf().getIUpdater().stateSize(numParams);
        INDArray updaterState = Nd4j.create(1, updaterStateSize);
        updater.setStateViewArray(layer, updaterState, true);


        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        double epsilon = 1e-8;

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
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
        assertEquals(rmsDecay, ((RmsProp)layer.layerConf().getIUpdater()).getRmsDecay(), 1e-4);
    }

    @Test
    public void testSGDUpdater() {
        double lr = 0.05;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Sgd(lr))
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);

        Gradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = val.mul(lr);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, ((Sgd)layer.layerConf().getIUpdater()).getLearningRate(), 1e-4);
    }


    @Test
    public void testNoOpUpdater() {
        Random r = new Random(12345L);
        double lr = 0.5;

        NeuralNetConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new NoOp())
                                        .layer(new DenseLayer.Builder().nIn(nIn).nOut(nOut).build())
                                        .build();

        long numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);

        for (int i = 0; i < weightGradient.length(); i++)
            weightGradient.putScalar(i, r.nextDouble());
        for (int i = 0; i < biasGradient.length(); i++)
            biasGradient.putScalar(i, r.nextDouble());

        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        gradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradient.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, bg);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        INDArray weightGradActual = gradient.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
        INDArray biasGradActual = gradient.getGradientFor(DefaultParamInitializer.BIAS_KEY);

        assertEquals(wg, weightGradActual);
        assertEquals(bg, biasGradActual);

    }

    @Test
    public void testMultiLayerUpdater() throws Exception {
        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).updater(new Sgd(lr)).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(6)
                                        .updater(new NoOp()).build())
                        .layer(2, new DenseLayer.Builder().nIn(6).nOut(7)
                                        .updater(new AdaGrad(lr)).build())
                        .layer(3, new OutputLayer.Builder().nIn(7).nOut(8)
                                        .updater(new Nesterovs(0.6))
                                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.MSE)
                                        .build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.fit(Nd4j.create(1, 4), Nd4j.create(1, 8));

        Updater updater = net.getUpdater();
        assertNotNull(updater);
        assertTrue(updater.getClass() == MultiLayerUpdater.class);

        MultiLayerUpdater mlu = (MultiLayerUpdater) updater;

        int count = 0;
        for (UpdaterBlock u : mlu.getUpdaterBlocks()) {
            GradientUpdater gu = u.getGradientUpdater();
            switch (count) {
                case 0:
                    assertTrue(gu instanceof SgdUpdater);
                    break;
                case 1:
                    assertTrue(gu instanceof org.nd4j.linalg.learning.NoOpUpdater);
                    break;
                case 2:
                    assertTrue(gu instanceof AdaGradUpdater);
                    break;
                case 3:
                    assertTrue(gu instanceof NesterovsUpdater);
                    break;
                default:
                    throw new RuntimeException();
            }
            count++;
        }


        GradientUpdater[] uArr = new GradientUpdater[4];
        uArr[0] = new SgdUpdater(new Sgd(lr));
        uArr[1] = new NoOpUpdater(new NoOp());
        uArr[2] = new AdaGradUpdater(new AdaGrad(lr, AdaGrad.DEFAULT_ADAGRAD_EPSILON));
        INDArray updaterState = Nd4j.create(1, 6 * 7 + 7, 'f');
        uArr[2].setStateViewArray(updaterState, new long[] {1, 6 * 7 + 7}, 'f', true);

        uArr[3] = new NesterovsUpdater(new Nesterovs(lr, 0.6));
        //        updaterStateSize = uArr[3].stateSizeForLayer(net.getLayer(3));
        updaterState = Nd4j.create(1, 7 * 8 + 8, 'f');
        uArr[3].setStateViewArray(updaterState, new long[] {1, 7 * 8 + 8}, 'f', true);

        int[] nIns = {4, 5, 6, 7};
        int[] nOuts = {5, 6, 7, 8};

        for (int i = 0; i < 5; i++) {
            Gradient gradient = new DefaultGradient();
            Map<String, INDArray> expectedGradient = new LinkedHashMap<>();

            for (int j = 0; j < net.getnLayers(); j++) {
                //Generate test gradient:
                INDArray wGrad = Nd4j.rand(nIns[j], nOuts[j]);
                INDArray bGrad = Nd4j.rand(1, nOuts[j]);

                String wKey = j + "_" + DefaultParamInitializer.WEIGHT_KEY;
                String bKey = j + "_" + DefaultParamInitializer.BIAS_KEY;

                gradient.setGradientFor(wKey, wGrad);
                gradient.setGradientFor(bKey, bGrad);

                //Also put copy of gradient through separate layer updaters to compare
                Gradient layerGradient = new DefaultGradient();
                layerGradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wGrad.dup());
                layerGradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, bGrad.dup());

//                uArr[j].getConfig().applySchedules(0, net.getLayer(j).conf().getLearningRateByParam("W"));
                for (String s : layerGradient.gradientForVariable().keySet()) {
                    expectedGradient.put(j + "_" + s, layerGradient.getGradientFor(s));
                }
            }

            updater.update(net, gradient, i, 0, 1, LayerWorkspaceMgr.noWorkspaces());
            assertEquals(gradient.gradientForVariable(), expectedGradient);
        }
    }


    @Test
    public void testSetGetUpdater() {

        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;

        int nIn = 4;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(lr,0.6)).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5)
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(6)
                                        .updater(new NoOp()).build())
                        .layer(2, new DenseLayer.Builder().nIn(6).nOut(7)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                        .layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.fit(Nd4j.rand(5, nIn), Nd4j.rand(5, nOut)); //Fit, to initialize optimizer/updater

        Updater updater = net.getUpdater();
        assertTrue(updater instanceof MultiLayerUpdater);

        Updater newUpdater = UpdaterCreator.getUpdater(net);
        net.setUpdater(newUpdater);
        assertTrue(newUpdater == net.getUpdater()); //Should be identical object
    }

    @Test
    public void testSetGetUpdater2() {
        //Same as above test, except that we are doing setUpdater on a new network
        Nd4j.getRandom().setSeed(12345L);
        double lr = 0.03;
        int nIn = 4;
        int nOut = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Nesterovs(lr,0.6)).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(5)
                                        .updater(org.deeplearning4j.nn.conf.Updater.SGD).build())
                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(6)
                                        .updater(new NoOp()).build())
                        .layer(2, new DenseLayer.Builder().nIn(6).nOut(7)
                                        .updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD).build())
                        .layer(3, new OutputLayer.Builder().nIn(7).nOut(nOut)
                                        .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).build())
                        .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        Updater newUpdater = UpdaterCreator.getUpdater(net);
        net.setUpdater(newUpdater);
        assertTrue(newUpdater == net.getUpdater()); //Should be identical object
    }

    @Test
    public void testPretrain() {

        gradients = Nd4j.ones(nIn * nOut + nOut + nIn);
        weightGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        INDArray vbiasGradient = gradients.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradient.setFlattenedGradient(gradients);


        //Test with pretrain = true
        double lr = 0.05;
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradient.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbiasGradient);


        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().updater(new Sgd(lr)).seed(42)
                        .layer(new AutoEncoder.Builder()
                                        .lossFunction(LossFunctions.LossFunction.COSINE_PROXIMITY)
                                        .activation(Activation.IDENTITY).nIn(nIn).nOut(nOut).build())
                        .build();
        long numParams = conf.getLayer().initializer().numParams(conf);
        conf.setPretrain(true);
        INDArray params = Nd4j.create(1, numParams);
        BaseLayer layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        Updater updater = UpdaterCreator.getUpdater(layer);

        DefaultGradient gradientCopyPreUpdate = new DefaultGradient();
        INDArray g = gradients.dup();
        INDArray wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        INDArray bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        INDArray vbg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);
        gradientCopyPreUpdate.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbg);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            val = entry.getValue();
            gradExpected = val.mul(lr);
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, ((Sgd)layer.layerConf().getIUpdater()).getLearningRate(), 1e-4);


        //Test with pretrain == false
        gradients = Nd4j.ones(nIn * nOut + nOut + nIn);
        weightGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        biasGradient = gradients.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        vbiasGradient = gradients.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradient.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, weightGradient);
        gradient.setGradientFor(DefaultParamInitializer.BIAS_KEY, biasGradient);
        gradient.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbiasGradient);
        gradient.setFlattenedGradient(gradients);

        gradientCopyPreUpdate = new DefaultGradient();
        g = gradients.dup();
        wg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn * nOut));
        bg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut, nIn * nOut + nOut));
        vbg = g.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn * nOut + nOut, nIn * nOut + nOut + nIn));
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.WEIGHT_KEY, wg);
        gradientCopyPreUpdate.setGradientFor(DefaultParamInitializer.BIAS_KEY, bg);
        gradientCopyPreUpdate.setGradientFor(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbg);
        gradientCopyPreUpdate.setFlattenedGradient(g);

        conf.setPretrain(false);
        params = Nd4j.create(1, numParams);
        layer = (BaseLayer) conf.getLayer().instantiate(conf, null, 0, params, true);
        layer.setBackpropGradientsViewArray(gradients);
        updater = UpdaterCreator.getUpdater(layer);

        updater.update(layer, gradient, -1, 0, 1, LayerWorkspaceMgr.noWorkspaces());

        for (Map.Entry<String, INDArray> entry : gradientCopyPreUpdate.gradientForVariable().entrySet()) {
            //            System.out.println(entry.getKey());
            val = entry.getValue();
            if (!entry.getKey().equals("vb")) {
                gradExpected = val.mul(lr);
            } else {
                //With pretrain == false, we shouldn't be updating the pretrain params (vb)
                gradExpected = val;
            }
            //            System.out.println(gradExpected + "\t" + gradient.getGradientFor(entry.getKey()));
            assertEquals(gradExpected, gradient.getGradientFor(entry.getKey()));
        }
        assertEquals(lr, ((Sgd)layer.layerConf().getIUpdater()).getLearningRate(), 1e-4);
    }

    @Test
    public void testUpdaterBlockMlnAndCG() {
        for (int i = 0; i < 2; i++) {

            List<UpdaterBlock> blocks;
            if (i == 0) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).name("l0")
                                                .updater(new Adam(0.5)).build())
                                .layer(1, new DenseLayer.Builder().nIn(10).nOut(10).name("l1")
                                                .updater(new Adam(0.5)).biasUpdater(new Adam(0.25))
                                                .build())
                                .layer(2, new DenseLayer.Builder().nIn(10).nOut(10).name("l2")
                                                .updater(new AdaDelta()).build())
                                .layer(3, new DenseLayer.Builder().nIn(10).nOut(10).name("l3")
                                                .updater(new AdaGrad(0.5)).build())
                                .layer(4, new OutputLayer.Builder().nIn(10).nOut(10).name("l4")
                                                .updater(new AdaMax(0.5)).build())
                                .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
                blocks = u.getUpdaterBlocks();
            } else {
                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                                .graphBuilder().addInputs("in")
                                .addLayer("l0", new DenseLayer.Builder().nIn(10).nOut(10)
                                        .updater(new Adam(0.5)).build(), "in")
                                .addLayer("l1", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(new Adam(0.5)).biasUpdater(new Adam(0.25))
                                                .build(), "l0")
                                .addLayer("l2", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(new AdaDelta()).build(), "l1")
                                .addLayer("l3", new DenseLayer.Builder().nIn(10).nOut(10)
                                                .updater(new AdaGrad(0.5)).build(), "l2")
                                .addLayer("l4", new OutputLayer.Builder().nIn(10).nOut(10)
                                                .updater(new AdaMax(0.5)).build(), "l3")
                                .setOutputs("l4").build();

                ComputationGraph net = new ComputationGraph(conf);
                net.init();

                ComputationGraphUpdater u = net.getUpdater();
                blocks = u.getUpdaterBlocks();
            }


            //Expect 4 blocks: (layer0 W, layer0 B, layer 1 W], [layer 1 B], [layer 2 W, layer 2 B],
            // [layer 3 W, layer 3 B], [layer 4 W, layer 4 B]
            assertEquals(5, blocks.size());


            //Check first updater block:
            UpdaterBlock ub0 = blocks.get(0);
            assertEquals(3, ub0.getLayersAndVariablesInBlock().size());
            assertEquals("l0", ub0.getLayersAndVariablesInBlock().get(0).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub0.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l0", ub0.getLayersAndVariablesInBlock().get(1).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub0.getLayersAndVariablesInBlock().get(1).getParamName());
            assertEquals("l1", ub0.getLayersAndVariablesInBlock().get(2).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub0.getLayersAndVariablesInBlock().get(2).getParamName());

            int nParams0 = 10 * 10 + 10 + 10 * 10;
            assertEquals(0, ub0.getParamOffsetStart());
            assertEquals(nParams0, ub0.getParamOffsetEnd());
            int nUpdaterVals0 = 2 * nParams0; //2x for Adam
            assertEquals(0, ub0.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0, ub0.getUpdaterViewOffsetEnd());

            //Check second updater block:
            UpdaterBlock ub1 = blocks.get(1);
            assertEquals(1, ub1.getLayersAndVariablesInBlock().size());
            assertEquals("l1", ub1.getLayersAndVariablesInBlock().get(0).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub1.getLayersAndVariablesInBlock().get(0).getParamName());

            int nParams1 = 10;
            assertEquals(nParams0, ub1.getParamOffsetStart());
            assertEquals(nParams0 + nParams1, ub1.getParamOffsetEnd());
            int nUpdaterVals1 = 2 * nParams1; //2x for Adam
            assertEquals(nUpdaterVals0, ub1.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1, ub1.getUpdaterViewOffsetEnd());

            //Check third updater block:
            UpdaterBlock ub2 = blocks.get(2);
            assertEquals(2, ub2.getLayersAndVariablesInBlock().size());
            assertEquals("l2", ub2.getLayersAndVariablesInBlock().get(0).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub2.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l2", ub2.getLayersAndVariablesInBlock().get(1).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub2.getLayersAndVariablesInBlock().get(1).getParamName());

            int nParams2 = 10 * 10 + 10;
            assertEquals(nParams0 + nParams1, ub2.getParamOffsetStart());
            assertEquals(nParams0 + nParams1 + nParams2, ub2.getParamOffsetEnd());
            int nUpdaterVals2 = 2 * nParams2; //2x for Adadelta
            assertEquals(nUpdaterVals0 + nUpdaterVals1, ub2.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2, ub2.getUpdaterViewOffsetEnd());

            //Check fourth updater block:
            UpdaterBlock ub3 = blocks.get(3);
            assertEquals(2, ub3.getLayersAndVariablesInBlock().size());
            assertEquals("l3", ub3.getLayersAndVariablesInBlock().get(0).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub3.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l3", ub3.getLayersAndVariablesInBlock().get(1).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub3.getLayersAndVariablesInBlock().get(1).getParamName());

            int nParams3 = 10 * 10 + 10;
            assertEquals(nParams0 + nParams1 + nParams2, ub3.getParamOffsetStart());
            assertEquals(nParams0 + nParams1 + nParams2 + nParams3, ub3.getParamOffsetEnd());
            int nUpdaterVals3 = nParams3; //1x for AdaGrad
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2, ub3.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2 + nUpdaterVals3, ub3.getUpdaterViewOffsetEnd());

            //Check fifth updater black
            UpdaterBlock ub4 = blocks.get(4);
            assertEquals(2, ub4.getLayersAndVariablesInBlock().size());
            assertEquals("l4", ub4.getLayersAndVariablesInBlock().get(0).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.WEIGHT_KEY, ub4.getLayersAndVariablesInBlock().get(0).getParamName());
            assertEquals("l4", ub4.getLayersAndVariablesInBlock().get(1).getLayer().getConfig().getLayerName());
            assertEquals(DefaultParamInitializer.BIAS_KEY, ub4.getLayersAndVariablesInBlock().get(1).getParamName());

            int nParams4 = 10 * 10 + 10;
            assertEquals(nParams0 + nParams1 + nParams2 + nParams3, ub4.getParamOffsetStart());
            assertEquals(nParams0 + nParams1 + nParams2 + nParams3 + nParams4, ub4.getParamOffsetEnd());
            int nUpdaterVals4 = 2 * nParams4; //2x for AdaGrad
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2 + nUpdaterVals3,
                            ub4.getUpdaterViewOffsetStart());
            assertEquals(nUpdaterVals0 + nUpdaterVals1 + nUpdaterVals2 + nUpdaterVals3 + nUpdaterVals4,
                            ub4.getUpdaterViewOffsetEnd());
        }
    }


    @Test
    public void testUpdaterBlockVae() {

        List<UpdaterBlock> blocks;
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Adam(0.5)).list()
                                        .layer(0, new VariationalAutoencoder.Builder().nIn(8).nOut(12)
                                                        .encoderLayerSizes(10, 11).decoderLayerSizes(13, 14).build())
                                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        MultiLayerUpdater u = (MultiLayerUpdater) net.getUpdater();
        blocks = u.getUpdaterBlocks();


        //Expect 2 blocks: Standard, and pretrain-only params
        assertEquals(2, blocks.size());


        //Check first updater block (all backprop-only params)
        UpdaterBlock ub0 = blocks.get(0);
        List<String> expParams = Arrays.asList("e0W", "e0b", "e1W", "e1b", "pZXMeanW", "pZXMeanb");
        List<String> actParams = new ArrayList<>();
        for (UpdaterBlock.ParamState vs : ub0.getLayersAndVariablesInBlock()) {
            actParams.add(vs.getParamName());
        }
        assertEquals(expParams, actParams);

        //Check second updater block
        UpdaterBlock ub1 = blocks.get(1);
        expParams = Arrays.asList("pZXLogStd2W", "pZXLogStd2b", "d0W", "d0b", "d1W", "d1b", "pXZW", "pXZb");
        actParams = new ArrayList<>();
        for (UpdaterBlock.ParamState vs : ub1.getLayersAndVariablesInBlock()) {
            actParams.add(vs.getParamName());
        }
        assertEquals(expParams, actParams);
    }
}
