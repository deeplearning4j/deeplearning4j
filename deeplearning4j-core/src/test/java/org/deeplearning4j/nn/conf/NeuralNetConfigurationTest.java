/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.junit.Test;

import java.util.HashMap;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetConfigurationTest {

    final DataSet trainingSet = createData();

    public DataSet createData() {
        int numFeatures = 40;

        INDArray input = Nd4j.create(2, numFeatures); // have to be at least two or else output layer gradient is a scalar and cause exception
        INDArray labels = Nd4j.create(2, 2);

        INDArray row0 = Nd4j.create(1, numFeatures);
        row0.assign(0.1);
        input.putRow(0, row0);
        labels.put(0, 1, 1); // set the 4th column

        INDArray row1 = Nd4j.create(1, numFeatures);
        row1.assign(0.2);

        input.putRow(1, row1);
        labels.put(1, 0, 1); // set the 2nd column

        return new DataSet(input, labels);
    }


    @Test
    public void testJson() {
        NeuralNetConfiguration conf = getRBMConfig(1, 1, WeightInit.SIZE);

        assertFalse(conf.useRegularization);
        String json = conf.toJson();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromJson(json);

        assertEquals(conf, read);
    }


    @Test
    public void testYaml() {
        NeuralNetConfiguration conf = getRBMConfig(1, 1, WeightInit.SIZE);

        assertFalse(conf.useRegularization);
        String json = conf.toYaml();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromYaml(json);

        assertEquals(conf,read);
    }

    @Test
    public void testClone() {
        NeuralNetConfiguration conf = getRBMConfig(1, 1, WeightInit.UNIFORM);
        conf.getLayer().setMomentumAfter(new HashMap<Integer, Double>());
        conf.setStepFunction(new DefaultStepFunction());

        NeuralNetConfiguration conf2 = conf.clone();

        assertEquals(conf, conf2);
        assertNotSame(conf, conf2);
        assertNotSame(conf.getLayer().getMomentumAfter(), conf2.getLayer().getMomentumAfter());
        assertNotSame(conf.getLayer(), conf2.getLayer());
        assertNotSame(conf.getLayer().getDist(), conf2.getLayer().getDist());
        assertNotSame(conf.getStepFunction(), conf2.getStepFunction());
    }

    @Test
    public void testRNG() {
        RBM layer = new RBM.Builder()
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.UNIFORM)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .activation("tanh")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(3)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .layer(layer)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedSize() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.SIZE);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.SIZE);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);
        assertEquals(modelWeights, modelWeights2);
    }


    @Test
    public void testSetSeedNormalized() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.NORMALIZED);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.NORMALIZED);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);
        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedUniform() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.UNIFORM);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.UNIFORM);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedVI() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.VI);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.VI);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedDistribution() {
        Nd4j.getRandom().setSeed(123);

        Layer model = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.DISTRIBUTION);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = getRBMLayer(trainingSet.numInputs(), trainingSet.numOutcomes(), WeightInit.DISTRIBUTION);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }


    @Test
    public void testTimeSeriesLength() {
        RBM layer = new RBM.Builder()
                .nIn(1)
                .nOut(1)
                .weightInit(WeightInit.UNIFORM).dist(new NormalDistribution(1, 1))
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .activation("tanh")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(3)
                .timeSeriesLength(1)
                .regularization(false)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .layer(layer)
                .build();

        assertEquals(conf.getTimeSeriesLength(), 1);

    }

    private static NeuralNetConfiguration getRBMConfig(int nIn, int nOut, WeightInit weightInit){
        RBM layer = new RBM.Builder()
                .nIn(nIn)
                .nOut(nOut)
                .weightInit(weightInit).dist(new NormalDistribution(1, 1))
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .activation("tanh")
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .iterations(3)
                .regularization(false)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .layer(layer)
                .build();
        return conf;

    }

    private static Layer getRBMLayer(int nIn, int nOut, WeightInit weightInit){
        NeuralNetConfiguration conf = getRBMConfig(nIn, nOut, weightInit);
        return LayerFactories.getFactory(conf).create(conf);

    }


}
