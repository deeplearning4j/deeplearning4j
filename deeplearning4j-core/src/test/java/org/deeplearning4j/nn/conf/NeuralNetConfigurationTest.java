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

import org.deeplearning4j.nn.conf.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
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
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.SIZE)
                .iterations(3)
                .regularization(false)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();

        assertFalse(conf.useRegularization);
        String json = conf.toJson();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromJson(json);

        assertEquals(conf, read);
    }



    @Test
    public void testYaml() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.SIZE)
                .iterations(3)
                .regularization(false)
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();

        assertFalse(conf.useRegularization);
        String json = conf.toYaml();
        NeuralNetConfiguration read = NeuralNetConfiguration.fromYaml(json);

        assertEquals(conf,read);
    }

    @Test
    public void testCopyConstructor() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .dist(new NormalDistribution(1, 1))
                .layer(new RBM())
                .build();

        NeuralNetConfiguration conf2 = new NeuralNetConfiguration(conf);
        assertEquals(conf,conf2);
    }

    @Test
    public void testRNG() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.NORMALIZED)
                .constrainGradientToUnitNorm(true)
                .seed(123)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedNormalized() {
        Nd4j.getRandom().setSeed(123);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.NORMALIZED)
                .constrainGradientToUnitNorm(true)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedUniform() {
        Nd4j.getRandom().setSeed(123);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.UNIFORM)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedVI() {
        Nd4j.getRandom().setSeed(123);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.VI)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);
    }

    @Test
    public void testSetSeedDistribution() {
        Nd4j.getRandom().setSeed(123);
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.DISTRIBUTION)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
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
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new RBM())
                .nIn(trainingSet.numInputs())
                .nOut(trainingSet.numOutcomes())
                .weightInit(WeightInit.SIZE)
                .iterations(3)
                .activationFunction("tanh")
                .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .build();
        Layer model = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights = model.getParam(DefaultParamInitializer.WEIGHT_KEY);

        Layer model2 = LayerFactories.getFactory(conf).create(conf);
        INDArray modelWeights2 = model2.getParam(DefaultParamInitializer.WEIGHT_KEY);

        assertEquals(modelWeights, modelWeights2);



    }


}
