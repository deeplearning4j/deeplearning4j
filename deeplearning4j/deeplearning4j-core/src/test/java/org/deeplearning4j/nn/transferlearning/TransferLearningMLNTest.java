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

package org.deeplearning4j.nn.transferlearning;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.constraint.UnitNormConstraint;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.*;

/**
 * Created by susaneraly on 2/15/17.
 */
@Slf4j
public class TransferLearningMLNTest extends BaseDL4JTest {

    @Test
    public void simpleFineTune() {

        long rng = 12345L;
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));
        //original conf
        NeuralNetConfiguration.Builder confToChange =
                        new NeuralNetConfiguration.Builder().seed(rng).optimizationAlgo(OptimizationAlgorithm.LBFGS)
                                        .updater(new Nesterovs(0.01, 0.99));

        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(confToChange.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                        .build())
                        .build());
        modelToFineTune.init();

        //model after applying changes with transfer learning
        MultiLayerNetwork modelNow =
                new TransferLearning.Builder(modelToFineTune)
                        .fineTuneConfiguration(new FineTuneConfiguration.Builder().seed(rng)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .updater(new RmsProp(0.5)) //Intent: override both weight and bias LR, unless bias LR is manually set also
                                .l2(0.4).build())
                        .build();

        for (org.deeplearning4j.nn.api.Layer l : modelNow.getLayers()) {
            BaseLayer bl = ((BaseLayer) l.conf().getLayer());
            assertEquals(new RmsProp(0.5), bl.getIUpdater());
        }


        NeuralNetConfiguration.Builder confSet = new NeuralNetConfiguration.Builder().seed(rng)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new RmsProp(0.5)).l2(0.4);

        MultiLayerNetwork expectedModel = new MultiLayerNetwork(confSet.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                        .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                        .build())
                        .build());
        expectedModel.init();
        expectedModel.setParams(modelToFineTune.params().dup());

        assertEquals(expectedModel.params(), modelNow.params());

        //Check json
        MultiLayerConfiguration expectedConf = expectedModel.getLayerWiseConfigurations();
        assertEquals(expectedConf.toJson(), modelNow.getLayerWiseConfigurations().toJson());

        //Check params after fit
        modelNow.fit(randomData);
        expectedModel.fit(randomData);

        assertEquals(modelNow.score(), expectedModel.score(), 1e-6);
        INDArray pExp = expectedModel.params();
        INDArray pNow = modelNow.params();
        assertEquals(pExp, pNow);
    }

    @Test
    public void testNoutChanges() {
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 2));

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1));
        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().updater(new Sgd(0.1))
                        .build();

        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(equivalentConf.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                        .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(3).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                        .build())
                        .build());
        modelToFineTune.init();
        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune).fineTuneConfiguration(overallConf)
                        .nOutReplace(3, 2, WeightInit.XAVIER, WeightInit.XAVIER)
                        .nOutReplace(0, 3, WeightInit.XAVIER, new NormalDistribution(1, 1e-1)).build();

        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(equivalentConf.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
                        .layer(1, new DenseLayer.Builder().nIn(3).nOut(2).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(3).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(3).nOut(2)
                                                        .build())
                        .build());
        modelExpectedArch.init();

        //Will fail - expected because of dist and weight init changes
        //assertEquals(modelExpectedArch.getLayerWiseConfigurations().toJson(), modelNow.getLayerWiseConfigurations().toJson());

        BaseLayer bl0 = ((BaseLayer) modelNow.getLayerWiseConfigurations().getConf(0).getLayer());
        BaseLayer bl1 = ((BaseLayer) modelNow.getLayerWiseConfigurations().getConf(1).getLayer());
        BaseLayer bl3 = ((BaseLayer) modelNow.getLayerWiseConfigurations().getConf(3).getLayer());
        assertEquals(bl0.getWeightInit(), WeightInit.XAVIER);
        assertEquals(bl0.getDist(), null);
        assertEquals(bl1.getWeightInit(), WeightInit.DISTRIBUTION);
        assertEquals(bl1.getDist(), new NormalDistribution(1, 1e-1));
        assertEquals(bl3.getWeightInit(), WeightInit.XAVIER);

        //modelNow should have the same architecture as modelExpectedArch
        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(2).params().shape(), modelNow.getLayer(2).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(3).params().shape(), modelNow.getLayer(3).params().shape());

        modelNow.setParams(modelExpectedArch.params());
        //fit should give the same results
        modelExpectedArch.fit(randomData);
        modelNow.fit(randomData);
        assertEquals(modelExpectedArch.score(), modelNow.score(), 0.000001);
        assertEquals(modelExpectedArch.params(), modelNow.params());
    }


    @Test
    public void testRemoveAndAdd() {
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.1));
        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().updater(new Sgd(0.1)).build();

        MultiLayerNetwork modelToFineTune = new MultiLayerNetwork(//overallConf.list()
                        equivalentConf.list().layer(0, new DenseLayer.Builder().nIn(4).nOut(5).build())
                                        .layer(1, new DenseLayer.Builder().nIn(5).nOut(2).build())
                                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(3).build())
                                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                                        LossFunctions.LossFunction.MCXENT)
                                                                        .activation(Activation.SOFTMAX).nIn(3).nOut(3)
                                                                        .build())
                                        .build());
        modelToFineTune.init();

        MultiLayerNetwork modelNow =
                        new TransferLearning.Builder(modelToFineTune).fineTuneConfiguration(overallConf)
                                        .nOutReplace(0, 7, WeightInit.XAVIER, WeightInit.XAVIER)
                                        .nOutReplace(2, 5, WeightInit.XAVIER).removeOutputLayer()
                                        .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(5)
                                                        .nOut(3).updater(new Sgd(0.5)).activation(Activation.SOFTMAX)
                                                        .build())
                                        .build();

        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(equivalentConf.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(7).build())
                        .layer(1, new DenseLayer.Builder().nIn(7).nOut(2).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(5).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                                                        .updater(new Sgd(0.5)).nIn(5).nOut(3).build())
                        .build());
        modelExpectedArch.init();

        //modelNow should have the same architecture as modelExpectedArch
        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(2).params().shape(), modelNow.getLayer(2).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(3).params().shape(), modelNow.getLayer(3).params().shape());

        modelNow.setParams(modelExpectedArch.params());
        //fit should give the same results
        modelExpectedArch.fit(randomData);
        modelNow.fit(randomData);
        assertTrue(modelExpectedArch.score() == modelNow.score());
        assertEquals(modelExpectedArch.params(), modelNow.params());
    }

    @Test
    public void testRemoveAndProcessing() {

        int V_WIDTH = 130;
        int V_HEIGHT = 130;
        int V_NFRAMES = 150;

        MultiLayerConfiguration confForArchitecture =
                        new NeuralNetConfiguration.Builder().seed(12345).l2(0.001) //l2 regularization on all layers
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .updater(new AdaGrad(0.4)).list()
                                        .layer(0, new ConvolutionLayer.Builder(10, 10).nIn(3) //3 channels: RGB
                                                        .nOut(30).stride(4, 4).activation(Activation.RELU).weightInit(
                                                                        WeightInit.RELU).build()) //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                                        .kernelSize(3, 3).stride(2, 2).build()) //(31-3+0)/2+1 = 15
                                        .layer(2, new ConvolutionLayer.Builder(3, 3).nIn(30).nOut(10).stride(2, 2)
                                                        .activation(Activation.RELU).weightInit(WeightInit.RELU)
                                                        .build()) //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                                        .layer(3, new DenseLayer.Builder().activation(Activation.RELU).nIn(490).nOut(50)
                                                        .weightInit(WeightInit.RELU).updater(new AdaGrad(0.5))
                                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                                        .gradientNormalizationThreshold(10).build())
                                        .layer(4, new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(50)
                                                        .nOut(50).weightInit(WeightInit.XAVIER).updater(new AdaGrad(0.6))
                                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                                        .gradientNormalizationThreshold(10).build())
                                        .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX).nIn(50).nOut(4) //4 possible shapes: circle, square, arc, line
                                                        .weightInit(WeightInit.XAVIER)
                                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                                        .gradientNormalizationThreshold(10).build())
                                        .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                                        .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
                                        .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                                        .backpropType(BackpropType.TruncatedBPTT)
                                        .tBPTTForwardLength(V_NFRAMES / 5).tBPTTBackwardLength(V_NFRAMES / 5).build();
        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(confForArchitecture);
        modelExpectedArch.init();

        MultiLayerNetwork modelToTweak =
                        new MultiLayerNetwork(
                                        new NeuralNetConfiguration.Builder().seed(12345)
                                                        .updater(new RmsProp(0.1))
                                                        .list()
                                                        .layer(0, new ConvolutionLayer.Builder(10, 10) //Only keep the first layer the same
                                                                        .nIn(3) //3 channels: RGB
                                                                        .nOut(30).stride(4, 4)
                                                                        .activation(Activation.RELU)
                                                                        .weightInit(WeightInit.RELU)
                                                                        .updater(new AdaGrad(0.1)).build()) //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                                                        .layer(1, new SubsamplingLayer.Builder(
                                                                        SubsamplingLayer.PoolingType.MAX) //change kernel size
                                                                                        .kernelSize(5, 5).stride(2, 2)
                                                                                        .build()) //(31-5+0)/2+1 = 14
                                                        .layer(2, new ConvolutionLayer.Builder(6, 6) //change here
                                                                        .nIn(30).nOut(10).stride(2, 2)
                                                                        .activation(Activation.RELU)
                                                                        .weightInit(WeightInit.RELU).build()) //Output: (14-6+0)/2+1 = 5 -> 5*5*10 = 250
                                                        .layer(3, new DenseLayer.Builder() //change here
                                                                        .activation(Activation.RELU).nIn(250).nOut(50)
                                                                        .weightInit(WeightInit.RELU)
                                                                        .gradientNormalization(
                                                                                        GradientNormalization.ClipElementWiseAbsoluteValue)
                                                                        .gradientNormalizationThreshold(10)
                                                                        .updater(new RmsProp(0.01)).build())
                                                        .layer(4, new GravesLSTM.Builder() //change here
                                                                        .activation(Activation.SOFTSIGN).nIn(50)
                                                                        .nOut(25).weightInit(WeightInit.XAVIER)
                                                                        .build())
                                                        .layer(5, new RnnOutputLayer.Builder(
                                                                        LossFunctions.LossFunction.MCXENT)
                                                                                        .activation(Activation.SOFTMAX)
                                                                                        .nIn(25).nOut(4)
                                                                                        .weightInit(WeightInit.XAVIER)
                                                                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                                                                        .gradientNormalizationThreshold(10)
                                                                                        .build())
                                                        .inputPreProcessor(0,new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                                                        .inputPreProcessor(3,new CnnToFeedForwardPreProcessor(5, 5, 10))
                                                        .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())

                                                        .backpropType(BackpropType.TruncatedBPTT)
                                                        .tBPTTForwardLength(V_NFRAMES / 5)
                                                        .tBPTTBackwardLength(V_NFRAMES / 5).build());
        modelToTweak.init();

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToTweak)
                        .fineTuneConfiguration(
                                        new FineTuneConfiguration.Builder().seed(12345).l2(0.001) //l2 regularization on all layers
                                                        .updater(new AdaGrad(0.4))
                                                        .weightInit(WeightInit.RELU).build())
                        .removeLayersFromOutput(5)
                        .addLayer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3)
                                        .stride(2, 2).build())
                        .addLayer(new ConvolutionLayer.Builder(3, 3).nIn(30).nOut(10).stride(2, 2)
                                        .activation(Activation.RELU).weightInit(WeightInit.RELU).build())
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(490).nOut(50)
                                        .weightInit(WeightInit.RELU).updater(new AdaGrad(0.5))
                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                        .gradientNormalizationThreshold(10).build())
                        .addLayer(new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(50).nOut(50)
                                        .weightInit(WeightInit.XAVIER).updater(new AdaGrad(0.6))
                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                        .gradientNormalizationThreshold(10).build())
                        .addLayer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(50).nOut(4) //4 possible shapes: circle, square, arc, line
                                        .weightInit(WeightInit.XAVIER)
                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                        .gradientNormalizationThreshold(10).build())
                        .setInputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
                        .setInputPreProcessor(4, new FeedForwardToRnnPreProcessor()).build();

        //modelNow should have the same architecture as modelExpectedArch
        assertEquals(modelExpectedArch.getLayerWiseConfigurations().getConf(0).toJson(),
                        modelNow.getLayerWiseConfigurations().getConf(0).toJson());
        //some learning related info the subsampling layer will not be overwritten
        //assertTrue(modelExpectedArch.getLayerWiseConfigurations().getConf(1).toJson().equals(modelNow.getLayerWiseConfigurations().getConf(1).toJson()));
        assertEquals(modelExpectedArch.getLayerWiseConfigurations().getConf(2).toJson(),
                        modelNow.getLayerWiseConfigurations().getConf(2).toJson());
        assertEquals(modelExpectedArch.getLayerWiseConfigurations().getConf(3).toJson(),
                        modelNow.getLayerWiseConfigurations().getConf(3).toJson());
        assertEquals(modelExpectedArch.getLayerWiseConfigurations().getConf(4).toJson(),
                        modelNow.getLayerWiseConfigurations().getConf(4).toJson());
        assertEquals(modelExpectedArch.getLayerWiseConfigurations().getConf(5).toJson(),
                        modelNow.getLayerWiseConfigurations().getConf(5).toJson());

        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        //subsampling has no params
        //assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(2).params().shape(), modelNow.getLayer(2).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(3).params().shape(), modelNow.getLayer(3).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(4).params().shape(), modelNow.getLayer(4).params().shape());
        assertArrayEquals(modelExpectedArch.getLayer(5).params().shape(), modelNow.getLayer(5).params().shape());

    }

    @Test
    public void testAllWithCNN() {

        DataSet randomData = new DataSet(Nd4j.rand(10, 28 * 28 * 3).reshape(10, 3, 28, 28), Nd4j.rand(10, 10));
        MultiLayerNetwork modelToFineTune =
                        new MultiLayerNetwork(
                                        new NeuralNetConfiguration.Builder().seed(123)
                                                        .weightInit(WeightInit.XAVIER)
                                                        .updater(new Nesterovs(0.01, 0.9))
                                                        .list()
                                                        .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).stride(1, 1)
                                                                        .nOut(20).activation(Activation.IDENTITY)
                                                                        .build())
                                                        .layer(1, new SubsamplingLayer.Builder(
                                                                        SubsamplingLayer.PoolingType.MAX)
                                                                                        .kernelSize(2, 2).stride(2, 2)
                                                                                        .build())
                                                        .layer(2, new ConvolutionLayer.Builder(5, 5).stride(1, 1)
                                                                        .nOut(50).activation(Activation.IDENTITY)
                                                                        .build())
                                                        .layer(3, new SubsamplingLayer.Builder(
                                                                        SubsamplingLayer.PoolingType.MAX)
                                                                                        .kernelSize(2, 2).stride(2, 2)
                                                                                        .build())
                                                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                                                        .nOut(500).build())
                                                        .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                                                                        .nOut(250).build())
                                                        .layer(6, new OutputLayer.Builder(
                                                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                                                        .nOut(100)
                                                                                        .activation(Activation.SOFTMAX)
                                                                                        .build())
                                                        .setInputType(InputType.convolutionalFlat(28, 28, 3))
                                                        .build());
        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2, randomData.getFeatures(), false).get(2); //10x20x12x12

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.2))
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().updater(new Sgd(0.2))
                        .build();

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune).fineTuneConfiguration(overallConf)
                        .setFeatureExtractor(1).nOutReplace(4, 600, WeightInit.XAVIER).removeLayersFromOutput(2)
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(600).nOut(300).build())
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(300).nOut(150).build())
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(150).nOut(50).build())
                        .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .activation(Activation.SOFTMAX).nIn(50).nOut(10).build())
                        .build();

        MultiLayerNetwork notFrozen = new MultiLayerNetwork(equivalentConf.list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(50)
                                        .activation(Activation.IDENTITY).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(2, new DenseLayer.Builder().activation(Activation.RELU).nOut(600).build())
                        .layer(3, new DenseLayer.Builder().activation(Activation.RELU).nOut(300).build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(150).build())
                        .layer(5, new DenseLayer.Builder().activation(Activation.RELU).nOut(50).build())
                        .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10)
                                        .activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutionalFlat(12, 12, 20)).build());
        notFrozen.init();

        assertArrayEquals(modelToFineTune.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        //subsampling has no params
        //assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(notFrozen.getLayer(0).params().shape(), modelNow.getLayer(2).params().shape());
        modelNow.getLayer(2).setParams(notFrozen.getLayer(0).params());
        //subsampling has no params
        //assertArrayEquals(notFrozen.getLayer(1).params().shape(), modelNow.getLayer(3).params().shape());
        assertArrayEquals(notFrozen.getLayer(2).params().shape(), modelNow.getLayer(4).params().shape());
        modelNow.getLayer(4).setParams(notFrozen.getLayer(2).params());
        assertArrayEquals(notFrozen.getLayer(3).params().shape(), modelNow.getLayer(5).params().shape());
        modelNow.getLayer(5).setParams(notFrozen.getLayer(3).params());
        assertArrayEquals(notFrozen.getLayer(4).params().shape(), modelNow.getLayer(6).params().shape());
        modelNow.getLayer(6).setParams(notFrozen.getLayer(4).params());
        assertArrayEquals(notFrozen.getLayer(5).params().shape(), modelNow.getLayer(7).params().shape());
        modelNow.getLayer(7).setParams(notFrozen.getLayer(5).params());
        assertArrayEquals(notFrozen.getLayer(6).params().shape(), modelNow.getLayer(8).params().shape());
        modelNow.getLayer(8).setParams(notFrozen.getLayer(6).params());

        int i = 0;
        while (i < 3) {
            notFrozen.fit(new DataSet(asFrozenFeatures, randomData.getLabels()));
            modelNow.fit(randomData);
            i++;
        }

        INDArray expectedParams = Nd4j.hstack(modelToFineTune.getLayer(0).params(), notFrozen.params());
        assertEquals(expectedParams, modelNow.params());
    }


    @Test
    public void testFineTuneOverride() {
        //Check that fine-tune overrides are selective - i.e., if I only specify a new LR, only the LR should be modified

        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().updater(new Adam(1e-4))
                                        .activation(Activation.TANH).weightInit(WeightInit.RELU)
                                        .l1(0.1).l2(0.2).list()
                                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(5).build()).layer(1,
                                                        new OutputLayer.Builder().nIn(5).nOut(4)
                                                                        .activation(Activation.HARDSIGMOID).build())
                                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        MultiLayerNetwork net2 = new TransferLearning.Builder(net)
                        .fineTuneConfiguration(new FineTuneConfiguration.Builder().updater(new Adam(2e-2))
                                        .backpropType(BackpropType.TruncatedBPTT) //Should be set on MLC
                                        .build())
                        .build();


        //Check original net isn't modified:
        BaseLayer l0 = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(new Adam(1e-4), l0.getIUpdater());
        assertEquals(Activation.TANH.getActivationFunction(), l0.getActivationFn());
        assertEquals(WeightInit.RELU, l0.getWeightInit());
        assertEquals(0.1, l0.getL1(), 1e-6);

        BaseLayer l1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(new Adam(1e-4), l1.getIUpdater());
        assertEquals(Activation.HARDSIGMOID.getActivationFunction(), l1.getActivationFn());
        assertEquals(WeightInit.RELU, l1.getWeightInit());
        assertEquals(0.2, l1.getL2(), 1e-6);

        assertEquals(BackpropType.Standard, conf.getBackpropType());

        //Check new net has only the appropriate things modified (i.e., LR)
        l0 = (BaseLayer) net2.getLayer(0).conf().getLayer();
        assertEquals(new Adam(2e-2), l0.getIUpdater());
        assertEquals(Activation.TANH.getActivationFunction(), l0.getActivationFn());
        assertEquals(WeightInit.RELU, l0.getWeightInit());
        assertEquals(0.1, l0.getL1(), 1e-6);

        l1 = (BaseLayer) net2.getLayer(1).conf().getLayer();
        assertEquals(new Adam(2e-2), l1.getIUpdater());
        assertEquals(Activation.HARDSIGMOID.getActivationFunction(), l1.getActivationFn());
        assertEquals(WeightInit.RELU, l1.getWeightInit());
        assertEquals(0.2, l1.getL2(), 1e-6);

        assertEquals(BackpropType.TruncatedBPTT, net2.getLayerWiseConfigurations().getBackpropType());
    }

    @Test
    public void testAllWithCNNNew() {

        DataSet randomData = new DataSet(Nd4j.rand(10, 28 * 28 * 3).reshape(10, 3, 28, 28), Nd4j.rand(10, 10));
        MultiLayerNetwork modelToFineTune =
                        new MultiLayerNetwork(
                                        new NeuralNetConfiguration.Builder().seed(123)
                                                        .weightInit(WeightInit.XAVIER)
                                                        .updater(new Nesterovs(0.01, 0.9))
                                                        .list()
                                                        .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(3).stride(1, 1)
                                                                        .nOut(20).activation(Activation.IDENTITY)
                                                                        .build())
                                                        .layer(1, new SubsamplingLayer.Builder(
                                                                        SubsamplingLayer.PoolingType.MAX)
                                                                                        .kernelSize(2, 2).stride(2, 2)
                                                                                        .build())
                                                        .layer(2, new ConvolutionLayer.Builder(5, 5).stride(1, 1)
                                                                        .nOut(50).activation(Activation.IDENTITY)
                                                                        .build())
                                                        .layer(3, new SubsamplingLayer.Builder(
                                                                        SubsamplingLayer.PoolingType.MAX)
                                                                                        .kernelSize(2, 2).stride(2, 2)
                                                                                        .build())
                                                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                                                        .nOut(500).build())
                                                        .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                                                                        .nOut(250).build())
                                                        .layer(6, new OutputLayer.Builder(
                                                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                                                        .nOut(100)
                                                                                        .activation(Activation.SOFTMAX)
                                                                                        .build())
                                                        .setInputType(InputType.convolutionalFlat(28, 28, 3)) //See note below
                                                        .build());
        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2, randomData.getFeatures(), false).get(2); //10x20x12x12

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().updater(new Sgd(0.2));
        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().updater(new Sgd(0.2)).build();

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToFineTune).fineTuneConfiguration(overallConf)
                        .setFeatureExtractor(1).removeLayersFromOutput(5)
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(12 * 12 * 20).nOut(300)
                                        .build())
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(300).nOut(150).build())
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(150).nOut(50).build())
                        .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .activation(Activation.SOFTMAX).nIn(50).nOut(10).build())
                        .setInputPreProcessor(2, new CnnToFeedForwardPreProcessor(12, 12, 20)).build();


        MultiLayerNetwork notFrozen = new MultiLayerNetwork(equivalentConf.list()
                        .layer(0, new DenseLayer.Builder().activation(Activation.RELU).nIn(12 * 12 * 20).nOut(300)
                                        .build())
                        .layer(1, new DenseLayer.Builder().activation(Activation.RELU).nIn(300).nOut(150).build())
                        .layer(2, new DenseLayer.Builder().activation(Activation.RELU).nIn(150).nOut(50).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(50)
                                        .nOut(10).activation(Activation.SOFTMAX).build())
                        .inputPreProcessor(0, new CnnToFeedForwardPreProcessor(12, 12, 20))
                        .build());
        notFrozen.init();

        assertArrayEquals(modelToFineTune.getLayer(0).params().shape(), modelNow.getLayer(0).params().shape());
        //subsampling has no params
        //assertArrayEquals(modelExpectedArch.getLayer(1).params().shape(), modelNow.getLayer(1).params().shape());
        assertArrayEquals(notFrozen.getLayer(0).params().shape(), modelNow.getLayer(2).params().shape());
        modelNow.getLayer(2).setParams(notFrozen.getLayer(0).params());
        assertArrayEquals(notFrozen.getLayer(1).params().shape(), modelNow.getLayer(3).params().shape());
        modelNow.getLayer(3).setParams(notFrozen.getLayer(1).params());
        assertArrayEquals(notFrozen.getLayer(2).params().shape(), modelNow.getLayer(4).params().shape());
        modelNow.getLayer(4).setParams(notFrozen.getLayer(2).params());
        assertArrayEquals(notFrozen.getLayer(3).params().shape(), modelNow.getLayer(5).params().shape());
        modelNow.getLayer(5).setParams(notFrozen.getLayer(3).params());

        int i = 0;
        while (i < 3) {
            notFrozen.fit(new DataSet(asFrozenFeatures, randomData.getLabels()));
            modelNow.fit(randomData);
            i++;
        }

        INDArray expectedParams = Nd4j.hstack(modelToFineTune.getLayer(0).params(), notFrozen.params());
        assertEquals(expectedParams, modelNow.params());
    }

    @Test
    public void testObjectOverrides(){
        //https://github.com/deeplearning4j/deeplearning4j/issues/4368
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dropOut(0.5)
                .weightNoise(new DropConnect(0.5))
                .l2(0.5)
                .constrainWeights(new UnitNormConstraint())
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork orig = new MultiLayerNetwork(conf);
        orig.init();

        FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
                .dropOut(0)
                .weightNoise(null)
                .constraints(null)
                .l2(0.0)
                .build();

        MultiLayerNetwork transfer = new TransferLearning.Builder(orig)
                .fineTuneConfiguration(ftc)
                .build();

        DenseLayer l = (DenseLayer) transfer.getLayer(0).conf().getLayer();

        assertNull(l.getIDropout());
        assertNull(l.getWeightNoise());
        assertNull(l.getConstraints());
        assertEquals(0.0, l.getL2(), 0.0);
    }


    @Test
    public void testTransferLearningSubsequent() {
        final INDArray input = Nd4j.create(6,6,6,6);
        final MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .list()
                .setInputType(InputType.inferInputTypes(input)[0])
                .layer(new Convolution2D.Builder(3, 3).nOut(10).build())
                .layer(new Convolution2D.Builder(1, 1).nOut(3).build())
                .layer(new OutputLayer.Builder().nOut(2).lossFunction(LossFunctions.LossFunction.MSE)
                        .build()).build());
        net.init();

        MultiLayerNetwork newGraph = new TransferLearning
                .Builder(net)
                .fineTuneConfiguration(new FineTuneConfiguration.Builder().build())
                .nOutReplace(0, 7, new ConstantDistribution(333))
                .nOutReplace(1, 3, new ConstantDistribution(111))
                .removeLayersFromOutput(1)
                .addLayer(new OutputLayer.Builder()
                        .nIn(48).nOut(2)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .setInputPreProcessor(2, new CnnToFeedForwardPreProcessor(4,4,3))
                .build();
        newGraph.init();

        assertEquals("Incorrect # inputs", 7, newGraph.layerInputSize(1));

        newGraph.output(input);
    }
}
