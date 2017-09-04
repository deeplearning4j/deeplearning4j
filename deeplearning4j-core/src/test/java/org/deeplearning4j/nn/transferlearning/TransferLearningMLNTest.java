package org.deeplearning4j.nn.transferlearning;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.*;

/**
 * Created by susaneraly on 2/15/17.
 */
@Slf4j
public class TransferLearningMLNTest {

    @Test
    public void simpleFineTune() {

        long rng = 12345L;
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));
        //original conf
        NeuralNetConfiguration.Builder confToChange =
                        new NeuralNetConfiguration.Builder().seed(rng).optimizationAlgo(OptimizationAlgorithm.LBFGS)
                                        .updater(Updater.NESTEROVS).momentum(0.99).learningRate(0.01);

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
                                                        .optimizationAlgo(
                                                                        OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .updater(Updater.RMSPROP).learningRate(0.5) //Intent: override both weight and bias LR, unless bias LR is manually set also
                                                        .l2(0.4).regularization(true).build())
                                        .build();

        for (org.deeplearning4j.nn.api.Layer l : modelNow.getLayers()) {
            BaseLayer bl = ((BaseLayer) l.conf().getLayer());
            assertEquals(Updater.RMSPROP, bl.getUpdater());
//            assertEquals(0.5, bl.getLearningRate(), 1e-6);
        }


        NeuralNetConfiguration.Builder confSet = new NeuralNetConfiguration.Builder().seed(rng)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.RMSPROP)
                        .learningRate(0.5).l2(0.4).regularization(true);

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

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().learningRate(0.1)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD);
        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().learningRate(0.1)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD)
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

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().learningRate(0.1)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD);
        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().learningRate(0.1)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD)
                        .build();

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
                                                        .nOut(3).learningRate(0.5).activation(Activation.SOFTMAX)
                                                        .build())
                                        .build();

        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(equivalentConf.list()
                        .layer(0, new DenseLayer.Builder().nIn(4).nOut(7).build())
                        .layer(1, new DenseLayer.Builder().nIn(7).nOut(2).build())
                        .layer(2, new DenseLayer.Builder().nIn(2).nOut(5).build())
                        .layer(3, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(
                                        LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
                                                        .learningRate(0.5).nIn(5).nOut(3).build())
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
                        new NeuralNetConfiguration.Builder().seed(12345).regularization(true).l2(0.001) //l2 regularization on all layers
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .iterations(1).learningRate(0.4).list()
                                        .layer(0, new ConvolutionLayer.Builder(10, 10).nIn(3) //3 channels: RGB
                                                        .nOut(30).stride(4, 4).activation(Activation.RELU).weightInit(
                                                                        WeightInit.RELU)
                                                        .updater(Updater.ADAGRAD).build()) //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                                        .kernelSize(3, 3).stride(2, 2).build()) //(31-3+0)/2+1 = 15
                                        .layer(2, new ConvolutionLayer.Builder(3, 3).nIn(30).nOut(10).stride(2, 2)
                                                        .activation(Activation.RELU).weightInit(WeightInit.RELU)
                                                        .updater(Updater.ADAGRAD).build()) //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                                        .layer(3, new DenseLayer.Builder().activation(Activation.RELU).nIn(490).nOut(50)
                                                        .weightInit(WeightInit.RELU).updater(Updater.ADAGRAD)
                                                        .gradientNormalization(
                                                                        GradientNormalization.ClipElementWiseAbsoluteValue)
                                                        .gradientNormalizationThreshold(10).learningRate(0.5).build())
                                        .layer(4, new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(50)
                                                        .nOut(50).weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD)
                                                        .gradientNormalization(
                                                                        GradientNormalization.ClipElementWiseAbsoluteValue)
                                                        .gradientNormalizationThreshold(10).learningRate(0.6)
                                                        .build())
                                        .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                        .activation(Activation.SOFTMAX).nIn(50).nOut(4) //4 possible shapes: circle, square, arc, line
                                                        .updater(Updater.ADAGRAD).weightInit(WeightInit.XAVIER)
                                                        .gradientNormalization(
                                                                        GradientNormalization.ClipElementWiseAbsoluteValue)
                                                        .gradientNormalizationThreshold(10).build())
                                        .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                                        .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
                                        .inputPreProcessor(4, new FeedForwardToRnnPreProcessor()).pretrain(false)
                                        .backprop(true).backpropType(BackpropType.TruncatedBPTT)
                                        .tBPTTForwardLength(V_NFRAMES / 5).tBPTTBackwardLength(V_NFRAMES / 5).build();
        MultiLayerNetwork modelExpectedArch = new MultiLayerNetwork(confForArchitecture);
        modelExpectedArch.init();

        MultiLayerNetwork modelToTweak =
                        new MultiLayerNetwork(
                                        new NeuralNetConfiguration.Builder().seed(12345)
                                                        //.regularization(true).l2(0.001) //change l2
                                                        .optimizationAlgo(
                                                                        OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .iterations(1).learningRate(0.1) //change learning rate
                                                        .updater(Updater.RMSPROP)// change updater
                                                        .list()
                                                        .layer(0, new ConvolutionLayer.Builder(10, 10) //Only keep the first layer the same
                                                                        .nIn(3) //3 channels: RGB
                                                                        .nOut(30).stride(4, 4)
                                                                        .activation(Activation.RELU)
                                                                        .weightInit(WeightInit.RELU)
                                                                        .updater(Updater.ADAGRAD).build()) //Output: (130-10+0)/4+1 = 31 -> 31*31*30
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
                                                                        .learningRate(0.01).build())
                                                        .layer(4, new GravesLSTM.Builder() //change here
                                                                        .activation(Activation.SOFTSIGN).nIn(50)
                                                                        .nOut(25).weightInit(WeightInit.XAVIER)
                                                                        .build())
                                                        .layer(5, new RnnOutputLayer.Builder(
                                                                        LossFunctions.LossFunction.MCXENT)
                                                                                        .activation(Activation.SOFTMAX)
                                                                                        .nIn(25).nOut(4)
                                                                                        .weightInit(WeightInit.XAVIER)
                                                                                        .gradientNormalization(
                                                                                                        GradientNormalization.ClipElementWiseAbsoluteValue)
                                                                                        .gradientNormalizationThreshold(
                                                                                                        10)
                                                                                        .build())
                                                        .inputPreProcessor(0,
                                                                        new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                                                        .inputPreProcessor(3,
                                                                        new CnnToFeedForwardPreProcessor(5, 5, 10))
                                                        .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                                                        .pretrain(false).backprop(true)
                                                        .backpropType(BackpropType.TruncatedBPTT)
                                                        .tBPTTForwardLength(V_NFRAMES / 5)
                                                        .tBPTTBackwardLength(V_NFRAMES / 5).build());
        modelToTweak.init();

        MultiLayerNetwork modelNow = new TransferLearning.Builder(modelToTweak)
                        .fineTuneConfiguration(
                                        new FineTuneConfiguration.Builder().seed(12345).regularization(true).l2(0.001) //l2 regularization on all layers
                                                        .optimizationAlgo(
                                                                        OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .updater(Updater.ADAGRAD).weightInit(WeightInit.RELU)
                                                        .iterations(1).learningRate(0.4).build())
                        .removeLayersFromOutput(5)
                        .addLayer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(3, 3)
                                        .stride(2, 2).build())
                        .addLayer(new ConvolutionLayer.Builder(3, 3).nIn(30).nOut(10).stride(2, 2)
                                        .activation(Activation.RELU).weightInit(WeightInit.RELU)
                                        .updater(Updater.ADAGRAD).build())
                        .addLayer(new DenseLayer.Builder().activation(Activation.RELU).nIn(490).nOut(50)
                                        .weightInit(WeightInit.RELU).updater(Updater.ADAGRAD)
                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                        .gradientNormalizationThreshold(10).learningRate(0.5).build())
                        .addLayer(new GravesLSTM.Builder().activation(Activation.SOFTSIGN).nIn(50).nOut(50)
                                        .weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD)
                                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                        .gradientNormalizationThreshold(10).learningRate(0.6).build())
                        .addLayer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .activation(Activation.SOFTMAX).nIn(50).nOut(4) //4 possible shapes: circle, square, arc, line
                                        .updater(Updater.ADAGRAD).weightInit(WeightInit.XAVIER)
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
                                        new NeuralNetConfiguration.Builder().seed(123).iterations(1).learningRate(.01)
                                                        .weightInit(WeightInit.XAVIER)
                                                        .optimizationAlgo(
                                                                        OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .updater(Updater.NESTEROVS).momentum(
                                                                        0.9)
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
                                                        .backprop(true).pretrain(false).build());
        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2, randomData.getFeatures(), false).get(2); //10x20x12x12

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().learningRate(0.2)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD);

        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().learningRate(0.2)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD)
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
                        .setInputType(InputType.convolutionalFlat(12, 12, 20)).backprop(true).pretrain(false).build());
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
                        new NeuralNetConfiguration.Builder().learningRate(1e-4).updater(Updater.ADAM)
                                        .activation(Activation.TANH).weightInit(WeightInit.RELU).regularization(true)
                                        .l1(0.1).l2(0.2).list()
                                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(5).build()).layer(1,
                                                        new OutputLayer.Builder().nIn(5).nOut(4)
                                                                        .activation(Activation.HARDSIGMOID).build())
                                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        MultiLayerNetwork net2 = new TransferLearning.Builder(net)
                        .fineTuneConfiguration(new FineTuneConfiguration.Builder().learningRate(2e-2) //Should be set on layers
                                        .backpropType(BackpropType.TruncatedBPTT) //Should be set on MLC
                                        .build())
                        .build();


        //Check original net isn't modified:
        BaseLayer l0 = (BaseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(Updater.ADAM, l0.getUpdater());
        assertEquals(Activation.TANH.getActivationFunction(), l0.getActivationFn());
//        assertEquals(1e-4, l0.getLearningRate(), 1e-8);
        assertEquals(WeightInit.RELU, l0.getWeightInit());
        assertEquals(0.1, l0.getL1(), 1e-6);

        BaseLayer l1 = (BaseLayer) net.getLayer(1).conf().getLayer();
        assertEquals(Updater.ADAM, l1.getUpdater());
        assertEquals(Activation.HARDSIGMOID.getActivationFunction(), l1.getActivationFn());
//        assertEquals(1e-4, l1.getLearningRate(), 1e-8);
        assertEquals(WeightInit.RELU, l1.getWeightInit());
        assertEquals(0.2, l1.getL2(), 1e-6);

        assertEquals(BackpropType.Standard, conf.getBackpropType());

        //Check new net has only the appropriate things modified (i.e., LR)
        l0 = (BaseLayer) net2.getLayer(0).conf().getLayer();
        assertEquals(Updater.ADAM, l0.getUpdater());
        assertEquals(Activation.TANH.getActivationFunction(), l0.getActivationFn());
//        assertEquals(2e-2, l0.getLearningRate(), 1e-8);
        assertEquals(WeightInit.RELU, l0.getWeightInit());
        assertEquals(0.1, l0.getL1(), 1e-6);

        l1 = (BaseLayer) net2.getLayer(1).conf().getLayer();
        assertEquals(Updater.ADAM, l1.getUpdater());
        assertEquals(Activation.HARDSIGMOID.getActivationFunction(), l1.getActivationFn());
//        assertEquals(2e-2, l1.getLearningRate(), 1e-8);
        assertEquals(WeightInit.RELU, l1.getWeightInit());
        assertEquals(0.2, l1.getL2(), 1e-6);

        assertEquals(BackpropType.TruncatedBPTT, net2.getLayerWiseConfigurations().getBackpropType());
    }

    @Test
    public void testAllWithCNNNew() {

        DataSet randomData = new DataSet(Nd4j.rand(10, 28 * 28 * 3).reshape(10, 3, 28, 28), Nd4j.rand(10, 10));
        MultiLayerNetwork modelToFineTune =
                        new MultiLayerNetwork(
                                        new NeuralNetConfiguration.Builder().seed(123).iterations(1).learningRate(.01)
                                                        .weightInit(WeightInit.XAVIER)
                                                        .optimizationAlgo(
                                                                        OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .updater(Updater.NESTEROVS).momentum(
                                                                        0.9)
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
                                                        .backprop(true).pretrain(false).build());
        modelToFineTune.init();
        INDArray asFrozenFeatures = modelToFineTune.feedForwardToLayer(2, randomData.getFeatures(), false).get(2); //10x20x12x12

        NeuralNetConfiguration.Builder equivalentConf = new NeuralNetConfiguration.Builder().learningRate(0.2)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD);
        FineTuneConfiguration overallConf = new FineTuneConfiguration.Builder().learningRate(0.2)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD)
                        .build();

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
                        .inputPreProcessor(0, new CnnToFeedForwardPreProcessor(12, 12, 20)).backprop(true)
                        .pretrain(false).build());
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


}
