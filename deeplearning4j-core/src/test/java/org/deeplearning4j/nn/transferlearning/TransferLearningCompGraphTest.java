package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 2/17/17.
 */
public class TransferLearningCompGraphTest {

    @Test
    public void simpleFineTune() {

        long rng = 12345L;
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));
        //original conf
        ComputationGraphConfiguration confToChange = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .updater(Updater.NESTEROVS).momentum(0.99)
                .learningRate(0.01)
                .graphBuilder()
                .addInputs("layer0In")
                .setInputTypes(InputType.feedForward(4))
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build(), "layer0In")
                .addLayer("layer1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(), "layer0")
                .setOutputs("layer1")
                .build();

        //conf with learning parameters changed
        ComputationGraphConfiguration expectedConf = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .learningRate(0.2)
                .regularization(true)
                .graphBuilder()
                .addInputs("layer0In")
                .setInputTypes(InputType.feedForward(4))
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build(), "layer0In")
                .addLayer("layer1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(), "layer0")
                .setOutputs("layer1")
                .build();
        ComputationGraph expectedModel = new ComputationGraph(expectedConf);
        expectedModel.init();

        ComputationGraph modelToFineTune = new ComputationGraph(expectedConf);
        modelToFineTune.init();
        modelToFineTune.setParams(expectedModel.params());
        //model after applying changes with transfer learning
        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(
                        new FineTuneConfiguration.Builder()
                                .seed(rng)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .updater(Updater.RMSPROP)
                                .learningRate(0.2)
                                .regularization(true).build())
                .build();

        //Check json
        assertEquals(expectedConf.toJson(), modelNow.getConfiguration().toJson());

        //Check params after fit
        modelNow.fit(randomData);
        expectedModel.fit(randomData);
        assertEquals(modelNow.score(), expectedModel.score(), 1e-8);
        assertEquals(modelNow.params(), expectedModel.params());
    }

    @Test
    public void testNoutChanges() {
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 2));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .activation(Activation.IDENTITY);
        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .activation(Activation.IDENTITY)
                .build();

        ComputationGraph modelToFineTune = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .build(), "layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .build(), "layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build(), "layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(), "layer2")
                .setOutputs("layer3")
                .build());
        modelToFineTune.init();
        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(fineTuneConfiguration)
                .nOutReplace("layer3", 2, WeightInit.XAVIER)
                .nOutReplace("layer0", 3, new NormalDistribution(1, 1e-1), WeightInit.XAVIER)
                .setOutputs("layer3")
                .build();

        assertEquals(modelNow.getLayer("layer0").conf().getLayer().getWeightInit(), WeightInit.DISTRIBUTION);
        assertEquals(modelNow.getLayer("layer0").conf().getLayer().getDist(), new NormalDistribution(1, 1e-1));
        assertEquals(modelNow.getLayer("layer1").conf().getLayer().getWeightInit(), WeightInit.XAVIER);
        assertEquals(modelNow.getLayer("layer1").conf().getLayer().getDist(), null);
        assertEquals(modelNow.getLayer("layer3").conf().getLayer().getWeightInit(), WeightInit.XAVIER);

        ComputationGraph modelExpectedArch = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build(), "layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .build(), "layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build(), "layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(2)
                        .build(), "layer2")
                .setOutputs("layer3")
                .build());

        modelExpectedArch.init();

        //modelNow should have the same architecture as modelExpectedArch
        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer0").params().shape(), modelNow.getLayer("layer0").params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer1").params().shape(), modelNow.getLayer("layer1").params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer2").params().shape(), modelNow.getLayer("layer2").params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer3").params().shape(), modelNow.getLayer("layer3").params().shape());

        modelNow.setParams(modelExpectedArch.params());
        //fit should give the same results
        modelExpectedArch.fit(randomData);
        modelNow.fit(randomData);
        assertEquals(modelExpectedArch.score(), modelNow.score(), 1e-8);
        assertEquals(modelExpectedArch.params(), modelNow.params());
    }

    @Test
    public void testRemoveAndAdd() {
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .activation(Activation.IDENTITY);
        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.SGD)
                .activation(Activation.IDENTITY)
                .build();

        ComputationGraph modelToFineTune = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .build(), "layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(5).nOut(2)
                        .build(), "layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build(), "layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(), "layer2")
                .setOutputs("layer3")
                .build());
        modelToFineTune.init();

        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(fineTuneConfiguration)
                .nOutReplace("layer0", 7, WeightInit.XAVIER, WeightInit.XAVIER)
                .nOutReplace("layer2", 5, WeightInit.XAVIER)
                .removeVertexKeepConnections("layer3")
                .addLayer("layer3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(5)
                        .nOut(3)
                        .activation(Activation.SOFTMAX).build(), "layer2")
                .setOutputs("layer3")
                .build();

        ComputationGraph modelExpectedArch = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(7)
                        .build(), "layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(7).nOut(2)
                        .build(), "layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(5)
                        .build(), "layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(5).nOut(3)
                        .build(), "layer2")
                .setOutputs("layer3")
                .build());

        modelExpectedArch.init();

        //modelNow should have the same architecture as modelExpectedArch
        assertArrayEquals(modelExpectedArch.params().shape(), modelNow.params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer0").params().shape(), modelNow.getLayer("layer0").params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer1").params().shape(), modelNow.getLayer("layer1").params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer2").params().shape(), modelNow.getLayer("layer2").params().shape());
        assertArrayEquals(modelExpectedArch.getLayer("layer3").params().shape(), modelNow.getLayer("layer3").params().shape());

        modelNow.setParams(modelExpectedArch.params());
        //fit should give the same results
        modelExpectedArch.fit(randomData);
        modelNow.fit(randomData);
        assertEquals(modelExpectedArch.score(), modelNow.score(), 1e-8);
        assertEquals(modelExpectedArch.params(), modelNow.params());
    }

    @Test
    public void testAllWithCNN() {

        DataSet randomData = new DataSet(Nd4j.rand(10, 28 * 28 * 3).reshape(10, 3, 28, 28), Nd4j.rand(10, 10));
        ComputationGraph modelToFineTune = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(1)
                .learningRate(.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .graphBuilder()
                .addInputs("layer0In")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 3))
                .addLayer("layer0", new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build(), "layer0In")
                .addLayer("layer1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "layer0")
                .addLayer("layer2", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build(), "layer1")
                .addLayer("layer3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "layer2")
                .addLayer("layer4", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build(), "layer3")
                .addLayer("layer5", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(250).build(), "layer4")
                .addLayer("layer6", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(100)
                        .activation(Activation.SOFTMAX)
                        .build(), "layer5")
                .setOutputs("layer5")
                .backprop(true).pretrain(false).build());
        modelToFineTune.init();

        //this will override the learning configuration set in the model
        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().seed(456).learningRate(0.001).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD);
        FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder().seed(456).learningRate(0.001).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).build();

        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(fineTuneConfiguration)
                .setFeatureExtractor("layer1")
                .nOutReplace("layer4", 600, WeightInit.XAVIER)
                .removeVertexAndConnections("layer5")
                .removeVertexAndConnections("layer6")
                .addLayer("layer5", new DenseLayer.Builder().activation(Activation.RELU).nIn(600).nOut(300).build(), "layer4")
                .addLayer("layer6", new DenseLayer.Builder().activation(Activation.RELU).nIn(300).nOut(150).build(), "layer5")
                .addLayer("layer7", new DenseLayer.Builder().activation(Activation.RELU).nIn(150).nOut(50).build(), "layer6")
                .addLayer("layer8", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(50).nOut(10).build(), "layer7")
                .setOutputs("layer8")
                .build();

        ComputationGraph modelExpectedArch = new ComputationGraph(overallConf
                .graphBuilder()
                .addInputs("layer0In")
                .setInputTypes(InputType.convolutionalFlat(28, 28, 3))
                .addLayer("layer0", new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build(), "layer0In")
                .addLayer("layer1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "layer0")
                .addLayer("layer2", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build(), "layer1")
                .addLayer("layer3", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build(), "layer2")
                .addLayer("layer4", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(600).build(), "layer3")
                .addLayer("layer5", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(300).build(), "layer4")
                .addLayer("layer6", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(150).build(), "layer5")
                .addLayer("layer7", new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(50).build(), "layer6")
                .addLayer("layer8", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build(), "layer7")
                .setOutputs("layer8")
                .backprop(true).pretrain(false).build());
        modelExpectedArch.init();
        modelExpectedArch.getVertex("layer0").setLayerAsFrozen();
        modelExpectedArch.getVertex("layer1").setLayerAsFrozen();

        assertEquals(modelExpectedArch.getConfiguration().toJson(), modelNow.getConfiguration().toJson());

        modelNow.setParams(modelExpectedArch.params());
        int i = 0;
        while (i < 5) {
            modelExpectedArch.fit(randomData);
            modelNow.fit(randomData);
            i++;
        }
        assertEquals(modelExpectedArch.params(), modelNow.params());

    }

}
