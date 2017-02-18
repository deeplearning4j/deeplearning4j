package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.*;

/**
 * Created by susaneraly on 2/17/17.
 */
public class TransferLearningCompGraphTest {

    @Test
    public void simpleFineTune() {

        long rng = 12345L;
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,3));
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
                        .build(),"layer0In")
                .addLayer("layer1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(),"layer0")
                .setOutputs("layer1")
                .build();

        //conf with learning parameters changed
        ComputationGraphConfiguration expectedConf = new NeuralNetConfiguration.Builder()
                .seed(rng)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .learningRate(0.0001)
                .regularization(true)
                .graphBuilder()
                .addInputs("layer0In")
                .setInputTypes(InputType.feedForward(4))
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build(),"layer0In")
                .addLayer("layer1", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(),"layer0")
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
                        new NeuralNetConfiguration.Builder()
                                .seed(rng)
                                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                .updater(Updater.RMSPROP)
                                .learningRate(0.0001)
                                .regularization(true))
                .build();

        //Check json
        assertEquals(expectedConf.toJson(), modelNow.getConfiguration().toJson());

        //Check params after fit
        modelNow.fit(randomData);
        expectedModel.fit(randomData);
        assertTrue(modelNow.score() == expectedModel.score());
        assertEquals(modelNow.params(), expectedModel.params());
    }

    @Test
    public void testNoutChanges(){
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,2));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).activation(Activation.IDENTITY);
        ComputationGraph modelToFineTune = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .build(),"layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .build(),"layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build(),"layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(),"layer2")
                .setOutputs("layer3")
                .build());
        modelToFineTune.init();
        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(overallConf)
                .nOutReplace("layer3", 2, WeightInit.XAVIER, WeightInit.XAVIER)
                .nOutReplace("layer0", 3, WeightInit.XAVIER, WeightInit.XAVIER)
                .build();

        ComputationGraph modelExpectedArch = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(3)
                        .build(),"layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .build(),"layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build(),"layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(2)
                        .build(),"layer2")
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
        assertTrue(modelExpectedArch.score() == modelNow.score());
        assertEquals(modelExpectedArch.params(), modelNow.params());
    }

    @Test
    public void testRemoveAndAdd() {
        DataSet randomData = new DataSet(Nd4j.rand(10,4),Nd4j.rand(10,3));

        NeuralNetConfiguration.Builder overallConf = new NeuralNetConfiguration.Builder().learningRate(0.1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.SGD).activation(Activation.IDENTITY);
        ComputationGraph modelToFineTune = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(5)
                        .build(),"layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(5).nOut(2)
                        .build(),"layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(3)
                        .build(),"layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(3).nOut(3)
                        .build(),"layer2")
                .setOutputs("layer3")
                .build());
        modelToFineTune.init();

       ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(overallConf)
                .nOutReplace("layer0", 7, WeightInit.XAVIER, WeightInit.XAVIER)
                .nOutReplace("layer2", 5, WeightInit.XAVIER)
                .removeVertex("layer3")
                .addLayer("layer3", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(3).activation(Activation.SOFTMAX).build(),"layer2")
                .setOutputs("layer3")
                .build();

        ComputationGraph modelExpectedArch = new ComputationGraph(overallConf.graphBuilder()
                .addInputs("layer0In")
                .addLayer("layer0", new DenseLayer.Builder()
                        .nIn(4).nOut(7)
                        .build(),"layer0In")
                .addLayer("layer1", new DenseLayer.Builder()
                        .nIn(7).nOut(2)
                        .build(),"layer0")
                .addLayer("layer2", new DenseLayer.Builder()
                        .nIn(2).nOut(5)
                        .build(),"layer1")
                .addLayer("layer3", new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(5).nOut(3)
                        .build(),"layer2")
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
        assertTrue(modelExpectedArch.score() == modelNow.score());
        assertEquals(modelExpectedArch.params(), modelNow.params());
    }

}
