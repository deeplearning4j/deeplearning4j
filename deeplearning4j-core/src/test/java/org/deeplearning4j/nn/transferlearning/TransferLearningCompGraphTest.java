package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
}
