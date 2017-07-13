package org.deeplearning4j.rl4j.network.ac;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 *
 */
@Value
public class ActorCriticFactoryCompGraphStdDense implements ActorCriticFactoryCompGraph {

    Configuration conf;

    public ActorCriticCompGraph buildActorCritic(int[] numInputs, int numOutputs) {

        ComputationGraphConfiguration.GraphBuilder confB =
                        new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED).iterations(1)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .learningRate(conf.getLearningRate())
                                        //.updater(Updater.NESTEROVS).momentum(0.9)
                                        //.updater(Updater.RMSPROP).rmsDecay(conf.getRmsDecay())
                                        .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
                                        .weightInit(WeightInit.XAVIER)
                                        .regularization(conf.getL2() > 0)
                                        .l2(conf.getL2()).graphBuilder()
                                        .setInputTypes(InputType.feedForward(numInputs[0])).addInputs("input")
                                        .addLayer("0", new DenseLayer.Builder().nIn(numInputs[0])
                                                        .nOut(conf.getNumHiddenNodes()).activation(Activation.RELU).build(),
                                                        "input");


        for (int i = 1; i < conf.getNumLayer(); i++) {
            confB.addLayer(i + "", new DenseLayer.Builder().nIn(conf.getNumHiddenNodes()).nOut(conf.getNumHiddenNodes())
                            .activation(Activation.RELU).build(), (i - 1) + "");
        }


        confB.addLayer("value",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(1).build(),
                        (getConf().getNumLayer() - 1) + "");

        confB.addLayer("softmax", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX) //fixthat
                        .nOut(numOutputs).build(), (getConf().getNumLayer() - 1) + "");

        confB.setOutputs("value", "softmax");


        ComputationGraphConfiguration cgconf = confB.pretrain(false).backprop(true).build();
        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));

        return new ActorCriticCompGraph(model);
    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class Configuration {

        int numLayer;
        int numHiddenNodes;
        double learningRate;
        double l2;
        IUpdater updater;

    }


}
