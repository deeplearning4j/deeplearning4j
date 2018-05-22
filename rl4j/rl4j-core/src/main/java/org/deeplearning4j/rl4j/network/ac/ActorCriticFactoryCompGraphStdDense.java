package org.deeplearning4j.rl4j.network.ac;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
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
        int nIn = 1;
        for (int i : numInputs) {
            nIn *= i;
        }
        ComputationGraphConfiguration.GraphBuilder confB =
                        new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
                                        .weightInit(WeightInit.XAVIER)
                                        .l2(conf.getL2()).graphBuilder()
                                        .setInputTypes(conf.isUseLSTM() ? InputType.recurrent(nIn)
                                                        : InputType.feedForward(nIn)).addInputs("input")
                                        .addLayer("0", new DenseLayer.Builder().nIn(nIn)
                                                        .nOut(conf.getNumHiddenNodes()).activation(Activation.RELU).build(),
                                                        "input");


        for (int i = 1; i < conf.getNumLayer(); i++) {
            confB.addLayer(i + "", new DenseLayer.Builder().nIn(conf.getNumHiddenNodes()).nOut(conf.getNumHiddenNodes())
                            .activation(Activation.RELU).build(), (i - 1) + "");
        }


        if (conf.isUseLSTM()) {
            confB.addLayer(getConf().getNumLayer() + "", new LSTM.Builder().activation(Activation.TANH)
                            .nOut(conf.getNumHiddenNodes()).build(), (getConf().getNumLayer() - 1) + "");

            confB.addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                            .nOut(1).build(), getConf().getNumLayer() + "");

            confB.addLayer("softmax", new RnnOutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                            .nOut(numOutputs).build(), getConf().getNumLayer() + "");
        } else {
            confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                            .nOut(1).build(), (getConf().getNumLayer() - 1) + "");

            confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                            .nOut(numOutputs).build(), (getConf().getNumLayer() - 1) + "");
        }

        confB.setOutputs("value", "softmax");


        ComputationGraphConfiguration cgconf = confB.pretrain(false).backprop(true).build();
        ComputationGraph model = new ComputationGraph(cgconf);
        model.init();
        if (conf.getListeners() != null) {
            model.setListeners(conf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return new ActorCriticCompGraph(model);
    }

    @AllArgsConstructor
    @Builder
    @Value
    public static class Configuration {

        int numLayer;
        int numHiddenNodes;
        double l2;
        IUpdater updater;
        TrainingListener[] listeners;
        boolean useLSTM;
    }


}
