package org.deeplearning4j.rl4j.network.ac;

import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 *
 */
@Value
public class ActorCriticFactorySeparateStdDense implements ActorCriticFactorySeparate {

    Configuration conf;

    public ActorCriticSeparate buildActorCritic(int[] numInputs, int numOutputs) {


        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(conf.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .updater(Updater.ADAM)
                //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
                //.regularization(true)
                //.l2(conf.getL2())
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs[0])
                        .nOut(conf.getNumHiddenNodes())
                        .activation("relu")
                        .build());


        for (int i = 1; i < conf.getNumLayer(); i++) {
            confB
                    .layer(i, new DenseLayer.Builder()
                            .nIn(conf.getNumHiddenNodes())
                            .nOut(conf.getNumHiddenNodes())
                            .activation("relu")
                            .build());
        }

        confB
                .layer(conf.getNumLayer(), new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(conf.getNumHiddenNodes())
                        .nOut(1)
                        .build());


        MultiLayerConfiguration mlnconf2 = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf2);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));

        NeuralNetConfiguration.ListBuilder confB2 = new NeuralNetConfiguration.Builder()
                .seed(Constants.NEURAL_NET_SEED)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(conf.getLearningRate())
                //.updater(Updater.NESTEROVS).momentum(0.9)
                .updater(Updater.ADAM)
                //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
                .weightInit(WeightInit.XAVIER)
                //.regularization(true)
                //.l2(conf.getL2())
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs[0])
                        .nOut(conf.getNumHiddenNodes())
                        .activation("relu")
                        .build());


        for (int i = 1; i < conf.getNumLayer(); i++) {
            confB2
                    .layer(i, new DenseLayer.Builder()
                            .nIn(conf.getNumHiddenNodes())
                            .nOut(conf.getNumHiddenNodes())
                            .activation("relu")
                            .build());
        }

        confB2
                .layer(conf.getNumLayer(), new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(conf.getNumHiddenNodes())
                        .nOut(numOutputs)
                        .build());


        MultiLayerConfiguration mlnconf = confB2.pretrain(false).backprop(true).build();
        MultiLayerNetwork model2 = new MultiLayerNetwork(mlnconf);
        model2.init();
        model2.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));


        return new ActorCriticSeparate(model, model2);
    }

    @Value
    @Builder
    public static class Configuration {

        int numLayer;
        int numHiddenNodes;
        double learningRate;
        double l2;

    }


}
