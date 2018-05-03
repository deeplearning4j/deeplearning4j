package org.deeplearning4j.rl4j.network.dqn;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/13/16.
 */

@Value
public class DQNFactoryStdDense implements DQNFactory {


    Configuration conf;

    public DQN buildDQN(int[] numInputs, int numOutputs) {
        int nIn = 1;
        for (int i : numInputs) {
            nIn *= i;
        }
        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        //.updater(Updater.NESTEROVS).momentum(0.9)
                        //.updater(Updater.RMSPROP).rho(conf.getRmsDecay())//.rmsDecay(conf.getRmsDecay())
                        .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
                        .weightInit(WeightInit.XAVIER)
                        .l2(conf.getL2())
                        .list().layer(0, new DenseLayer.Builder().nIn(nIn).nOut(conf.getNumHiddenNodes())
                                        .activation(Activation.RELU).build());


        for (int i = 1; i < conf.getNumLayer(); i++) {
            confB.layer(i, new DenseLayer.Builder().nIn(conf.getNumHiddenNodes()).nOut(conf.getNumHiddenNodes())
                            .activation(Activation.RELU).build());
        }

        confB.layer(conf.getNumLayer(), new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                        .nIn(conf.getNumHiddenNodes()).nOut(numOutputs).build());


        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        if (conf.getListeners() != null) {
            model.setListeners(conf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }
        return new DQN(model);
    }

    @AllArgsConstructor
    @Value
    @Builder
    public static class Configuration {

        int numLayer;
        int numHiddenNodes;
        double l2;
        IUpdater updater;
        TrainingListener[] listeners;
    }


}
