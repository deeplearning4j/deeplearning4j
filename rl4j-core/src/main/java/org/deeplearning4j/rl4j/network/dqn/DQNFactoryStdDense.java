package org.deeplearning4j.rl4j.network.dqn;

import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.util.Constants;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/13/16.
 */

@Value
public class DQNFactoryStdDense implements DQNFactory {


    Configuration conf;

    public DQN buildDQN(int[] numInputs, int numOutputs) {

        System.out.println(conf);

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
                        .nOut(numOutputs)
                        .build());


        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        return new DQN(model);
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
