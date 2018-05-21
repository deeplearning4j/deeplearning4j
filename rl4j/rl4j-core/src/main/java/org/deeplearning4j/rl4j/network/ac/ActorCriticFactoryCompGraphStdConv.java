package org.deeplearning4j.rl4j.network.ac;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
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
 * Standard factory for Conv net Actor Critic
 */
@Value
public class ActorCriticFactoryCompGraphStdConv implements ActorCriticFactoryCompGraph {


    Configuration conf;

    public ActorCriticCompGraph buildActorCritic(int shapeInputs[], int numOutputs) {

        if (shapeInputs.length == 1)
            throw new AssertionError("Impossible to apply convolutional layer on a shape == 1");

        int h = (((shapeInputs[1] - 8) / 4 + 1) - 4) / 2 + 1;
        int w = (((shapeInputs[2] - 8) / 4 + 1) - 4) / 2 + 1;

        ComputationGraphConfiguration.GraphBuilder confB =
                        new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
                                        .weightInit(WeightInit.XAVIER)
                                        .l2(conf.getL2()).graphBuilder()
                                        .addInputs("input").addLayer("0",
                                                        new ConvolutionLayer.Builder(8, 8).nIn(shapeInputs[0]).nOut(16)
                                                                        .stride(4, 4).activation(Activation.RELU).build(),
                                                        "input");

        confB.addLayer("1", new ConvolutionLayer.Builder(4, 4).nIn(16).nOut(32).stride(2, 2).activation(Activation.RELU).build(), "0");

        confB.addLayer("2", new DenseLayer.Builder().nIn(w * h * 32).nOut(256).activation(Activation.RELU).build(), "1");

        if (conf.isUseLSTM()) {
            confB.addLayer("3", new LSTM.Builder().nIn(256).nOut(256).activation(Activation.TANH).build(), "2");

            confB.addLayer("value", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                            .nIn(256).nOut(1).build(), "3");

            confB.addLayer("softmax", new RnnOutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                            .nIn(256).nOut(numOutputs).build(), "3");
        } else {
            confB.addLayer("value", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                            .nIn(256).nOut(1).build(), "2");

            confB.addLayer("softmax", new OutputLayer.Builder(new ActorCriticLoss()).activation(Activation.SOFTMAX)
                            .nIn(256).nOut(numOutputs).build(), "2");
        }

        confB.setOutputs("value", "softmax");

        if (conf.isUseLSTM()) {
            confB.inputPreProcessor("0", new RnnToCnnPreProcessor(shapeInputs[1], shapeInputs[2], shapeInputs[0]));
            confB.inputPreProcessor("2", new CnnToFeedForwardPreProcessor(h, w, 32));
            confB.inputPreProcessor("3", new FeedForwardToRnnPreProcessor());
        } else {
            confB.setInputTypes(InputType.convolutional(shapeInputs[1], shapeInputs[2], shapeInputs[0]));
        }

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

        double l2;
        IUpdater updater;
        TrainingListener[] listeners;
        boolean useLSTM;
    }

}
