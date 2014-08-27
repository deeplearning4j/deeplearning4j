package org.deeplearning4j.optimize.optimizers.autoencoder;

import org.deeplearning4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.optimize.optimizers.NeuralNetworkOptimizer;

/**
 *   Auto Encoder Optimizer
 *
 * @author Adam Gibson
 */
public class AutoEncoderOptimizer extends NeuralNetworkOptimizer {
    public AutoEncoderOptimizer(NeuralNetwork network, float lr, Object[] trainingParams, NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm, LossFunctions.LossFunction lossFunction) {
        super(network, lr, trainingParams, optimizationAlgorithm, lossFunction);
    }




}
