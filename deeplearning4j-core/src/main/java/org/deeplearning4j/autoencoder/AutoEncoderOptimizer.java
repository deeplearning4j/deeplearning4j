package org.deeplearning4j.autoencoder;

import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.optimize.NeuralNetworkOptimizer;
import org.jblas.DoubleMatrix;

/**
 *   Auto Encoder Optimizer
 *
 * @author Adam Gibson
 */
public class AutoEncoderOptimizer extends NeuralNetworkOptimizer {
    public AutoEncoderOptimizer(NeuralNetwork network, double lr, Object[] trainingParams, NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm, NeuralNetwork.LossFunction lossFunction) {
        super(network, lr, trainingParams, optimizationAlgorithm, lossFunction);
    }

    @Override
    public void getValueGradient(double[] buffer) {
        NeuralNetworkGradient g = network.getGradient(extraParams);
        /*
		 * Treat params as linear index. Always:
		 * W
		 * Visible Bias
		 * Hidden Bias
		 */
        int idx = 0;
        for (int i = 0; i < g.getwGradient().length; i++) {
            buffer[idx++] = g.getwGradient().get(i);
        }
        for (int i = 0; i < g.getvBiasGradient().length; i++) {
            buffer[idx++] = g.getvBiasGradient().get(i);
        }
        for (int i = 0; i < g.gethBiasGradient().length; i++) {
            buffer[idx++] = g.gethBiasGradient().get(i);
        }
    }
}
