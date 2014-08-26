package org.deeplearning4j.sda;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.optimize.NeuralNetworkOptimizer;

/**
 * Optimizes a denoising auto encoder.
 * Handles DA specific parameters such as corruption level
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoderOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 1815627091142129009L;

    /**
     * @param network
     * @param lr
     * @param trainingParams
     * @param optimizationAlgorithm
     * @param lossFunction
     */
    public DenoisingAutoEncoderOptimizer(NeuralNetwork network, float lr, Object[] trainingParams, OptimizationAlgorithm optimizationAlgorithm, LossFunction lossFunction) {
        super(network, lr, trainingParams, optimizationAlgorithm, lossFunction);
    }


    @Override
    public INDArray getParameters() {
        float corruptionLevel = (float) extraParams[0];
        NeuralNetworkGradient gradient = network.getGradient(new Object[]{corruptionLevel,lr,currIteration});
        INDArray L_W = gradient.getwGradient().ravel();
        INDArray L_vbias_mean = gradient.getvBiasGradient();
        INDArray L_hbias_mean = gradient.gethBiasGradient();
		float[] buffer = new float[getNumParameters()];
		/*
		 * Treat params as linear index. Always:
		 * W
		 * Visible Bias
		 * Hidden Bias
		 */
        int idx = 0;
        for (int i = 0; i < L_W.length(); i++) {
            buffer[idx++] = (float) L_W.getScalar(i).element();
        }
        for (int i = 0; i < L_vbias_mean.length(); i++) {
            buffer[idx++] = (float) L_vbias_mean.getScalar(i).element();
        }
        for (int i = 0; i < L_hbias_mean.length(); i++) {
            buffer[idx++] = (float) L_hbias_mean.getScalar(i).element();
        }

        return NDArrays.create(buffer);
    }



}
