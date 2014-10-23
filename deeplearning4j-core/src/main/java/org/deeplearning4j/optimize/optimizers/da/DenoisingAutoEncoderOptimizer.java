package org.deeplearning4j.optimize.optimizers.da;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.optimize.optimizers.NeuralNetworkOptimizer;

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
	public DenoisingAutoEncoderOptimizer(NeuralNetwork network, float lr, Object[] trainingParams, OptimizationAlgorithm optimizationAlgorithm, LossFunctions.LossFunction lossFunction) {
		super(network, lr, trainingParams, optimizationAlgorithm, lossFunction);
	}


	@Override
	public INDArray getParameters() {
		return network.params();
	}



}
