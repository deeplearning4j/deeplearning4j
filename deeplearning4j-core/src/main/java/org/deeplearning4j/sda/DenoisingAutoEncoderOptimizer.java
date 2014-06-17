package org.deeplearning4j.sda;

import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.optimize.NeuralNetworkOptimizer;
import org.jblas.DoubleMatrix;

/**
 * Optimizes a denoising auto encoder.
 * Handles DA specific parameters such as corruption level
 * @author Adam Gibson
 *
 */
public class DenoisingAutoEncoderOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 1815627091142129009L;

	public DenoisingAutoEncoderOptimizer(BaseNeuralNetwork network,double lr,Object[] trainingParams,OptimizationAlgorithm optimizationAlgorithm,LossFunction lossFunction) {
		super(network,lr,trainingParams,optimizationAlgorithm,lossFunction);
	
	}

    @Override
    public DoubleMatrix getParameters() {
        double corruptionLevel = (double) extraParams[0];
        NeuralNetworkGradient gradient = network.getGradient(new Object[]{corruptionLevel,lr,currIteration});
        DoubleMatrix L_W = gradient.getwGradient();
        DoubleMatrix L_vbias_mean = gradient.getvBiasGradient();
        DoubleMatrix L_hbias_mean = gradient.gethBiasGradient();
		double[] buffer = new double[getNumParameters()];
		/*
		 * Treat params as linear index. Always:
		 * W
		 * Visible Bias
		 * Hidden Bias
		 */
        int idx = 0;
        for (int i = 0; i < L_W.length; i++) {
            buffer[idx++] =L_W.get(i);
        }
        for (int i = 0; i < L_vbias_mean.length; i++) {
            buffer[idx++] = L_vbias_mean.get(i);
        }
        for (int i = 0; i < L_hbias_mean.length; i++) {
            buffer[idx++] = L_hbias_mean.get(i);
        }

        return new DoubleMatrix(buffer);
    }


}
