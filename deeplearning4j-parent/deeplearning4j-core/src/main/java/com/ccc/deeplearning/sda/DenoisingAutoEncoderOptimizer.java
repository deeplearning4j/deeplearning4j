package com.ccc.deeplearning.sda;

import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.da.DenoisingAutoEncoder;
import com.ccc.deeplearning.nn.BaseNeuralNetwork;
import com.ccc.deeplearning.optimize.NeuralNetworkOptimizer;

public class DenoisingAutoEncoderOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 1815627091142129009L;

	public DenoisingAutoEncoderOptimizer(BaseNeuralNetwork network, double lr,
			Object[] trainingParams) {
		super(network, lr, trainingParams);
	}

	@Override
	public void getValueGradient(double[] buffer) {
		double corruptionLevel = (double) extraParams[0];
		DenoisingAutoEncoder aE = (DenoisingAutoEncoder) network;

		DoubleMatrix tildeX = aE.getCorruptedInput(aE.input, corruptionLevel);
		DoubleMatrix y = aE.getHiddenValues(tildeX);
		DoubleMatrix z = aE.getReconstructedInput(y);

		DoubleMatrix L_h2 = aE.input.sub(z);

		DoubleMatrix L_h1 = L_h2.mmul(aE.W).mul(y).mul(oneMinus(y));

		DoubleMatrix L_vbias = L_h2;
		DoubleMatrix L_hbias = L_h1;


		/*
		 * Gradient down here.
		 */

		DoubleMatrix L_W = tildeX.transpose().mmul(L_h1).add(L_h2.transpose().mmul(y)).mul(lr);


		DoubleMatrix L_hbias_mean = L_hbias.columnMeans().mul(lr);
		DoubleMatrix L_vbias_mean = L_vbias.columnMeans().mul(lr);

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


	}

	


}
