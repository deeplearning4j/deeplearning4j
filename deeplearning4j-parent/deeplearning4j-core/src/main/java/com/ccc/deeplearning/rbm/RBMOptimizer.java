package com.ccc.deeplearning.rbm;

import static com.ccc.deeplearning.util.MatrixUtil.mean;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.nn.BaseNeuralNetwork;
import com.ccc.deeplearning.optimize.NeuralNetworkOptimizer;

public class RBMOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 3676032651650426749L;
	protected int k = -1;
	protected int numTimesIterated = 0;
	
	public RBMOptimizer(BaseNeuralNetwork network, double lr,
			Object[] trainingParams) {
		super(network, lr, trainingParams);
	}

	@Override
	public void getValueGradient(double[] buffer) {
		int k = (int) extraParams[0];
		numTimesIterated++;
		//adaptive k based on the number of iterations.
		//typically over time, you want to increase k.
		if(this.k <= 0)
			this.k = k;
		if(numTimesIterated % 10 == 0) {
			this.k++;
		}
		
		
		//Don't go over 15
		if(this.k >= 15) 
		     this.k = 15;
		
		k = this.k;
		/*
		 * Cost and updates dictionary.
		 * This is the update rules for weights and biases
		 */
		RBM r = (RBM) network;
		Pair<DoubleMatrix,DoubleMatrix> probHidden = r.sampleHiddenGivenVisible(r.input);

		/*
		 * Start the gibbs sampling.
		 */
		DoubleMatrix chainStart = probHidden.getSecond();

		/*
		 * Note that at a later date, we can explore alternative methods of 
		 * storing the chain transitions for different kinds of sampling
		 * and exploring the search space.
		 */
		Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>> matrices = null;
		//negative visible means or expected values
		@SuppressWarnings("unused")
		DoubleMatrix nvMeans = null;
		//negative value samples
		DoubleMatrix nvSamples = null;
		//negative hidden means or expected values
		DoubleMatrix nhMeans = null;
		//negative hidden samples
		DoubleMatrix nhSamples = null;

		/*
		 * K steps of gibbs sampling. THis is the positive phase of contrastive divergence.
		 * 
		 * There are 4 matrices being computed for each gibbs sampling.
		 * The samples from both the positive and negative phases and their expected values or averages.
		 * 
		 */

		for(int i = 0; i < k; i++) {


			if(i == 0) 
				matrices = r.gibbhVh(chainStart);
			else
				matrices = r.gibbhVh(nhSamples);
			//get the cost updates for sampling in the chain after k iterations
			nvMeans = matrices.getFirst().getFirst();
			nvSamples = matrices.getFirst().getSecond();
			nhMeans = matrices.getSecond().getFirst();
			nhSamples = matrices.getSecond().getSecond();
		}

		/*
		 * Update gradient parameters
		 */
		DoubleMatrix wAdd = r.input.transpose().mmul(probHidden.getSecond()).sub(nvSamples.transpose().mmul(nhMeans)).mul(lr).mul(0.1);

		DoubleMatrix  vBiasAdd = mean(r.input.sub(nvSamples), 0).mul(lr);


		//update rule: the expected values of the hidden input - the negative hidden  means adjusted by the learning rate
		DoubleMatrix hBiasAdd = mean(probHidden.getSecond().sub(nhMeans), 0).mul(lr);
		int idx = 0;
		for (int i = 0; i < wAdd.length; i++) 
			buffer[idx++] = wAdd.get(i);
		
		
		for (int i = 0; i < vBiasAdd.length; i++) 
			buffer[idx++] = vBiasAdd.get(i);
		

		
		for (int i = 0; i < hBiasAdd.length; i++) 
			buffer[idx++] = hBiasAdd.get(i);
		
	
	}



}
