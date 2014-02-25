package org.deeplearning4j.rbm;

import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.NeuralNetworkGradient;
import org.deeplearning4j.optimize.NeuralNetworkOptimizer;
import org.jblas.DoubleMatrix;


public class RBMOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 3676032651650426749L;
	protected int k = -1;
	protected int numTimesIterated = 0;
	
	public RBMOptimizer(BaseNeuralNetwork network, double lr,
			Object[] trainingParams) {
		super(network, lr, trainingParams);
	}

	@Override
	public  void getValueGradient(double[] buffer) {
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
		NeuralNetworkGradient gradient = network.getGradient(new Object[]{k,lr});
	
		DoubleMatrix wAdd = gradient.getwGradient();
		DoubleMatrix vBiasAdd = gradient.getvBiasGradient();
		DoubleMatrix hBiasAdd = gradient.gethBiasGradient();
		
		int idx = 0;
		for (int i = 0; i < wAdd.length; i++) 
			buffer[idx++] = wAdd.get(i);
		
		
		for (int i = 0; i < vBiasAdd.length; i++) 
			buffer[idx++] = vBiasAdd.get(i);
		

		
		for (int i = 0; i < hBiasAdd.length; i++) 
			buffer[idx++] = hBiasAdd.get(i);
		
	
	}



}
