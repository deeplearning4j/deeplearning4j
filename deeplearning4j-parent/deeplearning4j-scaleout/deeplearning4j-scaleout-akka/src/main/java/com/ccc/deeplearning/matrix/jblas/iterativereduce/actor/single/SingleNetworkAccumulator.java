package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.single;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;

public class SingleNetworkAccumulator {
	private List<BaseNeuralNetwork> workers = new ArrayList<BaseNeuralNetwork>();

	public void accumulate(BaseNeuralNetwork sda) {
		workers.add(sda);
	}

	public BaseNeuralNetwork averaged() {
		if(workers.isEmpty())
			return null;


	

		BaseNeuralNetwork first = workers.get(0);
		BaseNeuralNetwork ret = first;



		//sum and scale each of the weight vectors
		//start with the second worker as the baseline
		for(int worker = 1; worker <  workers.size(); worker++) {
			BaseNeuralNetwork network = workers.get(worker);
		
			ret.vBias = ret.vBias.add(network.vBias);
			ret.hBias = ret.hBias.add(network.hBias);
			ret.W = ret.W.add(network.W);

		}

		ret.vBias.div(workers.size());
		ret.hBias.div(workers.size());
		ret.W.div(workers.size());
	
		return ret;
	}

}
