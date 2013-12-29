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
			DoubleMatrix vBiasDelta = ret.vBias.sub(network.vBias);
			DoubleMatrix hBiasDelta = ret.hBias.sub(network.hBias);
			DoubleMatrix wDelta = ret.W.sub(network.W);
			
			ret.vBias = ret.vBias.add(vBiasDelta.div(workers.size()));
			ret.hBias = ret.hBias.add(hBiasDelta.div(workers.size()));
			ret.W = ret.W.add(wDelta.div(workers.size()));

		}

	
		return ret;
	}

}
