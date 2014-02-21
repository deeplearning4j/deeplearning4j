package org.deeplearning4j.iterativereduce.actor.single;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.jblas.DoubleMatrix;


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
			first.merge(network,workers.size());

		}

	
		return ret;
	}

}
