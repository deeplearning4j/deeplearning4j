package com.ccc.deeplearning.iterativereduce.akka;

import java.util.ArrayList;
import java.util.List;


import com.ccc.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;

public class DeepLearningAccumulator {

	private List<BaseMultiLayerNetwork> workers = new ArrayList<BaseMultiLayerNetwork>();

	public void accumulate(BaseMultiLayerNetwork sda) {
		workers.add(sda);
	}

	public BaseMultiLayerNetwork averaged() {
		if(workers.isEmpty())
			return null;


	

		BaseMultiLayerNetwork first = workers.get(0);
		BaseMultiLayerNetwork ret = first;



		//sum and scale each of the weight vectors
		//start with the second worker as the baseline
		for(int worker = 1; worker <  workers.size(); worker++) {
			BaseMultiLayerNetwork network = workers.get(worker);
			first.merge(network, workers.size());

		}

	
		return ret;
	}

}
