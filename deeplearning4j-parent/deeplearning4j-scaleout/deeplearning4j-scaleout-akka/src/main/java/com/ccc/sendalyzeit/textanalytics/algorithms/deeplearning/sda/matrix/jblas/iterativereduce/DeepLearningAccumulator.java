package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;

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



			for(int i = 0; i < ret.sigmoidLayers.length; i++) 
				ret.sigmoidLayers[i].W =  ret.sigmoidLayers[i].W.add(network.sigmoidLayers[i].W);

			for(int i = 0; i < ret.layers.length; i++) 
				ret.layers[i].setW(ret.layers[i].getW().add(network.layers[i].getW()));


			ret.logLayer.W = ret.logLayer.W.add(network.logLayer.W);

		}

		for(int i = 0; i < ret.layers.length; i++) {
			ret.layers[i].setW(ret.layers[i].getW().div(workers.size()));
		}
		for(int i = 0; i < ret.sigmoidLayers.length; i++) { 
			ret.sigmoidLayers[i].W = ret.sigmoidLayers[i].W.div(workers.size());
		}
		ret.logLayer.W = ret.logLayer.W.div(workers.size());
		return ret;
	}

}
